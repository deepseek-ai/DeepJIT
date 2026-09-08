#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <acl/acl.h>
#include <pybind11/pybind11.h>
#include <unistd.h>

#include <deep_jit/backend/ascend/backend.hpp>
#include <deep_jit/cache/memory.hpp>
#include <deep_jit/python_api.hpp>
#include <deep_jit/utils/command.hpp>
#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/filesystem.hpp>
#include <deep_jit/utils/hash.hpp>
#include <deep_jit/utils/json.hpp>
#include <deep_jit/utils/lazy.hpp>
#include <deep_jit/utils/parser.hpp>
#include <deep_jit/utils/str.hpp>
#include <deep_jit/utils/uuid.hpp>

namespace {

using Runtime = deep_jit::Runtime<deep_jit::Ascend>;
using CompilerOptions = deep_jit::ascend::CompilerOptions;
using LaunchOptions = deep_jit::ascend::LaunchOptions;
namespace fs = std::filesystem;

deep_jit::LazyInit<Runtime> python_api_jit(nullptr);
std::shared_ptr<Runtime> process_runtime;
std::shared_ptr<Runtime> gil_runtime;

void init_python_api_jit(const std::string& library_root);

const fs::path& get_test_project_dir() {
    static const fs::path path = [] {
        const char* value = std::getenv("DEEP_JIT_ASCEND_TEST_SOURCE_DIR");
        DJ_HOST_ASSERT(value != nullptr, "DEEP_JIT_ASCEND_TEST_SOURCE_DIR must be set");
        return fs::absolute(value).lexically_normal();
    }();
    return path;
}

template <typename Function>
void run_test(const std::string_view name, Function&& function) {
    std::printf("[ RUN      ] %.*s\n", static_cast<int>(name.size()), name.data());
    std::fflush(stdout);
    function();
    std::printf("[       OK ] %.*s\n", static_cast<int>(name.size()), name.data());
    std::fflush(stdout);
}

template <typename Function>
void expect_failure(Function&& function, const std::string_view expected_message) {
    try {
        function();
    } catch (const std::exception& exception) {
        DJ_HOST_ASSERT(std::string_view(exception.what()).find(expected_message) != std::string_view::npos,
                       "unexpected exception: {}", exception.what());
        return;
    }
    DJ_PANIC("expected failure containing: {}", expected_message);
}

void set_env(const std::string& name, const std::string& value) {
    DJ_HOST_ASSERT(::setenv(name.c_str(), value.c_str(), 1) == 0,
                   "failed to set environment variable: {}", name);
}

void unset_env(const std::string& name) {
    DJ_HOST_ASSERT(::unsetenv(name.c_str()) == 0,
                   "failed to unset environment variable: {}", name);
}

std::string get_source(const std::string& name) {
    return deep_jit::read(get_test_project_dir() / "kernels" / name);
}

std::string get_increment_source(const int bias) {
    return std::format("#define TEST_BIAS {}\n#include <kernels/scalar_increment.hpp>\n", bias);
}

std::shared_ptr<Runtime> make_runtime(const fs::path& include_dir,
                                      const std::string& env_prefix = "ASCEND_TEST",
                                      const std::string& extra_signature = "ascend-test-signature",
                                      const fs::path& third_party_dir = {}) {
    std::vector include_dirs = {include_dir, get_test_project_dir()};
    if (not third_party_dir.empty())
        include_dirs.emplace_back(third_party_dir);
    auto runtime = std::make_shared<Runtime>(deep_jit::Config(
        get_test_project_dir(), env_prefix, extra_signature,
        include_dirs, {"test_ascend/", "kernels/"}));
    runtime->default_compiler_options.bisheng_flags->emplace_back("-fcce-simt-lambda");
    return runtime;
}

void check_artifact(const fs::path& artifact_dir, const std::string& source) {
    DJ_HOST_ASSERT(fs::is_regular_file(artifact_dir / ".committed"),
                   "missing commit marker: {}", artifact_dir.string());
    DJ_HOST_ASSERT(fs::is_regular_file(artifact_dir / "kernel.asc"),
                   "missing Ascend source: {}", artifact_dir.string());
    DJ_HOST_ASSERT(fs::is_regular_file(artifact_dir / "kernel.o"),
                   "missing Ascend binary: {}", artifact_dir.string());
    DJ_HOST_ASSERT(fs::file_size(artifact_dir / "kernel.o") > 0,
                   "empty Ascend binary: {}", artifact_dir.string());
    DJ_HOST_ASSERT(fs::is_regular_file(artifact_dir / "meta.json"),
                   "missing metadata: {}", artifact_dir.string());
    DJ_HOST_ASSERT(deep_jit::read(artifact_dir / "kernel.asc") == source,
                   "cached source mismatch: {}", artifact_dir.string());
}

void check_tmp_is_empty(const fs::path& cache_root) {
    const auto tmp_dir = cache_root / "tmp";
    DJ_HOST_ASSERT(not fs::exists(tmp_dir) or fs::directory_iterator(tmp_dir) == fs::directory_iterator(),
                   "temporary artifacts were not cleaned: {}", tmp_dir.string());
}

class DeviceBuffer {
public:
    void* data = nullptr;
    size_t num_bytes = 0;

    explicit DeviceBuffer(const size_t num_bytes): num_bytes(num_bytes) {
        DJ_ACL_CHECK(aclrtMalloc(&data, num_bytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    }

    ~DeviceBuffer() {
        if (data != nullptr)
            aclrtFree(data);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

template <typename T>
void copy_to_device(DeviceBuffer& output, const std::vector<T>& input) {
    DJ_HOST_ASSERT(output.num_bytes == input.size() * sizeof(T));
    DJ_ACL_CHECK(aclrtMemcpy(
        output.data, output.num_bytes, input.data(), output.num_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
}

template <typename T>
std::vector<T> copy_from_device(const DeviceBuffer& input) {
    std::vector<T> output(input.num_bytes / sizeof(T));
    DJ_ACL_CHECK(aclrtMemcpy(
        output.data(), input.num_bytes, input.data, input.num_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    return output;
}

int launch_increment(Runtime& runtime,
                     const std::shared_ptr<deep_jit::ascend::Kernel>& kernel,
                     const int input) {
    DeviceBuffer output(sizeof(int));
    runtime.launch(kernel, {.num_blocks = 1}, output.data, input);
    DJ_ACL_CHECK(aclrtSynchronizeDevice());
    return copy_from_device<int>(output).front();
}

void test_environment(const fs::path& cache_root) {
    set_env("DJ_TEST_VALUE", "global");
    set_env("ASCEND_ENV_TEST_VALUE", "library");
    set_env("TEST_VALUE", "unprefixed");
    set_env("ASCEND_ENV_BOOL", "YeS");

    const deep_jit::Env env("ASCEND_ENV");
    DJ_HOST_ASSERT(env.get<std::string>("TEST_VALUE") == "library");
    DJ_HOST_ASSERT(env.get<bool>("BOOL") == true);
    unset_env("ASCEND_ENV_TEST_VALUE");
    DJ_HOST_ASSERT(env.get<std::string>("TEST_VALUE") == "global");
    unset_env("DJ_TEST_VALUE");
    DJ_HOST_ASSERT(not env.get<std::string>("TEST_VALUE").has_value());
    unset_env("ASCEND_ENV_BOOL");
    unset_env("TEST_VALUE");

    set_env("DJ_JIT_LAUNCH_TIMEOUT", "11");
    set_env("ASCEND_ENV_JIT_LAUNCH_TIMEOUT", "7");
    DJ_HOST_ASSERT(LaunchOptions::default_options(env).num_launch_timeout_secs == 7);
    unset_env("ASCEND_ENV_JIT_LAUNCH_TIMEOUT");
    DJ_HOST_ASSERT(LaunchOptions::default_options(env).num_launch_timeout_secs == 11);
    unset_env("DJ_JIT_LAUNCH_TIMEOUT");

    const auto first_cache = cache_root / "first";
    const auto second_cache = cache_root / "second";
    set_env("ASCEND_ENV_JIT_CACHE_DIR", first_cache.string() + ":" + second_cache.string());
    const auto disk_cache = deep_jit::DiskCache::from_env(env);
    DJ_HOST_ASSERT((disk_cache.paths == std::vector<fs::path>{first_cache, second_cache}));
    unset_env("ASCEND_ENV_JIT_CACHE_DIR");
}

void test_config() {
    const auto root = get_test_project_dir();
    const deep_jit::Config config(
        root, "ASCEND_CONFIG", "dependency-version",
        {root / "include_original"}, {"test_ascend/"});
    DJ_HOST_ASSERT(config.python_library_root == root);
    DJ_HOST_ASSERT(config.include_dirs == std::vector<fs::path>{root / "include_original"});
    const auto serialized = config.to_json().dump();
    DJ_HOST_ASSERT(serialized.find("\"extra_signature\":\"dependency-version\"") != std::string::npos);
    DJ_HOST_ASSERT(serialized.find("\"include_prefixes\":[\"test_ascend/\"]") != std::string::npos);

    expect_failure([] { deep_jit::Config({}, "ASCEND_CONFIG"); }, "root must not be empty");
    expect_failure([] { deep_jit::Config("relative", "ASCEND_CONFIG"); }, "root must be absolute");
    expect_failure([&] { deep_jit::Config(root, ""); }, "prefix must not be empty");
    expect_failure([&] { deep_jit::Config(root, "DJ"); }, "reserved for global environment variables");
    expect_failure([&] { deep_jit::Config(root, "ASCEND_CONFIG", {}, {fs::path{}}); },
                   "include directory must not be empty");
    expect_failure([&] { deep_jit::Config(root, "ASCEND_CONFIG", {}, {"relative"}); },
                   "include directory must be absolute");
}

void test_filesystem_and_command(const fs::path& cache_root) {
    deep_jit::make_dirs(cache_root);
    const auto path = cache_root / "binary_data";
    const std::string expected("a\0b", 3);
    deep_jit::write_file_sync(path, expected);
    DJ_HOST_ASSERT(deep_jit::read(path) == expected);
    deep_jit::safe_remove_all(path);
    DJ_HOST_ASSERT(not fs::exists(path));
    DJ_HOST_ASSERT(deep_jit::is_executable("/bin/sh"));

    DJ_HOST_ASSERT(deep_jit::call_external_command("sh -c 'printf stdout; printf stderr >&2'") == "stdoutstderr");
    expect_failure([] { deep_jit::call_external_command("sh -c 'exit 7'"); }, "exit code 7");

    const auto prefix = std::to_string(::getpid()) + "-";
    const auto first = deep_jit::get_uuid();
    const auto second = deep_jit::get_uuid();
    DJ_HOST_ASSERT(first.starts_with(prefix) and second.starts_with(prefix) and first != second);
}

void test_disk_and_memory_cache(const fs::path& cache_root) {
    const auto primary = cache_root / "disk_primary";
    const auto secondary = cache_root / "disk_secondary";
    deep_jit::DiskCache cache({primary, secondary});

    fs::path abandoned_path;
    {
        auto entry = cache.entry("abandoned", "digest");
        abandoned_path = entry.path;
        deep_jit::write_file_sync(entry.path / "partial", "partial");
    }
    DJ_HOST_ASSERT(not fs::exists(abandoned_path));

    fs::path committed_path;
    {
        auto entry = cache.entry("committed", "digest");
        deep_jit::write_file_sync(entry.path / "payload", "complete");
        committed_path = entry.commit();
    }
    DJ_HOST_ASSERT(deep_jit::read(committed_path / "payload") == "complete");
    DJ_HOST_ASSERT(cache.entry("committed", "digest").hit);
    expect_failure([&] { (void)cache.entry("invalid/tag", "digest"); }, "cache tag must contain only");

    struct Value { int number; };
    deep_jit::MemCache<std::string, Value> memory_cache;
    int num_factory_calls = 0;
    const auto first = memory_cache.get_or_create("same", [&] {
        ++ num_factory_calls;
        return std::make_shared<Value>(Value{7});
    });
    const auto second = memory_cache.get_or_create("same", [&] {
        ++ num_factory_calls;
        return std::make_shared<Value>(Value{8});
    });
    DJ_HOST_ASSERT(first == second and first->number == 7 and num_factory_calls == 1);
}

void test_json_and_lazy() {
    const deep_jit::json value = deep_jit::json::object_t {
        {"integer", 10},
        {"boolean", false},
        {"string", "line\n\"quoted\""},
        {"optional", std::optional<int>()},
        {"array", std::vector<int>{1, 2, 3}},
    };
    DJ_HOST_ASSERT(value.dump() == R"({"integer":10,"boolean":false,"string":"line\n\"quoted\"","optional":null,"array":[1,2,3]})");

    struct Value { int number; };
    int num_initializations = 0;
    deep_jit::LazyInit<Value> lazy([&] {
        ++ num_initializations;
        return std::make_shared<Value>(Value{17});
    });
    DJ_HOST_ASSERT(num_initializations == 0);
    DJ_HOST_ASSERT(lazy->number == 17 and num_initializations == 1);
    deep_jit::LazyInit<Value> empty(nullptr);
    expect_failure([&] { empty.get(); }, "lazy object must be initialized before use");
}

void test_parser(const fs::path& cache_root) {
    const auto root = get_test_project_dir();
    const auto source = std::string("#include <test_ascend/tracked_vector_add.hpp>\n");
    deep_jit::Parser original({root / "include_original"}, {"test_ascend/"});
    deep_jit::Parser same({root / "include_same_content"}, {"test_ascend/"});
    deep_jit::Parser changed({root / "include_changed_content"}, {"test_ascend/"});
    DJ_HOST_ASSERT(original.parse_into_hash(source) == same.parse_into_hash(source));
    DJ_HOST_ASSERT(original.parse_into_hash(source) != changed.parse_into_hash(source));
    DJ_HOST_ASSERT(original.parse_include(" # include <test_ascend/tracked_vector_add.hpp>") ==
                   "test_ascend/tracked_vector_add.hpp");
    expect_failure([&] { original.parse_into_hash("#include \"test_ascend/tracked_vector_add.hpp\"\n"); },
                   "non-standard include");

    deep_jit::Parser cycle({root / "include_cycle"}, {"test_ascend/"});
    expect_failure([&] { cycle.parse_into_hash("#include <test_ascend/circular_include_entry.hpp>\n"); },
                   "circular include");
    DJ_HOST_ASSERT(cycle.visiting.empty());

    const auto generated_root = cache_root / "generated_include";
    deep_jit::make_dirs(generated_root / "generated");
    deep_jit::write_file_sync(generated_root / "generated/dependency.hpp", "#pragma once\nconstexpr int kValue = 1;\n");
    deep_jit::Parser generated({generated_root}, {"generated/"});
    DJ_HOST_ASSERT(not generated.parse_into_hash("#include <generated/dependency.hpp>\n").empty());
}

void test_device(Runtime& runtime) {
    const auto& prop = runtime.device.get_prop();
    DJ_HOST_ASSERT(not prop.soc_name.empty());
    DJ_HOST_ASSERT(runtime.device.get_npu_arch() > 0);
    DJ_HOST_ASSERT(runtime.device.get_num_aicore_cores() > 0);
    DJ_HOST_ASSERT(runtime.device.get_num_vec_cores() > 0);
    DJ_HOST_ASSERT(runtime.device.get_num_cube_cores() > 0);
    DJ_HOST_ASSERT(runtime.device.get_num_vec_cores_per_ai_core() > 0);
    DJ_HOST_ASSERT(runtime.device.get_num_ubuf_bytes_per_vec_core() > 0);
}

void test_options(Runtime& runtime) {
    const auto& defaults = runtime.default_compiler_options;
    DJ_HOST_ASSERT(defaults.optimize_level == "2");
    DJ_HOST_ASSERT(defaults.arch == runtime.device.get_npu_arch());
    DJ_HOST_ASSERT(defaults.bisheng_flags.has_value());
    DJ_HOST_ASSERT(defaults.linker_flags.has_value());

    const auto overridden = defaults.override_with(CompilerOptions {
        .optimize_level = "3",
        .debug_info = true,
        .dump_asm = true,
        .extra_bisheng_flags = {"-DTEST_OPTION=1"},
        .extra_linker_flags = {"--test-linker-option"},
    });
    const auto bisheng_flags = overridden.get_bisheng_flags();
    DJ_HOST_ASSERT(std::ranges::find(bisheng_flags, "-O3") != bisheng_flags.end());
    DJ_HOST_ASSERT(std::ranges::find(bisheng_flags, "-DTEST_OPTION=1") != bisheng_flags.end());
    DJ_HOST_ASSERT(overridden.get_linker_flags().back() == "--test-linker-option");

    const auto launch_defaults = LaunchOptions::default_options(runtime.env);
    DJ_HOST_ASSERT(launch_defaults.num_ubuf_bytes == 0);
    DJ_HOST_ASSERT(launch_defaults.num_launch_timeout_secs == 10);
    const auto launch_overridden = launch_defaults.override_with(LaunchOptions {
        .num_blocks = 1,
        .num_ubuf_bytes = 1024,
        .num_launch_timeout_secs = 0,
    });
    DJ_HOST_ASSERT(launch_overridden.num_blocks == 1);
    DJ_HOST_ASSERT(launch_overridden.num_ubuf_bytes == 1024);
    DJ_HOST_ASSERT(launch_overridden.num_launch_timeout_secs == 0);

    auto bisheng_boundary = defaults;
    bisheng_boundary.bisheng_flags = std::vector<std::string>{"-DTEST_BOUNDARY=1", "ld.lld"};
    bisheng_boundary.linker_flags = std::vector<std::string>{"--test-linker-option"};
    auto linker_boundary = defaults;
    linker_boundary.bisheng_flags = std::vector<std::string>{"-DTEST_BOUNDARY=1"};
    linker_boundary.linker_flags = std::vector<std::string>{"ld.lld", "--test-linker-option"};
    const auto legacy_flatten = [](const CompilerOptions& options) {
        auto flags = options.get_bisheng_flags();
        flags.emplace_back("ld.lld");
        const auto linker_flags = options.get_linker_flags();
        flags.insert(flags.end(), linker_flags.begin(), linker_flags.end());
        return deep_jit::str::join(flags);
    };
    DJ_HOST_ASSERT(legacy_flatten(bisheng_boundary) == legacy_flatten(linker_boundary),
                   "compiler/linker boundary fixture must collide under the legacy flattening");
    DJ_HOST_ASSERT(runtime.cache_key(get_increment_source(1), bisheng_boundary) !=
                       runtime.cache_key(get_increment_source(1), linker_boundary),
                   "Bisheng and linker flag boundaries must affect the cache key");
}

void test_artifact_and_metadata(Runtime& runtime) {
    const auto source = get_increment_source(3);
    const auto artifact = runtime.compile_without_load("artifact_metadata", source);
    check_artifact(artifact, source);
    const auto metadata = deep_jit::read(artifact / "meta.json");
    DJ_HOST_ASSERT(metadata.find("\"command\":") != std::string::npos);
    DJ_HOST_ASSERT(metadata.find("\"compiler_info\":") != std::string::npos);
    DJ_HOST_ASSERT(metadata.find("\"compiler_options\":") != std::string::npos);
    DJ_HOST_ASSERT(metadata.find("-fcce-simt-lambda") != std::string::npos);
}

void test_increment_and_cache(Runtime& runtime) {
    const auto first_source = get_increment_source(3);
    const auto second_source = get_increment_source(4);
    const auto first_kernel = runtime.compile("scalar_increment", first_source);
    const auto first_kernel_again = runtime.compile("scalar_increment", first_source);
    const auto second_kernel = runtime.compile("scalar_increment", second_source);
    DJ_HOST_ASSERT(first_kernel == first_kernel_again);
    DJ_HOST_ASSERT(first_kernel != second_kernel);
    DJ_HOST_ASSERT(launch_increment(runtime, first_kernel, 9) == 12);
    DJ_HOST_ASSERT(launch_increment(runtime, first_kernel, 17) == 20);
    DJ_HOST_ASSERT(launch_increment(runtime, second_kernel, 9) == 13);
}

void test_launch_validation(Runtime& runtime) {
    const auto kernel = runtime.compile("launch_validation", get_increment_source(3));
    DeviceBuffer output(sizeof(int));
    expect_failure([&] { runtime.launch(kernel, {}, output.data, 1); }, "block count must be positive");
    expect_failure([&] { runtime.launch(kernel, {.num_blocks = 0}, output.data, 1); },
                   "block count must be positive");
    expect_failure([&] { runtime.launch(kernel, {.num_blocks = 1, .num_ubuf_bytes = -1}, output.data, 1); },
                   "dynamic UB size must not be negative");
    expect_failure([&] { runtime.launch(kernel, {.num_blocks = 1, .num_launch_timeout_secs = -1}, output.data, 1); },
                   "launch timeout is out of range");
    expect_failure([&] { runtime.launch(kernel, {.num_blocks = 1, .num_launch_timeout_secs = 65536}, output.data, 1); },
                   "launch timeout is out of range");
}

void test_mixed_arguments(Runtime& runtime) {
    const auto kernel = runtime.compile(
        "mixed_arguments", "#include <kernels/mixed_argument_echo.hpp>\n");
    DeviceBuffer output_i8(sizeof(int8_t));
    DeviceBuffer output_f32(sizeof(float));
    DeviceBuffer output_i64(sizeof(int64_t));
    DeviceBuffer output_f64(sizeof(double));
    const int8_t value_i8 = -7;
    const float value_f32 = 1.5f;
    const int64_t value_i64 = 1234567890123ll;
    const double value_f64 = 0.25;
    runtime.launch(
        kernel, {.num_blocks = 1},
        output_i8.data, value_i8,
        output_f32.data, value_f32,
        output_i64.data, value_i64,
        output_f64.data, value_f64);
    DJ_ACL_CHECK(aclrtSynchronizeDevice());
    DJ_HOST_ASSERT(copy_from_device<int8_t>(output_i8).front() == value_i8);
    DJ_HOST_ASSERT(copy_from_device<float>(output_f32).front() == value_f32);
    DJ_HOST_ASSERT(copy_from_device<int64_t>(output_i64).front() == value_i64);
    DJ_HOST_ASSERT(copy_from_device<double>(output_f64).front() == value_f64);
}

void test_dynamic_ubuf(Runtime& runtime) {
    const auto kernel = runtime.compile(
        "dynamic_ubuf", "#include <kernels/dynamic_ubuf_checksum.hpp>\n");
    for (const int num_bytes : {32, 1024, 8192, 65536}) {
        DeviceBuffer output(2 * sizeof(uint32_t));
        runtime.launch(
            kernel, {.num_blocks = 1, .num_ubuf_bytes = num_bytes},
            output.data, static_cast<uint32_t>(num_bytes));
        DJ_ACL_CHECK(aclrtSynchronizeDevice());
        const auto result = copy_from_device<uint32_t>(output);
        uint32_t expected_checksum = 0;
        for (int offset = 0; offset < num_bytes; ++ offset)
            expected_checksum += static_cast<uint8_t>(offset);
        DJ_HOST_ASSERT(result[0] == static_cast<uint32_t>(num_bytes));
        DJ_HOST_ASSERT(result[1] == expected_checksum);
    }
}

void test_static_ubuf(Runtime& runtime) {
    constexpr int num_bytes = 64 * 1024;
    std::vector<uint8_t> input(num_bytes);
    for (int index = 0; index < num_bytes; ++ index)
        input[index] = static_cast<uint8_t>(index);
    DeviceBuffer input_device(num_bytes);
    DeviceBuffer output_device(num_bytes);
    copy_to_device(input_device, input);
    const auto kernel = runtime.compile(
        "static_ubuf", "#include <kernels/static_ubuf_round_trip.hpp>\n");
    runtime.launch(kernel, {.num_blocks = 1}, input_device.data, output_device.data);
    DJ_ACL_CHECK(aclrtSynchronizeDevice());
    DJ_HOST_ASSERT(copy_from_device<uint8_t>(output_device) == input);
}

void test_simd_and_simt(Runtime& runtime) {
    constexpr int num_elements = 64;
    std::vector<float> lhs(num_elements), rhs(num_elements);
    for (int index = 0; index < num_elements; ++ index) {
        lhs[index] = static_cast<float>(index);
        rhs[index] = static_cast<float>(3 * index);
    }
    DeviceBuffer lhs_device(num_elements * sizeof(float));
    DeviceBuffer rhs_device(num_elements * sizeof(float));
    DeviceBuffer output_device(num_elements * sizeof(float));
    copy_to_device(lhs_device, lhs);
    copy_to_device(rhs_device, rhs);
    const auto simd_kernel = runtime.compile(
        "simd_vector_add", "#include <kernels/simd_vector_add.hpp>\n");
    runtime.launch(simd_kernel, {.num_blocks = 1}, lhs_device.data, rhs_device.data, output_device.data);
    DJ_ACL_CHECK(aclrtSynchronizeDevice());
    const auto output = copy_from_device<float>(output_device);
    for (int index = 0; index < num_elements; ++ index)
        DJ_HOST_ASSERT(output[index] == lhs[index] + rhs[index]);

    const auto simt_kernel = runtime.compile(
        "simt_sequence", "#include <kernels/simt_sequence.hpp>\n");
    runtime.launch(simt_kernel, {.num_blocks = 1}, output_device.data);
    DJ_ACL_CHECK(aclrtSynchronizeDevice());
    const auto sequence = copy_from_device<float>(output_device);
    for (int index = 0; index < num_elements; ++ index)
        DJ_HOST_ASSERT(sequence[index] == static_cast<float>(index + 1));
}

void test_macro_selected_argument(Runtime& runtime) {
    const auto source = std::string("#include <kernels/macro_selected_argument.hpp>\n");
    const auto narrow_kernel = runtime.compile("macro_argument_narrow", source);
    const auto wide_kernel = runtime.compile("macro_argument_wide", source, CompilerOptions {
        .extra_bisheng_flags = {"-DTEST_WIDE_ARGUMENT=1"},
    });
    DeviceBuffer output(sizeof(int64_t));
    const int32_t narrow_value = -12345;
    runtime.launch(narrow_kernel, {.num_blocks = 1}, output.data, narrow_value);
    DJ_ACL_CHECK(aclrtSynchronizeDevice());
    DJ_HOST_ASSERT(copy_from_device<int64_t>(output).front() == narrow_value);
    const int64_t wide_value = 0x123456789abcdefll;
    runtime.launch(wide_kernel, {.num_blocks = 1}, output.data, wide_value);
    DJ_ACL_CHECK(aclrtSynchronizeDevice());
    DJ_HOST_ASSERT(copy_from_device<int64_t>(output).front() == wide_value);
}

void test_include_contents(const fs::path& cache_root) {
    const auto root = get_test_project_dir();
    set_env("INCLUDE_ORIGINAL_JIT_CACHE_DIR", (cache_root / "include_shared").string());
    set_env("INCLUDE_SAME_JIT_CACHE_DIR", (cache_root / "include_shared").string());
    set_env("INCLUDE_CHANGED_JIT_CACHE_DIR", (cache_root / "include_shared").string());
    const auto source = std::string("#include <test_ascend/tracked_vector_add.hpp>\n");
    const auto original = make_runtime(root / "include_original", "INCLUDE_ORIGINAL");
    const auto same = make_runtime(root / "include_same_content", "INCLUDE_SAME");
    const auto changed = make_runtime(root / "include_changed_content", "INCLUDE_CHANGED");
    const auto original_artifact = original->compile_without_load("tracked_include", source);
    const auto same_artifact = same->compile_without_load("tracked_include", source);
    const auto changed_artifact = changed->compile_without_load("tracked_include", source);
    DJ_HOST_ASSERT(original_artifact == same_artifact);
    DJ_HOST_ASSERT(original_artifact != changed_artifact);
    unset_env("INCLUDE_ORIGINAL_JIT_CACHE_DIR");
    unset_env("INCLUDE_SAME_JIT_CACHE_DIR");
    unset_env("INCLUDE_CHANGED_JIT_CACHE_DIR");
}

void test_relocated_wheel_include_dir(const fs::path& cache_root) {
    const auto root = get_test_project_dir();
    const auto wheel_cache = cache_root / "wheel_include_relocation";
    const auto environment_a_include = cache_root / "environment_a/site-packages/deep_jit/include";
    const auto environment_b_include = cache_root / "environment_b/site-packages/deep_jit/include";
    const auto header = fs::path("deep_jit/wheel_marker.hpp");
    deep_jit::make_dirs((environment_a_include / header).parent_path());
    deep_jit::make_dirs((environment_b_include / header).parent_path());
    deep_jit::write_file_sync(environment_a_include / header, "#pragma once\n");
    deep_jit::write_file_sync(environment_b_include / header, "#pragma once\n");

    const auto source = "#include <deep_jit/wheel_marker.hpp>\n" + get_increment_source(73);
    const auto make_environment_runtime = [&](const fs::path& include_dir) {
        auto runtime = std::make_shared<Runtime>(deep_jit::Config(
            root, "WHEEL_INCLUDE", "ascend-test-signature",
            std::vector<fs::path>{include_dir, root},
            std::vector<std::string>{"deep_jit/", "kernels/"}));
        runtime->default_compiler_options.bisheng_flags->emplace_back("-fcce-simt-lambda");
        return runtime;
    };

    set_env("WHEEL_INCLUDE_JIT_CACHE_DIR", wheel_cache.string());
    const auto environment_a = make_environment_runtime(environment_a_include);
    const auto first_artifact = environment_a->compile_without_load("wheel_include_relocation", source);
    check_artifact(first_artifact, source);
    const auto first_metadata = deep_jit::read(first_artifact / "meta.json");
    DJ_HOST_ASSERT(first_metadata.find(environment_a_include.string()) != std::string::npos);

    const auto environment_b = make_environment_runtime(environment_b_include);
    DJ_HOST_ASSERT(environment_a->cache_key(source, environment_a->default_compiler_options) ==
                       environment_b->cache_key(source, environment_b->default_compiler_options),
                   "relocating an identical wheel include directory changed the Ascend cache key");
    environment_b->backend.toolkit.bisheng = cache_root / "compiler_must_not_run";
    environment_b->backend.toolkit.ld_lld = cache_root / "linker_must_not_run";
    const auto second_artifact = environment_b->compile_without_load("wheel_include_relocation", source);
    DJ_HOST_ASSERT(second_artifact == first_artifact,
                   "relocating an identical wheel include directory triggered Ascend compilation");
    DJ_HOST_ASSERT(deep_jit::read(second_artifact / "meta.json") == first_metadata,
                   "an Ascend cache hit rewrote metadata after the wheel include directory moved");
    check_tmp_is_empty(wheel_cache);
    unset_env("WHEEL_INCLUDE_JIT_CACHE_DIR");
}

void test_untracked_dependency() {
    const auto root = get_test_project_dir();
    const auto source = std::string("#include <kernels/untracked_dependency.hpp>\n");
    const auto original = make_runtime(
        root / "include_original", "UNTRACKED_ORIGINAL", "untracked",
        root / "third_party_original");
    const auto changed = make_runtime(
        root / "include_original", "UNTRACKED_CHANGED", "untracked",
        root / "third_party_changed_content");
    DJ_HOST_ASSERT(original->cache_key(source, original->default_compiler_options) ==
                   changed->cache_key(source, changed->default_compiler_options));
}

void test_kernel_count(Runtime& runtime) {
    const auto mixed_source = std::string("#include <kernels/mixed_kernel_metadata.hpp>\n");
    DJ_HOST_ASSERT(runtime.compile("mixed_kernel_metadata", mixed_source) != nullptr);

    const auto no_kernel_source = std::string("#include <kernels/no_entry_kernel.hpp>\n");
    check_artifact(runtime.compile_without_load("no_entry", no_kernel_source), no_kernel_source);
    expect_failure([&] { runtime.compile("no_entry", no_kernel_source); }, "expected exactly one kernel");

    const auto multiple_source = std::string("#include <kernels/multiple_entry_kernels.hpp>\n");
    check_artifact(runtime.compile_without_load("multiple_entries", multiple_source), multiple_source);
    expect_failure([&] { runtime.compile("multiple_entries", multiple_source); }, "expected exactly one kernel");
}

void test_compiler_options(Runtime& runtime) {
    const auto source = std::string("#include <kernels/compiler_option_value.hpp>\n");
    const auto first = runtime.compile("compiler_option", source, CompilerOptions {
        .extra_bisheng_flags = {"-DTEST_COMPILER_OPTION=31"},
    });
    const auto second = runtime.compile("compiler_option", source, CompilerOptions {
        .extra_bisheng_flags = {"-DTEST_COMPILER_OPTION=47"},
    });
    DeviceBuffer output(sizeof(int));
    runtime.launch(first, {.num_blocks = 1}, output.data);
    DJ_ACL_CHECK(aclrtSynchronizeDevice());
    DJ_HOST_ASSERT(copy_from_device<int>(output).front() == 31);
    runtime.launch(second, {.num_blocks = 1}, output.data);
    DJ_ACL_CHECK(aclrtSynchronizeDevice());
    DJ_HOST_ASSERT(copy_from_device<int>(output).front() == 47);
}

void test_compiler_failure_cleanup(Runtime& runtime, const fs::path& cache_root) {
    expect_failure(
        [&] { runtime.compile_without_load("invalid_source", "this is not valid CCE source\n"); },
        "command failed");
    check_tmp_is_empty(cache_root);
}

void test_dump_assembly(const fs::path& cache_root) {
    set_env("DUMP_TEST_JIT_CACHE_DIR", (cache_root / "dump").string());
    const auto runtime = make_runtime(
        get_test_project_dir() / "include_original", "DUMP_TEST", "dump-test");
    runtime->default_compiler_options.dump_asm = true;
    const auto source = get_increment_source(5);
    const auto artifact = runtime->compile_without_load("dump_assembly", source);
    bool has_assembly = false;
    for (const auto& entry : fs::directory_iterator(artifact / "asm"))
        has_assembly |= entry.path().extension() == ".asci" and entry.file_size() > 0;
    DJ_HOST_ASSERT(has_assembly, "Bisheng assembly dump was not generated");

    const auto artifact_without_dump = runtime->compile_without_load(
        "dump_cache_hit", source, CompilerOptions {.dump_asm = false});
    DJ_HOST_ASSERT(not fs::exists(artifact_without_dump / "asm"));
    DJ_HOST_ASSERT(runtime->compile_without_load("dump_cache_hit", source) == artifact_without_dump);
    DJ_HOST_ASSERT(not fs::exists(artifact_without_dump / "asm"),
                   "dump option unexpectedly rebuilt an existing cache entry");
    unset_env("DUMP_TEST_JIT_CACHE_DIR");
}

void test_multiple_runtimes(const fs::path& cache_root) {
    set_env("MULTI_A_JIT_CACHE_DIR", (cache_root / "multiple_runtimes").string());
    set_env("MULTI_B_JIT_CACHE_DIR", (cache_root / "multiple_runtimes").string());
    const auto first = make_runtime(get_test_project_dir() / "include_original", "MULTI_A", "multi-runtime");
    const auto second = make_runtime(get_test_project_dir() / "include_original", "MULTI_B", "multi-runtime");
    const auto source = get_increment_source(6);
    const auto first_artifact = first->compile_without_load("multiple_runtimes", source);
    const auto second_artifact = second->compile_without_load("multiple_runtimes", source);
    DJ_HOST_ASSERT(first_artifact == second_artifact);
    DJ_HOST_ASSERT(first->compile("multiple_runtimes", source) != second->compile("multiple_runtimes", source));
    unset_env("MULTI_A_JIT_CACHE_DIR");
    unset_env("MULTI_B_JIT_CACHE_DIR");
}

void test_secondary_cache(const fs::path& cache_root) {
    const auto primary = cache_root / "secondary_primary";
    const auto secondary = cache_root / "secondary_lookup";
    set_env("SECONDARY_WRITER_JIT_CACHE_DIR", secondary.string());
    const auto writer = make_runtime(
        get_test_project_dir() / "include_original", "SECONDARY_WRITER", "secondary-cache");
    const auto source = get_increment_source(12);
    const auto secondary_artifact = writer->compile_without_load("secondary_cache", source);
    set_env("SECONDARY_READER_JIT_CACHE_DIR", primary.string() + ":" + secondary.string());
    const auto reader = make_runtime(
        get_test_project_dir() / "include_original", "SECONDARY_READER", "secondary-cache");
    DJ_HOST_ASSERT(reader->compile_without_load("secondary_cache", source) == secondary_artifact);
    DJ_HOST_ASSERT(not fs::exists(primary / "cache"));
    unset_env("SECONDARY_WRITER_JIT_CACHE_DIR");
    unset_env("SECONDARY_READER_JIT_CACHE_DIR");
}

void test_architecture_override(const fs::path& cache_root) {
    set_env("ARCH_TEST_JIT_CACHE_DIR", (cache_root / "architecture").string());
    const auto runtime = make_runtime(
        get_test_project_dir() / "include_original", "ARCH_TEST", "architecture-test");
    const auto arch = runtime->device.get_npu_arch();
    const auto options = CompilerOptions {.arch = arch};
    const auto artifact = runtime->compile_without_load("architecture", get_increment_source(2), options);
    const auto metadata = deep_jit::read(artifact / "meta.json");
    DJ_HOST_ASSERT(metadata.find(std::format("\"arch\":{}", arch)) != std::string::npos);
    unset_env("ARCH_TEST_JIT_CACHE_DIR");
}

void test_launch_blocking(Runtime& runtime) {
    set_env("ASCEND_LAUNCH_BLOCKING", "1");
    const auto kernel = runtime.compile("launch_blocking", get_increment_source(8));
    DJ_HOST_ASSERT(launch_increment(runtime, kernel, 3) == 11);
    unset_env("ASCEND_LAUNCH_BLOCKING");
}

void test_launch_overhead(Runtime& runtime) {
    const auto kernel = runtime.compile(
        "launch_overhead", "#include <kernels/launch_overhead_noop.hpp>\n");
    constexpr int num_warmups = 100;
    constexpr int num_launches = 10000;
    for (int index = 0; index < num_warmups; ++ index)
        runtime.launch(kernel, {.num_blocks = 1}, 0);
    DJ_ACL_CHECK(aclrtSynchronizeDevice());
    const auto begin = std::chrono::steady_clock::now();
    for (int index = 0; index < num_launches; ++ index)
        runtime.launch(kernel, {.num_blocks = 1}, 0);
    DJ_ACL_CHECK(aclrtSynchronizeDevice());
    const auto elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
    std::printf("Ascend launch overhead: %.3f us\n", elapsed / num_launches);
}

void test_invalid_tag(Runtime& runtime) {
    expect_failure([&] { runtime.compile("invalid/tag", get_increment_source(1)); },
                   "cache tag must contain only letters, digits, or underscores");
    expect_failure([&] { runtime.compile("", get_increment_source(1)); },
                   "cache tag must contain only letters, digits, or underscores");
}

void test_python_api(pybind11::module_& module, const std::shared_ptr<Runtime>& runtime) {
    init_python_api_jit(get_test_project_dir().string());
    DJ_HOST_ASSERT(module.attr("get_jit")().cast<std::shared_ptr<Runtime>>() != nullptr);
    DJ_HOST_ASSERT(python_api_jit.get() != runtime);
}

void run_tests(pybind11::module_ module) {
    const char* cache_root_env = std::getenv("DEEP_JIT_ASCEND_TEST_CACHE_ROOT");
    DJ_HOST_ASSERT(cache_root_env != nullptr, "DEEP_JIT_ASCEND_TEST_CACHE_ROOT must be set");
    const auto cache_root = fs::absolute(cache_root_env).lexically_normal();

    run_test("environment precedence", [&] { test_environment(cache_root); });
    run_test("configuration", test_config);
    run_test("filesystem and command utilities", [&] { test_filesystem_and_command(cache_root); });
    run_test("disk and memory cache", [&] { test_disk_and_memory_cache(cache_root); });
    run_test("JSON and lazy init", test_json_and_lazy);
    run_test("include parser", [&] { test_parser(cache_root); });

    const auto runtime = make_runtime(get_test_project_dir() / "include_original");
    run_test("Ascend device", [&] { test_device(*runtime); });
    run_test("compiler and launch options", [&] { test_options(*runtime); });
    run_test("artifact and metadata", [&] { test_artifact_and_metadata(*runtime); });
    run_test("template source and runtime arguments", [&] { test_increment_and_cache(*runtime); });
    run_test("launch option validation", [&] { test_launch_validation(*runtime); });
    run_test("mixed kernel arguments", [&] { test_mixed_arguments(*runtime); });
    run_test("dynamic UB", [&] { test_dynamic_ubuf(*runtime); });
    run_test("static UB", [&] { test_static_ubuf(*runtime); });
    run_test("SIMD and SIMT", [&] { test_simd_and_simt(*runtime); });
    run_test("macro-selected argument ABI", [&] { test_macro_selected_argument(*runtime); });
    run_test("tracked include contents", [&] { test_include_contents(cache_root); });
    run_test("relocated wheel include directory", [&] { test_relocated_wheel_include_dir(cache_root); });
    run_test("untracked dependency", test_untracked_dependency);
    run_test("kernel count", [&] { test_kernel_count(*runtime); });
    run_test("compiler options", [&] { test_compiler_options(*runtime); });
    run_test("multiple runtimes", [&] { test_multiple_runtimes(cache_root); });
    run_test("secondary disk cache", [&] { test_secondary_cache(cache_root); });
    run_test("architecture override", [&] { test_architecture_override(cache_root); });
    run_test("launch blocking", [&] { test_launch_blocking(*runtime); });
    run_test("invalid cache tag", [&] { test_invalid_tag(*runtime); });
    run_test("compiler failure cleanup", [&] { test_compiler_failure_cleanup(*runtime, cache_root); });
    run_test("assembly dump", [&] { test_dump_assembly(cache_root); });
    run_test("launch overhead", [&] { test_launch_overhead(*runtime); });
    run_test("Python API", [&] { test_python_api(module, runtime); });
    check_tmp_is_empty(cache_root);

    std::puts("All DeepJIT Ascend tests passed");
}

void init_python_api_jit(const std::string& library_root) {
    const auto root = fs::absolute(library_root).lexically_normal();
    python_api_jit = deep_jit::create_lazy_jit<deep_jit::Ascend>(deep_jit::Config(
        root, "ASCEND_PYTHON_API", "python-api-test",
        {root / "include_original", root}, {"test_ascend/", "kernels/"}));
}

void prepare_process_runtime() {
    process_runtime = make_runtime(
        get_test_project_dir() / "include_original", "ASCEND_PROCESS", "process-test");
}

std::string compile_for_process(const std::string& tag, const int bias) {
    DJ_HOST_ASSERT(process_runtime != nullptr);
    return process_runtime->compile_without_load(tag, get_increment_source(bias)).string();
}

int launch_for_process(const std::string& tag, const int bias) {
    DJ_HOST_ASSERT(process_runtime != nullptr);
    const auto kernel = process_runtime->compile(tag, get_increment_source(bias));
    return launch_increment(*process_runtime, kernel, 1);
}

std::string publish_disk_cache_entry(const std::string& cache_root,
                                     const std::string& owner,
                                     const std::string& ready_path,
                                     const std::string& start_path) {
    deep_jit::DiskCache cache({fs::absolute(cache_root).lexically_normal()});
    auto entry = cache.entry("atomic_publication", "digest");
    DJ_HOST_ASSERT(not entry.hit);
    deep_jit::write_file_sync(entry.path / "owner_a", owner);
    deep_jit::write_file_sync(entry.path / "owner_b", owner);
    deep_jit::write_file_sync(ready_path, "ready");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (not fs::exists(start_path)) {
        DJ_HOST_ASSERT(std::chrono::steady_clock::now() < deadline,
                       "cache writer did not leave the commit barrier");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const auto published_path = entry.commit();
    const auto winner = deep_jit::read(published_path / "owner_a");
    DJ_HOST_ASSERT(deep_jit::read(published_path / "owner_b") == winner);
    return winner;
}

void prepare_gil_runtime() {
    gil_runtime = make_runtime(
        get_test_project_dir() / "include_original", "ASCEND_GIL", "gil-test");
}

std::string compile_for_gil_test() {
    DJ_HOST_ASSERT(gil_runtime != nullptr);
    return gil_runtime->compile_without_load("gil_compile", get_increment_source(21)).string();
}

int run_registered_jit(const int bias) {
    auto runtime = python_api_jit.get();
    const auto kernel = runtime->compile("python_api", get_increment_source(bias));
    return launch_increment(*runtime, kernel, 1);
}

int get_registered_jit_arch() {
    return *python_api_jit->default_compiler_options.arch;
}

void compile_invalid_registered_jit() {
    python_api_jit->compile_without_load("python_api_invalid", "this is not valid CCE source\n");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    deep_jit::register_python_api(module, python_api_jit);
    module.def("init_jit", &init_python_api_jit);
    module.def("run_tests", &run_tests);
    module.def("prepare_process_runtime", &prepare_process_runtime);
    module.def("compile_for_process", &compile_for_process);
    module.def("launch_for_process", &launch_for_process);
    module.def("publish_disk_cache_entry", &publish_disk_cache_entry);
    module.def("prepare_gil_runtime", &prepare_gil_runtime);
    module.def("compile_for_gil_test", &compile_for_gil_test);
    module.def("run_registered_jit", &run_registered_jit);
    module.def("get_registered_jit_arch", &get_registered_jit_arch);
    module.def("compile_invalid_registered_jit", &compile_invalid_registered_jit);
}
