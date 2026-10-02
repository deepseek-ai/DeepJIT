#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <deep_jit/backend/rocm/backend.hpp>
#include <deep_jit/python_api.hpp>

namespace {

using JIT = deep_jit::Runtime<deep_jit::ROCm>;
using CompilerOptions = JIT::CompilerOptions;
using LaunchOptions = JIT::LaunchOptions;
using Kernel = JIT::Kernel;

inline deep_jit::LazyInit<JIT> jit(nullptr);
const std::string increment_source = R"(
#include <hip/hip_runtime.h>
extern "C" __global__ void increment(int* output, int value) { *output = value + 1; }
)";
const std::string noop_source = R"(
#include <hip/hip_runtime.h>
extern "C" __global__ void noop() {}
)";

void expect_error(const std::function<void()>& function, const std::string& message) {
    try {
        function();
    } catch (const std::exception& error) {
        DJ_HOST_ASSERT(std::string(error.what()).find(message) != std::string::npos,
                       "expected '{}', got '{}'", message, error.what());
        return;
    }
    DJ_PANIC("expected failure containing '{}'", message);
}

CompilerOptions explicit_options() {
    return {.optimize_level = "3", .fast_math = false, .compiler_verbose = false,
            .check_no_spills = false, .check_no_local_memory = false, .with_line_info = false,
            .dump_llvm_ir = false, .dump_isa = false, .arch = "gfx90a:xnack-",
            .hipcc_flags = std::vector<std::string>{"-std=c++20"}};
}

std::string options_hash(const CompilerOptions& options) {
    deep_jit::hash::FNV1a hash;
    options.update_hash(hash);
    return hash.get_hex_digest();
}

void test_options(const std::filesystem::path& root) {
    auto options = explicit_options();
    options.fast_math = true;
    options.with_line_info = true;
    options.check_no_spills = true;
    options.extra_hipcc_flags = {"-DFIRST=1"};
    const auto overridden = options.override_with({.optimize_level = "0", .fast_math = false,
                                                   .check_no_spills = false, .with_line_info = false,
                                                   .extra_hipcc_flags = {"-DSECOND=2"}});
    DJ_HOST_ASSERT(*overridden.optimize_level == "0" and not *overridden.fast_math and
                   not *overridden.check_no_spills and not *overridden.with_line_info);
    DJ_HOST_ASSERT(overridden.extra_hipcc_flags == std::vector<std::string>({"-DFIRST=1", "-DSECOND=2"}));
    DJ_HOST_ASSERT(*options.fast_math and *options.with_line_info);
    DJ_HOST_ASSERT(options_hash(options) == options_hash(options.override_with({})));

    const auto baseline = explicit_options();
    for (const auto& change: std::vector<CompilerOptions>{
             {.optimize_level = "0"}, {.fast_math = true}, {.compiler_verbose = true},
             {.check_no_spills = true}, {.check_no_local_memory = true}, {.with_line_info = true},
             {.dump_llvm_ir = true}, {.dump_isa = true}, {.arch = "gfx942"}, {.extra_hipcc_flags = {"-DOTHER=1"}}}) {
        DJ_HOST_ASSERT(options_hash(baseline) != options_hash(baseline.override_with(change)));
    }
    DJ_HOST_ASSERT(options_hash(baseline.override_with({.hipcc_flags = std::vector<std::string>{"-DA=B C"}})) !=
                   options_hash(baseline.override_with({.hipcc_flags = std::vector<std::string>{"-DA=B", "C"}})));
    expect_error([&] { (void)baseline.override_with({.arch = "sm_90"}).get_flags(); }, "architecture");
    expect_error([&] { (void)baseline.override_with({.arch = "gfx90a;false"}).get_flags(); }, "architecture");
    expect_error([&] { (void)baseline.override_with({.optimize_level = "9"}).get_flags(); }, "optimization");
    expect_error([&] { (void)baseline.override_with({.extra_hipcc_flags = {"-Xptxas=-v"}}).get_flags(); }, "unsupported CUDA");

    auto launch = LaunchOptions::default_options(deep_jit::Env("ROCM_TEST"));
    DJ_HOST_ASSERT(not launch.stream);
    launch.stream = reinterpret_cast<hipStream_t>(uintptr_t(1));
    launch.num_smem_bytes = 1024;
    launch.cooperative = true;
    const auto updated = launch.override_with({.stream = static_cast<hipStream_t>(nullptr), .num_smem_bytes = 0,
                                               .cooperative = false});
    DJ_HOST_ASSERT(updated.stream.has_value() and *updated.stream == nullptr);
    DJ_HOST_ASSERT(updated.num_smem_bytes == 0 and not *updated.cooperative);
    DJ_HOST_ASSERT(launch.num_smem_bytes == 1024 and *launch.cooperative);

    const deep_jit::Config config(root, "ROCM_TEST");
    deep_jit::write_file_sync(root / "hook.py", "# first\n");
    auto hook = baseline.override_with({.post_hook = "hook.py"});
    const auto first = hook.get_post_hook_hash(config);
    deep_jit::write_file_sync(root / "hook.py", "# second\n");
    DJ_HOST_ASSERT(first != hook.get_post_hook_hash(config));
    DJ_HOST_ASSERT(baseline.get_post_hook_hash(config).empty());
    DJ_HOST_ASSERT(hook.to_json().dump().find("hook.py") != std::string::npos);

    deep_jit::ROCm::CompilerInfo info{.path = "/one/hipcc", .version = "HIP version: 10.0\nAMD clang version 22"};
    const auto first_compiler = info.get_hash();
    info.path = "/two/hipcc";
    DJ_HOST_ASSERT(first_compiler != info.get_hash());
    const auto second_compiler = info.get_hash();
    info.version += " updated";
    DJ_HOST_ASSERT(second_compiler != info.get_hash());
    const auto third_compiler = info.get_hash();
    info.environment = {"HIPCC_COMPILE_FLAGS_APPEND=-DFOO=1"};
    DJ_HOST_ASSERT(third_compiler != info.get_hash());
}

std::string metadata_kernel(const int sgpr, const int vgpr, const int scratch) {
    return "  - .args:\n      - .name: argument\n        .size: 8\n"
           "    .name: example\n    .symbol: example.kd\n"
           "    .sgpr_spill_count: " + std::to_string(sgpr) + "\n"
           "    .vgpr_spill_count: " + std::to_string(vgpr) + "\n"
           "    .private_segment_fixed_size: " + std::to_string(scratch) + "\n";
}

void test_metadata() {
    const std::string prefix = "AMDGPU Metadata: ---\namdhsa.kernels:\n";
    const auto good = prefix + metadata_kernel(0, 0, 0) + "amdhsa.version:\n  - 1\n  - 2\n...\n";
    deep_jit::ROCm::validate_metadata(good, true, true);
    deep_jit::ROCm::validate_metadata(prefix + metadata_kernel(0, 0, 128), true, false);
    deep_jit::ROCm::validate_metadata(prefix + metadata_kernel(1, 2, 0), false, true);
    deep_jit::ROCm::validate_metadata("amdhsa.kernels: []\n", true, true);
    expect_error([&] { deep_jit::ROCm::validate_metadata(prefix + metadata_kernel(1, 0, 0), true, false); }, "sgpr_spill_count");
    expect_error([&] { deep_jit::ROCm::validate_metadata(prefix + metadata_kernel(0, 1, 0), true, false); }, "vgpr_spill_count");
    expect_error([&] { deep_jit::ROCm::validate_metadata(prefix + metadata_kernel(0, 0, 4), false, true); }, "private_segment");
    expect_error([&] { deep_jit::ROCm::validate_metadata(prefix + metadata_kernel(0, 0, 0) +
                                                       "    .uses_dynamic_stack: true\n", false, true); }, "dynamic private stack");
    expect_error([&] { deep_jit::ROCm::validate_metadata(prefix + metadata_kernel(0, 0, 0) +
                                                       "    .uses_dynamic_stack: \n", false, true); }, "invalid empty AMDGPU");
    expect_error([&] { deep_jit::ROCm::validate_metadata("", true, true); }, "missing AMDGPU");
    expect_error([&] { deep_jit::ROCm::validate_metadata(prefix, true, true); }, "missing AMDGPU");
    expect_error([&] { deep_jit::ROCm::validate_metadata(good + good, true, true); }, "multiple AMDGPU");
    expect_error([&] { deep_jit::ROCm::validate_metadata(prefix + metadata_kernel(0, 0, 0) +
                                                       "    .sgpr_spill_count: 0\n", true, false); }, "duplicate AMDGPU");
    auto missing = metadata_kernel(0, 0, 0);
    const std::string field = "    .vgpr_spill_count: 0\n";
    missing.erase(missing.find(field), field.size());
    expect_error([&] { deep_jit::ROCm::validate_metadata(prefix + metadata_kernel(0, 0, 0) + missing, true, true); },
                 "missing AMDGPU .vgpr_spill_count");
    expect_error([&] { deep_jit::ROCm::validate_metadata(prefix + metadata_kernel(-1, 0, 0), true, false); }, "invalid AMDGPU");
}

void init_jit(const std::string& root) {
    jit = deep_jit::create_lazy_jit<deep_jit::ROCm>(deep_jit::Config(
        std::filesystem::absolute(root), "ROCM_TEST", {}, {std::filesystem::absolute(root) / "include"}, {"test/"}));
}

std::string compile_only(const std::string& name, const std::string& source, const bool checks,
                         const bool dumps, const std::string& hook) {
    CompilerOptions options{.check_no_spills = checks, .check_no_local_memory = checks,
                            .dump_llvm_ir = dumps, .dump_isa = dumps};
    if (not hook.empty())
        options.post_hook = hook;
    return jit->compile_without_load(name, source, options).string();
}

void build_fixture(const std::string& directory, const std::string& source, const bool checks,
                   const bool dumps, const std::string& hook, const bool disable_tools) {
    const deep_jit::Env env("ROCM_BUILD_TEST");
    const deep_jit::Config config(std::filesystem::absolute(directory), "ROCM_BUILD_TEST");
    deep_jit::ROCm backend(env);
    if (disable_tools) {
        backend.toolkit.llvm_objdump.reset();
        backend.toolkit.llvm_readobj.reset();
    }
    auto options = explicit_options().override_with({.check_no_spills = checks, .check_no_local_memory = checks,
                                                     .dump_llvm_ir = dumps, .dump_isa = dumps});
    if (not hook.empty())
        options.post_hook = hook;
    backend.compile(source, directory, env, config, options);
}

LaunchOptions launch_options(const dim3 grid = dim3(1), const dim3 block = dim3(1)) {
    return LaunchOptions::default_options(deep_jit::Env("ROCM_TEST")).override_with({.grid_dim = grid, .block_dim = block});
}

template <typename T>
struct Buffer {
    T* ptr = nullptr;
    explicit Buffer(const std::size_t count) { DJ_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T))); }
    ~Buffer() { if (ptr) (void)hipFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    std::vector<T> read(const std::size_t count) const {
        std::vector<T> result(count);
        DJ_HIP_CHECK(hipMemcpy(result.data(), ptr, count * sizeof(T), hipMemcpyDeviceToHost));
        return result;
    }
};

void test_launch_rejections(const std::shared_ptr<Kernel>& kernel) {
    const auto good = launch_options();
    expect_error([&] { kernel->launch(good.override_with({.num_smem_bytes = -1})); }, "must not be negative");
    expect_error([&] { kernel->launch(good.override_with({.num_smem_bytes = std::numeric_limits<int>::max()})); }, "shared-memory");
    expect_error([&] { kernel->launch(good.override_with({.grid_dim = dim3(0)})); }, "positive");
    expect_error([&] { kernel->launch(good.override_with({.block_dim = dim3(65535)})); }, "limit");
    expect_error([&] { kernel->launch(good.override_with({.cluster_dim = dim3(2)})); }, "clusters");
    expect_error([&] { kernel->launch(good.override_with({.cluster_dim = dim3(1, 2, 1)})); }, "clusters");
    expect_error([&] { kernel->launch(good.override_with({.enable_pdl = true})); }, "PDL");
    expect_error([&] { kernel->launch(good.override_with({.nonportable_cluster_size_allowed = true})); }, "clusters");
    auto missing = good;
    missing.block_dim.reset();
    expect_error([&] { kernel->launch(missing); }, "block dimension must be specified");
    expect_error([&] { jit->launch(std::shared_ptr<Kernel>{}, good); }, "must not be null");
    deep_jit::rocm::Device device;
    if (not device.get_prop().cooperativeLaunch) {
        expect_error([&] { kernel->launch(good.override_with({.cooperative = true})); }, "does not support cooperative");
    } else {
        expect_error([&] { kernel->launch(good.override_with({.grid_dim = dim3(1000000), .cooperative = true})); }, "capacity");
    }
}

void test_artifact_loading(const std::filesystem::path& root) {
    const auto dir = root / "invalid_artifact";
    std::filesystem::create_directories(dir);
    const deep_jit::Env env("ROCM_TEST");
    expect_error([&] { (void)Kernel::load(dir, env); }, "missing or empty");
    deep_jit::write_file_sync(dir / "kernel.hsaco", "");
    expect_error([&] { (void)Kernel::load(dir, env); }, "missing or empty");
    deep_jit::write_file_sync(dir / "kernel.hsaco", "not an AMD code object");
    expect_error([&] { (void)Kernel::load(dir, env); }, "HIP error");
#ifndef DJ_ROCM_MOCK_TESTS
    DJ_HOST_ASSERT(hipGetLastError() == hipErrorInvalidImage);
#endif
}

std::string run_device_tests(const std::string& root) {
    auto runtime = jit.get();
    DJ_HOST_ASSERT(runtime->device.get_arch().starts_with("gfx"));
    DJ_HOST_ASSERT(runtime->device.get_warp_size() > 0 and runtime->device.get_num_sms() > 0);
    DJ_HOST_ASSERT(runtime->device.get_clock_rate() > 0 and runtime->device.get_num_smem_bytes() > 0);
    const auto kernel = runtime->compile("noop", noop_source);
    DJ_HOST_ASSERT(kernel == runtime->compile("noop_again", noop_source));
    runtime->launch(kernel, launch_options());
    DJ_HIP_CHECK(hipDeviceSynchronize());
    test_launch_rejections(kernel);
    test_artifact_loading(root);

    const auto two = noop_source + "\nextern \"C\" __global__ void second() {}\n";
    (void)runtime->compile_without_load("two_build_only", two);
    expect_error([&] { (void)runtime->compile("two", two); }, "exactly one kernel");
    const std::string zero = "#include <hip/hip_runtime.h>\nextern \"C\" { __device__ int exported_value = 1; }\n";
    (void)runtime->compile_without_load("zero_build_only", zero);
    expect_error([&] { (void)runtime->compile("zero", zero); }, "exactly one kernel");
    expect_error([&] { (void)runtime->compile_without_load("syntax_error", "#error deliberate_rocm_compiler_failure\n"); },
                 "deliberate_rocm_compiler_failure");

    Buffer<long long> output(1);
    const std::string abi_source = R"(
#include <hip/hip_runtime.h>
struct alignas(16) Pair { long long x; double y; };
extern "C" __global__ void abi(long long* output, char a, short b, int c,
                              unsigned long long large, float f, double d, Pair pair) {
    *output = a + b + c + large + (long long)(2 * f) + (long long)(2 * d) + pair.x + (long long)pair.y;
}
)";
    struct alignas(16) Pair { long long x; double y; };
    struct Wrapped: deep_jit::NoRefPtr {};
    const unsigned long long large = 1ull << 34;
    const Pair pair{11, 13.0};
    auto abi = runtime->compile("abi", abi_source);
    auto* pointer = output.ptr;
    runtime->launch(abi, launch_options(), deep_jit::NoRefPtr{&pointer}, char(3), short(7), 19,
                    Wrapped{{const_cast<unsigned long long*>(&large)}}, 2.5f, 4.5, pair);
    DJ_HOST_ASSERT(output.read(1)[0] == static_cast<long long>(large) + 67);

    const std::string dimensions_source = R"(
#include <hip/hip_runtime.h>
extern "C" __global__ void dimensions(int* output) {
    const int block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    const int thread = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const int index = block * blockDim.x * blockDim.y * blockDim.z + thread;
    output[index] = index;
}
)";
    Buffer<int> dimensions(192);
    runtime->launch(runtime->compile("dimensions", dimensions_source), launch_options(dim3(2, 3, 2), dim3(4, 2, 2)), dimensions.ptr);
    const auto values = dimensions.read(192);
    for (int i = 0; i < 192; ++i)
        DJ_HOST_ASSERT(values[i] == i);

    const std::string shared_source = R"(
#include <hip/hip_runtime.h>
extern "C" __global__ void shared(int* output) {
    extern __shared__ int values[];
    values[threadIdx.x] = threadIdx.x;
    __syncthreads();
    if (threadIdx.x == 0) output[blockIdx.x] = values[blockDim.x - 1];
}
)";
    auto shared_options = launch_options(dim3(2), dim3(64));
    shared_options.num_smem_bytes = 64 * sizeof(int);
    Buffer<int> shared(2);
    runtime->launch(runtime->compile("shared", shared_source), shared_options, shared.ptr);
    DJ_HOST_ASSERT(shared.read(2) == std::vector<int>({63, 63}));

    std::shared_ptr<Kernel> surviving;
    {
        JIT temporary(deep_jit::Config(std::filesystem::absolute(root), "ROCM_TEST"));
        surviving = temporary.compile("lifetime", increment_source);
    }
    Buffer<int> lifetime(1);
    surviving->launch(launch_options(), lifetime.ptr, 41);
    DJ_HOST_ASSERT(lifetime.read(1)[0] == 42);

    if (runtime->device.get_prop().cooperativeLaunch) {
        const std::string cooperative_source = R"(
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
extern "C" __global__ void cooperative(int* output) {
    output[blockIdx.x] = 1;
    cooperative_groups::this_grid().sync();
    if (blockIdx.x == 0) {
        int total = 0;
        for (unsigned int i = 0; i < gridDim.x; ++i) total += output[i];
        output[0] = total;
    }
}
)";
        Buffer<int> cooperative(2);
        auto options = launch_options(dim3(2));
        options.cooperative = true;
        runtime->launch(runtime->compile("cooperative", cooperative_source), options, cooperative.ptr);
        DJ_HOST_ASSERT(cooperative.read(2)[0] == 2);
        return "cooperative launch validated";
    }
    return "cooperative launch unavailable on this device; rejection validated";
}

#ifdef DJ_ROCM_MOCK_TESTS
void run_mock_tests(const std::string& directory) {
    // This checks host control flow only. No mock result is a GPU execution test.
    using GetCount = int (*)();
    using SetInt = void (*)(int);
    using GetValue = uintptr_t (*)();
    using SetValue = void (*)(uintptr_t);
    const auto handle = deep_jit::rocm::driver::get_hip_handle();
    const auto count = reinterpret_cast<GetCount>(dlsym(handle, "mock_library_count"));
    const auto launch_count = reinterpret_cast<GetCount>(dlsym(handle, "mock_launch_count"));
    const auto last_stream = reinterpret_cast<GetValue>(dlsym(handle, "mock_last_stream"));
    const auto last_value = reinterpret_cast<GetValue>(dlsym(handle, "mock_last_value"));
    const auto set_stream = reinterpret_cast<SetValue>(dlsym(handle, "mock_set_stream"));
    const auto set_current = reinterpret_cast<SetInt>(dlsym(handle, "mock_set_device"));
    DJ_HOST_ASSERT(count and launch_count and last_stream and last_value and set_stream and set_current);
    const auto root = std::filesystem::path(directory);
    const auto dir = root / "mock_load";
    std::filesystem::create_directories(dir);
    const deep_jit::Env env("ROCM_TEST");
    test_artifact_loading(root);
    DJ_HOST_ASSERT(count() == 0);
    for (const auto& source: {std::string("MOCK\n"), std::string("MOCK\n") + noop_source + noop_source}) {
        deep_jit::write_file_sync(dir / "kernel.hsaco", source);
        expect_error([&] { (void)Kernel::load(dir, env); }, "exactly one kernel");
        DJ_HOST_ASSERT(count() == 0);
    }
    deep_jit::write_file_sync(dir / "kernel.hsaco", "MOCK\n" + noop_source);
    for (const auto* phase: {"count", "enumerate", "function", "attribute"}) {
        setenv("DJ_ROCM_MOCK_FAILURE", phase, 1);
        expect_error([&] { (void)Kernel::load(dir, env); }, "HIP error");
        unsetenv("DJ_ROCM_MOCK_FAILURE");
        DJ_HOST_ASSERT(count() == 0);
    }
    auto kernel = Kernel::load(dir, env);
    DJ_HOST_ASSERT(count() == 1);
    const auto before = launch_count();
    kernel->launch(launch_options());
    DJ_HOST_ASSERT(launch_count() == before + 1 and last_stream() == 7);
    set_stream(19);
    kernel->launch(launch_options());
    DJ_HOST_ASSERT(last_stream() == 19);
    kernel->launch(launch_options().override_with({.stream = static_cast<hipStream_t>(nullptr)}));
    DJ_HOST_ASSERT(last_stream() == 0);
    int value = 42;
    kernel->launch(launch_options().override_with({.stream = reinterpret_cast<hipStream_t>(uintptr_t(13))}), value);
    DJ_HOST_ASSERT(last_stream() == 13 and last_value() == 42);
    kernel->launch(launch_options(), deep_jit::NoRefPtr{&value});
    DJ_HOST_ASSERT(last_value() == 42);
    test_launch_rejections(kernel);
    set_current(1);
    expect_error([&] { kernel->launch(launch_options()); }, "device on which it was loaded");
    set_current(0);
    kernel->launch(launch_options().override_with({.cooperative = true}));
    auto second_owner = kernel;
    kernel.reset();
    DJ_HOST_ASSERT(count() == 1);
    second_owner.reset();
    DJ_HOST_ASSERT(count() == 0);
    set_stream(7);
}
#endif

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    deep_jit::register_python_api(module, jit);
    pybind11::class_<Kernel, std::shared_ptr<Kernel>>(module, "Kernel");
    module.def("init_jit", &init_jit);
    module.def("get_arch", [] { return jit->device.get_arch(); });
    module.def("increment_source", [] { return increment_source; });
    module.def("test_options", [](const std::string& root) { test_options(root); });
    module.def("test_metadata", &test_metadata);
    module.def("build_fixture", &build_fixture);
    module.def("compile_only", &compile_only, pybind11::arg("name"), pybind11::arg("source"),
               pybind11::arg("checks") = false, pybind11::arg("dumps") = false, pybind11::arg("hook") = "");
    module.def("compile_kernel", [](const std::string& source) { return jit->compile("python_kernel", source); });
    module.def("load_artifact", [](const std::string& directory) { return Kernel::load(directory, deep_jit::Env("ROCM_TEST")); });
    module.def("launch_increment", [](const std::shared_ptr<Kernel>& kernel, const uintptr_t pointer,
                                       const int value, const std::optional<uintptr_t> stream) {
        auto options = launch_options();
        if (stream)
            options.stream = reinterpret_cast<hipStream_t>(*stream);
        jit->launch(kernel, options, reinterpret_cast<int*>(pointer), value);
    }, pybind11::arg("kernel"), pybind11::arg("pointer"), pybind11::arg("value"), pybind11::arg("stream") = pybind11::none());
    module.def("find_toolkit", [] { return deep_jit::ROCm::find_rocm_toolkit(deep_jit::Env("ROCM_BUILD_TEST")).hipcc.string(); });
    module.def("run_device_tests", &run_device_tests);
#ifdef DJ_ROCM_MOCK_TESTS
    module.def("run_mock_tests", &run_mock_tests);
#endif
}
