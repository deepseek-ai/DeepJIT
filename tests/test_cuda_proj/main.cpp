#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>
#include <unistd.h>

#include <deep_jit/backend/cuda/backend.hpp>
#include <deep_jit/cache/memory.hpp>
#include <deep_jit/python_api.hpp>
#include <deep_jit/utils/command.hpp>
#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/filesystem.hpp>
#include <deep_jit/utils/hash.hpp>
#include <deep_jit/utils/json.hpp>
#include <deep_jit/utils/no_ref_ptr.hpp>
#include <deep_jit/utils/parser.hpp>
#include <deep_jit/utils/str.hpp>
#include <deep_jit/utils/uuid.hpp>

namespace {

using Runtime = deep_jit::Runtime<deep_jit::CUDA>;
using CompilerOptions = deep_jit::cuda::CompilerOptions;
using LaunchOptions = deep_jit::cuda::LaunchOptions;
namespace fs = std::filesystem;

deep_jit::LazyInit<Runtime> python_api_jit(nullptr);
int python_api_num_initializations = 0;
std::shared_ptr<Runtime> process_test_runtime;
std::shared_ptr<Runtime> gil_test_runtime;

const fs::path& get_test_cuda_project_dir() {
    static const fs::path path = [] {
        const auto value = std::getenv("DEEP_JIT_CUDA_TEST_SOURCE_DIR");
        DJ_HOST_ASSERT(value != nullptr, "DEEP_JIT_CUDA_TEST_SOURCE_DIR must be set");
        return fs::absolute(value).lexically_normal();
    }();
    return path;
}

template <typename Function>
void run_test(const std::string_view name, Function&& function) {
    std::printf("[ RUN      ] %.*s\n", static_cast<int>(name.size()), name.data());
    std::fflush(stdout);
    if constexpr (std::is_same_v<std::invoke_result_t<Function>, bool>) {
        if (not function()) {
            std::printf("[  SKIPPED ] %.*s\n", static_cast<int>(name.size()), name.data());
            std::fflush(stdout);
            return;
        }
    } else {
        function();
    }
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

template <typename Function>
void expect_any_failure(Function&& function) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    DJ_PANIC("expected failure");
}

void set_env(const std::string& name, const std::string& value) {
    DJ_HOST_ASSERT(::setenv(name.c_str(), value.c_str(), 1) == 0, "failed to set environment variable: {}", name);
}

void unset_env(const std::string& name) {
    DJ_HOST_ASSERT(::unsetenv(name.c_str()) == 0, "failed to unset environment variable: {}", name);
}

std::optional<std::string> get_raw_env(const std::string& name) {
    const auto value = std::getenv(name.c_str());
    return value == nullptr ? std::nullopt : std::optional<std::string>(value);
}

void restore_env(const std::string& name, const std::optional<std::string>& value) {
    if (value)
        set_env(name, *value);
    else
        unset_env(name);
}

void write_executable(const fs::path& path, const std::string& content) {
    deep_jit::write_file_sync(path, content);
    fs::permissions(
        path,
        fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec,
        fs::perm_options::add);
}

std::string get_source(const std::string& name) {
    return deep_jit::read(get_test_cuda_project_dir() / "kernels" / name);
}

std::string get_template_source(const int bias) {
    auto source = get_source("template_argument_hash.cu");
    const auto position = source.find("TEST_BIAS");
    DJ_HOST_ASSERT(position != std::string::npos, "template placeholder was not found");
    source.replace(position, std::string_view("TEST_BIAS").size(), std::to_string(bias));
    return source;
}

std::string get_float_template_source(const float value) {
    auto source = get_source("generated_float_literal.cu");
    const auto position = source.find("TEST_FLOAT_LITERAL");
    DJ_HOST_ASSERT(position != std::string::npos, "float template placeholder was not found");
    const auto literal = std::format("{}0x{:a}f", std::signbit(value) ? "-" : "", std::abs(value));
    source.replace(position, std::string_view("TEST_FLOAT_LITERAL").size(), literal);
    return source;
}

std::size_t count_different_bytes(const std::string_view first, const std::string_view second) {
    DJ_HOST_ASSERT(first.size() == second.size(), "byte strings must have equal lengths");
    std::size_t count = 0;
    for (std::size_t index = 0; index < first.size(); ++index)
        count += first[index] != second[index];
    return count;
}

std::shared_ptr<Runtime> make_runtime(const fs::path& include_dir,
                                      const std::string& extra_signature = "test-signature") {
    return std::make_shared<Runtime>(deep_jit::Config(
        get_test_cuda_project_dir(),
        "TEST",
        extra_signature,
        {include_dir},
        {"test_cuda/"}));
}

std::shared_ptr<Runtime> make_runtime_with_prefix(const std::string& env_prefix,
                                                  const fs::path& include_dir,
                                                  const std::string& extra_signature = "test-signature") {
    return std::make_shared<Runtime>(deep_jit::Config(
        get_test_cuda_project_dir(),
        env_prefix,
        extra_signature,
        {include_dir},
        {"test_cuda/"}));
}

void check_artifact(const fs::path& artifact_dir, const std::string& source) {
    DJ_HOST_ASSERT(fs::is_regular_file(artifact_dir / ".committed"), "missing commit marker: {}", artifact_dir.string());
    DJ_HOST_ASSERT(fs::is_regular_file(artifact_dir / "kernel.cu"), "missing CUDA source: {}", artifact_dir.string());
    DJ_HOST_ASSERT(fs::is_regular_file(artifact_dir / "kernel.cubin"), "missing CUBIN: {}", artifact_dir.string());
    DJ_HOST_ASSERT(fs::file_size(artifact_dir / "kernel.cubin") > 0, "empty CUBIN: {}", artifact_dir.string());
    DJ_HOST_ASSERT(fs::is_regular_file(artifact_dir / "meta.json"), "missing metadata: {}", artifact_dir.string());
    DJ_HOST_ASSERT(deep_jit::read(artifact_dir / "kernel.cu") == source, "cached source mismatch: {}", artifact_dir.string());
}

void check_metadata(const fs::path& artifact_dir,
                    const std::string_view expected_signature,
                    const std::string_view expected_text) {
    const auto metadata = deep_jit::read(artifact_dir / "meta.json");
    DJ_HOST_ASSERT(metadata.find("\"command\":") != std::string::npos, "metadata command is missing");
    DJ_HOST_ASSERT(metadata.find("\"config\":") != std::string::npos, "metadata config is missing");
    DJ_HOST_ASSERT(metadata.find("\"compiler_info\":") != std::string::npos, "metadata compiler info is missing");
    DJ_HOST_ASSERT(metadata.find("\"compiler_options\":") != std::string::npos, "metadata compiler options are missing");
    DJ_HOST_ASSERT(metadata.find(std::string(expected_signature)) != std::string::npos,
                   "metadata signature is missing: {}", expected_signature);
    DJ_HOST_ASSERT(metadata.find(std::string(expected_text)) != std::string::npos,
                   "metadata value is missing: {}", expected_text);
}

int launch_value(Runtime& runtime,
                 const std::shared_ptr<deep_jit::cuda::Kernel>& kernel,
                 const int input = 0) {
    int* output = nullptr;
    DJ_CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&output), sizeof(int)));
    try {
        runtime.launch(
            kernel, {
                .grid_dim = dim3(1, 1, 1),
                .block_dim = dim3(1, 1, 1),
            },
            output, input);
        DJ_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

        int result = 0;
        DJ_CUDA_RUNTIME_CHECK(cudaMemcpy(&result, output, sizeof(int), cudaMemcpyDeviceToHost));
        DJ_CUDA_RUNTIME_CHECK(cudaFree(output));
        return result;
    } catch (...) {
        cudaFree(output);
        throw;
    }
}

void check_tmp_is_empty(const fs::path& cache_root) {
    const auto tmp_dir = cache_root / "tmp";
    DJ_HOST_ASSERT(not fs::exists(tmp_dir) or fs::directory_iterator(tmp_dir) == fs::directory_iterator(),
                   "temporary artifacts were not cleaned: {}", tmp_dir.string());
}

void test_environment(const fs::path& cache_root) {
    set_env("DJ_TEST_VALUE", "global");
    set_env("TEST_ENV_TEST_VALUE", "library");
    set_env("TEST_VALUE", "unprefixed");
    set_env("TEST_ENV_BOOL", "YeS");

    const deep_jit::Env env("TEST_ENV");
    DJ_HOST_ASSERT(env.get<std::string>("TEST_VALUE") == "library");
    DJ_HOST_ASSERT(env.get<bool>("BOOL") == true);
    unset_env("TEST_ENV_TEST_VALUE");
    DJ_HOST_ASSERT(env.get<std::string>("TEST_VALUE") == "global");
    unset_env("DJ_TEST_VALUE");
    DJ_HOST_ASSERT(not env.get<std::string>("TEST_VALUE").has_value());
    unset_env("TEST_ENV_BOOL");
    unset_env("TEST_VALUE");

    set_env("TEST_ENV_BOOL_TRUE", "TrUe");
    set_env("TEST_ENV_BOOL_FALSE", "nO");
    set_env("TEST_ENV_BOOL_INTEGER", "-2");
    set_env("TEST_ENV_INTEGER", "-17");
    DJ_HOST_ASSERT(env.get<bool>("BOOL_TRUE") == true);
    DJ_HOST_ASSERT(env.get<bool>("BOOL_FALSE") == false);
    DJ_HOST_ASSERT(env.get<bool>("BOOL_INTEGER") == true);
    DJ_HOST_ASSERT(env.get<int>("INTEGER") == -17);
    unset_env("TEST_ENV_BOOL_TRUE");
    unset_env("TEST_ENV_BOOL_FALSE");
    unset_env("TEST_ENV_BOOL_INTEGER");
    unset_env("TEST_ENV_INTEGER");

    set_env("TEST_ENV_INVALID_BOOL", "sometimes");
    expect_failure([&] { env.get<bool>("INVALID_BOOL"); }, "invalid value");
    unset_env("TEST_ENV_INVALID_BOOL");
    set_env("TEST_ENV_INVALID_INTEGER", "12x");
    expect_failure([&] { env.get<int>("INVALID_INTEGER"); }, "invalid value");
    unset_env("TEST_ENV_INVALID_INTEGER");
    set_env("TEST_ENV_EMPTY_BOOL", "");
    expect_failure([&] { env.get<bool>("EMPTY_BOOL"); }, "invalid value");
    unset_env("TEST_ENV_EMPTY_BOOL");
    set_env("TEST_ENV_UNSIGNED", "-1");
    expect_failure([&] { env.get<unsigned>("UNSIGNED"); }, "invalid value");
    unset_env("TEST_ENV_UNSIGNED");
    set_env("TEST_ENV_OVERFLOW", "999999999999999999999999999999");
    expect_failure([&] { env.get<int64_t>("OVERFLOW"); }, "invalid value");
    unset_env("TEST_ENV_OVERFLOW");

    expect_failure([] { deep_jit::Env(""); }, "prefix must not be empty");
    expect_failure([] { deep_jit::Env("DJ"); }, "reserved for global environment variables");

    const auto first_cache = cache_root / "first";
    const auto second_cache = cache_root / "second";
    set_env("TEST_ENV_JIT_CACHE_DIR", first_cache.string() + ":" + second_cache.string());
    const auto disk_cache = deep_jit::DiskCache::from_env(env);
    DJ_HOST_ASSERT(disk_cache.paths.size() == 2);
    DJ_HOST_ASSERT(disk_cache.paths[0] == first_cache);
    DJ_HOST_ASSERT(disk_cache.paths[1] == second_cache);
    unset_env("TEST_ENV_JIT_CACHE_DIR");

    for (const auto& invalid_cache_paths : std::vector<std::string>{
             "",
             ":" + first_cache.string(),
             first_cache.string() + ":",
             first_cache.string() + "::" + second_cache.string(),
         }) {
        set_env("TEST_ENV_JIT_CACHE_DIR", invalid_cache_paths);
        expect_failure([&] { deep_jit::DiskCache::from_env(env); }, "contains an empty path");
    }
    unset_env("TEST_ENV_JIT_CACHE_DIR");

    const auto saved_home = get_raw_env("HOME");
    const auto saved_global_cache = get_raw_env("DJ_JIT_CACHE_DIR");
    DJ_HOST_ASSERT(saved_home.has_value() and saved_global_cache.has_value());
    DJ_HOST_ASSERT(deep_jit::DiskCache::from_env(env).paths ==
                       std::vector<fs::path>{fs::path(*saved_global_cache)},
                   "DJ cache directory was not used as the global fallback");
    set_env("JIT_CACHE_DIR", (cache_root / "ignored_unprefixed").string());
    DJ_HOST_ASSERT(deep_jit::DiskCache::from_env(env).paths ==
                       std::vector<fs::path>{fs::path(*saved_global_cache)},
                   "unprefixed cache directory unexpectedly took effect");
    unset_env("JIT_CACHE_DIR");
    unset_env("DJ_JIT_CACHE_DIR");
    DJ_HOST_ASSERT(deep_jit::DiskCache::from_env(env).paths ==
                   std::vector<fs::path>{fs::path(*saved_home) / ".dj"});
    unset_env("HOME");
    expect_failure([&] { deep_jit::DiskCache::from_env(env); }, "HOME environment variable must not be empty");
    restore_env("HOME", saved_home);
    restore_env("DJ_JIT_CACHE_DIR", saved_global_cache);
}

void test_config() {
    const auto root = get_test_cuda_project_dir();
    const auto include_dir = root / "include_original" / ".";
    const deep_jit::Config config(
        root / "scripts" / "..",
        "CONFIG_TEST",
        "dependency-version",
        {include_dir},
        {"test_cuda/"});
    DJ_HOST_ASSERT(fs::equivalent(config.python_library_root, root));
    DJ_HOST_ASSERT(config.include_dirs.size() == 1 and fs::equivalent(config.include_dirs.front(), root / "include_original"));
    DJ_HOST_ASSERT(fs::equivalent(config.get_python_path("scripts/../scripts/post_hook.py"), root / "scripts/post_hook.py"));
    const auto serialized = config.to_json().dump();
    DJ_HOST_ASSERT(serialized.find("\"extra_signature\":\"dependency-version\"") != std::string::npos);
    DJ_HOST_ASSERT(serialized.find("\"include_prefixes\":[\"test_cuda/\"]") != std::string::npos);

    expect_failure([] { deep_jit::Config({}, "CONFIG_TEST"); }, "Python library root must not be empty");
    expect_failure([] { deep_jit::Config("relative", "CONFIG_TEST"); }, "Python library root must be absolute");
    expect_failure([&] { deep_jit::Config(root, ""); }, "prefix must not be empty");
    expect_failure([&] { deep_jit::Config(root, "DJ"); }, "reserved for global environment variables");
    expect_failure([&] { deep_jit::Config(root, "CONFIG_TEST", {}, {fs::path{}}); }, "include directory must not be empty");
    expect_failure([&] { deep_jit::Config(root, "CONFIG_TEST", {}, {"relative"}); }, "include directory must be absolute");
}

void test_filesystem(const fs::path& cache_root) {
    deep_jit::make_dirs(cache_root);
    const auto path = cache_root / "binary_data";
    const std::string expected("a\0b", 3);
    deep_jit::write_file_sync(path, expected);
    DJ_HOST_ASSERT(deep_jit::read(path) == expected, "binary file contents changed while reading");
    DJ_HOST_ASSERT(fs::remove(path), "failed to remove binary test file: {}", path.string());
    expect_failure([&] { deep_jit::read(path); }, "failed to open for reading");

    std::string all_bytes;
    all_bytes.reserve(256 * 257);
    for (int repeat = 0; repeat < 257; ++repeat) {
        for (int value = 0; value < 256; ++value)
            all_bytes.push_back(static_cast<char>(value));
    }
    const auto all_bytes_path = cache_root / "all_byte_values";
    deep_jit::write_file_sync(all_bytes_path, all_bytes);
    DJ_HOST_ASSERT(deep_jit::read(all_bytes_path) == all_bytes,
                   "binary file read stopped or changed data at an embedded null byte");
    DJ_HOST_ASSERT(fs::remove(all_bytes_path));

    const auto empty_path = cache_root / "empty_file";
    deep_jit::write_file_sync(empty_path, "");
    DJ_HOST_ASSERT(deep_jit::read(empty_path).empty(), "empty file was not read correctly");
    DJ_HOST_ASSERT(fs::remove(empty_path));

    DJ_HOST_ASSERT(not deep_jit::normalize_path(std::nullopt).has_value());
    DJ_HOST_ASSERT(not deep_jit::normalize_path(fs::path{}).has_value());
    DJ_HOST_ASSERT(deep_jit::normalize_path(fs::path("relative/../normalized")) ==
                   fs::absolute("normalized").lexically_normal());
    DJ_HOST_ASSERT(deep_jit::is_executable("/bin/sh"));

    const auto remove_root = cache_root / "safe_remove_all";
    deep_jit::make_dirs(remove_root / "nested");
    deep_jit::write_file_sync(remove_root / "nested/file", "payload");
    deep_jit::safe_remove_all(remove_root);
    DJ_HOST_ASSERT(not fs::exists(remove_root), "safe_remove_all did not remove the directory tree");

    const auto single_file = cache_root / "single_file";
    deep_jit::write_file_sync(single_file, "payload");
    DJ_HOST_ASSERT(not deep_jit::is_executable(single_file), "regular data file was reported as executable");
    deep_jit::safe_remove_all(single_file);
    DJ_HOST_ASSERT(not fs::exists(single_file), "safe_remove_all did not remove a single file");
    DJ_HOST_ASSERT(not deep_jit::try_update_mtime(single_file), "mtime update unexpectedly succeeded for a missing file");
    deep_jit::safe_remove_all(single_file);
}

void test_command_and_uuid() {
    DJ_HOST_ASSERT(deep_jit::call_external_command("sh -c 'printf stdout; printf stderr >&2'") == "stdoutstderr",
                   "external command did not capture stdout and stderr");
    DJ_HOST_ASSERT(deep_jit::call_external_command("printf '%01024d' 0").size() == 1024,
                   "external command output was truncated");
    expect_failure([] { deep_jit::call_external_command(""); }, "command must not be empty");
    expect_failure([] { deep_jit::call_external_command("sh -c 'printf failure; exit 7'"); },
                   "command failed with exit code 7");
    expect_failure([] { deep_jit::call_external_command("kill -TERM $$"); }, "exit code 143");

    const auto prefix = std::to_string(::getpid()) + "-";
    for (int i = 0; i < 16; ++i) {
        const auto uuid = deep_jit::get_uuid();
        DJ_HOST_ASSERT(uuid.starts_with(prefix), "UUID does not contain the process id: {}", uuid);
        DJ_HOST_ASSERT(uuid.size() == prefix.size() + 26, "unexpected UUID length: {}", uuid);
        const auto suffix = std::string_view(uuid).substr(prefix.size());
        DJ_HOST_ASSERT(suffix[8] == '-' and suffix[17] == '-', "UUID separators are misplaced: {}", uuid);
        for (std::size_t index = 0; index < suffix.size(); ++index) {
            if (index == 8 or index == 17)
                continue;
            const auto value = suffix[index];
            DJ_HOST_ASSERT((value >= '0' and value <= '9') or (value >= 'a' and value <= 'f'),
                           "UUID contains an invalid character: {}", uuid);
        }
    }
}

void test_disk_cache(const fs::path& cache_root) {
    const auto root = cache_root / "disk_cache";
    const auto primary = root / "primary";
    const auto secondary = root / "secondary";
    deep_jit::DiskCache cache({primary, secondary});
    expect_failure([] { deep_jit::DiskCache(std::vector<fs::path>{}); }, "not this->paths.empty()");

    fs::path abandoned_path;
    {
        auto entry = cache.entry("abandoned", "digest");
        DJ_HOST_ASSERT(not entry.hit);
        abandoned_path = entry.path;
        deep_jit::write_file_sync(entry.path / "partial", "partial");
    }
    DJ_HOST_ASSERT(not fs::exists(abandoned_path), "uncommitted cache entry was not cleaned");

    fs::path committed_path;
    {
        auto entry = cache.entry("committed", "digest");
        DJ_HOST_ASSERT(not entry.hit);
        deep_jit::write_file_sync(entry.path / "payload", "complete");
        committed_path = entry.commit();
        DJ_HOST_ASSERT(entry.commit() == committed_path, "cache commit must be idempotent");
    }
    DJ_HOST_ASSERT(fs::is_regular_file(committed_path / deep_jit::kCommitFileName));
    DJ_HOST_ASSERT(deep_jit::read(committed_path / "payload") == "complete");
    {
        auto entry = cache.entry("committed", "digest");
        DJ_HOST_ASSERT(entry.hit and entry.path == committed_path, "committed cache entry was not reused");
        DJ_HOST_ASSERT(entry.commit() == committed_path, "committing a cache hit changed its path");
    }

    const auto secondary_path = secondary / "cache/secondary.digest";
    deep_jit::make_dirs(secondary_path);
    deep_jit::write_file_sync(secondary_path / deep_jit::kCommitFileName, "");
    deep_jit::write_file_sync(secondary_path / "payload", "secondary");
    {
        auto entry = cache.entry("secondary", "digest");
        DJ_HOST_ASSERT(entry.hit and entry.path == secondary_path, "secondary cache root was not searched");
    }

    const auto primary_priority_path = primary / "cache/priority.digest";
    const auto secondary_priority_path = secondary / "cache/priority.digest";
    for (const auto& path : {primary_priority_path, secondary_priority_path}) {
        deep_jit::make_dirs(path);
        deep_jit::write_file_sync(path / deep_jit::kCommitFileName, "");
    }
    deep_jit::write_file_sync(primary_priority_path / "payload", "primary");
    deep_jit::write_file_sync(secondary_priority_path / "payload", "secondary");
    {
        auto entry = cache.entry("priority", "digest");
        DJ_HOST_ASSERT(entry.hit and entry.path == primary_priority_path,
                       "the first valid cache root must take priority");
    }

    const auto incomplete_primary_path = primary / "cache/incomplete_primary.digest";
    const auto valid_secondary_path = secondary / "cache/incomplete_primary.digest";
    deep_jit::make_dirs(incomplete_primary_path);
    deep_jit::write_file_sync(incomplete_primary_path / "partial", "partial");
    deep_jit::make_dirs(valid_secondary_path);
    deep_jit::write_file_sync(valid_secondary_path / deep_jit::kCommitFileName, "");
    deep_jit::write_file_sync(valid_secondary_path / "payload", "secondary");
    {
        auto entry = cache.entry("incomplete_primary", "digest");
        DJ_HOST_ASSERT(entry.hit and entry.path == valid_secondary_path,
                       "an uncommitted primary entry hid a valid secondary entry");
    }

    const auto incomplete_secondary_path = secondary / "cache/incomplete_secondary.digest";
    deep_jit::make_dirs(incomplete_secondary_path);
    deep_jit::write_file_sync(incomplete_secondary_path / "partial", "partial");
    fs::path temporary_path;
    {
        auto entry = cache.entry("incomplete_secondary", "digest");
        DJ_HOST_ASSERT(not entry.hit and entry.path.parent_path() == primary / "tmp",
                       "an uncommitted secondary entry was treated as a cache hit");
        temporary_path = entry.path;
    }
    DJ_HOST_ASSERT(not fs::exists(temporary_path), "temporary cache miss was not cleaned");
    DJ_HOST_ASSERT(fs::is_regular_file(incomplete_secondary_path / "partial"),
                   "cache lookup modified an uncommitted secondary entry");

    const auto stale_temporary_path = primary / "tmp/stale";
    deep_jit::make_dirs(stale_temporary_path / "nested");
    deep_jit::write_file_sync(stale_temporary_path / "nested/payload", "stale");
    {
        auto entry = cache.entry("ignores_stale_tmp", "digest");
        DJ_HOST_ASSERT(not entry.hit and entry.path != stale_temporary_path,
                       "a stale temporary directory was reused");
    }
    DJ_HOST_ASSERT(fs::is_regular_file(stale_temporary_path / "nested/payload"),
                   "cache entry cleanup removed another writer's temporary directory");
    deep_jit::safe_remove_all(stale_temporary_path);

    expect_failure([&] { (void)cache.entry("", "digest"); }, "cache tag must contain only");
    expect_failure([&] { (void)cache.entry("path/tag", "digest"); }, "cache tag must contain only");
    expect_failure([&] { (void)cache.entry("dot.tag", "digest"); }, "cache tag must contain only");
}

void test_memory_cache() {
    struct Value {
        int number;
    };

    deep_jit::MemCache<std::string, Value> cache;
    int num_factory_calls = 0;
    const auto first = cache.get_or_create("same", [&] {
        ++num_factory_calls;
        return std::make_shared<Value>(Value{7});
    });
    const auto second = cache.get_or_create("same", [&] {
        ++num_factory_calls;
        return std::make_shared<Value>(Value{8});
    });
    DJ_HOST_ASSERT(first == second and first->number == 7);
    DJ_HOST_ASSERT(num_factory_calls == 1, "memory cache called the factory on a hit");

    expect_failure(
        [&] {
            cache.get_or_create("failure", []() -> std::shared_ptr<Value> {
                throw std::runtime_error("factory failed");
            });
        },
        "factory failed");
    DJ_HOST_ASSERT(not cache.cache.contains("failure"), "failed factory populated the memory cache");
}

void test_json() {
    static_assert(std::is_constructible_v<deep_jit::json, int>);
    static_assert(std::is_constructible_v<deep_jit::json, uint64_t>);
    static_assert(not std::is_constructible_v<deep_jit::json, double>);
    static_assert(not std::is_constructible_v<deep_jit::json, void*>);

    const deep_jit::json value = deep_jit::json::object_t {
        {"integer", 10},
        {"negative", -7},
        {"minimum", std::numeric_limits<int64_t>::min()},
        {"unsigned", std::numeric_limits<uint64_t>::max()},
        {"boolean", false},
        {"string", "line\n\"quoted\""},
        {"optional", std::optional<int>()},
        {"array", std::vector<int>{1, 2, 3}},
    };
    DJ_HOST_ASSERT(value.dump() == R"({"integer":10,"negative":-7,"minimum":-9223372036854775808,"unsigned":18446744073709551615,"boolean":false,"string":"line\n\"quoted\"","optional":null,"array":[1,2,3]})");
    DJ_HOST_ASSERT(deep_jit::json(std::string("\b\f\n\r\t\\\x01", 7)).dump() == "\"\\b\\f\\n\\r\\t\\\\\\u0001\"");
    DJ_HOST_ASSERT(deep_jit::json(std::string("a\0b", 3)).dump() == "\"a\\u0000b\"");
    DJ_HOST_ASSERT(deep_jit::json(deep_jit::json::object_t{{"key\n", 1}}).dump() == "{\"key\\n\":1}");
    const auto empty_options = CompilerOptions().to_json().dump();
    DJ_HOST_ASSERT(empty_options.find("\"optimize_level\":null") != std::string::npos);
    DJ_HOST_ASSERT(empty_options.find("\"extra_nvcc_flags\":[]") != std::string::npos);
}

void test_hash_boundaries() {
    const std::array<std::string, 6> inputs = {"", "a", "ab", "abc", "abcd", "message digest"};
    std::unordered_set<std::string> input_digests;
    for (const auto& input : inputs) {
        const auto digest = deep_jit::hash::get_hex_digest(input);
        DJ_HOST_ASSERT(digest == deep_jit::hash::get_hex_digest(input), "hash output is not deterministic");
        DJ_HOST_ASSERT(digest.size() == 32 and std::ranges::all_of(digest, [](const char value) {
                           return (value >= '0' and value <= '9') or (value >= 'a' and value <= 'f');
                       }),
                       "hash digest is not 128-bit lowercase hexadecimal: {}", digest);
        DJ_HOST_ASSERT(input_digests.emplace(digest).second, "distinct basic inputs collided: {}", input);
    }

    const auto first = deep_jit::hash::FNV1a().update("ab").update("c").get_hex_digest();
    const auto second = deep_jit::hash::FNV1a().update("a").update("bc").get_hex_digest();
    DJ_HOST_ASSERT(first != second, "hash updates must be separated");
    const std::string left_with_zero("a\0", 2);
    const std::string right_with_zero("\0b", 2);
    DJ_HOST_ASSERT(deep_jit::hash::FNV1a().update(left_with_zero).update("b").get_hex_digest() !=
                       deep_jit::hash::FNV1a().update("a").update(right_with_zero).get_hex_digest(),
                   "hash framing must remain unambiguous with embedded null bytes");
    DJ_HOST_ASSERT(deep_jit::hash::FNV1a().update("").update("a").get_hex_digest() !=
                       deep_jit::hash::FNV1a().update("a").update("").get_hex_digest(),
                   "empty hash components must preserve their position");
    DJ_HOST_ASSERT(deep_jit::hash::get_hex_digest(std::string("a\0b", 3)) != deep_jit::hash::get_hex_digest("a"),
                   "embedded null bytes must participate in hashing");
    const std::array binary_inputs = {
        std::string("\0", 1),
        std::string("\0tail", 5),
        std::string("head\0", 5),
        std::string("head\0tail", 9),
        std::string("a\0b\0c", 5),
    };
    for (const auto& input : binary_inputs) {
        const auto digest = deep_jit::hash::get_hex_digest(input);
        DJ_HOST_ASSERT(digest ==
                           deep_jit::hash::get_hex_digest(std::string_view(input.data(), input.size())),
                       "string and explicit string_view hashing diverged");
        const auto null_position = input.find('\0');
        DJ_HOST_ASSERT(digest != deep_jit::hash::get_hex_digest(
                                     std::string_view(input.data(), null_position)),
                       "binary hash input was truncated at an embedded null");
    }
    const std::string after_null_a("prefix\0A", 8);
    const std::string after_null_b("prefix\0B", 8);
    DJ_HOST_ASSERT(deep_jit::hash::get_hex_digest(after_null_a) != deep_jit::hash::get_hex_digest(after_null_b),
                   "bytes after an embedded null must affect the hash");

    const auto segmented_digest = deep_jit::hash::FNV1a()
        .update(std::string("a\0", 2)).update(std::string("\0b", 2))
        .get_hex_digest();
    DJ_HOST_ASSERT(segmented_digest != deep_jit::hash::get_hex_digest(std::string("a\0\0b", 4)),
                   "chained hash boundaries were lost");

    auto base = deep_jit::hash::FNV1a().update("base");
    auto copied = base;
    DJ_HOST_ASSERT(base.update("left").get_hex_digest() != copied.update("right").get_hex_digest(),
                   "copied hash states must evolve independently");

    DJ_HOST_ASSERT(deep_jit::str::join({}).empty());
    DJ_HOST_ASSERT(deep_jit::str::join({"one"}) == "one");
    DJ_HOST_ASSERT(deep_jit::str::join({"one", "two", "three"}, ":") == "one:two:three");
}

void test_hash_robustness() {
    std::string all_bytes;
    all_bytes.reserve(256);
    for (int value = 0; value < 256; ++value)
        all_bytes.push_back(static_cast<char>(value));
    auto changed_all_bytes = all_bytes;
    changed_all_bytes.back() ^= 1;
    DJ_HOST_ASSERT(deep_jit::hash::get_hex_digest(all_bytes) != deep_jit::hash::get_hex_digest(changed_all_bytes),
                   "changing one byte in a binary input did not change the hash");

    std::unordered_set<std::string> one_byte_digests;
    for (int value = 0; value < 256; ++value) {
        const char byte = static_cast<char>(value);
        one_byte_digests.emplace(deep_jit::hash::get_hex_digest(std::string_view(&byte, 1)));
    }
    DJ_HOST_ASSERT(one_byte_digests.size() == 256, "single-byte inputs collided");

    {
        constexpr uint64_t num_collision_inputs = 1'000'000;
        std::mt19937_64 collision_generator(0x741bc92du);
        std::unordered_set<std::string> collision_digests;
        collision_digests.max_load_factor(0.8f);
        collision_digests.reserve(num_collision_inputs);
        const auto begin = std::chrono::steady_clock::now();
        for (uint64_t index = 0; index < num_collision_inputs; ++index) {
            const std::array<uint64_t, 4> input = {
                index, collision_generator(), collision_generator(), collision_generator()};
            const auto bytes = std::string_view(
                reinterpret_cast<const char*>(input.data()), sizeof(input));
            DJ_HOST_ASSERT(collision_digests.emplace(deep_jit::hash::get_hex_digest(bytes)).second,
                           "hash collision at randomized input {}", index);
        }
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        std::printf("Hash collision test: %llu inputs in %.2f seconds\n",
                    static_cast<unsigned long long>(num_collision_inputs), elapsed);
    }

    constexpr int num_inputs = 10000;
    std::mt19937_64 generator(0x2d4a7f19u);
    uint64_t total_bit_distance = 0;
    uint64_t total_lane_distance = 0;
    std::array<int, 128> output_one_counts{};
    std::array<int, 128> changed_bit_counts{};
    std::array<int, 64> lane_difference_counts{};
    std::unordered_set<uint64_t> high_digests, low_digests;
    high_digests.reserve(num_inputs);
    low_digests.reserve(num_inputs);
    for (uint64_t index = 0; index < num_inputs; ++index) {
        std::string data(reinterpret_cast<const char*>(&index), sizeof(index));
        const auto random_size = static_cast<std::size_t>(generator() % 128);
        data.reserve(data.size() + random_size);
        for (std::size_t i = 0; i < random_size; ++i)
            data.push_back(static_cast<char>(generator()));

        const auto digest = deep_jit::hash::get_hex_digest(data);
        data[index % data.size()] ^= static_cast<char>(1u << (index % 8));
        const auto changed_digest = deep_jit::hash::get_hex_digest(data);
        const auto high = std::stoull(digest.substr(0, 16), nullptr, 16);
        const auto low = std::stoull(digest.substr(16), nullptr, 16);
        const auto changed_high = std::stoull(changed_digest.substr(0, 16), nullptr, 16);
        const auto changed_low = std::stoull(changed_digest.substr(16), nullptr, 16);
        const auto high_difference = high ^ changed_high;
        const auto low_difference = low ^ changed_low;
        total_bit_distance += std::popcount(high_difference) + std::popcount(low_difference);
        total_lane_distance += std::popcount(high ^ low);
        DJ_HOST_ASSERT(high_digests.emplace(high).second, "high hash lane collided at input {}", index);
        DJ_HOST_ASSERT(low_digests.emplace(low).second, "low hash lane collided at input {}", index);
        for (int bit = 0; bit < 64; ++bit) {
            output_one_counts[bit] += static_cast<int>((high >> bit) & 1);
            output_one_counts[64 + bit] += static_cast<int>((low >> bit) & 1);
            changed_bit_counts[bit] += static_cast<int>((high_difference >> bit) & 1);
            changed_bit_counts[64 + bit] += static_cast<int>((low_difference >> bit) & 1);
            lane_difference_counts[bit] += static_cast<int>(((high ^ low) >> bit) & 1);
        }
    }

    const auto average_bit_distance = static_cast<double>(total_bit_distance) / num_inputs;
    const auto average_lane_distance = static_cast<double>(total_lane_distance) / num_inputs;
    DJ_HOST_ASSERT(average_bit_distance > 60.0 and average_bit_distance < 68.0,
                   "weak hash diffusion: average bit distance is {}", average_bit_distance);
    DJ_HOST_ASSERT(average_lane_distance > 28.0 and average_lane_distance < 36.0,
                   "hash lanes are correlated: average bit distance is {}", average_lane_distance);
    const auto check_balanced_bits = [](const auto& counts, const std::string_view name) {
        for (std::size_t bit = 0; bit < counts.size(); ++bit) {
            DJ_HOST_ASSERT(counts[bit] > 3500 and counts[bit] < 6500,
                           "{} bit {} is biased: {} occurrences", name, bit, counts[bit]);
        }
    };
    check_balanced_bits(output_one_counts, "hash output");
    check_balanced_bits(changed_bit_counts, "hash avalanche");
    check_balanced_bits(lane_difference_counts, "hash lane difference");

    std::string benchmark_data(1 << 20, '\0');
    for (auto& value : benchmark_data)
        value = static_cast<char>(generator());
    constexpr int benchmark_iterations = 64;
    std::string benchmark_digest;
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < benchmark_iterations; ++i)
        benchmark_digest = deep_jit::hash::get_hex_digest(benchmark_data);
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    DJ_HOST_ASSERT(benchmark_digest.size() == 32);
    std::printf("Hash throughput: %.2f GiB/s\n",
                static_cast<double>(benchmark_data.size()) * benchmark_iterations / elapsed / (1ull << 30));
}

void test_lazy_init_and_kernel_arguments() {
    struct Value {
        int value;
    };

    int num_initializations = 0;
    deep_jit::LazyInit<Value> lazy([&] {
        ++num_initializations;
        return std::make_shared<Value>(Value{17});
    });
    DJ_HOST_ASSERT(num_initializations == 0, "lazy value was initialized eagerly");
    DJ_HOST_ASSERT(lazy->value == 17 and lazy.get()->value == 17, "lazy value is incorrect");
    DJ_HOST_ASSERT(num_initializations == 1, "lazy value was initialized more than once");

    const auto first_value = lazy.get();
    lazy = deep_jit::LazyInit<Value>([] { return std::make_shared<Value>(Value{29}); });
    DJ_HOST_ASSERT(lazy->value == 29, "reassigned lazy factory was not used");
    DJ_HOST_ASSERT(first_value->value == 17, "reassigning lazy initialization invalidated an existing shared owner");

    deep_jit::LazyInit<Value> empty(nullptr);
    expect_failure([&] { empty.get(); }, "lazy object must be initialized before use");

    deep_jit::LazyInit<Value> null_factory([] { return std::shared_ptr<Value>(); });
    expect_failure([&] { null_factory.get(); }, "lazy factory must not return nullptr");

    int num_attempts = 0;
    deep_jit::LazyInit<Value> retry([&] {
        if (++num_attempts == 1)
            throw std::runtime_error("initialization failed");
        return std::make_shared<Value>(Value{41});
    });
    expect_failure([&] { retry.get(); }, "initialization failed");
    DJ_HOST_ASSERT(retry->value == 41 and num_attempts == 2, "lazy initialization did not recover after an exception");

    int value = 0;
    DJ_HOST_ASSERT(deep_jit::cuda::kernel_arg_pointer(value) == &value);
    const deep_jit::NoRefPtr no_ref {&value};
    DJ_HOST_ASSERT(deep_jit::cuda::kernel_arg_pointer(no_ref) == &value);
    const deep_jit::NoRefPtr null_no_ref;
    DJ_HOST_ASSERT(deep_jit::cuda::kernel_arg_pointer(null_no_ref) == nullptr);
}

void test_parser(const fs::path& cache_root) {
    const auto include_original = get_test_cuda_project_dir() / "include_original";
    const auto include_same_content = get_test_cuda_project_dir() / "include_same_content";
    const auto include_changed_content = get_test_cuda_project_dir() / "include_changed_content";
    const auto source = get_source("tracked_include.cu");

    deep_jit::Parser parser_original({include_original}, {"test_cuda/"});
    deep_jit::Parser parser_same_content({include_same_content}, {"test_cuda/"});
    deep_jit::Parser parser_changed_content({include_changed_content}, {"test_cuda/"});
    DJ_HOST_ASSERT(parser_original.parse_into_hash(source) == parser_same_content.parse_into_hash(source),
                   "equal include contents must have equal hashes");
    DJ_HOST_ASSERT(parser_original.parse_into_hash(source) != parser_changed_content.parse_into_hash(source),
                   "changed include contents must change the hash");
    deep_jit::Parser parser_original_first({include_original, include_changed_content}, {"test_cuda/"});
    deep_jit::Parser parser_changed_first({include_changed_content, include_original}, {"test_cuda/"});
    DJ_HOST_ASSERT(parser_original_first.parse_into_hash(source) == parser_original.parse_into_hash(source));
    DJ_HOST_ASSERT(parser_changed_first.parse_into_hash(source) == parser_changed_content.parse_into_hash(source));

    deep_jit::Parser parser({include_original}, {"test_cuda/"});
    DJ_HOST_ASSERT(parser.parse_include(" \t# \tinclude \t<test_cuda/tracked_kernel.cuh> // comment") ==
                   "test_cuda/tracked_kernel.cuh");
    DJ_HOST_ASSERT(parser.parse_include("#include <cuda_runtime.h>").empty());
    DJ_HOST_ASSERT(parser.parse_include("#include <test_cuda_extra/tracked_kernel.cuh>").empty());
    DJ_HOST_ASSERT(parser.parse_include("#include TEST_HEADER").empty());
    DJ_HOST_ASSERT(parser.parse_include("#include /* comment */ <test_cuda/tracked_kernel.cuh>").empty());
    DJ_HOST_ASSERT(parser.parse_include("#include \\\n<test_cuda/tracked_kernel.cuh>").empty());
    expect_failure([&] { parser.parse_into_hash("#include \"test_cuda/tracked_kernel.cuh\"\n"); }, "non-standard include");
    expect_failure([&] { parser.parse_into_hash("#include \"cuda_runtime.h\"\n"); }, "non-standard include");

    deep_jit::Parser missing_parser({include_original}, {"test_cuda/"});
    expect_failure([&] { missing_parser.parse_into_hash("#include <test_cuda/missing_tracked_header.cuh>\n"); }, "failed to open");
    DJ_HOST_ASSERT(missing_parser.visiting.empty(), "missing include left stale parser state");

    deep_jit::Parser conditional_parser({include_original}, {"test_cuda/"});
    expect_failure(
        [&] {
            conditional_parser.parse_into_hash(
                "#if 0\n#include <test_cuda/missing_tracked_header.cuh>\n#endif\n");
        },
        "failed to open");

    deep_jit::Parser untracked_parser({include_original}, {"test_cuda/"});
    DJ_HOST_ASSERT(not untracked_parser.parse_into_hash("#include <cuda_runtime.h>\n").empty());
    const auto untracked_source = get_source("untracked_dependency.cu");
    deep_jit::Parser untracked_original({
        get_test_cuda_project_dir() / "third_party_original"}, {"test_cuda/"});
    deep_jit::Parser untracked_changed({
        get_test_cuda_project_dir() / "third_party_changed_content"}, {"test_cuda/"});
    DJ_HOST_ASSERT(untracked_original.parse_into_hash(untracked_source) ==
                       untracked_changed.parse_into_hash(untracked_source),
                   "untracked include contents unexpectedly affected the parser hash");

    expect_failure(
        [&] {
            untracked_parser.parse_into_hash(
                "/*\n#include <test_cuda/missing_tracked_header.cuh>\n*/\n");
        },
        "failed to open");

    const std::string binary_root_a("root\0A", 6);
    const std::string binary_root_b("root\0B", 6);
    DJ_HOST_ASSERT(untracked_parser.parse_into_hash(binary_root_a) != untracked_parser.parse_into_hash(binary_root_b),
                   "parser ignored bytes after a null in root source");
    const std::string binary_root_prefix_a("A\0tail", 6);
    const std::string binary_root_prefix_b("B\0tail", 6);
    DJ_HOST_ASSERT(untracked_parser.parse_into_hash(binary_root_prefix_a) !=
                       untracked_parser.parse_into_hash(binary_root_prefix_b),
                   "parser ignored bytes before a null in root source");

    const auto binary_include_a = cache_root / "binary_include_a";
    const auto binary_include_b = cache_root / "binary_include_b";
    deep_jit::make_dirs(binary_include_a / "binary");
    deep_jit::make_dirs(binary_include_b / "binary");
    deep_jit::write_file_sync(binary_include_a / "binary/dependency.cuh", std::string("header\0A", 8));
    deep_jit::write_file_sync(binary_include_b / "binary/dependency.cuh", std::string("header\0B", 8));
    deep_jit::write_file_sync(binary_include_a / "binary/prefix_dependency.cuh", std::string("A\0tail", 6));
    deep_jit::write_file_sync(binary_include_b / "binary/prefix_dependency.cuh", std::string("B\0tail", 6));
    deep_jit::Parser binary_parser_a({binary_include_a}, {"binary/"});
    deep_jit::Parser binary_parser_b({binary_include_b}, {"binary/"});
    const std::string binary_include_source = "#include <binary/dependency.cuh>\n";
    DJ_HOST_ASSERT(binary_parser_a.parse_into_hash(binary_include_source) !=
                       binary_parser_b.parse_into_hash(binary_include_source),
                   "parser ignored bytes after a null in an included file");
    const std::string binary_include_prefix_source = "#include <binary/prefix_dependency.cuh>\n";
    DJ_HOST_ASSERT(binary_parser_a.parse_into_hash(binary_include_prefix_source) !=
                       binary_parser_b.parse_into_hash(binary_include_prefix_source),
                   "parser ignored bytes before a null in an included file");

    const auto cycle_dir = get_test_cuda_project_dir() / "include_cycle";
    deep_jit::Parser cycle_parser({cycle_dir}, {"test_cuda/"});
    expect_failure(
        [&] { cycle_parser.parse_into_hash("#include <test_cuda/circular_include_entry.cuh>\n"); },
        "circular include");
    DJ_HOST_ASSERT(cycle_parser.visiting.empty(), "circular include left stale parser state");

    const auto recovery_dir = cache_root / "parser_recovery";
    deep_jit::make_dirs(recovery_dir / "recovery");
    deep_jit::Parser recovery_parser({recovery_dir}, {"recovery/"});
    const std::string recovery_source = "#include <recovery/generated_header.cuh>\n";
    expect_failure([&] { recovery_parser.parse_into_hash(recovery_source); }, "failed to open");
    deep_jit::write_file_sync(recovery_dir / "recovery/generated_header.cuh", "#pragma once\n");
    DJ_HOST_ASSERT(not recovery_parser.parse_into_hash(recovery_source).empty(),
                   "parser did not recover after a missing include was created");
}

void test_generated_include_graph(const fs::path& cache_root) {
    constexpr int num_headers = 64;
    const auto root_a = cache_root / "generated_includes_a";
    const auto root_b = cache_root / "generated_includes_b";
    deep_jit::make_dirs(root_a / "generated");
    deep_jit::make_dirs(root_b / "generated");

    for (int index = 0; index < num_headers; ++index) {
        std::string code = "#pragma once\n";
        if (index > 0)
            code += std::format("#include <generated/generated_dependency_{}.cuh>\n", index - 1);
        if (index > 2)
            code += std::format("#include <generated/generated_dependency_{}.cuh>\n", index / 2);
        code += std::format("constexpr int kGeneratedDependency{} = {};\n", index, index);
        const auto filename = std::format("generated/generated_dependency_{}.cuh", index);
        deep_jit::write_file_sync(root_a / filename, code);
        deep_jit::write_file_sync(root_b / filename, code);
    }

    const auto source = std::format("#include <generated/generated_dependency_{}.cuh>\n", num_headers - 1);
    deep_jit::Parser parser_a({root_a}, {"generated/"});
    deep_jit::Parser parser_b({root_b}, {"generated/"});
    DJ_HOST_ASSERT(parser_a.parse_into_hash(source) == parser_b.parse_into_hash(source),
                   "equivalent generated include graphs must be path-independent");

    deep_jit::write_file_sync(
        root_b / "generated/generated_dependency_0.cuh",
        "#pragma once\nconstexpr int kGeneratedDependency0 = 999;\n");
    DJ_HOST_ASSERT(parser_a.parse_into_hash(source) == parser_b.parse_into_hash(source),
                   "existing parser unexpectedly refreshed its include cache");
    deep_jit::Parser changed_parser({root_b}, {"generated/"});
    DJ_HOST_ASSERT(parser_a.parse_into_hash(source) != changed_parser.parse_into_hash(source),
                   "a transitive generated include change did not affect the hash");
}

void test_options(Runtime& runtime) {
    const auto& defaults = runtime.default_compiler_options;
    DJ_HOST_ASSERT(defaults.optimize_level == "3");
    DJ_HOST_ASSERT(defaults.fast_math == false);
    DJ_HOST_ASSERT(defaults.ptxas_verbose == false);
    DJ_HOST_ASSERT(defaults.ptxas_register_usage_level == 10);
    DJ_HOST_ASSERT(defaults.check_no_spills == false);
    DJ_HOST_ASSERT(defaults.check_no_local_memory == false);
    DJ_HOST_ASSERT(defaults.with_line_info == false);
    DJ_HOST_ASSERT(defaults.dump_ptx == false and defaults.dump_sass == false);
    DJ_HOST_ASSERT(defaults.arch.has_value() and defaults.nvcc_flags.has_value());
    DJ_HOST_ASSERT(not defaults.post_hook.has_value());
    const auto& default_launch_options = runtime.default_launch_options;
    DJ_HOST_ASSERT(not default_launch_options.stream.has_value());
    DJ_HOST_ASSERT(default_launch_options.num_smem_bytes == 0);
    DJ_HOST_ASSERT(not default_launch_options.grid_dim.has_value() and not default_launch_options.block_dim.has_value());
    DJ_HOST_ASSERT(default_launch_options.cluster_dim->x == 1 and
                   default_launch_options.cluster_dim->y == 1 and default_launch_options.cluster_dim->z == 1);
    DJ_HOST_ASSERT(default_launch_options.cooperative == false and default_launch_options.enable_pdl == false and
                   default_launch_options.nonportable_cluster_size_allowed == false);
    const std::vector<std::string> expected_default_flags = {
        "--gpu-architecture=sm_" + *defaults.arch,
        "-O3",
        "--compiler-options=-O3",
        "--ptxas-options=--register-usage-level=10",
        "-std=c++20",
        "--compiler-options=-fPIC",
        "--compiler-options=-fconcepts",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    };
    DJ_HOST_ASSERT(defaults.get_flags() == expected_default_flags, "unexpected default NVCC flag order");

    struct BoolEnvironmentOption {
        std::string_view suffix;
        std::optional<bool> CompilerOptions::*member;
    };
    const std::array bool_environment_options = {
        BoolEnvironmentOption{"JIT_PTXAS_VERBOSE", &CompilerOptions::ptxas_verbose},
        BoolEnvironmentOption{"JIT_CHECK_NO_SPILLS", &CompilerOptions::check_no_spills},
        BoolEnvironmentOption{"JIT_CHECK_NO_LOCAL_MEMORY", &CompilerOptions::check_no_local_memory},
        BoolEnvironmentOption{"JIT_WITH_LINEINFO", &CompilerOptions::with_line_info},
        BoolEnvironmentOption{"JIT_DUMP_PTX", &CompilerOptions::dump_ptx},
        BoolEnvironmentOption{"JIT_DUMP_SASS", &CompilerOptions::dump_sass},
    };
    for (const auto& option : bool_environment_options) {
        const auto unprefixed_name = std::string(option.suffix);
        const auto global_name = "DJ_" + unprefixed_name;
        const auto library_name = "OPTIONS_TEST_" + unprefixed_name;
        unset_env(unprefixed_name);
        unset_env(global_name);
        unset_env(library_name);

        set_env(unprefixed_name, "1");
        DJ_HOST_ASSERT((CompilerOptions::default_options(deep_jit::Env("OPTIONS_TEST"), runtime.device).*(option.member)) == false,
                       "unprefixed environment variable unexpectedly took effect: {}", option.suffix);
        set_env(global_name, "1");
        DJ_HOST_ASSERT((CompilerOptions::default_options(deep_jit::Env("OPTIONS_TEST"), runtime.device).*(option.member)) == true,
                       "DJ environment fallback did not take effect: {}", option.suffix);
        set_env(library_name, "0");
        DJ_HOST_ASSERT((CompilerOptions::default_options(deep_jit::Env("OPTIONS_TEST"), runtime.device).*(option.member)) == false,
                       "library false did not override DJ true: {}", option.suffix);
        set_env(global_name, "0");
        set_env(library_name, "1");
        DJ_HOST_ASSERT((CompilerOptions::default_options(deep_jit::Env("OPTIONS_TEST"), runtime.device).*(option.member)) == true,
                       "library true did not override DJ false: {}", option.suffix);

        unset_env(unprefixed_name);
        unset_env(global_name);
        unset_env(library_name);
    }

    set_env("DJ_JIT_DEBUG", "1");
    set_env("OPTIONS_TEST_JIT_DEBUG", "0");
    auto env_options = CompilerOptions::default_options(deep_jit::Env("OPTIONS_TEST"), runtime.device);
    DJ_HOST_ASSERT(env_options.ptxas_verbose == false and env_options.with_line_info == false and
                       env_options.dump_ptx == false and env_options.dump_sass == false,
                   "library debug=false did not override the DJ debug fallback");
    set_env("OPTIONS_TEST_JIT_DEBUG", "1");
    set_env("OPTIONS_TEST_JIT_PTXAS_VERBOSE", "0");
    set_env("OPTIONS_TEST_JIT_WITH_LINEINFO", "0");
    set_env("OPTIONS_TEST_JIT_DUMP_PTX", "0");
    set_env("OPTIONS_TEST_JIT_DUMP_SASS", "0");
    env_options = CompilerOptions::default_options(deep_jit::Env("OPTIONS_TEST"), runtime.device);
    DJ_HOST_ASSERT(env_options.ptxas_verbose == true and env_options.with_line_info == true and
                       env_options.dump_ptx == true and env_options.dump_sass == true,
                   "debug=true must enable all derived compiler diagnostics");
    unset_env("OPTIONS_TEST_JIT_DEBUG");
    unset_env("DJ_JIT_DEBUG");
    unset_env("OPTIONS_TEST_JIT_PTXAS_VERBOSE");
    unset_env("OPTIONS_TEST_JIT_WITH_LINEINFO");
    unset_env("OPTIONS_TEST_JIT_DUMP_PTX");
    unset_env("OPTIONS_TEST_JIT_DUMP_SASS");

    set_env("OPTIONS_TEST_JIT_DUMP_ASM", "1");
    set_env("OPTIONS_TEST_JIT_DUMP_PTX", "0");
    set_env("OPTIONS_TEST_JIT_DUMP_SASS", "0");
    env_options = CompilerOptions::default_options(deep_jit::Env("OPTIONS_TEST"), runtime.device);
    DJ_HOST_ASSERT(env_options.dump_ptx == true and env_options.dump_sass == true,
                   "dump-asm=true must enable both PTX and SASS dumps");
    unset_env("OPTIONS_TEST_JIT_DUMP_ASM");
    unset_env("OPTIONS_TEST_JIT_DUMP_PTX");
    unset_env("OPTIONS_TEST_JIT_DUMP_SASS");

    env_options.check_no_local_memory = true;
    DJ_HOST_ASSERT(env_options.override_with(CompilerOptions{.check_no_local_memory = false}).check_no_local_memory == false,
                   "an explicit false option must override a true default");

    const CompilerOptions overrides {
        .optimize_level = "0",
        .fast_math = true,
        .check_no_spills = true,
        .check_no_local_memory = true,
        .with_line_info = true,
        .extra_nvcc_flags = {"-DTEST_OPTION=17"},
    };
    const auto options = defaults.override_with(overrides);
    DJ_HOST_ASSERT(defaults.fast_math == false and defaults.with_line_info == false,
                   "compiler option override mutated the defaults");
    const auto flags = options.get_flags();
    const auto has_flag = [&](const std::string_view expected) {
        return std::ranges::find(flags, expected) != flags.end();
    };
    DJ_HOST_ASSERT(has_flag("-O0"));
    DJ_HOST_ASSERT(has_flag("--compiler-options=-O0"));
    DJ_HOST_ASSERT(has_flag("--use_fast_math"));
    DJ_HOST_ASSERT(has_flag("--ptxas-options=--register-usage-level=10"));
    DJ_HOST_ASSERT(has_flag("--ptxas-options=--warn-on-spills"));
    DJ_HOST_ASSERT(has_flag("--ptxas-options=--warn-on-local-memory-usage"));
    DJ_HOST_ASSERT(has_flag("--generate-line-info"));
    DJ_HOST_ASSERT(has_flag("-std=c++20"));
    DJ_HOST_ASSERT(has_flag("--compiler-options=-fPIC"));
    DJ_HOST_ASSERT(has_flag("--compiler-options=-fconcepts"));
    DJ_HOST_ASSERT(has_flag("--expt-relaxed-constexpr"));
    DJ_HOST_ASSERT(has_flag("--expt-extended-lambda"));
    DJ_HOST_ASSERT(has_flag("-DTEST_OPTION=17"));
    DJ_HOST_ASSERT(flags.back() == "-DTEST_OPTION=17", "extra NVCC flags must be appended last");

    const CompilerOptions replacement {
        .nvcc_flags = std::vector<std::string>{"-std=c++20", "-DREPLACED_FLAGS=1"},
    };
    const auto replacement_flags = defaults.override_with(replacement).get_flags();
    DJ_HOST_ASSERT(std::ranges::find(replacement_flags, "-DREPLACED_FLAGS=1") != replacement_flags.end());
    DJ_HOST_ASSERT(std::ranges::find(replacement_flags, "--compiler-options=-fPIC") == replacement_flags.end(),
                   "nvcc_flags must replace the default free-form flags");

    auto defaults_with_extra = defaults;
    defaults_with_extra.extra_nvcc_flags = {"-DDEFAULT_EXTRA=1"};
    const auto appended = defaults_with_extra.override_with(CompilerOptions {
        .extra_nvcc_flags = {"-DPER_KERNEL_EXTRA=1"},
    });
    const std::vector<std::string> expected_extra_flags = {"-DDEFAULT_EXTRA=1", "-DPER_KERNEL_EXTRA=1"};
    DJ_HOST_ASSERT(appended.extra_nvcc_flags == expected_extra_flags,
                   "per-kernel extra NVCC flags must append to runtime defaults");

    const auto size_options = defaults.override_with(CompilerOptions {
        .optimize_level = "s",
        .ptxas_register_usage_level = 0,
    });
    const auto size_flags = size_options.get_flags();
    DJ_HOST_ASSERT(std::ranges::find(size_flags, "-Os") != size_flags.end());
    DJ_HOST_ASSERT(std::ranges::find(size_flags, "--compiler-options=-Os") != size_flags.end());
    DJ_HOST_ASSERT(std::ranges::find(size_flags, "--ptxas-options=--register-usage-level=0") != size_flags.end());

    auto hook_defaults = defaults;
    hook_defaults.post_hook = "scripts/post_hook.py";
    const auto inherited_hook = hook_defaults.override_with(CompilerOptions {.post_hook = std::nullopt});
    DJ_HOST_ASSERT(inherited_hook.post_hook.has_value() and *inherited_hook.post_hook == "scripts/post_hook.py",
                   "a null post-hook override must inherit the runtime default");
    const auto empty_hook = defaults.override_with(CompilerOptions {.post_hook = ""});
    DJ_HOST_ASSERT(empty_hook.post_hook.has_value() and empty_hook.post_hook->empty(),
                   "an empty post hook must remain explicitly configured");

    auto enabled = defaults;
    enabled.fast_math = true;
    enabled.ptxas_verbose = true;
    enabled.check_no_spills = true;
    enabled.check_no_local_memory = true;
    enabled.with_line_info = true;
    enabled.dump_ptx = true;
    enabled.dump_sass = true;
    const auto disabled = enabled.override_with(CompilerOptions {
        .fast_math = false,
        .ptxas_verbose = false,
        .check_no_spills = false,
        .check_no_local_memory = false,
        .with_line_info = false,
        .dump_ptx = false,
        .dump_sass = false,
    });
    DJ_HOST_ASSERT(disabled.fast_math == false and disabled.ptxas_verbose == false and
                   disabled.check_no_spills == false and disabled.check_no_local_memory == false and
                   disabled.with_line_info == false and disabled.dump_ptx == false and disabled.dump_sass == false,
                   "explicit false compiler options must override true defaults");

    auto missing_arch = defaults;
    missing_arch.arch.reset();
    expect_failure([&] { (void)missing_arch.get_flags(); }, "CUDA architecture must be specified");
    auto missing_optimization = defaults;
    missing_optimization.optimize_level.reset();
    expect_failure([&] { (void)missing_optimization.get_flags(); }, "optimization level must be specified");
    auto missing_register_level = defaults;
    missing_register_level.ptxas_register_usage_level.reset();
    expect_failure([&] { (void)missing_register_level.get_flags(); }, "register usage level must be specified");

    auto launch_defaults = LaunchOptions::default_options(runtime.env);
    launch_defaults.stream = reinterpret_cast<CUstream>(0x1);
    launch_defaults.num_smem_bytes = 16;
    launch_defaults.grid_dim = dim3(2, 3, 4);
    launch_defaults.block_dim = dim3(32, 2, 1);
    launch_defaults.cluster_dim = dim3(2, 1, 1);
    launch_defaults.cooperative = true;
    launch_defaults.enable_pdl = true;
    launch_defaults.nonportable_cluster_size_allowed = true;
    const auto launch_options = launch_defaults.override_with(LaunchOptions {
        .stream = nullptr,
        .num_smem_bytes = 0,
        .grid_dim = dim3(1, 1, 1),
        .block_dim = dim3(1, 1, 1),
        .cluster_dim = dim3(1, 1, 1),
        .cooperative = false,
        .enable_pdl = false,
        .nonportable_cluster_size_allowed = false,
    });
    DJ_HOST_ASSERT(launch_options.stream.has_value() and *launch_options.stream == nullptr);
    DJ_HOST_ASSERT(launch_options.num_smem_bytes == 0);
    DJ_HOST_ASSERT(launch_options.grid_dim->x == 1 and launch_options.block_dim->x == 1 and launch_options.cluster_dim->x == 1);
    DJ_HOST_ASSERT(launch_options.cooperative == false and launch_options.enable_pdl == false and
                   launch_options.nonportable_cluster_size_allowed == false,
                   "explicit false launch options must override true defaults");
    DJ_HOST_ASSERT(launch_defaults.cooperative == true and launch_defaults.enable_pdl == true and
                   launch_defaults.nonportable_cluster_size_allowed == true,
                   "launch option override mutated the defaults");

    auto incomplete_defaults = LaunchOptions::default_options(runtime.env);
    incomplete_defaults.num_smem_bytes.reset();
    const auto incomplete_options = incomplete_defaults.override_with(LaunchOptions {
        .grid_dim = dim3(1, 1, 1),
        .block_dim = dim3(1, 1, 1),
    });
    DJ_HOST_ASSERT(not incomplete_options.num_smem_bytes.has_value());
}

void test_library_environment_compatibility(Runtime& runtime, const fs::path& cache_root) {
    set_env("DJ_JIT_DEBUG", "1");
    set_env("DJ_JIT_CPP_STANDARD", "17");
    set_env("DJ_JIT_CHECK_NO_LOCAL_MEMORY", "1");
    set_env("DG_JIT_DEBUG", "0");
    set_env("DG_JIT_CPP_STANDARD", "20");
    set_env("DG_JIT_CHECK_NO_SPILLS", "1");
    set_env("DG_JIT_CHECK_NO_LOCAL_MEMORY", "0");
    set_env("EP_JIT_CPP_STANDARD", "23");
    set_env("EP_JIT_CHECK_NO_LOCAL_MEMORY", "1");
    set_env("JIT_CPP_STANDARD", "14");
    set_env("JIT_CHECK_NO_SPILLS", "1");

    const auto dg_options = CompilerOptions::default_options(deep_jit::Env("DG"), runtime.device);
    const auto ep_options = CompilerOptions::default_options(deep_jit::Env("EP"), runtime.device);
    const auto other_options = CompilerOptions::default_options(deep_jit::Env("OTHER"), runtime.device);
    DJ_HOST_ASSERT(dg_options.ptxas_verbose == false and dg_options.with_line_info == false and
                   dg_options.dump_ptx == false and dg_options.dump_sass == false,
                   "DG_JIT_DEBUG=0 must override DJ_JIT_DEBUG=1");
    DJ_HOST_ASSERT(dg_options.check_no_spills == true);
    DJ_HOST_ASSERT(dg_options.check_no_local_memory == false,
                   "DG_JIT_CHECK_NO_LOCAL_MEMORY=0 must override the DJ default");
    DJ_HOST_ASSERT(std::ranges::find(*dg_options.nvcc_flags, "-std=c++20") != dg_options.nvcc_flags->end());
    DJ_HOST_ASSERT(ep_options.ptxas_verbose == true and ep_options.with_line_info == true and
                   ep_options.dump_ptx == true and ep_options.dump_sass == true,
                   "EP must inherit DJ_JIT_DEBUG");
    DJ_HOST_ASSERT(ep_options.check_no_local_memory == true);
    DJ_HOST_ASSERT(std::ranges::find(*ep_options.nvcc_flags, "-std=c++23") != ep_options.nvcc_flags->end());
    DJ_HOST_ASSERT(other_options.check_no_spills == false,
                   "unprefixed JIT_CHECK_NO_SPILLS must not be read");
    DJ_HOST_ASSERT(other_options.check_no_local_memory == true,
                   "DJ_JIT_CHECK_NO_LOCAL_MEMORY must be the global fallback");
    DJ_HOST_ASSERT(std::ranges::find(*other_options.nvcc_flags, "-std=c++17") != other_options.nvcc_flags->end(),
                   "DJ_JIT_CPP_STANDARD must be the global fallback");

    const auto dg_cache = cache_root / "dg_environment_cache";
    const auto ep_cache = cache_root / "ep_environment_cache";
    set_env("DG_JIT_CACHE_DIR", dg_cache.string());
    set_env("EP_JIT_CACHE_DIR", ep_cache.string());
    DJ_HOST_ASSERT(deep_jit::DiskCache::from_env(deep_jit::Env("DG")).paths == std::vector<fs::path>{dg_cache});
    DJ_HOST_ASSERT(deep_jit::DiskCache::from_env(deep_jit::Env("EP")).paths == std::vector<fs::path>{ep_cache});

    const auto compiler_override_dir = cache_root / "compiler_override";
    const auto compiler_override_cache = compiler_override_dir / "cache_root";
    const auto compiler_override = compiler_override_dir / "valid_nvcc";
    deep_jit::make_dirs(compiler_override_dir);
    write_executable(compiler_override, "#!/bin/sh\nprintf '%s\\n' 'Cuda compilation tools, release 13.1, V13.1.0'\n");
    set_env("COMPILER_OVERRIDE_JIT_CACHE_DIR", compiler_override_cache.string());
    set_env("COMPILER_OVERRIDE_JIT_NVCC_COMPILER", compiler_override.string());
    const auto override_runtime = make_runtime_with_prefix(
        "COMPILER_OVERRIDE", get_test_cuda_project_dir() / "include_original");
    DJ_HOST_ASSERT(override_runtime->backend.toolkit.nvcc == fs::absolute(compiler_override).lexically_normal(),
                   "library-prefixed NVCC override was not selected");

    set_env("INVALID_COMPILER_JIT_NVCC_COMPILER", (compiler_override_dir / "missing_nvcc").string());
    expect_failure(
        [&] { make_runtime_with_prefix("INVALID_COMPILER", get_test_cuda_project_dir() / "include_original"); },
        "NVCC compiler is not executable");

    const auto malformed_compiler = compiler_override_dir / "malformed_nvcc";
    write_executable(malformed_compiler, "#!/bin/sh\nprintf '%s\\n' 'not an NVCC version'\n");
    set_env("MALFORMED_COMPILER_JIT_NVCC_COMPILER", malformed_compiler.string());
    expect_failure(
        [&] { make_runtime_with_prefix("MALFORMED_COMPILER", get_test_cuda_project_dir() / "include_original"); },
        "failed to parse NVCC version");

    const auto old_compiler = compiler_override_dir / "old_nvcc";
    write_executable(old_compiler, "#!/bin/sh\nprintf '%s\\n' 'Cuda compilation tools, release 12.8, V12.8.0'\n");
    set_env("OLD_COMPILER_JIT_NVCC_COMPILER", old_compiler.string());
    expect_failure(
        [&] { make_runtime_with_prefix("OLD_COMPILER", get_test_cuda_project_dir() / "include_original"); },
        "NVCC version must be at least 12.9");

    unset_env("DJ_JIT_DEBUG");
    unset_env("DJ_JIT_CPP_STANDARD");
    unset_env("DJ_JIT_CHECK_NO_LOCAL_MEMORY");
    unset_env("DG_JIT_DEBUG");
    unset_env("DG_JIT_CPP_STANDARD");
    unset_env("DG_JIT_CHECK_NO_SPILLS");
    unset_env("DG_JIT_CHECK_NO_LOCAL_MEMORY");
    unset_env("EP_JIT_CPP_STANDARD");
    unset_env("EP_JIT_CHECK_NO_LOCAL_MEMORY");
    unset_env("DG_JIT_CACHE_DIR");
    unset_env("EP_JIT_CACHE_DIR");
    unset_env("COMPILER_OVERRIDE_JIT_CACHE_DIR");
    unset_env("COMPILER_OVERRIDE_JIT_NVCC_COMPILER");
    unset_env("INVALID_COMPILER_JIT_NVCC_COMPILER");
    unset_env("MALFORMED_COMPILER_JIT_NVCC_COMPILER");
    unset_env("OLD_COMPILER_JIT_NVCC_COMPILER");
    unset_env("JIT_CPP_STANDARD");
    unset_env("JIT_CHECK_NO_SPILLS");

    set_env("JIT_DEBUG", "1");
    const auto unprefixed_options = CompilerOptions::default_options(deep_jit::Env("UNPREFIXED"), runtime.device);
    DJ_HOST_ASSERT(unprefixed_options.ptxas_verbose == false and unprefixed_options.with_line_info == false and
                   unprefixed_options.dump_ptx == false and unprefixed_options.dump_sass == false,
                   "unprefixed JIT_DEBUG must not affect compiler defaults");
    unset_env("JIT_DEBUG");
}

void test_cuda_toolkit_discovery(const fs::path& cache_root) {
    const auto saved_cuda_home = get_raw_env("CUDA_HOME");
    const auto saved_cuda_path = get_raw_env("CUDA_PATH");
    const auto saved_path = get_raw_env("PATH");
    DJ_HOST_ASSERT(saved_path.has_value());

    const auto toolkit_a = cache_root / "toolkit_a";
    const auto toolkit_b = cache_root / "toolkit_b";
    deep_jit::make_dirs(toolkit_a / "bin");
    deep_jit::make_dirs(toolkit_b / "bin");
    write_executable(toolkit_a / "bin/nvcc", "#!/bin/sh\nexit 0\n");
    write_executable(toolkit_b / "bin/nvcc", "#!/bin/sh\nexit 0\n");

    const deep_jit::Env env("TOOLKIT_DISCOVERY");
    set_env("CUDA_HOME", toolkit_a.string());
    set_env("CUDA_PATH", toolkit_b.string());
    DJ_HOST_ASSERT(deep_jit::CUDA::find_cuda_toolkit(env).nvcc == toolkit_a / "bin/nvcc",
                   "CUDA_HOME must take priority over CUDA_PATH");

    set_env("DJ_JIT_NVCC_COMPILER", (toolkit_b / "bin/nvcc").string());
    DJ_HOST_ASSERT(deep_jit::CUDA::find_cuda_toolkit(env).nvcc == toolkit_b / "bin/nvcc",
                   "DJ NVCC override was not used as the global fallback");
    set_env("TOOLKIT_DISCOVERY_JIT_NVCC_COMPILER", (toolkit_a / "bin/nvcc").string());
    DJ_HOST_ASSERT(deep_jit::CUDA::find_cuda_toolkit(env).nvcc == toolkit_a / "bin/nvcc",
                   "library NVCC override did not take priority over the DJ fallback");
    unset_env("TOOLKIT_DISCOVERY_JIT_NVCC_COMPILER");
    unset_env("DJ_JIT_NVCC_COMPILER");
    set_env("JIT_NVCC_COMPILER", (toolkit_b / "bin/nvcc").string());
    DJ_HOST_ASSERT(deep_jit::CUDA::find_cuda_toolkit(env).nvcc == toolkit_a / "bin/nvcc",
                   "unprefixed NVCC override unexpectedly took effect");
    unset_env("JIT_NVCC_COMPILER");

    set_env("CUDA_HOME", "");
    DJ_HOST_ASSERT(deep_jit::CUDA::find_cuda_toolkit(env).nvcc == toolkit_b / "bin/nvcc",
                   "CUDA_PATH was not used when CUDA_HOME was empty");

    unset_env("CUDA_HOME");
    unset_env("CUDA_PATH");
    set_env("PATH", (toolkit_a / "bin").string() + ":/usr/bin:/bin");
    DJ_HOST_ASSERT(deep_jit::CUDA::find_cuda_toolkit(env).nvcc == toolkit_a / "bin/nvcc",
                   "which nvcc discovery did not select the PATH compiler");

    if (deep_jit::is_executable("/usr/local/cuda/bin/nvcc")) {
        const auto empty_path = cache_root / "empty_path";
        deep_jit::make_dirs(empty_path);
        set_env("PATH", empty_path.string());
        DJ_HOST_ASSERT(deep_jit::CUDA::find_cuda_toolkit(env).nvcc == "/usr/local/cuda/bin/nvcc",
                       "/usr/local/cuda fallback was not used after which nvcc failed");
    }

    set_env("CUDA_HOME", (cache_root / "missing_cuda_home").string());
    expect_failure([&] { deep_jit::CUDA::find_cuda_toolkit(env); }, "not home_path.empty()");

    restore_env("CUDA_HOME", saved_cuda_home);
    restore_env("CUDA_PATH", saved_cuda_path);
    restore_env("PATH", saved_path);
}

void test_multiple_runtimes(const fs::path& cache_root) {
    const auto include_dir = get_test_cuda_project_dir() / "include_original";
    const auto source = get_source("compiler_options.cu");
    const auto cache_a = cache_root / "runtime_a";
    const auto cache_b = cache_root / "runtime_b";
    set_env("RUNTIME_A_JIT_CACHE_DIR", cache_a.string());
    set_env("RUNTIME_B_JIT_CACHE_DIR", cache_b.string());

    const auto runtime_a = make_runtime_with_prefix("RUNTIME_A", include_dir);
    const auto runtime_b = make_runtime_with_prefix("RUNTIME_B", include_dir);
    runtime_a->default_compiler_options.extra_nvcc_flags.emplace_back("-DTEST_OPTION=11");
    runtime_b->default_compiler_options.extra_nvcc_flags.emplace_back("-DTEST_OPTION=29");
    runtime_a->default_launch_options.enable_pdl = true;
    DJ_HOST_ASSERT(runtime_b->default_launch_options.enable_pdl == false,
                   "default launch options leaked between runtimes");
    runtime_a->default_launch_options.enable_pdl = false;
    DJ_HOST_ASSERT(runtime_a->disk_cache.paths.front() == cache_a and runtime_b->disk_cache.paths.front() == cache_b,
                   "runtime cache roots are not isolated");
    DJ_HOST_ASSERT(runtime_a->cache_key(source, runtime_a->default_compiler_options) !=
                   runtime_b->cache_key(source, runtime_b->default_compiler_options),
                   "runtime-specific compiler defaults must affect the cache key");
    DJ_HOST_ASSERT(launch_value(*runtime_a, runtime_a->compile("runtime_isolation", source)) == 11);
    DJ_HOST_ASSERT(runtime_b->mem_cache.cache.empty(), "one runtime populated another runtime's memory cache");
    DJ_HOST_ASSERT(launch_value(*runtime_b, runtime_b->compile("runtime_isolation", source)) == 29);

    const auto shared_cache = cache_root / "shared_runtime";
    set_env("SHARED_RUNTIME_JIT_CACHE_DIR", shared_cache.string());
    const auto shared_a = make_runtime_with_prefix("SHARED_RUNTIME", include_dir);
    const auto shared_b = make_runtime_with_prefix("SHARED_RUNTIME", include_dir);
    const auto shared_source = get_template_source(13);
    const auto artifact_a = shared_a->compile_without_load("shared_runtime", shared_source);
    const auto artifact_b = shared_b->compile_without_load("shared_runtime", shared_source);
    DJ_HOST_ASSERT(artifact_a == artifact_b, "equivalent runtimes did not share the disk cache");
    const auto kernel_a = shared_a->compile("shared_runtime", shared_source);
    const auto kernel_b = shared_b->compile("shared_runtime", shared_source);
    DJ_HOST_ASSERT(kernel_a != kernel_b, "different runtimes unexpectedly shared the memory cache");
    DJ_HOST_ASSERT(launch_value(*shared_a, kernel_a, 1) == 14 and launch_value(*shared_b, kernel_b, 2) == 15);

    const auto snapshot_cache_a = cache_root / "snapshot_a";
    const auto snapshot_cache_b = cache_root / "snapshot_b";
    set_env("SNAPSHOT_JIT_CACHE_DIR", snapshot_cache_a.string());
    set_env("SNAPSHOT_JIT_CPP_STANDARD", "17");
    const auto snapshot_a = make_runtime_with_prefix("SNAPSHOT", include_dir);
    set_env("SNAPSHOT_JIT_CACHE_DIR", snapshot_cache_b.string());
    set_env("SNAPSHOT_JIT_CPP_STANDARD", "23");
    const auto snapshot_b = make_runtime_with_prefix("SNAPSHOT", include_dir);
    DJ_HOST_ASSERT(snapshot_a->disk_cache.paths.front() == snapshot_cache_a and
                   snapshot_b->disk_cache.paths.front() == snapshot_cache_b,
                   "runtime construction did not snapshot its cache environment");
    DJ_HOST_ASSERT(std::ranges::find(*snapshot_a->default_compiler_options.nvcc_flags, "-std=c++17") !=
                       snapshot_a->default_compiler_options.nvcc_flags->end());
    DJ_HOST_ASSERT(std::ranges::find(*snapshot_b->default_compiler_options.nvcc_flags, "-std=c++23") !=
                       snapshot_b->default_compiler_options.nvcc_flags->end());
    DJ_HOST_ASSERT(snapshot_a->env.get<int>("JIT_CPP_STANDARD") == 23,
                   "runtime Env should continue to reflect process environment changes");

    unset_env("RUNTIME_A_JIT_CACHE_DIR");
    unset_env("RUNTIME_B_JIT_CACHE_DIR");
    unset_env("SHARED_RUNTIME_JIT_CACHE_DIR");
    unset_env("SNAPSHOT_JIT_CACHE_DIR");
    unset_env("SNAPSHOT_JIT_CPP_STANDARD");
}

void test_lazy_runtime(const fs::path& cache_root) {
    const auto include_dir = get_test_cuda_project_dir() / "include_original";
    const auto first_cache = cache_root / "lazy_runtime_first";
    const auto second_cache = cache_root / "lazy_runtime_second";
    set_env("LAZY_RUNTIME_JIT_CACHE_DIR", first_cache.string());
    auto lazy = deep_jit::create_lazy_jit<deep_jit::CUDA>(deep_jit::Config(
        get_test_cuda_project_dir(),
        "LAZY_RUNTIME",
        "lazy-runtime",
        {include_dir},
        {"test_cuda/"}));
    set_env("LAZY_RUNTIME_JIT_CACHE_DIR", second_cache.string());

    const auto first = lazy.get();
    const auto second = lazy.get();
    DJ_HOST_ASSERT(first == second, "create_lazy_jit initialized more than once");
    DJ_HOST_ASSERT(first->disk_cache.paths.front() == second_cache,
                   "create_lazy_jit constructed the runtime before first use");
    DJ_HOST_ASSERT(not fs::exists(first_cache), "unused pre-initialization cache path was created");
    unset_env("LAZY_RUNTIME_JIT_CACHE_DIR");
}

void test_consumer_default_compiler_flags(const fs::path& cache_root) {
    const auto include_dir = get_test_cuda_project_dir() / "include_original";
    const auto consumer_cache = cache_root / "consumer_defaults";
    set_env("CONSUMER_DEFAULTS_JIT_CACHE_DIR", consumer_cache.string());
    const auto runtime = make_runtime_with_prefix("CONSUMER_DEFAULTS", include_dir);
    const auto source = get_source("compiler_options.cu");
    const auto original_key = runtime->cache_key(source, runtime->default_compiler_options);

    auto& nvcc_flags = *runtime->default_compiler_options.nvcc_flags;
    nvcc_flags.emplace_back("--diag-suppress=39,161,174,177,186,940,3012");
    nvcc_flags.emplace_back("--compiler-options=-Wno-deprecated-declarations,-Wno-abi");
    nvcc_flags.emplace_back("-I" + include_dir.string());
    nvcc_flags.emplace_back("-DEP_NUM_TOPK_IDX_BITS=64");
    nvcc_flags.emplace_back("-DTEST_OPTION=64");
    DJ_HOST_ASSERT(runtime->cache_key(source, runtime->default_compiler_options) != original_key,
                   "consumer default compiler flags did not affect the cache key");

    const auto artifact = runtime->compile_without_load("consumer_defaults", source);
    const auto metadata = deep_jit::read(artifact / "meta.json");
    const std::vector<std::string> expected_flags = {
        "--diag-suppress=39,161,174,177,186,940,3012",
        "--compiler-options=-Wno-deprecated-declarations,-Wno-abi",
        "-I" + include_dir.string(),
        "-DEP_NUM_TOPK_IDX_BITS=64",
    };
    for (const auto& expected : expected_flags) {
        DJ_HOST_ASSERT(metadata.find(expected) != std::string::npos,
                       "consumer compiler flag is missing from metadata: {}", expected);
    }
    DJ_HOST_ASSERT(launch_value(*runtime, runtime->compile("consumer_defaults", source)) == 64,
                   "consumer default macro was not compiled");
    unset_env("CONSUMER_DEFAULTS_JIT_CACHE_DIR");
}

void test_debug_environment(const fs::path& cache_root) {
    const auto include_dir = get_test_cuda_project_dir() / "include_original";
    const auto debug_cache = cache_root / "debug_environment";
    set_env("DEBUG_ENV_JIT_CACHE_DIR", debug_cache.string());
    set_env("DEBUG_ENV_JIT_DEBUG", "1");
    const auto runtime = make_runtime_with_prefix("DEBUG_ENV", include_dir);
    DJ_HOST_ASSERT(runtime->default_compiler_options.ptxas_verbose == true);
    DJ_HOST_ASSERT(runtime->default_compiler_options.with_line_info == true);
    DJ_HOST_ASSERT(runtime->default_compiler_options.dump_ptx == true);
    DJ_HOST_ASSERT(runtime->default_compiler_options.dump_sass == true);

    const auto source = get_template_source(47);
    const CompilerOptions options {
        .dump_sass = runtime->backend.toolkit.cuobjdump.has_value(),
    };
    const auto artifact = runtime->compile_without_load("debug_environment", source, options);
    DJ_HOST_ASSERT(fs::file_size(artifact / "kernel.ptx") > 0);
    DJ_HOST_ASSERT(not runtime->backend.toolkit.cuobjdump or fs::file_size(artifact / "kernel.sass") > 0);
    DJ_HOST_ASSERT(launch_value(*runtime, runtime->compile("debug_environment", source, options), 1) == 48);
    unset_env("DEBUG_ENV_JIT_CACHE_DIR");
    unset_env("DEBUG_ENV_JIT_DEBUG");
}

void test_untracked_dependency_include_flags(const fs::path& cache_root) {
    const auto include_dir = get_test_cuda_project_dir() / "include_original";
    const auto dependency_original = get_test_cuda_project_dir() / "third_party_original";
    const auto dependency_same_content = get_test_cuda_project_dir() / "third_party_same_content";
    const auto dependency_changed_content = get_test_cuda_project_dir() / "third_party_changed_content";
    const auto dependency_cache = cache_root / "untracked_dependency";
    set_env("UNTRACKED_DEPENDENCY_JIT_CACHE_DIR", dependency_cache.string());
    const auto runtime = make_runtime_with_prefix("UNTRACKED_DEPENDENCY", include_dir);
    const auto source = get_source("untracked_dependency.cu");
    DJ_HOST_ASSERT(count_different_bytes(
                       deep_jit::read(dependency_original / "third_party/dependency_value.cuh"),
                       deep_jit::read(dependency_changed_content / "third_party/dependency_value.cuh")) == 1,
                   "untracked dependency fixture must differ by exactly one byte");

    const auto make_options = [&](const fs::path& dependency_dir) {
        auto options = runtime->default_compiler_options;
        options.nvcc_flags->emplace_back("-I" + dependency_dir.string());
        return options;
    };
    const auto original_options = make_options(dependency_original);
    const auto same_content_options = make_options(dependency_same_content);
    const auto changed_content_options = make_options(dependency_changed_content);
    DJ_HOST_ASSERT(runtime->cache_key(source, original_options) != runtime->cache_key(source, same_content_options),
                   "untracked dependency include paths must remain in the cache key");
    DJ_HOST_ASSERT(runtime->cache_key(source, original_options) != runtime->cache_key(source, changed_content_options));

    DJ_HOST_ASSERT(launch_value(*runtime, runtime->compile("untracked_dependency", source, original_options)) == 17);
    DJ_HOST_ASSERT(launch_value(*runtime, runtime->compile("untracked_dependency", source, changed_content_options)) == 18);

    const auto transitive_source = get_source("tracked_to_untracked_dependency.cu");
    deep_jit::Parser transitive_parser_original(
        {include_dir, dependency_original}, {"test_cuda/"});
    deep_jit::Parser transitive_parser_changed(
        {include_dir, dependency_changed_content}, {"test_cuda/"});
    DJ_HOST_ASSERT(transitive_parser_original.parse_into_hash(transitive_source) ==
                       transitive_parser_changed.parse_into_hash(transitive_source),
                   "parser must ignore an untracked dependency included by a tracked header");
    DJ_HOST_ASSERT(runtime->cache_key(transitive_source, original_options) !=
                       runtime->cache_key(transitive_source, changed_content_options),
                   "untracked transitive include paths must remain in the compiler-option hash");
    DJ_HOST_ASSERT(launch_value(
        *runtime,
        runtime->compile("tracked_to_untracked_dependency", transitive_source, original_options)) == 17);
    DJ_HOST_ASSERT(launch_value(
        *runtime,
        runtime->compile("tracked_to_untracked_dependency", transitive_source, changed_content_options)) == 18);
    unset_env("UNTRACKED_DEPENDENCY_JIT_CACHE_DIR");
}

bool test_multiple_device_runtimes(const fs::path& cache_root) {
    int num_devices = 0;
    DJ_CUDA_RUNTIME_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices < 2)
        return false;

    int original_device = 0;
    DJ_CUDA_RUNTIME_CHECK(cudaGetDevice(&original_device));
    const auto include_dir = get_test_cuda_project_dir() / "include_original";
    const auto shared_cache = cache_root / "multiple_devices";
    set_env("MULTI_DEVICE_JIT_CACHE_DIR", shared_cache.string());

    try {
        DJ_CUDA_RUNTIME_CHECK(cudaSetDevice(0));
        int device_0_clock_rate = 0;
        DJ_CUDA_RUNTIME_CHECK(cudaDeviceGetAttribute(&device_0_clock_rate, cudaDevAttrClockRate, 0));
        const auto runtime_0 = make_runtime_with_prefix("MULTI_DEVICE", include_dir);
        const auto runtime_0_arch = *runtime_0->default_compiler_options.arch;
        const auto runtime_0_clock_rate = runtime_0->device.get_clock_rate();
        const auto source = get_template_source(21);
        const auto artifact_0 = runtime_0->compile_without_load("multiple_devices", source);
        const auto kernel_0 = runtime_0->compile("multiple_devices", source);
        DJ_HOST_ASSERT(launch_value(*runtime_0, kernel_0, 1) == 22);

        DJ_CUDA_RUNTIME_CHECK(cudaSetDevice(1));
        const auto runtime_1 = make_runtime_with_prefix("MULTI_DEVICE", include_dir);
        const auto runtime_1_arch = *runtime_1->default_compiler_options.arch;
        const auto artifact_1 = runtime_1->compile_without_load("multiple_devices", source);
        if (runtime_0_arch == runtime_1_arch) {
            DJ_HOST_ASSERT(artifact_0 == artifact_1,
                           "matching architectures did not share the disk cache across devices");
        } else {
            DJ_HOST_ASSERT(artifact_0 != artifact_1,
                           "different architectures unexpectedly shared the disk cache");
        }
        DJ_HOST_ASSERT(runtime_0->device.get_arch() == runtime_0_arch,
                       "runtime device properties changed with the process current device");
        DJ_HOST_ASSERT(runtime_0_clock_rate == static_cast<int64_t>(device_0_clock_rate) * 1000);
        DJ_HOST_ASSERT(runtime_0->device.get_clock_rate() == runtime_0_clock_rate,
                       "runtime clock rate was not cached");
        const auto kernel_1 = runtime_1->compile("multiple_devices", source);
        DJ_HOST_ASSERT(launch_value(*runtime_1, kernel_1, 2) == 23);

        DJ_CUDA_RUNTIME_CHECK(cudaSetDevice(0));
        DJ_HOST_ASSERT(launch_value(*runtime_0, kernel_0, 3) == 24);
        DJ_CUDA_RUNTIME_CHECK(cudaSetDevice(1));
        DJ_HOST_ASSERT(launch_value(*runtime_1, kernel_1, 4) == 25);
    } catch (...) {
        cudaSetDevice(original_device);
        unset_env("MULTI_DEVICE_JIT_CACHE_DIR");
        throw;
    }

    DJ_CUDA_RUNTIME_CHECK(cudaSetDevice(original_device));
    unset_env("MULTI_DEVICE_JIT_CACHE_DIR");
    return true;
}

void test_arch_override(const fs::path& cache_root) {
    const auto include_dir = get_test_cuda_project_dir() / "include_original";
    const auto arch_cache = cache_root / "arch_override";
    set_env("ARCH_OVERRIDE_JIT_CACHE_DIR", arch_cache.string());
    const auto runtime = make_runtime_with_prefix("ARCH_OVERRIDE", include_dir);
    const auto source = get_template_source(38);
    const CompilerOptions options {
        .arch = runtime->device.get_arch(false),
    };
    const auto kernel = runtime->compile("arch_override", source, options);
    DJ_HOST_ASSERT(launch_value(*runtime, kernel, 1) == 39,
                   "kernel compiled for the concrete architecture did not launch");
    const auto metadata = deep_jit::read(runtime->compile_without_load("arch_override", source, options) / "meta.json");
    DJ_HOST_ASSERT(metadata.find("\"arch\":\"" + runtime->device.get_arch(false) + "\"") != std::string::npos,
                   "concrete architecture override is missing from metadata");
    unset_env("ARCH_OVERRIDE_JIT_CACHE_DIR");
}

void test_kernel_lifecycle(const fs::path& cache_root) {
    const auto include_dir = get_test_cuda_project_dir() / "include_original";
    const auto lifecycle_cache = cache_root / "kernel_lifecycle";
    set_env("KERNEL_LIFECYCLE_JIT_CACHE_DIR", lifecycle_cache.string());
    const auto runtime = make_runtime_with_prefix("KERNEL_LIFECYCLE", include_dir);

    const auto missing_dir = cache_root / "missing_cubin";
    deep_jit::make_dirs(missing_dir);
    expect_failure([&] { (void)deep_jit::CUDA::load(missing_dir, runtime->env); }, "missing CUDA CUBIN");

    const auto invalid_dir = cache_root / "invalid_cubin";
    deep_jit::make_dirs(invalid_dir);
    deep_jit::write_file_sync(invalid_dir / "kernel.cubin", "not a cubin");
    expect_any_failure([&] { (void)deep_jit::CUDA::load(invalid_dir, runtime->env); });

    const auto source = get_template_source(34);
    const auto kernel = runtime->compile("kernel_lifecycle", source);
    DJ_HOST_ASSERT(launch_value(*runtime, kernel, 1) == 35);
    kernel->unload();
    kernel->unload();
    expect_failure(
        [&] {
            runtime->launch(
                kernel, {
                    .grid_dim = dim3(1, 1, 1),
                    .block_dim = dim3(1, 1, 1),
                });
        },
        "kernel must be loaded before launch");

    const auto reloaded_runtime = make_runtime_with_prefix("KERNEL_LIFECYCLE", include_dir);
    DJ_HOST_ASSERT(launch_value(*reloaded_runtime, reloaded_runtime->compile("kernel_lifecycle", source), 2) == 36,
                   "kernel could not be reloaded from disk cache after unload");
    unset_env("KERNEL_LIFECYCLE_JIT_CACHE_DIR");
}

void test_python_api(pybind11::module_& module, const std::shared_ptr<Runtime>& runtime) {
    const auto initialized_runtime = python_api_jit.get();
    const auto first = module.attr("get_jit")().cast<std::shared_ptr<Runtime>>();
    const auto second = module.attr("get_jit")().cast<std::shared_ptr<Runtime>>();
    DJ_HOST_ASSERT(first == initialized_runtime and second == initialized_runtime,
                   "Python get_jit() did not return the registered runtime");

    python_api_jit = deep_jit::LazyInit<Runtime>([runtime] {
        ++python_api_num_initializations;
        return runtime;
    });
    const auto third = module.attr("get_jit")().cast<std::shared_ptr<Runtime>>();
    DJ_HOST_ASSERT(third == runtime and third != first, "Python get_jit() did not follow lazy runtime reassignment");
    DJ_HOST_ASSERT(first == initialized_runtime, "reassigning the Python JIT invalidated an existing runtime reference");
    DJ_HOST_ASSERT(python_api_num_initializations == 1, "replacement runtime was initialized more than once");
}

void test_device(Runtime& runtime) {
    deep_jit::cuda::Device num_sms_device;
    DJ_HOST_ASSERT(num_sms_device.get_num_sms() > 0,
                   "get_num_sms did not initialize CUDA device properties lazily");
    deep_jit::cuda::Device l2_device;
    DJ_HOST_ASSERT(l2_device.get_num_l2_cache_bytes() > 0,
                   "get_num_l2_cache_bytes did not initialize CUDA device properties lazily");
    deep_jit::cuda::Device smem_device;
    DJ_HOST_ASSERT(smem_device.get_num_smem_bytes() > 0,
                   "get_num_smem_bytes did not initialize CUDA device properties lazily");
    deep_jit::cuda::Device arch_device;
    DJ_HOST_ASSERT(not arch_device.get_arch().empty(),
                   "get_arch did not initialize CUDA device properties lazily");
    deep_jit::cuda::Device clock_device;
    DJ_HOST_ASSERT(clock_device.get_clock_rate() > 0, "get_clock_rate failed on a fresh device object");

    const auto [major, minor] = runtime.device.get_arch_pair();
    const auto arch_number = std::to_string(major * 10 + minor);
    const auto& first_prop = runtime.device.get_prop();
    const auto& second_prop = runtime.device.get_prop();
    DJ_HOST_ASSERT(&first_prop == &second_prop, "CUDA device properties were not cached");
    DJ_HOST_ASSERT(runtime.device.get_num_sms() > 0, "CUDA device has no SMs");
    DJ_HOST_ASSERT(runtime.device.get_num_l2_cache_bytes() > 0, "CUDA device has no L2 cache");
    DJ_HOST_ASSERT(runtime.device.get_num_smem_bytes() > 0, "CUDA device has no shared memory");
    DJ_HOST_ASSERT(runtime.device.get_clock_rate() > 0, "CUDA device clock rate is invalid");
    DJ_HOST_ASSERT(runtime.device.get_arch_major() == major and runtime.device.get_arch_minor() == minor);
    DJ_HOST_ASSERT(runtime.device.get_clock_rate() == runtime.device.get_clock_rate(), "CUDA clock rate was not cached");
    DJ_HOST_ASSERT(runtime.device.get_arch(true, true) == arch_number);
    DJ_HOST_ASSERT(runtime.device.get_arch(false, true) == arch_number);
    if (major < 9) {
        DJ_HOST_ASSERT(runtime.device.get_arch() == arch_number and runtime.device.get_arch(false) == arch_number);
    } else if (major == 9) {
        DJ_HOST_ASSERT(runtime.device.get_arch() == arch_number + "a" and runtime.device.get_arch(false) == arch_number + "a");
    } else {
        DJ_HOST_ASSERT(runtime.device.get_arch() == arch_number + "f" and runtime.device.get_arch(false) == arch_number + "a");
    }
}

void test_tma_driver_wrapper(Runtime& runtime) {
    void* tensor_data = nullptr;
    unsigned int* output = nullptr;
    try {
        DJ_CUDA_RUNTIME_CHECK(cudaMalloc(&tensor_data, 64 * 64 * sizeof(float)));
        DJ_CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&output), sizeof(int)));
        CUtensorMap tensor_map{};
        const cuuint64_t global_dims[2] = {64, 64};
        const cuuint64_t global_strides[1] = {64 * sizeof(float)};
        const cuuint32_t box_dims[2] = {32, 32};
        const cuuint32_t element_strides[2] = {1, 1};
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(
            &tensor_map,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            2,
            tensor_data,
            global_dims,
            global_strides,
            box_dims,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

        const auto bytes = std::bit_cast<std::array<unsigned char, sizeof(CUtensorMap)>>(tensor_map);
        unsigned int expected = 2166136261u;
        for (const auto byte : bytes)
            expected = (expected ^ byte) * 16777619u;

        const auto kernel = runtime.compile("tensor_map_argument", get_source("tensor_map_argument.cu"));
        runtime.launch(
            kernel, {
                .grid_dim = dim3(1, 1, 1),
                .block_dim = dim3(1, 1, 1),
            },
            tensor_map, output);
        DJ_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        unsigned int result = 0;
        DJ_CUDA_RUNTIME_CHECK(cudaMemcpy(&result, output, sizeof(result), cudaMemcpyDeviceToHost));
        DJ_HOST_ASSERT(result == expected, "CUtensorMap kernel argument contents changed");

        DJ_CUDA_RUNTIME_CHECK(cudaFree(output));
        output = nullptr;
        DJ_CUDA_RUNTIME_CHECK(cudaFree(tensor_data));
        tensor_data = nullptr;
    } catch (...) {
        if (output != nullptr)
            cudaFree(output);
        if (tensor_data != nullptr)
            cudaFree(tensor_data);
        throw;
    }
}

void test_runtime_launch_features(Runtime& runtime) {
    struct RuntimeLaunchArguments {
        int base;
        int multiplier;
        uint64_t delay_cycles;
    };

    const auto source = get_source("runtime_launch_features.cu");
    const auto kernel = runtime.compile("runtime_launch_features", source);
    int* output = nullptr;
    cudaStream_t stream = nullptr;
    DJ_CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&output), 2 * sizeof(int)));
    DJ_CUDA_RUNTIME_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const auto check_output = [&](const int first, const int second) {
        std::array<int, 2> host{};
        DJ_CUDA_RUNTIME_CHECK(cudaMemcpy(host.data(), output, sizeof(host), cudaMemcpyDeviceToHost));
        DJ_HOST_ASSERT(host[0] == first and host[1] == second,
                       "unexpected runtime launch output: {}, {}", host[0], host[1]);
    };
    const auto check_graph = [](const cudaGraph_t graph) {
        std::size_t num_nodes = 0;
        DJ_CUDA_RUNTIME_CHECK(cudaGraphGetNodes(graph, nullptr, &num_nodes));
        DJ_HOST_ASSERT(num_nodes == 1, "captured launch must produce exactly one graph node");
        cudaGraphNode_t node = nullptr;
        DJ_CUDA_RUNTIME_CHECK(cudaGraphGetNodes(graph, &node, &num_nodes));
        cudaGraphNodeType type{};
        DJ_CUDA_RUNTIME_CHECK(cudaGraphNodeGetType(node, &type));
        DJ_HOST_ASSERT(type == cudaGraphNodeTypeKernel, "captured launch did not produce a kernel node");
    };

    try {
        RuntimeLaunchArguments arguments {3, 4, 0};
        const deep_jit::NoRefPtr argument_storage {&arguments};
        const LaunchOptions inherit_runtime_defaults {
            .num_smem_bytes = sizeof(int),
            .grid_dim = dim3(2, 1, 1),
            .block_dim = dim3(32, 1, 1),
        };

        runtime.launch(
            kernel, {
                .stream = stream,
                .num_smem_bytes = sizeof(int),
                .grid_dim = dim3(2, 1, 1),
                .block_dim = dim3(32, 1, 1),
            },
            output, argument_storage, 5);
        DJ_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
        check_output(23, 24);

        runtime.default_launch_options.stream = stream;
        runtime.launch(
            kernel, inherit_runtime_defaults,
            output, argument_storage, 6);
        DJ_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
        check_output(27, 28);

        int device_index = 0;
        int supports_cluster_launch = 0;
        int supports_cooperative_launch = 0;
        DJ_CUDA_RUNTIME_CHECK(cudaGetDevice(&device_index));
        DJ_CUDA_RUNTIME_CHECK(cudaDeviceGetAttribute(
            &supports_cluster_launch, cudaDevAttrClusterLaunch, device_index));
        DJ_CUDA_RUNTIME_CHECK(cudaDeviceGetAttribute(
            &supports_cooperative_launch, cudaDevAttrCooperativeLaunch, device_index));
        if (runtime.device.get_arch_major() >= 9 and supports_cluster_launch and supports_cooperative_launch) {
            runtime.launch(
                kernel, {
                    .stream = stream,
                    .num_smem_bytes = sizeof(int),
                    .grid_dim = dim3(2, 1, 1),
                    .block_dim = dim3(32, 1, 1),
                    .cluster_dim = dim3(2, 1, 1),
                    .cooperative = true,
                    .nonportable_cluster_size_allowed = true,
                },
                output, argument_storage, 7);
            DJ_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
            check_output(31, 32);
        }

        runtime.default_launch_options.stream = std::nullopt;
        runtime.default_launch_options.enable_pdl = false;
        const auto current_stream = c10::cuda::getStreamFromPool();

        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graph_exec = nullptr;
        {
            const c10::cuda::CUDAStreamGuard stream_guard(current_stream);
            DJ_CUDA_RUNTIME_CHECK(cudaStreamBeginCapture(current_stream.stream(), cudaStreamCaptureModeGlobal));
            runtime.launch(
                kernel, {
                    .num_smem_bytes = sizeof(int),
                    .grid_dim = dim3(2, 1, 1),
                    .block_dim = dim3(32, 1, 1),
                },
                output, argument_storage, 10);
            DJ_CUDA_RUNTIME_CHECK(cudaStreamEndCapture(current_stream.stream(), &graph));
        }
        check_graph(graph);
        DJ_CUDA_RUNTIME_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        DJ_CUDA_RUNTIME_CHECK(cudaGraphLaunch(graph_exec, current_stream.stream()));
        DJ_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(current_stream.stream()));
        check_output(43, 44);
        DJ_CUDA_RUNTIME_CHECK(cudaGraphExecDestroy(graph_exec));
        DJ_CUDA_RUNTIME_CHECK(cudaGraphDestroy(graph));

        graph = nullptr;
        graph_exec = nullptr;
        {
            const c10::cuda::CUDAStreamGuard stream_guard(current_stream);
            DJ_CUDA_RUNTIME_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            runtime.launch(
                kernel, {
                    .stream = stream,
                    .num_smem_bytes = sizeof(int),
                    .grid_dim = dim3(2, 1, 1),
                    .block_dim = dim3(32, 1, 1),
                },
                output, argument_storage, 11);
            DJ_CUDA_RUNTIME_CHECK(cudaStreamEndCapture(stream, &graph));
        }
        check_graph(graph);
        DJ_CUDA_RUNTIME_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        DJ_CUDA_RUNTIME_CHECK(cudaGraphLaunch(graph_exec, stream));
        DJ_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
        check_output(47, 48);
        DJ_CUDA_RUNTIME_CHECK(cudaGraphExecDestroy(graph_exec));
        DJ_CUDA_RUNTIME_CHECK(cudaGraphDestroy(graph));

        DJ_CUDA_RUNTIME_CHECK(cudaStreamDestroy(stream));
        stream = nullptr;
        DJ_CUDA_RUNTIME_CHECK(cudaFree(output));
    } catch (...) {
        if (stream != nullptr)
            cudaStreamDestroy(stream);
        cudaFree(output);
        throw;
    }
}

bool test_cooperative_launch(Runtime& runtime) {
    const auto kernel = runtime.compile("cooperative_grid_sync", get_source("cooperative_grid_sync.cu"));
    int device_index = 0;
    int supported = 0;
    DJ_CUDA_RUNTIME_CHECK(cudaGetDevice(&device_index));
    DJ_CUDA_RUNTIME_CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch, device_index));
    if (not supported)
        return false;

    constexpr int num_blocks = 2;
    int* values = nullptr;
    int* result = nullptr;
    try {
        DJ_CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&values), num_blocks * sizeof(int)));
        DJ_CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&result), 2 * sizeof(int)));
        DJ_CUDA_RUNTIME_CHECK(cudaMemset(values, 0, num_blocks * sizeof(int)));
        DJ_CUDA_RUNTIME_CHECK(cudaMemset(result, 0, 2 * sizeof(int)));
        runtime.launch(
            kernel, {
                .grid_dim = dim3(num_blocks, 1, 1),
                .block_dim = dim3(32, 1, 1),
                .cluster_dim = dim3(1, 1, 1),
                .cooperative = false,
                .enable_pdl = false,
                .nonportable_cluster_size_allowed = false,
            },
            values, result);
        DJ_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        std::array<int, 2> host_result{};
        DJ_CUDA_RUNTIME_CHECK(cudaMemcpy(host_result.data(), result, sizeof(host_result), cudaMemcpyDeviceToHost));
        const std::array<int, 2> non_cooperative_expected {0, 0};
        DJ_HOST_ASSERT(host_result == non_cooperative_expected,
                       "non-cooperative launch unexpectedly exposed a valid grid group");

        DJ_CUDA_RUNTIME_CHECK(cudaMemset(values, 0, num_blocks * sizeof(int)));
        DJ_CUDA_RUNTIME_CHECK(cudaMemset(result, 0, 2 * sizeof(int)));
        runtime.launch(
            kernel, {
                .grid_dim = dim3(num_blocks, 1, 1),
                .block_dim = dim3(32, 1, 1),
                .cluster_dim = dim3(1, 1, 1),
                .cooperative = true,
                .enable_pdl = false,
                .nonportable_cluster_size_allowed = false,
            },
            values, result);
        DJ_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

        DJ_CUDA_RUNTIME_CHECK(cudaMemcpy(host_result.data(), result, sizeof(host_result), cudaMemcpyDeviceToHost));
        DJ_HOST_ASSERT(host_result[0] == 1, "cooperative grid was not valid inside the kernel");
        DJ_HOST_ASSERT(host_result[1] == num_blocks * (num_blocks + 1) / 2,
                       "cooperative grid synchronization produced an incorrect sum");
        DJ_CUDA_RUNTIME_CHECK(cudaFree(result));
        result = nullptr;
        DJ_CUDA_RUNTIME_CHECK(cudaFree(values));
        values = nullptr;
    } catch (...) {
        if (result != nullptr)
            cudaFree(result);
        if (values != nullptr)
            cudaFree(values);
        throw;
    }
    return true;
}

bool test_cluster_launch(Runtime& runtime) {
    const auto kernel = runtime.compile("cluster_shared_memory", get_source("cluster_shared_memory.cu"));
    int device_index = 0;
    int supported = 0;
    DJ_CUDA_RUNTIME_CHECK(cudaGetDevice(&device_index));
    DJ_CUDA_RUNTIME_CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrClusterLaunch, device_index));
    if (runtime.device.get_arch_major() < 9 or not supported)
        return false;

    int* output = nullptr;
    try {
        DJ_CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&output), 4 * sizeof(int)));
        DJ_CUDA_RUNTIME_CHECK(cudaMemset(output, 0, 4 * sizeof(int)));
        runtime.launch(
            kernel, {
                .num_smem_bytes = sizeof(int),
                .grid_dim = dim3(2, 1, 1),
                .block_dim = dim3(32, 1, 1),
                .cluster_dim = dim3(1, 1, 1),
                .cooperative = false,
                .enable_pdl = false,
                .nonportable_cluster_size_allowed = false,
            },
            output);
        DJ_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        std::array<int, 4> result{};
        DJ_CUDA_RUNTIME_CHECK(cudaMemcpy(result.data(), output, sizeof(result), cudaMemcpyDeviceToHost));
        const std::array<int, 4> single_cta_expected {-1, -1, 1, 1};
        DJ_HOST_ASSERT(result == single_cta_expected,
                       "single-CTA clusters reported unexpected cluster semantics");

        DJ_CUDA_RUNTIME_CHECK(cudaMemset(output, 0, 4 * sizeof(int)));
        runtime.launch(
            kernel, {
                .num_smem_bytes = sizeof(int),
                .grid_dim = dim3(2, 1, 1),
                .block_dim = dim3(32, 1, 1),
                .cluster_dim = dim3(2, 1, 1),
                .cooperative = false,
                .enable_pdl = false,
                .nonportable_cluster_size_allowed = false,
            },
            output);
        DJ_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

        DJ_CUDA_RUNTIME_CHECK(cudaMemcpy(result.data(), output, sizeof(result), cudaMemcpyDeviceToHost));
        const std::array<int, 4> two_cta_expected {2, 1, 2, 2};
        DJ_HOST_ASSERT(result == two_cta_expected,
                       "cluster rank or distributed shared memory is incorrect");
        DJ_CUDA_RUNTIME_CHECK(cudaFree(output));
        output = nullptr;
    } catch (...) {
        if (output != nullptr)
            cudaFree(output);
        throw;
    }
    return true;
}

void test_multidimensional_launch(Runtime& runtime) {
    constexpr dim3 grid_dim(2, 3, 2);
    constexpr dim3 block_dim(4, 2, 2);
    constexpr int num_blocks = grid_dim.x * grid_dim.y * grid_dim.z;
    constexpr int threads_per_block = block_dim.x * block_dim.y * block_dim.z;
    constexpr int num_values = num_blocks * threads_per_block;
    const auto kernel = runtime.compile("multidimensional_launch", get_source("multidimensional_launch.cu"));

    int* output = nullptr;
    DJ_CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&output), num_values * sizeof(int)));
    try {
        runtime.launch(
            kernel, {
                .grid_dim = grid_dim,
                .block_dim = block_dim,
            },
            output);
        DJ_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

        std::array<int, num_values> result{};
        DJ_CUDA_RUNTIME_CHECK(cudaMemcpy(result.data(), output, sizeof(result), cudaMemcpyDeviceToHost));
        for (int block_index = 0; block_index < num_blocks; ++block_index) {
            for (int thread_index = 0; thread_index < threads_per_block; ++thread_index) {
                const auto index = block_index * threads_per_block + thread_index;
                DJ_HOST_ASSERT(result[index] == block_index * 1000 + thread_index,
                               "multidimensional launch produced an incorrect value at {}", index);
            }
        }

        DJ_CUDA_RUNTIME_CHECK(cudaFree(output));
    } catch (...) {
        cudaFree(output);
        throw;
    }
}

void test_mixed_kernel_arguments(Runtime& runtime) {
    struct MixedArgumentPayload {
        int first;
        unsigned int second;
        long long third;
    };

    const auto kernel = runtime.compile("mixed_arguments", get_source("mixed_arguments.cu"));
    long long* output = nullptr;
    int* input = nullptr;
    try {
        DJ_CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&output), sizeof(long long)));
        DJ_CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&input), sizeof(int)));
        const int input_value = 3;
        DJ_CUDA_RUNTIME_CHECK(cudaMemcpy(input, &input_value, sizeof(input_value), cudaMemcpyHostToDevice));
        const int* optional_input = nullptr;
        void* optional_pointer = nullptr;
        const signed char value_i8 = -2;
        const unsigned short value_u16 = 7;
        const int value_i32 = -11;
        const unsigned int value_u32 = 13;
        const long long value_i64 = -17;
        const unsigned long long value_u64 = 19;
        const float value_f32 = 2.5f;
        const double value_f64 = 3.25;
        const bool flag = true;
        const MixedArgumentPayload direct_payload {29, 31, 37};
        MixedArgumentPayload indirect_payload {41, 43, 47};
        const deep_jit::NoRefPtr indirect_storage {&indirect_payload};

        runtime.launch(
            kernel, {
                .grid_dim = dim3(1, 1, 1),
                .block_dim = dim3(1, 1, 1),
            },
            output,
            input,
            optional_input,
            value_i8,
            value_u16,
            value_i32,
            value_u32,
            value_i64,
            value_u64,
            value_f32,
            value_f64,
            flag,
            direct_payload,
            indirect_storage,
            optional_pointer);
        DJ_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

        long long result = 0;
        DJ_CUDA_RUNTIME_CHECK(cudaMemcpy(&result, output, sizeof(result), cudaMemcpyDeviceToHost));
        DJ_HOST_ASSERT(result == 339, "mixed kernel arguments produced an incorrect value: {}", result);

        DJ_CUDA_RUNTIME_CHECK(cudaFree(input));
        input = nullptr;
        DJ_CUDA_RUNTIME_CHECK(cudaFree(output));
        output = nullptr;
    } catch (...) {
        if (input != nullptr)
            cudaFree(input);
        if (output != nullptr)
            cudaFree(output);
        throw;
    }
}

void test_large_kernel_arguments(Runtime& runtime) {
    struct LargeArgumentPayload {
        unsigned long long values[128];
    };
    struct IndirectArgumentPayload {
        int values[8];
    };
    static_assert(sizeof(LargeArgumentPayload) == 1024);

    const auto make_payload = [] {
        LargeArgumentPayload payload{};
        for (int i = 0; i < 128; ++i)
            payload.values[i] = static_cast<unsigned long long>(i * 17 + 3);
        return payload;
    };
    std::array<CUtensorMap, 18> tensor_maps{};
    for (std::size_t map_index = 0; map_index < tensor_maps.size(); ++map_index) {
        auto bytes = reinterpret_cast<unsigned char*>(&tensor_maps[map_index]);
        for (std::size_t byte_index = 0; byte_index < sizeof(CUtensorMap); ++byte_index)
            bytes[byte_index] = static_cast<unsigned char>(map_index * 19 + byte_index * 7);
    }
    IndirectArgumentPayload indirect_payload {{2, 3, 5, 7, 11, 13, 17, 19}};

    unsigned long long expected = 1469598103934665603ull;
    const auto payload_for_checksum = make_payload();
    for (const auto value : payload_for_checksum.values)
        expected = (expected ^ value) * 1099511628211ull;
    for (const auto& tensor_map : tensor_maps) {
        const auto bytes = reinterpret_cast<const unsigned char*>(&tensor_map);
        for (std::size_t i = 0; i < sizeof(CUtensorMap); ++i)
            expected = (expected ^ bytes[i]) * 1099511628211ull;
    }
    for (const auto value : indirect_payload.values)
        expected = (expected ^ static_cast<unsigned int>(value)) * 1099511628211ull;

    const auto kernel = runtime.compile("large_kernel_arguments", get_source("large_kernel_arguments.cu"));
    unsigned long long* output = nullptr;
    DJ_CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&output), sizeof(unsigned long long)));
    try {
        const deep_jit::NoRefPtr indirect_storage {&indirect_payload};
        void* optional_pointer = nullptr;
        runtime.launch(
            kernel, {
                .grid_dim = dim3(1, 1, 1),
                .block_dim = dim3(1024, 1, 1),
            },
            output,
            make_payload(),
            tensor_maps[0], tensor_maps[1], tensor_maps[2], tensor_maps[3], tensor_maps[4], tensor_maps[5],
            tensor_maps[6], tensor_maps[7], tensor_maps[8], tensor_maps[9], tensor_maps[10], tensor_maps[11],
            tensor_maps[12], tensor_maps[13], tensor_maps[14], tensor_maps[15], tensor_maps[16], tensor_maps[17],
            indirect_storage,
            optional_pointer);
        DJ_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

        unsigned long long result = 0;
        DJ_CUDA_RUNTIME_CHECK(cudaMemcpy(&result, output, sizeof(result), cudaMemcpyDeviceToHost));
        DJ_HOST_ASSERT(result == expected, "large kernel arguments produced an incorrect checksum");
        DJ_CUDA_RUNTIME_CHECK(cudaFree(output));
        output = nullptr;
    } catch (...) {
        if (output != nullptr)
            cudaFree(output);
        throw;
    }
}

void test_macro_selected_abi(Runtime& runtime) {
    const auto source = get_source("macro_selected_abi.cu");
    const CompilerOptions options_32 {.extra_nvcc_flags = {"-DTEST_INDEX_BITS=32"}};
    const CompilerOptions options_64 {.extra_nvcc_flags = {"-DTEST_INDEX_BITS=64"}};
    const auto kernel_32 = runtime.compile("macro_selected_abi", source, options_32);
    const auto kernel_64 = runtime.compile("macro_selected_abi", source, options_64);
    DJ_HOST_ASSERT(kernel_32 != kernel_64, "compiler macro did not select a distinct kernel ABI");

    long long* output = nullptr;
    DJ_CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&output), sizeof(long long)));
    try {
        const int value_32 = 17;
        runtime.launch(
            kernel_32, {
                .grid_dim = dim3(1, 1, 1),
                .block_dim = dim3(1, 1, 1),
            },
            output, value_32);
        DJ_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        long long result = 0;
        DJ_CUDA_RUNTIME_CHECK(cudaMemcpy(&result, output, sizeof(result), cudaMemcpyDeviceToHost));
        DJ_HOST_ASSERT(result == 4017, "32-bit macro-selected ABI produced {}", result);

        const long long value_64 = 23;
        runtime.launch(
            kernel_64, {
                .grid_dim = dim3(1, 1, 1),
                .block_dim = dim3(1, 1, 1),
            },
            output, value_64);
        DJ_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        DJ_CUDA_RUNTIME_CHECK(cudaMemcpy(&result, output, sizeof(result), cudaMemcpyDeviceToHost));
        DJ_HOST_ASSERT(result == 8023, "64-bit macro-selected ABI produced {}", result);

        DJ_CUDA_RUNTIME_CHECK(cudaFree(output));
        output = nullptr;
    } catch (...) {
        if (output != nullptr)
            cudaFree(output);
        throw;
    }
}

void test_large_dynamic_shared_memory(Runtime& runtime) {
    const int num_smem_bytes = runtime.device.get_num_smem_bytes();
    const auto kernel = runtime.compile("large_dynamic_shared_memory", get_source("large_dynamic_shared_memory.cu"));

    int* output = nullptr;
    DJ_CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&output), sizeof(int)));
    try {
        runtime.launch(
            kernel, {
                .num_smem_bytes = num_smem_bytes,
                .grid_dim = dim3(1, 1, 1),
                .block_dim = dim3(32, 1, 1),
            },
            output, 73, num_smem_bytes / static_cast<int>(sizeof(int)));
        DJ_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        int result = 0;
        DJ_CUDA_RUNTIME_CHECK(cudaMemcpy(&result, output, sizeof(int), cudaMemcpyDeviceToHost));
        DJ_HOST_ASSERT(result == 73, "large dynamic shared-memory launch returned {}", result);
        DJ_CUDA_RUNTIME_CHECK(cudaFree(output));
    } catch (...) {
        cudaFree(output);
        throw;
    }
}

void test_generated_float_literals(const fs::path& cache_root) {
    const auto include_dir = get_test_cuda_project_dir() / "include_original";
    const auto float_cache = cache_root / "generated_float_literals";
    set_env("FLOAT_LITERAL_JIT_CACHE_DIR", float_cache.string());
    const auto runtime = make_runtime_with_prefix("FLOAT_LITERAL", include_dir);

    const std::array<float, 4> values = {10.0f, -10.0f, -0.0f, 0.125f};
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto source = get_float_template_source(values[index]);
        const auto kernel = runtime->compile(std::format("generated_float_literal_{}", index), source);
        float* output = nullptr;
        DJ_CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&output), sizeof(float)));
        try {
            runtime->launch(
                kernel, {
                    .grid_dim = dim3(1, 1, 1),
                    .block_dim = dim3(1, 1, 1),
                },
                output);
            DJ_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
            float result = 0;
            DJ_CUDA_RUNTIME_CHECK(cudaMemcpy(&result, output, sizeof(float), cudaMemcpyDeviceToHost));
            DJ_HOST_ASSERT(std::bit_cast<uint32_t>(result) == std::bit_cast<uint32_t>(values[index]),
                           "generated float literal changed value at index {}", index);
            DJ_CUDA_RUNTIME_CHECK(cudaFree(output));
        } catch (...) {
            cudaFree(output);
            throw;
        }
    }
    unset_env("FLOAT_LITERAL_JIT_CACHE_DIR");
}

void test_template_hash_and_launch(Runtime& runtime) {
    const auto source_1 = get_template_source(1);
    const auto source_2 = get_template_source(2);
    DJ_HOST_ASSERT(count_different_bytes(source_1, source_2) == 1,
                   "template source fixture must differ by exactly one byte");
    const auto initial_memory_cache_size = runtime.mem_cache.cache.size();
    const auto artifact_1 = runtime.compile_without_load("template_add", source_1);
    check_artifact(artifact_1, source_1);
    DJ_HOST_ASSERT(runtime.mem_cache.cache.size() == initial_memory_cache_size,
                   "compile_without_load populated the memory cache");
    DJ_HOST_ASSERT(runtime.compile_without_load("template_add", source_1) == artifact_1,
                   "identical template source must reuse the cache entry");

    const auto kernel_1 = runtime.compile("template_add", source_1);
    DJ_HOST_ASSERT(runtime.mem_cache.cache.size() == initial_memory_cache_size + 1,
                   "compile did not populate the memory cache");
    expect_failure(
        [&] { runtime.launch(std::shared_ptr<deep_jit::cuda::Kernel>{}, LaunchOptions {}); },
        "kernel must not be null");
    expect_failure(
        [&] { runtime.launch(kernel_1, {}, static_cast<int*>(nullptr), 0); },
        "CUDA grid dimension must be specified");
    expect_failure(
        [&] {
            runtime.launch(
                kernel_1, {
                    .grid_dim = dim3(1, 1, 1),
                },
                static_cast<int*>(nullptr), 0);
        },
        "CUDA block dimension must be specified");
    expect_failure(
        [&] {
            runtime.launch(
                kernel_1, {
                    .grid_dim = dim3(1, 1, 1),
                    .block_dim = dim3(1, 1, 1),
                    .cluster_dim = dim3(1, 2, 1),
                },
                static_cast<int*>(nullptr), 0);
        },
        "only one-dimensional CUDA clusters are supported");
    expect_failure(
        [&] {
            runtime.launch(
                kernel_1, {
                    .num_smem_bytes = -1,
                    .grid_dim = dim3(1, 1, 1),
                    .block_dim = dim3(1, 1, 1),
                },
                static_cast<int*>(nullptr), 0);
        },
        "shared-memory size must not be negative");
    expect_failure(
        [&] {
            runtime.launch(
                kernel_1, {
                    .grid_dim = dim3(0, 1, 1),
                    .block_dim = dim3(1, 1, 1),
                },
                static_cast<int*>(nullptr), 0);
        },
        "grid dimensions must be positive");
    expect_failure(
        [&] {
            runtime.launch(
                kernel_1, {
                    .grid_dim = dim3(1, 1, 1),
                    .block_dim = dim3(0, 1, 1),
                },
                static_cast<int*>(nullptr), 0);
        },
        "block dimensions must be positive");
    expect_failure(
        [&] {
            runtime.launch(
                kernel_1, {
                    .grid_dim = dim3(1, 1, 1),
                    .block_dim = dim3(1, 1, 1),
                    .cluster_dim = dim3(0, 1, 1),
                },
                static_cast<int*>(nullptr), 0);
        },
        "cluster dimension must be positive");

    auto saved_defaults = runtime.default_launch_options;
    runtime.default_launch_options.num_smem_bytes.reset();
    expect_failure(
        [&] {
            runtime.launch(
                kernel_1, {
                    .grid_dim = dim3(1, 1, 1),
                    .block_dim = dim3(1, 1, 1),
                },
                static_cast<int*>(nullptr), 0);
        },
        "shared-memory size must be specified");

    runtime.default_launch_options = saved_defaults;
    runtime.default_launch_options.cluster_dim.reset();
    expect_failure(
        [&] {
            runtime.launch(
                kernel_1, {
                    .grid_dim = dim3(1, 1, 1),
                    .block_dim = dim3(1, 1, 1),
                },
                static_cast<int*>(nullptr), 0);
        },
        "cluster dimension must be specified");

    runtime.default_launch_options = saved_defaults;
    runtime.default_launch_options.cooperative.reset();
    expect_failure(
        [&] {
            runtime.launch(
                kernel_1, {
                    .grid_dim = dim3(1, 1, 1),
                    .block_dim = dim3(1, 1, 1),
                },
                static_cast<int*>(nullptr), 0);
        },
        "cooperative option must be specified");

    runtime.default_launch_options = saved_defaults;
    runtime.default_launch_options.enable_pdl.reset();
    expect_failure(
        [&] {
            runtime.launch(
                kernel_1, {
                    .grid_dim = dim3(1, 1, 1),
                    .block_dim = dim3(1, 1, 1),
                },
                static_cast<int*>(nullptr), 0);
        },
        "PDL option must be specified");

    runtime.default_launch_options = saved_defaults;
    runtime.default_launch_options.nonportable_cluster_size_allowed.reset();
    expect_failure(
        [&] {
            runtime.launch(
                kernel_1, {
                    .grid_dim = dim3(1, 1, 1),
                    .block_dim = dim3(1, 1, 1),
                },
                static_cast<int*>(nullptr), 0);
        },
        "non-portable cluster option must be specified");
    runtime.default_launch_options = saved_defaults;

    expect_any_failure([&] {
        runtime.launch(
            kernel_1, {
                .num_smem_bytes = runtime.device.get_num_smem_bytes() + 1,
                .grid_dim = dim3(1, 1, 1),
                .block_dim = dim3(1, 1, 1),
            },
            static_cast<int*>(nullptr), 0);
    });
    expect_any_failure([&] {
        runtime.launch(
            kernel_1, {
                .grid_dim = dim3(1, 1, 1),
                .block_dim = dim3(runtime.device.get_prop().maxThreadsPerBlock + 1, 1, 1),
            },
            static_cast<int*>(nullptr), 0);
    });

    const auto cache_size = runtime.mem_cache.cache.size();
    DJ_HOST_ASSERT(launch_value(runtime, kernel_1, 7) == 8, "unexpected first launch result");
    DJ_HOST_ASSERT(launch_value(runtime, kernel_1, 41) == 42, "runtime argument changed compiled behavior");
    DJ_HOST_ASSERT(runtime.mem_cache.cache.size() == cache_size, "runtime arguments must not change the cache key");
    DJ_HOST_ASSERT(runtime.compile("template_add", source_1) == kernel_1, "memory cache did not reuse the kernel");

    constexpr int cache_hit_iterations = 10000;
    const auto cache_hit_begin = std::chrono::steady_clock::now();
    for (int i = 0; i < cache_hit_iterations; ++i)
        DJ_HOST_ASSERT(runtime.compile("template_add", source_1) == kernel_1);
    const auto cache_hit_elapsed = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - cache_hit_begin).count();
    std::printf("JIT memory-cache hit overhead: %.3f us\n", cache_hit_elapsed / cache_hit_iterations);

    const auto artifact_2 = runtime.compile_without_load("template_add", source_2);
    check_artifact(artifact_2, source_2);
    DJ_HOST_ASSERT(artifact_1 != artifact_2, "changing a template argument must change the cache key");
    DJ_HOST_ASSERT(launch_value(runtime, runtime.compile("template_add", source_2), 7) == 9,
                   "changed template argument was not compiled");
}

void test_secondary_disk_cache(const fs::path& cache_root) {
    const auto primary_cache = cache_root / "secondary_lookup_primary";
    const auto secondary_cache = cache_root / "secondary_lookup_secondary";
    const auto include_dir = get_test_cuda_project_dir() / "include_original";
    const auto source = get_template_source(1);

    set_env("SECONDARY_SOURCE_JIT_CACHE_DIR", secondary_cache.string());
    const auto source_runtime = make_runtime_with_prefix("SECONDARY_SOURCE", include_dir);
    const auto expected_artifact = source_runtime->compile_without_load("template_add", source);
    unset_env("SECONDARY_SOURCE_JIT_CACHE_DIR");

    const auto commit_path = expected_artifact / deep_jit::kCommitFileName;
    DJ_HOST_ASSERT(fs::remove(commit_path));
    fs::create_symlink("/proc/version", commit_path);
    DJ_HOST_ASSERT(not deep_jit::try_update_mtime(commit_path),
                   "read-only secondary cache marker unexpectedly allowed an mtime update");

    set_env("SECONDARY_LOOKUP_JIT_CACHE_DIR", primary_cache.string() + ":" + secondary_cache.string());
    const auto runtime = make_runtime_with_prefix(
        "SECONDARY_LOOKUP", include_dir);
    const auto artifact = runtime->compile_without_load("template_add", source);
    DJ_HOST_ASSERT(artifact == expected_artifact, "runtime did not reuse the secondary cache entry");
    DJ_HOST_ASSERT(launch_value(*runtime, runtime->compile("template_add", source), 4) == 5,
                   "kernel loaded from the secondary cache root did not launch");
    DJ_HOST_ASSERT(not fs::exists(primary_cache / "cache"),
                   "secondary cache hit unexpectedly wrote to the primary cache root");
    unset_env("SECONDARY_LOOKUP_JIT_CACHE_DIR");
}

void test_compiler_options(Runtime& runtime) {
    const auto source = get_source("compiler_options.cu");
    const deep_jit::CUDA::CompilerInfo compiler_a {"/toolkit/a/nvcc", "same-version"};
    const deep_jit::CUDA::CompilerInfo compiler_b {"/toolkit/b/nvcc", "same-version"};
    const deep_jit::CUDA::CompilerInfo compiler_c {"/toolkit/a/nvcc", "different-version"};
    const deep_jit::CUDA::CompilerInfo compiler_binary_a {
        "/toolkit/a/nvcc", std::string("same\0A", 6)};
    const deep_jit::CUDA::CompilerInfo compiler_binary_b {
        "/toolkit/a/nvcc", std::string("same\0B", 6)};
    const deep_jit::CUDA::CompilerInfo compiler_binary_prefix_a {
        "/toolkit/a/nvcc", std::string("A\0same", 6)};
    const deep_jit::CUDA::CompilerInfo compiler_binary_prefix_b {
        "/toolkit/a/nvcc", std::string("B\0same", 6)};
    DJ_HOST_ASSERT(compiler_a.get_hash() == compiler_b.get_hash(), "compiler path must not affect the cache key");
    DJ_HOST_ASSERT(compiler_a.get_hash() != compiler_c.get_hash(), "compiler version must affect the cache key");
    DJ_HOST_ASSERT(compiler_binary_a.get_hash() != compiler_binary_b.get_hash(),
                   "compiler-info hash ignored bytes after a null");
    DJ_HOST_ASSERT(compiler_binary_prefix_a.get_hash() != compiler_binary_prefix_b.get_hash(),
                   "compiler-info hash ignored bytes before a null");

    const CompilerOptions overrides {
        .optimize_level = "0",
        .fast_math = true,
        .with_line_info = true,
        .extra_nvcc_flags = {"-DTEST_OPTION=17"},
    };
    const auto options = runtime.default_compiler_options.override_with(overrides);
    const auto default_key = runtime.cache_key(source, runtime.default_compiler_options);
    const auto options_key = runtime.cache_key(source, options);
    DJ_HOST_ASSERT(default_key != options_key, "compiler options must change the cache key");

    auto dump_options = runtime.default_compiler_options;
    dump_options.dump_ptx = true;
    dump_options.dump_sass = true;
    DJ_HOST_ASSERT(runtime.cache_key(source, dump_options) == default_key,
                   "dump-only options must not change the cache key");

    auto include_flag_a = runtime.default_compiler_options;
    auto include_flag_b = runtime.default_compiler_options;
    include_flag_a.extra_nvcc_flags = {"-I/opt/nccl-a/include"};
    include_flag_b.extra_nvcc_flags = {"-I/opt/nccl-b/include"};
    DJ_HOST_ASSERT(runtime.cache_key(source, include_flag_a) != runtime.cache_key(source, include_flag_b),
                   "untracked dependency include paths in compiler flags must change the cache key");

    auto ordered_flags = runtime.default_compiler_options;
    auto reversed_flags = runtime.default_compiler_options;
    ordered_flags.extra_nvcc_flags = {"-DFIRST=1", "-DSECOND=2"};
    reversed_flags.extra_nvcc_flags = {"-DSECOND=2", "-DFIRST=1"};
    DJ_HOST_ASSERT(runtime.cache_key(source, ordered_flags) != runtime.cache_key(source, reversed_flags),
                   "compiler flag order must affect the cache key");

    const auto artifact = runtime.compile_without_load("compiler_options", source, overrides);
    check_artifact(artifact, source);
    check_metadata(artifact, "test-signature", "-DTEST_OPTION=17");
    DJ_HOST_ASSERT(launch_value(runtime, runtime.compile("compiler_options", source, overrides)) == 17,
                   "compiler option was not applied");
}

void test_cache_key_option_matrix(Runtime& runtime) {
    const auto source = get_source("compiler_options.cu");
    const auto default_key = runtime.cache_key(source, runtime.default_compiler_options);
    const auto expected_base = deep_jit::hash::FNV1a()
        .update(runtime.config.extra_signature)
        .update(runtime.backend.compiler_info.get_hash())
        .get_hex_digest();
    DJ_HOST_ASSERT(runtime.hash_base.get_hex_digest() == expected_base,
                   "runtime base hash does not contain signature and compiler info in order");
    auto expected_hash = runtime.hash_base;
    expected_hash.update(deep_jit::str::join(runtime.default_compiler_options.get_flags()));
    expected_hash.update(runtime.default_compiler_options.get_post_hook_hash(runtime.config));
    expected_hash.update(runtime.parser.parse_into_hash(source));
    DJ_HOST_ASSERT(default_key == expected_hash.get_hex_digest(), "cache key components were combined in the wrong order");

    const std::string binary_source_a = source + std::string("\n// binary\0A", 12);
    const std::string binary_source_b = source + std::string("\n// binary\0B", 12);
    DJ_HOST_ASSERT(runtime.cache_key(binary_source_a, runtime.default_compiler_options) !=
                       runtime.cache_key(binary_source_b, runtime.default_compiler_options),
                   "kernel cache key ignored source bytes after a null");
    const auto expect_changed = [&](const CompilerOptions& overrides, const std::string_view name) {
        const auto options = runtime.default_compiler_options.override_with(overrides);
        DJ_HOST_ASSERT(runtime.cache_key(source, options) != default_key,
                       "compiler option did not affect cache key: {}", name);
    };

    expect_changed(CompilerOptions {.optimize_level = "0"}, "optimize_level");
    expect_changed(CompilerOptions {.fast_math = true}, "fast_math");
    expect_changed(CompilerOptions {.ptxas_verbose = true}, "ptxas_verbose");
    expect_changed(CompilerOptions {.ptxas_register_usage_level = 0}, "ptxas_register_usage_level");
    expect_changed(CompilerOptions {.check_no_spills = true}, "check_no_spills");
    expect_changed(CompilerOptions {.check_no_local_memory = true}, "check_no_local_memory");
    expect_changed(CompilerOptions {.with_line_info = true}, "with_line_info");
    expect_changed(CompilerOptions {.arch = "90"}, "arch");
    expect_changed(CompilerOptions {.nvcc_flags = std::vector<std::string>{"-std=c++20"}}, "nvcc_flags");
    expect_changed(CompilerOptions {.extra_nvcc_flags = {"-DKEY_OPTION=1"}}, "extra_nvcc_flags");

    auto binary_flag_a = runtime.default_compiler_options;
    auto binary_flag_b = runtime.default_compiler_options;
    binary_flag_a.extra_nvcc_flags = {std::string("-DKEY=\0A", 8)};
    binary_flag_b.extra_nvcc_flags = {std::string("-DKEY=\0B", 8)};
    DJ_HOST_ASSERT(runtime.cache_key(source, binary_flag_a) != runtime.cache_key(source, binary_flag_b),
                   "compiler-option hash ignored bytes after a null");
    binary_flag_a.extra_nvcc_flags = {std::string("A\0-DKEY", 7)};
    binary_flag_b.extra_nvcc_flags = {std::string("B\0-DKEY", 7)};
    DJ_HOST_ASSERT(runtime.cache_key(source, binary_flag_a) != runtime.cache_key(source, binary_flag_b),
                   "compiler-option hash ignored bytes before a null");

    auto dump_ptx_options = runtime.default_compiler_options;
    dump_ptx_options.dump_ptx = true;
    DJ_HOST_ASSERT(runtime.cache_key(source, dump_ptx_options) == default_key,
                   "PTX dump option must not affect the cache key");
    auto dump_sass_options = runtime.default_compiler_options;
    dump_sass_options.dump_sass = true;
    DJ_HOST_ASSERT(runtime.cache_key(source, dump_sass_options) == default_key,
                   "SASS dump option must not affect the cache key");

    std::unordered_set<std::string> randomized_keys;
    randomized_keys.reserve(5000);
    std::mt19937_64 generator(0x741bc92du);
    for (int index = 0; index < 5000; ++index) {
        const auto randomized_source = source + std::format("\n// {} {}\n", index, generator());
        DJ_HOST_ASSERT(randomized_keys.emplace(
                           runtime.cache_key(randomized_source, runtime.default_compiler_options)).second,
                       "randomized cache key collision at input {}", index);
    }
}

void test_include_dirs(const fs::path& cache_root) {
    const auto source = get_source("tracked_include.cu");
    const auto include_original = get_test_cuda_project_dir() / "include_original";
    const auto include_same_content = get_test_cuda_project_dir() / "include_same_content";
    const auto include_changed_content = get_test_cuda_project_dir() / "include_changed_content";
    const auto runtime_original = make_runtime(include_original);
    const auto runtime_same_content = make_runtime(include_same_content);
    const auto runtime_changed_content = make_runtime(include_changed_content);
    DJ_HOST_ASSERT(count_different_bytes(
                       deep_jit::read(include_original / "test_cuda/detail/tracked_constant.cuh"),
                       deep_jit::read(include_changed_content / "test_cuda/detail/tracked_constant.cuh")) == 1,
                   "tracked include fixture must differ by exactly one byte");

    const auto artifact_original = runtime_original->compile_without_load("include_add", source);
    check_artifact(artifact_original, source);
    DJ_HOST_ASSERT(launch_value(*runtime_original, runtime_original->compile("include_add", source), 1) == 12,
                   "tracked include file was not compiled");
    const auto artifact_same_content = runtime_same_content->compile_without_load("include_add", source);
    DJ_HOST_ASSERT(artifact_original == artifact_same_content, "include directory paths must not affect the cache key");
    DJ_HOST_ASSERT(launch_value(*runtime_same_content, runtime_same_content->compile("include_add", source), 1) == 12,
                   "equivalent include directory did not reuse a valid kernel");

    const auto artifact_changed_content = runtime_changed_content->compile_without_load("include_add", source);
    check_artifact(artifact_changed_content, source);
    DJ_HOST_ASSERT(artifact_original != artifact_changed_content, "changed include contents must change the cache key");
    DJ_HOST_ASSERT(launch_value(*runtime_changed_content, runtime_changed_content->compile("include_add", source), 1) == 13,
                   "changed include file was not compiled");

    const auto runtime_original_first = std::make_shared<Runtime>(deep_jit::Config(
        get_test_cuda_project_dir(), "TEST", "test-signature",
        std::vector<fs::path>{include_original, include_changed_content},
        std::vector<std::string>{"test_cuda/"}));
    const auto runtime_changed_first = std::make_shared<Runtime>(deep_jit::Config(
        get_test_cuda_project_dir(), "TEST", "test-signature",
        std::vector<fs::path>{include_changed_content, include_original},
        std::vector<std::string>{"test_cuda/"}));
    DJ_HOST_ASSERT(launch_value(*runtime_original_first, runtime_original_first->compile("include_add", source), 1) == 12,
                   "compiler did not use the first matching include directory");
    DJ_HOST_ASSERT(launch_value(*runtime_changed_first, runtime_changed_first->compile("include_add", source), 1) == 13,
                   "compiler and parser include-directory order diverged");

    const auto signature_runtime = make_runtime(include_original, "different-signature");
    DJ_HOST_ASSERT(signature_runtime->cache_key(source, signature_runtime->default_compiler_options) !=
                       runtime_original->cache_key(source, runtime_original->default_compiler_options),
                   "extra signature must change the cache key");
    const auto signature_suffix_a = make_runtime(include_original, std::string("signature\0A", 11));
    const auto signature_suffix_b = make_runtime(include_original, std::string("signature\0B", 11));
    DJ_HOST_ASSERT(signature_suffix_a->cache_key(source, signature_suffix_a->default_compiler_options) !=
                       signature_suffix_b->cache_key(source, signature_suffix_b->default_compiler_options),
                   "extra-signature hash ignored bytes after a null");
    const auto signature_prefix_a = make_runtime(include_original, std::string("A\0signature", 11));
    const auto signature_prefix_b = make_runtime(include_original, std::string("B\0signature", 11));
    DJ_HOST_ASSERT(signature_prefix_a->cache_key(source, signature_prefix_a->default_compiler_options) !=
                       signature_prefix_b->cache_key(source, signature_prefix_b->default_compiler_options),
                   "extra-signature hash ignored bytes before a null");
    const auto untracked_runtime = std::make_shared<Runtime>(deep_jit::Config(
        get_test_cuda_project_dir(), "TEST", "test-signature", {include_original}, {"another_library/"}));
    DJ_HOST_ASSERT(untracked_runtime->cache_key(source, untracked_runtime->default_compiler_options) !=
                       runtime_original->cache_key(source, runtime_original->default_compiler_options),
                   "include prefixes must determine whether dependency contents enter the cache key");
    check_tmp_is_empty(cache_root);
}

void test_relocated_wheel_include_dir(const fs::path& cache_root) {
    const auto root = get_test_cuda_project_dir();
    const auto wheel_cache = cache_root / "wheel_include_relocation";
    const auto environment_a_include = cache_root / "environment_a/site-packages/deep_jit/include";
    const auto environment_b_include = cache_root / "environment_b/site-packages/deep_jit/include";
    const auto header = fs::path("deep_jit/wheel_marker.cuh");
    deep_jit::make_dirs((environment_a_include / header).parent_path());
    deep_jit::make_dirs((environment_b_include / header).parent_path());
    deep_jit::write_file_sync(environment_a_include / header, "#pragma once\n");
    deep_jit::write_file_sync(environment_b_include / header, "#pragma once\n");

    const auto source = "#include <deep_jit/wheel_marker.cuh>\n" + get_template_source(73);
    const auto make_environment_runtime = [&](const fs::path& include_dir) {
        return std::make_shared<Runtime>(deep_jit::Config(
            root, "WHEEL_INCLUDE", "test-signature",
            std::vector<fs::path>{include_dir},
            std::vector<std::string>{"deep_jit/"}));
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
                   "relocating an identical wheel include directory changed the CUDA cache key");
    environment_b->backend.toolkit.nvcc = cache_root / "compiler_must_not_run";
    const auto second_artifact = environment_b->compile_without_load("wheel_include_relocation", source);
    DJ_HOST_ASSERT(second_artifact == first_artifact,
                   "relocating an identical wheel include directory triggered CUDA compilation");
    DJ_HOST_ASSERT(deep_jit::read(second_artifact / "meta.json") == first_metadata,
                   "a CUDA cache hit rewrote metadata after the wheel include directory moved");
    check_tmp_is_empty(wheel_cache);
    unset_env("WHEEL_INCLUDE_JIT_CACHE_DIR");
}

void test_post_hook(Runtime& runtime, const fs::path& cache_root) {
    const auto source = get_template_source(3);
    DJ_HOST_ASSERT(runtime.default_compiler_options.get_post_hook_hash(runtime.config).empty());
    const auto without_hook = runtime.compile_without_load("post_hook", source);
    check_artifact(without_hook, source);
    const CompilerOptions options {
        .post_hook = "scripts/post_hook.py",
    };
    const auto expected_hook_hash = deep_jit::hash::FNV1a()
        .update("scripts/post_hook.py")
        .update(deep_jit::read(runtime.config.get_python_path("scripts/post_hook.py")))
        .get_hex_digest();
    const auto hook_hash = options.get_post_hook_hash(runtime.config);
    DJ_HOST_ASSERT(hook_hash == expected_hook_hash and options.get_post_hook_hash(runtime.config) == hook_hash,
                   "post hook hash was not stable");
    const auto artifact = runtime.compile_without_load("post_hook", source, options);
    check_artifact(artifact, source);
    DJ_HOST_ASSERT(artifact != without_hook, "post hook must change the cache key");
    const auto original_cubin = deep_jit::read(without_hook / "kernel.cubin");
    const auto hooked_cubin = deep_jit::read(artifact / "kernel.cubin");
    DJ_HOST_ASSERT(not original_cubin.starts_with("HOOK"), "unmodified CUBIN has the hook marker");
    DJ_HOST_ASSERT(hooked_cubin.starts_with("HOOK"), "post hook did not modify the first 32 bits");
    DJ_HOST_ASSERT(hooked_cubin.size() == original_cubin.size(), "post hook changed the CUBIN size");
    DJ_HOST_ASSERT(hooked_cubin != original_cubin, "post hook did not change the CUBIN contents");
    check_metadata(artifact, "test-signature", "\"post_hook\":\"scripts/post_hook.py\"");
    const auto memory_cache_size = runtime.mem_cache.cache.size();
    expect_any_failure([&] { (void)runtime.compile("post_hook", source, options); });
    DJ_HOST_ASSERT(runtime.mem_cache.cache.size() == memory_cache_size,
                   "failed CUBIN load populated the memory cache");

    const CompilerOptions failing_options {
        .post_hook = "scripts/failing_post_hook.py",
    };
    DJ_HOST_ASSERT(failing_options.get_post_hook_hash(runtime.config) != hook_hash,
                   "different post hooks must have different hashes");
    const auto effective_failing_options = runtime.default_compiler_options.override_with(failing_options);
    const auto failing_artifact = runtime.disk_cache.paths.front() / "cache" /
        std::format("failing_post_hook.{}", runtime.cache_key(source, effective_failing_options));

    const auto hook_root_a = cache_root / "hook_root_a";
    const auto hook_root_b = cache_root / "hook_root_b";
    const auto hook_root_changed = cache_root / "hook_root_changed";
    for (const auto& root : {hook_root_a, hook_root_b, hook_root_changed})
        deep_jit::make_dirs(root / "hooks");
    deep_jit::write_file_sync(hook_root_a / "hooks/shared.py", "print('same')\n");
    deep_jit::write_file_sync(hook_root_a / "hooks/name_a.py", "print('same')\n");
    deep_jit::write_file_sync(hook_root_a / "hooks/name_b.py", "print('same')\n");
    deep_jit::write_file_sync(hook_root_b / "hooks/shared.py", "print('same')\n");
    deep_jit::write_file_sync(hook_root_changed / "hooks/shared.py", "print('changed')\n");
    const CompilerOptions shared_hook {.post_hook = "hooks/shared.py"};
    const deep_jit::Config hook_config_a(hook_root_a, "HOOK_A");
    const deep_jit::Config hook_config_b(hook_root_b, "HOOK_B");
    const deep_jit::Config hook_config_changed(hook_root_changed, "HOOK_CHANGED");
    DJ_HOST_ASSERT(shared_hook.get_post_hook_hash(hook_config_a) == shared_hook.get_post_hook_hash(hook_config_b),
                   "equal post hook contents must be installation-path independent");
    DJ_HOST_ASSERT(shared_hook.get_post_hook_hash(hook_config_a) != shared_hook.get_post_hook_hash(hook_config_changed),
                   "changed post hook contents must change the hash across runtimes");
    const CompilerOptions named_hook_a {.post_hook = "hooks/name_a.py"};
    const CompilerOptions named_hook_b {.post_hook = "hooks/name_b.py"};
    DJ_HOST_ASSERT(count_different_bytes(*named_hook_a.post_hook, *named_hook_b.post_hook) == 1);
    DJ_HOST_ASSERT(named_hook_a.get_post_hook_hash(hook_config_a) != named_hook_b.get_post_hook_hash(hook_config_a),
                   "a one-byte post hook name change did not affect the hash");

    deep_jit::write_file_sync(hook_root_a / "hooks/binary.py", std::string("hook\0A", 6));
    deep_jit::write_file_sync(hook_root_b / "hooks/binary.py", std::string("hook\0B", 6));
    deep_jit::write_file_sync(hook_root_a / "hooks/binary_prefix.py", std::string("A\0hook", 6));
    deep_jit::write_file_sync(hook_root_b / "hooks/binary_prefix.py", std::string("B\0hook", 6));
    const CompilerOptions binary_hook {.post_hook = "hooks/binary.py"};
    DJ_HOST_ASSERT(binary_hook.get_post_hook_hash(hook_config_a) != binary_hook.get_post_hook_hash(hook_config_b),
                   "post-hook hash ignored bytes after a null");
    const CompilerOptions binary_prefix_hook {.post_hook = "hooks/binary_prefix.py"};
    DJ_HOST_ASSERT(binary_prefix_hook.get_post_hook_hash(hook_config_a) !=
                       binary_prefix_hook.get_post_hook_hash(hook_config_b),
                   "post-hook hash ignored bytes before a null");

    const auto root_hash = shared_hook.get_post_hook_hash(hook_config_a);
    DJ_HOST_ASSERT(root_hash == deep_jit::hash::FNV1a()
                                    .update("hooks/shared.py")
                                    .update(deep_jit::read(hook_root_a / "hooks/shared.py"))
                                    .get_hex_digest(),
                   "post hook hash depends on lookup order across runtimes");
    deep_jit::write_file_sync(hook_root_a / "hooks/shared.py", "print('modified after hashing')\n");
    DJ_HOST_ASSERT(shared_hook.get_post_hook_hash(hook_config_a) == root_hash,
                   "post hook hash cache unexpectedly refreshed within the process");

    const CompilerOptions missing_options {
        .post_hook = "scripts/missing_post_hook.py",
    };
    expect_failure(
        [&] { runtime.compile_without_load("missing_post_hook", source, missing_options); },
        "failed to open for reading");
    check_tmp_is_empty(cache_root);

    expect_failure(
        [&] { runtime.compile_without_load("failing_post_hook", source, failing_options); },
        "command failed with exit code 7");
    DJ_HOST_ASSERT(not fs::exists(failing_artifact), "failed post hook published a cache artifact");
    check_tmp_is_empty(cache_root);
}

void test_loadable_post_hook(Runtime& runtime) {
    const auto source = get_template_source(67);
    const CompilerOptions options {
        .post_hook = "scripts/loadable_post_hook.py",
    };
    const auto kernel = runtime.compile("loadable_post_hook", source, options);
    DJ_HOST_ASSERT(launch_value(runtime, kernel, 1) == 68, "post-processed CUBIN could not be launched");

    const auto artifact = runtime.compile_without_load("loadable_post_hook", source, options);
    const auto cubin = deep_jit::read(artifact / "kernel.cubin");
    DJ_HOST_ASSERT(cubin.size() >= 16 and cubin.starts_with("\x7f" "ELF") and cubin[12] == 1,
                   "loadable post hook did not modify the ELF padding");
}

void test_default_post_hook(const fs::path& cache_root) {
    const auto include_dir = get_test_cuda_project_dir() / "include_original";
    const auto hook_cache = cache_root / "default_post_hook";
    set_env("DEFAULT_HOOK_JIT_CACHE_DIR", hook_cache.string());
    const auto runtime = make_runtime_with_prefix("DEFAULT_HOOK", include_dir);
    runtime->default_compiler_options.post_hook = "scripts/loadable_post_hook.py";

    const auto source = get_template_source(71);
    const auto kernel = runtime->compile("default_post_hook", source);
    DJ_HOST_ASSERT(launch_value(*runtime, kernel, 1) == 72, "default post hook produced an unloadable CUBIN");
    const auto artifact = runtime->compile_without_load("default_post_hook", source);
    const auto cubin = deep_jit::read(artifact / "kernel.cubin");
    DJ_HOST_ASSERT(cubin.size() >= 16 and cubin[12] == 1,
                   "default post hook was not applied");
    unset_env("DEFAULT_HOOK_JIT_CACHE_DIR");
}

void test_dump_and_launch_overhead(Runtime& runtime) {
    const auto source = get_source("launch_overhead.cu");
    CompilerOptions options;
    options.dump_ptx = true;
    options.dump_sass = runtime.backend.toolkit.cuobjdump.has_value();
    const auto artifact = runtime.compile_without_load("launch_overhead", source, options);
    check_artifact(artifact, source);
    DJ_HOST_ASSERT(fs::is_regular_file(artifact / "kernel.ptx") and fs::file_size(artifact / "kernel.ptx") > 0,
                   "PTX dump was not generated");
    if (*options.dump_sass)
        DJ_HOST_ASSERT(fs::is_regular_file(artifact / "kernel.sass") and fs::file_size(artifact / "kernel.sass") > 0,
                       "SASS dump was not generated");
    DJ_HOST_ASSERT(runtime.compile_without_load("launch_overhead", source) == artifact,
                   "dump options must not change the cache key");

    const auto kernel = runtime.compile("launch_overhead", source);
    const LaunchOptions launch_options {
        .grid_dim = dim3(1, 1, 1),
        .block_dim = dim3(1, 1, 1),
    };
    constexpr int warmup_iterations = 100;
    constexpr int num_rounds = 5;
    constexpr int benchmark_iterations = 10000;
    for (int i = 0; i < warmup_iterations; ++i)
        runtime.launch(kernel, launch_options);
    DJ_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

    std::array<double, num_rounds> samples{};
    for (auto& sample : samples) {
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < benchmark_iterations; ++i)
            runtime.launch(kernel, launch_options);
        const auto end = std::chrono::steady_clock::now();
        DJ_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        sample = std::chrono::duration<double, std::micro>(end - begin).count() / benchmark_iterations;
    }

    std::ranges::sort(samples);
    std::printf("CUDA launch CPU overhead with GIL: median %.3f us, min %.3f us\n",
                samples[num_rounds / 2], samples.front());
}

void test_dump_options_on_cache_hit(const fs::path& cache_root) {
    const auto include_dir = get_test_cuda_project_dir() / "include_original";
    const auto dump_cache = cache_root / "dump_cache_hit";
    set_env("DUMP_CACHE_HIT_JIT_CACHE_DIR", dump_cache.string());
    const auto runtime = make_runtime_with_prefix("DUMP_CACHE_HIT", include_dir);
    const auto source = get_template_source(55);
    const auto artifact = runtime->compile_without_load("dump_cache_hit", source);
    DJ_HOST_ASSERT(not fs::exists(artifact / "kernel.ptx") and not fs::exists(artifact / "kernel.sass"));

    CompilerOptions dump_options;
    dump_options.dump_ptx = true;
    dump_options.dump_sass = runtime->backend.toolkit.cuobjdump.has_value();
    DJ_HOST_ASSERT(runtime->compile_without_load("dump_cache_hit", source, dump_options) == artifact);
    DJ_HOST_ASSERT(not fs::exists(artifact / "kernel.ptx") and not fs::exists(artifact / "kernel.sass"),
                   "dump options unexpectedly rebuilt an existing cache entry");
    const auto metadata = deep_jit::read(artifact / "meta.json");
    DJ_HOST_ASSERT(metadata.find("\"dump_ptx\":false") != std::string::npos and
                   metadata.find("\"dump_sass\":false") != std::string::npos,
                   "cache-hit metadata must describe the original compilation");
    unset_env("DUMP_CACHE_HIT_JIT_CACHE_DIR");
}

void test_ptxas_checks(const fs::path& cache_root) {
    const auto include_dir = get_test_cuda_project_dir() / "include_original";
    const auto compiler_path = cache_root / "ptxas_checks/nvcc";
    const auto ptxas_cache = cache_root / "ptxas_checks/cache_root";
    deep_jit::make_dirs(compiler_path.parent_path());
    write_executable(
        compiler_path,
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then\n"
        "  printf '%s\\n' 'Cuda compilation tools, release 13.1, V13.1.0'\n"
        "  exit 0\n"
        "fi\n"
        "previous=''\n"
        "for argument in \"$@\"; do\n"
        "  if [ \"$previous\" = \"--output-file\" ]; then\n"
        "    printf '%s' 'fake cubin' > \"$argument\"\n"
        "    break\n"
        "  fi\n"
        "  previous=\"$argument\"\n"
        "done\n"
        "printf '%s\\n' \"$DEEP_JIT_TEST_PTXAS_OUTPUT\"\n");
    set_env("PTXAS_CHECK_JIT_CACHE_DIR", ptxas_cache.string());
    set_env("PTXAS_CHECK_JIT_NVCC_COMPILER", compiler_path.string());
    const auto ptxas_runtime = make_runtime_with_prefix("PTXAS_CHECK", include_dir);
    const auto source = get_source("launch_overhead.cu");
    const CompilerOptions spill_check {
        .check_no_spills = true,
    };
    const CompilerOptions local_memory_check {
        .check_no_local_memory = true,
    };

    set_env("DEEP_JIT_TEST_PTXAS_OUTPUT", "ptxas warning : Local memory used for function 'kernel'");
    check_artifact(ptxas_runtime->compile_without_load("local_memory_is_not_a_spill", source, spill_check), source);
    set_env("DEEP_JIT_TEST_PTXAS_OUTPUT", "ptxas warning : Register is SPILLED   TO LOCAL MEMORY in function 'kernel'");
    check_artifact(ptxas_runtime->compile_without_load("a_spill_is_not_local_memory_usage", source, local_memory_check), source);
    const auto spill_options = ptxas_runtime->default_compiler_options.override_with(spill_check);
    const auto spill_artifact = ptxas_cache / "cache" /
        std::format("register_spill.{}", ptxas_runtime->cache_key(source, spill_options));
    expect_failure(
        [&] { ptxas_runtime->compile_without_load("register_spill", source, spill_check); },
        "register spills");
    DJ_HOST_ASSERT(not fs::exists(spill_artifact), "register-spill failure published a cache artifact");
    check_tmp_is_empty(ptxas_cache);

    set_env("DEEP_JIT_TEST_PTXAS_OUTPUT", "PTXAS WARNING : LOCAL   MEMORY USED for function 'kernel'");
    const auto local_options = ptxas_runtime->default_compiler_options.override_with(local_memory_check);
    const auto local_artifact = ptxas_cache / "cache" /
        std::format("local_memory.{}", ptxas_runtime->cache_key(source, local_options));
    expect_failure(
        [&] { ptxas_runtime->compile_without_load("local_memory", source, local_memory_check); },
        "local memory usage");
    DJ_HOST_ASSERT(not fs::exists(local_artifact), "local-memory failure published a cache artifact");
    check_tmp_is_empty(ptxas_cache);
    unset_env("DEEP_JIT_TEST_PTXAS_OUTPUT");
    unset_env("PTXAS_CHECK_JIT_CACHE_DIR");
    unset_env("PTXAS_CHECK_JIT_NVCC_COMPILER");
}

void test_compiler_failure_cleanup(Runtime& runtime, const fs::path& cache_root) {
    const std::string source = "extern \"C\" __global__ void invalid_cuda_source( {\n";
    const auto artifact = runtime.disk_cache.paths.front() / "cache" /
        std::format("invalid_cuda_source.{}", runtime.cache_key(source, runtime.default_compiler_options));
    expect_failure(
        [&] { runtime.compile_without_load("invalid_cuda_source", source); },
        "command failed with exit code");
    DJ_HOST_ASSERT(not fs::exists(artifact), "failed NVCC compilation published a cache artifact");
    check_tmp_is_empty(cache_root);
}

void test_backend_output_validation(Runtime& runtime, const fs::path& cache_root) {
    const auto include_dir = get_test_cuda_project_dir() / "include_original";
    const auto source = get_template_source(94);
    const auto tool_dir = cache_root / "fake_compiler_tools";
    deep_jit::make_dirs(tool_dir);

    const auto failing_after_cubin_nvcc = tool_dir / "nvcc_failing_after_cubin";
    write_executable(
        failing_after_cubin_nvcc,
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then\n"
        "  printf '%s\\n' 'Cuda compilation tools, release 13.1, V13.1.0'\n"
        "  exit 0\n"
        "fi\n"
        "previous=''\n"
        "for argument in \"$@\"; do\n"
        "  if [ \"$previous\" = \"--output-file\" ]; then\n"
        "    printf '%s' 'partial cubin' > \"$argument\"\n"
        "    break\n"
        "  fi\n"
        "  previous=\"$argument\"\n"
        "done\n"
        "exit 37\n");
    const auto failing_after_cubin_cache = cache_root / "failing_after_cubin";
    set_env("FAILING_AFTER_CUBIN_JIT_CACHE_DIR", failing_after_cubin_cache.string());
    set_env("FAILING_AFTER_CUBIN_JIT_NVCC_COMPILER", failing_after_cubin_nvcc.string());
    const auto failing_after_cubin_runtime = make_runtime_with_prefix("FAILING_AFTER_CUBIN", include_dir);
    expect_failure(
        [&] { failing_after_cubin_runtime->compile_without_load("failed_after_cubin", source); },
        "command failed with exit code 37");
    DJ_HOST_ASSERT(not fs::exists(failing_after_cubin_cache / "cache"),
                   "failed compiler output published a cache artifact");
    check_tmp_is_empty(failing_after_cubin_cache);
    write_executable(
        failing_after_cubin_nvcc,
        std::format("#!/bin/sh\nexec {} \"$@\"\n", runtime.backend.toolkit.nvcc.string()));
    DJ_HOST_ASSERT(launch_value(
        *failing_after_cubin_runtime,
        failing_after_cubin_runtime->compile("failed_after_cubin", source), 1) == 95,
        "runtime did not recover after the compiler failed with partial output");
    unset_env("FAILING_AFTER_CUBIN_JIT_CACHE_DIR");
    unset_env("FAILING_AFTER_CUBIN_JIT_NVCC_COMPILER");

    const auto no_cubin_nvcc = tool_dir / "nvcc_without_cubin";
    write_executable(
        no_cubin_nvcc,
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then\n"
        "  printf '%s\\n' 'Cuda compilation tools, release 13.1, V13.1.0'\n"
        "fi\n"
        "exit 0\n");
    const auto no_cubin_cache = cache_root / "no_cubin";
    set_env("NO_CUBIN_JIT_CACHE_DIR", no_cubin_cache.string());
    set_env("NO_CUBIN_JIT_NVCC_COMPILER", no_cubin_nvcc.string());
    const auto no_cubin_runtime = make_runtime_with_prefix("NO_CUBIN", include_dir);
    expect_failure(
        [&] { no_cubin_runtime->compile_without_load("missing_cubin_output", source); },
        "NVCC did not produce a valid CUBIN");
    DJ_HOST_ASSERT(not fs::exists(no_cubin_cache / "cache"),
                   "missing CUBIN output published a cache artifact");
    check_tmp_is_empty(no_cubin_cache);
    write_executable(
        no_cubin_nvcc,
        "#!/bin/sh\n"
        "previous=''\n"
        "for argument in \"$@\"; do\n"
        "  if [ \"$previous\" = \"--output-file\" ]; then\n"
        "    : > \"$argument\"\n"
        "    exit 0\n"
        "  fi\n"
        "  previous=\"$argument\"\n"
        "done\n"
        "exit 0\n");
    expect_failure(
        [&] { no_cubin_runtime->compile_without_load("missing_cubin_output", source); },
        "NVCC did not produce a valid CUBIN");
    DJ_HOST_ASSERT(not fs::exists(no_cubin_cache / "cache"),
                   "empty CUBIN output published a cache artifact");
    check_tmp_is_empty(no_cubin_cache);
    write_executable(
        no_cubin_nvcc,
        std::format("#!/bin/sh\nexec {} \"$@\"\n", runtime.backend.toolkit.nvcc.string()));
    DJ_HOST_ASSERT(launch_value(
        *no_cubin_runtime,
        no_cubin_runtime->compile("missing_cubin_output", source), 1) == 95,
        "runtime did not recover after missing CUBIN output");
    unset_env("NO_CUBIN_JIT_CACHE_DIR");
    unset_env("NO_CUBIN_JIT_NVCC_COMPILER");

    const auto no_ptx_nvcc = tool_dir / "nvcc_without_ptx";
    write_executable(
        no_ptx_nvcc,
        std::format(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ]; then\n"
            "  exec {} \"$@\"\n"
            "fi\n"
            "for argument in \"$@\"; do\n"
            "  if [ \"$argument\" = \"--ptx\" ]; then\n"
            "    exit 0\n"
            "  fi\n"
            "done\n"
            "exec {} \"$@\"\n",
            runtime.backend.toolkit.nvcc.string(),
            runtime.backend.toolkit.nvcc.string()));
    const auto no_ptx_cache = cache_root / "no_ptx";
    set_env("NO_PTX_JIT_CACHE_DIR", no_ptx_cache.string());
    set_env("NO_PTX_JIT_NVCC_COMPILER", no_ptx_nvcc.string());
    const auto no_ptx_runtime = make_runtime_with_prefix("NO_PTX", include_dir);
    expect_failure(
        [&] {
            no_ptx_runtime->compile_without_load(
                "missing_ptx_output", source, CompilerOptions {.dump_ptx = true});
        },
        "NVCC did not produce a valid PTX");
    DJ_HOST_ASSERT(not fs::exists(no_ptx_cache / "cache"),
                   "missing PTX output published a cache artifact");
    check_tmp_is_empty(no_ptx_cache);
    write_executable(
        no_ptx_nvcc,
        std::format(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ]; then\n"
            "  exec {} \"$@\"\n"
            "fi\n"
            "is_ptx=0\n"
            "previous=''\n"
            "for argument in \"$@\"; do\n"
            "  if [ \"$argument\" = \"--ptx\" ]; then\n"
            "    is_ptx=1\n"
            "  fi\n"
            "  if [ \"$previous\" = \"--output-file\" ] && [ \"$is_ptx\" = 1 ]; then\n"
            "    : > \"$argument\"\n"
            "    exit 0\n"
            "  fi\n"
            "  previous=\"$argument\"\n"
            "done\n"
            "exec {} \"$@\"\n",
            runtime.backend.toolkit.nvcc.string(),
            runtime.backend.toolkit.nvcc.string()));
    expect_failure(
        [&] {
            no_ptx_runtime->compile_without_load(
                "missing_ptx_output", source, CompilerOptions {.dump_ptx = true});
        },
        "NVCC did not produce a valid PTX");
    DJ_HOST_ASSERT(not fs::exists(no_ptx_cache / "cache"),
                   "empty PTX output published a cache artifact");
    check_tmp_is_empty(no_ptx_cache);
    write_executable(
        no_ptx_nvcc,
        std::format("#!/bin/sh\nexec {} \"$@\"\n", runtime.backend.toolkit.nvcc.string()));
    const auto recovered_ptx_artifact = no_ptx_runtime->compile_without_load(
        "missing_ptx_output", source, CompilerOptions {.dump_ptx = true});
    DJ_HOST_ASSERT(fs::file_size(recovered_ptx_artifact / "kernel.ptx") > 0,
                   "runtime did not recover after missing PTX output");
    unset_env("NO_PTX_JIT_CACHE_DIR");
    unset_env("NO_PTX_JIT_NVCC_COMPILER");

    const auto metadata_hook_root = tool_dir / "metadata_hook_root";
    const auto metadata_hook_path = metadata_hook_root / "hooks/metadata_conflict.py";
    const auto metadata_cache = cache_root / "metadata_failure";
    deep_jit::make_dirs(metadata_hook_path.parent_path());
    deep_jit::write_file_sync(
        metadata_hook_path,
        "from pathlib import Path\n"
        "Path('meta.json').mkdir()\n");
    set_env("METADATA_FAILURE_JIT_CACHE_DIR", metadata_cache.string());
    const auto metadata_runtime = std::make_shared<Runtime>(deep_jit::Config(
        fs::absolute(metadata_hook_root),
        "METADATA_FAILURE",
        "test-signature",
        std::vector<fs::path>{include_dir},
        std::vector<std::string>{"test_cuda/"}));
    const CompilerOptions metadata_options {.post_hook = "hooks/metadata_conflict.py"};
    const auto effective_metadata_options =
        metadata_runtime->default_compiler_options.override_with(metadata_options);
    const auto metadata_artifact = metadata_cache / "cache" /
        std::format("metadata_failure.{}", metadata_runtime->cache_key(source, effective_metadata_options));
    expect_failure(
        [&] { metadata_runtime->compile_without_load("metadata_failure", source, metadata_options); },
        "failed to open for writing");
    DJ_HOST_ASSERT(not fs::exists(metadata_artifact), "metadata write failure published a cache artifact");
    check_tmp_is_empty(metadata_cache);
    deep_jit::write_file_sync(metadata_hook_path, "# successful retry\n");
    DJ_HOST_ASSERT(metadata_runtime->compile_without_load("metadata_failure", source, metadata_options) == metadata_artifact,
                   "metadata failure retry changed the cached post-hook key");
    DJ_HOST_ASSERT(launch_value(
        *metadata_runtime,
        metadata_runtime->compile("metadata_failure", source, metadata_options), 1) == 95,
        "runtime did not recover after a metadata write failure");
    unset_env("METADATA_FAILURE_JIT_CACHE_DIR");

    if (not runtime.backend.toolkit.cuobjdump)
        return;

    const auto empty_cuobjdump = tool_dir / "empty_cuobjdump";
    write_executable(empty_cuobjdump, "#!/bin/sh\nexit 0\n");
    const auto empty_sass_cache = cache_root / "empty_sass";
    set_env("EMPTY_SASS_JIT_CACHE_DIR", empty_sass_cache.string());
    const auto empty_sass_runtime = make_runtime_with_prefix("EMPTY_SASS", include_dir);
    empty_sass_runtime->backend.toolkit.cuobjdump = empty_cuobjdump;
    expect_failure(
        [&] {
            empty_sass_runtime->compile_without_load(
                "empty_sass_output", source, CompilerOptions {.dump_sass = true});
        },
        "cuobjdump did not produce valid SASS");
    DJ_HOST_ASSERT(not fs::exists(empty_sass_cache / "cache"),
                   "empty SASS output published a cache artifact");
    check_tmp_is_empty(empty_sass_cache);
    write_executable(
        empty_cuobjdump,
        std::format("#!/bin/sh\nexec {} \"$@\"\n", runtime.backend.toolkit.cuobjdump->string()));
    const auto recovered_sass_artifact = empty_sass_runtime->compile_without_load(
        "empty_sass_output", source, CompilerOptions {.dump_sass = true});
    DJ_HOST_ASSERT(fs::file_size(recovered_sass_artifact / "kernel.sass") > 0,
                   "runtime did not recover after empty SASS output");
    unset_env("EMPTY_SASS_JIT_CACHE_DIR");

    const auto missing_cuobjdump_cache = cache_root / "missing_cuobjdump";
    set_env("MISSING_CUOBJDUMP_JIT_CACHE_DIR", missing_cuobjdump_cache.string());
    const auto missing_cuobjdump_runtime = make_runtime_with_prefix("MISSING_CUOBJDUMP", include_dir);
    missing_cuobjdump_runtime->backend.toolkit.cuobjdump.reset();
    expect_failure(
        [&] {
            missing_cuobjdump_runtime->compile_without_load(
                "missing_cuobjdump", source, CompilerOptions {.dump_sass = true});
        },
        "toolkit.cuobjdump.has_value()");
    DJ_HOST_ASSERT(not fs::exists(missing_cuobjdump_cache / "cache"),
                   "missing cuobjdump published a cache artifact");
    check_tmp_is_empty(missing_cuobjdump_cache);
    missing_cuobjdump_runtime->backend.toolkit.cuobjdump = runtime.backend.toolkit.cuobjdump;
    const auto recovered_cuobjdump_artifact = missing_cuobjdump_runtime->compile_without_load(
        "missing_cuobjdump", source, CompilerOptions {.dump_sass = true});
    DJ_HOST_ASSERT(fs::file_size(recovered_cuobjdump_artifact / "kernel.sass") > 0,
                   "runtime did not recover after restoring cuobjdump");
    unset_env("MISSING_CUOBJDUMP_JIT_CACHE_DIR");

}

void test_invalid_tag(Runtime& runtime) {
    expect_failure(
        [&] { runtime.compile_without_load("invalid-tag", get_template_source(0)); },
        "cache tag must contain only letters, digits, or underscores");
    expect_failure(
        [&] { runtime.compile("invalid-tag", get_template_source(91)); },
        "cache tag must contain only letters, digits, or underscores");
    expect_failure(
        [&] { runtime.compile("", get_template_source(92)); },
        "cache tag must contain only letters, digits, or underscores");
}

void test_kernel_count(Runtime& runtime) {
    const auto no_kernel_source = get_source("no_kernel.cu");
    check_artifact(runtime.compile_without_load("no_kernel", no_kernel_source), no_kernel_source);
    expect_failure(
        [&] { runtime.compile("no_kernel", no_kernel_source); },
        "expected exactly one kernel");

    const auto source = get_source("multiple_kernels.cu");
    check_artifact(runtime.compile_without_load("multiple_kernels", source), source);
    expect_failure(
        [&] { runtime.compile("multiple_kernels", source); },
        "expected exactly one kernel");
}

void run_tests(pybind11::module_ module) {
    const auto cache_root_env = std::getenv("DEEP_JIT_CUDA_TEST_CACHE_ROOT");
    DJ_HOST_ASSERT(cache_root_env != nullptr, "DEEP_JIT_CUDA_TEST_CACHE_ROOT must be set");
    const auto cache_root = fs::absolute(cache_root_env).lexically_normal();

    run_test("environment precedence", [&] { test_environment(cache_root); });
    run_test("configuration", test_config);
    run_test("filesystem utilities", [&] { test_filesystem(cache_root); });
    run_test("external command and UUID", test_command_and_uuid);
    run_test("disk cache lifecycle", [&] { test_disk_cache(cache_root); });
    run_test("memory cache", test_memory_cache);
    run_test("JSON serialization", test_json);
    run_test("hash boundaries", test_hash_boundaries);
    run_test("hash robustness", test_hash_robustness);
    run_test("lazy init and kernel arguments", test_lazy_init_and_kernel_arguments);
    run_test("include parser", [&] { test_parser(cache_root); });
    run_test("generated include graph", [&] { test_generated_include_graph(cache_root); });
    run_test("CUDA toolkit discovery", [&] { test_cuda_toolkit_discovery(cache_root); });

    const auto include_dir = get_test_cuda_project_dir() / "include_original";
    const auto runtime = make_runtime(include_dir);
    DJ_HOST_ASSERT(runtime->disk_cache.paths.front() == cache_root,
                   "library-prefixed cache directory was not selected");
    run_test("CUDA device", [&] { test_device(*runtime); });
    run_test("TMA driver wrapper and kernel argument", [&] { test_tma_driver_wrapper(*runtime); });
    run_test("compiler options", [&] { test_options(*runtime); });
    run_test("runtime launch features", [&] { test_runtime_launch_features(*runtime); });
    run_test("cooperative launch", [&] { return test_cooperative_launch(*runtime); });
    run_test("cluster launch", [&] { return test_cluster_launch(*runtime); });
    run_test("multidimensional launch", [&] { test_multidimensional_launch(*runtime); });
    run_test("mixed kernel arguments", [&] { test_mixed_kernel_arguments(*runtime); });
    run_test("large kernel arguments", [&] { test_large_kernel_arguments(*runtime); });
    run_test("macro-selected kernel ABI", [&] { test_macro_selected_abi(*runtime); });
    run_test("large dynamic shared memory", [&] { test_large_dynamic_shared_memory(*runtime); });
    run_test("generated float literals", [&] { test_generated_float_literals(cache_root); });
    run_test("DG and EP environment compatibility", [&] { test_library_environment_compatibility(*runtime, cache_root); });
    run_test("multiple runtimes", [&] { test_multiple_runtimes(cache_root); });
    run_test("lazy runtime", [&] { test_lazy_runtime(cache_root); });
    run_test("consumer default compiler flags", [&] { test_consumer_default_compiler_flags(cache_root); });
    run_test("debug environment", [&] { test_debug_environment(cache_root); });
    run_test("untracked dependency include flags", [&] { test_untracked_dependency_include_flags(cache_root); });
    run_test("multiple device runtimes", [&] { return test_multiple_device_runtimes(cache_root); });
    run_test("architecture override", [&] { test_arch_override(cache_root); });
    run_test("kernel lifecycle", [&] { test_kernel_lifecycle(cache_root); });
    run_test("Python API", [&] { test_python_api(module, runtime); });
    run_test("template hash and runtime arguments", [&] { test_template_hash_and_launch(*runtime); });
    run_test("secondary disk cache", [&] { test_secondary_disk_cache(cache_root); });
    run_test("compiler option hash and metadata", [&] { test_compiler_options(*runtime); });
    run_test("cache key option matrix", [&] { test_cache_key_option_matrix(*runtime); });
    run_test("include directory and signature hashes", [&] { test_include_dirs(cache_root); });
    run_test("relocated wheel include directory", [&] { test_relocated_wheel_include_dir(cache_root); });
    run_test("post hook", [&] { test_post_hook(*runtime, cache_root); });
    run_test("loadable post hook", [&] { test_loadable_post_hook(*runtime); });
    run_test("default post hook", [&] { test_default_post_hook(cache_root); });
    run_test("invalid cache tag", [&] { test_invalid_tag(*runtime); });
    run_test("kernel count", [&] { test_kernel_count(*runtime); });
    run_test("PTXAS checks", [&] { test_ptxas_checks(cache_root); });
    run_test("compiler failure cleanup", [&] { test_compiler_failure_cleanup(*runtime, cache_root); });
    run_test("backend output validation", [&] { test_backend_output_validation(*runtime, cache_root); });
    run_test("dump artifacts and launch overhead", [&] { test_dump_and_launch_overhead(*runtime); });
    run_test("dump options on cache hit", [&] { test_dump_options_on_cache_hit(cache_root); });
    check_tmp_is_empty(cache_root);

    std::puts("All DeepJIT tests passed");
}

std::string compile_for_process(const std::string& tag, const int bias) {
    DJ_HOST_ASSERT(process_test_runtime != nullptr, "process test runtime was not prepared");
    return process_test_runtime->compile_without_load(tag, get_template_source(bias)).string();
}

std::string publish_disk_cache_entry(const std::string& cache_root,
                                     const std::string& owner,
                                     const std::string& ready_path,
                                     const std::string& start_path) {
    deep_jit::DiskCache cache({fs::absolute(cache_root).lexically_normal()});
    auto entry = cache.entry("atomic_publication", "digest");
    DJ_HOST_ASSERT(not entry.hit, "direct cache writer unexpectedly found a cache hit");
    deep_jit::write_file_sync(entry.path / "owner_a", owner);
    deep_jit::write_file_sync(entry.path / "owner_b", owner);
    deep_jit::write_file_sync(ready_path, "ready");

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (not fs::exists(start_path)) {
        DJ_HOST_ASSERT(std::chrono::steady_clock::now() < deadline,
                       "direct cache writer did not leave the barrier");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const auto published_path = entry.commit();
    const auto winner = deep_jit::read(published_path / "owner_a");
    DJ_HOST_ASSERT(deep_jit::read(published_path / "owner_b") == winner,
                   "atomic cache publication mixed files from different writers");
    return winner;
}

int launch_for_process(const std::string& tag, const int bias) {
    DJ_HOST_ASSERT(process_test_runtime != nullptr, "process test runtime was not prepared");
    const auto kernel = process_test_runtime->compile(tag, get_template_source(bias));
    return launch_value(*process_test_runtime, kernel, 1);
}

void prepare_process_runtime() {
    const auto include_dir = get_test_cuda_project_dir() / "include_original";
    process_test_runtime = make_runtime_with_prefix("PROCESS_TEST", include_dir, "process-cache-test");
}

void prepare_gil_runtime() {
    const auto include_dir = get_test_cuda_project_dir() / "include_original";
    gil_test_runtime = make_runtime_with_prefix("GIL_TEST", include_dir, "gil-test");
}

std::string compile_for_gil_test() {
    DJ_HOST_ASSERT(gil_test_runtime != nullptr, "GIL test runtime was not prepared");
    return gil_test_runtime->compile_without_load("gil_compile", get_template_source(81)).string();
}

void init_python_api_jit(const std::string& library_root) {
    const auto root = fs::absolute(library_root).lexically_normal();
    python_api_jit = deep_jit::create_lazy_jit<deep_jit::CUDA>(deep_jit::Config(
        root,
        "PYTHON_API",
        "python-api-test",
        {root / "include_original"},
        {"test_cuda/"}));
}

int run_registered_jit(const int bias) {
    auto runtime = python_api_jit.get();
    return launch_value(*runtime, runtime->compile("python_api", get_template_source(bias)), 1);
}

std::string get_registered_jit_arch() {
    return *python_api_jit->default_compiler_options.arch;
}

void compile_invalid_registered_jit() {
    python_api_jit->compile_without_load(
        "python_api_invalid", "extern \"C\" __global__ void python_api_invalid( {\n");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    deep_jit::register_python_api(module, python_api_jit);
    module.def("init_jit", &init_python_api_jit);
    module.def("run_registered_jit", &run_registered_jit);
    module.def("get_registered_jit_arch", &get_registered_jit_arch);
    module.def("compile_invalid_registered_jit", &compile_invalid_registered_jit);
    module.def("run_tests", &run_tests);
    module.def("prepare_process_runtime", &prepare_process_runtime);
    module.def("compile_for_process", &compile_for_process);
    module.def("launch_for_process", &launch_for_process);
    module.def("publish_disk_cache_entry", &publish_disk_cache_entry);
    module.def("prepare_gil_runtime", &prepare_gil_runtime);
    module.def("compile_for_gil_test", &compile_for_gil_test);
}
