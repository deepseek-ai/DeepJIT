#include <algorithm>
#include <chrono>
#include <csignal>
#include <iostream>
#include <regex>
#include <thread>
#include <unordered_set>
#include <vector>

#include <deep_jit/cache/disk.hpp>
#include <deep_jit/utils/command.hpp>
#include <deep_jit/utils/lazy.hpp>

#if defined(DJ_TEST_CUDA)
#include <deep_jit/backend/cuda/backend.hpp>
#endif

#if defined(min) || defined(max)
#error Windows headers must not define min/max macros
#endif

namespace fs = std::filesystem;
using namespace deep_jit;

#if defined(_WIN32)
DJ_DECL_LAZY_DL_HANDLE(test_library, "kernel32.dll");
DJ_DECL_LAZY_DL_FUNCTION(test_library, GetCurrentProcessId);
#else
DJ_DECL_LAZY_DL_HANDLE(test_library, "libc.so.6");
DJ_DECL_LAZY_DL_FUNCTION(test_library, getpid);
#endif
DJ_DECL_LAZY_DL_HANDLE(missing_library, "deep_jit_missing_test_library");
extern "C" int deep_jit_missing_test_symbol();
DJ_DECL_LAZY_DL_FUNCTION(test_library, deep_jit_missing_test_symbol);

template <typename Function>
void expect_error(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const exception::Exception& error) {
        DJ_HOST_ASSERT(std::string(error.what()).find(message) != std::string::npos,
                       "unexpected error: {}", error.what());
        return;
    }
    DJ_PANIC("expected an error containing: {}", message);
}

void test_filesystem(const fs::path& root) {
    const auto directory = root / "nested directory";
    make_dirs(directory);
    const auto file = directory / "data.bin";
    const std::string contents("binary\0data\n", 12);
    write_file_sync(file, contents);
    DJ_HOST_ASSERT(read(file) == contents);
    write_file_sync(directory / "empty", "");
    fsync_dir(root, false);
    DJ_HOST_ASSERT(try_update_mtime(file));
    DJ_HOST_ASSERT(not try_update_mtime(root / "missing"));
    DJ_HOST_ASSERT(not is_executable(directory));
    expect_error([&] { fsync_file(root / "missing"); }, "failed to open for fsync");
    safe_remove_all(directory);
    DJ_HOST_ASSERT(not fs::exists(directory));
}

void test_cache(const fs::path& root) {
    DiskCache cache({root / "writable"});
    auto first = cache.entry("test", "race");
    auto second = cache.entry("test", "race");
    DJ_HOST_ASSERT(not first.hit and not second.hit and first.path != second.path);
    const auto first_temporary = first.path;
    const auto second_temporary = second.path;
    make_dirs(first.path / "nested");
    write_file_sync(first.path / "nested" / "result", "first");
    write_file_sync(second.path / "result", "second");
    DJ_HOST_ASSERT(not fs::exists(first.commit_path));
    const auto published = first.commit();
    DJ_HOST_ASSERT(first.commit() == published);
    DJ_HOST_ASSERT(fs::is_regular_file(published / kCommitFileName));
    DJ_HOST_ASSERT(not fs::exists(first_temporary));
    DJ_HOST_ASSERT(second.commit() == published);
    DJ_HOST_ASSERT(not fs::exists(second_temporary));
    DJ_HOST_ASSERT(read(published / "nested" / "result") == "first");
    DJ_HOST_ASSERT(not fs::exists(published / "result"));
    DJ_HOST_ASSERT(cache.entry("test", "race").hit);

    DiskCache lookup({root / "unused", root / "writable"});
    DJ_HOST_ASSERT(lookup.entry("test", "race").path == published);
    DJ_HOST_ASSERT(not fs::exists(root / "unused"));

    fs::path abandoned;
    {
        auto entry = cache.entry("test", "abandoned");
        abandoned = entry.path;
        write_file_sync(entry.path / "partial", "incomplete");
    }
    DJ_HOST_ASSERT(not fs::exists(abandoned));

    fs::path failed_temporary;
    {
        auto entry = cache.entry("test", "failed");
        failed_temporary = entry.path;
        make_dirs(entry.commit_path);
        write_file_sync(entry.commit_path / "uncommitted", "do not overwrite");
        expect_error([&] { entry.commit(); }, "failed to publish cache entry");
        DJ_HOST_ASSERT(not entry.hit and not entry.committed);
        DJ_HOST_ASSERT(fs::exists(entry.path));
        DJ_HOST_ASSERT(read(entry.commit_path / "uncommitted") == "do not overwrite");
    }
    DJ_HOST_ASSERT(not fs::exists(failed_temporary));
}

void cache_worker(const fs::path& root) {
    DiskCache cache({root / "cache"});
    auto entry = cache.entry("test", "concurrent");
    DJ_HOST_ASSERT(not entry.hit);
    const auto id = get_uuid();
    write_file_sync(entry.path / "result", id);
    write_file_sync(root / id, "ready");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (not fs::exists(root / "go")) {
        DJ_HOST_ASSERT(std::chrono::steady_clock::now() < deadline, "cache worker timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << entry.commit().string() << '\n';
}

void test_command(const fs::path& root, const fs::path& executable) {
    const auto child = quote_command_path(executable);
#if defined(_WIN32)
    const auto argument = root / "spaces & (parentheses)" / "";
    const auto cd = "cd /D " + quote_command_path(root) + " && ";
    expect_error([&] { quote_command_path(root / "%PATH%"); }, "unsupported shell metacharacter");
    expect_error([&] { quote_command_path(root / "!PATH!"); }, "unsupported shell metacharacter");
#else
    const auto argument = root / "argument";
    const auto cd = "cd " + root.string() + " && ";
#endif
    const auto output = call_external_command(child + " child-output " + quote_command_path(argument));
    DJ_HOST_ASSERT(output == argument.string() + "\nstderr output\n", "unexpected output: {}", output);
    DJ_HOST_ASSERT(call_external_command(cd + child + " child-cwd") == root.string() + "\n");
    expect_error([&] { call_external_command(child + " child-failure"); }, "exit code 23");
    expect_error([&] { call_external_command(child + " child-failure"); }, "child failure");
    expect_error([] { call_external_command(""); }, "command must not be empty");
#if !defined(_WIN32)
    expect_error([&] { call_external_command(child + " child-signal"); }, "exit code 143");
#endif
}

void test_lazy() {
#if defined(_WIN32)
    DJ_HOST_ASSERT(lazy_GetCurrentProcessId() == ::GetCurrentProcessId());
#else
    DJ_HOST_ASSERT(lazy_getpid() == ::getpid());
#endif
    DJ_HOST_ASSERT(test_library() == test_library());
    for (int retry = 0; retry < 2; ++retry) {
        expect_error([] { missing_library(); }, "failed to load deep_jit_missing_test_library");
        expect_error([] { lazy_deep_jit_missing_test_symbol(); }, "failed to load deep_jit_missing_test_symbol");
    }
}

void test_threads() {
    std::vector<std::vector<std::string>> results(8);
    std::vector<std::thread> threads;
    for (auto& result : results) {
        threads.emplace_back([&result] {
            for (int i = 0; i < 128; ++i) {
                result.emplace_back(get_uuid());
                expect_error([] { DJ_PANIC("threaded failure"); }, "threaded failure");
            }
        });
    }
    for (auto& thread : threads)
        thread.join();
    std::unordered_set<std::string> unique;
    const std::regex uuid_pattern(R"(\d+-[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8})");
    for (const auto& result : results) {
        for (const auto& uuid : result) {
            DJ_HOST_ASSERT(std::regex_match(uuid, uuid_pattern));
            DJ_HOST_ASSERT(unique.insert(uuid).second);
        }
    }
}

#if defined(DJ_TEST_CUDA)
void test_cuda(const fs::path& root, const bool dump_sass) {
    const Env env("PLATFORM_TEST");
    CUDA backend(env);
    DJ_HOST_ASSERT(is_executable(backend.toolkit.nvcc));
    DJ_HOST_ASSERT(not dump_sass or backend.toolkit.cuobjdump.has_value());
    const auto include_dir = root / "include directory";
    make_dirs(include_dir);
    write_file_sync(include_dir / "constant.cuh", "#define TEST_VALUE 7\n");
    const auto hook = root / "post hook.py";
    write_file_sync(hook, "import pathlib, sys\npathlib.Path(sys.argv[1]).with_suffix('.hook').write_text('hook ran')\n");
    const Config config(root, "PLATFORM_TEST", "", {include_dir});
    CUDA::CompilerOptions options {
        .optimize_level = "3",
        .ptxas_register_usage_level = 10,
        .dump_ptx = true,
        .dump_sass = dump_sass,
        .arch = "80",
        .post_hook = hook.filename().string(),
    };
    DJ_HOST_ASSERT(not options.get_post_hook_hash(config).empty());
    DJ_HOST_ASSERT(options.get_post_hook_hash(config) == options.get_post_hook_hash(config));
    for (const auto& level : {"0", "1", "2", "3"}) {
        options.optimize_level = level;
        const auto flags = options.get_flags();
#if defined(_WIN32)
        const auto host_flag = std::string("--compiler-options=") +
            (std::string(level) == "0" ? "/Od" : std::string(level) == "1" ? "/O1" : "/O2");
#else
        const auto host_flag = std::string("--compiler-options=-O") + level;
#endif
        DJ_HOST_ASSERT(std::ranges::find(flags, host_flag) != flags.end());
    }
    DiskCache cache({root / "cache directory"});
    auto entry = cache.entry("cuda", "toolchain");
    backend.compile("#include <constant.cuh>\nextern \"C\" __global__ void test(int* p) { *p = TEST_VALUE; }\n",
                    entry.path, env, config, options);
    const auto output = entry.commit();
    for (const auto& name : {"kernel.cubin", "kernel.ptx", "meta.json", "kernel.hook"})
        DJ_HOST_ASSERT(fs::is_regular_file(output / name) and fs::file_size(output / name) > 0, "missing {}", name);
    if (dump_sass)
        DJ_HOST_ASSERT(fs::file_size(output / "kernel.sass") > 0);
}
#endif

int main(const int argc, const char* const* argv) {
    try {
        DJ_HOST_ASSERT(argc >= 2);
        const std::string mode = argv[1];
        if (mode == "child-output") {
            DJ_HOST_ASSERT(argc == 3);
            std::cout << argv[2] << std::endl;
            std::cerr << "stderr output\n";
        } else if (mode == "child-cwd") {
            std::cout << fs::current_path().string() << '\n';
        } else if (mode == "child-failure") {
            std::cerr << "child failure\n";
            return 23;
        } else if (mode == "child-signal") {
            std::raise(SIGTERM);
        } else if (mode == "environment") {
            for (const auto& path : DiskCache::from_env(Env("PLATFORM_TEST")).paths)
                std::cout << path.string() << '\n';
#if defined(DJ_TEST_CUDA)
        } else if (mode == "cuda-toolkit") {
            const auto toolkit = CUDA::find_cuda_toolkit(Env("PLATFORM_TEST"));
            std::cout << toolkit.nvcc.string() << '\n';
            if (toolkit.cuobjdump)
                std::cout << toolkit.cuobjdump->string() << '\n';
#endif
        } else {
            DJ_HOST_ASSERT(argc == 3);
            const fs::path root = fs::absolute(argv[2]);
            if (mode == "filesystem")
                test_filesystem(root);
            else if (mode == "cache")
                test_cache(root);
            else if (mode == "cache-worker")
                cache_worker(root);
            else if (mode == "command")
                test_command(root, fs::absolute(argv[0]));
            else if (mode == "lazy")
                test_lazy();
            else if (mode == "threads")
                test_threads();
#if defined(DJ_TEST_CUDA)
            else if (mode == "cuda" or mode == "cuda-sass")
                test_cuda(root, mode == "cuda-sass");
#endif
            else
                DJ_PANIC("unknown test: {}", mode);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
