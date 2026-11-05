#define DJ_DISABLE_GIL 1
#include <deep_jit/runtime/runtime.hpp>
#include <deep_jit/utils/command.hpp>
#include <atomic>
#include <cassert>
#include <cstring>
#include <dlfcn.h>
#include <iostream>

using namespace deep_jit;
using namespace std::chrono_literals;
static std::atomic<bool> fail_all{false}, fail_primary{false}, stall_primary{false}, release_stall{false}, stall_done{false};
static std::string fd_path(int fd) {
    char target[4096];
    const auto size = ::readlink(("/proc/self/fd/" + std::to_string(fd)).c_str(), target, sizeof(target));
    return size < 0 ? "" : std::string(target, static_cast<size_t>(size));
}
extern "C" int __real_fsync(int);
extern "C" int __wrap_fsync(int fd) {
    const auto path = fd_path(fd);
    if (fail_all) { errno = ENOSPC; return -1; }
    if (path.find("/primary/") != std::string::npos) {
        if (fail_primary) { errno = ENOSPC; return -1; }
        if (stall_primary) {
            while (not release_stall.load()) std::this_thread::sleep_for(1ms);
            const auto result = __real_fsync(fd);
            stall_done = true;
            return result;
        }
    }
    return __real_fsync(fd);
}
// std::filesystem's shared-library calls resolve this interposed symbol.
extern "C" int rename(const char* source, const char* target) noexcept {
    static auto real = reinterpret_cast<decltype(&::rename)>(::dlsym(RTLD_NEXT, "rename"));
    if (std::strstr(source, "/rename_error/")) { errno = EIO; return -1; }
    return real(source, target);
}

static std::atomic<int> load_fault{0}; // 1: short read, 2: blocked read, 3: rejected image
static std::atomic<bool> release_read{false}, read_done{false};
static thread_local bool loading = false, returned_partial = false;
static std::atomic<int> load_calls{0};
static const auto caller_thread = std::this_thread::get_id();
extern "C" ssize_t read(int fd, void* buffer, size_t count) {
    static auto real = reinterpret_cast<decltype(&::read)>(::dlsym(RTLD_NEXT, "read"));
    if (loading and fd_path(fd).find("/primary/") != std::string::npos and
        fd_path(fd).ends_with("/binary")) {
        if (load_fault == 1) {
            if (returned_partial) return 0;
            returned_partial = true;
            return real(fd, buffer, std::min(count, size_t(3)));
        }
        if (load_fault == 2) {
            if (not returned_partial) {
                returned_partial = true;
                return real(fd, buffer, std::min(count, size_t(3)));
            }
            while (not release_read) std::this_thread::sleep_for(1ms);
            read_done = true;
        }
    }
    return real(fd, buffer, count);
}
static std::atomic<bool> partial_write{false}, release_write{false}, write_done{false};

static int compile_mode = 0;
static std::filesystem::path compile_barrier;
static std::atomic<int> compile_errors{0};

struct FakeBackend {
    struct Device {};
    struct Kernel { std::string data; };
    struct Options {
        template<class... Args> static Options default_options(const Args&...) { return {}; }
        Options override_with(const Options&) const { return {}; }
        void update_hash(hash::FNV1a&) const {}
        std::string get_post_hook_hash(const Config&) const { return {}; }
    };
    using CompilerOptions = Options;
    using LaunchOptions = Options;
    struct Info { std::string get_hash() const { return "fake"; } } compiler_info;
    explicit FakeBackend(const Env&) {}
    void compile(const std::string& source, const std::filesystem::path& dir,
                 const Env&, const Config&, const Options&) const {
        if (compile_mode) {
            assert(not detail::in_iothread);
            if (compile_mode == 1)
                call_external_command("sleep 0.2");
            if (compile_mode == 2 and dir.string().find("/primary/") != std::string::npos) {
                try { call_external_command("exit 7"); }
                catch (const exception::Exception& error) {
                    assert(std::string(error.what()).find("exit code 7") != std::string::npos);
                    ++compile_errors;
                    throw;
                }
            }
            if (compile_mode == 3) {
                const auto barrier = compile_barrier.string();
                call_external_command("touch " + barrier + "/" + source +
                    "; while [ ! -f " + barrier + "/one ] || [ ! -f " + barrier +
                    "/two ]; do sleep 0.01; done; sleep 0.2");
            }
        }
        run_in_iothread([dir, source] {
            if (partial_write and dir.string().find("/primary/") != std::string::npos) {
                write_file_sync(dir / "binary", source.substr(0, 3));
                while (not release_write) std::this_thread::sleep_for(1ms);
                write_done = true;
            }
            write_file_sync(dir / "binary", source);
        });
    }
    static std::shared_ptr<Kernel> load(const std::filesystem::path& dir, const Env&) {
        ++load_calls;
        assert(std::this_thread::get_id() == caller_thread);
        auto data = run_in_iothread([dir] {
            struct Loading {
                Loading() { loading = true; returned_partial = false; }
                ~Loading() { loading = false; }
            } guard;
            return read_bytes(dir / "binary");
        });
        if (load_fault == 3 and dir.string().find("/primary/") != std::string::npos)
            DJ_PANIC("invalid binary image");
        DJ_HOST_ASSERT(data == "contents", "invalid fake binary image");
        return std::make_shared<Kernel>(Kernel{std::move(data)});
    }
};

template<class F> static void expect_io_error(F f) {
    try { f(); assert(false); }
    catch (const std::exception&) {}
}
static void configure(const std::filesystem::path& root) {
    const auto paths = (root / "primary").string() + ":" + (root / "secondary").string();
    ::setenv("FAULT_JIT_CACHE_DIR", paths.c_str(), 1);
}
int main(int argc, char** argv) {
    assert(argc == 2);
    const std::filesystem::path root(argv[1]);
    make_dirs(root);
    write_file_sync(root / "file", "0123456789");
    assert(read(root / "file") == "0123456789");
    try { read(root / "missing"); assert(false); }
    catch (const exception::Exception&) {}

    // Genuine Runtime retries its compile+publish transaction on the next root.
    configure(root / "fallback");
    fail_primary = true;
    Runtime<FakeBackend> runtime(Config(root, "FAULT"));
    const auto artifact = runtime.compile_without_load("kernel", "contents");
    assert(load_calls == 0);
    assert(artifact.string().find("/secondary/") != std::string::npos);
    assert(read(artifact / "binary") == "contents");
    fail_primary = false;
    auto kernel = runtime.compile("kernel", "contents");
    assert(kernel->data == "contents");
    assert(runtime.compile("kernel", "contents") == kernel);
    assert(load_calls == 1);

    // Exhausting all roots reports an error; loaded kernels remain usable.
    configure(root / "all_failed");
    fail_all = true;
    Runtime<FakeBackend> all_failed(Config(root, "FAULT"));
    expect_io_error([&] { all_failed.compile_without_load("kernel", "contents"); });
    assert(runtime.compile("kernel", "contents") == kernel);
    fail_all = false;

    // Directory setup errors also skip to the next root.
    write_file_sync(root / "not_a_directory", "file");
    DiskCache directory_fallback({root / "not_a_directory", root / "good_directory"});
    assert(directory_fallback.entry("kernel", "key").root_index == 1);

    // Timeout abandons the original temporary directory and continues on root 2.
    configure(root / "timeout");
    stall_primary = true;
    Runtime<FakeBackend> timeout(Config(root, "FAULT"));
    const auto fallback = timeout.compile_without_load("kernel", "late data");
    assert(fallback.string().find("/secondary/") != std::string::npos);
    release_stall = true;
    while (not stall_done.load()) std::this_thread::sleep_for(1ms);
    stall_primary = false;
    assert(read(fallback / "binary") == "late data");

    // Publication retains the original empty marker.
    DiskCache cache({root / "cache"});
    {
        auto entry = cache.entry("kernel", "key");
        write_file_sync(entry.path / "binary", "data");
        assert(read(entry.commit() / ".committed").empty());
    }
    assert(cache.entry("kernel", "key").hit);
    const auto legacy = root / "legacy/cache/kernel.key";
    make_dirs(legacy);
    write_file_sync(legacy / ".committed", "");
    assert(DiskCache({root / "legacy"}).entry("kernel", "key").hit);

    for (const auto* name : {"rename_error"}) {
        DiskCache broken({root / name});
        auto entry = broken.entry("kernel", "key");
        write_file_sync(entry.path / "binary", "data");
        expect_io_error([&] { entry.commit(); });
    }
    // Two concurrent publishers: complete existing entry can win.
    DiskCache collision({root / "collision"});
    auto first = collision.entry("kernel", "key");
    auto second = collision.entry("kernel", "key");
    write_file_sync(first.path / "binary", "same");
    write_file_sync(second.path / "binary", "same");
    assert(first.commit() == second.commit());
    // Corruption after publication, including same-size damage, must not load.
    for (const bool truncate : {false, true}) {
        const auto test_root = root / (truncate ? "truncated" : "corrupt");
        configure(test_root);
        Runtime<FakeBackend> writer(Config(root, "FAULT"));
        const auto original = writer.compile_without_load("kernel", "contents");
        write_file_sync(original / "binary", truncate ? "con" : "xxxxxxxx");
        Runtime<FakeBackend> reader(Config(root, "FAULT"));
        assert(reader.compile("kernel", "contents")->data == "contents");
        assert(std::filesystem::exists(test_root / "secondary/cache"));
    }

    // Lookup succeeds; the subsequent load read fails halfway.
    for (int mode : {1, 2, 3}) {
        const auto test_root = root / ("load_" + std::to_string(mode));
        configure(test_root);
        Runtime<FakeBackend> writer(Config(root, "FAULT"));
        writer.compile_without_load("kernel", "contents");
        load_fault = mode;
        Runtime<FakeBackend> reader(Config(root, "FAULT"));
        assert(reader.compile("kernel", "contents")->data == "contents");
        assert(std::filesystem::exists(test_root / "secondary/cache"));
        if (mode == 2) {
            release_read = true;
            while (not read_done) std::this_thread::sleep_for(1ms);
        }
        load_fault = 0;
    }

    // Timeout halfway through writing; the abandoned task owns its inputs even
    // after Runtime destruction and never touches the fallback's temporary.
    const auto write_root = root / "partial_write";
    configure(write_root);
    partial_write = true;
    {
        Runtime<FakeBackend> writer(Config(root, "FAULT"));
        assert(writer.compile("kernel", "contents")->data == "contents");
        assert(std::filesystem::exists(write_root / "secondary/cache"));
    }
    release_write = true;
    while (not write_done) std::this_thread::sleep_for(1ms);
    partial_write = false;

    // A failed request must not poison a later request on the same Runtime.
    configure(root / "transient");
    Runtime<FakeBackend> transient(Config(root, "FAULT"));
    fail_all = true;
    expect_io_error([&] { transient.compile("kernel", "contents"); });
    fail_all = false;
    assert(transient.compile("kernel", "contents")->data == "contents");

    // A failed hit on root 2 advances to root 3, not back to root 1's miss.
    const auto ordered_root = root / "ordered";
    const auto ordered_dirs = (ordered_root / "first").string() + ":" +
        (ordered_root / "second").string() + ":" + (ordered_root / "third").string();
    ::setenv("FAULT_JIT_CACHE_DIR", ordered_dirs.c_str(), 1);
    Runtime<FakeBackend> ordered(Config(root, "FAULT"));
    const auto ordered_key = ordered.cache_key("contents", {});
    DiskCache seed({ordered_root / "second"});
    auto seeded = seed.entry("kernel", ordered_key);
    write_file_sync(seeded.path / "binary", "contents");
    const auto seeded_path = seeded.commit();
    write_file_sync(seeded_path / "binary", "bad");
    assert(ordered.compile("kernel", "contents")->data == "contents");
    assert(std::filesystem::exists(ordered_root / "third/cache" / ("kernel." + ordered_key)));
    assert(not std::filesystem::exists(ordered_root / "first/tmp"));

    // A rename collision is benign only if the winner is complete.
    DiskCache corrupt_winner({root / "corrupt_winner"});
    auto winner = corrupt_winner.entry("kernel", "key");
    auto loser = corrupt_winner.entry("kernel", "key");
    write_file_sync(winner.path / "binary", "same");
    write_file_sync(loser.path / "binary", "same");
    const auto winning_path = winner.commit();
    write_file_sync(winning_path / "binary", "bad");
    assert(loser.commit() == winning_path); // the rename race itself succeeds
    expect_io_error([&] { FakeBackend::load(winning_path, Env("FAULT")); });

    // Synchronous compiler processes are independent of the 100 ms IO timeout.
    configure(root / "slow_compile");
    compile_mode = 1;
    Runtime<FakeBackend> slow_compile(Config(root, "FAULT"));
    const auto slow_path = slow_compile.compile_without_load("kernel", "contents");
    assert(slow_path.string().find("/primary/") != std::string::npos);
    std::cout << "PASS: 200 ms compiler exceeds 100 ms IO deadline without fallback\n";

    configure(root / "compile_failure");
    compile_mode = 2;
    Runtime<FakeBackend> compile_failure(Config(root, "FAULT"));
    const auto failure_path = compile_failure.compile_without_load("kernel", "contents");
    assert(failure_path.string().find("/secondary/") != std::string::npos);
    assert(compile_errors == 1);
    std::cout << "PASS: compiler failure falls back with its exit-code diagnostic\n";

    // Both subprocesses must start before either can finish. A serial IO worker
    // cannot pass this barrier. Use independent runtimes, as in separate clients.
    configure(root / "parallel_compile");
    compile_mode = 3;
    compile_barrier = root / "compile_barrier";
    make_dirs(compile_barrier);
    Runtime<FakeBackend> parallel_one(Config(root, "FAULT")), parallel_two(Config(root, "FAULT"));
    auto one = std::async(std::launch::async, [&] { return parallel_one.compile_without_load("kernel", "one"); });
    auto two = std::async(std::launch::async, [&] { return parallel_two.compile_without_load("kernel", "two"); });
    assert(one.get().string().find("/primary/") != std::string::npos);
    assert(two.get().string().find("/primary/") != std::string::npos);
    compile_mode = 0;
    std::cout << "PASS: two compiler subprocesses pass a mutual start barrier concurrently\n";

    std::cout << "Cache fault injection tests passed\n";
}
