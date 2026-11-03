#define DJ_DISABLE_GIL 1
#include <deep_jit/runtime/runtime.hpp>
#include <deep_jit/utils/command.hpp>

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <cstdarg>
#include <dirent.h>
#include <dlfcn.h>
#include <iostream>
#include <mutex>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>

using namespace deep_jit;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

// Each scenario runs in a separate process. Permanent stalls are never released.
// This models a syscall which does not return; it does not inject kernel D-state.
static std::string scenario;
static std::atomic<bool> armed{false};
static std::atomic<int> injections{0}, blocked{0}, builds{0};
static std::atomic<bool> partial{false};
static std::atomic<int> primary_io{0}, primary_cleanup{0};
static std::atomic<int> root_fault{0}; // 1: read stall, 2: commit stall, 3: compiler error, 4: EIO, 5: write fsync stall
static const std::string binary = "FAKE-BINARY-v1\n" + std::string(32768, 'b') + "\nEND\n";
static const std::string metadata = "FAKE-METADATA-v1\n";

static std::string fd_path(int fd) {
    char path[4096];
    const auto count = ::readlink(("/proc/self/fd/" + std::to_string(fd)).c_str(), path, sizeof(path));
    return count < 0 ? "" : std::string(path, static_cast<size_t>(count));
}
static bool primary(const std::string& path) {
    return path.find("/primary/") != std::string::npos or path.ends_with("/primary");
}
static bool target(int fd) {
    if (not armed) return false;
    const auto path = fd_path(fd);
    return primary(path) and path.ends_with("/binary");
}
[[noreturn]] static void hang() {
    ++injections;
    ++blocked;
    // Heap lifetime is intentional for a never-returning test thread.
    auto* mutex = new std::mutex;
    auto* cv = new std::condition_variable;
    std::unique_lock lock(*mutex);
    cv->wait(lock, [] { return false; });
    std::abort();
}

// Enumerate actual calls first; replay with exactly one numbered call failing.
static int site_number = 0, selected_site = 0;
static std::string site_fault, selected_operation, previous_operation, previous_path;
static bool retryable_parent_probe = false;
static bool site(const char* operation, const std::string& path) {
    if (armed and scenario.starts_with("root_") and primary(path)) {
        ++primary_io;
        if (std::string_view(operation) == "remove") ++primary_cleanup;
    }
    if (not armed or not scenario.starts_with("sites_") or not primary(path)) return false;
    const auto number = ++site_number;
    std::fprintf(stderr, "SITE\t%d\t%s\t%s\n", number, operation, path.c_str());
    const bool parent_probe = std::string_view(operation) == "stat" and previous_operation == "stat" and
        fs::path(previous_path).parent_path() == path;
    previous_operation = operation;
    previous_path = path;
    if (number != selected_site) return false;
    selected_operation = operation;
    struct stat info{};
    retryable_parent_probe = parent_probe and
        ::syscall(SYS_newfstatat, AT_FDCWD, path.c_str(), &info, 0) == 0;
    std::fprintf(stderr, "HIT\t%d\t%s\t%s\n", number, operation, path.c_str());
    if (site_fault == "hang") hang();
    ++injections;
    errno = EIO;
    return true;
}
#include "sites.hpp"

extern "C" ssize_t read(int fd, void* data, size_t size) {
    if (site("read", fd_path(fd))) return -1;
    if (not target(fd)) return ::syscall(SYS_read, fd, data, size);
    if (root_fault == 1 or scenario == "hang_read") hang();
    if (scenario == "hang_read_after_prefix") {
        if (partial.exchange(true)) hang();
        return ::syscall(SYS_read, fd, data, std::min(size, size_t(17)));
    }
    if (scenario == "read_eio_after_prefix") {
        if (partial.exchange(true)) { ++injections; errno = EIO; return -1; }
        return ::syscall(SYS_read, fd, data, std::min(size, size_t(17)));
    }
    const auto result = ::syscall(SYS_read, fd, data, size);
    if (scenario.starts_with("drop_read_") and result > 3) {
        const size_t count = result, drop = count / 3;
        const size_t start = scenario == "drop_read_prefix" ? 0 :
            scenario == "drop_read_middle" ? count / 3 : count - drop;
        auto* bytes = static_cast<char*>(data);
        std::memmove(bytes + start, bytes + start + drop, count - start - drop);
        ++injections;
        return count - drop;
    }
    if (scenario == "corrupt_read" and result > 0) {
        static_cast<char*>(data)[0] ^= 1;
        ++injections;
    }
    return result;
}

static ssize_t write_data(int fd, const void* data, size_t size) {
    if (site("write", fd_path(fd))) return -1;
    if (not target(fd)) return ::syscall(SYS_write, fd, data, size);
    if (scenario == "hang_write") hang();
    if (scenario == "hang_write_after_prefix") {
        ::syscall(SYS_write, fd, data, std::min(size, size_t(17)));
        hang();
    }
    if (scenario == "write_enospc_after_prefix") {
        if (partial.exchange(true)) { ++injections; errno = ENOSPC; return -1; }
        return ::syscall(SYS_write, fd, data, std::min(size, size_t(17)));
    }
    if (scenario == "short_write_retried" and not partial.exchange(true)) {
        ++injections;
        return ::syscall(SYS_write, fd, data, std::min(size, size_t(17)));
    }
    if (scenario.starts_with("drop_write_") and size > 3) {
        const size_t drop = size / 3;
        const size_t start = scenario == "drop_write_prefix" ? 0 :
            scenario == "drop_write_middle" ? size / 3 : size - drop;
        const auto* bytes = static_cast<const char*>(data);
        if (start) assert(::syscall(SYS_write, fd, bytes, start) == static_cast<ssize_t>(start));
        if (size > start + drop)
            assert(::syscall(SYS_write, fd, bytes + start + drop, size - start - drop) ==
                   static_cast<ssize_t>(size - start - drop));
        ++injections;
        return size; // filesystem reports success despite losing part of the data
    }
    return ::syscall(SYS_write, fd, data, size);
}
extern "C" ssize_t write(int fd, const void* data, size_t size) {
    return write_data(fd, data, size);
}
extern "C" ssize_t writev(int fd, const iovec* vectors, int count) {
    if (site("writev", fd_path(fd))) return -1;
    if (scenario.starts_with("sites_") or not target(fd)) return ::syscall(SYS_writev, fd, vectors, count);
    std::string data;
    for (int i = 0; i < count; ++i)
        data.append(static_cast<const char*>(vectors[i].iov_base), vectors[i].iov_len);
    return write_data(fd, data.data(), data.size());
}
extern "C" int __wrap_fsync(int fd) {
    const auto path = fd_path(fd);
    if (site("fsync", path)) return -1;
    if (armed and (primary(path) or scenario == "all_roots_hang")) {
        if ((root_fault == 2 and path.ends_with("/.committed")) or
            (root_fault == 5 and path.ends_with("/binary"))) hang();
        if (root_fault == 4) { ++injections; errno = EIO; return -1; }
        if (scenario == "hang_fsync" or scenario == "all_roots_hang") hang();
        if (scenario == "hang_commit" and path.ends_with("/.committed")) hang();
        if (scenario == "fsync_eio" or scenario == "fsync_enospc" or scenario == "fsync_erofs") {
            ++injections;
            errno = scenario == "fsync_eio" ? EIO : scenario == "fsync_enospc" ? ENOSPC : EROFS;
            return -1;
        }
    }
    return ::syscall(SYS_fsync, fd);
}
extern "C" int mkdir(const char* path, mode_t mode) noexcept {
    static const auto real = reinterpret_cast<decltype(&::mkdir)>(::dlsym(RTLD_NEXT, "mkdir"));
    if (site("mkdir", path)) return -1;
    if (armed and primary(path)) {
        if (scenario == "hang_mkdir") hang();
        if (scenario == "mkdir_eacces") { ++injections; errno = EACCES; return -1; }
    }
    return real(path, mode);
}
extern "C" int stat(const char* path, struct stat* info) noexcept {
    static const auto real = reinterpret_cast<decltype(&::stat)>(::dlsym(RTLD_NEXT, "stat"));
    if (site("stat", path)) return -1;
    if (armed and primary(path) and std::string_view(path).ends_with("/.committed")) {
        if (scenario == "hang_lookup") hang();
        if (scenario == "lookup_eio") { ++injections; errno = EIO; return -1; }
    }
    return real(path, info);
}
extern "C" int rename(const char* source, const char* destination) noexcept {
    static const auto real = reinterpret_cast<decltype(&::rename)>(::dlsym(RTLD_NEXT, "rename"));
    if (site("rename", source)) return -1;
    if (armed and primary(source)) {
        if (scenario == "hang_rename") hang();
        if (scenario == "rename_eio") { ++injections; errno = EIO; return -1; }
        if (scenario.starts_with("rename_collision")) {
            ++injections;
            const fs::path winner(destination);
            make_dirs(winner);
            write_file_sync(winner / "binary", scenario == "rename_collision_bad_winner" ? "bad" : binary);
            write_file_sync(winner / "metadata", metadata);
            if (scenario != "rename_collision_uncommitted")
                write_file_sync(winner / ".committed", "");
        }
    }
    return real(source, destination);
}

struct Backend {
    struct Device {};
    struct Kernel { std::string data; fs::path path; };
    struct Options {
        template<class... Args> static Options default_options(const Args&...) { return {}; }
        Options override_with(const Options&) const { return {}; }
        void update_hash(hash::FNV1a&) const {}
        std::string get_post_hook_hash(const Config&) const { return {}; }
    };
    using CompilerOptions = Options;
    using LaunchOptions = Options;
    struct Info { std::string get_hash() const { return "fault-injection"; } } compiler_info;
    explicit Backend(const Env&) {}
    void compile(const std::string&, const fs::path& path, const Env&, const Config&, const Options&) const {
        ++builds;
        if (armed and primary(path.string()) and root_fault == 3)
            call_external_command("exit 7");
        if (armed and primary(path.string()) and scenario.starts_with("command_")) {
            const std::string command = scenario == "command_fifo" ?
                "mkfifo " + (path / "compiler.fifo").string() + "; exec cat " + (path / "compiler.fifo").string() :
                scenario == "command_closed_output" ? "exec 1>&- 2>&-; exec sleep infinity" :
                scenario == "command_active_output" ? "while :; do echo progress; sleep 0.01; done" :
                "exec sleep infinity";
            ++injections;
            call_external_command(command, false, 100ms);
        }
        run_in_iothread([path] {
            write_file_sync(path / "binary", binary);
            write_file_sync(path / "metadata", metadata);
        });
    }
    static std::shared_ptr<Kernel> load(const fs::path& path, const Env&) {
        auto data = run_in_iothread([path] {
            const auto content = read_bytes(path / "binary");
            DJ_HOST_ASSERT(read_bytes(path / "metadata") == metadata, "invalid metadata");
            return content;
        });
        // This models backend format validation, not a production checksum.
        DJ_HOST_ASSERT(data == binary, "invalid binary image");
        return std::make_shared<Kernel>(Kernel{std::move(data), path});
    }
};
using TestRuntime = Runtime<Backend>;

static void configure(const fs::path& root) {
    const auto dirs = (root / "primary").string() + ":" + (root / "secondary").string();
    assert(::setenv("FAULT_JIT_CACHE_DIR", dirs.c_str(), 1) == 0);
}
static void seed(const fs::path& path, bool complete = true) {
    make_dirs(path);
    if (complete) {
        write_file_sync(path / "binary", binary);
        write_file_sync(path / "metadata", metadata);
    }
    write_file_sync(path / ".committed", "");
}

static void test_root_timeouts(const fs::path& root, const fs::path& primary_path) {
    TestRuntime runtime(Config(root, "FAULT"));
    auto request = [&](const std::string& source = "source", bool load = true) {
        runtime.mem_cache.cache.clear(); // force another disk request on the same Runtime
        return load ? runtime.compile("kernel", source)->path : runtime.compile_without_load("kernel", source);
    };
    if (scenario == "root_read_timeouts") {
        seed(primary_path);
        armed = true;
        root_fault = 1;
        for (int i = 0; i < 2; ++i) {
            assert(not primary(request().string()));
            assert(blocked == i + 1);
            assert(primary(request("source", false).string())); // marker hits do not clear the count
        }
        root_fault = 0;
        assert(primary(request().string())); // a full successful load also does not clear the count
        root_fault = 1;
        assert(not primary(request().string()));
        assert(blocked == 3);
        const auto io_before_skip = primary_io.load();
        root_fault = 0;
        assert(not primary(request().string())); // remains disabled even after the filesystem recovers
        assert(not primary(request("new source").string())); // skipped for writes as well
        assert(primary_io == io_before_skip);
        std::cout << "PASS: disabled root receives zero IO on later reads and writes\n";
        TestRuntime fresh(Config(root, "FAULT"));
        assert(primary(fresh.compile("kernel", "source")->path.string())); // a new Runtime starts fresh
        assert(not primary(request().string())); // does not reset the original Runtime's count
    } else if (scenario == "root_commit_timeouts" or scenario == "root_write_timeouts") {
        armed = true;
        root_fault = scenario == "root_commit_timeouts" ? 2 : 5;
        for (int i = 0; i < 3; ++i) {
            assert(not primary(request("source" + std::to_string(i), false).string()));
            assert(blocked == i + 1);
            assert(primary_cleanup == 0); // a timed-out task still owns the temporary directory
        }
        const auto io_before_skip = primary_io.load();
        assert(not primary(request("another source", false).string()));
        assert(primary_io == io_before_skip);
        assert(blocked == 3); // no fourth IO task on the disabled root
    } else if (scenario == "root_other_errors") {
        armed = true;
        for (const int fault : {3, 4}) {
            root_fault = fault;
            for (int i = 0; i < 3; ++i)
                assert(not primary(request(std::to_string(fault) + ":" + std::to_string(i)).string()));
            assert(fs::is_empty(root / "primary/tmp")); // non-timeout failures clean up temporary directories
            root_fault = 0;
            assert(primary(request("healthy" + std::to_string(fault)).string()));
        }
        assert(blocked == 0); // compiler failure and EIO do not disable a root
    } else if (scenario == "root_queue_wait") {
        seed(primary_path);
        armed = true;
        for (int i = 0; i < 3; ++i) {
            auto started = std::make_shared<std::promise<void>>();
            auto ready = started->get_future();
            auto owner = std::async(std::launch::async, [started] {
                try { run_in_iothread([started] { started->set_value(); hang(); }); assert(false); }
                catch (const IOTimeoutError&) {}
            });
            ready.get();
            // Waiting behind unrelated blocked IO must not fail a healthy root.
            assert(primary(request().string()));
            owner.get();
        }
        assert(primary(request().string())); // waiting must not disable this healthy root
    } else if (scenario == "root_concurrent_reports") {
        seed(primary_path);
        const IOTimeoutError error;
        std::vector<std::thread> reporters;
        for (int i = 0; i < 4; ++i)
            reporters.emplace_back([&] { runtime.disk_cache.report_error(0, error); });
        for (auto& thread : reporters) thread.join();
        armed = true;
        assert(not primary(request().string()));
        assert(primary_io == 0); // atomic counts remain disabled even when they exceed three
    } else if (scenario == "root_all_disabled") {
        const auto first = root / "first/primary", second = root / "second/primary";
        const auto dirs = first.string() + ":" + second.string();
        assert(::setenv("FAULT_JIT_CACHE_DIR", dirs.c_str(), 1) == 0);
        TestRuntime runtime(Config(root, "FAULT"));
        const auto key = runtime.cache_key("source", {});
        seed(first / "cache" / ("kernel." + key));
        seed(second / "cache" / ("kernel." + key));
        auto warm = runtime.compile("kernel", "warm source");
        armed = true;
        root_fault = 1;
        for (int i = 0; i < 3; ++i) {
            try { runtime.compile("kernel", "source"); assert(false); }
            catch (const IOTimeoutError&) {}
            assert(blocked == 2 * (i + 1));
        }
        const auto io_before_skip = primary_io.load();
        try { runtime.compile("kernel", "source"); assert(false); }
        catch (const exception::Exception& error) {
            assert(std::string(error.what()).find("all cache roots failed") != std::string::npos);
        }
        assert(blocked == 6);
        assert(primary_io == io_before_skip);
        assert(runtime.compile("kernel", "warm source") == warm);
    }
    std::cout << "PASS: " << scenario << "; permanently blocked=" << blocked << "\n";
}

int main(int argc, char** argv) {
    assert(argc == 3 or argc == 5);
    if (argc == 5) { selected_site = std::stoi(argv[3]); site_fault = argv[4]; }
    scenario = argv[1];
    const fs::path root(argv[2]);
    make_dirs(root);
    configure(root);
    TestRuntime runtime(Config(root, "FAULT"));
    const auto key = runtime.cache_key("source", {});
    const auto primary_path = root / "primary/cache" / ("kernel." + key);
    const auto secondary_path = root / "secondary/cache" / ("kernel." + key);

    if (scenario == "disable_fallback") {
        assert(::setenv("DJ_DISABLE_CACHE_FALLBACK", "1", 1) == 0);
        assert(detail::disable_cache_fallback());
        const auto caller = std::this_thread::get_id();
        assert(run_in_iothread([] { return std::this_thread::get_id(); }) == caller);

        // Normal build and secondary lookup hit work inline without starting a worker.
        auto built = runtime.compile("kernel", "healthy_source");
        assert(built->data == binary and primary(built->path.string()));
        seed(secondary_path);
        runtime.mem_cache.cache.clear();
        auto hit = runtime.compile("kernel", "source");
        assert(hit->data == binary and hit->path == secondary_path);
        fs::remove_all(secondary_path);

        // Lookup, write, and compile errors fail immediately without falling back.
        armed = true;
        for (const int fault : {3, 4}) {
            root_fault = fault;
            runtime.mem_cache.cache.clear();
            bool failed = false;
            try { runtime.compile("kernel", "fail_" + std::to_string(fault)); }
            catch (const std::exception&) { failed = true; }
            assert(failed);
        }
        assert(not fs::exists(root / "secondary/cache") or fs::is_empty(root / "secondary/cache"));
        std::cout << "PASS: disable_fallback runs inline and does not fall back across roots\n";
        return 0;
    }
    if (scenario.starts_with("root_")) {
        test_root_timeouts(root, primary_path);
        return 0;
    }
    if (scenario == "sites_hit") seed(primary_path);
    if (scenario == "sites_cleanup") {
        // Exercise uncommitted entry cleanup directly, including cleanup faults.
        armed = true;
        {
            auto entry = runtime.disk_cache.entry("kernel", key);
            try {
                run_in_iothread([path = entry.path] { write_file_sync(path / "binary", binary); });
            } catch (const IOTimeoutError&) {
                entry.abandoned = true;
            } catch (const exception::Exception&) {}
        }
        assert(injections == (selected_site ? 1 : 0));
        assert(blocked == (site_fault == "hang" ? 1 : 0));
        assert(run_in_iothread([] { return 42; }) == 42);
        std::cout << "PASS: uncommitted entry cleanup; site=" << selected_site << "\n";
        return 0;
    }
    if (scenario == "sites_shared") {
        make_dirs(root / "primary");
        fs::permissions(root / "primary", fs::perms::set_gid, fs::perm_options::add);
    }

    if (scenario.starts_with("drop_read_") or scenario == "corrupt_read" or
        scenario.starts_with("hang_read") or scenario == "read_eio_after_prefix")
        seed(primary_path);
    if (scenario == "missing_binary" or scenario == "missing_metadata" or
        scenario == "empty_binary" or scenario == "truncated_binary" or scenario == "stale_metadata") {
        seed(primary_path);
        if (scenario == "missing_binary") fs::remove(primary_path / "binary");
        if (scenario == "missing_metadata") fs::remove(primary_path / "metadata");
        if (scenario == "empty_binary") write_file_sync(primary_path / "binary", "");
        if (scenario == "truncated_binary") write_file_sync(primary_path / "binary", binary.substr(0, binary.size() / 2));
        if (scenario == "stale_metadata") write_file_sync(primary_path / "metadata", "old metadata");
    }
    if (scenario == "all_roots_hang") {
        auto warm = runtime.compile("kernel", "warm source");
        armed = true;
        const auto start = std::chrono::steady_clock::now();
        try { runtime.compile("kernel", "source"); assert(false); }
        catch (const IOTimeoutError& error) {
            assert(std::string(error.what()).find("timed out") != std::string::npos);
        }
        assert(std::chrono::steady_clock::now() - start < 2s);
        assert(blocked == 2);
        assert(runtime.compile("kernel", "warm source") == warm);
        std::cout << "PASS: both roots hang permanently; request fails and memory hit remains usable\n";
        return 0;
    }

    armed = true;
    const auto pid = ::getpid();
    const auto start = std::chrono::steady_clock::now();
    auto kernel = runtime.compile("kernel", "source");
    assert(::getpid() == pid); // recovery occurs in the original caller process
    assert(std::chrono::steady_clock::now() - start < 2s);
    assert(kernel->data == binary);
    if (scenario.starts_with("sites_")) {
        assert(injections == (selected_site ? 1 : 0));
        // mtime/read/directory close errors are ignored by the existing code.
        // create_directories can recover from a failed parent probe by mkdir.
        const bool best_effort = site_fault == "error" and
            (selected_operation == "utimensat" or selected_operation == "fclose_read" or
             selected_operation == "closedir" or retryable_parent_probe);
        const bool fallback = selected_site and not best_effort;
        assert(kernel->path == (fallback ? secondary_path : primary_path));
        if (site_fault == "hang") {
            assert(blocked == 1);
            assert(run_in_iothread([] { return 42; }) == 42);
        }
        armed = false;
        assert(runtime.compile("kernel", "source") == kernel);
        std::cout << "PASS: " << scenario << "; site=" << selected_site
                  << "; result=" << (fallback ? "secondary" : "primary") << "\n";
        return 0;
    }
    const bool healthy_primary = scenario == "short_write_retried" or scenario == "rename_collision";
    assert(kernel->path == (healthy_primary ? primary_path : secondary_path));
    if (not scenario.starts_with("missing_") and scenario != "empty_binary" and
        scenario != "truncated_binary" and scenario != "stale_metadata")
        assert(injections > 0); // a passing test must have actually reached its injection point
    if (scenario.starts_with("command_"))
        assert(not fs::is_empty(root / "primary/tmp")); // killed commands may still be in uninterruptible IO
    if (scenario.starts_with("hang_")) {
        assert(blocked == 1);
        assert(run_in_iothread([] { return 42; }) == 42);
        // Do not release the blocked operation; it remains stuck until process exit.
    }
    assert(runtime.compile("kernel", "source") == kernel);
    std::cout << "PASS: " << scenario << "; " << (healthy_primary ? "primary retained" : "fallback to secondary")
              << "; injections=" << injections << "; permanently blocked=" << blocked << "\n";
}
