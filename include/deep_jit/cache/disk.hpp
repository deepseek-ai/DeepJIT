#pragma once

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/filesystem.hpp>
#include <deep_jit/utils/executor.hpp>
#include <deep_jit/utils/uuid.hpp>

namespace deep_jit {

inline constexpr std::string_view kCommitFileName = ".committed";

// One disk-cache slot. On a miss the entry points at a private temporary
// directory; commit() publishes it atomically (whole-directory rename) after
// fsync, so concurrent processes on (distributed) filesystems observe either
// nothing or a complete entry. Uncommitted temporaries are removed on
// destruction unless abandoned after an I/O timeout.
class DiskCacheEntry {
    const bool shared;

public:
    bool hit = false, committed = false, abandoned = false;
    size_t root_index = 0;

    // `path` is the temporary path while building, commit path after commit
    std::filesystem::path path;
    std::filesystem::path commit_path;

    DiskCacheEntry(const DiskCacheEntry&) = delete;
    DiskCacheEntry& operator=(const DiskCacheEntry&) = delete;
    DiskCacheEntry(DiskCacheEntry&&) = delete;
    DiskCacheEntry& operator=(DiskCacheEntry&&) = delete;

    DiskCacheEntry(const bool hit, std::filesystem::path path, std::filesystem::path commit_path = {}, const bool shared = false, size_t root_index = 0)
        : shared(shared), hit(hit), root_index(root_index), path(std::move(path)), commit_path(std::move(commit_path)) {}

    std::filesystem::path commit() {
        if (hit or committed)
            return path;

        try {
            run_in_iothread([source = path, target = commit_path, shared = shared] {
                write_file_sync(source / kCommitFileName, "", shared);
                fsync_dir(source, shared);
                make_dirs(target.parent_path(), shared);
                std::error_code error;
                std::filesystem::rename(source, target, error);
                if (error) {
                    if ((error != std::errc::file_exists and error != std::errc::directory_not_empty) or
                        not std::filesystem::exists(target / kCommitFileName))
                        DJ_PANIC("cache rename failed: {}", error.message());
                    safe_remove_all(source);
                }
            });
        } catch (const IOTimeoutError&) {
            abandoned = true; // a timed-out task may still be using this directory
            throw;
        }

        path = commit_path;
        hit = true;
        committed = true;
        return path;
    }

    ~DiskCacheEntry() {
        // Best-effort cleanup; abrupt process termination may bypass the destructor.
        if (not hit and not committed and not abandoned)
            try { run_in_iothread([path = path] { safe_remove_all(path); }); }
            catch (...) {}
    }
};

// Directory-granularity cache on a possibly distributed filesystem.
// Entries live at `<root>/cache/<tag>.<digest>/` and are published with a
// ".committed" marker; builds go through `<root>/tmp/<uuid>`.
struct DiskCache {
    const std::vector<std::filesystem::path> paths = {};

private:
    mutable std::vector<std::atomic<unsigned>> num_timeouts;

public:
    explicit DiskCache(std::vector<std::filesystem::path> paths)
        : paths(std::move(paths)), num_timeouts(this->paths.size()) {
        DJ_HOST_ASSERT(not this->paths.empty());
    }

    void report_error(size_t root, const std::exception& error) const {
        std::fprintf(stderr, "DeepJIT cache warning: %s: %s; trying next root\n",
                     paths[root].c_str(), error.what());
        if (dynamic_cast<const IOTimeoutError*>(&error) and ++num_timeouts[root] == 3)
            std::fprintf(stderr, "DeepJIT cache warning: %s: disabled after 3 IO timeouts\n",
                         paths[root].c_str());
    }

    static DiskCache from_env(const Env& env) {
        // Parse roots in preference order. Later roots also serve as write fallbacks.
        std::vector<std::filesystem::path> paths;
        if (const auto env_opt = env.get<std::string>("JIT_CACHE_DIR")) {
            size_t begin = 0;
            while (true) {
                const auto end = env_opt->find(':', begin);
                const auto item = env_opt->substr(begin, end - begin);
                DJ_HOST_ASSERT(not item.empty(), "disk cache path list contains an empty path: {}", env_opt.value());
                paths.emplace_back(item);
                if (end == std::string_view::npos)
                    break;
                begin = end + 1;
            }
        } else {
            const auto home = get_env<std::string>("HOME");
            DJ_HOST_ASSERT(not home.empty(), "HOME environment variable must not be empty");
            paths.emplace_back(std::filesystem::path(home) / ".dj");
        }

        return DiskCache(paths);
    }

    [[nodiscard]] DiskCacheEntry entry(const std::string& tag, const std::string& digest, size_t first_root = 0) const {
        const auto is_valid_char = [](const char& c) {
            return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or
                   (c >= '0' and c <= '9') or c == '_';
        };
        DJ_HOST_ASSERT(not tag.empty() and std::ranges::all_of(tag, is_valid_char),
                       "cache tag must contain only letters, digits, or underscores: {}", tag);

        // Lookup failures must not be retried as writes in this attempt.
        std::vector<size_t> failed;
        const auto entry_name = std::format("{}.{}", tag, digest);
        for (size_t i = first_root; i < paths.size(); ++i) {
            if (num_timeouts[i] >= 3) continue;
            const auto path = paths[i] / "cache" / entry_name;
            try {
                const bool hit = run_in_iothread([path] {
                    const auto marker = path / kCommitFileName;
                    if (not std::filesystem::exists(marker)) return false;
                    try_update_mtime(marker);
                    return true;
                });
                if (hit)
                    return DiskCacheEntry{true, path, {}, false, i};
            } catch (const std::exception& error) {
                if (detail::disable_cache_fallback())
                    throw;
                failed.push_back(i);
                report_error(i, error);
            }
        }

        for (size_t i = first_root; i < paths.size(); ++i) {
            if (num_timeouts[i] >= 3 or
                std::ranges::find(failed, i) != failed.end()) continue;
            const auto temporary_path = paths[i] / "tmp" / get_uuid();
            try {
                const bool shared = run_in_iothread([root = paths[i], temporary_path] {
                    std::error_code error;
                    const auto status = std::filesystem::status(root, error);
                    if (error and error != std::errc::no_such_file_or_directory)
                        DJ_PANIC("cache root status failed: {}", error.message());
                    const bool shared = std::filesystem::exists(status) and
                        (status.permissions() & std::filesystem::perms::set_gid) != std::filesystem::perms::none;
                    make_dirs(temporary_path.parent_path(), shared);
                    make_dirs(temporary_path, shared);
                    return shared;
                });
                return DiskCacheEntry{false, temporary_path, paths[i] / "cache" / entry_name, shared, i};
            } catch (const std::exception& error) {
                if (detail::disable_cache_fallback())
                    throw;
                report_error(i, error);
            }
        }
        DJ_PANIC("all cache roots failed");
    }
};

}  // namespace deep_jit
