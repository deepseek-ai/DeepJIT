#pragma once

#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <fcntl.h>
#include <unistd.h>

#include <deep_jit/utils/exception.hpp>

namespace deep_jit {

inline std::string read(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    DJ_HOST_ASSERT(input.is_open(), "failed to open for reading: {}", path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

inline void fsync_file(const std::filesystem::path& path) {
    const auto fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        const std::error_code error(errno, std::generic_category());
        DJ_PANIC("failed to open for fsync: {}: {}", path.string(), error.message());
    }
    if (::fsync(fd) != 0) {
        const std::error_code error(errno, std::generic_category());
        ::close(fd);
        DJ_PANIC("failed to fsync: {}: {}", path.string(), error.message());
    }
    if (::close(fd) != 0) {
        const std::error_code error(errno, std::generic_category());
        DJ_PANIC("failed to close after fsync: {}: {}", path.string(), error.message());
    }
}

// Shared cache access, independent of umask; directories retain inherited SGID.
inline void set_shared_permissions(const std::filesystem::path& path, const bool directory, const bool shared) {
    if (not shared)
        return;
    const auto sgid = directory ? std::filesystem::status(path).permissions() & std::filesystem::perms::set_gid
                                : std::filesystem::perms::none;
    const auto mode = static_cast<std::filesystem::perms>(directory ? 0775 : 0664) | sgid;
    std::error_code error;
    std::filesystem::permissions(path, mode, std::filesystem::perm_options::replace, error);
    if (error)
        DJ_PANIC("failed to set permissions: {}: {}", path.string(), error.message());
}

// Apply shared permissions to compiler outputs too, then fsync bottom-up.
inline void fsync_dir(const std::filesystem::path& dir_path, const bool shared) {  // NOLINT(*-no-recursion)
    set_shared_permissions(dir_path, true, shared);
    for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
        const auto status = entry.symlink_status();
        if (std::filesystem::is_directory(status))
            fsync_dir(entry.path(), shared);
        else if (std::filesystem::is_regular_file(status)) {
            set_shared_permissions(entry.path(), false, shared);
            fsync_file(entry.path());
        }
    }
    fsync_file(dir_path);
}

inline void make_dirs(const std::filesystem::path& path, const bool shared = false) {
    // OK if already exists
    std::error_code error_code;
    const bool created = std::filesystem::create_directories(path, error_code);
    if (not (created or error_code.value() == 0))
        DJ_PANIC("failed to create directory: {}", path.string());
    if (created)
        set_shared_permissions(path, true, shared);
}

// Best-effort cache-access bookkeeping. Failure to update an mtime (for
// example, on a read-only lookup cache) must never turn a valid cache hit into
// a miss or error. Unlike cache publication, this does not create or modify
// file contents.
inline bool try_update_mtime(const std::filesystem::path& path) noexcept {
    std::error_code error_code;
    std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now(), error_code);
    return not error_code;
}

// Write a file and fsync it: on distributed filesystems `close()` alone does
// not guarantee that other processes can see the data.
// Shared cache writes use 0664; other writes retain normal creation permissions.
inline void write_file_sync(const std::filesystem::path& path, const std::string_view& data, const bool shared = false) {
    std::ofstream output(path, std::ios::binary);
    if (not output)
        DJ_PANIC("failed to open for writing: {}", path.string());
    if (not output.write(data.data(), static_cast<std::streamsize>(data.size())))
        DJ_PANIC("failed to write: {}", path.string());
    output.close();
    if (not output)
        DJ_PANIC("failed to close after writing: {}", path.string());
    set_shared_permissions(path, false, shared);
    fsync_file(path);
}

// Remove a directory tree without `std::filesystem::remove_all`, which can
// segfault on distributed filesystems when concurrent processes operate on the
// same parent directory.
inline void safe_remove_all(const std::filesystem::path& path) {
    std::error_code error_code;
    if (not std::filesystem::exists(path, error_code) or error_code)
        return;

    // A single file
    if (not std::filesystem::is_directory(path, error_code) or error_code) {
        std::filesystem::remove(path, error_code);
        return;
    }

    auto iterator = std::filesystem::directory_iterator(path, std::filesystem::directory_options::skip_permission_denied, error_code);
    for (const auto end = std::filesystem::directory_iterator(); iterator != end and not error_code;) {
        const auto entry_path = iterator->path();
        iterator.increment(error_code);  // advance before recursing
        if (error_code)
            break;
        safe_remove_all(entry_path);
    }
    std::filesystem::remove(path, error_code);
}

inline std::optional<std::filesystem::path> normalize_path(const std::optional<std::filesystem::path>& path) {
    if (not path or path->empty())
        return std::nullopt;
    return std::filesystem::absolute(*path).lexically_normal();
}

inline bool is_executable(const std::filesystem::path& path) {
    return std::filesystem::is_regular_file(path) and ::access(path.c_str(), X_OK) == 0;
}

}  // namespace deep_jit
