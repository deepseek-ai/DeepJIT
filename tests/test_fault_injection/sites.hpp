// Extra libc interception for the numbered-site sweep. Only primary-cache
// paths are eligible; all faults and tracing remain entirely in the test.
extern "C" int open(const char* path, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list args;
        va_start(args, flags);
        mode = va_arg(args, int);
        va_end(args);
    }
    if (site("open", path)) return -1;
    return ::syscall(SYS_openat, AT_FDCWD, path, flags, mode);
}
extern "C" int close(int fd) {
    if (site("close", fd_path(fd))) {
        ::syscall(SYS_close, fd); // close can report an error after releasing fd
        errno = EIO;
        return -1;
    }
    return ::syscall(SYS_close, fd);
}
extern "C" FILE* fopen(const char* path, const char* mode) {
    static const auto real = reinterpret_cast<decltype(&::fopen)>(::dlsym(RTLD_NEXT, "fopen"));
    if (site(*mode == 'r' ? "fopen_read" : "fopen_write", path)) return nullptr;
    return real(path, mode);
}
extern "C" int fclose(FILE* stream) {
    static const auto real = reinterpret_cast<decltype(&::fclose)>(::dlsym(RTLD_NEXT, "fclose"));
    const auto fd = ::fileno(stream);
    const bool reading = (::fcntl(fd, F_GETFL) & O_ACCMODE) == O_RDONLY;
    if (site(reading ? "fclose_read" : "fclose_write", fd_path(fd))) {
        real(stream);
        errno = EIO;
        return EOF;
    }
    return real(stream);
}
extern "C" dirent* readdir(DIR* dir) {
    static const auto real = reinterpret_cast<decltype(&::readdir)>(::dlsym(RTLD_NEXT, "readdir"));
    if (site("readdir", fd_path(::dirfd(dir)))) return nullptr;
    return real(dir);
}
extern "C" int closedir(DIR* dir) {
    static const auto real = reinterpret_cast<decltype(&::closedir)>(::dlsym(RTLD_NEXT, "closedir"));
    if (site("closedir", fd_path(::dirfd(dir)))) {
        real(dir);
        errno = EIO;
        return -1;
    }
    return real(dir);
}
extern "C" int lstat(const char* path, struct stat* info) noexcept {
    static const auto real = reinterpret_cast<decltype(&::lstat)>(::dlsym(RTLD_NEXT, "lstat"));
    if (site("lstat", path)) return -1;
    return real(path, info);
}
extern "C" int utimensat(int fd, const char* path, const timespec times[2], int flags) noexcept {
    if (site("utimensat", path)) return -1;
    return ::syscall(SYS_utimensat, fd, path, times, flags);
}

extern "C" int openat(int dir, const char* path, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list args;
        va_start(args, flags);
        mode = va_arg(args, int);
        va_end(args);
    }
    const auto full = *path == '/' ? std::string(path) : fd_path(dir) + "/" + path;
    if (site("openat", full)) return -1;
    return ::syscall(SYS_openat, dir, path, flags, mode);
}
extern "C" int __openat_2(int dir, const char* path, int flags) {
    return openat(dir, path, flags);
}
extern "C" DIR* fdopendir(int fd) {
    static const auto real = reinterpret_cast<decltype(&::fdopendir)>(::dlsym(RTLD_NEXT, "fdopendir"));
    if (site("fdopendir", fd_path(fd))) return nullptr;
    return real(fd);
}
extern "C" int fchmodat(int dir, const char* path, mode_t mode, int flags) noexcept {
    static const auto real = reinterpret_cast<decltype(&::fchmodat)>(::dlsym(RTLD_NEXT, "fchmodat"));
    const auto full = *path == '/' ? std::string(path) : fd_path(dir) + "/" + path;
    if (site("fchmodat", full)) return -1;
    return real(dir, path, mode, flags);
}
extern "C" int remove(const char* path) noexcept {
    static const auto real = reinterpret_cast<decltype(&::remove)>(::dlsym(RTLD_NEXT, "remove"));
    if (site("remove", path)) return -1;
    return real(path);
}
