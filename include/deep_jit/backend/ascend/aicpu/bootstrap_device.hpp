#pragma once

#include <string_view>

namespace deep_jit::ascend::aicpu {

// Embedded sources keep consumers independent of the DeepJIT installation at runtime.
inline constexpr std::string_view device_source = R"DJCPP(
#include "abi.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <memory>
#include <stdexcept>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
namespace abi = deep_jit::ascend::aicpu;

struct Module {
    int fd = -1;
    void* library = nullptr;
    uint32_t (*entry)(void*) = nullptr;

    ~Module() {
        if (library != nullptr)
            dlclose(library);
        if (fd >= 0)
            close(fd);
    }
};

uint64_t load(const abi::ControlRequest& request) {
    auto module = std::make_unique<Module>();
    // Keep the fd until dlclose: reusing a live /proc/self/fd/N path can make
    // the dynamic linker return a different JIT variant's library.
    module->fd = static_cast<int>(syscall(SYS_memfd_create, "deep-jit-aicpu", 1u));
    if (module->fd < 0)
        throw std::runtime_error("memfd_create failed");
    const auto* image = reinterpret_cast<const unsigned char*>(request.image);
    for (size_t offset = 0; offset < request.image_size;) {
        const auto written = write(module->fd, image + offset, request.image_size - offset);
        if (written < 0 and errno == EINTR)
            continue;
        if (written <= 0)
            throw std::runtime_error("writing AICPU image failed");
        offset += static_cast<size_t>(written);
    }

    char path[64];
    std::snprintf(path, sizeof(path), "/proc/self/fd/%d", module->fd);
    module->library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (module->library == nullptr)
        throw std::runtime_error(dlerror());
    dlerror();
    module->entry = reinterpret_cast<decltype(module->entry)>(dlsym(module->library, request.symbol));
    if (const char* error = dlerror())
        throw std::runtime_error(error);
    if (module->entry == nullptr)
        throw std::runtime_error("AICPU entry is null");
    return reinterpret_cast<uint64_t>(module.release());
}

void remove_bootstrap() {
    Dl_info info{};
    if (not dladdr(reinterpret_cast<void*>(&remove_bootstrap), &info) or info.dli_fname == nullptr)
        throw std::runtime_error("cannot locate AICPU bootstrap");
    const char* name = std::strrchr(info.dli_fname, '/');
    if (name == nullptr or std::strncmp(name + 1, "libdeep_jit_aicpu_", 17) != 0)
        throw std::runtime_error("unexpected AICPU bootstrap path");
    if (unlink(info.dli_fname) != 0 and errno != ENOENT)
        throw std::runtime_error("cannot remove AICPU bootstrap");
}

void publish(abi::Result* output, const abi::Result& result) {
    *output = result;
    uint64_t ctr;
    asm volatile("mrs %0, ctr_el0" : "=r"(ctr));
    const uintptr_t line = 4ull << ((ctr >> 16) & 15);
    const auto begin = reinterpret_cast<uintptr_t>(output);
    for (auto address = begin & ~(line - 1); address < begin + sizeof(*output); address += line)
        asm volatile("dc cvac, %0" :: "r"(address) : "memory");
    asm volatile("dsb sy" ::: "memory");
}
}  // namespace

extern "C" __attribute__((visibility("default"))) uint32_t deep_jit_aicpu_loader_v1(void* raw) {
    const auto& header = *static_cast<const abi::Request*>(raw);
    // The module pointer stays valid until the Host drains queued invocations
    // and submits Unload. Invoke needs no registry lookup, lock or allocation.
    if (header.operation == abi::Operation::Invoke) {
        const auto* module = reinterpret_cast<const Module*>(header.module);
        try {
            return module->entry(static_cast<unsigned char*>(raw) + sizeof(header));
        } catch (...) {
            return 1;
        }
    }

    const auto& request = *static_cast<const abi::ControlRequest*>(raw);
    abi::Result result{};
    try {
        switch (header.operation) {
            case abi::Operation::Load:
                result.module = load(request);
                break;
            case abi::Operation::Unload:
                delete reinterpret_cast<Module*>(header.module);
                break;
            case abi::Operation::RemoveBootstrap:
                remove_bootstrap();
                break;
            default:
                throw std::runtime_error("unknown AICPU loader operation");
        }
    } catch (const std::exception& error) {
        result.status = 1;
        result.sys_errno = errno;
        std::snprintf(result.message, sizeof(result.message), "%s", error.what());
    } catch (...) {
        result.status = 1;
        std::snprintf(result.message, sizeof(result.message), "unknown AICPU loader error");
    }
    publish(reinterpret_cast<abi::Result*>(request.result), result);
    return 0;  // Report load failures without poisoning the ACL stream.
}
)DJCPP";

}  // namespace deep_jit::ascend::aicpu
