#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <spawn.h>
#include <string>
#include <system_error>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include <deep_jit/backend/ascend/aicpu/bootstrap_abi.hpp>
#include <deep_jit/backend/ascend/aicpu/bootstrap_device.hpp>
#include <deep_jit/backend/ascend/aicpu/bootstrap_helper.hpp>
#include <deep_jit/backend/ascend/driver.hpp>
#include <deep_jit/cache/disk.hpp>
#include <deep_jit/utils/command.hpp>
#include <deep_jit/utils/filesystem.hpp>
#include <deep_jit/utils/gil.hpp>
#include <deep_jit/utils/hash.hpp>
#include <deep_jit/utils/str.hpp>

extern char** environ;

namespace deep_jit::ascend::aicpu {

// Map the Host .so image for the AICPU scheduler process to read during Load.
// The mapping is released after Load. Invocations execute the loaded device code.
class KernelImage {
    void* host = nullptr;
    void* device = nullptr;
    size_t length = 0;
    size_t mapped_length = 0;
    bool registered = false;

    void release() noexcept {
        try {
            // Retain pages if unregister fails, rather than freeing a live mapping.
            if (registered and driver::lazy_aclrtHostUnregister(host) != ACL_SUCCESS)
                return;
            if (host != nullptr)
                munmap(host, mapped_length);
        } catch (...) {
        }
    }

public:
    explicit KernelImage(const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        DJ_HOST_ASSERT(input.is_open(), "failed to open AICPU binary: {}", path.string());
        const auto size = input.tellg();
        DJ_HOST_ASSERT(size > 0, "empty or unreadable AICPU binary: {}", path.string());
        const auto page_size = sysconf(_SC_PAGESIZE);
        DJ_HOST_ASSERT(page_size > 0, "cannot determine Host page size");
        const auto page = static_cast<size_t>(page_size);
        DJ_HOST_ASSERT(static_cast<uintmax_t>(size) <= std::numeric_limits<size_t>::max() - (page - 1) and
                       static_cast<uintmax_t>(size) <= static_cast<uintmax_t>(std::numeric_limits<std::streamsize>::max()),
                       "AICPU kernel image is too large: {}", path.string());
        length = static_cast<size_t>(size);
        mapped_length = (length + page - 1) / page * page;
        try {
            void* mapping = mmap(nullptr, mapped_length, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            DJ_HOST_ASSERT(mapping != MAP_FAILED, "mmap AICPU image failed: {}", std::strerror(errno));
            host = mapping;
            input.seekg(0);
            DJ_HOST_ASSERT(input.read(static_cast<char*>(host), static_cast<std::streamsize>(length)),
                           "failed to read AICPU binary: {}", path.string());
            DJ_ACL_CHECK(driver::lazy_aclrtHostRegisterV2(host, mapped_length, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED));
            registered = true;
            DJ_ACL_CHECK(driver::lazy_aclrtHostGetDevicePointer(host, &device, 0));
        } catch (...) {
            release();
            throw;
        }
    }

    ~KernelImage() { release(); }
    KernelImage(const KernelImage&) = delete;
    KernelImage& operator=(const KernelImage&) = delete;

    void* data() const { return device; }
    size_t size() const { return length; }
};

inline constexpr std::string_view loader_cache_tag = "aicpu_bootstrap";
inline constexpr std::string_view loader_flags = "-std=c++17 -O2 -shared -fPIC -fvisibility=hidden -ldl";

// Build the loader once and publish it to the JIT disk cache.
inline std::filesystem::path build_loader(const std::filesystem::path& cann, const DiskCache& cache) {
    const auto compiler = cann / "tools/hcc/bin/aarch64-target-linux-gnu-g++";
    DJ_HOST_ASSERT(is_executable(compiler),
                   "AICPU bootstrap deployment failed: loader compiler is not executable: {}",
                   compiler.string());
    const auto version = call_external_command(compiler.string() + " --version");
    const auto digest = hash::FNV1a()
                            .update(loader_flags)
                            .update(abi_source)
                            .update(device_source)
                            .update(version)
                            .get_hex_digest();
    auto entry = cache.entry(std::string(loader_cache_tag), digest);
    const auto library = entry.path / "libdeep_jit_aicpu_loader.so";
    if (entry.hit) {
        DJ_HOST_ASSERT(std::filesystem::is_regular_file(library),
                       "AICPU bootstrap cache entry is incomplete: {}", entry.path.string());
        return library;
    }

    write_file_sync(entry.path / "abi.hpp", abi_source);
    write_file_sync(entry.path / "loader.cpp", device_source);
    const std::vector<std::string> command = {
        compiler.string(), "-std=c++17", "-O2", "-shared", "-fPIC", "-fvisibility=hidden",
        (entry.path / "loader.cpp").string(), "-ldl", "-o", library.string(),
    };
    call_external_command(str::join(command));
    DJ_HOST_ASSERT(std::filesystem::is_regular_file(library) and std::filesystem::file_size(library) > 0,
                   "loader compiler did not produce the AICPU bootstrap library");
    return entry.commit() / "libdeep_jit_aicpu_loader.so";
}

// Package the cached loader and deploy its library to the device filesystem.
// A fresh helper process selects the private package before initializing CANN.
inline void deploy_loader(const std::filesystem::path& cann,
                          const std::filesystem::path& directory,
                          const std::filesystem::path& loader,
                          const int device) {
    write_file_sync(directory / "helper.py", helper_source);
    const auto log = directory / "deploy.log";
    std::string python = "python3", script = (directory / "helper.py").string();
    auto toolkit = cann.string(), output = directory.string(), binary = loader.string(), index = std::to_string(device);
    std::string symbol = entry;
    char* argv[] = {
        python.data(), script.data(), toolkit.data(), output.data(),
        binary.data(), index.data(), symbol.data(), nullptr,
    };

    posix_spawn_file_actions_t actions;
    DJ_HOST_ASSERT(posix_spawn_file_actions_init(&actions) == 0, "cannot initialize AICPU helper actions");
    struct ActionsGuard {
        posix_spawn_file_actions_t* actions;
        ~ActionsGuard() { posix_spawn_file_actions_destroy(actions); }
    } guard{&actions};
    DJ_HOST_ASSERT(posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, log.c_str(),
                                                   O_WRONLY | O_CREAT | O_TRUNC, 0600) == 0 and
                   posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO) == 0,
                   "cannot redirect AICPU helper output");
    pid_t pid;
    const int error = posix_spawnp(&pid, python.c_str(), &actions, nullptr, argv, environ);
    DJ_HOST_ASSERT(error == 0, "cannot start AICPU bootstrap helper: {}", std::strerror(error));
    int status = 0;
    pid_t waited;
    do {
        waited = waitpid(pid, &status, 0);
    } while (waited < 0 and errno == EINTR);
    DJ_HOST_ASSERT(waited == pid and WIFEXITED(status) and WEXITSTATUS(status) == 0,
                   "AICPU bootstrap deployment failed:\n{}", read(log));
}

}  // namespace deep_jit::ascend::aicpu

namespace deep_jit::ascend {

// One bootstrap per Runtime, created on the first AICPU load.
// Deploys the loader library and registers its ACL entry in the current context.
// The application's AICPU scheduler process loads the library on first dispatch
// and handles one request per launch. JIT kernels can access that process's Jetty mappings.
class AicpuBootstrap {
    std::filesystem::path directory_;
    aclrtBinHandle binary_handle = nullptr;
    aclrtFuncHandle loader_handle = nullptr;
    aclrtStream loader_stream = nullptr;
    void* result_buffer = nullptr;
    mutable std::mutex loader_mutex;
    mutable bool loader_healthy = true;

    void release() noexcept {
        try {
            if (loader_handle != nullptr and loader_healthy)
                execute({.header = {.operation = aicpu::Operation::RemoveBootstrap}});
        } catch (...) {
        }
        try {
            if (loader_stream != nullptr)
                driver::lazy_aclrtSynchronizeStream(loader_stream);
        } catch (...) {
        }
        try {
            if (loader_stream != nullptr)
                driver::lazy_aclrtDestroyStream(loader_stream);
        } catch (...) {
        }
        try {
            if (result_buffer != nullptr)
                driver::lazy_aclrtFree(result_buffer);
        } catch (...) {
        }
        try {
            if (binary_handle != nullptr)
                driver::lazy_aclrtBinaryUnLoad(binary_handle);
        } catch (...) {
        }
        // remove_all does not follow the shadow's links into the installed toolkit.
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
    }

public:
    ~AicpuBootstrap() { release(); }
    AicpuBootstrap(const AicpuBootstrap&) = delete;
    AicpuBootstrap& operator=(const AicpuBootstrap&) = delete;

    // Return the loader kernel handle used to dispatch AICPU requests.
    aclrtFuncHandle get_loader_handle() const { return loader_handle; }
    const std::filesystem::path& get_directory() const { return directory_; }

    explicit AicpuBootstrap(const std::filesystem::path& cann, const DiskCache& cache) {
        int device;
        DJ_ACL_CHECK(driver::lazy_aclrtGetDevice(&device));
        // Create a private directory for the deployed bootstrap
        // TODO: Sharing the bootstrap among Runtime instances created in the same context
        auto pattern = (std::filesystem::temp_directory_path() / "deep-jit-aicpu-XXXXXX").string();
        DJ_HOST_ASSERT(mkdtemp(pattern.data()) != nullptr, "cannot create AICPU bootstrap work directory: {}", std::strerror(errno));
        directory_ = pattern;
        try {
            DJ_ACL_CHECK(driver::lazy_aclrtCreateStream(&loader_stream));
            DJ_ACL_CHECK(driver::lazy_aclrtMalloc(&result_buffer, sizeof(aicpu::LoaderResult), ACL_MEM_MALLOC_HUGE_FIRST));
            const auto loader = aicpu::build_loader(cann, cache);
            aicpu::deploy_loader(cann, directory_, loader, device);
            // We can use mode 0 to register the deployed library's entry
            const auto config = directory_ / "shadow/conf/bootstrap.json";
            aclrtBinaryLoadOption option{};
            option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
            option.value.cpuKernelMode = 0;
            aclrtBinaryLoadOptions options{&option, 1};
            DJ_ACL_CHECK(driver::lazy_aclrtBinaryLoadFromFile(config.c_str(), &options, &binary_handle));
            DJ_ACL_CHECK(driver::lazy_aclrtBinaryGetFunction(binary_handle, aicpu::entry, &loader_handle));
        } catch (...) {
            release();
            throw;
        }
    }

    // Execute a loader request synchronously, serializing access to the shared result buffer.
    aicpu::LoaderResult execute(aicpu::LoaderRequest request) const {
        [[maybe_unused]] GilScopedRelease gil_release;
        const std::lock_guard lock(loader_mutex);
        DJ_HOST_ASSERT(loader_healthy, "AICPU loader is unavailable after an ACL failure");
        request.result = reinterpret_cast<uint64_t>(result_buffer);
        aclrtLaunchKernelAttr attribute{};
        attribute.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
        attribute.value.timeout = 30;
        aclrtLaunchKernelCfg config{&attribute, 1};
        // ACL failures disable reuse; errors reported in LoaderResult do not.
        loader_healthy = false;
        // Launch the loader once to handle this request.
        DJ_ACL_CHECK(driver::lazy_aclrtLaunchKernelWithHostArgs(
            loader_handle, 1, loader_stream, &config, &request, sizeof(request), nullptr, 0));
        DJ_ACL_CHECK(driver::lazy_aclrtSynchronizeStream(loader_stream));
        aicpu::LoaderResult result{};
        DJ_ACL_CHECK(driver::lazy_aclrtMemcpy(&result, sizeof(result), result_buffer,
                                            sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST));
        loader_healthy = true;
        result.message[sizeof(result.message) - 1] = '\0';
        DJ_HOST_ASSERT(result.status == 0, "AICPU loader failed (errno {}): {}", result.sys_errno, result.message);
        return result;
    }

    // Map the kernel image and ask the loader to load it and return the module
    uint64_t load(const std::filesystem::path& path, const std::string& symbol) const {
        aicpu::LoaderRequest request;
        DJ_HOST_ASSERT(symbol.size() < sizeof(request.symbol), "AICPU entry name is too long");
        const aicpu::KernelImage image(path);
        request.image = reinterpret_cast<uint64_t>(image.data());
        request.image_size = image.size();
        std::memcpy(request.symbol, symbol.c_str(), symbol.size() + 1);
        return execute(request).module;
    }

    // Tell the bootstrap to unload a kernel
    void unload(const uint64_t module) const {
        // Launch is asynchronous: code and memfd must survive all queued calls.
        DJ_ACL_CHECK(driver::lazy_aclrtSynchronizeDevice());
        execute({.header = {.operation = aicpu::Operation::Unload, .module = module}});
    }
};

}  // namespace deep_jit::ascend
