#pragma once

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <type_traits>

#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <hip/hip_runtime.h>

#include <deep_jit/backend/rocm/driver.hpp>
#include <deep_jit/backend/rocm/options.hpp>
#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/gil.hpp>
#include <deep_jit/utils/no_ref_ptr.hpp>

namespace deep_jit::rocm {

template <typename T>
inline void* kernel_arg_pointer(const T& value) {
    if constexpr (std::is_base_of_v<NoRefPtr, std::decay_t<T>>) {
        return value.ptr;
    } else {
        return const_cast<void*>(static_cast<const void*>(&value));
    }
}

inline hipStream_t current_stream(const int device_index) {
    // ROCm tensors use DeviceType::CUDA. This wrapper supports both PyTorch
    // HIPify stream APIs without depending on their renamed free functions.
    const c10::impl::VirtualGuardImpl guard(c10::DeviceType::CUDA);
    const auto stream = guard.getStream(c10::Device(c10::DeviceType::CUDA, device_index));
    return c10::hip::HIPStreamMasqueradingAsCUDA(stream).stream();
}

// HIP library ownership is shared with the runtime's memory cache and callers.
class Kernel {
    int device_index = -1;
    hipDeviceProp_t prop{};
    int max_threads = 0;
    int static_smem_bytes = 0;
    int max_dynamic_smem_bytes = 0;

public:
    hipLibrary_t library_handle{};
    hipFunction_t kernel_handle{};

    Kernel() = default;
    ~Kernel() { unload(); }
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;
    Kernel(Kernel&&) = delete;
    Kernel& operator=(Kernel&&) = delete;

    static std::shared_ptr<Kernel> load(const std::filesystem::path& dir, const Env& env) {
        GilScopedRelease gil_release;

        const auto hsaco_path = dir / "kernel.hsaco";
        DJ_HOST_ASSERT(std::filesystem::is_regular_file(hsaco_path) and std::filesystem::file_size(hsaco_path) != 0,
                       "missing or empty ROCm HSACO: {}", hsaco_path.string());
        const bool debug = env.get<bool>("JIT_DEBUG", false);
        const bool print_load_time = debug or env.get<bool>("JIT_PRINT_LOAD_TIME", false);
        if (debug)
            std::fputs(std::format("Loading HSACO: {}\n", hsaco_path.string()).c_str(), stdout);
        const auto start_time = std::chrono::steady_clock::now();

        driver::require_library_api();
        // Allocate ownership before acquiring resources, so every failure cleans up.
        auto kernel = std::make_shared<Kernel>();
        DJ_HIP_CHECK(hipGetDevice(&kernel->device_index));
        DJ_HIP_CHECK(hipFree(nullptr));
        DJ_HIP_CHECK(hipGetDeviceProperties(&kernel->prop, kernel->device_index));
        DJ_HIP_CHECK(driver::lazy_hipLibraryLoadFromFile(
            &kernel->library_handle, hsaco_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
        unsigned int num_kernels = 0;
        DJ_HIP_CHECK(driver::lazy_hipLibraryGetKernelCount(&num_kernels, kernel->library_handle));
        DJ_HOST_ASSERT(num_kernels == 1, "expected exactly one kernel in {} (found {})", hsaco_path.string(), num_kernels);
        hipKernel_t library_kernel_handle{};
        DJ_HIP_CHECK(driver::lazy_hipLibraryEnumerateKernels(&library_kernel_handle, 1, kernel->library_handle));
        DJ_HIP_CHECK(driver::lazy_hipKernelGetFunction(&kernel->kernel_handle, library_kernel_handle));
        DJ_HIP_CHECK(driver::lazy_hipFuncGetAttribute(
            &kernel->max_threads, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel->kernel_handle));
        DJ_HIP_CHECK(driver::lazy_hipFuncGetAttribute(
            &kernel->static_smem_bytes, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel->kernel_handle));
        DJ_HIP_CHECK(driver::lazy_hipFuncGetAttribute(
            &kernel->max_dynamic_smem_bytes, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernel->kernel_handle));

        if (print_load_time) {
            const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start_time;
            std::fputs(std::format("Load time ({}): {:.2f} ms\n", dir.string(), elapsed.count()).c_str(), stdout);
        }
        return kernel;
    }

    template <typename... Args>
    void launch(const LaunchOptions& options, const Args&... args) const {
        GilScopedRelease gil_release;

        DJ_HOST_ASSERT(kernel_handle != nullptr, "kernel must be loaded before launch");
        DJ_HOST_ASSERT(options.num_smem_bytes.has_value(), "ROCm dynamic shared-memory size must be specified");
        DJ_HOST_ASSERT(options.grid_dim.has_value(), "ROCm grid dimension must be specified");
        DJ_HOST_ASSERT(options.block_dim.has_value(), "ROCm block dimension must be specified");
        DJ_HOST_ASSERT(options.cooperative.has_value(), "ROCm cooperative option must be specified");
        DJ_HOST_ASSERT(not options.enable_pdl.value_or(false), "ROCm does not support CUDA PDL");
        DJ_HOST_ASSERT(not options.nonportable_cluster_size_allowed.value_or(false),
                       "ROCm does not support CUDA non-portable clusters");
        if (options.cluster_dim) {
            DJ_HOST_ASSERT(options.cluster_dim->x == 1 and options.cluster_dim->y == 1 and options.cluster_dim->z == 1,
                           "ROCm does not support CUDA thread-block clusters");
        }
        DJ_HOST_ASSERT(*options.num_smem_bytes >= 0, "ROCm dynamic shared-memory size must not be negative");
        DJ_HOST_ASSERT(*options.num_smem_bytes <= max_dynamic_smem_bytes and
                       static_smem_bytes >= 0 and
                       static_cast<uint64_t>(*options.num_smem_bytes) + static_smem_bytes <= prop.sharedMemPerBlock,
                       "ROCm shared-memory request exceeds the kernel or device limit");

        int current_device = -1;
        DJ_HIP_CHECK(hipGetDevice(&current_device));
        DJ_HOST_ASSERT(current_device == device_index, "ROCm kernel must be launched on the device on which it was loaded");
        const auto grid = *options.grid_dim;
        const auto block = *options.block_dim;
        const std::array<unsigned int, 3> grids = {grid.x, grid.y, grid.z};
        const std::array<unsigned int, 3> blocks = {block.x, block.y, block.z};
        uint64_t num_threads = 1;
        for (int i = 0; i < 3; ++i) {
            DJ_HOST_ASSERT(grids[i] > 0 and blocks[i] > 0, "ROCm grid and block dimensions must be positive");
            DJ_HOST_ASSERT(prop.maxGridSize[i] > 0 and grids[i] <= static_cast<unsigned int>(prop.maxGridSize[i]) and
                           prop.maxThreadsDim[i] > 0 and blocks[i] <= static_cast<unsigned int>(prop.maxThreadsDim[i]),
                           "ROCm launch dimensions exceed device limits");
            DJ_HOST_ASSERT(static_cast<uint64_t>(grids[i]) * blocks[i] < (uint64_t(1) << 32),
                           "ROCm grid times block must be less than 2^32 in each dimension");
            num_threads *= blocks[i];
            DJ_HOST_ASSERT(max_threads > 0 and num_threads <= static_cast<uint64_t>(max_threads) and
                           prop.maxThreadsPerBlock > 0 and num_threads <= static_cast<uint64_t>(prop.maxThreadsPerBlock),
                           "ROCm block size exceeds the kernel or device limit");
        }

        // An explicit null stream must not be replaced with the current stream.
        const auto stream = options.stream ? *options.stream : current_stream(device_index);
        void* kernel_arg_ptrs[sizeof...(Args) + 1] = {kernel_arg_pointer(args)..., nullptr};
        auto* params = sizeof...(Args) == 0 ? nullptr : kernel_arg_ptrs;
        if (*options.cooperative) {
            DJ_HOST_ASSERT(prop.cooperativeLaunch != 0, "ROCm device does not support cooperative launch");
            int blocks_per_sm = 0;
            DJ_HIP_CHECK(driver::lazy_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks_per_sm, kernel_handle, static_cast<int>(num_threads), *options.num_smem_bytes));
            DJ_HOST_ASSERT(blocks_per_sm > 0 and prop.multiProcessorCount > 0, "ROCm cooperative kernel has no active blocks");
            uint64_t capacity = static_cast<uint64_t>(blocks_per_sm) * prop.multiProcessorCount;
            for (const auto dimension: grids) {
                DJ_HOST_ASSERT(dimension <= capacity, "ROCm cooperative grid exceeds resident block capacity");
                capacity /= dimension;
            }
            DJ_HIP_CHECK(driver::lazy_hipModuleLaunchCooperativeKernel(
                kernel_handle, grid.x, grid.y, grid.z, block.x, block.y, block.z,
                *options.num_smem_bytes, stream, params));
        } else {
            DJ_HIP_CHECK(driver::lazy_hipModuleLaunchKernel(
                kernel_handle, grid.x, grid.y, grid.z, block.x, block.y, block.z,
                *options.num_smem_bytes, stream, params, nullptr));
        }
    }

    void unload() noexcept {
        if (library_handle == nullptr)
            return;
        try {
            DJ_HIP_CHECK(driver::lazy_hipLibraryUnload(library_handle));
        } catch (...) {
        }
        library_handle = nullptr;
        kernel_handle = nullptr;
    }
};

}  // namespace deep_jit::rocm
