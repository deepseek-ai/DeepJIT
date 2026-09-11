#pragma once

#include <hip/hip_runtime.h>

#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/lazy.hpp>

#if not defined(__HIP_PLATFORM_AMD__)
#error "DeepJIT ROCm requires the AMD HIP platform"
#endif

namespace deep_jit::rocm::driver {

DJ_DECL_LAZY_DL_HANDLE(get_hip_handle, "libamdhip64.so");

DJ_DECL_LAZY_DL_FUNCTION(get_hip_handle, hipLibraryLoadFromFile);
DJ_DECL_LAZY_DL_FUNCTION(get_hip_handle, hipLibraryUnload);
DJ_DECL_LAZY_DL_FUNCTION(get_hip_handle, hipLibraryGetKernelCount);
DJ_DECL_LAZY_DL_FUNCTION(get_hip_handle, hipLibraryEnumerateKernels);
DJ_DECL_LAZY_DL_FUNCTION(get_hip_handle, hipKernelGetFunction);
DJ_DECL_LAZY_DL_FUNCTION(get_hip_handle, hipFuncGetAttribute);
DJ_DECL_LAZY_DL_FUNCTION(get_hip_handle, hipModuleLaunchKernel);
DJ_DECL_LAZY_DL_FUNCTION(get_hip_handle, hipModuleLaunchCooperativeKernel);
DJ_DECL_LAZY_DL_FUNCTION(get_hip_handle, hipModuleOccupancyMaxActiveBlocksPerMultiprocessor);

inline void require_library_api() {
    // Check before acquiring a library, including its cleanup entry point.
    static const bool available = [] {
        for (const auto* name: {"hipLibraryLoadFromFile", "hipLibraryUnload", "hipLibraryGetKernelCount",
                               "hipLibraryEnumerateKernels", "hipKernelGetFunction"}) {
            DJ_HOST_ASSERT(::dlsym(get_hip_handle(), name) != nullptr,
                           "ROCm 10 HIP library API is required; missing {} in libamdhip64.so. "
                           "Check the HIP runtime and library search path", name);
        }
        return true;
    }();
    (void)available;
}

inline void check_hip(const hipError_t error, const char* expression) {
    if (error != hipSuccess) {
        DJ_PANIC("{} failed with HIP error {} ({}): {}",
                 expression,
                 static_cast<int>(error),
                 hipGetErrorName(error),
                 hipGetErrorString(error));
    }
}

#ifndef DJ_HIP_CHECK
#define DJ_HIP_CHECK(expr) ::deep_jit::rocm::driver::check_hip((expr), #expr)
#endif

}  // namespace deep_jit::rocm::driver
