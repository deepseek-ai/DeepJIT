#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime_api.h>

// Expose the CUDA declarations in Torch's shims without adding USE_CUDA to
// the rest of the including translation unit.
#ifndef USE_CUDA
#define DEEP_JIT_UNDEFINE_USE_CUDA
#define USE_CUDA
#endif
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/headeronly/util/shim_utils.h>
#ifdef DEEP_JIT_UNDEFINE_USE_CUDA
#undef USE_CUDA
#undef DEEP_JIT_UNDEFINE_USE_CUDA
#endif

namespace deep_jit::cuda {

// Returns the current CUDA stream for use with the CUDA Driver API.
inline CUstream get_current_cuda_stream(const int32_t device_index) {
    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
    return static_cast<CUstream>(stream_ptr);
}

// Upstream torch plans to add stable API's for get_stream_from_pool and TorchCUDAStreamGuard.
// Once they are added and TORCH_TARGET_VERSION catches up, this file can be deleted
// and the APIs can be called directly.

inline cudaStream_t get_stream_from_pool(const int32_t device_index) {
    void* stream = nullptr;
    TORCH_ERROR_CODE_CHECK(
        torch_get_cuda_stream_from_pool(false, device_index, &stream));
    return static_cast<cudaStream_t>(stream);
}

class TorchCUDAStreamGuard {
public:
    TorchCUDAStreamGuard() = delete;

    explicit TorchCUDAStreamGuard(cudaStream_t stream, int32_t device_index) {
        TORCH_ERROR_CODE_CHECK(
            aoti_torch_create_cuda_stream_guard(
                static_cast<void*>(stream), device_index, &guard_));
    }

    ~TorchCUDAStreamGuard() {
        if (guard_)
            (void)aoti_torch_delete_cuda_stream_guard(guard_);
    }

    // Match c10::cuda::CUDAStreamGuard: copying is disallowed because the
    // guard has unique ownership, and moving is disallowed because an RAII
    // stream guard has no uninitialized state to leave behind.
    TorchCUDAStreamGuard(const TorchCUDAStreamGuard&) = delete;
    TorchCUDAStreamGuard& operator=(const TorchCUDAStreamGuard&) = delete;
    TorchCUDAStreamGuard(TorchCUDAStreamGuard&&) = delete;
    TorchCUDAStreamGuard& operator=(TorchCUDAStreamGuard&&) = delete;

private:
    CUDAStreamGuardHandle guard_ = nullptr;
};

}  // namespace deep_jit::cuda
