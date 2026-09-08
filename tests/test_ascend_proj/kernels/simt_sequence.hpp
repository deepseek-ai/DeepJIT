#pragma once

#include <kernel_operator.h>
#include <simt_api/asc_simt.h>
#include <simt_api/device_functions.h>

using namespace AscendC;

inline constexpr uint32_t kSimtSequenceElements = 64;

__simt_vf__ void write_simt_sequence(__ubuf__ float* output) {
    using namespace __cce_simt;
    if (threadIdx.x < kSimtSequenceElements)
        output[threadIdx.x] = static_cast<float>(threadIdx.x + 1);
}

extern "C" __global__ __vector__ void simt_sequence_kernel(__gm__ float* output) {
    struct Buffer {
        float output[kSimtSequenceElements];
    };
    constexpr __ubuf__ Buffer* buffer = nullptr;
    asc_vf_call<write_simt_sequence>(dim3(kSimtSequenceElements), buffer->output);
    pipe_barrier(pipe_t::PIPE_ALL);
    constexpr uint32_t num_bytes = kSimtSequenceElements * sizeof(float);
    copy_ubuf_to_gm_align_v2(
        output, buffer->output, 0, 1, num_bytes, 0, static_cast<uint64_t>(num_bytes), num_bytes);
}
