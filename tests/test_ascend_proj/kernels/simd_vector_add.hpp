#pragma once

#include <kernel_operator.h>

using namespace AscendC;
using namespace __cce_simd;

inline constexpr uint32_t kSimdVectorElements = 64;

__simd_vf__ inline void simd_vector_add_vf(__ubuf__ float* output, __ubuf__ float* lhs, __ubuf__ float* rhs) {
    const vector_bool mask = pset_b32(PAT_ALL);
    vector_f32 lhs_vector, rhs_vector, output_vector;
    vlds(lhs_vector, lhs, 0, NORM);
    vlds(rhs_vector, rhs, 0, NORM);
    vadd(output_vector, lhs_vector, rhs_vector, mask);
    vsts(output_vector, output, 0, NORM_B32, mask);
}

extern "C" __global__ __vector__ void simd_vector_add_kernel(
    __gm__ float* lhs, __gm__ float* rhs, __gm__ float* output) {
    struct Buffer {
        float lhs[kSimdVectorElements];
        float rhs[kSimdVectorElements];
        float output[kSimdVectorElements];
    };
    constexpr __ubuf__ Buffer* buffer = nullptr;
    constexpr uint32_t num_bytes = kSimdVectorElements * sizeof(float);
    copy_gm_to_ubuf_align_v2(
        buffer->lhs, lhs, 0, 1, num_bytes, 0, 0, false, 0, static_cast<uint64_t>(num_bytes), num_bytes);
    copy_gm_to_ubuf_align_v2(
        buffer->rhs, rhs, 0, 1, num_bytes, 0, 0, false, 0, static_cast<uint64_t>(num_bytes), num_bytes);
    pipe_barrier(pipe_t::PIPE_ALL);
    simd_vector_add_vf(buffer->output, buffer->lhs, buffer->rhs);
    pipe_barrier(pipe_t::PIPE_ALL);
    copy_ubuf_to_gm_align_v2(
        output, buffer->output, 0, 1, num_bytes, 0, static_cast<uint64_t>(num_bytes), num_bytes);
}
