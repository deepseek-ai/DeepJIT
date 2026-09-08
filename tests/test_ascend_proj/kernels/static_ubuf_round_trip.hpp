#pragma once

#include <kernel_operator.h>

inline constexpr uint32_t kStaticUbufBytes = 64 * 1024;

extern "C" __global__ __vector__ void static_ubuf_round_trip_kernel(
    __gm__ uint8_t* input, __gm__ uint8_t* output) {
    __ubuf__ uint8_t buffer[kStaticUbufBytes];
    copy_gm_to_ubuf_align_v2(
        buffer, input, 0, 1, kStaticUbufBytes, 0, 0, false, 0,
        static_cast<uint64_t>(kStaticUbufBytes), kStaticUbufBytes);
    pipe_barrier(pipe_t::PIPE_ALL);
    copy_ubuf_to_gm_align_v2(
        output, buffer, 0, 1, kStaticUbufBytes, 0,
        static_cast<uint64_t>(kStaticUbufBytes), kStaticUbufBytes);
}
