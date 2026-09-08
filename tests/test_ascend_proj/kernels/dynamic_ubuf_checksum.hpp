#pragma once

#include <kernel_operator.h>

extern "C" __global__ __vector__ void dynamic_ubuf_checksum_kernel(__gm__ uint32_t* output, uint32_t num_bytes) {
    extern __ubuf__ uint8_t dynamic_buffer[];
    uint32_t checksum = 0;
    for (uint32_t offset = 0; offset < num_bytes; ++ offset) {
        dynamic_buffer[offset] = static_cast<uint8_t>(offset);
        checksum += dynamic_buffer[offset];
    }
    output[0] = num_bytes;
    output[1] = checksum;
}
