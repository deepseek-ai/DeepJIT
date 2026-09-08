#pragma once

#include <kernel_operator.h>

extern "C" __global__ __mix__(1, 2) void mixed_kernel_metadata_kernel(int) {
    AscendC::InitSocState();
}
