#pragma once

#include <kernel_operator.h>

extern "C" __global__ __vector__ void first_entry_kernel(__gm__ int* output) {
    output[0] = 1;
}

extern "C" __global__ __vector__ void second_entry_kernel(__gm__ int* output) {
    output[0] = 2;
}
