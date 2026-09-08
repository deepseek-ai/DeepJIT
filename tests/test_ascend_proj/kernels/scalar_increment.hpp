#pragma once

#include <kernel_operator.h>

#ifndef TEST_BIAS
#define TEST_BIAS 0
#endif

extern "C" __global__ __vector__ void scalar_increment_kernel(__gm__ int* output, int input) {
    output[0] = input + TEST_BIAS;
}
