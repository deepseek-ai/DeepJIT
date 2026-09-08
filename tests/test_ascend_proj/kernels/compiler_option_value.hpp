#pragma once

#include <kernel_operator.h>

#ifndef TEST_COMPILER_OPTION
#define TEST_COMPILER_OPTION 0
#endif

extern "C" __global__ __vector__ void compiler_option_value_kernel(__gm__ int* output) {
    output[0] = TEST_COMPILER_OPTION;
}
