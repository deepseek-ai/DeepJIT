#pragma once

#include <kernel_operator.h>

extern "C" __global__ __vector__ void mixed_argument_echo_kernel(
    __gm__ int8_t* output_i8, int8_t value_i8,
    __gm__ float* output_f32, float value_f32,
    __gm__ int64_t* output_i64, int64_t value_i64,
    __gm__ double* output_f64, double value_f64) {
    output_i8[0] = value_i8;
    output_f32[0] = value_f32;
    output_i64[0] = value_i64;
    output_f64[0] = value_f64;
}
