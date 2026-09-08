#pragma once

#include <test_cuda/detail/tracked_constant.cuh>

template <int kExtra>
__global__ void add_from_tracked_include(int* output, const int input) {
    if (blockIdx.x == 0 and threadIdx.x == 0)
        output[0] = input + kTrackedBias + kExtra;
}
