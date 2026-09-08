#pragma once

#include <kernel_operator.h>

#include <test_ascend/detail/tracked_offset.hpp>

extern "C" __global__ __vector__ void tracked_vector_add_kernel(
    __gm__ const float* lhs, __gm__ const float* rhs, __gm__ float* output, int64_t count) {
    for (int64_t index = get_block_idx(); index < count; index += get_block_num())
        output[index] = lhs[index] + rhs[index] + static_cast<float>(kTrackedOffset);
}
