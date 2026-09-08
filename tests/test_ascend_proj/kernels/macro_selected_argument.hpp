#pragma once

#include <kernel_operator.h>

#ifdef TEST_WIDE_ARGUMENT
using test_argument_t = int64_t;
#else
using test_argument_t = int32_t;
#endif

extern "C" __global__ __vector__ void macro_selected_argument_kernel(
    __gm__ int64_t* output, test_argument_t value) {
    output[0] = static_cast<int64_t>(value);
}
