#pragma once

#include <kernel_operator.h>
#include <third_party/dependency_value.hpp>

extern "C" __global__ __vector__ void untracked_dependency_kernel(__gm__ int* output) {
    output[0] = kThirdPartyValue;
}
