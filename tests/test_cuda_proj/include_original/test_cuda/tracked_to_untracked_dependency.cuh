#pragma once

#include <third_party/dependency_value.cuh>

extern "C" __global__ void tracked_to_untracked_dependency_kernel(int* output) {
    output[0] = kThirdPartyDependencyValue;
}
