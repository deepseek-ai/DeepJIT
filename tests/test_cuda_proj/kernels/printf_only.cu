#include <cstdio>

extern "C" __global__ void printf_only_kernel() {
    if (blockIdx.x == 0 and threadIdx.x == 0)
        printf("DeepJIT printf test\n");
}
