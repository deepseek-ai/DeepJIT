#include <cuda_runtime.h>

#define FOR_EACH_REGISTER(M) \
    M(0)  M(1)  M(2)  M(3)  M(4)  M(5)  M(6)  M(7)  \
    M(8)  M(9)  M(10) M(11) M(12) M(13) M(14) M(15) \
    M(16) M(17) M(18) M(19) M(20) M(21) M(22) M(23) \
    M(24) M(25) M(26) M(27) M(28) M(29) M(30) M(31) \
    M(32) M(33) M(34) M(35) M(36) M(37) M(38) M(39) \
    M(40) M(41) M(42) M(43) M(44) M(45) M(46) M(47)

extern "C" __global__ void forced_register_spill_kernel(
    float* output,
    const float* input,
    const int count) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = gridDim.x * blockDim.x;

#define DECLARE_REGISTER(i) float value_##i = input[index + (i) * stride];
    FOR_EACH_REGISTER(DECLARE_REGISTER)
#undef DECLARE_REGISTER

    for (int i = index; i < count; i += stride) {
        const auto value = input[i];
#define UPDATE_REGISTER(j) value_##j = fmaf(value_##j, 1.0001f + (j) * 0.0001f, value);
        FOR_EACH_REGISTER(UPDATE_REGISTER)
#undef UPDATE_REGISTER
    }

#define STORE_REGISTER(i) output[index + (i) * stride] = value_##i;
    FOR_EACH_REGISTER(STORE_REGISTER)
#undef STORE_REGISTER
}

#undef FOR_EACH_REGISTER
