extern "C" __global__ void explicit_local_memory_kernel(
    float* output,
    const float* input,
    const int dynamic_index) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = gridDim.x * blockDim.x;
    volatile float values[64];

#pragma unroll
    for (int i = 0; i < 64; ++i)
        values[i] = input[index + i * stride];
    output[index] = values[dynamic_index & 63];
}
