struct RuntimeLaunchArguments {
    int base;
    int multiplier;
    unsigned long long delay_cycles;
};

extern "C" __global__ void runtime_launch_features_kernel(
    int* output,
    const RuntimeLaunchArguments arguments,
    const int input) {
    extern __shared__ int shared[];
    if (threadIdx.x == 0) {
        const auto start = clock64();
        while (clock64() - start < arguments.delay_cycles) {}
        shared[0] = arguments.base + arguments.multiplier * input + static_cast<int>(blockIdx.x);
    }
    __syncthreads();
    if (threadIdx.x == 0)
        output[blockIdx.x] = shared[0];
}
