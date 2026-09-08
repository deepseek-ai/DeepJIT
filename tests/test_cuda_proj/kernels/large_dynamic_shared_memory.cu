extern "C" __global__ void large_dynamic_shared_memory_kernel(
    int* output,
    const int value,
    const int num_elements) {
    extern __shared__ int shared[];
    if (threadIdx.x == 0)
        shared[num_elements - 1] = value;
    __syncthreads();
    if (threadIdx.x == 0)
        output[0] = shared[num_elements - 1];
}
