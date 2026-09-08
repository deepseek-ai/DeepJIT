extern "C" __global__ void multidimensional_launch_kernel(int* output) {
    const int block_index = static_cast<int>(
        blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));
    const int thread_index = static_cast<int>(
        threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));
    const int threads_per_block = static_cast<int>(blockDim.x * blockDim.y * blockDim.z);
    output[block_index * threads_per_block + thread_index] = block_index * 1000 + thread_index;
}
