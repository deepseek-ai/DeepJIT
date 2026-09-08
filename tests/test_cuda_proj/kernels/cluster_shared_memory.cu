#include <cooperative_groups.h>

namespace cg = cooperative_groups;

extern "C" __global__ void cluster_shared_memory_kernel(int* output) {
#if __CUDA_ARCH__ >= 900
    extern __shared__ int shared[];
    const auto cluster = cg::this_cluster();
    const auto rank = cluster.block_rank();
    const auto num_blocks = cluster.num_blocks();
    if (threadIdx.x == 0) {
        shared[0] = static_cast<int>(rank) + 1;
        output[2 + blockIdx.x] = static_cast<int>(num_blocks);
    }
    cluster.sync();

    int peer_value = -1;
    if (num_blocks == 2 and threadIdx.x == 0)
        peer_value = cluster.map_shared_rank(shared, 1 - rank)[0];
    cluster.sync();

    if (threadIdx.x == 0)
        output[blockIdx.x] = peer_value;
#else
    if (threadIdx.x == 0)
        output[blockIdx.x] = -2;
#endif
}
