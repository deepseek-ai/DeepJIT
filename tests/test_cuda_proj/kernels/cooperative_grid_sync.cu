#include <cooperative_groups.h>

namespace cg = cooperative_groups;

extern "C" __global__ void cooperative_grid_sync_kernel(int* values, int* result) {
    const auto grid = cg::this_grid();
    if (not grid.is_valid()) {
        if (blockIdx.x == 0 and threadIdx.x == 0)
            result[0] = 0;
        return;
    }

    if (threadIdx.x == 0)
        values[blockIdx.x] = static_cast<int>(blockIdx.x) + 1;
    grid.sync();

    if (blockIdx.x == 0 and threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < static_cast<int>(gridDim.x); ++i)
            sum += values[i];
        result[0] = 1;
        result[1] = sum;
    }
}
