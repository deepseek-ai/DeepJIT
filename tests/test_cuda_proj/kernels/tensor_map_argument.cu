#include <cuda.h>

extern "C" __global__ void tensor_map_argument_kernel(const CUtensorMap tensor_map, unsigned int* output) {
    const auto bytes = reinterpret_cast<const unsigned char*>(&tensor_map);
    unsigned int checksum = 2166136261u;
    for (int i = 0; i < static_cast<int>(sizeof(CUtensorMap)); ++i)
        checksum = (checksum ^ bytes[i]) * 16777619u;
    if (blockIdx.x == 0 and threadIdx.x == 0)
        output[0] = checksum;
}
