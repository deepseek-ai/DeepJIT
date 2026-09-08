#include <cuda.h>

struct LargeArgumentPayload {
    unsigned long long values[128];
};

struct IndirectArgumentPayload {
    int values[8];
};

__device__ void update_checksum(unsigned long long& checksum, const CUtensorMap& tensor_map) {
    const auto bytes = reinterpret_cast<const unsigned char*>(&tensor_map);
    for (int i = 0; i < static_cast<int>(sizeof(CUtensorMap)); ++i)
        checksum = (checksum ^ bytes[i]) * 1099511628211ull;
}

extern "C" __global__ void large_kernel_arguments_kernel(
    unsigned long long* output,
    const __grid_constant__ LargeArgumentPayload payload,
    const __grid_constant__ CUtensorMap tensor_map_0,
    const __grid_constant__ CUtensorMap tensor_map_1,
    const __grid_constant__ CUtensorMap tensor_map_2,
    const __grid_constant__ CUtensorMap tensor_map_3,
    const __grid_constant__ CUtensorMap tensor_map_4,
    const __grid_constant__ CUtensorMap tensor_map_5,
    const __grid_constant__ CUtensorMap tensor_map_6,
    const __grid_constant__ CUtensorMap tensor_map_7,
    const __grid_constant__ CUtensorMap tensor_map_8,
    const __grid_constant__ CUtensorMap tensor_map_9,
    const __grid_constant__ CUtensorMap tensor_map_10,
    const __grid_constant__ CUtensorMap tensor_map_11,
    const __grid_constant__ CUtensorMap tensor_map_12,
    const __grid_constant__ CUtensorMap tensor_map_13,
    const __grid_constant__ CUtensorMap tensor_map_14,
    const __grid_constant__ CUtensorMap tensor_map_15,
    const __grid_constant__ CUtensorMap tensor_map_16,
    const __grid_constant__ CUtensorMap tensor_map_17,
    const IndirectArgumentPayload indirect_payload,
    const void* optional_pointer) {
    if (blockIdx.x != 0 or threadIdx.x != 0)
        return;

    unsigned long long checksum = 1469598103934665603ull;
    for (const auto value : payload.values)
        checksum = (checksum ^ value) * 1099511628211ull;

#define UPDATE_TENSOR_MAP(index) update_checksum(checksum, tensor_map_##index)
    UPDATE_TENSOR_MAP(0);
    UPDATE_TENSOR_MAP(1);
    UPDATE_TENSOR_MAP(2);
    UPDATE_TENSOR_MAP(3);
    UPDATE_TENSOR_MAP(4);
    UPDATE_TENSOR_MAP(5);
    UPDATE_TENSOR_MAP(6);
    UPDATE_TENSOR_MAP(7);
    UPDATE_TENSOR_MAP(8);
    UPDATE_TENSOR_MAP(9);
    UPDATE_TENSOR_MAP(10);
    UPDATE_TENSOR_MAP(11);
    UPDATE_TENSOR_MAP(12);
    UPDATE_TENSOR_MAP(13);
    UPDATE_TENSOR_MAP(14);
    UPDATE_TENSOR_MAP(15);
    UPDATE_TENSOR_MAP(16);
    UPDATE_TENSOR_MAP(17);
#undef UPDATE_TENSOR_MAP

    for (const auto value : indirect_payload.values)
        checksum = (checksum ^ static_cast<unsigned int>(value)) * 1099511628211ull;
    output[0] = optional_pointer == nullptr ? checksum : 0;
}
