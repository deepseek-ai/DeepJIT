#if TEST_INDEX_BITS == 32
using test_index_t = int;
#elif TEST_INDEX_BITS == 64
using test_index_t = long long;
#else
#error TEST_INDEX_BITS must be 32 or 64
#endif

extern "C" __global__ void macro_selected_abi_kernel(long long* output, const test_index_t value) {
    if (blockIdx.x == 0 and threadIdx.x == 0)
        output[0] = static_cast<long long>(sizeof(test_index_t)) * 1000 + value;
}
