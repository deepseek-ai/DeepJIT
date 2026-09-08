#ifndef TEST_OPTION
#error TEST_OPTION must be defined
#endif

extern "C" __global__ void compiler_options_kernel(int* output, const int) {
    if (blockIdx.x == 0 and threadIdx.x == 0)
        output[0] = TEST_OPTION;
}
