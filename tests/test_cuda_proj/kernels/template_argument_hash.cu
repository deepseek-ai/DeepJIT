template <int kBias>
__global__ void add_template(int* output, const int input) {
    if (blockIdx.x == 0 and threadIdx.x == 0)
        output[0] = input + kBias;
}

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&add_template<TEST_BIAS>);
}
