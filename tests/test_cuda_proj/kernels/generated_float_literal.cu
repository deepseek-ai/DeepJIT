template <float kValue>
__global__ void generated_float_literal_kernel(float* output) {
    output[0] = kValue;
}

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&generated_float_literal_kernel<TEST_FLOAT_LITERAL>);
}
