struct MixedArgumentPayload {
    int first;
    unsigned int second;
    long long third;
};

extern "C" __global__ void mixed_arguments_kernel(
    long long* output,
    const int* input,
    const int* optional_input,
    const signed char value_i8,
    const unsigned short value_u16,
    const int value_i32,
    const unsigned int value_u32,
    const long long value_i64,
    const unsigned long long value_u64,
    const float value_f32,
    const double value_f64,
    const bool flag,
    const MixedArgumentPayload direct_payload,
    const MixedArgumentPayload indirect_payload,
    const void* optional_pointer) {
    if (blockIdx.x != 0 or threadIdx.x != 0)
        return;

    output[0] = *input + (optional_input == nullptr ? 5 : *optional_input) +
                value_i8 + value_u16 + value_i32 + value_u32 + value_i64 +
                static_cast<long long>(value_u64) + static_cast<long long>(value_f32 * 2) +
                static_cast<long long>(value_f64 * 4) + (flag ? 23 : 0) +
                direct_payload.first + direct_payload.second + direct_payload.third +
                indirect_payload.first + indirect_payload.second + indirect_payload.third +
                (optional_pointer == nullptr ? 53 : 0);
}
