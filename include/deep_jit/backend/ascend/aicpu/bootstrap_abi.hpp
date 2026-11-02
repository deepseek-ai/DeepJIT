#pragma once

#include <cstdint>
#include <string_view>

namespace deep_jit::ascend::aicpu {

// Define the shared Host/Device ABI once and embed it for the device compiler.
#define DJ_AICPU_ABI(...) \
    __VA_ARGS__ \
    inline constexpr std::string_view abi_source = \
        "#include <cstdint>\nnamespace deep_jit::ascend::aicpu {\n" #__VA_ARGS__ "\n}\n";

DJ_AICPU_ABI(
inline constexpr const char* entry = "deep_jit_aicpu_loader_v1";

enum class Operation : uint64_t { Load, Invoke, Unload, RemoveBootstrap };

struct alignas(16) RequestHeader {
    Operation operation = Operation::Load;
    uint64_t module = 0;
};

struct LoaderRequest {
    RequestHeader header;
    uint64_t image = 0;
    uint64_t image_size = 0;
    uint64_t result = 0;
    char symbol[128]{};
};

struct LoaderResult {
    uint64_t module = 0;
    int32_t status = 0;
    int32_t sys_errno = 0;
    char message[256]{};
};

static_assert(sizeof(RequestHeader) == 16);
static_assert(sizeof(LoaderRequest) == 176);
static_assert(sizeof(LoaderResult) == 272);
)

#undef DJ_AICPU_ABI

}  // namespace deep_jit::ascend::aicpu
