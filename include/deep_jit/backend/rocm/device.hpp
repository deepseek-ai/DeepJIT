#pragma once

#include <cstdint>
#include <string>

#include <hip/hip_runtime.h>

#include <deep_jit/backend/rocm/driver.hpp>

namespace deep_jit::rocm {

class Device {
    hipDeviceProp_t prop{};
    bool initialized = false;

public:
    const hipDeviceProp_t& get_prop() {
        if (not initialized) {
            // Establish the current context only when the device is first used.
            int device_index = 0;
            DJ_HIP_CHECK(hipGetDevice(&device_index));
            DJ_HIP_CHECK(hipFree(nullptr));
            DJ_HIP_CHECK(hipGetDeviceProperties(&prop, device_index));
            initialized = true;
        }
        return prop;
    }

    int get_num_sms() { return get_prop().multiProcessorCount; }

    int get_num_l2_cache_bytes() { return get_prop().l2CacheSize; }

    int get_num_smem_bytes() { return static_cast<int>(get_prop().sharedMemPerBlock); }

    int64_t get_clock_rate() { return static_cast<int64_t>(get_prop().clockRate) * 1000; }

    int get_warp_size() { return get_prop().warpSize; }

    std::string get_arch() {
        // Preserve target features (e.g. :xnack-), not CUDA compute capability.
        const std::string arch = get_prop().gcnArchName;
        DJ_HOST_ASSERT(arch.starts_with("gfx"), "invalid AMD GPU target identifier: {}", arch);
        return arch;
    }
};

}  // namespace deep_jit::rocm
