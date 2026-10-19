#include <acl/acl.h>

#ifdef DEEP_JIT_RECORD_LAUNCH

// Record the ACL boundary without loading a binary or running device work.
extern "C" {
int recorded_timeout = -1;
int recorded_ubuf = 0;
int recorded_attribute_count = 0;
int recorded_launch_count = 0;

aclError aclrtLaunchKernelWithHostArgs(
    aclrtFuncHandle, uint32_t, aclrtStream, aclrtLaunchKernelCfg* config,
    void*, size_t, aclrtPlaceHolderInfo*, size_t) {
    recorded_timeout = -1;
    recorded_ubuf = 0;
    recorded_attribute_count = config == nullptr ? 0 : config->numAttrs;
    ++ recorded_launch_count;
    if (config != nullptr) {
        for (size_t index = 0; index < config->numAttrs; ++ index) {
            const auto& attribute = config->attrs[index];
            if (attribute.id == ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT)
                recorded_timeout = attribute.value.timeout;
            if (attribute.id == ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE)
                recorded_ubuf = static_cast<int>(attribute.value.dynUBufSize);
        }
    }
    return ACL_SUCCESS;
}

const char* aclGetRecentErrMsg() { return "recording ACL"; }
aclError aclrtBinaryUnLoad(aclrtBinHandle) { return ACL_SUCCESS; }
aclError aclrtSynchronizeDevice() { return ACL_SUCCESS; }
}

#else

#include <cstdlib>
#include <iostream>
#include <string_view>

#include <deep_jit/backend/ascend/kernel.hpp>

using deep_jit::ascend::LaunchOptions;

int main() try {
    unsetenv("ASCEND_LAUNCH_BLOCKING");
    unsetenv("DJ_JIT_LAUNCH_TIMEOUT");
    unsetenv("ASCEND_LAUNCH_TEST_JIT_LAUNCH_TIMEOUT");
    const deep_jit::Env env("ASCEND_LAUNCH_TEST");
    auto defaults = LaunchOptions::default_options(env);
    defaults.stream = nullptr;
    defaults.num_blocks = 1;

    deep_jit::ascend::Kernel kernel;
    int handle = 0;
    kernel.kernel_handle = &handle;
    const auto recorded = [](const char* name) -> const int& {
        void* symbol = dlsym(deep_jit::ascend::driver::get_acl_handle(), name);
        DJ_HOST_ASSERT(symbol != nullptr, "missing recording symbol: {}", name);
        return *static_cast<const int*>(symbol);
    };
    int checked = 0;
    const auto check = [&](const LaunchOptions& options, const int timeout, const int ubuf) {
        const int previous_launches = recorded("recorded_launch_count");
        kernel.launch(options);
        DJ_HOST_ASSERT(recorded("recorded_launch_count") == previous_launches + 1);
        DJ_HOST_ASSERT(recorded("recorded_timeout") == timeout,
                       "timeout {} was not forwarded to ACL", timeout);
        DJ_HOST_ASSERT(recorded("recorded_ubuf") == ubuf);
        DJ_HOST_ASSERT(recorded("recorded_attribute_count") == 1 + (ubuf > 0));
        ++ checked;
    };

    check(defaults.override_with({}), 300, 0);
    for (const int timeout: {0, 1, 300, 65535}) {
        for (const int ubuf: {0, 4096}) {
            check(defaults.override_with({
                .num_ubuf_bytes = ubuf,
                .num_launch_timeout_secs = timeout,
            }), timeout, ubuf);
        }
    }

    setenv("ASCEND_LAUNCH_TEST_JIT_LAUNCH_TIMEOUT", "0", 1);
    auto from_env = LaunchOptions::default_options(env).override_with({.stream = nullptr, .num_blocks = 1});
    check(from_env, 0, 0);
    check(from_env.override_with({.num_launch_timeout_secs = 1}), 1, 0);
    unsetenv("ASCEND_LAUNCH_TEST_JIT_LAUNCH_TIMEOUT");
    setenv("DJ_JIT_LAUNCH_TIMEOUT", "0", 1);
    from_env = LaunchOptions::default_options(env).override_with({.stream = nullptr, .num_blocks = 1});
    check(from_env, 0, 0);

    for (const auto timeout: {std::optional<int>{}, std::optional<int>{-1}, std::optional<int>{65536}}) {
        auto invalid = defaults;
        invalid.num_launch_timeout_secs = timeout;
        const int previous_launches = recorded("recorded_launch_count");
        bool rejected = false;
        try {
            kernel.launch(invalid);
        } catch (const std::exception& exception) {
            rejected = std::string_view(exception.what()).find("launch timeout is out of range") != std::string_view::npos;
        }
        DJ_HOST_ASSERT(rejected, "invalid timeout must fail before calling ACL");
        DJ_HOST_ASSERT(recorded("recorded_launch_count") == previous_launches);
        ++ checked;
    }
    std::cout << "validated " << checked << " Ascend launch attribute cases\n";
} catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
}

#endif
