#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <acl/acl.h>

#include <deep_jit/backend/ascend/device.hpp>
#include <deep_jit/runtime/config.hpp>
#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/hash.hpp>
#include <deep_jit/utils/json.hpp>
#include <deep_jit/utils/str.hpp>

namespace deep_jit::ascend {

// Per-compile Bisheng options. Unset fields inherit from backend defaults.
struct CompilerOptions {
    std::optional<bool> aicpu;
    std::optional<std::string> optimize_level;
    std::optional<int> arch;
    std::optional<bool> debug_info;
    std::optional<bool> dump_asm;
    std::optional<std::vector<std::string>> bisheng_flags;
    std::optional<std::vector<std::string>> linker_flags;
    std::optional<std::vector<std::string>> aicpu_flags;
    std::optional<std::vector<std::string>> aicpu_linker_flags;

    std::vector<std::string> extra_bisheng_flags;
    std::vector<std::string> extra_linker_flags;
    std::vector<std::string> extra_aicpu_flags;
    std::vector<std::string> extra_aicpu_linker_flags;

    static CompilerOptions default_options(const Env& env, Device& device) {
        return {
            .aicpu = false,
            .optimize_level = "2",
            .arch = device.get_npu_arch(),
            .debug_info = env.get<bool>("JIT_KERNEL_DEBUG_INFO", false),
            .dump_asm = env.get<bool>("JIT_DUMP_ASM", false),
            .bisheng_flags = std::vector<std::string> {
                "--cce-aicore-only",
                "-mllvm", "-enable-hiipu-vf-loop-unroll",
                "-mllvm", "-cce-aicore-dcpreload-args=false",
            },
            .linker_flags = std::vector<std::string> {
                "-m", "aicorelinux",
                "-Ttext", "0",
                "--no-mmap-output-file",
            },
            .aicpu_flags = std::vector<std::string> {
                "--cce-aicore-lang", "--cce-lite-aicpu", "--cce-aicpu-only",
                "--cce-aicpu-launch-with-interface",
                // CANN 9.2 requires an AICore arch even for the Cortex-A55 pass.
                "--cce-aicore-arch=dav-c310-vec", "-fPIC", "-c",
            },
            .aicpu_linker_flags = std::vector<std::string> {
                // Keep function addresses nonzero and bind local definitions
                // so JIT variants do not interpose through RTLD_GLOBAL.
                "-m", "aicpulinux", "-shared", "-soname=libkernel.aicpu.so",
                "-Bsymbolic", "--no-mmap-output-file",
            },
        };
    }

    [[nodiscard]] CompilerOptions override_with(const CompilerOptions& overrides) const {
        CompilerOptions result = *this;
        if (overrides.aicpu)
            result.aicpu = overrides.aicpu;
        if (overrides.optimize_level)
            result.optimize_level = overrides.optimize_level;
        if (overrides.arch)
            result.arch = overrides.arch;
        if (overrides.debug_info)
            result.debug_info = overrides.debug_info;
        if (overrides.dump_asm)
            result.dump_asm = overrides.dump_asm;
        if (overrides.bisheng_flags)
            result.bisheng_flags = overrides.bisheng_flags;
        if (overrides.linker_flags)
            result.linker_flags = overrides.linker_flags;
        if (overrides.aicpu_flags)
            result.aicpu_flags = overrides.aicpu_flags;
        if (overrides.aicpu_linker_flags)
            result.aicpu_linker_flags = overrides.aicpu_linker_flags;
        result.extra_bisheng_flags.insert(
            result.extra_bisheng_flags.end(),
            overrides.extra_bisheng_flags.begin(),
            overrides.extra_bisheng_flags.end());
        result.extra_linker_flags.insert(
            result.extra_linker_flags.end(),
            overrides.extra_linker_flags.begin(),
            overrides.extra_linker_flags.end());
        result.extra_aicpu_flags.insert(
            result.extra_aicpu_flags.end(),
            overrides.extra_aicpu_flags.begin(),
            overrides.extra_aicpu_flags.end());
        result.extra_aicpu_linker_flags.insert(
            result.extra_aicpu_linker_flags.end(),
            overrides.extra_aicpu_linker_flags.begin(),
            overrides.extra_aicpu_linker_flags.end());
        return result;
    }

    [[nodiscard]] std::vector<std::string> get_bisheng_flags() const {
        DJ_HOST_ASSERT(optimize_level.has_value() and not optimize_level->empty(),
                       "optimization level must be specified");
        if (not aicpu.value_or(false))
            DJ_HOST_ASSERT(arch.has_value() and *arch > 0, "Ascend architecture must be specified");

        std::vector<std::string> flags = {
            "-O" + *optimize_level,
        };
        if (not aicpu.value_or(false))
            flags.emplace_back("--npu-arch=dav-" + std::to_string(*arch));
        if (debug_info.value_or(false)) {
            flags.insert(flags.end(), {
                "-g",
                "-gdwarf-5",
                "-gembed-source",
                "-gline-tables-only",
            });
        }
        const auto& mode_flags = aicpu.value_or(false) ? aicpu_flags : bisheng_flags;
        if (mode_flags)
            flags.insert(flags.end(), mode_flags->begin(), mode_flags->end());
        const auto& extra_flags = aicpu.value_or(false) ? extra_aicpu_flags : extra_bisheng_flags;
        flags.insert(flags.end(), extra_flags.begin(), extra_flags.end());
        return flags;
    }

    [[nodiscard]] std::vector<std::string> get_linker_flags() const {
        std::vector<std::string> flags;
        const auto& mode_flags = aicpu.value_or(false) ? aicpu_linker_flags : linker_flags;
        if (mode_flags)
            flags = *mode_flags;
        const auto& extra_flags = aicpu.value_or(false) ? extra_aicpu_linker_flags : extra_linker_flags;
        flags.insert(flags.end(), extra_flags.begin(), extra_flags.end());
        return flags;
    }

    void update_hash(hash::FNV1a& hasher) const {
        if (aicpu.value_or(false))
            hasher.update("aicpu");
        hasher.update(str::join(get_bisheng_flags()));
        hasher.update(str::join(get_linker_flags()));
    }

    [[nodiscard]] std::string get_post_hook_hash(const Config&) const { return {}; }

    [[nodiscard]] json to_json() const {
        return json::object_t {
            {"aicpu", aicpu},
            {"optimize_level", optimize_level},
            {"arch", arch},
            {"debug_info", debug_info},
            {"dump_asm", dump_asm},
            {"bisheng_flags", bisheng_flags},
            {"linker_flags", linker_flags},
            {"aicpu_flags", aicpu_flags},
            {"aicpu_linker_flags", aicpu_linker_flags},
            {"extra_bisheng_flags", extra_bisheng_flags},
            {"extra_linker_flags", extra_linker_flags},
            {"extra_aicpu_flags", extra_aicpu_flags},
            {"extra_aicpu_linker_flags", extra_aicpu_linker_flags},
        };
    }
};

// Per-launch Ascend options. Unset fields inherit from runtime defaults.
struct LaunchOptions {
    // Use current torch_npu stream if unset
    std::optional<aclrtStream> stream;

    std::optional<int> num_blocks;
    std::optional<int> num_ubuf_bytes;
    std::optional<int> num_launch_timeout_secs;

    static LaunchOptions default_options(const Env& env) {
        return {
            .stream = std::nullopt,
            .num_blocks = std::nullopt,
            .num_ubuf_bytes = 0,
            .num_launch_timeout_secs = env.get<int>("JIT_LAUNCH_TIMEOUT", 300),
        };
    }

    [[nodiscard]] LaunchOptions override_with(const LaunchOptions& overrides) const {
        LaunchOptions result = *this;
        if (overrides.stream)
            result.stream = overrides.stream;
        if (overrides.num_blocks)
            result.num_blocks = overrides.num_blocks;
        if (overrides.num_ubuf_bytes)
            result.num_ubuf_bytes = overrides.num_ubuf_bytes;
        if (overrides.num_launch_timeout_secs)
            result.num_launch_timeout_secs = overrides.num_launch_timeout_secs;
        return result;
    }
};

}  // namespace deep_jit::ascend
