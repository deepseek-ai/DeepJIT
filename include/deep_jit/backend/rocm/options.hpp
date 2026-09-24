#pragma once

#include <format>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include <deep_jit/backend/rocm/device.hpp>
#include <deep_jit/runtime/config.hpp>
#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/filesystem.hpp>
#include <deep_jit/utils/hash.hpp>
#include <deep_jit/utils/json.hpp>

namespace deep_jit::rocm {

// Per-compile HIPCC options. Unset fields inherit from backend defaults.
struct CompilerOptions {
    std::optional<std::string> optimize_level;
    std::optional<bool> fast_math;
    std::optional<bool> compiler_verbose;
    std::optional<bool> check_no_spills;
    std::optional<bool> check_no_local_memory;
    std::optional<bool> with_line_info;
    std::optional<bool> dump_llvm_ir;
    std::optional<bool> dump_isa;
    std::optional<std::string> arch;
    std::optional<std::vector<std::string>> hipcc_flags;
    std::vector<std::string> extra_hipcc_flags;
    std::optional<std::string> post_hook;

    static CompilerOptions default_options(const Env& env, Device& device) {
        const bool debug = env.get<bool>("JIT_DEBUG", false);
        const bool dump_asm = debug or env.get<bool>("JIT_DUMP_ASM", false);
        return {
            .optimize_level = "3",
            .fast_math = false,
            .compiler_verbose = debug or env.get<bool>("JIT_HIPCC_VERBOSE", false),
            .check_no_spills = env.get<bool>("JIT_CHECK_NO_SPILLS", false),
            .check_no_local_memory = env.get<bool>("JIT_CHECK_NO_LOCAL_MEMORY", false),
            .with_line_info = debug or env.get<bool>("JIT_WITH_LINEINFO", false),
            .dump_llvm_ir = dump_asm or env.get<bool>("JIT_DUMP_LLVM_IR", false),
            .dump_isa = dump_asm or env.get<bool>("JIT_DUMP_ISA", false),
            .arch = device.get_arch(),
            .hipcc_flags = std::vector<std::string> {
                std::format("-std=c++{}", env.get<int>("JIT_CPP_STANDARD", 20)),
            },
            .post_hook = std::nullopt
        };
    }

    [[nodiscard]] CompilerOptions override_with(const CompilerOptions& overrides) const {
        CompilerOptions result = *this;
        if (overrides.optimize_level)
            result.optimize_level = overrides.optimize_level;
        if (overrides.fast_math)
            result.fast_math = overrides.fast_math;
        if (overrides.compiler_verbose)
            result.compiler_verbose = overrides.compiler_verbose;
        if (overrides.check_no_spills)
            result.check_no_spills = overrides.check_no_spills;
        if (overrides.check_no_local_memory)
            result.check_no_local_memory = overrides.check_no_local_memory;
        if (overrides.with_line_info)
            result.with_line_info = overrides.with_line_info;
        if (overrides.dump_llvm_ir)
            result.dump_llvm_ir = overrides.dump_llvm_ir;
        if (overrides.dump_isa)
            result.dump_isa = overrides.dump_isa;
        if (overrides.arch)
            result.arch = overrides.arch;
        if (overrides.hipcc_flags)
            result.hipcc_flags = overrides.hipcc_flags;
        result.extra_hipcc_flags.insert(result.extra_hipcc_flags.end(), overrides.extra_hipcc_flags.begin(), overrides.extra_hipcc_flags.end());

        if (overrides.post_hook)
            result.post_hook = overrides.post_hook;
        return result;
    }

    [[nodiscard]] std::vector<std::string> get_flags() const {
        // Use a single explicit AMD target; never fall back to a host-selected GPU.
        DJ_HOST_ASSERT(arch.has_value() and
                       std::regex_match(*arch, std::regex(R"(gfx[0-9a-f]+(:[a-zA-Z0-9_]+[+-])*)")),
                       "ROCm architecture must be an AMD target identifier (gfx...)");
        DJ_HOST_ASSERT(optimize_level.has_value() and
                       (*optimize_level == "0" or *optimize_level == "1" or *optimize_level == "2" or
                        *optimize_level == "3" or *optimize_level == "s" or *optimize_level == "z"),
                       "unsupported HIP optimization level");
        std::vector<std::string> flags = {"--offload-arch=" + *arch, "-O" + *optimize_level, "--no-gpu-bundle-output"};
        if (fast_math.value_or(false))
            flags.emplace_back("-ffast-math");
        if (compiler_verbose.value_or(false))
            flags.emplace_back("-v");
        if (with_line_info.value_or(false))
            flags.emplace_back("-gline-tables-only");

        // Other flags
        if (hipcc_flags)
            flags.insert(flags.end(), hipcc_flags->begin(), hipcc_flags->end());
        flags.insert(flags.end(), extra_hipcc_flags.begin(), extra_hipcc_flags.end());
        // HIPCC may ignore some CUDA-only switches. Do not advertise them as supported.
        for (const auto& flag: flags) {
            DJ_HOST_ASSERT(not flag.starts_with("--ptxas") and not flag.starts_with("-Xptxas") and
                           not flag.starts_with("--gpu-architecture") and not flag.starts_with("-arch=") and
                           flag != "--ptx" and flag != "--cubin",
                           "unsupported CUDA compiler option in ROCm backend: {}", flag);
        }
        return flags;
    }

    void update_hash(hash::FNV1a& hasher) const {
        // Preserve argument boundaries, including spaces within a single flag.
        hasher.update("rocm-options-v1");
        for (const auto& flag: get_flags())
            hasher.update(flag);
        // These requests do not change the HSACO flags, but must not be skipped on a cache hit.
        hasher.update(json(json::array_t {
            check_no_spills.value_or(false), check_no_local_memory.value_or(false),
            dump_llvm_ir.value_or(false), dump_isa.value_or(false),
        }).dump());
    }

    [[nodiscard]] std::string get_post_hook_hash(const Config& config) const {
        if (not post_hook)
            return "";
        const auto path = config.get_python_path(*post_hook);
        return hash::FNV1a().update(*post_hook).update(read(path)).get_hex_digest();
    }

    [[nodiscard]] json to_json() const {
        return json::object_t {
            {"optimize_level", optimize_level},
            {"use_fast_math", fast_math},
            {"compiler_verbose", compiler_verbose},
            {"check_no_spills", check_no_spills},
            {"check_no_local_memory", check_no_local_memory},
            {"with_line_info", with_line_info},
            {"dump_llvm_ir", dump_llvm_ir},
            {"dump_isa", dump_isa},
            {"arch", arch},
            {"hipcc_flags", hipcc_flags},
            {"extra_hipcc_flags", extra_hipcc_flags},
            {"post_hook", post_hook},
        };
    }
};

// Per-launch ROCm options. Unset fields inherit from runtime defaults.
struct LaunchOptions {
    // Use current stream if unset
    std::optional<hipStream_t> stream;

    // Dynamic shared memory size
    std::optional<int> num_smem_bytes;

    // Grid
    std::optional<dim3> grid_dim;
    std::optional<dim3> block_dim;
    std::optional<dim3> cluster_dim;

    // Cooperative launch
    std::optional<bool> cooperative;

    // CUDA-only requests are accepted as fields solely to reject non-default values.
    // Dependent kernel launch
    std::optional<bool> enable_pdl;

    // Others
    std::optional<bool> nonportable_cluster_size_allowed;

    static LaunchOptions default_options(const Env&) {
        return {
            .stream = std::nullopt,
            .num_smem_bytes = 0,
            .grid_dim = std::nullopt,
            .block_dim = std::nullopt,
            .cluster_dim = dim3(1, 1, 1),
            .cooperative = false,
            .enable_pdl = false,
            .nonportable_cluster_size_allowed = false,
        };
    }

    [[nodiscard]] LaunchOptions override_with(const LaunchOptions& overrides) const {
        LaunchOptions result = *this;
        if (overrides.stream)
            result.stream = overrides.stream;
        if (overrides.num_smem_bytes)
            result.num_smem_bytes = overrides.num_smem_bytes;
        if (overrides.grid_dim)
            result.grid_dim = overrides.grid_dim;
        if (overrides.block_dim)
            result.block_dim = overrides.block_dim;
        if (overrides.cluster_dim)
            result.cluster_dim = overrides.cluster_dim;
        if (overrides.cooperative)
            result.cooperative = overrides.cooperative;
        if (overrides.enable_pdl)
            result.enable_pdl = overrides.enable_pdl;
        if (overrides.nonportable_cluster_size_allowed)
            result.nonportable_cluster_size_allowed = overrides.nonportable_cluster_size_allowed;
        return result;
    }
};

}  // namespace deep_jit::rocm
