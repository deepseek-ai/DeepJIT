#pragma once

#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <deep_jit/backend/rocm/device.hpp>
#include <deep_jit/backend/rocm/kernel.hpp>
#include <deep_jit/backend/rocm/options.hpp>
#include <deep_jit/runtime/config.hpp>
#include <deep_jit/runtime/runtime.hpp>
#include <deep_jit/utils/command.hpp>
#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/filesystem.hpp>
#include <deep_jit/utils/gil.hpp>
#include <deep_jit/utils/hash.hpp>
#include <deep_jit/utils/json.hpp>
#include <deep_jit/utils/str.hpp>

namespace deep_jit {

class ROCm {
public:
    using Device = rocm::Device;
    using Kernel = rocm::Kernel;
    using CompilerOptions = rocm::CompilerOptions;
    using LaunchOptions = rocm::LaunchOptions;

    struct CompilerInfo {
        std::filesystem::path path;
        std::string version;
        std::vector<std::string> environment;

        [[nodiscard]] std::string get_hash() const {
            auto hasher = hash::FNV1a().update("rocm-hipcc-v1").update(path.string()).update(version);
            for (const auto& value: environment)
                hasher.update(value);
            return hasher.get_hex_digest();
        }

        [[nodiscard]] json to_json() const {
            return json::object_t {
                {"path", path.string()},
                {"version", version},
                {"environment", environment},
            };
        }
    };

    struct Toolkit {
        std::filesystem::path hipcc;
        std::optional<std::filesystem::path> llvm_objdump;
        std::optional<std::filesystem::path> llvm_readobj;
    };

    Toolkit toolkit;
    CompilerInfo compiler_info;

    explicit ROCm(const Env& env)
        : toolkit(find_rocm_toolkit(env)),
          compiler_info(get_compiler_info()) {}

    // Quote individual arguments locally; do not change other backends' commands.
    static std::string quote(const std::string& value) {
        std::string result = "'";
        for (const auto c: value)
            result += c == '\'' ? "'\\''" : std::string(1, c);
        return result + "'";
    }

    static std::string command_string(const std::vector<std::string>& args) {
        std::vector<std::string> quoted;
        quoted.reserve(args.size());
        for (const auto& arg: args)
            quoted.emplace_back(quote(arg));
        return str::join(quoted);
    }

    [[nodiscard]] CompilerInfo get_compiler_info() const {
        const auto version = call_external_command("HIP_PLATFORM=amd " + quote(toolkit.hipcc.string()) + " --version");
        DJ_HOST_ASSERT(std::regex_search(version, std::regex(R"(HIP version:\s*\d+\.\d+)")) and
                       version.find("clang version") != std::string::npos,
                       "expected AMD HIPCC with Clang; compiler reported:\n{}", version);
        // ROCm SDK and HIP component major versions need not be the same.
        // The required library API is checked lazily against the loaded runtime.
        std::vector<std::string> environment;
        for (const auto* name: {"HIPCC_COMPILE_FLAGS_APPEND", "HIPCC_LINK_FLAGS_APPEND", "HIP_CLANG_PATH",
                               "HIP_DEVICE_LIB_PATH", "HIP_PATH", "ROCM_PATH", "CPATH", "CPLUS_INCLUDE_PATH",
                               "C_INCLUDE_PATH"}) {
            environment.emplace_back(std::string(name) + "=" + get_env<std::string>(name));
        }
        return {.path = toolkit.hipcc, .version = version, .environment = std::move(environment)};
    }

    // Consume LLVM's decoded AMDGPU metadata, not the ELF/MessagePack binary.
    // Validate every kernel separately: one kernel's zero must not mask another's missing field.
    static void validate_metadata(const std::string& metadata, const bool no_spills, const bool no_local_memory) {
        const std::regex list_pattern(R"((^|\n)([ \t]*)amdhsa\.kernels:[ \t]*([^\r\n]*))");
        std::smatch list;
        DJ_HOST_ASSERT(std::regex_search(metadata, list, list_pattern), "missing AMDGPU kernel metadata for requested checks");
        const auto rest = metadata.substr(list.position() + list.length());
        DJ_HOST_ASSERT(not std::regex_search(rest, list_pattern), "multiple AMDGPU metadata notes are not supported by safety checks");
        if (list[3].str() == "[]")
            return;
        DJ_HOST_ASSERT(list[3].str().empty(), "unrecognized AMDGPU kernel metadata list");

        const auto list_indent = list[2].length();
        std::optional<std::size_t> kernel_indent;
        std::vector<std::string> kernels;
        std::istringstream lines(rest);
        for (std::string line; std::getline(lines, line);) {
            const auto indent = line.find_first_not_of(" \t\r");
            if (indent == std::string::npos)
                continue;
            if (indent <= list_indent)
                break;
            if (not kernel_indent) {
                DJ_HOST_ASSERT(line.substr(indent).starts_with("- "), "unrecognized AMDGPU kernel metadata entry");
                kernel_indent = indent;
            }
            if (indent == *kernel_indent and line.substr(indent).starts_with("- "))
                kernels.emplace_back();
            DJ_HOST_ASSERT(not kernels.empty(), "missing AMDGPU kernel metadata entry");
            kernels.back() += line + "\n";
        }
        DJ_HOST_ASSERT(not kernels.empty(), "missing AMDGPU kernel metadata for requested checks");
        for (const auto& kernel: kernels) {
            const auto field = [&](const std::string& name, const bool required = true) {
                const std::regex pattern("(^|\\n)[ \\t]*(-[ \\t]+)?\\." + name + ":[ \\t]*([^\\r\\n]*)");
                std::smatch match;
                if (not std::regex_search(kernel, match, pattern)) {
                    DJ_HOST_ASSERT(not required, "missing AMDGPU .{} metadata for requested check", name);
                    return std::string();
                }
                const auto tail = kernel.substr(match.position() + match.length());
                DJ_HOST_ASSERT(not std::regex_search(tail, pattern), "duplicate AMDGPU .{} metadata", name);
                auto value = match[3].str();
                const auto end = value.find_last_not_of(" \t\r");
                DJ_HOST_ASSERT(end != std::string::npos, "invalid empty AMDGPU .{} metadata", name);
                return value.substr(0, end + 1);
            };
            DJ_HOST_ASSERT(not field("symbol").empty(), "missing AMDGPU kernel symbol metadata");
            const auto check_zero = [&](const std::string& name) {
                const auto value = field(name);
                DJ_HOST_ASSERT(std::regex_match(value, std::regex("[0-9]+")), "invalid AMDGPU .{} metadata: {}", name, value);
                DJ_HOST_ASSERT(value.find_first_not_of('0') == std::string::npos,
                               "AMDGPU .{} is {} (requested zero)", name, value);
            };
            if (no_spills) {
                check_zero("sgpr_spill_count");
                check_zero("vgpr_spill_count");
            }
            if (no_local_memory) {
                check_zero("private_segment_fixed_size");
                const auto dynamic_stack = field("uses_dynamic_stack", false);
                DJ_HOST_ASSERT(dynamic_stack.empty() or dynamic_stack == "false",
                               "AMDGPU dynamic private stack is not allowed: {}", dynamic_stack);
            }
        }
    }

    void compile(const std::string& source,
                 std::filesystem::path dir,
                 const Env& env,
                 const Config& config,
                 const CompilerOptions& options) const {
        GilScopedRelease gil_release;

        dir = std::filesystem::absolute(dir).lexically_normal();
        const auto source_path = dir / "kernel.hip";
        const auto hsaco_path = dir / "kernel.hsaco";
        const bool debug = env.get<bool>("JIT_DEBUG", false);
        const bool print_command = debug or env.get<bool>("JIT_PRINT_COMPILER_COMMAND", false);
        const auto validate_artifact = [](const std::filesystem::path& path) {
            DJ_HOST_ASSERT(std::filesystem::is_regular_file(path) and std::filesystem::file_size(path) != 0,
                           "HIP toolchain did not produce a nonempty artifact: {}", path.string());
        };

        write_file_sync(source_path, source);
        std::vector<std::string> args = {toolkit.hipcc.string(), "-x", "hip", source_path.string(),
                                         "--genco", "-o", hsaco_path.string()};
        const auto flags = options.get_flags();
        args.insert(args.end(), flags.begin(), flags.end());
        for (const auto& include_dir: config.include_dirs) {
            args.emplace_back("-I");
            args.emplace_back(include_dir.string());
        }
        // Use a temporary empty working directory, as with the existing backends.
        const auto cd_command = "cd " + quote(dir.string()) + " && ";
        const auto command = "HIP_PLATFORM=amd " + command_string(args);
        const auto output = call_external_command(cd_command + command, print_command);
        write_file_sync(dir / "compiler.log", output);
        if (options.compiler_verbose.value_or(false)) {
            std::fputs(output.c_str(), stdout);
            std::fflush(stdout);
        }
        validate_artifact(hsaco_path);

        if (options.post_hook) {
            call_external_command(cd_command + command_string({"python", config.get_python_path(*options.post_hook).string(),
                                                                hsaco_path.string()}), print_command);
            validate_artifact(hsaco_path);
        }

        // Inspect the final artifact, after any post-compilation hook.
        if (options.check_no_spills.value_or(false) or options.check_no_local_memory.value_or(false)) {
            DJ_HOST_ASSERT(toolkit.llvm_readobj.has_value(), "llvm-readobj is required for ROCm spill/private-memory checks");
            const auto notes = call_external_command(cd_command + command_string(
                {toolkit.llvm_readobj->string(), "--notes", hsaco_path.string()}), print_command);
            write_file_sync(dir / "kernel.metadata", notes);
            validate_metadata(notes, options.check_no_spills.value_or(false), options.check_no_local_memory.value_or(false));
        }
        if (options.dump_llvm_ir.value_or(false)) {
            auto ir_args = args;
            ir_args[6] = (dir / "kernel.ll").string();
            ir_args.insert(ir_args.end(), {"-S", "-emit-llvm"});
            call_external_command(cd_command + "HIP_PLATFORM=amd " + command_string(ir_args), print_command);
            validate_artifact(dir / "kernel.ll");
        }
        if (options.dump_isa.value_or(false)) {
            DJ_HOST_ASSERT(toolkit.llvm_objdump.has_value(), "llvm-objdump is required for ROCm ISA dumps");
            const auto isa = call_external_command(cd_command + command_string(
                {toolkit.llvm_objdump->string(), "--disassemble", "--mcpu=" + options.arch->substr(0, options.arch->find(':')),
                 hsaco_path.string()}), print_command);
            DJ_HOST_ASSERT(not isa.empty(), "llvm-objdump did not produce an ISA dump");
            write_file_sync(dir / "kernel.isa", isa);
        }

        const json metadata = json::object_t {
            {"backend", "ROCm"},
            {"command", command},
            {"config", config.to_json()},
            {"compiler_info", compiler_info.to_json()},
            {"compiler_options", options.to_json()},
        };
        write_file_sync(dir / "meta.json", metadata.dump());
    }

    [[nodiscard]] static std::shared_ptr<Kernel> load(const std::filesystem::path& dir, const Env& env) {
        return Kernel::load(dir, env);
    }

    static std::optional<std::filesystem::path> find_executable(const std::filesystem::path& name) {
        if (name.empty())
            return std::nullopt;
        if (name.has_parent_path()) {
            const auto path = std::filesystem::absolute(name).lexically_normal();
            return is_executable(path) ? std::optional(path) : std::nullopt;
        }
        std::istringstream paths(get_env<std::string>("PATH"));
        for (std::string dir; std::getline(paths, dir, ':');) {
            const auto path = std::filesystem::absolute(std::filesystem::path(dir) / name).lexically_normal();
            if (is_executable(path))
                return path;
        }
        return std::nullopt;
    }

    static Toolkit find_rocm_toolkit(const Env& env) {
        std::filesystem::path home;
        for (const auto* name: {"ROCM_HOME", "ROCM_PATH", "HIP_PATH"}) {
            if (home.empty())
                home = get_env<std::string>(name);
        }
        std::optional<std::filesystem::path> hipcc;
        if (const auto compiler = env.get<std::string>("JIT_HIPCC_COMPILER")) {
            hipcc = find_executable(*compiler);
            DJ_HOST_ASSERT(hipcc.has_value(), "HIPCC compiler is not executable: {}", *compiler);
        } else if (not home.empty()) {
            hipcc = find_executable(home / "bin/hipcc");
            DJ_HOST_ASSERT(hipcc.has_value(), "HIPCC compiler is not executable under {}", home.string());
        } else {
            hipcc = find_executable("hipcc");
            for (const auto* prefix: {"/opt/rocm/core-10.0", "/opt/rocm", "/opt/rocm/core"}) {
                if (not hipcc)
                    hipcc = find_executable(std::filesystem::path(prefix) / "bin/hipcc");
            }
        }
        DJ_HOST_ASSERT(hipcc.has_value(), "ROCm HIPCC was not found; set ROCM_PATH or JIT_HIPCC_COMPILER with your library prefix");

        std::vector<std::filesystem::path> roots = {hipcc->parent_path().parent_path(),
                                                   std::filesystem::canonical(*hipcc).parent_path().parent_path()};
        if (not home.empty())
            roots.emplace_back(home);
        const auto find_tool = [&](const std::string& suffix, const std::string& name) -> std::optional<std::filesystem::path> {
            if (const auto override = env.get<std::string>(suffix)) {
                const auto path = find_executable(*override);
                DJ_HOST_ASSERT(path.has_value(), "{} is not executable: {}", name, *override);
                return path;
            }
            for (const auto& root: roots) {
                for (const auto* bin: {"bin", "llvm/bin", "lib/llvm/bin"}) {
                    if (const auto path = find_executable(root / bin / name))
                        return path;
                }
            }
            if (const auto clang_path = get_env<std::string>("HIP_CLANG_PATH"); not clang_path.empty()) {
                if (const auto path = find_executable(std::filesystem::path(clang_path) / name))
                    return path;
            }
            return find_executable(name);
        };
        return {.hipcc = *hipcc,
                .llvm_objdump = find_tool("JIT_LLVM_OBJDUMP", "llvm-objdump"),
                .llvm_readobj = find_tool("JIT_LLVM_READOBJ", "llvm-readobj")};
    }
};

}  // namespace deep_jit
