#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <deep_jit/backend/ascend/device.hpp>
#include <deep_jit/backend/ascend/kernel.hpp>
#include <deep_jit/backend/ascend/options.hpp>
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

class Ascend {
public:
    using Device = ascend::Device;
    using Kernel = ascend::Kernel;
    using CompilerOptions = ascend::CompilerOptions;
    using LaunchOptions = ascend::LaunchOptions;

    struct CompilerInfo {
        std::filesystem::path path;
        std::string version;
        std::filesystem::path linker_path;
        std::string linker_version;

        [[nodiscard]] std::string get_hash() const {
            return hash::FNV1a().update(version).update(linker_version).get_hex_digest();
        }

        [[nodiscard]] json to_json() const {
            return json::object_t {
                {"path", path.string()},
                {"version", version},
                {"linker_path", linker_path.string()},
                {"linker_version", linker_version},
            };
        }
    };

    struct Toolkit {
        std::filesystem::path bisheng;
        std::filesystem::path ld_lld;
        std::filesystem::path asc_devkit_include;
    };

    Toolkit toolkit;
    CompilerInfo compiler_info;
    mutable std::shared_ptr<ascend::AicpuBootstrap> aicpu_bootstrap;

    explicit Ascend(const Env&)
        : toolkit(find_ascend_toolkit()),
          compiler_info(get_compiler_info()) {}

    [[nodiscard]] CompilerInfo get_compiler_info() const {
        return {
            .path = toolkit.bisheng,
            .version = call_external_command(toolkit.bisheng.string() + " --version"),
            .linker_path = toolkit.ld_lld,
            .linker_version = call_external_command(toolkit.ld_lld.string() + " --version"),
        };
    }

    void compile(const std::string& source,
                 std::filesystem::path dir,
                 const Env& env,
                 const Config& config,
                 const CompilerOptions& options) const {
        // Release GIL to let other Python threads run
        [[maybe_unused]] GilScopedRelease gil_release;

        const bool aicpu = options.aicpu.value_or(false);
        const bool print_compiler_command = env.get<bool>("JIT_DEBUG", false) or
                                            env.get<bool>("JIT_PRINT_COMPILER_COMMAND", false);

        // Prepare inputs in one IO task. Own captures in case the task times out.
        dir = run_in_iothread([dir, source, dump_asm = options.dump_asm.value_or(false)] {
            const auto path = std::filesystem::absolute(dir).lexically_normal();
            write_file_sync(path / "kernel.asc", source);
            if (dump_asm)
                make_dirs(path / "asm");
            return path;
        });
        const auto source_path = dir / "kernel.asc";
        const auto relocatable_path = dir / "kernel.rel.o";
        const auto binary_path = dir / (aicpu ? "kernel.aicpu.so" : "kernel.o");

        // Build commands
        std::vector<std::string> base_bisheng_args = {toolkit.bisheng.string()};
        const auto bisheng_flags = options.get_bisheng_flags();
        base_bisheng_args.insert(base_bisheng_args.end(), bisheng_flags.begin(), bisheng_flags.end());
        for (const auto& include_dir: config.include_dirs)
            base_bisheng_args.emplace_back("-I" + include_dir.string());
        base_bisheng_args.emplace_back("-I" + toolkit.asc_devkit_include.string());
        if (aicpu) {
            // HCOMM's ABI headers require AscendC SDK include directories.
            const auto asc_include = toolkit.asc_devkit_include.parent_path();
            for (const auto& include_dir: {asc_include, asc_include / "basic_api", asc_include.parent_path(),
                                           asc_include.parent_path().parent_path() / "include"})
                base_bisheng_args.emplace_back("-I" + include_dir.string());
        }

        auto bisheng_args = base_bisheng_args;
        bisheng_args.insert(bisheng_args.end(), {
            source_path.string(),
            "-o",
            relocatable_path.string(),
        });
        const auto bisheng_command = str::join(bisheng_args);

        std::vector<std::string> linker_args = {toolkit.ld_lld.string()};
        const auto linker_flags = options.get_linker_flags();
        linker_args.insert(linker_args.end(), linker_flags.begin(), linker_flags.end());
        linker_args.insert(linker_args.end(), {
            relocatable_path.string(),
            "-o",
            binary_path.string(),
        });
        const auto linker_command = str::join(linker_args);

        // Compile and link
        const auto cd_command = "cd " + dir.string() + " && ";
        call_external_command(cd_command + bisheng_command, print_compiler_command);
        call_external_command(cd_command + linker_command, print_compiler_command);

        // Dump assembly
        if (options.dump_asm.value_or(false)) {
            const auto asm_dir = dir / "asm";
            auto asm_args = base_bisheng_args;
            asm_args.insert(asm_args.end(), {
                aicpu ? "-S" : "-save-temps",
                source_path.string(),
                "-o",
                (asm_dir / (aicpu ? "kernel.s" : "kernel.rel.o")).string(),
            });
            call_external_command("cd " + asm_dir.string() + " && " + str::join(asm_args),
                                  print_compiler_command);
        }

        // Validate outputs and finish writes in one IO task.
        const json metadata = json::object_t {
            {"command", bisheng_command + " && " + linker_command},
            {"config", config.to_json()},
            {"compiler_info", compiler_info.to_json()},
            {"compiler_options", options.to_json()},
        };
        run_in_iothread([dir, binary_path, relocatable_path, data = metadata.dump()] {
            DJ_HOST_ASSERT(std::filesystem::is_regular_file(binary_path) and std::filesystem::file_size(binary_path) != 0,
                           "Bisheng did not produce a valid Ascend binary: {}", binary_path.string());
            safe_remove_all(relocatable_path);
            write_file_sync(dir / "meta.json", data);
        });
    }

    [[nodiscard]] std::shared_ptr<Kernel> load(const std::filesystem::path& dir,
                                               const Env& env,
                                               const DiskCache& disk_cache) const {
        // Release GIL to let other Python threads run since booting AICPU bootstrap takes some time
        [[maybe_unused]] GilScopedRelease gil_release;
        const bool is_aicpu = run_in_iothread([dir] {
            return std::filesystem::is_regular_file(dir / "kernel.aicpu.so");
        });
        // Initialize bootstrap at the first AICPU kernel load
        if (aicpu_bootstrap == nullptr and is_aicpu)
            aicpu_bootstrap = std::make_shared<ascend::AicpuBootstrap>(
                toolkit.bisheng.parent_path().parent_path(), disk_cache);
        return Kernel::load(dir, env, aicpu_bootstrap);
    }

    static Toolkit find_ascend_toolkit() {
        // Find Ascend home
        std::filesystem::path home_path = get_env<std::string>("ASCEND_HOME_PATH");
        if (home_path.empty())
            home_path = get_env<std::string>("ASCEND_TOOLKIT_HOME");
        if (home_path.empty() and std::filesystem::exists("/usr/local/Ascend/ascend-toolkit/latest"))
            home_path = "/usr/local/Ascend/ascend-toolkit/latest";
        if (home_path.empty() and std::filesystem::exists("/usr/local/Ascend/cann"))
            home_path = "/usr/local/Ascend/cann";

        DJ_HOST_ASSERT(not home_path.empty() and std::filesystem::is_directory(home_path),
                       "Ascend toolkit home was not found");
        home_path = std::filesystem::absolute(home_path).lexically_normal();

        const auto bisheng = home_path / "bin/bisheng";
        const auto ld_lld = home_path / "bin/ld.lld";
        DJ_HOST_ASSERT(is_executable(bisheng), "Bisheng compiler is not executable: {}", bisheng.string());
        DJ_HOST_ASSERT(is_executable(ld_lld), "Ascend linker is not executable: {}", ld_lld.string());

        const auto asc_devkit_include = home_path / "aarch64-linux/asc/include/adv_api";
        DJ_HOST_ASSERT(std::filesystem::is_directory(asc_devkit_include),
                       "Ascend adv_api include directory was not found: {}", asc_devkit_include.string());
        return {
            .bisheng = bisheng,
            .ld_lld = ld_lld,
            .asc_devkit_include = asc_devkit_include,
        };
    }
};

}  // namespace deep_jit
