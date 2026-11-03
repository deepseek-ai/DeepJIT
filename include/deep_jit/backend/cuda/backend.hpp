#pragma once

#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <deep_jit/backend/cuda/device.hpp>
#include <deep_jit/backend/cuda/kernel.hpp>
#include <deep_jit/backend/cuda/options.hpp>
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

class CUDA {
public:
    using Device = cuda::Device;
    using Kernel = cuda::Kernel;
    using CompilerOptions = cuda::CompilerOptions;
    using LaunchOptions = cuda::LaunchOptions;

    struct CompilerInfo {
        std::filesystem::path path;
        std::string version;

        [[nodiscard]] std::string get_hash() const {
            return hash::get_hex_digest(version);
        }

        [[nodiscard]] json to_json() const {
            return json::object_t {
                {"path", path.string()},
                {"version", version},
            };
        }
    };

    struct Toolkit {
        std::filesystem::path nvcc;
        std::optional<std::filesystem::path> cuobjdump;
    };

    Toolkit toolkit;
    CompilerInfo compiler_info;

    explicit CUDA(const Env& env)
        : toolkit(find_cuda_toolkit(env)),
          compiler_info(get_compiler_info()) {}

    [[nodiscard]] CompilerInfo get_compiler_info() const {
        const auto version = call_external_command(quote_command_path(toolkit.nvcc) + " --version");

        // Should support arch-family
        std::smatch match;
        DJ_HOST_ASSERT(std::regex_search(version, match, std::regex(R"(release (\d+)\.(\d+))")),
                       "failed to parse NVCC version from:\n{}", version);
        const int major = std::stoi(match[1].str());
        const int minor = std::stoi(match[2].str());
        DJ_HOST_ASSERT(major > 12 or (major == 12 and minor >= 9), "NVCC version must be at least 12.9");

        return { .path = toolkit.nvcc, .version = version };
    }

    void compile(const std::string& source,
                 std::filesystem::path dir,
                 const Env& env,
                 const Config& config,
                 const CompilerOptions& options) const {
        // Release GIL to let other Python threads run
        [[maybe_unused]] GilScopedRelease gil_release;

        const bool debug = env.get<bool>("JIT_DEBUG", false);
        const bool print_compiler_command = debug or env.get<bool>("JIT_PRINT_COMPILER_COMMAND", false);

        // Prepare inputs in one IO task. Own captures in case the task times out.
        dir = run_in_iothread([dir, source] {
            const auto path = std::filesystem::absolute(dir).lexically_normal();
            write_file_sync(path / "kernel.cu", source);
            return path;
        });
        const auto source_path = dir / "kernel.cu";
        const auto cubin_path = dir / "kernel.cubin";

        // Build the full command
        std::vector<std::string> args = {
            quote_command_path(toolkit.nvcc),
            quote_command_path(source_path),
            "--cubin",
            "--output-file",
            quote_command_path(cubin_path),
        };
        const auto option_flags = options.get_flags();
        args.insert(args.end(), option_flags.begin(), option_flags.end());
        for (const auto& include_dir: config.include_dirs) {
            args.emplace_back("--include-path");
            args.emplace_back(quote_command_path(include_dir));
        }
        const auto command = str::join(args);

        // NOTES: change directory into a temporary empty directory to prevent same name include files
#if defined(_WIN32)
        const auto cd_command = "cd /D " + quote_command_path(dir) + " && ";
#else
        const auto cd_command = "cd " + dir.string() + " && ";
#endif

        // Compile
        const auto compiler_output = call_external_command(cd_command + command, print_compiler_command);
        if (options.ptxas_verbose.value_or(false)) {
            std::fputs(compiler_output.c_str(), stdout);
            std::fflush(stdout);
        }
        DJ_HOST_ASSERT(not options.check_no_spills.value_or(false) or
                       not std::regex_search(compiler_output, std::regex(R"(spilled\s+to\s+local\s+memory)", std::regex::icase)),
                       "PTXAS reported register spills:\n{}",
                       compiler_output);
        DJ_HOST_ASSERT(not options.check_no_local_memory.value_or(false) or
                       not std::regex_search(compiler_output, std::regex(R"(local\s+memory\s+used)", std::regex::icase)),
                       "PTXAS reported local memory usage:\n{}",
                       compiler_output);

        // Run post hook
        if (options.post_hook) {
            const auto hook_path = config.get_python_path(*options.post_hook);
            const auto hook_command = cd_command +
                                      "python " + quote_command_path(hook_path) + " " + quote_command_path(cubin_path);
            call_external_command(hook_command, print_compiler_command);
        }

        // Dump PTX
        if (options.dump_ptx.value_or(false)) {
            const auto ptx_path = dir / "kernel.ptx";
            auto ptx_args = args;
            ptx_args[2] = "--ptx", ptx_args[4] = quote_command_path(ptx_path);
            call_external_command(cd_command + str::join(ptx_args), print_compiler_command);
        }

        // Dump SASS
        std::string sass;
        if (options.dump_sass.value_or(false)) {
            DJ_HOST_ASSERT(toolkit.cuobjdump.has_value());
            const auto sass_command = quote_command_path(*toolkit.cuobjdump) + " --dump-sass " + quote_command_path(cubin_path);
            sass = call_external_command(cd_command + sass_command, print_compiler_command);
            DJ_HOST_ASSERT(not sass.empty(), "cuobjdump did not produce valid SASS for {}", cubin_path.string());
        }

        // Validate outputs and finish writes in one IO task.
        const json metadata = json::object_t {
            {"command", command},
            {"config", config.to_json()},
            {"compiler_info", compiler_info.to_json()},
            {"compiler_options", options.to_json()},
        };
        run_in_iothread([dir, cubin_path, dump_ptx = options.dump_ptx.value_or(false),
                         sass = std::move(sass), data = metadata.dump()] {
            DJ_HOST_ASSERT(std::filesystem::is_regular_file(cubin_path) and std::filesystem::file_size(cubin_path) != 0,
                           "NVCC did not produce a valid CUBIN: {}", cubin_path.string());
            if (dump_ptx) {
                const auto ptx_path = dir / "kernel.ptx";
                DJ_HOST_ASSERT(std::filesystem::is_regular_file(ptx_path) and std::filesystem::file_size(ptx_path) != 0,
                               "NVCC did not produce a valid PTX: {}", ptx_path.string());
            }
            if (not sass.empty())
                write_file_sync(dir / "kernel.sass", sass);
            write_file_sync(dir / "meta.json", data);
        });
    }

    [[nodiscard]] static std::shared_ptr<Kernel> load(const std::filesystem::path& dir, const Env& env) {
        return Kernel::load(dir, env);
    }

    static Toolkit find_cuda_toolkit(const Env& env) {
        std::filesystem::path home_path;

        // Find CUDA home
        // 1. `CUDA_HOME` or `CUDA_PATH`
        if (home_path.empty()) {
            home_path = get_env<std::string>("CUDA_HOME");
            home_path = home_path.empty() ? std::filesystem::path(get_env<std::string>("CUDA_PATH")) : home_path;
        }

        // 2. Search PATH
        if (home_path.empty()) {
            try {
#if defined(_WIN32)
                auto path = call_external_command("where.exe nvcc.exe");
                if (const auto end = path.find_first_of("\r\n"); end != std::string::npos)
                    path.resize(end);
#else
                auto path = call_external_command("which nvcc");
#endif
                while (not path.empty() and (path.back() == '\r' or path.back() == '\n'))
                    path.pop_back();
                if (not path.empty())
                    home_path = std::filesystem::path(path).parent_path().parent_path();
            } catch (...) {}
        }

#if !defined(_WIN32)
        // 3. /usr/local/cuda
        if (home_path.empty() and std::filesystem::exists("/usr/local/cuda"))
            home_path = "/usr/local/cuda";
#endif

        // Canonicalize
        DJ_HOST_ASSERT(not home_path.empty() and std::filesystem::exists(home_path),
                       "CUDA toolkit not found; set CUDA_HOME or CUDA_PATH to a valid toolkit root");
        home_path = std::filesystem::absolute(home_path).lexically_normal();

        // Find NVCC
        std::filesystem::path nvcc;
        if (const auto path = env.get<std::string>("JIT_NVCC_COMPILER"); path and not path->empty()) {
            nvcc = std::filesystem::absolute(*path).lexically_normal();
        } else {
#if defined(_WIN32)
            nvcc = home_path / "bin" / "nvcc.exe";
#else
            nvcc = home_path / "bin/nvcc";
#endif
        }
        DJ_HOST_ASSERT(is_executable(nvcc), "NVCC compiler is not executable: {}", nvcc.string());

        // Try to find `cuobjdump`
#if defined(_WIN32)
        const auto cuobjdump_path = home_path / "bin" / "cuobjdump.exe";
#else
        const auto cuobjdump_path = home_path / "bin/cuobjdump";
#endif
        std::optional<std::filesystem::path> cuobjdump =
            is_executable(cuobjdump_path) ? std::optional(cuobjdump_path) : std::nullopt;
        return {.nvcc = std::move(nvcc), .cuobjdump = std::move(cuobjdump)};
    }
};

}  // namespace deep_jit
