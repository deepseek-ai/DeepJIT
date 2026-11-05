#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <elf.h>
#include <filesystem>
#include <format>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <deep_jit/backend/ascend/aicpu/bootstrap.hpp>
#include <deep_jit/backend/ascend/driver.hpp>
#include <deep_jit/backend/ascend/options.hpp>
#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/executor.hpp>
#include <deep_jit/utils/filesystem.hpp>
#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/gil.hpp>

namespace deep_jit::ascend {

template <typename T>
inline constexpr size_t kernel_arg_alignment = alignof(T) < 4 ? 4 : alignof(T);

template <typename T>
constexpr size_t align_kernel_arg(const size_t offset) {
    return (offset + kernel_arg_alignment<T> - 1) & ~(kernel_arg_alignment<T> - 1);
}

// Immutable Ascend kernel handles with shared ownership. Driver resources are
// unloaded when the last shared owner is destroyed.
class Kernel {
    bool aicpu = false;
    // Opaque handle to a loaded module in the AICPU scheduler process.
    uint64_t aicpu_module = 0;
    std::shared_ptr<AicpuBootstrap> aicpu_bootstrap;

public:
    aclrtBinHandle binary_handle{};
    // ACL launch handle: the custom kernel for AICore, or the loader for AICPU.
    aclrtFuncHandle kernel_handle{};

    Kernel() = default;

    ~Kernel() { unload(); }
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;
    Kernel(Kernel&&) = delete;
    Kernel& operator=(Kernel&&) = delete;

    static std::vector<std::string> parse_kernel_names(const std::string& data) {
        // Read header
        const auto file_size = data.size();
        DJ_HOST_ASSERT(file_size >= sizeof(Elf64_Ehdr),
                       "invalid Ascend binary");

        Elf64_Ehdr header{};
        std::memcpy(&header, data.data(), sizeof(header));
        DJ_HOST_ASSERT(std::memcmp(header.e_ident, ELFMAG, SELFMAG) == 0,
                       "not an ELF file");
        DJ_HOST_ASSERT(header.e_ident[EI_CLASS] == ELFCLASS64 and header.e_ident[EI_DATA] == ELFDATA2LSB,
                       "unsupported ELF format");
        DJ_HOST_ASSERT(header.e_shentsize == sizeof(Elf64_Shdr) and header.e_shnum > 0,
                       "invalid ELF section table");

        // Read sections
        const auto section_table_size = static_cast<uint64_t>(header.e_shnum) * sizeof(Elf64_Shdr);
        DJ_HOST_ASSERT(header.e_shoff <= static_cast<uint64_t>(file_size) and
                       section_table_size <= static_cast<uint64_t>(file_size) - header.e_shoff,
                       "truncated ELF section table");
        std::vector<Elf64_Shdr> sections(header.e_shnum);
        std::memcpy(sections.data(), data.data() + header.e_shoff, section_table_size);

        if (header.e_type == ET_DYN and header.e_machine == EM_AARCH64) {
            const auto check_range = [&](uint64_t offset, uint64_t size) {
                DJ_HOST_ASSERT(offset <= data.size() and size <= data.size() - offset,
                               "truncated AICPU ELF");
            };
            // Match exported <name>/<name>_interface function pairs, as Bisheng
            // generates them only for __aicpu__ kernels, not ordinary helpers.
            std::set<std::string> symbols;
            for (const auto& section: sections) {
                if (section.sh_type != SHT_DYNSYM)
                    continue;
                DJ_HOST_ASSERT(section.sh_entsize == sizeof(Elf64_Sym) and
                               section.sh_size % sizeof(Elf64_Sym) == 0 and section.sh_link < sections.size(),
                               "invalid AICPU ELF symbol table");
                const auto& strings = sections[section.sh_link];
                DJ_HOST_ASSERT(strings.sh_type == SHT_STRTAB, "invalid AICPU ELF string table");
                check_range(section.sh_offset, section.sh_size);
                check_range(strings.sh_offset, strings.sh_size);
                const std::string_view names(data.data() + strings.sh_offset, strings.sh_size);
                for (uint64_t offset = 0; offset < section.sh_size; offset += sizeof(Elf64_Sym)) {
                    Elf64_Sym symbol{};
                    std::memcpy(&symbol, data.data() + section.sh_offset + offset, sizeof(symbol));
                    if (ELF64_ST_TYPE(symbol.st_info) != STT_FUNC or ELF64_ST_BIND(symbol.st_info) != STB_GLOBAL or
                        ELF64_ST_VISIBILITY(symbol.st_other) != STV_DEFAULT or symbol.st_shndx == SHN_UNDEF)
                        continue;
                    DJ_HOST_ASSERT(symbol.st_name < names.size(), "invalid AICPU ELF symbol name");
                    const auto end = names.find('\0', symbol.st_name);
                    DJ_HOST_ASSERT(end != std::string_view::npos, "unterminated AICPU ELF symbol name");
                    symbols.emplace(names.substr(symbol.st_name, end - symbol.st_name));
                }
            }
            std::vector<std::string> kernels;
            constexpr std::string_view suffix = "_interface";
            for (const auto& symbol: symbols) {
                if (symbol.ends_with(suffix)) {
                    const auto name = symbol.substr(0, symbol.size() - suffix.size());
                    if (symbols.contains(name))
                        kernels.push_back(name);
                }
            }
            return kernels;
        }

        // Read section names
        const auto string_table_index = header.e_shstrndx == SHN_XINDEX
            ? sections.front().sh_link
            : header.e_shstrndx;
        DJ_HOST_ASSERT(string_table_index < sections.size(),
                       "invalid ELF string table index");
        const auto& string_table_section = sections[string_table_index];
        DJ_HOST_ASSERT(string_table_section.sh_offset <= static_cast<uint64_t>(file_size) and
                       string_table_section.sh_size <= static_cast<uint64_t>(file_size) - string_table_section.sh_offset,
                       "truncated ELF string table");

        const std::string_view string_table(data.data() + string_table_section.sh_offset,
                                            string_table_section.sh_size);

        // Collect kernel names
        constexpr std::string_view prefix = ".ascend.meta.";
        std::vector<std::string> kernel_names;
        for (const auto& section: sections) {
            if (section.sh_name >= string_table.size())
                continue;
            const auto end = string_table.find('\0', section.sh_name);
            if (end == std::string::npos)
                continue;
            std::string_view name(string_table.data() + section.sh_name, end - section.sh_name);
            if (not name.starts_with(prefix))
                continue;

            name.remove_prefix(prefix.size());
            for (const std::string_view suffix: {"_mix_aic", "_mix_aiv"}) {
                if (name.ends_with(suffix)) {
                    name.remove_suffix(suffix.size());
                    break;
                }
            }
            kernel_names.emplace_back(name);
        }
        std::ranges::sort(kernel_names);
        kernel_names.erase(std::ranges::unique(kernel_names).begin(), kernel_names.end());
        return kernel_names;
    }

    static std::vector<std::string> parse_kernel_names(const std::filesystem::path& path) {
        return parse_kernel_names(read_bytes(path));
    }

    static std::shared_ptr<Kernel> load(const std::filesystem::path& dir, const Env& env,
                                        std::shared_ptr<AicpuBootstrap> bootstrap = nullptr) {
        // Release GIL to let other Python threads run
        [[maybe_unused]] GilScopedRelease gil_release;

        // Record start time
        const bool debug = env.get<bool>("JIT_DEBUG", false);
        const bool print_load_time = debug or env.get<bool>("JIT_PRINT_LOAD_TIME", false);
        const auto start_time = print_load_time ? std::chrono::steady_clock::now()
                                                : std::chrono::steady_clock::time_point{};
        auto [is_aicpu, binary_path, data] = run_in_iothread([dir] {
            const bool aicpu = std::filesystem::is_regular_file(dir / "kernel.aicpu.so");
            const auto path = dir / (aicpu ? "kernel.aicpu.so" : "kernel.o");
            return std::tuple{aicpu, path, read_bytes(path)};
        });
        if (debug)
            std::fputs(std::format("Loading Ascend binary: {}\n", binary_path.string()).c_str(), stdout);

        // Load kernel
        const auto kernel_names = parse_kernel_names(data);
        DJ_HOST_ASSERT(kernel_names.size() == 1,
                       "expected exactly one kernel in {}, found {}",
                       binary_path.string(), kernel_names.size());

        auto kernel = std::make_shared<Kernel>();
        kernel->aicpu = is_aicpu;
        if (is_aicpu) {
            DJ_HOST_ASSERT(bootstrap != nullptr, "AICPU kernels require an Ascend backend bootstrap");
            kernel->aicpu_module = bootstrap->load(binary_path, kernel_names.front() + "_interface");
            kernel->kernel_handle = bootstrap->get_loader_handle();
            kernel->aicpu_bootstrap = std::move(bootstrap);
        } else {
            DJ_ACL_CHECK(driver::lazy_aclrtBinaryLoadFromData(
                data.data(), data.size(), nullptr, &kernel->binary_handle));
            DJ_ACL_CHECK(driver::lazy_aclrtBinaryGetFunction(
                kernel->binary_handle, kernel_names.front().c_str(), &kernel->kernel_handle));
        }

        // Print and return
        if (print_load_time) {
            const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start_time;
            std::fputs(std::format("Load time ({}): {:.2f} ms\n", dir.string(), elapsed.count()).c_str(), stdout);
        }
        return kernel;
    }

    template <typename... Args>
    void launch(const LaunchOptions& launch_options, const Args&... args) const {
        // Release GIL to let other Python threads run
        [[maybe_unused]] GilScopedRelease gil_release;

        // Checks
        DJ_HOST_ASSERT(kernel_handle != nullptr, "kernel must be loaded before launch");
        if (not aicpu) {
            DJ_HOST_ASSERT(launch_options.num_blocks.has_value() and *launch_options.num_blocks > 0,
                           "Ascend block count must be positive");
            DJ_HOST_ASSERT(launch_options.num_ubuf_bytes.has_value() and *launch_options.num_ubuf_bytes >= 0,
                           "Ascend dynamic UB size must not be negative");
        }
        DJ_HOST_ASSERT(launch_options.num_launch_timeout_secs.has_value() and
                       *launch_options.num_launch_timeout_secs >= 0 and
                       *launch_options.num_launch_timeout_secs <= std::numeric_limits<uint16_t>::max(),
                       "Ascend launch timeout is out of range");

        // Pack arguments
        constexpr size_t parameter_size = [] {
            size_t offset = 0;
            ((offset = align_kernel_arg<Args>(offset) + sizeof(Args)), ...);
            // ACL requires a nonempty host payload even for zero-argument kernels.
            return std::max<size_t>(8, (offset + 7) & ~static_cast<size_t>(7));
        }();
        std::array<uint8_t, parameter_size> parameter_buffer{};
        size_t offset = 0;
        ((offset = align_kernel_arg<Args>(offset),
          std::memcpy(parameter_buffer.data() + offset, &args, sizeof(args)),
          offset += sizeof(args)), ...);

        // Set launch attributes
        std::array<aclrtLaunchKernelAttr, 2> attributes{};
        size_t num_attributes = 0;
        if (not aicpu and *launch_options.num_ubuf_bytes > 0) {
            auto& attribute = attributes[num_attributes ++];
            attribute.id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
            attribute.value.dynUBufSize = static_cast<uint32_t>(*launch_options.num_ubuf_bytes);
        }
        if (*launch_options.num_launch_timeout_secs > 0) {
            auto& attribute = attributes[num_attributes ++];
            attribute.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
            attribute.value.timeout = static_cast<uint16_t>(*launch_options.num_launch_timeout_secs);
        }

        // Launch
        aclrtLaunchKernelCfg config{};
        aclrtLaunchKernelCfg* config_ptr = nullptr;
        if (num_attributes > 0) {
            config.attrs = attributes.data();
            config.numAttrs = num_attributes;
            config_ptr = &config;
        }

        const aclrtStream stream = launch_options.stream
            ? *launch_options.stream
            : c10_npu::getCurrentNPUStream();
        if (aicpu) {
            const aicpu::RequestHeader header{.operation = aicpu::Operation::Invoke, .module = aicpu_module};
            alignas(aicpu::RequestHeader) std::array<uint8_t, sizeof(header) + parameter_size> packet{};
            std::memcpy(packet.data(), &header, sizeof(header));
            std::memcpy(packet.data() + sizeof(header), parameter_buffer.data(), parameter_size);
            DJ_ACL_CHECK(driver::lazy_aclrtLaunchKernelWithHostArgs(
                kernel_handle, 1, stream, config_ptr, packet.data(), packet.size(), nullptr, 0));
        } else {
            DJ_ACL_CHECK(driver::lazy_aclrtLaunchKernelWithHostArgs(
                kernel_handle, *launch_options.num_blocks, stream, config_ptr,
                parameter_buffer.data(), parameter_size, nullptr, 0));
        }

        // Launch blocking
        if (get_env<bool>("ASCEND_LAUNCH_BLOCKING", false))
            DJ_ACL_CHECK(driver::lazy_aclrtSynchronizeDevice());
    }

    void unload() noexcept {
        try {
            // Release the GIL while synchronizing the device and unloading the kernel.
            [[maybe_unused]] GilScopedRelease gil_release;
            if (aicpu_module != 0)
                aicpu_bootstrap->unload(aicpu_module);
            else if (binary_handle != nullptr)
                DJ_ACL_CHECK(driver::lazy_aclrtBinaryUnLoad(binary_handle));
        } catch (...) {
        }
        binary_handle = nullptr;
        kernel_handle = nullptr;
        aicpu_module = 0;
        aicpu_bootstrap.reset();
    }
};

}  // namespace deep_jit::ascend
