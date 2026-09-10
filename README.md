# DeepJIT

DeepJIT is a lightweight, header-only C++20 JIT runtime for **NVIDIA CUDA GPUs** and **HUAWEI Ascend (昇腾) NPUs**. It gives C++/Python extension authors a shared interface for compiling kernel source at runtime, caching the resulting binaries, loading them onto the device, and launching them with backend-specific options.

DeepJIT handles the JIT infrastructure so that kernel libraries can focus on their device code. Both backends share runtime configuration, source and include hashing, in-memory and on-disk caches, and lazy initialization. Kernel source and compiler/launch options remain specific to the selected backend.

**Main authors:** [@guyan364](https://github.com/guyan364), [@kurisu6912](https://github.com/kurisu6912), [@LyricZhao](https://github.com/LyricZhao).

## Features

- **CUDA and Ascend backends:** use `deep_jit::Runtime<deep_jit::CUDA>` or `deep_jit::Runtime<deep_jit::Ascend>` with the same compile/load/launch workflow.
- **Kernel caching:** reuse loaded kernels in memory and compiled artifacts on disk. Cache keys account for source, tracked includes, compiler versions, effective compiler options, and an application-provided dependency signature.
- **Distributed filesystems and shared caches:** share one cache directory across users, processes, and nodes to reuse compiled kernels. Both backends support local and distributed filesystems with the required POSIX filesystem semantics; see [Shared cache](#shared-cache) for configuration.
- **Lazy initialization:** defer device and compiler discovery until the runtime is first used.
- **PyTorch integration:** use the current PyTorch CUDA or `torch_npu` stream by default, and expose the configured runtime through pybind11 with `get_jit()`.
- **Compilation controls and diagnostics:** configure runtime defaults and per-kernel overrides, inspect compilation metadata, and dump CUDA PTX/SASS or Ascend assembly. CUDA also supports a Python post-compilation hook.

### In development (WIP)

- **Cache warmup from history:** use historical cache entries to anticipate kernels that future runs may need and warm up their cache in advance, reducing compilation delays during execution. This feature is under development and is not yet available.
- **Python compilation API:** pass kernel source code directly from Python to compile CUDA or Ascend kernels. This feature is under development and is not yet available.

## Supported backends

| Backend | Device toolchain and runtime | Integration requirements |
| --- | --- | --- |
| **CUDA** | NVCC compiles CUDA source to CUBIN; the CUDA Driver API loads and launches kernels. | CUDA headers 12.4+, NVCC 12.9+, and PyTorch with CUDA support. |
| **Ascend** | Bisheng and ld.lld compile and link Ascend kernel source; ACL loads and launches kernels. | CANN with `bin/bisheng`, `bin/ld.lld`, and the Ascend `adv_api` headers; ACL and `torch_npu` headers and runtime. |

The host environment must provide Linux, a C++20 compiler and standard library with `std::format` support, Python, pybind11, and the dependencies for the selected backend. Installing the elfutils development headers (`libdw-dev` on Debian/Ubuntu) additionally enables file and line information in C++ backtraces; without them backtraces keep function names only. DeepJIT is intended to be embedded into your extension as a header-only dependency.

See [Integration](#integration) for setup, [CUDA](#cuda) for GPU usage, and [Ascend](#ascend) for NPU usage.

## Shared cache

CUDA and Ascend use the same disk-cache implementation. It supports local and distributed filesystems that provide atomic directory rename within a filesystem and file/directory `fsync`. Builds use unique temporary directories, synchronize their contents, and publish complete entries through an atomic rename. Concurrent processes can compile the same entry and reuse the published result.

Multiple users, processes, and nodes can point to the same cache directory:

```bash
export DJ_JIT_CACHE_DIR=/shared/deep_jit
```

Configure directory permissions so participating users can read shared artifacts and writers can create and publish entries under the cache root. As with all DeepJIT caches, use a trusted shared directory. Matching compilation inputs and cache tags allow users to reuse each other's compiled kernels.

You can also combine a writable personal cache with a shared lookup cache:

```bash
export DJ_JIT_CACHE_DIR="$HOME/.dj:/shared/deep_jit"
```

DeepJIT searches all roots in order and writes cache misses only to the first root. The shared lookup cache can be read-only. To configure a single consumer library, use its prefix instead, for example `MYLIB_JIT_CACHE_DIR`.

## Repository layout

| Path | Contents |
| --- | --- |
| [`include/deep_jit/runtime/`](include/deep_jit/runtime/) | Shared runtime and configuration. |
| [`include/deep_jit/backend/cuda/`](include/deep_jit/backend/cuda/) | CUDA compiler, device queries, kernel loading, and launch options. |
| [`include/deep_jit/backend/ascend/`](include/deep_jit/backend/ascend/) | Ascend compiler/linker integration, device queries, kernel loading, and launch options. |
| [`include/deep_jit/cache/`](include/deep_jit/cache/) | In-memory and on-disk kernel caches. |
| [`include/deep_jit/python_api.hpp`](include/deep_jit/python_api.hpp) | pybind11 registration for a consumer library's runtime. |
| [`tests/`](tests/) | CUDA and Ascend integration tests, example extensions, and device kernels. |

The root `CMakeLists.txt` is for debugging and IDE indexing. Integrate the headers into your own extension as described below; the projects under [`tests/test_cuda_proj/`](tests/test_cuda_proj/) and [`tests/test_ascend_proj/`](tests/test_ascend_proj/) provide working integration examples.

## Integration

Add `DeepJIT/include` to the include path of the host target, then include:

```cmake
target_include_directories(my_target PRIVATE third-party/deep_jit/include)
```

Include exactly one backend entry header. For CUDA:

```cpp
#include <deep_jit/backend/cuda/backend.hpp>
```

For Ascend:

```cpp
#include <deep_jit/backend/ascend/backend.hpp>
```

The selected header exposes its backend type:

```cpp
using JIT = deep_jit::Runtime<deep_jit::CUDA>;
```

For the Ascend header, use `deep_jit::Runtime<deep_jit::Ascend>` instead.

`create_lazy_jit` delays construction of the runtime until its first use. This also delays device and compiler discovery:

```cpp
inline auto jit = deep_jit::create_lazy_jit<deep_jit::CUDA>(
    deep_jit::Config("/absolute/path/to/my_library", "MYLIB"));
```

If configuration is only known during library initialization, start with an empty lazy object and assign its factory later:

```cpp
#include <filesystem>
#include <string>

#include <cutlass/version.h>
#include <deep_jit/backend/cuda/backend.hpp>

namespace my_library {

inline deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>> jit(nullptr);

inline void init_jit(const std::string& library_root) {
    const auto library_root_path = std::filesystem::absolute(library_root);
    const auto include_dir = library_root_path / "include";

    jit = deep_jit::create_lazy_jit<deep_jit::CUDA>(
        deep_jit::Config(
            library_root_path,
            "MYLIB",
            "cutlass-" + std::to_string(CUTLASS_VERSION),
            {include_dir},
            {"my_library/"}));
}

}  // namespace my_library
```

The lazy object must receive a factory before `jit->...` or Python `get_jit()` is called.

### Config

`deep_jit::Config` has the following constructor:

```cpp
Config(std::filesystem::path python_library_root,
       std::string env_prefix,
       std::string extra_signature = {},
       std::vector<std::filesystem::path> include_dirs = {},
       std::vector<std::string> include_prefixes = {});
```

- `python_library_root` resolves relative `post_hook` paths. It must be non-empty and absolute.
- `env_prefix` selects the library-specific environment-variable prefix. It must be non-empty and cannot be `DJ`, which is reserved for global defaults.
- `extra_signature` represents dependencies that affect generated code but are not tracked by the include parser. Change it when such a dependency changes, for example `"cutlass-" + std::to_string(CUTLASS_VERSION)`.
- `include_dirs` are passed to the compiler and searched by the include parser. Every path must be absolute.
- `include_prefixes` select which angle-bracket includes are recursively tracked. For example, `"my_library/"` tracks `#include <my_library/kernel.cuh>`.

Configuration is snapshotted when `Runtime` is constructed. Do not mutate `config`, `backend`, `parser`, `disk_cache`, or `hash_base` afterward. The supported mutable per-runtime settings are `default_compiler_options` and `default_launch_options`.

Compiler, cache, include, and hook paths, as well as free-form compiler flags, are currently passed through a simple shell command and therefore must not contain whitespace or shell metacharacters.

### Registering `get_jit()`

For a pybind11 extension, DeepJIT can register the runtime type and `get_jit()` directly. The host library does not need to implement its own `get_jit` binding:

```cpp
#include <pybind11/pybind11.h>

#include <deep_jit/python_api.hpp>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    deep_jit::register_python_api(module, my_library::jit);
}
```

Python can then obtain the same process-local runtime object:

```python
jit = my_library._C.get_jit()
```

Calling `get_jit()` initializes the lazy runtime if it has not already been initialized.

## CUDA

The CUDA backend requires CUDA headers 12.4 or newer and NVCC 12.9 or newer. It uses NVCC to generate a CUBIN. Loading through `compile()` requires exactly one CUDA kernel; `compile_without_load()` only builds the artifact and does not perform that check.

### Compile and launch

```cpp
const auto kernel = jit->compile("scale", R"(
extern "C" __global__ void scale(float* output, const float* input, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count)
        output[index] = input[index] * 2.0f;
}
)");

jit->launch(
    kernel,
    {
        .grid_dim = dim3((count + 255) / 256, 1, 1),
        .block_dim = dim3(256, 1, 1),
    },
    output,
    input,
    count);
```

The compile tag must contain only letters, digits, and underscores. `compile()` compiles on a cache miss, loads the CUBIN, and returns a process-cached `std::shared_ptr<deep_jit::cuda::Kernel>`. `compile_without_load()` only returns the artifact directory.

Unset `deep_jit::cuda::CompilerOptions` fields inherit from the runtime defaults:

```cpp
deep_jit::cuda::CompilerOptions options {
    .optimize_level = "3",
    .fast_math = true,
    .check_no_spills = true,
    .arch = jit->device.get_arch(/* use_arch_family = */ false),
    .extra_nvcc_flags = {"-DMY_OPTION=1"},
    .post_hook = "hooks/hook_1.py",
};

const auto kernel = jit->compile("scale", source, options);
```

`nvcc_flags` replaces the default free-form NVCC flag list; structured options such as `optimize_level` are generated separately. `extra_nvcc_flags` appends per-kernel flags such as `-D` definitions. To extend the runtime defaults, append flags directly to `*jit->default_compiler_options.nvcc_flags` during initialization.

CUDA launch options include the stream, dynamic shared-memory size, grid, block, cluster, cooperative-launch, PDL, and non-portable cluster controls. Unset fields inherit from `jit->default_launch_options`, following the same override model as compiler options. Do not clear the initialized runtime defaults back to `std::nullopt`. Grid and block dimensions are required and must be positive; dynamic shared-memory size cannot be negative. An unset effective stream uses the current PyTorch CUDA stream; an explicitly supplied null stream remains the CUDA default stream.

Only one-dimensional clusters are supported (`cluster_dim.y == cluster_dim.z == 1`). Enabling `nonportable_cluster_size_allowed` sets a persistent CUDA function attribute; later launches with the option disabled do not reset that attribute.

Compile and load kernels before CUDA Graph capture. Launching an already loaded kernel is capture-compatible, including when the stream is inherited from the current PyTorch stream.

Library-wide launch defaults can be changed once during initialization:

```cpp
jit->default_launch_options.enable_pdl = true;
```

Device information is available from `jit->device`:

```cpp
const int num_sms = jit->device.get_num_sms();
const int l2_bytes = jit->device.get_num_l2_cache_bytes();
const int smem_bytes = jit->device.get_num_smem_bytes();
const int64_t clock_rate = jit->device.get_clock_rate();
const auto [major, minor] = jit->device.get_arch_pair();
const std::string family_arch = jit->device.get_arch();
const std::string concrete_arch = jit->device.get_arch(false);
```

A runtime snapshots the current CUDA device when its default architecture is initialized. Use a separate runtime per CUDA device, construct and use it while that device is current, and do not move a loaded kernel between device contexts.

### Post hook

`CompilerOptions::post_hook` selects one optional Python file under `Config::python_library_root`. For example:

```text
my_library/hooks/hook_1.py
```

Pass the file's relative path through the compiler options:

```cpp
const auto kernel = jit->compile("scale", source, {
    .post_hook = "hooks/hook_1.py",
});
```

`post_hook` inherits from `jit->default_compiler_options` like every other compiler option and is unset by default. Only `std::nullopt` disables it; an empty string is still treated as a configured hook. Once a default hook is set, the current override model cannot disable it for one kernel. The caller must provide a trusted relative path. When set, it runs after NVCC produces the CUBIN and before the artifact is published:

```bash
cd <temporary-artifact-directory>
python <absolute-python-library-root>/hooks/hook_1.py <absolute-cubin-path>
```

The script receives the absolute CUBIN path as its only argument and must modify that file in place. The configured relative `post_hook` path and file-content hash are included in the kernel cache digest; the path is also recorded in `meta.json`. Hook hashes are cached per thread, so restart the process after changing a hook file.

A hook must be deterministic for a given input CUBIN and tracked signature. Imported Python modules, auxiliary files, Python/package versions, environment variables, random state, time, and network data are not discovered automatically; represent every such dependency in `extra_signature` if it can affect output.

### Cache-key and artifact rules

The CUDA cache digest is built, in order, from:

1. `Config::extra_signature`.
2. The hash of the complete `nvcc --version` output.
3. The effective compiler flags returned by `CompilerOptions::get_flags()`. Paths in `Config::include_dirs` are intentionally excluded, while any `-I...` placed directly in `nvcc_flags` or `extra_nvcc_flags` remains part of the digest.
4. The selected `post_hook` path and file-content hash.
5. The parser digest of the source and its tracked include tree.

Each component is prefixed by its fixed-width byte length before it is added to the two-state FNV-1a hash, so boundaries remain unambiguous even for binary strings containing zero bytes. The final digest is a 32-character hexadecimal string. This is a fast cache checksum, not a cryptographic hash.

The compile tag is not part of the digest. A disk entry is stored as:

```text
<cache-root>/cache/<tag>.<digest>/
```

Cache roots must be trusted. A directory carrying a `.committed` marker is treated as a completed artifact and its CUBIN may be loaded directly.

An entry contains `kernel.cu`, `kernel.cubin`, `meta.json`, `.committed`, and optional `kernel.ptx` or `kernel.sass` files. `meta.json` records the NVCC command arguments used in the temporary build directory, config fields, compiler information, and effective compiler options, including the selected `post_hook` path. Temporary paths in that command may no longer exist after publication. It does not record the surrounding `cd`, stderr redirection, post-hook command, or dump commands.

`dump_ptx` and `dump_sass` control extra artifacts but do not affect the cache digest. They only run when compilation actually occurs; a pre-existing cache hit is not rebuilt to add missing dumps.

The cache key assumes the external compiler environment is stable. Variables such as `NVCC_PREPEND_FLAGS`, `NVCC_APPEND_FLAGS`, `CPATH`, host-compiler selection, and transitive third-party header changes are not discovered automatically. Avoid such implicit inputs or represent them in `extra_signature`/explicit compiler flags. Excluding tracked include-directory paths is safe only when moving the same header contents does not change compilation behavior such as embedded `__FILE__` strings.

### Include parser

The root source is always hashed. An included file is recursively tracked only when both of the following are true:

- The directive uses a literal angle-bracket include: `#include <...>`.
- The included filename starts with one of `Config::include_prefixes`.

The parser is intentionally a line-oriented scanner, not a C preprocessor. A tracked directive must use the canonical single-line form `#include <...>` (whitespace around `#`, `include`, and the filename is allowed). It does not interpret comments between tokens, macro-expanded includes, backslash continuations, or conditional compilation; an include inside `#if 0` is still scanned.

Tracked files are resolved by checking `include_dirs` in order and using the first matching path. A tracked file that cannot be found is an error. The same rules are applied recursively to includes inside tracked files.

Quoted includes such as `#include "kernel.cuh"` are rejected. Macro-based includes are not tracked. Angle-bracket includes that do not match any configured prefix are accepted by the compiler but ignored by the parser.

For example, with:

```cpp
deep_jit::Config(
    std::filesystem::path("/opt/my_library"),
    "MYLIB",
    "cutlass-40000",
    {std::filesystem::path("/opt/my_library/include")},
    {"my_library/"});
```

`#include <my_library/kernel.cuh>` is tracked, while `#include <cutlass/cutlass.h>` is not. The CUTLASS version is instead represented by `extra_signature`.

Tracked include graphs must not contain cycles. Header digests are cached for the lifetime of a runtime, so recreate the runtime after changing tracked header files.

### Environment variables

For `Config("/absolute/path/to/my_library", "MYLIB")`, every DeepJIT setting is resolved in this order:

```text
MYLIB_<SUFFIX> > DJ_<SUFFIX> > built-in default
```

For example:

```text
MYLIB_JIT_CACHE_DIR > DJ_JIT_CACHE_DIR > $HOME/.dj
```

The library prefix therefore allows one consumer to be configured independently, while the reserved `DJ_` prefix provides process-wide defaults. Boolean values accept `true`/`false`, `yes`/`no`, or any integer, case-insensitively.

Only the library-prefixed and `DJ_` forms are read. An unprefixed variable such as `JIT_CACHE_DIR` or `JIT_DEBUG` has no effect.

Set JIT environment variables before the first runtime construction (normally before the first `get_jit()` or `jit->...`). Cache roots, compiler selection, C++ standard, and default compiler options are snapshotted then. Compiler-command printing and load-time diagnostics are read again when compile/load runs; changing variables after initialization can therefore produce a mixed configuration and is unsupported.

| Suffix | Default | Behavior |
| --- | --- | --- |
| `JIT_CACHE_DIR` | `$HOME/.dj` | Cache root, or a colon-separated list. All roots are searched in order; misses are compiled into the first root. Empty values or empty list elements are rejected. |
| `JIT_DEBUG` | `0` | Enables compiler-command and load diagnostics. CUDA also enables PTXAS output, line info, and PTX/SASS dumps. |
| `JIT_NVCC_COMPILER` | `<discovered-toolkit>/bin/nvcc` | Overrides the NVCC executable. |
| `JIT_CPP_STANDARD` | `20` | Selects the C++ standard passed to NVCC as `-std=c++<value>`. |
| `JIT_KERNEL_DEBUG_INFO` | `0` | Adds Bisheng kernel debug information. |
| `JIT_LAUNCH_TIMEOUT` | `10` | Sets the Ascend kernel launch timeout in seconds; `0` disables it. |
| `JIT_PRINT_COMPILER_COMMAND` | `0` | Prints compiler and disassembler commands. |
| `JIT_PTXAS_VERBOSE` | `0` | Adds verbose PTXAS output and prints it after compilation. |
| `JIT_CHECK_NO_SPILLS` | `0` | Adds `--warn-on-spills` and rejects register spills. |
| `JIT_CHECK_NO_LOCAL_MEMORY` | `0` | Adds `--warn-on-local-memory-usage` and rejects any local-memory usage. |
| `JIT_PRINT_LOAD_TIME` | `0` | Prints kernel-binary loading time. |
| `JIT_WITH_LINEINFO` | `0` | Adds CUDA source line information. |
| `JIT_DUMP_ASM` | `0` | Generates CUDA PTX/SASS or Ascend assembly artifacts on a cache miss. |
| `JIT_DUMP_PTX` | `0` | Generates a PTX artifact on a cache miss. |
| `JIT_DUMP_SASS` | `0` | Generates a SASS artifact on a cache miss. |

CUDA toolkit and cache discovery also use these standard environment variables:

| Variable | Behavior |
| --- | --- |
| `HOME` | Required for the default `$HOME/.dj` cache path. |
| `CUDA_HOME` | First CUDA toolkit-root candidate. |
| `CUDA_PATH` | CUDA toolkit-root fallback when `CUDA_HOME` is unset or empty. |
| `PATH` | Used by `which nvcc` when neither CUDA root variable identifies a toolkit. |

If both CUDA root variables are unset or empty and `which nvcc` fails, DeepJIT tries `/usr/local/cuda`. A non-empty but invalid `CUDA_HOME` or `CUDA_PATH` is treated as an error rather than skipped.

`JIT_NVCC_COMPILER` overrides the executable after CUDA-home discovery. A valid CUDA root must still be discoverable through `CUDA_HOME`, `CUDA_PATH`, `PATH`, or `/usr/local/cuda`. SASS dumping additionally requires an executable `cuobjdump` under that discovered toolkit root.

## Ascend

The Ascend backend requires ACL and torch_npu headers plus a CANN toolkit containing `bin/bisheng` and `bin/ld.lld`. It compiles `kernel.asc` to `kernel.rel.o`, links `kernel.o`, parses the unique `.ascend.meta.*` kernel name, and loads it through ACL.

```cpp
inline auto jit = deep_jit::create_lazy_jit<deep_jit::Ascend>(
    deep_jit::Config(
        "/absolute/path/to/my_library",
        "MYLIB",
        {},
        {"/absolute/path/to/my_library/include"},
        {"my_library/"}));

const auto kernel = jit->compile("scale", source);
jit->launch(kernel, {.num_blocks = num_blocks}, output, input, count);
```

An unset stream uses the current torch_npu stream. `num_blocks` is required. `num_ubuf_bytes` controls dynamic UB size, and `num_launch_timeout_secs` defaults to `JIT_LAUNCH_TIMEOUT`.

When `ASCEND_LAUNCH_BLOCKING` is enabled, DeepJIT synchronizes the device after every launch.

`deep_jit::ascend::CompilerOptions` supports the optimization level, `dav-*` architecture, debug information, assembly dumping, and replace/append lists for Bisheng and linker flags. The defaults are `-O2`, `--cce-aicore-only`, VF loop unrolling, and `ld.lld -m aicorelinux -Ttext 0 --no-mmap-output-file`. `<toolkit>/aarch64-linux/asc/include/adv_api` is required and added automatically.

Assembly dumping writes Bisheng saved intermediates under the artifact's `asm/` directory. Like CUDA's PTX/SASS dumps, it does not change the cache digest and only runs on a cache miss.

Device queries are available through `jit->device`, including `get_npu_arch()`, `get_num_sms()`/`get_num_aicore_cores()`, vector and cube core counts, UB size, and L2 size.

Toolkit discovery checks `ASCEND_HOME_PATH`, `ASCEND_TOOLKIT_HOME`, `/usr/local/Ascend/ascend-toolkit/latest`, and `/usr/local/Ascend/cann`, in that order.
