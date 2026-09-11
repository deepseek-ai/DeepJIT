import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import torch


if not __debug__:
    raise RuntimeError('DeepJIT tests require Python assertions to be enabled')


ROOT = Path(__file__).resolve().parent.parent
TEST_ROCM_PROJECT = ROOT / 'tests' / 'test_rocm_proj'
MODULE_NAME = 'deep_jit_rocm_test'


@contextmanager
def environment(values):
    previous = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def expect_error(function, message):
    try:
        function()
    except RuntimeError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f'expected failure containing {message!r}')


def import_extension(module_path):
    mock = Path(module_path).parent / 'mock'
    flags = sys.getdlopenflags()
    if mock.is_dir():
        import ctypes
        library = next((Path(directory) / 'libamdhip64.so'
                        for directory in os.environ.get('LD_LIBRARY_PATH', '').split(':')
                        if Path(directory).is_relative_to(mock) and
                        (Path(directory) / 'libamdhip64.so').is_file()),
                       mock / 'libamdhip64.so')
        mock_library = ctypes.CDLL(str(library), mode=os.RTLD_LOCAL | os.RTLD_DEEPBIND)
        sys.setdlopenflags(flags | os.RTLD_DEEPBIND)
    try:
        spec = importlib.util.spec_from_file_location(MODULE_NAME, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.setdlopenflags(flags)


def write_script(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!{sys.executable}\n' + source, encoding='utf-8')
    path.chmod(0o755)
    return path


def create_toolkit(root):
    hipcc = write_script(root / 'bin' / 'hipcc', r'''
import json
import os
import sys
import time
from pathlib import Path

if '--version' in sys.argv:
    print('HIP version: 10.0.0\nAMD clang version 22.0.0 (DeepJIT test fixture)')
    sys.exit(0)
args = sys.argv[1:]
mode = os.environ.get('DJ_ROCM_TEST_COMPILER_MODE', 'success')
if mode == 'failure':
    print('deliberate HIP compiler failure', file=sys.stderr)
    sys.exit(17)
if mode == 'missing':
    sys.exit(0)
if os.environ.get('DJ_ROCM_TEST_BARRIER'):
    barrier = Path(os.environ['DJ_ROCM_TEST_BARRIER'])
    (barrier / str(os.getpid())).touch()
    deadline = time.monotonic() + 20
    while True:
        release = os.environ.get('DJ_ROCM_TEST_RELEASE')
        ready = Path(release).exists() if release else len(list(barrier.iterdir())) >= int(os.environ['DJ_ROCM_TEST_WORKERS'])
        if ready:
            break
        if time.monotonic() > deadline:
            raise RuntimeError('compiler barrier timed out; Python may still hold the GIL')
        time.sleep(0.01)
output = Path(args[args.index('-o') + 1])
source = next(Path(value) for value in args if value.endswith('kernel.hip'))
if mode == 'empty':
    output.write_bytes(b'')
elif '-emit-llvm' in args:
    output.write_text('; test fixture IR\ntarget triple = "amdgcn-amd-amdhsa"\n')
else:
    output.write_text('MOCK\n' + source.read_text())
Path('argv.json').write_text(json.dumps(args))
print('captured HIP compiler output')
''')
    write_script(root / 'bin' / 'llvm-readobj', r'''
import os
print(os.environ.get('DJ_ROCM_TEST_METADATA', ''' + repr(
        'amdhsa.kernels:\n  - .symbol: example.kd\n    .sgpr_spill_count: 0\n'
        '    .vgpr_spill_count: 0\n    .private_segment_fixed_size: 0\namdhsa.version: [1, 2]\n'
    ) + '''))
''')
    write_script(root / 'bin' / 'llvm-objdump', "print('test fixture AMD ISA: s_endpgm')\n")
    return hipcc


def validate_toolchain_discovery(module, root, hipcc):
    settings = {'ROCM_BUILD_TEST_JIT_HIPCC_COMPILER': hipcc, 'ROCM_HOME': '/nonexistent',
                'ROCM_PATH': '/nonexistent', 'HIP_PATH': '/nonexistent'}
    with environment(settings):
        assert Path(module.find_toolkit()) == hipcc
    with environment(dict(settings, ROCM_BUILD_TEST_JIT_HIPCC_COMPILER='/nonexistent/hipcc')):
        expect_error(module.find_toolkit, 'not executable')
    for variable in ('ROCM_HOME', 'ROCM_PATH', 'HIP_PATH'):
        settings = {'ROCM_BUILD_TEST_JIT_HIPCC_COMPILER': None, 'ROCM_HOME': None, 'ROCM_PATH': None, 'HIP_PATH': None}
        settings[variable] = hipcc.parent.parent
        with environment(settings):
            assert Path(module.find_toolkit()) == hipcc
    with environment({'ROCM_BUILD_TEST_JIT_HIPCC_COMPILER': None, 'ROCM_HOME': None, 'ROCM_PATH': None,
                      'HIP_PATH': None, 'PATH': str(hipcc.parent)}):
        assert Path(module.find_toolkit()) == hipcc
    print('validated explicit, prefixed, relocated, and PATH toolchain discovery', flush=True)


def validate_build_pipeline(module, root, hipcc):
    directory = root / "build with spaces and 'quotes'"
    directory.mkdir()
    settings = {'ROCM_BUILD_TEST_JIT_HIPCC_COMPILER': hipcc,
                'ROCM_BUILD_TEST_JIT_LLVM_READOBJ': hipcc.parent / 'llvm-readobj',
                'ROCM_BUILD_TEST_JIT_LLVM_OBJDUMP': hipcc.parent / 'llvm-objdump'}
    with environment(settings):
        def build(checks=False, dumps=False, hook='', disable_tools=False):
            module.build_fixture(str(directory), module.increment_source(), checks, dumps, hook, disable_tools)

        build(checks=True, dumps=True)
        for name in ('kernel.hip', 'kernel.hsaco', 'compiler.log', 'kernel.metadata', 'kernel.ll', 'kernel.isa', 'meta.json'):
            assert (directory / name).is_file(), name
        metadata = json.loads((directory / 'meta.json').read_text())
        assert metadata['backend'] == 'ROCm'
        assert metadata['compiler_info']['path'] == str(hipcc)
        assert metadata['compiler_options']['arch'] == 'gfx90a:xnack-'
        assert 'captured HIP compiler output' in (directory / 'compiler.log').read_text()
        arguments = json.loads((directory / 'argv.json').read_text())
        assert str(directory / 'kernel.hip') in arguments
        assert '--offload-arch=gfx90a:xnack-' in arguments
        for mode, message in [('failure', 'deliberate HIP compiler failure'), ('missing', 'nonempty artifact'), ('empty', 'nonempty artifact')]:
            (directory / 'kernel.hsaco').unlink(missing_ok=True)
            with environment({'DJ_ROCM_TEST_COMPILER_MODE': mode}):
                expect_error(build, message)
        (directory / 'hook.py').write_text(
            'import sys\nfrom pathlib import Path\np = Path(sys.argv[1])\n'
            'assert p.name == "kernel.hsaco"\np.with_name("hook_ran").write_text(str(p))\n')
        build(hook='hook.py')
        assert (directory / 'hook_ran').read_text() == str(directory / 'kernel.hsaco')
        (directory / 'hook.py').write_text('raise RuntimeError("deliberate hook failure")\n')
        expect_error(lambda: build(hook='hook.py'), 'deliberate hook failure')
        (directory / 'hook.py').write_text('import sys\nfrom pathlib import Path\nPath(sys.argv[1]).unlink()\n')
        expect_error(lambda: build(hook='hook.py'), 'nonempty artifact')
        expect_error(lambda: build(checks=True, disable_tools=True), 'llvm-readobj is required')
        expect_error(lambda: build(dumps=True, disable_tools=True), 'llvm-objdump is required')
        with environment({'DJ_ROCM_TEST_METADATA': 'no AMDGPU metadata'}):
            expect_error(lambda: build(checks=True), 'missing AMDGPU')
    print('validated compiler output, artifacts, quoting, hooks, dumps, and fail-closed diagnostics', flush=True)


def validate_cache(module, root):
    include = root / 'include' / 'test'
    include.mkdir(parents=True, exist_ok=True)
    (include / 'inner.hpp').write_text('#define TEST_VALUE 1\n')
    (include / 'outer.hpp').write_text('#include <test/inner.hpp>\n')
    source = '#include <test/outer.hpp>\n' + module.increment_source()
    module.init_jit(str(root))
    first = Path(module.compile_only('tracked', source))
    assert (first / '.committed').is_file()
    assert first == Path(module.compile_only('tracked', source))
    checked = Path(module.compile_only('tracked', source, checks=True))
    dumped = Path(module.compile_only('tracked', source, dumps=True))
    assert len({first, checked, dumped}) == 3
    assert (checked / 'kernel.metadata').is_file() and (dumped / 'kernel.ll').is_file()
    (include / 'inner.hpp').write_text('#define TEST_VALUE 2\n')
    # Tracked includes are snapshotted by the existing runtime, so use a new instance.
    module.init_jit(str(root))
    assert Path(module.compile_only('tracked', source)) != first
    (root / 'post.py').write_text('# initial hook\n')
    hook_first = Path(module.compile_only('hooked', source, hook='post.py'))
    (root / 'post.py').write_text('# changed hook\n')
    assert Path(module.compile_only('hooked', source, hook='post.py')) != hook_first
    with environment({'DJ_ROCM_TEST_COMPILER_MODE': 'failure'}):
        # Cache hits must not recompile; failures must not publish new entries.
        assert Path(module.compile_only('tracked', source)).is_dir()
        expect_error(lambda: module.compile_only('failed_publication', source + '\n// unique'), 'deliberate HIP compiler failure')
    cache = Path(os.environ['ROCM_TEST_JIT_CACHE_DIR'])
    assert not list((cache / 'cache').glob('failed_publication.*'))
    assert not list((cache / 'tmp').iterdir())
    print('validated cache reuse, include and hook hashing, safety/dump keys, and failed-publication cleanup', flush=True)


def validate_gil(module, root):
    barrier = root / 'gil_barrier'
    barrier.mkdir()
    release = root / 'python_progress'
    finished = threading.Event()
    progress = []

    def worker():
        while not finished.is_set():
            if list(barrier.iterdir()):
                progress.append(True)
                release.touch()
                return
            time.sleep(0.001)

    with environment({'DJ_ROCM_TEST_BARRIER': barrier, 'DJ_ROCM_TEST_RELEASE': release}):
        module.init_jit(str(root))
        module.get_jit()
        thread = threading.Thread(target=worker)
        thread.start()
        try:
            module.compile_only('gil', module.increment_source() + '\n// GIL test')
        finally:
            finished.set()
            thread.join(timeout=5)
    assert progress and not thread.is_alive(), 'compilation did not release the GIL'
    print('validated Python thread progress during external compilation', flush=True)


CHILD_IMPORT = f'''
import runpy
import sys
from pathlib import Path
import torch
module = runpy.run_path({str(Path(__file__).resolve())!r})['import_extension'](sys.argv[1])
'''


def run_child(code, module_path, root, env=None):
    process = subprocess.Popen([sys.executable, '-c', CHILD_IMPORT + code, str(module_path), str(root)],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        output, error = process.communicate(timeout=120)
        assert process.returncode == 0, output + error
        return output
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()


def validate_lazy_import(module_path, root):
    cache = root / 'lazy_cache'
    env = dict(os.environ, ROCR_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCM_TEST_JIT_CACHE_DIR=str(cache))
    run_child('''
try:
    module.get_jit()
except RuntimeError as error:
    assert 'lazy object must be initialized before use' in str(error)
else:
    raise AssertionError('empty lazy runtime was initialized')
module.init_jit(sys.argv[2])
assert not torch.cuda.is_initialized()
try:
    module.get_jit()
except RuntimeError as error:
    assert 'lazy object must be initialized before use' not in str(error)
else:
    raise AssertionError('runtime initialized with no visible device')
''', module_path, root, env)
    assert not cache.exists()
    print('validated import and deferred initialization without a visible device', flush=True)


def validate_fork(module_path, root, host_only):
    code = '''
import os
import traceback
module.init_jit(sys.argv[2])
assert not torch.cuda.is_initialized()
child = os.fork()
if child == 0:
    try:
        HOST_SETUP
        assert module.get_arch().startswith('gfx')
        assert Path(module.compile_only('forked', module.increment_source())).is_dir()
    except Exception:
        traceback.print_exc()
        os._exit(1)
    os._exit(0)
_, status = os.waitpid(child, 0)
assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
assert not torch.cuda.is_initialized()
'''.replace('HOST_SETUP', 'pass' if host_only else 'torch.cuda.set_device(0)')
    run_child(code, module_path, root)
    print('validated fork after lazy setup and process-local initialization', flush=True)


def validate_shared_cache(module_path, root):
    barrier = root / 'shared_barrier'
    barrier.mkdir()
    cache = root / 'shared_cache'
    env = dict(os.environ, ROCM_TEST_JIT_CACHE_DIR=str(cache), DJ_ROCM_TEST_BARRIER=str(barrier), DJ_ROCM_TEST_WORKERS='3')
    code = CHILD_IMPORT + '''
module.init_jit(sys.argv[2])
path = Path(module.compile_only('shared', module.increment_source()))
assert (path / '.committed').is_file()
assert (path / 'kernel.hsaco').stat().st_size > 0
assert (path / 'meta.json').is_file()
print(path)
'''
    processes = []
    outputs = []
    try:
        for _ in range(3):
            processes.append(subprocess.Popen([sys.executable, '-c', code, str(module_path), str(root)], env=env,
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True))
        for process in processes:
            output, error = process.communicate(timeout=120)
            assert process.returncode == 0, output + error
            outputs.append(output.strip().splitlines()[-1])
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
    assert len(set(outputs)) == 1
    assert len(list((cache / 'cache').iterdir())) == 1
    assert not list((cache / 'tmp').iterdir())
    # A fresh process reuses the published entry even with a deliberately broken compiler.
    replay_env = dict(env, DJ_ROCM_TEST_COMPILER_MODE='failure')
    replay_env.pop('DJ_ROCM_TEST_BARRIER')
    assert run_child(code[len(CHILD_IMPORT):], module_path, root, replay_env).strip().endswith(outputs[0])
    print('validated three-process shared-cache publication and fresh-process reuse', flush=True)


def validate_header_self_containment(root, includes, cflags):
    include_root = ROOT / 'include'
    headers = sorted(path.relative_to(include_root) for path in (include_root / 'deep_jit').rglob('*.hpp')
                     if 'backend/cuda/' not in path.as_posix() and 'backend/ascend/' not in path.as_posix())
    source = root / 'header_self_containment.cpp'
    for header in headers:
        source.write_text(f'#include <{header.as_posix()}>\n')
        command = [os.environ.get('CXX', 'c++'), '-std=c++20', '-fsyntax-only', '-Werror',
                   '-Wno-attributes', '-Wno-deprecated-declarations', '-Wno-missing-field-initializers',
                   '-Wno-psabi', *cflags, str(source)]
        for path in includes:
            command.extend(['-isystem', str(path)])
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, f'{header}:\n{result.stdout}{result.stderr}'
    print(f'validated {len(headers)} self-contained ROCm/public headers', flush=True)


def validate_streams_and_graph(module):
    module.get_jit()
    kernel = module.compile_kernel(module.increment_source())
    output = torch.empty(1, device='cuda', dtype=torch.int32)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        output.fill_(-1)
        module.launch_increment(kernel, output.data_ptr(), 40)
    stream.synchronize()
    assert output.item() == 41
    explicit = torch.cuda.Stream()
    module.launch_increment(kernel, output.data_ptr(), 41, explicit.cuda_stream)
    explicit.synchronize()
    assert output.item() == 42
    # Null is an explicit default-stream request, even while another stream is current.
    with torch.cuda.stream(stream):
        module.launch_increment(kernel, output.data_ptr(), 42, 0)
    torch.cuda.default_stream().synchronize()
    assert output.item() == 43
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        module.launch_increment(kernel, output.data_ptr(), 44)
    graph.replay()
    torch.cuda.synchronize()
    assert output.item() == 45
    # Keep kernel ownership until all graph executions are complete.
    del graph
    print('validated current, explicit, and null streams and graph capture of a loaded kernel', flush=True)


# Test doubles are generated into temporary directories and used only with --host-only.
# They deliberately do not emulate GPU execution or establish ROCm ABI compatibility.
MOCK_HIP_HEADER = r'''
#pragma once
#include <cstddef>
#include <cstdint>
struct dim3 { unsigned int x, y, z; constexpr dim3(unsigned int x=1, unsigned int y=1, unsigned int z=1): x(x), y(y), z(z) {} };
struct ihipStream; struct ihipLibrary; struct ihipKernel; struct ihipFunction;
using hipStream_t = ihipStream*; using hipLibrary_t = ihipLibrary*;
using hipKernel_t = ihipKernel*; using hipFunction_t = ihipFunction*;
enum hipError_t { hipSuccess=0, hipErrorInvalidValue=1, hipErrorInvalidImage=2, hipErrorNoDevice=100 };
enum hipJitOption { mockJitOption }; enum hipLibraryOption { mockLibraryOption };
enum hipFunction_attribute { HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                             HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES };
enum hipMemcpyKind { hipMemcpyHostToDevice, hipMemcpyDeviceToHost };
struct hipDeviceProp_t {
    int multiProcessorCount, l2CacheSize, clockRate, warpSize, maxThreadsPerBlock, cooperativeLaunch;
    int maxThreadsDim[3], maxGridSize[3]; std::size_t sharedMemPerBlock; char gcnArchName[256];
};
#define hipGetDeviceProperties hipGetDevicePropertiesR0600
extern "C" {
hipError_t hipGetDevice(int*); hipError_t hipFree(void*); hipError_t hipMalloc(void**, std::size_t);
hipError_t hipGetDeviceProperties(hipDeviceProp_t*, int); hipError_t hipDeviceSynchronize();
hipError_t hipMemcpy(void*, const void*, std::size_t, hipMemcpyKind);
const char* hipGetErrorName(hipError_t); const char* hipGetErrorString(hipError_t);
hipError_t hipLibraryLoadFromFile(hipLibrary_t*, const char*, hipJitOption*, void**, unsigned int, hipLibraryOption*, void**, unsigned int);
hipError_t hipLibraryUnload(hipLibrary_t); hipError_t hipLibraryGetKernelCount(unsigned int*, hipLibrary_t);
hipError_t hipLibraryEnumerateKernels(hipKernel_t*, unsigned int, hipLibrary_t);
hipError_t hipKernelGetFunction(hipFunction_t*, hipKernel_t);
hipError_t hipFuncGetAttribute(int*, hipFunction_attribute, hipFunction_t);
hipError_t hipModuleLaunchKernel(hipFunction_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                               unsigned int, hipStream_t, void**, void**);
hipError_t hipModuleLaunchCooperativeKernel(hipFunction_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                          unsigned int, unsigned int, hipStream_t, void**);
hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int*, hipFunction_t, int, std::size_t);
hipStream_t mock_current_stream();
}
'''

MOCK_HIP_SOURCE = r'''
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
struct ihipLibrary { std::string source; };
static int libraries = 0, launches = 0, device = 0;
static uintptr_t stream_value = 0, argument_value = 0, current_stream_value = 7;
static bool fails(const char* phase) { const auto* value = std::getenv("DJ_ROCM_MOCK_FAILURE"); return value and std::strcmp(value, phase) == 0; }
extern "C" {
hipError_t hipGetDevice(int* value) {
    const auto* visible = std::getenv("ROCR_VISIBLE_DEVICES");
    if (visible and *visible == '\0') return hipErrorNoDevice;
    *value = device; return hipSuccess;
}
hipError_t hipFree(void* p) { std::free(p); return hipSuccess; }
hipError_t hipMalloc(void** p, std::size_t size) { *p = std::malloc(size); return *p ? hipSuccess : hipErrorInvalidValue; }
hipError_t hipDeviceSynchronize() { return hipSuccess; }
hipError_t hipMemcpy(void* to, const void* from, std::size_t size, hipMemcpyKind) { std::memcpy(to, from, size); return hipSuccess; }
const char* hipGetErrorName(hipError_t) { return "mockHIPError"; }
const char* hipGetErrorString(hipError_t) { return "test double; no GPU execution"; }
hipError_t hipGetDeviceProperties(hipDeviceProp_t* p, int index) {
    *p = {}; p->multiProcessorCount=8; p->l2CacheSize=4194304; p->sharedMemPerBlock=65536; p->clockRate=1500000;
    p->warpSize=64; p->maxThreadsPerBlock=1024; p->cooperativeLaunch=1;
    for (int i=0; i<3; ++i) { p->maxThreadsDim[i]=1024; p->maxGridSize[i]=2147483647; }
    std::strcpy(p->gcnArchName, index == 0 ? "gfx90a:xnack-" : "gfx942"); return hipSuccess;
}
hipError_t hipLibraryLoadFromFile(hipLibrary_t* value, const char* file, hipJitOption*, void**, unsigned int, hipLibraryOption*, void**, unsigned int) {
    std::ifstream input(file); if (not input) return hipErrorInvalidValue;
    *value = new ihipLibrary{std::string(std::istreambuf_iterator<char>(input), {})}; ++libraries; return hipSuccess;
}
hipError_t hipLibraryUnload(hipLibrary_t value) { delete value; --libraries; return hipSuccess; }
hipError_t hipLibraryGetKernelCount(unsigned int* count, hipLibrary_t value) {
    if (fails("count") or not value->source.starts_with("MOCK\n")) return hipErrorInvalidImage;
    *count=0; std::size_t pos=0;
    while ((pos=value->source.find("__global__", pos)) != std::string::npos) { ++*count; pos += 10; }
    return hipSuccess;
}
#ifndef DJ_ROCM_MISSING_ENUMERATION
hipError_t hipLibraryEnumerateKernels(hipKernel_t* value, unsigned int, hipLibrary_t library) {
    if (fails("enumerate")) return hipErrorInvalidValue; *value=reinterpret_cast<hipKernel_t>(library); return hipSuccess;
}
#endif
hipError_t hipKernelGetFunction(hipFunction_t* value, hipKernel_t kernel) {
    if (fails("function")) return hipErrorInvalidValue; *value=reinterpret_cast<hipFunction_t>(kernel); return hipSuccess;
}
hipError_t hipFuncGetAttribute(int* value, hipFunction_attribute attribute, hipFunction_t) {
    if (fails("attribute")) return hipErrorInvalidValue;
    *value = attribute == HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK ? 1024 : attribute == HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES ? 0 : 65536;
    return hipSuccess;
}
hipError_t hipModuleLaunchKernel(hipFunction_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                               unsigned int, hipStream_t stream, void** args, void**) {
    ++launches; stream_value=reinterpret_cast<uintptr_t>(stream); argument_value=0;
    if (args) { int value; std::memcpy(&value, args[0], sizeof(value)); argument_value=value; }
    return hipSuccess;
}
hipError_t hipModuleLaunchCooperativeKernel(hipFunction_t f, unsigned int x, unsigned int y, unsigned int z,
                                          unsigned int bx, unsigned int by, unsigned int bz, unsigned int mem, hipStream_t s, void** args) {
    return hipModuleLaunchKernel(f,x,y,z,bx,by,bz,mem,s,args,nullptr);
}
hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int* n, hipFunction_t, int, std::size_t) { *n=2; return hipSuccess; }
hipStream_t mock_current_stream() { return reinterpret_cast<hipStream_t>(current_stream_value); }
void mock_set_stream(uintptr_t value) { current_stream_value=value; }
int mock_library_count() { return libraries; } int mock_launch_count() { return launches; }
uintptr_t mock_last_stream() { return stream_value; } uintptr_t mock_last_value() { return argument_value; }
void mock_set_device(int index) { device=index; }
}
'''


def build_mock_extension(root, includes):
    mock = root / 'mock'
    (mock / 'hip').mkdir(parents=True)
    (mock / 'hip' / 'hip_runtime.h').write_text(MOCK_HIP_HEADER)
    guard = mock / 'c10' / 'core' / 'impl' / 'VirtualGuardImpl.h'
    guard.parent.mkdir(parents=True)
    guard.write_text('''#pragma once
#include <hip/hip_runtime.h>
namespace c10 {
enum class DeviceType { CUDA };
struct Device { Device(DeviceType, int) {} };
struct Stream { hipStream_t handle; };
namespace impl { struct VirtualGuardImpl {
    explicit VirtualGuardImpl(DeviceType) {}
    Stream getStream(Device) const { return {mock_current_stream()}; }
}; }
}
''')
    wrapper = mock / 'ATen' / 'hip' / 'impl' / 'HIPStreamMasqueradingAsCUDA.h'
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text('''#pragma once
#include <c10/core/impl/VirtualGuardImpl.h>
namespace c10::hip {
struct HIPStreamMasqueradingAsCUDA {
    c10::Stream value;
    explicit HIPStreamMasqueradingAsCUDA(c10::Stream stream): value(stream) {}
    hipStream_t stream() const { return value.handle; }
};
}
''')
    source = mock / 'mock.cpp'
    source.write_text(MOCK_HIP_SOURCE)
    compiler = os.environ.get('CXX', 'c++')
    for directory, flags in [(mock, []), (mock / 'missing', ['-DDJ_ROCM_MISSING_ENUMERATION'])]:
        directory.mkdir(exist_ok=True)
        subprocess.run([compiler, '-std=c++20', '-shared', '-fPIC', '-O0', '-I', str(mock), str(source),
                        '-Wl,-soname,libamdhip64.so', *flags, '-o', str(directory / 'libamdhip64.so')], check=True, timeout=120)
    includes = [mock, *includes]
    output = root / (MODULE_NAME + sysconfig.get_config_var('EXT_SUFFIX'))
    command = [compiler, '-std=c++20', '-shared', '-fPIC', '-O0', '-g1', '-Wno-attributes',
               '-D__HIP_PLATFORM_AMD__', '-DDJ_ROCM_MOCK_TESTS', f'-DTORCH_EXTENSION_NAME={MODULE_NAME}',
               str(TEST_ROCM_PROJECT / 'main.cpp'), '-o', str(output), '-L', str(mock), '-lamdhip64', '-ldl',
               f'-Wl,-rpath,{mock}']
    for path in includes:
        command.extend(['-isystem', str(path)])
    subprocess.run(command, check=True, timeout=180)
    return output, includes, mock


def find_rocm_home():
    home = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH') or os.environ.get('HIP_PATH')
    if home:
        return Path(home).resolve()
    hipcc = shutil.which('hipcc')
    if hipcc:
        return Path(hipcc).resolve().parent.parent
    for home in ('/opt/rocm/core-10.0', '/opt/rocm', '/opt/rocm/core'):
        if (Path(home) / 'bin' / 'hipcc').is_file():
            return Path(home)
    raise RuntimeError('ROCm development toolkit was not found; set ROCM_PATH')


def build_rocm_extension(root, includes):
    home = find_rocm_home()
    includes = [*includes, home / 'include']
    torch_lib = Path(torch.__file__).resolve().parent / 'lib'
    libraries = []
    for alternatives in [('torch_hip', 'torch_cuda'), ('c10_hip', 'c10_cuda')]:
        name = next((name for name in alternatives if (torch_lib / f'lib{name}.so').is_file()), None)
        if name is None:
            raise RuntimeError(f'missing ROCm PyTorch library: {alternatives}')
        libraries.append('-l' + name)
    library_dir = next((home / name for name in ('lib', 'lib64') if (home / name / 'libamdhip64.so').exists()), None)
    if library_dir is None:
        raise RuntimeError('ROCm development runtime libamdhip64.so was not found')
    from torch.utils.cpp_extension import load
    build = root / 'extension'
    build.mkdir()
    module = load(name=MODULE_NAME, sources=[str(TEST_ROCM_PROJECT / 'main.cpp')],
                  extra_include_paths=[str(path) for path in includes],
                  extra_cflags=['-std=c++20', '-O0', '-g1', '-D__HIP_PLATFORM_AMD__', '-Wno-attributes'],
                  extra_ldflags=[f'-L{library_dir}', f'-Wl,-rpath,{library_dir}', '-lamdhip64', *libraries],
                  with_cuda=False, build_directory=str(build), verbose=True)
    return Path(module.__file__), includes, home


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host-only', action='store_true', help='run host contract tests using generated HIP/stream test doubles; no GPU validation')
    args = parser.parse_args()
    if not args.host_only and (not torch.version.hip or not torch.cuda.is_available()):
        raise RuntimeError('ROCm integration tests require ROCm-enabled PyTorch and an AMD GPU; --host-only runs mock host tests instead')
    torch_include = Path(torch.__file__).resolve().parent / 'include'
    includes = [ROOT / 'include', torch_include, torch_include / 'torch' / 'csrc' / 'api' / 'include',
                Path(sysconfig.get_paths()['include'])]
    # Only this separate runner sanitizes its own process environment.
    suffixes = ('JIT_DEBUG', 'JIT_DUMP_ASM', 'JIT_DUMP_LLVM_IR', 'JIT_DUMP_ISA', 'JIT_CHECK_NO_SPILLS',
                'JIT_CHECK_NO_LOCAL_MEMORY', 'JIT_HIPCC_VERBOSE', 'JIT_WITH_LINEINFO', 'JIT_CPP_STANDARD',
                'JIT_HIPCC_COMPILER', 'JIT_LLVM_READOBJ', 'JIT_LLVM_OBJDUMP', 'JIT_CACHE_DIR')
    clean = {name: None for name in os.environ if any(name.endswith('_' + suffix) for suffix in suffixes)}
    with environment(clean), tempfile.TemporaryDirectory(prefix='deep_jit_rocm_') as temporary:
        root = Path(temporary)
        hipcc = create_toolkit(root / 'toolkit')
        if args.host_only:
            module_path, includes, mock = build_mock_extension(root, includes)
        else:
            module_path, includes, home = build_rocm_extension(root, includes)
        validate_header_self_containment(root, includes, ['-D__HIP_PLATFORM_AMD__'])
        module = import_extension(module_path)
        module.test_options(str(root))
        module.test_metadata()
        print('validated option overrides, cache discrimination, and per-kernel diagnostic metadata', flush=True)
        validate_toolchain_discovery(module, root, hipcc)
        validate_build_pipeline(module, root, hipcc)
        settings = {'ROCM_TEST_JIT_HIPCC_COMPILER': hipcc, 'ROCM_TEST_JIT_CACHE_DIR': root / 'cache',
                    'ROCM_TEST_JIT_LLVM_READOBJ': hipcc.parent / 'llvm-readobj',
                    'ROCM_TEST_JIT_LLVM_OBJDUMP': hipcc.parent / 'llvm-objdump'}
        with environment(settings):
            module.init_jit(str(root))
            if args.host_only:
                module.run_mock_tests(str(root))
                print('validated mock library cleanup, cardinality, marshalling, and launch dispatch', flush=True)
                missing_env = dict(os.environ, LD_LIBRARY_PATH=str(mock / 'missing') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
                run_child('''
try:
    module.load_artifact(str(Path(sys.argv[2]) / 'mock_load'))
except RuntimeError as error:
    assert 'ROCm 10 HIP library API is required' in str(error)
    assert 'hipLibraryEnumerateKernels' in str(error)
else:
    raise AssertionError('missing HIP library API was accepted')
''', module_path, root, missing_env)
            validate_cache(module, root)
            validate_gil(module, root)
            validate_lazy_import(module_path, root)
            validate_fork(module_path, root, args.host_only)
            validate_shared_cache(module_path, root)
        if not args.host_only:
            with environment({'ROCM_TEST_JIT_CACHE_DIR': root / 'device_cache', 'ROCM_TEST_JIT_HIPCC_COMPILER': home / 'bin' / 'hipcc'}):
                module.init_jit(str(root))
                print(module.run_device_tests(str(root)), flush=True)
                diagnostics = Path(module.compile_only('real_diagnostics', module.increment_source(), checks=True, dumps=True))
                assert 'amdgcn' in (diagnostics / 'kernel.ll').read_text()
                assert (diagnostics / 'kernel.isa').stat().st_size > 0
                validate_streams_and_graph(module)
        print('ROCm host contract tests passed (mock HIP; not GPU validation)' if args.host_only else 'ROCm integration tests passed', flush=True)


if __name__ == '__main__':
    main()
