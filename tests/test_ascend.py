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
from pathlib import Path

import torch
import torch_npu
from torch.utils.cpp_extension import include_paths as torch_include_paths
from torch.utils.cpp_extension import library_paths as torch_library_paths
from torch.utils.cpp_extension import load


if not __debug__:
    raise RuntimeError('DeepJIT tests require Python assertions to be enabled')


ROOT = Path(__file__).resolve().parent.parent
TEST_ASCEND_PROJECT = ROOT / 'tests' / 'test_ascend_proj'


def get_ascend_home():
    ascend_home = os.environ.get('ASCEND_HOME_PATH') or os.environ.get(
        'ASCEND_TOOLKIT_HOME', '/usr/local/Ascend/ascend-toolkit/latest')
    if not Path(ascend_home).is_dir():
        ascend_home = '/usr/local/Ascend/cann'
    if not Path(ascend_home).is_dir():
        raise RuntimeError('Ascend toolkit was not found')
    return Path(ascend_home).resolve()


def clear_external_jit_environment():
    suffixes = {
        'JIT_CACHE_DIR',
        'JIT_DEBUG',
        'JIT_DUMP_ASM',
        'JIT_KERNEL_DEBUG_INFO',
        'JIT_LAUNCH_TIMEOUT',
        'JIT_PRINT_COMPILER_COMMAND',
        'JIT_PRINT_LOAD_TIME',
    }
    for name in tuple(os.environ):
        if name in suffixes or any(name.endswith('_' + suffix) for suffix in suffixes):
            os.environ.pop(name)
    os.environ.pop('ASCEND_LAUNCH_BLOCKING', None)


def import_extension(module_path):
    spec = importlib.util.spec_from_file_location('deep_jit_ascend_test', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_header_self_containment(temporary_dir, include_paths):
    include_root = ROOT / 'include'
    headers = sorted(
        path.relative_to(include_root)
        for path in (include_root / 'deep_jit').rglob('*.hpp')
        if 'backend/cuda/' not in path.as_posix()
    )
    source_path = temporary_dir / 'header_self_containment.cpp'
    for header in headers:
        source_path.write_text(f'#include <{header.as_posix()}>\n', encoding='utf-8')
        command = [
            os.environ.get('CXX', 'c++'), '-std=c++20', '-fsyntax-only', '-Werror',
            '-Wno-attributes', '-Wno-deprecated-declarations', '-Wno-unused-function',
            '-Wno-missing-field-initializers', str(source_path),
        ]
        for include_path in include_paths:
            command.extend(['-isystem', str(include_path)])
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, f'{header}:\n{result.stdout}{result.stderr}'
    print(f'validated {len(headers)} self-contained Ascend/public headers', flush=True)


def validate_lazy_import(module_path, temporary_dir):
    code = '''
import importlib.util
import os
import sys
from pathlib import Path

import torch

os.environ['ASCEND_HOME_PATH'] = '/path/that/does/not/exist'
os.environ['ASCEND_TOOLKIT_HOME'] = '/path/that/does/not/exist'
spec = importlib.util.spec_from_file_location('deep_jit_ascend_test', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    module.get_jit()
except RuntimeError as exception:
    assert 'lazy object must be initialized before use' in str(exception)
else:
    raise AssertionError('empty lazy JIT unexpectedly initialized')

module.init_jit(sys.argv[2])
assert not Path(sys.argv[3]).exists()
try:
    module.get_jit()
except RuntimeError as exception:
    assert 'lazy object must be initialized before use' not in str(exception)
else:
    raise AssertionError('runtime unexpectedly initialized with an invalid toolkit')
'''
    lazy_cache = temporary_dir / 'lazy_import_cache'
    env = os.environ.copy()
    env['DJ_JIT_CACHE_DIR'] = str(lazy_cache)
    subprocess.run(
        [sys.executable, '-c', code, str(module_path), str(TEST_ASCEND_PROJECT), str(lazy_cache)],
        env=env, check=True, timeout=60,
    )
    assert not lazy_cache.exists()


def validate_compile_releases_gil(module, temporary_dir, ascend_home):
    fake_toolkit = temporary_dir / 'delayed_toolkit'
    (fake_toolkit / 'bin').mkdir(parents=True)
    (fake_toolkit / 'aarch64-linux/asc').mkdir(parents=True)
    shutil.copy2(TEST_ASCEND_PROJECT / 'scripts' / 'delayed_bisheng.py', fake_toolkit / 'bin/bisheng')
    (fake_toolkit / 'bin/bisheng').chmod(0o755)
    (fake_toolkit / 'bin/ld.lld').symlink_to(ascend_home / 'bin/ld.lld')
    (fake_toolkit / 'aarch64-linux/asc/include').symlink_to(
        ascend_home / 'aarch64-linux/asc/include', target_is_directory=True)

    cache_root = temporary_dir / 'gil_cache'
    barrier_dir = temporary_dir / 'bisheng_barrier'
    progress_path = temporary_dir / 'python_progress'
    environment = {
        'ASCEND_HOME_PATH': str(fake_toolkit),
        'ASCEND_GIL_JIT_CACHE_DIR': str(cache_root),
        'DEEP_JIT_TEST_REAL_BISHENG': str(ascend_home / 'bin/bisheng'),
        'DEEP_JIT_TEST_BISHENG_BARRIER_DIR': str(barrier_dir),
        'DEEP_JIT_TEST_PYTHON_PROGRESS_PATH': str(progress_path),
    }
    saved = {name: os.environ.get(name) for name in environment}
    os.environ.update(environment)
    module.prepare_gil_runtime()

    finished = threading.Event()
    progress = [0]

    def worker():
        while not barrier_dir.exists() and not finished.is_set():
            time.sleep(0.001)
        if finished.is_set():
            return
        while not any(barrier_dir.iterdir()) and not finished.is_set():
            time.sleep(0.001)
        if finished.is_set():
            return
        progress_path.touch()
        while not finished.is_set():
            progress[0] += 1

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        artifact = Path(module.compile_for_gil_test())
    finally:
        finished.set()
        thread.join(timeout=5)
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    assert not thread.is_alive()
    assert progress_path.is_file()
    assert progress[0] > 0
    assert (artifact / '.committed').is_file()
    print(f'validated compile-time GIL release ({progress[0]} observer iterations)', flush=True)


def validate_direct_cache_publication(module_path, temporary_dir):
    cache_root = temporary_dir / 'direct_cache'
    coordination = temporary_dir / 'direct_cache_coordination'
    coordination.mkdir()
    start_path = coordination / 'start'
    processes = []
    for index in range(4):
        processes.append(subprocess.Popen(
            [
                sys.executable,
                str(TEST_ASCEND_PROJECT / 'scripts/cache_publication_worker.py'),
                str(module_path), str(cache_root), str(index),
                str(coordination / f'ready_{index}'), str(start_path),
            ],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        ))
    deadline = time.monotonic() + 60
    while len(list(coordination.glob('ready_*'))) != len(processes):
        assert time.monotonic() < deadline
        assert all(process.poll() is None for process in processes)
        time.sleep(0.01)
    start_path.touch()
    winners = []
    for process in processes:
        output, _ = process.communicate(timeout=60)
        assert process.returncode == 0, output
        lines = [line for line in output.splitlines() if line.startswith('WINNER=')]
        assert len(lines) == 1, output
        winners.append(lines[0].removeprefix('WINNER='))
    assert len(set(winners)) == 1
    final_path = cache_root / 'cache/atomic_publication.digest'
    assert (final_path / '.committed').is_file()
    assert (final_path / 'owner_a').read_text() == winners[0]
    assert (final_path / 'owner_b').read_text() == winners[0]
    print('validated direct 4-process atomic publication', flush=True)


def validate_multiprocess_compile(module_path, temporary_dir):
    cache_root = temporary_dir / 'process_cache'
    env = os.environ.copy()
    env['ASCEND_PROCESS_JIT_CACHE_DIR'] = str(cache_root)
    cases = [('shared_process_kernel', 13)] * 4
    processes = [subprocess.Popen(
        [
            sys.executable,
            str(TEST_ASCEND_PROJECT / 'scripts/process_worker.py'),
            str(module_path), 'compile', tag, str(bias),
        ],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    ) for tag, bias in cases]
    artifacts = []
    for process in processes:
        output, _ = process.communicate(timeout=240)
        assert process.returncode == 0, output
        lines = [line for line in output.splitlines() if line.startswith('ARTIFACT=')]
        assert len(lines) == 1, output
        artifacts.append(Path(lines[0].removeprefix('ARTIFACT=')))
    assert len(set(artifacts)) == 1
    assert (artifacts[0] / '.committed').is_file()

    launch = subprocess.run(
        [
            sys.executable,
            str(TEST_ASCEND_PROJECT / 'scripts/process_worker.py'),
            str(module_path), 'launch', 'shared_process_kernel', '13',
        ],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert launch.returncode == 0, launch.stdout + launch.stderr
    assert 'RESULT=14' in launch.stdout
    print('validated 4-process same-key compilation and cache-hit launch', flush=True)


def validate_diagnostic_output(module_path, temporary_dir):
    code = '''
import importlib.util
import sys
import torch_npu

torch_npu.npu.set_device(0)
spec = importlib.util.spec_from_file_location('deep_jit_ascend_test', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.init_jit(sys.argv[2])
print(f'RESULT={module.run_registered_jit(37)}')
'''
    env = os.environ.copy()
    env['ASCEND_PYTHON_API_JIT_CACHE_DIR'] = str(temporary_dir / 'diagnostic_cache')
    env['ASCEND_PYTHON_API_JIT_PRINT_COMPILER_COMMAND'] = '1'
    env['ASCEND_PYTHON_API_JIT_PRINT_LOAD_TIME'] = '1'
    result = subprocess.run(
        [sys.executable, '-c', code, str(module_path), str(TEST_ASCEND_PROJECT)],
        env=env, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Running command:' in result.stdout
    assert 'Load time (' in result.stdout
    assert 'RESULT=38' in result.stdout
    print('validated compiler and load diagnostics', flush=True)


def validate_artifacts(temporary_dir):
    metadata_paths = sorted(temporary_dir.rglob('meta.json'))
    assert metadata_paths
    for metadata_path in metadata_paths:
        artifact = metadata_path.parent
        assert (artifact / '.committed').is_file(), artifact
        assert (artifact / 'kernel.asc').is_file(), artifact
        assert (artifact / 'kernel.o').stat().st_size > 0, artifact
        assert not (artifact / 'kernel.rel.o').exists(), artifact
        metadata = json.loads(metadata_path.read_text())
        assert set(metadata) == {'command', 'config', 'compiler_info', 'compiler_options'}
        assert set(metadata['compiler_info']) == {'path', 'version', 'linker_path', 'linker_version'}
        assert set(metadata['compiler_options']) == {
            'optimize_level', 'arch', 'debug_info', 'dump_asm',
            'bisheng_flags', 'linker_flags', 'extra_bisheng_flags', 'extra_linker_flags',
        }
        assert metadata['compiler_info']['path']
        assert metadata['compiler_info']['version']
        assert metadata['compiler_info']['linker_path']
        assert metadata['compiler_info']['linker_version']
        assert metadata['compiler_options']['arch'] > 0
        assert metadata['command']
    print(f'validated {len(metadata_paths)} Ascend artifacts', flush=True)


def run_worker():
    clear_external_jit_environment()
    ascend_home = get_ascend_home()
    torch_npu_dir = Path(torch_npu.__file__).resolve().parent
    include_paths = [
        ROOT / 'include',
        Path(sysconfig.get_paths()['include']),
        ascend_home / 'include',
        ascend_home / 'aarch64-linux/include',
        torch_npu_dir / 'include',
        torch_npu_dir / 'include/third_party/acl/inc',
        *map(Path, torch_include_paths()),
    ]
    library_paths = [
        ascend_home / 'lib64',
        torch_npu_dir / 'lib',
        *map(Path, torch_library_paths()),
    ]

    temporary_parent = os.environ.get('DEEP_JIT_TEST_TMPDIR')
    with tempfile.TemporaryDirectory(prefix='deep-jit-ascend-', dir=temporary_parent) as temporary_dir, \
            tempfile.TemporaryDirectory(prefix='deep-jit-ascend-build-') as build_dir:
        temporary_dir = Path(temporary_dir)
        build_dir = Path(build_dir)
        cache_root = temporary_dir / 'cache'
        os.environ['DEEP_JIT_ASCEND_TEST_SOURCE_DIR'] = str(TEST_ASCEND_PROJECT)
        os.environ['DEEP_JIT_ASCEND_TEST_CACHE_ROOT'] = str(cache_root)
        os.environ['DJ_JIT_CACHE_DIR'] = str(temporary_dir / 'global_cache')
        os.environ['ASCEND_TEST_JIT_CACHE_DIR'] = str(cache_root)
        os.environ['ASCEND_TEST_JIT_DEBUG'] = '0'
        os.environ['ASCEND_TEST_JIT_DUMP_ASM'] = '0'
        os.environ['ASCEND_TEST_JIT_KERNEL_DEBUG_INFO'] = '0'
        os.environ['ASCEND_TEST_JIT_LAUNCH_TIMEOUT'] = '10'
        os.environ['ASCEND_TEST_JIT_PRINT_COMPILER_COMMAND'] = '0'
        os.environ['ASCEND_TEST_JIT_PRINT_LOAD_TIME'] = '0'
        os.environ['ASCEND_HOME_PATH'] = str(ascend_home)
        os.environ.setdefault('MAX_JOBS', '8')

        torch_npu.npu.set_device(0)
        extra_ldflags = []
        for library_path in library_paths:
            extra_ldflags.append(f'-L{library_path}')
        extra_ldflags.extend([
            '-ldl', '-lascendcl', '-ltorch_npu',
            f'-Wl,-rpath,{ascend_home / "lib64"}',
            f'-Wl,-rpath,{torch_npu_dir / "lib"}',
        ])
        module = load(
            name='deep_jit_ascend_test',
            sources=[str(TEST_ASCEND_PROJECT / 'main.cpp')],
            extra_cflags=[
                '-std=c++20', '-O3', '-fPIC', '-Wall', '-Wextra', '-Werror',
                '-Wno-attributes', '-Wno-deprecated-declarations',
                '-Wno-missing-field-initializers', '-Wno-unused-function',
            ],
            extra_include_paths=[str(path) for path in include_paths],
            extra_ldflags=extra_ldflags,
            build_directory=str(build_dir),
            with_cuda=False,
            verbose=True,
        )

        validate_header_self_containment(temporary_dir, include_paths)
        validate_lazy_import(module.__file__, temporary_dir)
        os.environ['ASCEND_PYTHON_API_JIT_CACHE_DIR'] = str(temporary_dir / 'python_api_cache')
        module.init_jit(str(TEST_ASCEND_PROJECT))
        assert module.get_jit() is not None
        assert module.run_registered_jit(31) == 32
        try:
            module.compile_invalid_registered_jit()
        except RuntimeError as exception:
            assert 'command failed with exit code' in str(exception)
        else:
            raise AssertionError('invalid Ascend source unexpectedly compiled')
        assert module.run_registered_jit(33) == 34
        validate_compile_releases_gil(module, temporary_dir, ascend_home)
        validate_direct_cache_publication(module.__file__, temporary_dir)
        module.run_tests(module)
        validate_multiprocess_compile(module.__file__, temporary_dir)
        validate_diagnostic_output(module.__file__, temporary_dir)
        validate_artifacts(temporary_dir)


def main():
    if len(sys.argv) == 2 and sys.argv[1] == '--worker':
        run_worker()
        return

    assert len(sys.argv) == 1, sys.argv
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), '--worker'],
        start_new_session=True,
    )
    try:
        return_code = process.wait(timeout=1800)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
    if return_code != 0:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise subprocess.CalledProcessError(return_code, process.args)


if __name__ == '__main__':
    main()
