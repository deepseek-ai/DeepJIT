import json
import os
import signal
import shlex
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME, load


if not __debug__:
    raise RuntimeError('DeepJIT tests require Python assertions to be enabled')


ROOT = Path(__file__).resolve().parent.parent
TEST_CUDA_PROJECT = ROOT / 'tests' / 'test_cuda_proj'


def register_process_group(process):
    group_dir = os.environ.get('DEEP_JIT_TEST_CHILD_PROCESS_GROUP_DIR')
    if group_dir:
        (Path(group_dir) / str(process.pid)).touch()
    return process


def unregister_process_group(process):
    group_dir = os.environ.get('DEEP_JIT_TEST_CHILD_PROCESS_GROUP_DIR')
    if group_dir:
        (Path(group_dir) / str(process.pid)).unlink(missing_ok=True)


def terminate_process_group(process):
    group_dir = os.environ.get('DEEP_JIT_TEST_CHILD_PROCESS_GROUP_DIR')
    marker = Path(group_dir) / str(process.pid) if group_dir else None
    if marker is None or marker.exists():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
    unregister_process_group(process)


def terminate_registered_process_groups(group_dir):
    for marker in Path(group_dir).iterdir():
        try:
            os.killpg(int(marker.name), signal.SIGKILL)
        except (ProcessLookupError, ValueError):
            pass


def clear_external_jit_environment():
    suffixes = {
        'JIT_CACHE_DIR',
        'JIT_CHECK_NO_LOCAL_MEMORY',
        'JIT_CHECK_NO_SPILLS',
        'JIT_CPP_STANDARD',
        'JIT_DEBUG',
        'JIT_DUMP_ASM',
        'JIT_DUMP_PTX',
        'JIT_DUMP_SASS',
        'JIT_NVCC_COMPILER',
        'JIT_PRINT_COMPILER_COMMAND',
        'JIT_PRINT_LOAD_TIME',
        'JIT_PTXAS_VERBOSE',
        'JIT_WITH_LINEINFO',
    }
    for name in tuple(os.environ):
        if name in suffixes or any(name.endswith('_' + suffix) for suffix in suffixes):
            os.environ.pop(name)
    for name in (
        'CPATH',
        'CPLUS_INCLUDE_PATH',
        'C_INCLUDE_PATH',
        'CUDAHOSTCXX',
        'CUDA_LAUNCH_BLOCKING',
        'NVCC_CCBIN',
        'NVCC_PREPEND_FLAGS',
        'NVCC_APPEND_FLAGS',
    ):
        os.environ.pop(name, None)


def validate_lazy_import(module_path, temporary_dir):
    code = '''
import importlib.util
import sys
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location('deep_jit_cuda_test', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert hasattr(module, 'get_jit')
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
    raise AssertionError('runtime unexpectedly initialized without a visible CUDA device')
'''
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['CUDA_HOME'] = '/path/that/does/not/exist'
    env['CUDA_PATH'] = '/path/that/does/not/exist'
    lazy_cache = temporary_dir / 'lazy_import_cache'
    env['DJ_JIT_CACHE_DIR'] = str(lazy_cache)
    subprocess.run(
        [sys.executable, '-c', code, str(module_path), str(TEST_CUDA_PROJECT), str(lazy_cache)],
        env=env,
        check=True,
        timeout=60,
    )
    assert not lazy_cache.exists(), 'importing the extension initialized the lazy runtime'


def validate_fork_after_lazy_init(module_path, temporary_dir):
    code = '''
import importlib.util
import os
import signal
import sys
import time
import traceback

import torch

spec = importlib.util.spec_from_file_location('deep_jit_cuda_test', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.init_jit(sys.argv[2])
assert not torch.cuda.is_initialized()

for device_index in range(min(int(sys.argv[3]), 2)):
    child = os.fork()
    if child == 0:
        try:
            torch.cuda.set_device(device_index)
            assert torch.cuda.current_device() == device_index
            major, minor = torch.cuda.get_device_capability(device_index)
            number = str(major * 10 + minor)
            expected_arch = number if major < 9 else number + ('a' if major == 9 else 'f')
            assert module.get_registered_jit_arch() == expected_arch
            assert module.run_registered_jit(31 + device_index) == 32 + device_index
            assert torch.cuda.current_device() == device_index
        except Exception:
            traceback.print_exc()
            os._exit(1)
        os._exit(0)

    deadline = time.monotonic() + 240
    while True:
        waited, status = os.waitpid(child, os.WNOHANG)
        if waited == child:
            break
        if time.monotonic() >= deadline:
            os.kill(child, signal.SIGKILL)
            os.waitpid(child, 0)
            raise TimeoutError(f'forked CUDA child {child} did not exit')
        time.sleep(0.01)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, status

assert not torch.cuda.is_initialized()
'''
    env = os.environ.copy()
    fork_cache = temporary_dir / 'fork_cache'
    env['PYTHON_API_JIT_CACHE_DIR'] = str(fork_cache)
    subprocess.run(
        [
            sys.executable, '-c', code, str(module_path), str(TEST_CUDA_PROJECT),
            os.environ['DEEP_JIT_TEST_DEVICE_COUNT'],
        ],
        env=env,
        check=True,
        timeout=300,
    )
    if int(os.environ['DEEP_JIT_TEST_DEVICE_COUNT']) > 0:
        assert (fork_cache / 'cache').is_dir(), 'forked children did not compile through the lazy runtime'


def validate_header_self_containment(temporary_dir):
    assert CUDA_HOME is not None
    include_root = ROOT / 'include'
    torch_include = Path(torch.__file__).resolve().parent / 'include'
    include_paths = [
        include_root,
        torch_include,
        torch_include / 'torch' / 'csrc' / 'api' / 'include',
        Path(sysconfig.get_paths()['include']),
        Path(CUDA_HOME) / 'include',
    ]
    headers = sorted(
        path.relative_to(include_root)
        for path in (include_root / 'deep_jit').rglob('*.hpp')
        if 'backend/ascend/' not in path.as_posix()
    )
    source_path = temporary_dir / 'header_self_containment.cpp'
    for header in headers:
        source_path.write_text(f'#include <{header.as_posix()}>\n', encoding='utf-8')
        command = [
            os.environ.get('CXX', 'c++'), '-std=c++20', '-fsyntax-only', '-Werror',
            '-Wno-attributes', '-Wno-deprecated-declarations',
            '-Wno-missing-field-initializers', '-Wno-psabi',
            str(source_path),
        ]
        for include_path in include_paths:
            command.extend(['-isystem', str(include_path)])
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, f'{header}:\n{result.stdout}{result.stderr}'
    print(f'validated {len(headers)} self-contained CUDA/public headers', flush=True)


def validate_compile_releases_gil(module, temporary_dir):
    cache_root = temporary_dir / 'gil_cache'
    compiler_barrier_dir = temporary_dir / 'gil_compiler_barrier'
    python_progress_path = temporary_dir / 'gil_python_progress'
    compiler_barrier_dir.mkdir()
    environment = {
        'GIL_TEST_JIT_CACHE_DIR': str(cache_root),
        'GIL_TEST_JIT_NVCC_COMPILER': str(TEST_CUDA_PROJECT / 'scripts' / 'nvcc_barrier.py'),
        'DEEP_JIT_TEST_REAL_NVCC': str(Path(CUDA_HOME) / 'bin' / 'nvcc'),
        'DEEP_JIT_TEST_NVCC_BARRIER_DIR': str(compiler_barrier_dir),
        'DEEP_JIT_TEST_NVCC_BARRIER_SIZE': '1',
        'DEEP_JIT_TEST_PYTHON_PROGRESS_PATH': str(python_progress_path),
    }
    os.environ.update(environment)
    module.prepare_gil_runtime()

    ready = threading.Event()
    finished = threading.Event()
    progress = [0]

    def worker():
        ready.set()
        while not any(compiler_barrier_dir.iterdir()) and not finished.is_set():
            time.sleep(0.001)
        if finished.is_set():
            return
        progress[0] += 1
        python_progress_path.touch()
        while not finished.is_set():
            progress[0] += 1

    thread = threading.Thread(target=worker)
    thread.start()
    assert ready.wait(timeout=5)
    try:
        artifact = Path(module.compile_for_gil_test())
    finally:
        finished.set()
        thread.join(timeout=5)
        for name in environment:
            os.environ.pop(name, None)

    assert not thread.is_alive(), 'GIL observer thread did not stop'
    assert any(compiler_barrier_dir.iterdir()), 'delayed NVCC wrapper was not executed'
    assert python_progress_path.is_file(), 'Python observer did not run after NVCC started'
    assert progress[0] > 0, 'Python thread made no progress while JIT compilation held the GIL'
    assert (artifact / '.committed').is_file(), artifact
    print(f'validated compile-time GIL release ({progress[0]} observer iterations)', flush=True)


def validate_diagnostic_output(module_path, temporary_dir):
    code = '''
import importlib.util
import sys

import torch

spec = importlib.util.spec_from_file_location('deep_jit_cuda_test', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.init_jit(sys.argv[2])
print(f'RESULT={module.run_registered_jit(37)}')
'''
    command = [sys.executable, '-c', code, str(module_path), str(TEST_CUDA_PROJECT)]

    diagnostic_names = ('JIT_DEBUG', 'JIT_PRINT_COMPILER_COMMAND', 'JIT_PTXAS_VERBOSE', 'JIT_PRINT_LOAD_TIME')

    def run_case(name, values):
        env = os.environ.copy()
        for prefix in ('PYTHON_API_', 'DJ_', ''):
            for diagnostic_name in diagnostic_names:
                env.pop(prefix + diagnostic_name, None)
        env['PYTHON_API_JIT_CACHE_DIR'] = str(temporary_dir / f'diagnostic_cache_{name}')
        env.update(values)
        return subprocess.run(command, env=env, capture_output=True, text=True, check=True, timeout=180)

    first = run_case('library', {
        'PYTHON_API_JIT_DEBUG': '0',
        'PYTHON_API_JIT_PRINT_COMPILER_COMMAND': '1',
        'PYTHON_API_JIT_PTXAS_VERBOSE': '1',
        'PYTHON_API_JIT_PRINT_LOAD_TIME': '1',
    })
    assert 'Running command:' in first.stdout, first.stdout
    assert 'ptxas info' in first.stdout, first.stdout
    assert 'Load time (' in first.stdout, first.stdout
    assert 'RESULT=38' in first.stdout, first.stdout

    second = run_case('library', {
        'PYTHON_API_JIT_DEBUG': '0',
        'PYTHON_API_JIT_PRINT_COMPILER_COMMAND': '1',
        'PYTHON_API_JIT_PTXAS_VERBOSE': '1',
        'PYTHON_API_JIT_PRINT_LOAD_TIME': '1',
    })
    assert 'Running command:' not in second.stdout, second.stdout
    assert 'ptxas info' not in second.stdout, second.stdout
    assert 'Load time (' in second.stdout, second.stdout
    assert 'RESULT=38' in second.stdout, second.stdout

    global_fallback = run_case('global_fallback', {
        'DJ_JIT_PRINT_COMPILER_COMMAND': '1',
        'DJ_JIT_PTXAS_VERBOSE': '1',
        'DJ_JIT_PRINT_LOAD_TIME': '1',
    })
    assert 'Running command:' in global_fallback.stdout, global_fallback.stdout
    assert 'ptxas info' in global_fallback.stdout, global_fallback.stdout
    assert 'Load time (' in global_fallback.stdout, global_fallback.stdout

    library_false = run_case('library_false', {
        'DJ_JIT_PRINT_COMPILER_COMMAND': '1',
        'DJ_JIT_PTXAS_VERBOSE': '1',
        'DJ_JIT_PRINT_LOAD_TIME': '1',
        'PYTHON_API_JIT_PRINT_COMPILER_COMMAND': '0',
        'PYTHON_API_JIT_PTXAS_VERBOSE': '0',
        'PYTHON_API_JIT_PRINT_LOAD_TIME': '0',
    })
    assert 'Running command:' not in library_false.stdout, library_false.stdout
    assert 'ptxas info' not in library_false.stdout, library_false.stdout
    assert 'Load time (' not in library_false.stdout, library_false.stdout

    unprefixed = run_case('unprefixed', {
        'JIT_PRINT_COMPILER_COMMAND': '1',
        'JIT_PTXAS_VERBOSE': '1',
        'JIT_PRINT_LOAD_TIME': '1',
    })
    assert 'Running command:' not in unprefixed.stdout, unprefixed.stdout
    assert 'ptxas info' not in unprefixed.stdout, unprefixed.stdout
    assert 'Load time (' not in unprefixed.stdout, unprefixed.stdout
    print('validated compiler and load diagnostic output', flush=True)


def validate_multiprocess_cache(module_path, temporary_dir):
    scenarios = [
        (f'same_key_{round_index}', [('process_cache', 89)] * 8)
        for round_index in range(3)
    ]
    scenarios.append((
        'mixed_keys',
        [('process_cache_a', 89)] * 2 +
        [('process_cache_b', 89)] * 2 +
        [('process_cache_a', 97)] * 2 +
        [('process_cache_b', 97)] * 2,
    ))

    for scenario_name, cases in scenarios:
        cache_root = temporary_dir / f'multiprocess_cache_{scenario_name}'
        coordination_dir = temporary_dir / f'multiprocess_coordination_{scenario_name}'
        compiler_barrier_dir = coordination_dir / 'compiler_barrier'
        compiler_barrier_dir.mkdir(parents=True)
        env = os.environ.copy()
        env['PROCESS_TEST_JIT_CACHE_DIR'] = str(cache_root)
        env['PROCESS_TEST_JIT_DEBUG'] = '0'
        env['PROCESS_TEST_JIT_PRINT_COMPILER_COMMAND'] = '0'
        env['PROCESS_TEST_JIT_PTXAS_VERBOSE'] = '0'
        env['PROCESS_TEST_JIT_CHECK_NO_SPILLS'] = '0'
        env['PROCESS_TEST_JIT_CHECK_NO_LOCAL_MEMORY'] = '0'
        env['PROCESS_TEST_JIT_WITH_LINEINFO'] = '0'
        env['PROCESS_TEST_JIT_DUMP_ASM'] = '0'
        env['PROCESS_TEST_JIT_DUMP_PTX'] = '0'
        env['PROCESS_TEST_JIT_DUMP_SASS'] = '0'
        env['PROCESS_TEST_JIT_CPP_STANDARD'] = '20'
        env['PROCESS_TEST_JIT_NVCC_COMPILER'] = str(TEST_CUDA_PROJECT / 'scripts' / 'nvcc_barrier.py')
        env['DEEP_JIT_TEST_REAL_NVCC'] = str(Path(CUDA_HOME) / 'bin' / 'nvcc')
        env['DEEP_JIT_TEST_NVCC_BARRIER_DIR'] = str(compiler_barrier_dir)
        env['DEEP_JIT_TEST_NVCC_BARRIER_SIZE'] = str(len(cases))
        start_path = coordination_dir / 'start'
        processes = []

        try:
            for index, (tag, bias) in enumerate(cases):
                worker_env = env.copy()
                worker_env['DEEP_JIT_TEST_WORKER_ID'] = str(index)
                processes.append(register_process_group(subprocess.Popen(
                    [
                        sys.executable,
                        str(TEST_CUDA_PROJECT / 'scripts' / 'compile_process.py'),
                        str(module_path),
                        str(coordination_dir / f'ready_{index}'),
                        str(start_path),
                        tag,
                        str(bias),
                    ],
                    env=worker_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )))

            deadline = time.monotonic() + 60
            while len(list(coordination_dir.glob('ready_*'))) != len(processes):
                for process in processes:
                    if process.poll() is not None:
                        output, _ = process.communicate(timeout=5)
                        raise AssertionError(output)
                assert time.monotonic() < deadline, 'workers did not reach the compile barrier'
                time.sleep(0.01)
            start_path.touch()

            records = []
            for process, (tag, bias) in zip(processes, cases):
                output, _ = process.communicate(timeout=180)
                assert process.returncode == 0, output
                unregister_process_group(process)
                artifact_lines = [line for line in output.splitlines() if line.startswith('ARTIFACT=')]
                result_lines = [line for line in output.splitlines() if line.startswith('RESULT=')]
                assert len(artifact_lines) == 1, output
                assert result_lines == [f'RESULT={bias + 1}'], output
                records.append((tag, bias, Path(artifact_lines[0].removeprefix('ARTIFACT='))))
        finally:
            for process in processes:
                terminate_process_group(process)

        assert len(list(compiler_barrier_dir.iterdir())) == len(cases)
        expected_pairs = set(cases)
        artifacts_by_case = {}
        for tag, bias, artifact in records:
            assert artifact.parent == cache_root / 'cache', artifact
            artifact_tag, digest = artifact.name.rsplit('.', 1)
            assert artifact_tag == tag, artifact
            assert len(digest) == 32 and all(character in '0123456789abcdef' for character in digest), artifact
            assert (artifact / '.committed').is_file(), artifact
            assert (artifact / 'kernel.cu').is_file(), artifact
            assert f'add_template<{bias}>' in (artifact / 'kernel.cu').read_text(), artifact
            assert (artifact / 'kernel.cubin').stat().st_size > 0, artifact
            with (artifact / 'meta.json').open() as file:
                assert json.load(file)['command'], artifact
            artifacts_by_case.setdefault((tag, bias), set()).add(artifact)

        assert set(artifacts_by_case) == expected_pairs, artifacts_by_case
        assert all(len(paths) == 1 for paths in artifacts_by_case.values()), artifacts_by_case
        assert len(list((cache_root / 'cache').iterdir())) == len(expected_pairs)
        if scenario_name == 'mixed_keys':
            digests = {
                case: next(iter(paths)).name.rsplit('.', 1)[1]
                for case, paths in artifacts_by_case.items()
            }
            assert digests[('process_cache_a', 89)] == digests[('process_cache_b', 89)], digests
            assert digests[('process_cache_a', 97)] == digests[('process_cache_b', 97)], digests
            assert digests[('process_cache_a', 89)] != digests[('process_cache_a', 97)], digests
        tmp_dir = cache_root / 'tmp'
        assert not tmp_dir.exists() or not any(tmp_dir.iterdir()), tmp_dir

        markers_before_cache_hits = set(compiler_barrier_dir.iterdir())
        for index, (tag, bias) in enumerate(sorted(expected_pairs)):
            artifact = next(iter(artifacts_by_case[(tag, bias)]))
            cache_hit_env = env.copy()
            cache_hit_env['DEEP_JIT_TEST_WORKER_ID'] = f'cache_hit_{index}'
            cache_hit = subprocess.run(
                [
                    sys.executable,
                    str(TEST_CUDA_PROJECT / 'scripts' / 'compile_process.py'),
                    str(module_path),
                    str(coordination_dir / f'ready_cache_hit_{index}'),
                    str(start_path),
                    tag,
                    str(bias),
                ],
                env=cache_hit_env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert cache_hit.returncode == 0, cache_hit.stdout + cache_hit.stderr
            assert f'ARTIFACT={artifact}' in cache_hit.stdout, cache_hit.stdout
            assert f'RESULT={bias + 1}' in cache_hit.stdout, cache_hit.stdout
        assert set(compiler_barrier_dir.iterdir()) == markers_before_cache_hits, \
            'cache hits unexpectedly invoked NVCC'

    print('validated same-key and mixed-key 8-process atomic cache publication', flush=True)


def validate_direct_disk_cache_publication(module_path, temporary_dir):
    cache_root = temporary_dir / 'direct_disk_cache'
    coordination_dir = temporary_dir / 'direct_disk_cache_coordination'
    coordination_dir.mkdir()
    start_path = coordination_dir / 'start'
    final_path = cache_root / 'cache' / 'atomic_publication.digest'
    processes = []

    try:
        for index in range(8):
            processes.append(register_process_group(subprocess.Popen(
                [
                    sys.executable,
                    str(TEST_CUDA_PROJECT / 'scripts' / 'disk_cache_process.py'),
                    str(module_path),
                    str(cache_root),
                    str(index),
                    str(coordination_dir / f'ready_{index}'),
                    str(start_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )))

        deadline = time.monotonic() + 60
        while len(list(coordination_dir.glob('ready_*'))) != len(processes):
            for process in processes:
                if process.poll() is not None:
                    output, _ = process.communicate(timeout=5)
                    raise AssertionError(output)
            assert time.monotonic() < deadline, 'direct cache writers did not reach the commit barrier'
            time.sleep(0.01)
        assert not final_path.exists(), 'cache entry became visible before commit'
        start_path.touch()

        winners = []
        for process in processes:
            output, _ = process.communicate(timeout=60)
            assert process.returncode == 0, output
            unregister_process_group(process)
            winner_lines = [line for line in output.splitlines() if line.startswith('WINNER=')]
            assert len(winner_lines) == 1, output
            winners.append(winner_lines[0].removeprefix('WINNER='))
    finally:
        for process in processes:
            terminate_process_group(process)

    assert len(set(winners)) == 1, winners
    assert (final_path / '.committed').is_file(), final_path
    assert (final_path / 'owner_a').read_text() == winners[0], final_path
    assert (final_path / 'owner_b').read_text() == winners[0], final_path
    tmp_dir = cache_root / 'tmp'
    assert not tmp_dir.exists() or not any(tmp_dir.iterdir()), tmp_dir
    print('validated direct 8-process atomic directory publication', flush=True)


def validate_crashed_disk_cache_writer(module_path, temporary_dir):
    cache_root = temporary_dir / 'crashed_disk_cache'
    coordination_dir = temporary_dir / 'crashed_disk_cache_coordination'
    coordination_dir.mkdir()
    ready_path = coordination_dir / 'crashed_ready'
    blocked_start_path = coordination_dir / 'blocked_start'
    final_path = cache_root / 'cache' / 'atomic_publication.digest'
    command = [
        sys.executable,
        str(TEST_CUDA_PROJECT / 'scripts' / 'disk_cache_process.py'),
        str(module_path),
        str(cache_root),
        'crashed',
        str(ready_path),
        str(blocked_start_path),
    ]
    crashed = register_process_group(subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ))
    try:
        deadline = time.monotonic() + 60
        while not ready_path.exists():
            if crashed.poll() is not None:
                output, _ = crashed.communicate(timeout=5)
                raise AssertionError(output)
            assert time.monotonic() < deadline, 'crashed cache writer did not reach the commit barrier'
            time.sleep(0.01)
        assert not final_path.exists(), 'uncommitted cache entry became visible'
    finally:
        terminate_process_group(crashed)
    assert not final_path.exists(), 'killed cache writer published an entry'
    stale_entries = list((cache_root / 'tmp').iterdir())
    assert len(stale_entries) == 1, stale_entries

    recovery_ready_path = coordination_dir / 'recovery_ready'
    recovery_start_path = coordination_dir / 'recovery_start'
    recovery_start_path.touch()
    recovery = subprocess.run(
        [
            sys.executable,
            str(TEST_CUDA_PROJECT / 'scripts' / 'disk_cache_process.py'),
            str(module_path),
            str(cache_root),
            'recovered',
            str(recovery_ready_path),
            str(recovery_start_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert recovery.returncode == 0, recovery.stdout + recovery.stderr
    assert 'WINNER=recovered' in recovery.stdout, recovery.stdout
    assert (final_path / '.committed').is_file(), final_path
    assert (final_path / 'owner_a').read_text() == 'recovered', final_path
    assert (final_path / 'owner_b').read_text() == 'recovered', final_path
    assert stale_entries[0].is_dir(), 'recovery removed another process\'s stale temporary entry'
    print('validated recovery after a cache writer is killed before commit', flush=True)


def validate_artifacts(cache_root):
    cache_dir = cache_root / 'cache'
    cache_entries = sorted(cache_dir.iterdir())
    assert all(path.is_dir() for path in cache_entries), cache_entries
    artifacts = cache_entries
    tags = [artifact.name.rsplit('.', 1)[0] for artifact in artifacts]
    assert Counter(tags) == Counter({
        'cluster_shared_memory': 1,
        'compiler_options': 1,
        'cooperative_grid_sync': 1,
        'include_add': 2,
        'large_dynamic_shared_memory': 1,
        'large_kernel_arguments': 1,
        'launch_overhead': 1,
        'loadable_post_hook': 1,
        'mixed_arguments': 1,
        'macro_selected_abi': 2,
        'multidimensional_launch': 1,
        'multiple_kernels': 1,
        'no_kernel': 1,
        'post_hook': 2,
        'runtime_launch_features': 1,
        'template_add': 2,
        'tensor_map_argument': 1,
    }), artifacts

    required_metadata_keys = {'command', 'config', 'compiler_info', 'compiler_options'}
    required_option_keys = {
        'optimize_level',
        'use_fast_math',
        'ptxas_verbose',
        'ptxas_register_usage_level',
        'check_no_spills',
        'check_no_local_memory',
        'with_line_info',
        'dump_ptx',
        'dump_sass',
        'arch',
        'nvcc_flags',
        'extra_nvcc_flags',
        'post_hook',
    }

    metadata_by_artifact = {}
    hook_artifacts = {}
    for artifact in artifacts:
        tag, digest = artifact.name.rsplit('.', 1)
        assert tag.replace('_', '').isalnum(), artifact
        assert len(digest) == 32 and all(character in '0123456789abcdef' for character in digest), artifact
        assert (artifact / '.committed').is_file(), artifact
        assert (artifact / 'kernel.cu').is_file(), artifact
        assert (artifact / 'kernel.cubin').stat().st_size > 0, artifact

        with (artifact / 'meta.json').open() as file:
            metadata = json.load(file)
        metadata_by_artifact[artifact] = metadata
        assert set(metadata) == required_metadata_keys, metadata
        assert set(metadata['config']) == {
            'python_library_root', 'extra_signature', 'include_dirs', 'include_prefixes'
        }, metadata
        assert set(metadata['compiler_info']) == {'path', 'version'}, metadata
        assert set(metadata['compiler_options']) == required_option_keys, metadata
        assert metadata['compiler_info']['path'], metadata
        assert metadata['compiler_info']['version'], metadata
        assert metadata['config']['extra_signature'] == 'test-signature', metadata
        assert isinstance(metadata['compiler_options']['optimize_level'], str), metadata
        assert isinstance(metadata['compiler_options']['use_fast_math'], bool), metadata
        assert isinstance(metadata['compiler_options']['ptxas_verbose'], bool), metadata
        assert metadata['compiler_options']['ptxas_register_usage_level'] == 10, metadata
        assert not isinstance(metadata['compiler_options']['ptxas_register_usage_level'], bool), metadata
        assert isinstance(metadata['compiler_options']['check_no_spills'], bool), metadata
        assert isinstance(metadata['compiler_options']['check_no_local_memory'], bool), metadata
        assert isinstance(metadata['compiler_options']['with_line_info'], bool), metadata
        assert isinstance(metadata['compiler_options']['dump_ptx'], bool), metadata
        assert isinstance(metadata['compiler_options']['dump_sass'], bool), metadata
        assert isinstance(metadata['compiler_options']['arch'], str), metadata
        assert isinstance(metadata['compiler_options']['nvcc_flags'], list), metadata
        assert isinstance(metadata['compiler_options']['extra_nvcc_flags'], list), metadata
        expected_files = {'.committed', 'kernel.cu', 'kernel.cubin', 'meta.json'}
        if metadata['compiler_options']['dump_ptx']:
            expected_files.add('kernel.ptx')
        if metadata['compiler_options']['dump_sass']:
            expected_files.add('kernel.sass')
        assert {path.name for path in artifact.iterdir()} == expected_files, artifact
        assert metadata['command'], metadata
        assert metadata['config']['python_library_root'] == str(TEST_CUDA_PROJECT), metadata
        assert metadata['config']['include_prefixes'] == ['test_cuda/'], metadata
        command = shlex.split(metadata['command'])
        assert command[0] == metadata['compiler_info']['path'], metadata
        assert '--cubin' in command and '--output-file' in command, metadata
        assert sum(Path(argument).name == 'kernel.cu' for argument in command) == 1, metadata
        assert sum(Path(argument).name == 'kernel.cubin' for argument in command) == 1, metadata
        command_include_dirs = [
            command[index + 1]
            for index, argument in enumerate(command[:-1])
            if argument == '--include-path'
        ]
        assert command_include_dirs == metadata['config']['include_dirs'], metadata
        assert metadata['command'] not in (artifact / 'kernel.cu').read_text(), metadata
        if metadata['compiler_options']['post_hook']:
            hook_artifacts[metadata['compiler_options']['post_hook']] = artifact

    compiler_options_artifact = next(
        artifact for artifact in artifacts if artifact.name.startswith('compiler_options.'))
    compiler_options_metadata = metadata_by_artifact[compiler_options_artifact]
    compiler_options = compiler_options_metadata['compiler_options']
    assert compiler_options['optimize_level'] == '0', compiler_options
    assert compiler_options['use_fast_math'] is True, compiler_options
    assert compiler_options['with_line_info'] is True, compiler_options
    assert compiler_options['extra_nvcc_flags'] == ['-DTEST_OPTION=17'], compiler_options
    command = compiler_options_metadata['command'].split()
    for flag in ('-O0', '--compiler-options=-O0', '--use_fast_math', '--generate-line-info', '-DTEST_OPTION=17'):
        assert flag in command, compiler_options_metadata

    dump_artifact = next(artifact for artifact in artifacts if artifact.name.startswith('launch_overhead.'))
    dump_options = metadata_by_artifact[dump_artifact]['compiler_options']
    assert dump_options['dump_ptx'] is True, dump_options

    assert set(hook_artifacts) == {'scripts/post_hook.py', 'scripts/loadable_post_hook.py'}, hook_artifacts
    assert (hook_artifacts['scripts/post_hook.py'] / 'kernel.cubin').read_bytes()[:4] == b'HOOK'
    assert (hook_artifacts['scripts/loadable_post_hook.py'] / 'kernel.cubin').read_bytes()[12] == 1

    tmp_dir = cache_root / 'tmp'
    assert not tmp_dir.exists() or not any(tmp_dir.iterdir()), 'temporary artifacts were not cleaned'
    print(f'validated {len(artifacts)} cache artifacts', flush=True)


def run_worker():
    clear_external_jit_environment()
    temporary_parent = os.environ.get('DEEP_JIT_TEST_TMPDIR')
    with tempfile.TemporaryDirectory(prefix='deep-jit-test-', dir=temporary_parent) as temporary_dir, \
            tempfile.TemporaryDirectory(prefix='deep-jit-build-') as build_dir:
        temporary_dir = Path(temporary_dir)
        build_dir = Path(build_dir)
        cache_root = temporary_dir / 'cache'
        bin_dir = temporary_dir / 'bin'
        bin_dir.mkdir()
        (bin_dir / 'python').symlink_to(sys.executable)

        os.environ['PATH'] = str(bin_dir) + os.pathsep + os.environ.get('PATH', '')
        os.environ['DEEP_JIT_CUDA_TEST_SOURCE_DIR'] = str(TEST_CUDA_PROJECT)
        os.environ['DEEP_JIT_CUDA_TEST_CACHE_ROOT'] = str(cache_root)
        os.environ['DJ_JIT_CACHE_DIR'] = str(temporary_dir / 'global_cache')
        os.environ['TEST_JIT_CACHE_DIR'] = str(cache_root)
        os.environ['TEST_JIT_DEBUG'] = '0'
        os.environ['TEST_JIT_PRINT_COMPILER_COMMAND'] = '0'
        os.environ['TEST_JIT_PTXAS_VERBOSE'] = '0'
        os.environ['TEST_JIT_CHECK_NO_SPILLS'] = '0'
        os.environ['TEST_JIT_CHECK_NO_LOCAL_MEMORY'] = '0'
        os.environ['TEST_JIT_WITH_LINEINFO'] = '0'
        os.environ['TEST_JIT_DUMP_ASM'] = '0'
        os.environ['TEST_JIT_DUMP_PTX'] = '0'
        os.environ['TEST_JIT_DUMP_SASS'] = '0'
        os.environ['TEST_JIT_PRINT_LOAD_TIME'] = '0'
        os.environ['TEST_JIT_CPP_STANDARD'] = '20'

        module = load(
            name='deep_jit_cuda_test',
            sources=[str(TEST_CUDA_PROJECT / 'main.cpp')],
            extra_cflags=[
                '-std=c++20', '-O3', '-fPIC', '-Wall', '-Wextra', '-Werror',
                '-Wno-attributes', '-Wno-missing-field-initializers',
                '-Wno-psabi', '-Wno-deprecated-declarations',
            ],
            extra_include_paths=[str(ROOT / 'include')],
            extra_ldflags=['-ldl'],
            build_directory=str(build_dir),
            with_cuda=True,
            verbose=True,
        )
        validate_header_self_containment(temporary_dir)
        validate_lazy_import(module.__file__, temporary_dir)
        validate_fork_after_lazy_init(module.__file__, temporary_dir)
        os.environ['PYTHON_API_JIT_CACHE_DIR'] = str(temporary_dir / 'python_api_cache')
        module.init_jit(str(TEST_CUDA_PROJECT))
        assert module.get_jit() is not None
        assert module.run_registered_jit(31) == 32
        try:
            module.compile_invalid_registered_jit()
        except RuntimeError as exception:
            assert 'command failed with exit code' in str(exception)
        else:
            raise AssertionError('invalid CUDA source unexpectedly compiled')
        assert module.run_registered_jit(33) == 34
        validate_compile_releases_gil(module, temporary_dir)
        validate_direct_disk_cache_publication(module.__file__, temporary_dir)
        validate_crashed_disk_cache_writer(module.__file__, temporary_dir)
        module.run_tests(module)
        validate_artifacts(cache_root)
        validate_diagnostic_output(module.__file__, temporary_dir)
        validate_multiprocess_cache(module.__file__, temporary_dir)


def main():
    if len(sys.argv) == 2 and sys.argv[1] == '--worker':
        run_worker()
        return

    assert len(sys.argv) == 1, sys.argv
    with tempfile.TemporaryDirectory(prefix='deep-jit-process-groups-') as group_dir:
        env = os.environ.copy()
        env['DEEP_JIT_TEST_DEVICE_COUNT'] = str(torch.cuda.device_count())
        env['DEEP_JIT_TEST_CHILD_PROCESS_GROUP_DIR'] = group_dir
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), '--worker'],
            env=env,
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
            terminate_registered_process_groups(group_dir)
            raise
        if return_code != 0:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            terminate_registered_process_groups(group_dir)
            raise subprocess.CalledProcessError(return_code, process.args)
        assert not any(Path(group_dir).iterdir()), 'child process-group registrations were not cleaned'


if __name__ == '__main__':
    main()
