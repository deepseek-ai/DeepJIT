import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile

import pybind11


def test_precompiler_runtime():
    root = Path(__file__).resolve().parents[1]
    compiler = os.environ.get('CXX', 'c++')
    with tempfile.TemporaryDirectory(prefix='deep-jit-precompiler-runtime-') as directory:
        target = Path(directory)
        common = [compiler, '-std=c++20', '-O2', '-pthread', '-Wall', '-Wextra',
                  '-Werror', '-I', str(root / 'include')]
        executable = str(target / 'test_precompiler_runtime')
        subprocess.run([*common, '-DDJ_DISABLE_GIL=1',
                        str(root / 'tests/test_precompiler_runtime.cpp'), '-ldl',
                        '-o', executable], check=True, timeout=90)
        subprocess.run([executable, str(target / 'native-cache')], check=True, timeout=30)

        extension = target / ('_deepjit_precompiler_test' + sysconfig.get_config_var('EXT_SUFFIX'))
        subprocess.run([*common, '-shared', '-fPIC', '-fvisibility=hidden', '-I', pybind11.get_include(),
                        '-I', sysconfig.get_path('include'),
                        str(root / 'tests/test_precompiler_binding.cpp'), '-ldl',
                        '-o', str(extension)], check=True, timeout=90)
        env = {**os.environ, 'PYTHONPATH': str(target), 'TEST_JIT_DRY_RUN': '1',
               'TEST_JIT_COMPILE_THREADS': '2', 'DJ_DISABLE_CACHE_FALLBACK': '1'}
        success = '''
import threading
import time
import _deepjit_precompiler_test as module
module.submit("blocking", "block")
deadline = time.monotonic() + 5
while not module.started():
    assert time.monotonic() < deadline
    time.sleep(0.01)
jit = module.get_jit()
errors = []
def finish_again():
    try:
        jit.finish_precompilation()
    except Exception as error:
        errors.append(str(error))
other = threading.Thread(target=finish_again)
other.start()
thread = threading.Thread(target=lambda: (time.sleep(0.05), module.release()))
thread.start()
jit.finish_precompilation()
thread.join(timeout=5)
other.join(timeout=5)
assert not thread.is_alive()
assert not other.is_alive() and not errors
assert module.calls() == 1
jit.finish_precompilation()
try:
    module.submit("late", "later")
except RuntimeError as error:
    assert "cannot enqueue" in str(error)
else:
    raise AssertionError("compile after finish should fail")
'''
        failure = '''
import _deepjit_precompiler_test as module
module.submit("failure", "fail")
jit = module.get_jit()
for _ in range(2):
    try:
        jit.finish_precompilation()
    except RuntimeError as error:
        assert "injected compiler failure" in str(error)
    else:
        raise AssertionError("finish should report the failure")
'''
        for name, script in [('success', success), ('failure', failure)]:
            subprocess.run([sys.executable, '-c', script],
                           env={**env, 'TEST_JIT_CACHE_DIR': str(target / name)},
                           check=True, timeout=30)


if __name__ == '__main__':
    test_precompiler_runtime()
