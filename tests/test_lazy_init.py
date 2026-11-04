import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile


def test_lazy_init():
    root = Path(__file__).resolve().parents[1]
    cxx = os.environ.get('CXX', 'c++')
    with tempfile.TemporaryDirectory(prefix='deep-jit-lazy-') as directory:
        executable = str(Path(directory) / 'test_lazy_init')
        subprocess.run([
            cxx, '-std=c++20', '-O2', '-pthread', '-Wall', '-Wextra', '-Werror',
            '-DDJ_DISABLE_GIL=1', '-I', str(root / 'include'),
            str(root / 'tests/test_lazy_init.cpp'), '-ldl', '-o', executable,
        ], check=True, timeout=60)
        subprocess.run([executable], check=True, timeout=10)


def test_lazy_init_gil():
    from torch.utils.cpp_extension import include_paths

    root = Path(__file__).resolve().parents[1]
    cxx = os.environ.get('CXX', 'c++')
    with tempfile.TemporaryDirectory(prefix='deep-jit-lazy-gil-') as directory:
        module = str(Path(directory) / ('test_lazy_init_gil' + sysconfig.get_config_var('EXT_SUFFIX')))
        subprocess.run([
            cxx, '-std=c++20', '-O2', '-pthread', '-shared', '-fPIC', '-fvisibility=hidden',
            '-Wall', '-Wextra', '-Werror', '-I', str(root / 'include'),
            '-I', include_paths()[0], '-I', sysconfig.get_paths()['include'],
            str(root / 'tests/test_lazy_init_gil.cpp'), '-ldl', '-o', module,
        ], check=True, timeout=60)
        program = '''
import threading
import test_lazy_init_gil as module
results = []
threads = [threading.Thread(target=lambda: results.append(module.get())) for _ in range(2)]
for thread in threads: thread.start()
for thread in threads: thread.join()
assert module.calls() == 1 and results == [1, 1], (module.calls(), results)
'''
        subprocess.run([sys.executable, '-c', program],
                       env={**os.environ, 'PYTHONPATH': directory}, check=True, timeout=12)
