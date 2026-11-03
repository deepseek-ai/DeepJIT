import os
from pathlib import Path
import subprocess
import tempfile


def test_executor():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='deep-jit-executor-') as directory:
        executable = str(Path(directory) / 'test_executor')
        subprocess.run([
            os.environ.get('CXX', 'c++'), '-std=c++20', '-O2', '-pthread',
            '-Wall', '-Wextra', '-Werror', '-I', str(root / 'include'),
            str(root / 'tests/test_executor.cpp'), '-o', executable,
        ], check=True, timeout=60)
        subprocess.run([executable], check=True, timeout=30)
        subprocess.run([executable, 'stuck-worker'], check=True, timeout=5)
        subprocess.run([executable, 'disable-fallback'],
                       env={**os.environ, 'DJ_DISABLE_CACHE_FALLBACK': '1'},
                       check=True, timeout=5)


if __name__ == '__main__':
    test_executor()
