import os
from pathlib import Path
import subprocess
import tempfile


def test_precompiler():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='deep-jit-precompiler-') as directory:
        executable = str(Path(directory) / 'test_precompiler')
        subprocess.run([
            os.environ.get('CXX', 'c++'), '-std=c++20', '-O2', '-pthread',
            '-Wall', '-Wextra', '-Werror', '-I', str(root / 'include'),
            str(root / 'tests/test_precompiler.cpp'), '-o', executable,
        ], check=True, timeout=60)
        subprocess.run([executable], check=True, timeout=15)
        result = subprocess.run([executable, 'unobserved-failure'], capture_output=True,
                                text=True, timeout=15)
        assert result.returncode != 0
        assert 'DeepJIT precompilation failed: unobserved error' in result.stderr


if __name__ == '__main__':
    test_precompiler()
