import os
from pathlib import Path
import subprocess
import tempfile


def test_executor():
    root = Path(__file__).resolve().parents[1]
    windows = os.name == 'nt'
    with tempfile.TemporaryDirectory(prefix='deep-jit-executor-') as directory:
        executable = str(Path(directory) / ('test_executor.exe' if windows else 'test_executor'))
        compiler = os.environ.get('CXX', 'cl' if windows else 'c++')
        if windows:
            # Throw-only test tasks make promise completion unreachable under MSVC.
            flags = [
                '/nologo', '/std:c++20', '/EHsc', '/MD', '/O2', '/W4', '/WX', '/wd4702',
                '/Zc:preprocessor', '/Zc:__cplusplus', '/D_CRT_SECURE_NO_WARNINGS',
                f'/I{root / "include"}',
                f'/Fe{executable}',
            ]
        else:
            flags = [
                '-std=c++20', '-O2', '-pthread', '-Wall', '-Wextra', '-Werror',
                '-I', str(root / 'include'), '-o', executable,
            ]
        subprocess.run([compiler, *flags, str(root / 'tests' / 'test_executor.cpp')],
                       cwd=directory, check=True, timeout=60)
        subprocess.run([executable], check=True, timeout=30)
        subprocess.run([executable, 'stuck-worker'], check=True, timeout=5)
        subprocess.run([executable, 'disable-fallback'],
                       env={**os.environ, 'DJ_DISABLE_CACHE_FALLBACK': '1'},
                       check=True, timeout=5)


if __name__ == '__main__':
    test_executor()
