"""CPU-only disk-cache publication regressions; run with python tests/test_disk_cache.py."""

import os
import shlex
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix='deep-jit-disk-cache-') as directory:
        build = Path(directory)
        executable = build / 'disk_cache_test'
        subprocess.run([
            *shlex.split(os.environ.get('CXX', 'c++')),
            '-std=c++20', '-O2', '-Wall', '-Wextra',
            '-I', str(ROOT / 'include'),
            str(ROOT / 'tests/test_disk_cache_proj/main.cpp'),
            '-o', str(executable), '-ldl', '-pthread',
        ], check=True)
        scenarios = ['normal', 'winner', 'incomplete', 'permission', 'marker_permission']
        for scenario in scenarios:
            if scenario in ('permission', 'marker_permission') and os.geteuid() == 0:
                print(f'{scenario}: skipped (requires an unprivileged user)', flush=True)
                continue
            subprocess.run([str(executable), scenario, str(build / scenario)], check=True, timeout=30)


if __name__ == '__main__':
    main()
