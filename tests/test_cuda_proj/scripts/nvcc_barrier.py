#!/usr/bin/env python3

import os
import sys
import time
from pathlib import Path


nvcc = os.environ['DEEP_JIT_TEST_REAL_NVCC']
if '--version' in sys.argv[1:]:
    os.execv(nvcc, [nvcc, *sys.argv[1:]])

barrier_dir = Path(os.environ['DEEP_JIT_TEST_NVCC_BARRIER_DIR'])
marker = barrier_dir / os.environ.get('DEEP_JIT_TEST_WORKER_ID', str(os.getpid()))
with marker.open('x'):
    pass
barrier_size = int(os.environ['DEEP_JIT_TEST_NVCC_BARRIER_SIZE'])
deadline = time.monotonic() + 60
while len(list(barrier_dir.iterdir())) < barrier_size:
    if time.monotonic() >= deadline:
        raise TimeoutError('NVCC workers did not reach the compiler barrier')
    time.sleep(0.01)

if progress_path := os.environ.get('DEEP_JIT_TEST_PYTHON_PROGRESS_PATH'):
    deadline = time.monotonic() + 60
    while not Path(progress_path).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('Python thread made no progress while NVCC was waiting')
        time.sleep(0.01)

time.sleep(float(os.environ.get('DEEP_JIT_TEST_NVCC_DELAY_SECONDS', '0')))
os.execv(nvcc, [nvcc, *sys.argv[1:]])
