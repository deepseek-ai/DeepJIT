#!/usr/bin/env python3

import os
import subprocess
import sys
import time
from pathlib import Path


real_bisheng = os.environ['DEEP_JIT_TEST_REAL_BISHENG']
arguments = sys.argv[1:]
environment = os.environ.copy()
environment['ASCEND_HOME_PATH'] = str(Path(real_bisheng).parent.parent)
environment['ASCEND_TOOLKIT_HOME'] = environment['ASCEND_HOME_PATH']
if '--version' in arguments:
    raise SystemExit(subprocess.call([real_bisheng, *arguments], env=environment))

barrier_dir = Path(os.environ['DEEP_JIT_TEST_BISHENG_BARRIER_DIR'])
barrier_dir.mkdir(parents=True, exist_ok=True)
(barrier_dir / str(os.getpid())).touch()
progress_path = Path(os.environ['DEEP_JIT_TEST_PYTHON_PROGRESS_PATH'])
deadline = time.monotonic() + 60
while not progress_path.exists():
    if time.monotonic() >= deadline:
        raise TimeoutError('Python thread made no progress while Bisheng was blocked')
    time.sleep(0.001)

raise SystemExit(subprocess.call([real_bisheng, *arguments], env=environment))
