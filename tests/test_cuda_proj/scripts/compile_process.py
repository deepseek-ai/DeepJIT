import importlib.util
import sys
import time
from pathlib import Path

import torch


spec = importlib.util.spec_from_file_location('deep_jit_cuda_test', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.prepare_process_runtime()
Path(sys.argv[2]).write_text('ready', encoding='utf-8')
start_path = Path(sys.argv[3])
tag = sys.argv[4]
bias = int(sys.argv[5])
deadline = time.monotonic() + 60
while not start_path.exists():
    if time.monotonic() >= deadline:
        raise TimeoutError('workers did not leave the compile barrier')
    time.sleep(0.01)
print(f'ARTIFACT={module.compile_for_process(tag, bias)}', flush=True)
print(f'RESULT={module.launch_for_process(tag, bias)}', flush=True)
