import importlib.util
import sys

import torch


spec = importlib.util.spec_from_file_location('deep_jit_cuda_test', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
winner = module.publish_disk_cache_entry(*sys.argv[2:])
print(f'WINNER={winner}', flush=True)
