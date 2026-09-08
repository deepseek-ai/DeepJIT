import importlib.util
import sys

import torch


module_path, cache_root, owner, ready_path, start_path = sys.argv[1:]
spec = importlib.util.spec_from_file_location('deep_jit_ascend_test', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(f'WINNER={module.publish_disk_cache_entry(cache_root, owner, ready_path, start_path)}', flush=True)
