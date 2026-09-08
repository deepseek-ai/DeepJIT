import importlib.util
import sys

import torch_npu


module_path, action, tag, bias = sys.argv[1:]
torch_npu.npu.set_device(0)
spec = importlib.util.spec_from_file_location('deep_jit_ascend_test', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.prepare_process_runtime()
if action == 'compile':
    print(f'ARTIFACT={module.compile_for_process(tag, int(bias))}', flush=True)
elif action == 'launch':
    print(f'RESULT={module.launch_for_process(tag, int(bias))}', flush=True)
else:
    raise ValueError(action)
