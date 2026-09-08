import sys
from pathlib import Path


cubin_path = Path(sys.argv[1])
assert len(sys.argv) == 2
assert cubin_path.is_absolute()
assert Path.cwd() == cubin_path.parent
data = bytearray(cubin_path.read_bytes())
data[:4] = b'FAIL'
cubin_path.write_bytes(data)
(cubin_path.parent / 'partial_hook_output').write_text('partial', encoding='utf-8')
raise SystemExit(7)
