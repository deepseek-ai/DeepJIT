import sys
from pathlib import Path


cubin_path = Path(sys.argv[1])
assert len(sys.argv) == 2
assert cubin_path.is_absolute()
assert Path.cwd() == cubin_path.parent
data = bytearray(cubin_path.read_bytes())
assert data[:4] == b'\x7fELF'
assert data[12] == 0
data[12] = 1
cubin_path.write_bytes(data)
