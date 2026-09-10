import os
import sys
from pathlib import Path


cubin_path = Path(sys.argv[1])
assert cubin_path.is_file()
outside = Path(os.environ['DEEP_JIT_TEST_CLEANUP_OUTSIDE'])
(cubin_path.parent / 'directory_link').symlink_to(outside, target_is_directory=True)
(cubin_path.parent / 'file_link').symlink_to(outside / 'sentinel')
(cubin_path.parent / 'dangling_link').symlink_to(outside / 'missing')
raise SystemExit(7)
