#pragma once

#include <string_view>

namespace deep_jit::ascend::aicpu {

// Host helper that packages and deploys the cached loader with a private CANN package configuration.
inline constexpr std::string_view helper_source = R"DJPY(
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import uuid


def overlay(source, destination, private):
    destination.mkdir(exist_ok=True)
    if source.is_dir():
        for child in source.iterdir():
            if child.name not in private:
                (destination / child.name).symlink_to(child)


def main():
    cann, state, loader = (Path(value) for value in sys.argv[1:4])
    device, entry = int(sys.argv[4]), sys.argv[5]
    shadow = state / 'shadow'
    name = 'deep_jit_aicpu_' + uuid.uuid4().hex
    library = 'lib' + name + '.so'
    stage = shadow / 'build/aicpu_kernels_device'
    stage.mkdir(parents=True)
    shutil.copyfile(loader, stage / library)
    digest = hashlib.sha256((stage / library).read_bytes()).hexdigest()
    (stage / 'bin_hash.cfg').write_text(f'{library}={digest}\n')

    # Reuse the installed toolkit while keeping package and operator configs private.
    overlay(cann, shadow, {'conf', 'opp', 'build'})
    overlay(cann / 'conf', shadow / 'conf', {'ascend_package_load.ini', 'bootstrap.json'})
    overlay(cann / 'opp', shadow / 'opp', {'vendors'})
    overlay(cann / 'opp/vendors', shadow / 'opp/vendors', {name})
    package_dir = Path('opp/vendors') / name / 'aicpu/kernel'
    (shadow / package_dir).mkdir(parents=True)
    with tarfile.open(shadow / package_dir / (name + '.tar.gz'), 'w:gz') as archive:
        archive.add(stage, arcname=stage.name)
    manifest = (cann / 'conf/ascend_package_load.ini').read_text().rstrip() + '\n'
    manifest += (f'name:{name}.tar.gz\ninstall_path:2\noptional:true\n'
                 f'package_path:{package_dir}\nload_as_per_soc:false\n')
    (shadow / 'conf/ascend_package_load.ini').write_text(manifest)
    (shadow / 'conf/bootstrap.json').write_text(json.dumps({entry: {'opInfo': {
        'functionName': entry, 'kernelSo': library, 'opKernelLib': 'AICPUKernel',
    }}}))

    # A fresh process selects the private package before initializing CANN.
    # The application's environment and existing context are unaffected.
    os.environ.update(ASCEND_HOME_PATH=str(shadow), ASCEND_LATEST_INSTALL_PATH=str(shadow),
                      ASCEND_AICPU_PATH=str(shadow))
    acl = ctypes.CDLL(str(cann / 'lib64/libascendcl.so'))
    # aclrtSetDevice deploys the package; its files remain after this helper exits.
    for function, arguments in [('aclInit', [None]), ('aclrtSetDevice', [device]),
                                ('aclrtResetDevice', [device]), ('aclFinalize', [])]:
        status = getattr(acl, function)(*arguments)
        if status:
            raise RuntimeError(f'{function} failed: {status}')


if __name__ == '__main__':
    main()
)DJPY";

}  // namespace deep_jit::ascend::aicpu
