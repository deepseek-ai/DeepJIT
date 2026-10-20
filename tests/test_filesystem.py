"""CPU-only cleanup regressions; run with python tests/test_filesystem.py."""

import tempfile
from pathlib import Path

from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[1]

CPP_SOURCE = r'''
#include <pybind11/pybind11.h>
#include <deep_jit/runtime/runtime.hpp>
#include <stdexcept>

namespace fs = std::filesystem;

// Exercise the real runtime's exception path without a GPU or external compiler.
struct FailingBackend {
    struct Device {};
    struct Kernel {};
    struct CompilerOptions {
        static CompilerOptions default_options(const deep_jit::Env&, Device&) { return {}; }
    };
    struct LaunchOptions {
        static LaunchOptions default_options(const deep_jit::Env&) { return {}; }
    };
    struct CompilerInfo {
        std::string get_hash() const { return "cleanup-test"; }
    } compiler_info;

    explicit FailingBackend(const deep_jit::Env&) {}

    void compile(const std::string&, const fs::path& path, const deep_jit::Env&,
                 const deep_jit::Config& config, const CompilerOptions&) const {
        deep_jit::write_file_sync(path / "partial", "incomplete build");
        fs::create_directory_symlink(config.python_library_root / "outside", path / "link");
        fs::create_symlink(config.python_library_root / "missing", path / "dangling");
        throw std::runtime_error("synthetic compiler failure");
    }
};

void remove_tree(const std::string& path) {
    deep_jit::safe_remove_all(path);
}

void fail_compilation(const std::string& root) {
    deep_jit::Runtime<FailingBackend> runtime(deep_jit::Config(root, "CLEANUP_TEST"));
    runtime.compile("cleanup", "source", "testkey", {});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("remove_tree", &remove_tree);
    module.def("fail_compilation", &fail_compilation);
}
'''


def check_cleanup(module, root: Path, kind: str, nested: bool) -> None:
    outside = root / 'outside'
    outside.mkdir()
    sentinel = outside / 'sentinel'
    sentinel.write_text('keep me')
    target = root / 'cleanup'
    link = target
    if nested:
        target.mkdir()
        (target / 'partial').write_text('partial build')
        link = target / 'link'
    if kind == 'directory':
        link.symlink_to(outside, target_is_directory=True)
    elif kind == 'file':
        link.symlink_to(sentinel)
    else:
        link.symlink_to(root / 'missing')

    module.remove_tree(str(target))
    assert sentinel.is_file(), f'{kind}, nested={nested}: outside sentinel deleted'
    assert sentinel.read_text() == 'keep me'
    assert not target.exists() and not target.is_symlink(), f'{kind}, nested={nested}: cleanup left entries'
    module.remove_tree(str(target))  # Repeated cleanup of an absent path is harmless.


def main() -> None:
    import os

    # Keep build products and disposable test trees in the checkout, not /tmp.
    with tempfile.TemporaryDirectory(prefix='.filesystem-test-', dir=ROOT) as directory:
        root = Path(directory)
        build = root / 'build'
        build.mkdir()
        module = load_inline(
            name='deep_jit_filesystem_test',
            cpp_sources=CPP_SOURCE,
            extra_cflags=['-std=c++20', '-O0'],
            extra_include_paths=[str(ROOT / 'include')],
            build_directory=str(build),
            with_cuda=False,
            verbose=True,
        )
        failures = []
        for kind in ('directory', 'file', 'dangling'):
            for nested in (False, True):
                case = root / f'{kind}-{nested}'
                case.mkdir()
                try:
                    check_cleanup(module, case, kind, nested)
                    print(f'PASS: {case.name}')
                except AssertionError as error:
                    failures.append(str(error))

        ordinary = root / 'ordinary'
        (ordinary / 'nested').mkdir(parents=True)
        (ordinary / 'nested' / 'file').write_text('ordinary')
        module.remove_tree(str(ordinary))
        assert not ordinary.exists()
        print('PASS: ordinary tree')

        runtime_root = root / 'runtime'
        (runtime_root / 'outside').mkdir(parents=True)
        sentinel = runtime_root / 'outside' / 'sentinel'
        sentinel.write_text('keep me')
        env_name = 'CLEANUP_TEST_JIT_CACHE_DIR'
        previous = os.environ.get(env_name)
        os.environ[env_name] = str(runtime_root)
        try:
            try:
                module.fail_compilation(str(runtime_root))
            except RuntimeError as error:
                assert str(error) == 'synthetic compiler failure'
            else:
                raise AssertionError('compilation did not fail')
        finally:
            if previous is None:
                del os.environ[env_name]
            else:
                os.environ[env_name] = previous
        if not sentinel.is_file() or sentinel.read_text() != 'keep me':
            failures.append('compilation failure deleted outside sentinel')
        if list((runtime_root / 'tmp').iterdir()):
            failures.append('compilation failure left a temporary build')
        if not failures:
            print('PASS: compilation failure cleanup')
        assert not failures, '\n'.join(failures)


if __name__ == '__main__':
    main()
