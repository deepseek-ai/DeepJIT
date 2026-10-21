import os
import shutil
import subprocess
import sysconfig
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = os.name == 'nt'
WITH_CUDA = os.environ.get('DEEP_JIT_TEST_CUDA') == '1'


class PlatformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='deep-jit-platform-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.build = cls.directory / ('build with spaces' if WINDOWS else 'build')
        cls.build.mkdir()
        cls.executable = cls.build / ('platform test.exe' if WINDOWS else 'platform_test')
        cls.compiler = os.environ.get('CXX', 'cl' if WINDOWS else 'c++')
        if WINDOWS:
            cls.flags = [
                '/nologo', '/std:c++20', '/EHsc', '/MD', '/O2',
                '/Zc:preprocessor', '/Zc:__cplusplus', '/D_CRT_SECURE_NO_WARNINGS',
                f'/I{ROOT / "include"}',
            ]
        else:
            cls.flags = ['-std=c++20', '-O2', '-g1', '-pthread', '-I', str(ROOT / 'include')]
        if WITH_CUDA:
            import torch

            nvcc = shutil.which('nvcc')
            cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
            if not cuda_home and nvcc:
                cuda_home = str(Path(nvcc).parent.parent)
            if not cuda_home:
                raise RuntimeError('CUDA toolkit not found')
            cls.cuda_home = Path(cuda_home)
            torch_root = Path(torch.__file__).resolve().parent
            includes = [
                torch_root / 'include',
                torch_root / 'include' / 'torch' / 'csrc' / 'api' / 'include',
                Path(cuda_home) / 'include',
                Path(sysconfig.get_paths()['include']),
            ]
            cls.flags.extend(
                ('/D' if WINDOWS else '-D') + value
                for value in ('DJ_TEST_CUDA=1', 'DJ_DISABLE_GIL=1', 'TORCH_TARGET_VERSION=0x020a000000000000', 'USE_CUDA')
            )
            cls.flags.extend(('/I' if WINDOWS else '-I') + str(path) for path in includes)
        command = [cls.compiler, *cls.flags, str(ROOT / 'tests' / 'test_platform.cpp')]
        if WINDOWS:
            command.extend([f'/Fe{cls.executable}', f'/Fo{cls.build}{os.sep}'])
        else:
            command.extend(['-o', str(cls.executable), '-ldl'])
        subprocess.run(command, cwd=cls.build, check=True, timeout=180)

    def run_case(self, name, *, environment=None, success=True):
        directory = self.directory / self._testMethodName
        directory.mkdir(exist_ok=True)
        env = os.environ.copy()
        env.pop('DJ_JIT_CACHE_DIR', None)
        env.pop('PLATFORM_TEST_JIT_CACHE_DIR', None)
        env.pop('DJ_JIT_NVCC_COMPILER', None)
        env.pop('PLATFORM_TEST_JIT_NVCC_COMPILER', None)
        for key, value in (environment or {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        result = subprocess.run(
            [str(self.executable), name, str(directory)], env=env,
            capture_output=True, text=True, timeout=180,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_filesystem(self):
        self.run_case('filesystem')

    def test_cache(self):
        self.run_case('cache')

    @unittest.skipIf(WINDOWS, 'POSIX cache permissions')
    def test_cache_permissions(self):
        root = self.directory / self._testMethodName / 'writable'
        root.mkdir(parents=True)
        root.chmod(0o2770)
        previous_umask = os.umask(0o077)
        try:
            self.run_case('cache')
        finally:
            os.umask(previous_umask)
        published = root / 'cache' / 'test.race'
        self.assertEqual(published.stat().st_mode & 0o7777, 0o2775)
        self.assertEqual((published / 'nested').stat().st_mode & 0o7777, 0o2775)
        self.assertEqual((published / 'nested' / 'result').stat().st_mode & 0o777, 0o664)
        self.assertEqual((published / '.committed').stat().st_mode & 0o777, 0o664)

    def test_concurrent_cache(self):
        directory = self.directory / 'concurrent'
        directory.mkdir()
        workers = []
        try:
            for _ in range(4):
                workers.append(subprocess.Popen(
                    [str(self.executable), 'cache-worker', str(directory)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                ))
            deadline = time.monotonic() + 30
            while len(list(directory.glob('*-*-*-*'))) != len(workers):
                for worker in workers:
                    if worker.poll() is not None:
                        self.fail(f'cache worker exited early: {worker.communicate()}')
                self.assertLess(time.monotonic(), deadline, 'cache workers did not become ready')
                time.sleep(0.01)
            (directory / 'go').touch()
            outputs = []
            for worker in workers:
                stdout, stderr = worker.communicate(timeout=30)
                self.assertEqual(worker.returncode, 0, stdout + stderr)
                outputs.append(stdout.strip())
            self.assertEqual(len(set(outputs)), 1)
            published = Path(outputs[0])
            self.assertTrue((published / '.committed').is_file())
            self.assertTrue((directory / (published / 'result').read_text()).is_file())
            self.assertEqual(list((directory / 'cache' / 'tmp').iterdir()), [])
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.kill()
                worker.communicate(timeout=10)

    def test_command(self):
        self.run_case('command')

    def test_lazy(self):
        self.run_case('lazy')

    def test_threads(self):
        self.run_case('threads')

    def test_cache_paths(self):
        roots = [self.directory / 'writable cache', self.directory / 'lookup cache']
        result = self.run_case('environment', environment={
            'PLATFORM_TEST_JIT_CACHE_DIR': os.pathsep.join(map(str, roots)),
            'DJ_JIT_CACHE_DIR': str(self.directory / 'ignored'),
        })
        self.assertEqual(result.stdout.splitlines(), list(map(str, roots)))

    def test_invalid_cache_paths(self):
        for value in ('', os.pathsep + 'cache', 'cache' + os.pathsep, 'a' + os.pathsep * 2 + 'b'):
            with self.subTest(value=value):
                result = self.run_case('environment', environment={
                    'PLATFORM_TEST_JIT_CACHE_DIR': value,
                }, success=False)
                self.assertIn('disk cache path list contains an empty path', result.stderr)

    def test_default_cache(self):
        home = self.directory / 'home'
        result = self.run_case('environment', environment={'HOME': str(home)})
        self.assertEqual(result.stdout.strip(), str(home / '.dj'))

    @unittest.skipUnless(WINDOWS, 'Windows profile fallback')
    def test_profile_cache(self):
        profile = self.directory / 'profile'
        for home in (None, ''):
            with self.subTest(home=home):
                result = self.run_case('environment', environment={'HOME': home, 'USERPROFILE': str(profile)})
                self.assertEqual(result.stdout.strip(), str(profile / '.dj'))

    def test_missing_home(self):
        result = self.run_case('environment', environment={'HOME': None, 'USERPROFILE': None}, success=False)
        self.assertIn('environment variable must not be empty', result.stderr)

    def test_header_self_containment(self):
        headers = [
            'cache/disk.hpp', 'utils/command.hpp', 'utils/exception.hpp',
            'utils/filesystem.hpp', 'utils/lazy.hpp', 'utils/uuid.hpp', 'utils/windows.hpp',
        ]
        if WITH_CUDA:
            headers.extend(f'backend/cuda/{name}.hpp' for name in ('backend', 'driver', 'kernel', 'options'))
        source = self.build / 'header.cpp'
        for header in headers:
            with self.subTest(header=header):
                source.write_text(f'#include <deep_jit/{header}>\n', encoding='utf-8')
                command = [
                    self.compiler, *self.flags,
                    '/Zs' if WINDOWS else '-fsyntax-only', str(source),
                ]
                result = subprocess.run(command, cwd=self.build, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(WITH_CUDA and WINDOWS, 'Set DEEP_JIT_TEST_CUDA=1 for the Windows CUDA toolchain smoke test')
    def test_cuda_toolchain(self):
        self.run_case('cuda')

    @unittest.skipUnless(WITH_CUDA and WINDOWS, 'Set DEEP_JIT_TEST_CUDA=1 for Windows toolkit discovery')
    def test_cuda_discovery(self):
        first = self.directory / 'first toolkit'
        second = self.directory / 'second toolkit'
        for root in (first, second):
            (root / 'bin').mkdir(parents=True)
            shutil.copyfile(self.executable, root / 'bin' / 'nvcc.exe')
        shutil.copyfile(self.executable, first / 'bin' / 'cuobjdump.exe')
        expected = [(first / 'bin' / name).resolve() for name in ('nvcc.exe', 'cuobjdump.exe')]
        cases = [
            {'CUDA_HOME': str(first), 'CUDA_PATH': str(second)},
            {'CUDA_HOME': None, 'CUDA_PATH': str(first)},
            {
                'CUDA_HOME': None, 'CUDA_PATH': None,
                'PATH': os.pathsep.join([str(first / 'bin'), str(second / 'bin'), os.environ['PATH']]),
            },
        ]
        for index, environment in enumerate(cases):
            with self.subTest(index=index):
                result = self.run_case('cuda-toolkit', environment=environment)
                self.assertEqual([Path(path).resolve() for path in result.stdout.splitlines()], expected)
        result = self.run_case('cuda-toolkit', environment={
            'CUDA_HOME': str(first),
            'PLATFORM_TEST_JIT_NVCC_COMPILER': str(second / 'bin' / 'nvcc.exe'),
        })
        self.assertEqual(
            [Path(path).resolve() for path in result.stdout.splitlines()],
            [(second / 'bin' / 'nvcc.exe').resolve(), expected[1]],
        )
        result = self.run_case('cuda-toolkit', environment={
            'CUDA_HOME': str(self.directory / 'missing toolkit'), 'CUDA_PATH': str(first),
        }, success=False)
        self.assertIn('CUDA toolkit not found', result.stderr)

    @unittest.skipUnless(WITH_CUDA and WINDOWS, 'Set DEEP_JIT_TEST_CUDA=1 for Windows CUDA SASS coverage')
    def test_cuda_sass(self):
        if not (self.cuda_home / 'bin' / 'cuobjdump.exe').is_file():
            self.skipTest('cuobjdump.exe is not installed in this CUDA toolkit')
        self.run_case('cuda-sass')

    @unittest.skipUnless(WITH_CUDA, 'Set DEEP_JIT_TEST_CUDA=1 for CUDA consumer compilation')
    def test_cuda_consumer(self):
        source = self.build / 'consumer.cpp'
        source.write_text('''
#undef DJ_DISABLE_GIL
#include <deep_jit/backend/cuda/backend.hpp>
#include <deep_jit/python_api.hpp>

void register_consumer(pybind11::module_& module, const std::string& root) {
    static auto jit = deep_jit::create_lazy_jit<deep_jit::CUDA>(deep_jit::Config(root, "CONSUMER"));
    deep_jit::register_python_api(module, jit);
}

void compile_and_launch(deep_jit::Runtime<deep_jit::CUDA>& jit, const std::string& code) {
    jit.launch(jit.compile("test", code), {.grid_dim = dim3(1), .block_dim = dim3(1)});
}
''', encoding='utf-8')
        result = subprocess.run(
            [self.compiler, *self.flags, '/Zs' if WINDOWS else '-fsyntax-only', str(source)],
            cwd=self.build, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
