#include "precompiler_fake_backend.hpp"

#include <cassert>
#include <string>

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::filesystem::path root(argv[1]);
    ::setenv("TEST_JIT_CACHE_DIR", argv[1], 1);
    ::setenv("TEST_JIT_DRY_RUN", "1", 1);
    ::setenv("TEST_JIT_COMPILE_THREADS", "2", 1);
    ::setenv("DJ_DISABLE_CACHE_FALLBACK", "1", 1);

    {
        precompiler_test::Runtime runtime(deep_jit::Config(root, "TEST"));
        assert(runtime.compile("first", "success") == nullptr);
        assert(runtime.compile("failed", "fail") == nullptr);
        for (int i = 0; i < 2; ++i) {
            try {
                runtime.finish_precompilation();
                assert(false && "expected precompilation failure");
            } catch (const std::runtime_error& error) {
                assert(std::string(error.what()) == "injected compiler failure");
            }
        }
        assert(precompiler_test::compile_calls == 2);
        try {
            runtime.compile("late", "later");
            assert(false && "compile after finish should fail");
        } catch (const std::logic_error&) {}
        const auto path = runtime.compile_without_load("sync", "synchronous");
        assert(std::filesystem::exists(path / "kernel"));
    }

    ::setenv("TEST_JIT_DRY_RUN", "0", 1);
    precompiler_test::Runtime normal(deep_jit::Config(root, "TEST"));
    normal.finish_precompilation();
    normal.finish_precompilation();
    assert(normal.compile("normal", "normal mode") != nullptr);
}
