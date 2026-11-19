#include <deep_jit/runtime/precompiler.hpp>

#include <atomic>
#include <cassert>
#include <future>
#include <stdexcept>
#include <string>
#include <string_view>

void test_finish_success() {
    std::atomic<int> calls = 0;
    {
        deep_jit::Precompiler precompiler(2);
        precompiler.enqueue("a", [&] { ++calls; });
        precompiler.enqueue("b", [&] { ++calls; });
        precompiler.enqueue("a", [&] { ++calls; });
        precompiler.finish();
        precompiler.finish();
        assert(calls == 2);
        try {
            precompiler.enqueue("c", [&] { ++calls; });
            assert(false && "enqueue after finish should fail");
        } catch (const std::logic_error&) {}
    }
    assert(calls == 2);
}

void test_finish_failure() {
    std::atomic<int> calls = 0;
    {
        deep_jit::Precompiler precompiler(2);
        precompiler.enqueue("first", [&] { ++calls; throw std::runtime_error("first error"); });
        precompiler.enqueue("second", [&] { ++calls; throw std::runtime_error("second error"); });
        precompiler.enqueue("third", [&] { ++calls; });
        for (int i = 0; i < 2; ++i) {
            try {
                precompiler.finish();
                assert(false && "finish should report the first error");
            } catch (const std::runtime_error& error) {
                assert(std::string(error.what()) == "first error");
            }
        }
        assert(calls == 3);
    }
}

void test_empty_and_automatic_finish() {
    {
        deep_jit::Precompiler precompiler(1);
        precompiler.finish();
        precompiler.finish();
    }
    std::atomic<int> calls = 0;
    {
        deep_jit::Precompiler precompiler(2);
        precompiler.enqueue("auto", [&] { ++calls; });
    }
    assert(calls == 1);
}

void test_concurrent_finish_and_enqueue() {
    for (int i = 0; i < 8; ++i) {
        std::atomic<int> calls = 0;
        deep_jit::Precompiler precompiler(2);
        precompiler.enqueue("first", [&] { ++calls; });
        auto first = std::async(std::launch::async, [&] { precompiler.finish(); });
        auto second = std::async(std::launch::async, [&] { precompiler.finish(); });
        bool accepted = true;
        try {
            precompiler.enqueue("second", [&] { ++calls; });
        } catch (const std::logic_error&) {
            accepted = false;
        }
        first.get();
        second.get();
        assert(calls == 1 + static_cast<int>(accepted));
    }
}

int main(int argc, char** argv) {
    if (argc > 1 and std::string_view(argv[1]) == "unobserved-failure") {
        deep_jit::Precompiler precompiler(1);
        precompiler.enqueue("failure", [] { throw std::runtime_error("unobserved error"); });
        return 0;
    }
    test_finish_success();
    test_finish_failure();
    test_empty_and_automatic_finish();
    test_concurrent_finish_and_enqueue();
}
