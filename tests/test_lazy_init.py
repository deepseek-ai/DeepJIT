import os
import tempfile
from pathlib import Path

from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[1]
INCLUDE_ROOT = Path(os.environ.get("DEEP_JIT_TEST_INCLUDE_ROOT", ROOT / "include"))


CPP_SOURCE = r'''
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <deep_jit/utils/lazy.hpp>

namespace py = pybind11;

py::tuple initialize_concurrently() {
    constexpr int num_threads = 16;
    std::atomic<int> factory_calls = 0;
    std::atomic<bool> start = false;
    std::atomic<bool> release_factory = false;
    auto value = std::make_shared<int>(42);
    deep_jit::LazyInit<int> lazy([&] {
        factory_calls.fetch_add(1, std::memory_order_relaxed);
        while (not release_factory.load(std::memory_order_acquire))
            std::this_thread::yield();
        return value;
    });

    std::vector<std::uintptr_t> addresses(num_threads);
    std::vector<int> values(num_threads);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int index = 0; index < num_threads; ++index) {
        threads.emplace_back([&, index] {
            while (not start.load(std::memory_order_acquire))
                std::this_thread::yield();
            const auto result = lazy.get();
            addresses[index] = reinterpret_cast<std::uintptr_t>(result.get());
            values[index] = *result;
        });
    }

    start.store(true, std::memory_order_release);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (factory_calls.load(std::memory_order_relaxed) == 0 and
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    release_factory.store(true, std::memory_order_release);
    for (auto& thread : threads)
        thread.join();

    return py::make_tuple(factory_calls.load(), addresses, values);
}

py::tuple retry_after_failure() {
    int attempts = 0;
    deep_jit::LazyInit<int> lazy([&] {
        if (attempts++ == 0)
            throw std::runtime_error("expected factory failure");
        return std::make_shared<int>(7);
    });
    try {
        (void)lazy.get();
    } catch (const std::runtime_error&) {
    }
    return py::make_tuple(attempts, *lazy.get());
}

py::tuple compatibility_paths() {
    int empty_errors = 0;
    deep_jit::LazyInit<int> empty(nullptr);
    for (int attempt = 0; attempt < 2; ++attempt) {
        try {
            (void)empty.get();
        } catch (const std::exception&) {
            ++empty_errors;
        }
    }

    int null_calls = 0;
    int null_errors = 0;
    deep_jit::LazyInit<int> null_factory([&] {
        ++null_calls;
        return std::shared_ptr<int>();
    });
    for (int attempt = 0; attempt < 2; ++attempt) {
        try {
            (void)null_factory.get();
        } catch (const std::exception&) {
            ++null_errors;
        }
    }

    deep_jit::LazyInit<int> assigned(nullptr);
    assigned = deep_jit::LazyInit<int>([] { return std::make_shared<int>(11); });

    int copied_factory_calls = 0;
    deep_jit::LazyInit<int> original([&] {
        ++copied_factory_calls;
        return std::make_shared<int>(12);
    });
    auto copied = original;
    const auto original_value = original.get();
    const auto copied_value = copied.get();

    deep_jit::LazyInit<int> source([] { return std::make_shared<int>(13); });
    deep_jit::LazyInit<int> moved(std::move(source));
    int moved_from_errors = 0;
    try {
        (void)source.get();
    } catch (const std::exception&) {
        ++moved_from_errors;
    }
    return py::make_tuple(
        empty_errors,
        null_calls,
        null_errors,
        *assigned.get(),
        copied_factory_calls,
        original_value.get() == copied_value.get(),
        *moved.get(),
        moved_from_errors);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("initialize_concurrently", &initialize_concurrently);
    module.def("retry_after_failure", &retry_after_failure);
    module.def("compatibility_paths", &compatibility_paths);
}
'''


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="deep-jit-lazy-") as directory:
        module = load_inline(
            name="deep_jit_lazy_test",
            cpp_sources=CPP_SOURCE,
            extra_cflags=[
                "-std=c++20",
                "-O2",
                "-pthread",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-Wno-attributes",
            ],
            extra_ldflags=["-ldw", "-ldl", "-pthread"],
            extra_include_paths=[str(INCLUDE_ROOT)],
            build_directory=directory,
            with_cuda=False,
            verbose=True,
        )

        for _ in range(25):
            factory_calls, addresses, values = module.initialize_concurrently()
            assert factory_calls == 1, factory_calls
            assert len(set(addresses)) == 1, addresses
            assert values == [42] * 16, values
        assert module.retry_after_failure() == (2, 7)
        assert module.compatibility_paths() == (2, 2, 2, 11, 1, True, 13, 1)


if __name__ == "__main__":
    main()
