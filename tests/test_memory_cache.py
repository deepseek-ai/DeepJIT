import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CPP_SOURCE = r'''
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <deep_jit/cache/memory.hpp>

namespace {

void require(bool condition) {
    if (not condition)
        std::abort();
}

void test_same_key_single_flight() {
    deep_jit::MemCache<std::string, int> cache;
    constexpr int num_threads = 16;
    std::barrier start(num_threads);
    std::atomic<int> factory_calls = 0;
    std::vector<std::shared_ptr<int>> values(num_threads);
    std::vector<std::thread> threads;

    for (int index = 0; index < num_threads; ++index) {
        threads.emplace_back([&, index] {
            start.arrive_and_wait();
            values[index] = cache.get_or_create("same", [&] {
                ++factory_calls;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                return std::make_shared<int>(17);
            });
        });
    }
    for (auto& thread : threads)
        thread.join();

    require(factory_calls == 1);
    for (const auto& value : values)
        require(value == values.front() and *value == 17);
}

void test_different_keys_remain_parallel() {
    deep_jit::MemCache<int, int> cache;
    std::barrier start(2);
    std::atomic<int> active = 0;
    std::atomic<int> max_active = 0;

    auto worker = [&](int key) {
        start.arrive_and_wait();
        return cache.get_or_create(key, [&] {
            const int current = ++active;
            int observed = max_active.load();
            while (observed < current and
                   not max_active.compare_exchange_weak(observed, current)) {}
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            --active;
            return std::make_shared<int>(key);
        });
    };

    std::shared_ptr<int> first;
    std::shared_ptr<int> second;
    std::thread thread_a([&] { first = worker(1); });
    std::thread thread_b([&] { second = worker(2); });
    thread_a.join();
    thread_b.join();

    require(max_active == 2);
    require(*first == 1 and *second == 2);
}

void test_failure_is_shared_and_retryable() {
    deep_jit::MemCache<std::string, int> cache;
    constexpr int num_threads = 8;
    std::barrier start(num_threads);
    std::atomic<int> factory_calls = 0;
    std::atomic<int> failures = 0;
    std::vector<std::thread> threads;

    for (int index = 0; index < num_threads; ++index) {
        threads.emplace_back([&] {
            start.arrive_and_wait();
            try {
                (void)cache.get_or_create("retry", [&]() -> std::shared_ptr<int> {
                    ++factory_calls;
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    throw std::runtime_error("expected failure");
                });
            } catch (const std::runtime_error&) {
                ++failures;
            }
        });
    }
    for (auto& thread : threads)
        thread.join();

    require(factory_calls == 1);
    require(failures == num_threads);
    const auto recovered = cache.get_or_create("retry", [&] {
        ++factory_calls;
        return std::make_shared<int>(41);
    });
    require(factory_calls == 2 and *recovered == 41);
}

}  // namespace

int main() {
    test_same_key_single_flight();
    test_different_keys_remain_parallel();
    test_failure_is_shared_and_retryable();
}
'''


def main() -> None:
    compiler = os.environ.get('CXX', 'c++')
    with tempfile.TemporaryDirectory(prefix='deep-jit-memory-cache-') as directory:
        directory = Path(directory)
        source = directory / 'test.cpp'
        executable = directory / 'test'
        source.write_text(CPP_SOURCE, encoding='utf-8')
        subprocess.run(
            [
                compiler,
                '-std=c++20',
                '-O2',
                '-pthread',
                '-Wall',
                '-Wextra',
                '-Werror',
                '-I',
                str(ROOT / 'include'),
                str(source),
                '-o',
                str(executable),
            ],
            check=True,
            timeout=60,
        )
        subprocess.run([str(executable)], check=True, timeout=30)


if __name__ == '__main__':
    main()
