#include <deep_jit/utils/executor.hpp>

#include <cassert>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace std::chrono_literals;
using deep_jit::IOTimeoutError;
using deep_jit::ThreadExecutor;

thread_local int worker_marker = 0;

template<typename F>
void expect_timeout(F&& f) {
    try {
        f();
        assert(false && "expected a timeout");
    } catch (const IOTimeoutError&) {}
}

void test_results_and_errors() {
    ThreadExecutor executor;
    auto id = executor.run([] { return std::this_thread::get_id(); });
    assert(id != std::this_thread::get_id());
    assert(executor.run([] { return std::this_thread::get_id(); }) == id);
    assert(executor.run([p = std::make_unique<int>(42)] { return *p; }) == 42);
    auto value = executor.run([] { return std::make_unique<int>(17); });
    assert(*value == 17);
    int count = 0;
    auto increment = [&] { ++count; };
    executor.run(increment, 0ms);
    assert(count == 1);
    assert(&executor.run([&]() -> int& { return count; }) == &count);
    try {
        executor.run([] {
            worker_marker = 77;
            throw std::runtime_error("injected I/O failure");
        });
        assert(false);
    } catch (const std::runtime_error& error) {
        assert(std::string(error.what()) == "injected I/O failure");
    }
    assert(executor.run([] { return worker_marker; }) == 0);
}

void test_independent_timeouts() {
    ThreadExecutor executor;
    std::promise<void> started, release, finished;
    auto gate = release.get_future().share();
    auto old_worker = std::async(std::launch::async, [&] {
        expect_timeout([&] {
            executor.run([&, gate] {
                started.set_value();
                gate.wait();
                finished.set_value();
            }, 100ms);
        });
    });
    started.get_future().get();
    std::vector<std::future<int>> callers;
    for (int i = 0; i < 4; ++i) {
        callers.push_back(std::async(std::launch::async, [&executor, i] {
            return executor.run([i] {
                std::this_thread::sleep_for(60ms);
                return i;
            }, 200ms);
        }));
    }
    old_worker.get();
    for (int i = 0; i < 4; ++i)
        assert(callers[i].get() == i);
    release.set_value();
    finished.get_future().get();
    assert(executor.run([] { return 42; }) == 42);
}

void test_capture_destruction() {
    ThreadExecutor executor;
    std::promise<void> release, destroyed;
    auto gate = release.get_future().share();
    const auto caller = std::this_thread::get_id();
    auto deleter = [&, gate, caller](int* pointer) {
        assert(std::this_thread::get_id() != caller);
        gate.wait();
        delete pointer;
        destroyed.set_value();
    };
    expect_timeout([&] {
        executor.run([p = std::unique_ptr<int, decltype(deleter)>(new int(7), deleter)] {
            return *p;
        }, 50ms);
    });
    assert(executor.run([] { return 23; }) == 23);
    release.set_value();
    destroyed.get_future().get();
}

void test_timeout_completion_race() {
    ThreadExecutor executor;
    for (int i = 0; i < 150; ++i) {
        try {
            executor.run([] { std::this_thread::sleep_for(1ms); }, 1ms);
        } catch (const IOTimeoutError&) {}
        assert(executor.run([i] { return i; }, 500ms) == i);
    }
}

void test_consecutive_timeouts() {
    ThreadExecutor executor;
    std::promise<void> release;
    auto gate = release.get_future().share();
    std::vector<std::future<void>> finished;
    for (int i = 0; i < 3; ++i) {
        std::promise<void> completion;
        finished.push_back(completion.get_future());
        const auto start = std::chrono::steady_clock::now();
        expect_timeout([&] {
            executor.run([gate, completion = std::move(completion)]() mutable {
                gate.wait();
                completion.set_value();
            }, 40ms);
        });
        assert(std::chrono::steady_clock::now() - start >= 40ms);
    }
    assert(executor.run([] { return 11; }) == 11);
    release.set_value();
    for (auto& future : finished)
        future.get();
}

void test_run_in_iothread() {
    const auto worker = deep_jit::run_in_iothread([] { return std::this_thread::get_id(); });
    assert(worker != std::this_thread::get_id());
    assert(deep_jit::run_in_iothread([worker] {
        assert(std::this_thread::get_id() == worker);
        return deep_jit::run_in_iothread([p = std::make_unique<int>(42)] { return *p; });
    }) == 42);
    try {
        deep_jit::run_in_iothread([] {
            deep_jit::run_in_iothread([] { throw std::runtime_error("nested failure"); });
        });
        assert(false);
    } catch (const std::runtime_error& error) {
        assert(std::string(error.what()) == "nested failure");
    }
    assert(deep_jit::run_in_iothread([] { return 7; }) == 7);
}

int forked_child_result(bool concurrent) {
    ::alarm(3);  // Bound a hang in the child without affecting other processes.
    try {
        if (concurrent) {
            std::vector<std::thread::id> ids(8);
            std::vector<std::thread> threads;
            for (size_t i = 0; i != ids.size(); ++i) {
                threads.emplace_back([&, i] {
                    ids[i] = deep_jit::run_in_iothread([] { return std::this_thread::get_id(); });
                });
            }
            for (auto& thread : threads) thread.join();
            for (const auto& id : ids) if (id != ids.front()) return 2;
        } else if (deep_jit::run_in_iothread([] { return 42; }) != 42) {
            return 3;
        }
        return 0;
    } catch (...) {
        return 4;
    }
}

void wait_for_child(pid_t child) {
    assert(child > 0);
    int status = 0;
    assert(::waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) and WEXITSTATUS(status) == 0);
}

void test_forked_shared_executor() {
    const auto parent_worker = deep_jit::run_in_iothread([] { return std::this_thread::get_id(); });
    auto child = ::fork();
    assert(child >= 0);
    if (child == 0) ::_exit(forked_child_result(true));
    wait_for_child(child);
    assert(deep_jit::run_in_iothread([] { return std::this_thread::get_id(); }) == parent_worker);

    // Fork while a different parent thread holds the executor submission lock.
    std::promise<void> entered, release;
    auto entered_future = entered.get_future();
    auto gate = release.get_future().share();
    std::thread submitter([&] {
        deep_jit::run_in_iothread([&] { entered.set_value(); gate.wait(); });
    });
    entered_future.wait();
    child = ::fork();
    assert(child >= 0);
    if (child == 0) ::_exit(forked_child_result(false));
    release.set_value();
    submitter.join();
    wait_for_child(child);

    // A fork from the I/O worker must not treat the child's main thread as
    // an I/O worker just because the thread-local marker was inherited.
    deep_jit::run_in_iothread([] {
        const auto child = ::fork();
        assert(child >= 0);
        if (child == 0) {
            ::alarm(3);
            try {
                const auto caller = std::this_thread::get_id();
                const auto worker = deep_jit::run_in_iothread([] { return std::this_thread::get_id(); });
                ::_exit(worker == caller ? 5 : 0);
            } catch (...) {
                ::_exit(6);
            }
        }
        wait_for_child(child);
    });
    assert(deep_jit::run_in_iothread([] { return 7; }) == 7);
}

int main(int argc, char** argv) {
    if (argc > 1) {
        if (std::string_view(argv[1]) == "fork") {
            test_forked_shared_executor();
            return 0;
        }
        if (std::string_view(argv[1]) == "disable-fallback") {
            const auto caller = std::this_thread::get_id();
            assert(deep_jit::detail::disable_cache_fallback());
            assert(deep_jit::run_in_iothread([] { return std::this_thread::get_id(); }) == caller);
            assert(not deep_jit::detail::in_iothread);
            return 0;
        }
        // The process must exit without joining this permanently blocked worker.
        std::promise<void> release;
        auto gate = release.get_future().share();
        ThreadExecutor executor;
        expect_timeout([&] { executor.run([gate] { gate.wait(); }, 20ms); });
        return 0;
    }
    test_results_and_errors();
    test_independent_timeouts();
    test_capture_destruction();
    test_timeout_completion_race();
    test_consecutive_timeouts();
    test_run_in_iothread();
    std::cout << "Executor tests passed\n";
}
