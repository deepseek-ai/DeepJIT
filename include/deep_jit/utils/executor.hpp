#pragma once

#include <atomic>
#include <charconv>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <unistd.h>

namespace deep_jit {

namespace detail {
inline thread_local bool in_iothread = false;
inline thread_local pid_t iothread_pid = 0;
}

class IOTimeoutError : public std::runtime_error {
public:
    IOTimeoutError() : std::runtime_error("I/O task timed out") {}
};

class ThreadExecutor {
    struct Context {
        std::mutex lock;
        std::condition_variable cv;
        std::packaged_task<void()> task;
        bool done = false;
        bool stopped = false;
    };

    // Serialize submissions through completion/retirement. Queueing does not
    // consume a task's timeout, and an old timeout cannot retire a newer task.
    std::mutex run_lock;
    std::shared_ptr<Context> context;
    std::thread worker;

    static void thread_main(std::shared_ptr<Context> ctx) {
        detail::in_iothread = true;
        detail::iothread_pid = ::getpid();
        while (true) {
            std::unique_lock<std::mutex> lock(ctx->lock);
            ctx->cv.wait(lock, [&] { return ctx->stopped or ctx->task.valid(); });
            auto task = std::move(ctx->task);
            const bool stopped = ctx->stopped;
            lock.unlock();

            // Even a task abandoned before execution is destroyed on the worker.
            if (stopped)
                return;
            task();
            task = {};

            // Capture destruction is part of the timed operation, outside locks.
            lock.lock();
            ctx->done = true;
            lock.unlock();
            ctx->cv.notify_one();
        }
    }

    // Called with run_lock held, or during destruction. Never join stuck I/O.
    void stop() {
        if (not context)
            return;
        {
            const std::lock_guard lock(context->lock);
            context->stopped = true;
        }
        context->cv.notify_one();
        worker.detach();
        context.reset();
    }

public:
    ThreadExecutor() = default;
    ThreadExecutor(const ThreadExecutor&) = delete;
    ThreadExecutor& operator=(const ThreadExecutor&) = delete;

    ~ThreadExecutor() {
        stop();
    }

    // Captures must own everything used after a timeout; borrowed objects must
    // outlive the worker. Concurrent run() calls are supported, but recursive
    // calls on the same executor and concurrent destruction are not.
    // Zero disables the timeout. Timing starts at submission to the idle worker.
    template<typename F>
        requires std::invocable<std::decay_t<F>&>
    auto run(F&& f, std::chrono::milliseconds timeout = std::chrono::seconds(10))
        -> std::invoke_result_t<std::decay_t<F>&> {
        using Result = std::invoke_result_t<std::decay_t<F>&>;
        if (timeout.count() < 0)
            throw std::invalid_argument("I/O timeout must be nonnegative");
        const std::lock_guard submission_lock(run_lock);

        std::promise<Result> promise;
        auto result = promise.get_future();
        std::packaged_task<void()> task(
            [f = std::forward<F>(f), promise = std::move(promise)]() mutable {
                try {
                    if constexpr (std::is_void_v<Result>) {
                        std::invoke(f);
                        promise.set_value();
                    } else {
                        promise.set_value(std::invoke(f));
                    }
                } catch (...) {
                    promise.set_exception(std::current_exception());
                }
            });

        if (not context) {
            auto ctx = std::make_shared<Context>();
            worker = std::thread(thread_main, ctx);
            context = std::move(ctx);
        }
        std::unique_lock lock(context->lock);
        context->done = false;
        context->task = std::move(task);
        context->cv.notify_one();
        bool done = true;
        if (timeout.count() == 0)
            context->cv.wait(lock, [&] { return context->done; });
        else
            done = context->cv.wait_for(lock, timeout, [&] { return context->done; });
        lock.unlock();

        if (not done) {
            stop();
            throw IOTimeoutError();
        }
        try {
            return result.get();
        } catch (...) {
            stop();
            throw;
        }
    }
};

namespace detail {

// Initialize before global JIT handles so their workers can use IO during shutdown.
// A forked child cannot use the parent's worker or any inherited mutexes.
// Publish process-owned state through a lock-free pointer so the first child
// caller can replace it without touching the inherited executor. The old
// state is deliberately left alone in the child: it may contain a locked
// mutex or a std::thread referring to a thread that no longer exists.
class ProcessIOExecutor {
    struct State {
        const pid_t pid;
        ThreadExecutor executor;
        explicit State(pid_t pid) : pid(pid) {}
    };

    static_assert(std::atomic<State*>::is_always_lock_free);
    std::atomic<State*> state{nullptr};

public:
    constexpr ProcessIOExecutor() = default;
    ~ProcessIOExecutor() {
        auto* current = state.load(std::memory_order_acquire);
        if (current != nullptr and current->pid == ::getpid())
            delete current;
    }

    ThreadExecutor& get() {
        const auto pid = ::getpid();
        auto* current = state.load(std::memory_order_acquire);
        while (current == nullptr or current->pid != pid) {
            auto next = std::make_unique<State>(pid);
            if (state.compare_exchange_strong(current, next.get(), std::memory_order_acq_rel))
                return next.release()->executor;
        }
        return current->executor;
    }
};

inline constinit ProcessIOExecutor process_io_executor;

inline ThreadExecutor& io_executor() {
    return process_io_executor.get();
}

inline std::chrono::milliseconds io_timeout() {
    static const auto timeout = [] {
        int ms = 10'000;
        if (const char* raw = std::getenv("DJ_FS_TIMEOUT_MS")) {
            const auto end = raw + std::strlen(raw);
            const auto [ptr, error] = std::from_chars(raw, end, ms);
            if (error != std::errc{} or ptr != end or ms < 0)
                throw std::invalid_argument("invalid DJ_FS_TIMEOUT_MS");
        }
        return std::chrono::milliseconds(ms);
    }();
    return timeout;
}

inline bool disable_cache_fallback() {
    static const bool disabled = [] {
        int value = 0;
        if (const char* raw = std::getenv("DJ_DISABLE_CACHE_FALLBACK")) {
            const auto end = raw + std::strlen(raw);
            const auto [ptr, error] = std::from_chars(raw, end, value);
            if (error != std::errc{} or ptr != end)
                throw std::invalid_argument("invalid DJ_DISABLE_CACHE_FALLBACK");
        }
        return value != 0;
    }();
    return disabled;
}

} // namespace detail

// Own captures: a timed-out operation may outlive its caller. Nested calls
// execute directly on the worker under the outer operation's timeout.
template<typename F>
auto run_in_iothread(F&& f) -> std::invoke_result_t<std::decay_t<F>&> {
    if ((detail::in_iothread and detail::iothread_pid == ::getpid()) or detail::disable_cache_fallback())
        return std::invoke(f);
    return detail::io_executor().run(std::forward<F>(f), detail::io_timeout());
}

} // namespace deep_jit
