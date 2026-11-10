#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace deep_jit {

// One producer submits requests; workers only consume jobs. The producer owns
// submitted, so deduplication needs no lock. Destruction drains the queue and
// exits with failure if compilation failed.
class Precompiler {
    using Job = std::packaged_task<void()>;

    std::unordered_set<std::string> submitted;
    std::vector<std::future<void>> results;
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<Job> jobs;
    std::vector<std::thread> workers;
    bool stopping = false;
    bool finished = false;
    std::exception_ptr failure;
    size_t num_completed = 0;
    std::chrono::steady_clock::time_point start_time;

    // Called with mutex held, or after all workers have joined.
    void print_progress() const {
        const auto elapsed = static_cast<long long>(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time).count());
        const auto total = results.size();
        const auto queued = jobs.size();
        const auto running = total - num_completed - queued;

        std::fprintf(stderr,
                     "\r[DeepJIT] %c done %zu/%zu"
                     " | running %zu | queued %zu"
                     " | elapsed %02lld:%02lld\033[K",
                     "|/-\\"[num_completed % 4],
                     num_completed, total, running, queued,
                     elapsed / 60, elapsed % 60);
        std::fflush(stderr);
    }

    void thread_main() {
        while (true) {
            std::unique_lock lock(mutex);
            cv.wait(lock, [&] { return stopping or not jobs.empty(); });
            if (jobs.empty())
                return;
            auto job = std::move(jobs.front());
            jobs.pop_front();
            print_progress();
            lock.unlock();

            job();

            lock.lock();
            ++num_completed;
            print_progress();
        }
    }

    void stop_and_join() noexcept {
        std::unique_lock lock(mutex);
        stopping = true;
        lock.unlock();
        cv.notify_all();
        for (auto& worker: workers)
            worker.join();
        workers.clear();
    }

public:
    explicit Precompiler(const int num_workers) {
        if (num_workers <= 0)
            throw std::invalid_argument("precompiler worker count must be positive");
        try {
            workers.reserve(num_workers);
            for (int worker_idx = 0; worker_idx < num_workers; ++worker_idx)
                workers.emplace_back([this] { thread_main(); });
        } catch (...) {
            stop_and_join();
            throw;
        }
    }

    ~Precompiler() {
        if (finished)
            return;
        try {
            finish();
            return;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "\nDeepJIT precompilation failed: %s\n", error.what());
        } catch (...) {
            std::fputs("\nDeepJIT precompilation failed: unknown exception\n", stderr);
        }
        std::fflush(stderr);
        std::_Exit(EXIT_FAILURE);
    }

    template <typename F>
    void enqueue(const std::string& key, F&& function) {
        if (stopping)
            throw std::logic_error("cannot enqueue after precompilation has finished");
        const auto [position, inserted] = submitted.insert(key);
        if (not inserted)
            return;
        try {
            // NOTE: Monitor memory usage; futures may retain task captures until finish().
            Job job(std::forward<F>(function));
            const std::lock_guard lock(mutex);
            results.push_back(job.get_future());
            try {
                jobs.push_back(std::move(job));
            } catch (...) {
                results.pop_back();
                throw;
            }
            if (results.size() == 1)
                start_time = std::chrono::steady_clock::now();
            print_progress();
        } catch (...) {
            submitted.erase(position);
            throw;
        }
        cv.notify_one();
    }

    // Join all workers, then report the first error in submission order.
    // Repeated calls return or rethrow the same error without consuming futures again.
    void finish() {
        if (finished) {
            if (failure)
                std::rethrow_exception(failure);
            return;
        }
        stop_and_join();
        for (auto& result: results) {
            try {
                result.get();
            } catch (...) {
                if (not failure)
                    failure = std::current_exception();
            }
        }
        finished = true;
        if (not results.empty()) {
            print_progress();
            std::fputc('\n', stderr);
            std::fflush(stderr);
        }
        if (failure)
            std::rethrow_exception(failure);
    }
};

}  // namespace deep_jit
