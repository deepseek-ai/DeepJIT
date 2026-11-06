#pragma once

#include <chrono>
#include <format>
#include <random>
#include <string>
#include <thread>
#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace deep_jit {

// Process-unique id: pid + three random words. The generator is thread_local
// and seeded per thread so concurrent callers never collide.
inline std::string get_uuid() {
    thread_local std::mt19937 generator([]() {
        std::random_device rd;
        return static_cast<std::mt19937::result_type>(rd()) ^
            static_cast<std::mt19937::result_type>(std::chrono::steady_clock::now().time_since_epoch().count()) ^
            static_cast<std::mt19937::result_type>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    }());
    thread_local std::uniform_int_distribution<uint32_t> distribution;
#if defined(_WIN32)
    const auto pid = ::_getpid();
#else
    const auto pid = ::getpid();
#endif
    return std::format("{}-{:08x}-{:08x}-{:08x}", pid, distribution(generator), distribution(generator), distribution(generator));
}

}  // namespace deep_jit
