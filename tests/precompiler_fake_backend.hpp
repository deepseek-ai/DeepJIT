#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>

#include <deep_jit/runtime/runtime.hpp>

namespace precompiler_test {

inline std::atomic<int> compile_calls = 0;
inline std::atomic<bool> started = false;
inline std::mutex gate_mutex;
inline std::condition_variable gate_cv;
inline bool released = false;

struct Device {};

struct CompilerOptions {
    static CompilerOptions default_options(const deep_jit::Env&, const Device&) { return {}; }
    CompilerOptions override_with(const CompilerOptions&) const { return {}; }
    void update_hash(deep_jit::hash::FNV1a&) const {}
    std::string get_post_hook_hash(const deep_jit::Config&) const { return {}; }
};

struct LaunchOptions {
    static LaunchOptions default_options(const deep_jit::Env&) { return {}; }
    LaunchOptions override_with(const LaunchOptions&) const { return {}; }
};

struct Kernel {
    template <typename... Args>
    void launch(const LaunchOptions&, const Args&...) {}
};

struct Backend {
    using Device = precompiler_test::Device;
    using CompilerOptions = precompiler_test::CompilerOptions;
    using LaunchOptions = precompiler_test::LaunchOptions;
    using Kernel = precompiler_test::Kernel;

    struct CompilerInfo {
        std::string get_hash() const { return "fake-compiler"; }
    } compiler_info;

    explicit Backend(const deep_jit::Env&) {}

    void compile(const std::string& source, const std::filesystem::path& path,
                 const deep_jit::Env&, const deep_jit::Config&, const CompilerOptions&) const {
        ++compile_calls;
        if (source == "fail")
            throw std::runtime_error("injected compiler failure");
        if (source == "block") {
            std::unique_lock lock(gate_mutex);
            started = true;
            gate_cv.wait(lock, [] { return released; });
        }
        std::ofstream(path / "kernel") << source;
    }

    static std::shared_ptr<Kernel> load(const std::filesystem::path&, const deep_jit::Env&) {
        return std::make_shared<Kernel>();
    }
};

using Runtime = deep_jit::Runtime<Backend>;

inline void release() {
    {
        const std::lock_guard lock(gate_mutex);
        released = true;
    }
    gate_cv.notify_all();
}

}  // namespace precompiler_test
