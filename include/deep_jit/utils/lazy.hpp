#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <dlfcn.h>

#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/gil.hpp>

#define DJ_DECLARE_STATIC_VAR_IN_CLASS(cls, name) decltype(cls::name) cls::name

namespace deep_jit {

template <typename T>
class LazyInit {
    std::shared_ptr<T> ptr;
    std::function<std::shared_ptr<T>()> factory;
    std::atomic<T*> ready{nullptr};
    std::mutex mutex;

    std::shared_ptr<T> initialize() {
        if (ready.load(std::memory_order_acquire) != nullptr)
            return ptr;
        std::unique_lock lock(mutex, std::defer_lock);
        {
            // Another factory may need the GIL before it can finish and
            // release this mutex. Preserve the caller's GIL state for the
            // factory itself, but never hold it while waiting for the lock.
            [[maybe_unused]] GilScopedRelease gil_release;
            lock.lock();
        }
        DJ_HOST_ASSERT(factory != nullptr, "lazy object must be initialized before use");
        if (ptr == nullptr)
            ptr = factory();
        DJ_HOST_ASSERT(ptr != nullptr, "lazy factory must not return nullptr");
        ready.store(ptr.get(), std::memory_order_release);
        return ptr;
    }

public:
    explicit LazyInit(std::nullptr_t) {}

    explicit LazyInit(std::function<std::shared_ptr<T>()> factory)
        : factory(std::move(factory)) {}

    // Copy, assignment, and destruction must not race with access. As before,
    // copying an uninitialized object copies its factory, while copying an
    // initialized object shares its value.
    LazyInit(const LazyInit& other)
        : ptr(other.ptr), factory(other.factory), ready(ptr.get()) {}
    LazyInit& operator=(const LazyInit& other) {
        if (this != &other) {
            auto new_ptr = other.ptr;
            auto new_factory = other.factory;
            ptr = std::move(new_ptr);
            factory = std::move(new_factory);
            ready.store(ptr.get(), std::memory_order_release);
        }
        return *this;
    }
    LazyInit(LazyInit&& other) noexcept
        : ptr(std::move(other.ptr)), factory(std::move(other.factory)), ready(ptr.get()) {
        other.ready.store(nullptr, std::memory_order_relaxed);
    }
    LazyInit& operator=(LazyInit&& other) noexcept {
        if (this != &other) {
            ptr = std::move(other.ptr);
            factory = std::move(other.factory);
            ready.store(ptr.get(), std::memory_order_release);
            other.ready.store(nullptr, std::memory_order_relaxed);
        }
        return *this;
    }

    T* operator->() {
        if (auto* value = ready.load(std::memory_order_acquire))
            return value;
        return initialize().get();
    }

    std::shared_ptr<T> get() {
        return initialize();
    }
};

}  // namespace deep_jit

#define DJ_DECL_LAZY_DL_HANDLE(handle_func_name, lib)                                         \
    inline void* handle_func_name() {                                                         \
        static void* handle = [] {                                                            \
            ::dlerror();                                                                      \
            void* value = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);                                \
            if (value == nullptr) {                                                           \
                const char* error = ::dlerror();                                              \
                DJ_PANIC("failed to load {}: {}", lib, error == nullptr ? "unknown" : error); \
            }                                                                                 \
            return value;                                                                     \
        }();                                                                                  \
        return handle;                                                                        \
    }

#define DJ_STRINGIFY(name) #name

#define DJ_DECL_LAZY_DL_FUNCTION(handle_func_name, name)                                                                   \
    template <typename... Args>                                                                                            \
    static auto lazy_##name(Args&&... args) {                                                                              \
        static const auto func = []() {                                                                                    \
            void* symbol = ::dlsym(handle_func_name(), DJ_STRINGIFY(name));                                                \
            if (symbol == nullptr) {                                                                                       \
                const char* error = ::dlerror();                                                                           \
                DJ_PANIC("failed to load {} from {}: {}", DJ_STRINGIFY(name),                                              \
                         #handle_func_name, error == nullptr ? "unknown" : error);                                         \
            }                                                                                                              \
            return reinterpret_cast<decltype(&name)>(symbol);                                                              \
        }();                                                                                                               \
        return func(std::forward<Args>(args)...);                                                                          \
    }
