#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <dlfcn.h>

#include <deep_jit/utils/exception.hpp>

#define DJ_DECLARE_STATIC_VAR_IN_CLASS(cls, name) decltype(cls::name) cls::name

namespace deep_jit {

// Concurrent first use initializes one shared object. Copies share the same
// initialization state, and a factory that throws can be retried by a later
// call.
template <typename T>
class LazyInit {
    struct State {
        std::once_flag init_once;
        std::shared_ptr<T> ptr;
        std::function<std::shared_ptr<T>()> factory;

        State() = default;

        explicit State(std::function<std::shared_ptr<T>()> factory)
            : factory(std::move(factory)) {}
    };

    std::shared_ptr<State> state;

public:
    explicit LazyInit(std::nullptr_t)
        : state(std::make_shared<State>()) {}

    explicit LazyInit(std::function<std::shared_ptr<T>()> factory)
        : state(std::make_shared<State>(std::move(factory))) {}

    T* operator->() {
        return get().get();
    }

    std::shared_ptr<T> get() {
        const auto current = state;
        DJ_HOST_ASSERT(current != nullptr, "lazy object must be initialized before use");
        std::call_once(current->init_once, [&] {
            DJ_HOST_ASSERT(current->factory != nullptr, "lazy object must be initialized before use");
            auto ptr = current->factory();
            DJ_HOST_ASSERT(ptr != nullptr, "lazy factory must not return nullptr");
            current->ptr = std::move(ptr);
        });
        return current->ptr;
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

#define DJ_DECL_LAZY_DL_FUNCTION(handle_func_name, name)                                                                   \
    template <typename... Args>                                                                                            \
    static auto lazy_##name(Args&&... args) {                                                                              \
        static const auto func = []() {                                                                                    \
            void* symbol = ::dlsym(handle_func_name(), #name);                                                             \
            if (symbol == nullptr) {                                                                                       \
                const char* error = ::dlerror();                                                                           \
                DJ_PANIC("failed to load {} from {}: {}", #name, #handle_func_name, error == nullptr ? "unknown" : error); \
            }                                                                                                              \
            return reinterpret_cast<decltype(&name)>(symbol);                                                              \
        }();                                                                                                               \
        return func(std::forward<Args>(args)...);                                                                          \
    }
