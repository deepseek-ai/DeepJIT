#pragma once

#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace deep_jit {

// In-memory cache keyed by value type.
template <typename Key, typename Value>
class MemCache {
    using ValuePtr = std::shared_ptr<Value>;

    std::unordered_map<Key, std::shared_future<ValuePtr>> pending;
    mutable std::mutex mutex;

public:
    // Kept public for source compatibility. Concurrent callers must use
    // get_or_create() rather than accessing the container directly.
    std::unordered_map<Key, ValuePtr> cache;

    template <typename Factory>
    ValuePtr get_or_create(const Key& key, Factory&& factory) {
        std::shared_future<ValuePtr> future;
        std::shared_ptr<std::promise<ValuePtr>> producer;
        {
            std::lock_guard lock(mutex);
            if (const auto iterator = cache.find(key); iterator != cache.end())
                return iterator->second;
            if (const auto iterator = pending.find(key); iterator != pending.end()) {
                future = iterator->second;
            } else {
                producer = std::make_shared<std::promise<ValuePtr>>();
                future = producer->get_future().share();
                pending.emplace(key, future);
            }
        }

        // A miss for a different key must not wait for this factory. Callers
        // that lost the race for the same key wait on the shared result.
        if (producer == nullptr)
            return future.get();

        try {
            auto value = factory();
            {
                std::lock_guard lock(mutex);
                cache.emplace(key, value);
                pending.erase(key);
            }
            producer->set_value(value);
            return value;
        } catch (...) {
            const auto exception = std::current_exception();
            producer->set_exception(exception);
            {
                std::lock_guard lock(mutex);
                pending.erase(key);
            }
            std::rethrow_exception(exception);
        }
    }
};

}  // namespace deep_jit
