#include <deep_jit/utils/lazy.hpp>

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

struct Value { int id; };
struct ThrowingFactory {
    std::shared_ptr<bool> fail_copy;
    explicit ThrowingFactory(std::shared_ptr<bool> fail_copy) : fail_copy(std::move(fail_copy)) {}
    ThrowingFactory(const ThrowingFactory& other) : fail_copy(other.fail_copy) {
        if (*fail_copy) throw std::runtime_error("copy failed");
    }
    std::shared_ptr<Value> operator()() const { return std::make_shared<Value>(Value{11}); }
};

int main() {
    std::mutex mutex;
    std::condition_variable cv;
    int calls = 0;
    bool second_entered = false, first_returned = false;
    deep_jit::LazyInit<Value> lazy([&] {
        std::unique_lock lock(mutex);
        const int id = ++calls;
        if (id == 1) {
            cv.wait_for(lock, std::chrono::seconds(2), [&] { return second_entered; });
        } else {
            second_entered = true;
            cv.notify_all();
            cv.wait(lock, [&] { return first_returned; });
        }
        return std::make_shared<Value>(Value{id});
    });
    std::shared_ptr<Value> values[2];
    auto run = [&](int index) {
        values[index] = lazy.get();
        if (values[index]->id == 1) {
            std::lock_guard lock(mutex);
            first_returned = true;
            cv.notify_all();
        }
    };
    std::thread a(run, 0), b(run, 1);
    a.join();
    b.join();
    assert(calls == 1 and values[0] == values[1]);

    int attempts = 0;
    deep_jit::LazyInit<Value> retry([&] {
        if (++attempts == 1) throw std::runtime_error("first attempt");
        return std::make_shared<Value>(Value{7});
    });
    try { retry.get(); assert(false); } catch (const std::runtime_error&) {}
    assert(retry.get()->id == 7 and attempts == 2);

    auto copy = retry;
    assert(copy.get() == retry.get());
    deep_jit::LazyInit<Value> moved(std::move(copy));
    assert(moved.get() == retry.get());
    deep_jit::LazyInit<Value> assigned(nullptr);
    assigned = deep_jit::LazyInit<Value>([] { return std::make_shared<Value>(Value{9}); });
    assert(assigned->id == 9);
    auto fail_copy = std::make_shared<bool>(false);
    deep_jit::LazyInit<Value> source{std::function<std::shared_ptr<Value>()>(ThrowingFactory(fail_copy))};
    *fail_copy = true;
    try { assigned = source; assert(false); } catch (const std::runtime_error&) {}
    assert(assigned->id == 9);
}
