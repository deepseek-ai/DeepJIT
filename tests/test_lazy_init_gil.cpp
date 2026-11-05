#include <deep_jit/utils/lazy.hpp>
#include <pybind11/pybind11.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

namespace py = pybind11;
static std::atomic<int> entered{0}, calls{0};
static deep_jit::LazyInit<int> lazy([] {
    int id = ++calls;
    if (id == 1) {
        py::gil_scoped_release release;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (entered.load() < 2 and std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return std::make_shared<int>(id);
});

PYBIND11_MODULE(test_lazy_init_gil, module) {
    module.def("get", [] { ++entered; return *lazy.get(); });
    module.def("calls", [] { return calls.load(); });
}
