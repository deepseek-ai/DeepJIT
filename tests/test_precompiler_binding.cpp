#include "precompiler_fake_backend.hpp"

#include <pybind11/pybind11.h>
#include <deep_jit/python_api.hpp>

namespace py = pybind11;

inline deep_jit::LazyInit<precompiler_test::Runtime> jit([] {
    return std::make_shared<precompiler_test::Runtime>(
        deep_jit::Config(std::getenv("TEST_JIT_CACHE_DIR"), "TEST"));
});

PYBIND11_MODULE(_deepjit_precompiler_test, module) {
    deep_jit::register_python_api(module, jit);
    module.def("submit", [](const std::string& name, const std::string& source) {
        jit->compile(name, source);
    });
    module.def("release", &precompiler_test::release);
    module.def("calls", [] { return precompiler_test::compile_calls.load(); });
    module.def("started", [] { return precompiler_test::started.load(); });
}
