#include <pybind11/pybind11.h>

#include <deep_jit/backend/cuda/backend.hpp>
#include <deep_jit/python_api.hpp>

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    // Consumer libraries register their configured runtime with `register_python_api`.
}
