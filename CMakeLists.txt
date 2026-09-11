# NOTES: this CMake is only for debugging/IDE indexing; for setup, please use Python extension
cmake_minimum_required(VERSION 3.20)
project(deep_jit LANGUAGES CXX)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wno-psabi -Wno-deprecated-declarations")
set(CMAKE_CXX_STANDARD 20)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

include_directories(
    include
    ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS}
)
link_directories(${TORCH_INSTALL_PREFIX}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)

pybind11_add_module(_C csrc/python_api.cpp)
target_compile_definitions(_C PRIVATE
    TORCH_TARGET_VERSION=0x020a000000000000
    USE_CUDA
)
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} torch_python)
target_link_options(_C PRIVATE "-Wl,-Bsymbolic-functions")
