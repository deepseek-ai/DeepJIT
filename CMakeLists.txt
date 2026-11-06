# NOTES: this CMake is only for debugging/IDE indexing; for setup, please use Python extension
cmake_minimum_required(VERSION 3.20)
project(deep_jit LANGUAGES CXX)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 20)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

include_directories(
    include
    ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS}
)
link_directories(${TORCH_INSTALL_PREFIX}/lib)

pybind11_add_module(_C csrc/python_api.cpp)
target_compile_definitions(_C PRIVATE
    TORCH_TARGET_VERSION=0x020a000000000000
    USE_CUDA
)
if(MSVC)
    target_compile_options(_C PRIVATE /O2 /Zc:preprocessor /Zc:__cplusplus)
    target_compile_definitions(_C PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    target_compile_options(_C PRIVATE -O3 -fPIC -Wno-psabi -Wno-deprecated-declarations)
    target_link_options(_C PRIVATE "-Wl,-Bsymbolic-functions")
endif()
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} torch_python CUDA::cudart)
