#include <test_cuda/tracked_kernel.cuh>

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&add_from_tracked_include<0>);
}
