#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_auction(
          hb_tensor_t* result,
          hb_tensor_t* cost) {

    auto result_ten = HBTensor<float>(result);
    auto cost_ten = HBTensor<float>(cost);

    bsg_cuda_print_stat_kernel_start();

    uint32_t X = cost_ten.dim(0);
    uint32_t Y = cost_ten.dim(1);

    for (size_t x = 0; x < X; ++x) {
        uint32_t argmax = 0;
        float valmax = 0;
        for (size_t y = 0; y < Y; ++y) {
            float val = cost_ten(x, y);
            if (val >= valmax) {
                valmax = val;
                argmax = y;
            }
        }
        result_ten(x, 0) = argmax;
    }

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_auction, hb_tensor_t*, hb_tensor_t*)

}
