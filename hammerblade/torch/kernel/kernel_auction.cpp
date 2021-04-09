#include <kernel_common.hpp>
#include <stdio.h>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_auction(
          hb_tensor_t* result_p,
          hb_tensor_t* cost_p) {

    HBTensor<float> result(result_p);
    HBTensor<float> cost(cost_p);

    bsg_cuda_print_stat_kernel_start();

    uint32_t X = cost.dim(0);
    uint32_t Y = cost.dim(1);

    uint32_t mask = 0;

    if (__bsg_id == 0) {

    for (size_t x = 0; x < X; ++x) {
        uint32_t argmax = 0;
        float valmax = 0;
        for (size_t y = 0; y < Y; ++y) {
            if (mask & (1 << y)) {
                continue;
            }

            float val = cost(x, y);
            if (val >= valmax) {
                valmax = val;
                argmax = y;
            }
        }

        result(x) = argmax;
        mask |= 1 << argmax;
    }

    }

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_auction, hb_tensor_t*, hb_tensor_t*)

}
