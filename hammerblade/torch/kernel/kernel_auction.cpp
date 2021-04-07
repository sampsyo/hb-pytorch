#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_auction(
          hb_tensor_t* result,
          hb_tensor_t* cost) {

    auto result_ten = HBTensor<float>(result);
    auto cost_ten = HBTensor<float>(cost);

    bsg_cuda_print_stat_kernel_start();

    // TODO

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_auction, hb_tensor_t*, hb_tensor_t*)

}
