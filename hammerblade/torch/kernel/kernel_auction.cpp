#include <kernel_common.hpp>
#include <stdio.h>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_auction(
          hb_tensor_t* result_p,
          hb_tensor_t* cost_p) {

    HBTensor<float> result(result_p);
    HBTensor<float> cost(cost_p);

    size_t* buffer = (size_t*)g_reduction_buffer;

    bsg_cuda_print_stat_kernel_start();

    uint32_t X = cost.dim(0);
    uint32_t Y = cost.dim(1);

    // 32-bit bitmask for "taken" Ys.
    uint32_t mask = 0;

    for (size_t x = 0; x < X; ++x) {
        // Take the max of our chunk.
        hb_tiled_range(Y, [&](size_t ystart, size_t yend) {
            uint32_t argmax = UINT32_MAX;
            float valmax = 0;

            for (int y = ystart; y < yend; ++y) {
                if (mask & (1 << y)) {
                    continue;
                }

                float val = cost(x, y);
                if (val >= valmax) {
                    valmax = val;
                    argmax = y;
                }
            }

            // printf("%i for x=%i: range %i - %i: winner %i\n", __bsg_id, x, ystart, yend, argmax);
            buffer[__bsg_id] = argmax;
        });

        g_barrier.sync();

        // Top-level output reduction.
        if (__bsg_id == 0) {
            // printf("finalize for x=%i\n", x);

            uint32_t argmax = 0;
            float valmax = 0;
            for(size_t idx = 0; idx < bsg_tiles_X * bsg_tiles_Y; idx++) {
                size_t y = buffer[idx];
                if (y == UINT32_MAX) {
                    continue;
                }
                if (mask & (1 << y)) {
                    continue;
                }

                float val = cost(x, y);
                // printf("idx=%i, y=%i, val=%f\n", idx, y, val);
                if (val >= valmax) {
                    valmax = val;
                    argmax = y;
                }
            }
            // printf("valmax=%f, argmax=%i\n", valmax, argmax);

            // Record the overall winner.
            result(x) = argmax;
            mask |= 1 << argmax;
        }

        g_barrier.sync();
    }

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_auction, hb_tensor_t*, hb_tensor_t*)

}
