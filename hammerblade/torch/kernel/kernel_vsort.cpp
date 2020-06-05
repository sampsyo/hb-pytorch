  
//====================================================================
// Vector merge sort kernel
// 06/04/2020 Janice Wei (cw655@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

// Helper function to perform Merge Sort
void merge(
	HBTensor<float>* result_p, 
	HBTensor<float>* vec0,
	int32_t begin0,
	int32_t end0,
	HBTensor<float>* vec1,
	int32_t begin1,
	int32_t end1) {
	int32_t size = (end0 - begin0) + (end1 - begin1);
	int32_t idx0 = begin0;
	int32_t idx1 = begin1;
	for (size_t i = 0; i < size; i++) {
		if ( idx0 == end0 ) {
			(*result_p)(i) = (*vec1)(idx1);
			idx1 ++;
		}
		else if ( idx1 == end1 ) {
			(*result_p)(i) = (*vec0)(idx0);
			idx0 ++;
		}
		else if ((*vec0)(idx0) <= (*vec1)(idx1)) {
			(*result_p)(i) = (*vec0)(idx0);
			idx0 ++;
		}
		else {
			(*result_p)(i) = (*vec1)(idx1);
			idx1 ++;
		}
	}
}

// Recursive merge helper function
void msort_op_h(HBTensor<float>* result_p,
	HBTensor<float>* vec,
	int32_t begin,
	int32_t end) {
	int32_t size = end - begin;
	if (size <= 1) {
		return;
	}
	int32_t mid = (begin + end) / 2;
	msort_op_h(result_p, vec, begin, mid);
	msort_op_h(result_p, vec, mid, end);
	merge(result_p, vec, begin, mid, vec, mid, end);
}

extern "C" {

  __attribute__ ((noinline))  int tensorlib_vsort(
          hb_tensor_t* result_p,
          hb_tensor_t* self_p) {

    // Convert low level pointers to Tensor objects
    HBTensor<float> result(result_p);
    HBTensor<float> self(self_p);

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

	msort_op_h(&result, &self, 0, self.numel());

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    // Sync
    g_barrier.sync();
    return 0;
  }

  // Register the HB kernel with emulation layer
  HB_EMUL_REG_KERNEL(tensorlib_vsort, hb_tensor_t*, hb_tensor_t*)

}
