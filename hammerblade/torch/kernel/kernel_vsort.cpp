  
//====================================================================
// Vector merge sort kernel
// 06/04/2020 Janice Wei (cw655@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

// Helper function to perform Merge Sort
static void merge(
  HBTensor<float>* tmp_p,
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
      (*tmp_p)(i) = (*vec1)(idx1);
      idx1 ++;
    }
    else if ( idx1 == end1 ) {
      (*tmp_p)(i) = (*vec0)(idx0);
      idx0 ++;
    }
    else if ((*vec0)(idx0) <= (*vec1)(idx1)) {
      (*tmp_p)(i)= (*vec0)(idx0);
      idx0 ++;
    }
    else {
      (*tmp_p)(i) = (*vec1)(idx1);
      idx1 ++;
    }
  }
}

// Recursive merge helper function
static void msort_op_h(HBTensor<float>* result_p,
  HBTensor<float>* tmp_p,
  HBTensor<float>* vec,
  int32_t begin,
  int32_t end) {
  int32_t size = end - begin;
  if (size <= 1) {
    (*result_p)(begin) = (*vec)(begin);
    return;
  }
  int32_t mid =int32_t( (begin + end) / 2 );
  msort_op_h(result_p, tmp_p, vec, begin, mid);
  msort_op_h(result_p, tmp_p, vec, mid, end);
  merge(tmp_p, result_p, begin, mid, result_p, mid, end);
  int32_t x = 0;
  for (size_t i = begin; i < end; i++) {
    (*result_p)(i) = (*tmp_p)(x);
    x++;
  }
}

extern "C" {

  __attribute__ ((noinline))  int tensorlib_vsort(
          hb_tensor_t* result_p,
          hb_tensor_t* tmp_p,
          hb_tensor_t* self_p) {

    // Convert low level pointers to Tensor objects
    HBTensor<float> result(result_p);
    HBTensor<float> self(self_p);
    HBTensor<float> tmp(tmp_p);
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    
    // Use a single tile only
    if(__bsg_id == 0) {
      msort_op_h(&result, &tmp, &self, 0, self.numel());
    }
  //quicksort(&result, 0, self.numel()-1);
    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    // Sync
    g_barrier.sync();
    return 0;
  }

  // Register the HB kernel with emulation layer
  HB_EMUL_REG_KERNEL(tensorlib_vsort, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}
