#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor& auction_hb(const Tensor& cost) {
  Tensor result = at::empty(cost.sizes(), cost.options());
  auto iter = TensorIterator::binary_op(result, cost,
    /*check_mem_overlap=*/true);
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "auction", [&]() {
      offload_op_binary(iter, "tensorlib_auction");
      });
  return result;
}

}}
