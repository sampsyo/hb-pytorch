#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor auction_hb(const Tensor& cost) {
  Tensor result = at::empty({cost.size(0)}, {at::device(at::kHAMMERBLADE).dtype(at::kFloat)});
  hb_offload_kernel(result, cost, "tensorlib_auction");
  return result;
}

}}
