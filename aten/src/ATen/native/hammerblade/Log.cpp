#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {

static void log_kernel_hb(TensorIterator& iter) {
    AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "log_hb", [&]() {
        offload_op_unary(iter, "tensorlib_log");
        });
}

} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(log_stub, &log_kernel_hb);

}} // namespace at::native
