#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {

using namespace at::sparse;


Tensor sddmm_cpu(
  const SparseTensor& a_sparse_tensor,
  const Tensor& b_dense_tensor,
  const Tensor& c_dense_tensor) {
  Tensor out_tensor = at::zeros(a_sparse_tensor.sizes(), {at::requires_grad().device(at::kCPU).dtype(at::kFloat)});
  AT_DISPATCH_ALL_TYPES(b_dense_tensor.scalar_type(), "sddmm_cpu", [&] {

    auto a_indices = a_sparse_tensor._indices().accessor<int64_t, 2>();
    auto b_dense = b_dense_tensor.accessor<scalar_t, 2>();
    auto c_dense = c_dense_tensor.accessor<scalar_t, 2>();
    auto out = out_tensor.accessor<scalar_t, 2>();

    int dot_len = b_dense.size(1);
    for (int k = 0; k < a_sparse_tensor._nnz(); k++) {
      int ai = a_indices[0][k]; //0
      int aj = a_indices[1][k];  //1

      float dot_total = 0;
      for (int i = 0; i < dot_len; i++)
        dot_total += b_dense[ai][i] * c_dense[i][aj];
        
      out[ai][aj] = dot_total;
    }
  
  });
  return out_tensor;
}

} // namespace native
} // namespace at