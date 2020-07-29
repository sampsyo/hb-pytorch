#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {

using namespace at::sparse;

template <typename scalar_t>
void inline sddmm_kernel(
    const Tensor& a_sparse,
    const Tensor& b_dense,
    const Tensor& c_dense,
    Tensor& out_tensor) {

    auto a_indices = a_sparse._indices();

    int dot_len = b_dense.size(1);
    for (int k = 0; k < a_sparse._nnz(); k++) {
      int ai = a_indices(0, k);//0
      int aj = a_indices(1, k);  //1

      float dot_total = 0;
      for (int i = 0; i < dot_len; i++)
        dot_total += b_dense(ai,i) * c_dense(i,aj);
        
      out_tensor(ai,aj) = dot_total;
    }
}

Tensor sddmm_cpu(
  const SparseTensor& a_sparse,
  const Tensor& b_dense,
  const Tensor& c_dense) {
  Tensor out_dense = at::zeros(a_sparse.sizes(), {at::requires_grad().device(at::kCPU).dtype(at::kFloat)});

  sddmm_kernel<float>(a_sparse, b_dense, c_dense, out_dense);
  return out_dense;
}


} // namespace native
} // namespace at