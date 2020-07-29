#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/native/hammerblade/Offload.h>



namespace at {
namespace native {

using namespace at::sparse;

// IntTensor _to_csr_int( const IntTensor& rowIndices, int64_t dim, int64_t nnz);


// void inline dstmm_kernel(
//     Tensor& res, //destination
//     const Tensor& a, //dense
//     const IntTensor& b_csc,
//     const IntTensor& b_rows,
//     const Tensor& b_vals
//   ) {

//     auto a_nrows = res.size(0);
//     auto b_ncols = res.size(1);
//     auto nnz = b_vals.numel();

//     float sum;
//     for (int a_row = 0; a_row < a_nrows; a_row++){
//       for (int b_col = 0; b_col < b_ncols; b_col++){
//         sum = 0;
//         for (int b_row_idx = b_csc(b_col); b_row_idx < b_csc(b_col+1); b_row_idx++){
//           int b_row = b_rows(b_row_idx);
//           float b_val = b_vals(b_row_idx);
//           sum += b_val * a(a_row, b_row); 
//         }
//         res(a_row, b_col) = sum;
//       }
//     }

// }

Tensor dstmm_cpu(
  const Tensor& a_dense,
  const SparseTensor& bT_sparse) {
  // Tensor out_dense = at::zeros({a_dense.size(0), bT_sparse.size(0)}, {at::requires_grad().device(at::kCPU).dtype(at::kFloat)});

  // IntTensor indices = bT_sparse._indices();
  // IntTensor b_rowIndices = indices.select(0, 1);
  // IntTensor b_colIndices = indices.select(0, 0);
  // int64_t b_nnz = bT_sparse._nnz();
  // int64_t b_dim = bT_sparse.size(0);
  // IntTensor b_csc = _to_csr_int(b_colIndices, b_dim, b_nnz);

  // Tensor b_values = bT_sparse._values();

  // dstmm_kernel(out_dense, a_dense, b_csc, b_rowIndices, b_values);

  // return out_dense;
}


} // namespace native
} // namespace at