/**
 *  Copyright (c) 2023 by Contributors
 * @file matrix_ops.cc
 * @brief DGL C++ matrix operators.
 */
#include <sparse/matrix_ops.h>
#include <torch/script.h>

namespace dgl {
namespace sparse {

/**
 * @brief Compute the intersection of two COO matrices. Return the intersection
 * COO matrix, and the indices of the intersection in the left-hand-side and
 * right-hand-side COO matrices.
 *
 * @param lhs The left-hand-side COO matrix.
 * @param rhs The right-hand-side COO matrix.
 *
 * @return A tuple of COO matrix, lhs indices, and rhs indices.
 */
std::tuple<std::shared_ptr<COO>, torch::Tensor, torch::Tensor> COOIntersection(
    const std::shared_ptr<COO>& lhs, const std::shared_ptr<COO>& rhs) {
  // 1. Encode the two COO matrices into arrays of integers.
  auto lhs_arr =
      lhs->indices.index({0}) * lhs->num_cols + lhs->indices.index({1});
  auto rhs_arr =
      rhs->indices.index({0}) * rhs->num_cols + rhs->indices.index({1});
  // 2. Concatenate the two arrays.
  auto arr = torch::cat({lhs_arr, rhs_arr});
  // 3. Unique the concatenated array.
  torch::Tensor unique, inverse, counts;
  std::tie(unique, inverse, counts) =
      torch::unique_dim(arr, 0, false, true, true);
  // 4. Find the indices of the counts greater than 1 in the unique array.
  auto mask = counts > 1;
  // 5. Map the inverse array to the original array to generate indices.
  auto lhs_inverse = inverse.slice(0, 0, lhs_arr.numel());
  auto rhs_inverse = inverse.slice(0, lhs_arr.numel(), arr.numel());
  auto map_to_original = torch::empty_like(unique);
  map_to_original.index_put_(
      {lhs_inverse},
      torch::arange(lhs_inverse.numel(), map_to_original.options()));
  auto lhs_indices = map_to_original.index({mask});
  map_to_original.index_put_(
      {rhs_inverse},
      torch::arange(rhs_inverse.numel(), map_to_original.options()));
  auto rhs_indices = map_to_original.index({mask});
  // 6. Decode the indices to get the intersection COO matrix.
  auto ret_arr = unique.index({mask});
  auto ret_indices = torch::stack(
      {ret_arr.floor_divide(lhs->num_cols), ret_arr % lhs->num_cols}, 0);
  auto ret_coo = std::make_shared<COO>(
      COO{lhs->num_rows, lhs->num_cols, ret_indices, false, false});
  return {ret_coo, lhs_indices, rhs_indices};
}

std::tuple<torch::Tensor, torch::Tensor> CompactIndices(
    const torch::Tensor& row,
    const torch::optional<torch::Tensor>& leading_indices) {
  torch::Tensor sort_row, sort_idx;
  std::tie(sort_row, sort_idx) = row.sort(-1);
  torch::Tensor rev_sort_idx = torch::empty_like(sort_idx);
  rev_sort_idx.index_put_({sort_idx}, torch::arange(0, sort_idx.numel()));

  torch::Tensor uniqued, uniq_idx;
  int64_t n_leading_indices = 0;
  if (leading_indices.has_value()) {
    n_leading_indices = leading_indices.value().numel();
    std::tie(uniqued, uniq_idx) = torch::_unique(
        torch::cat({leading_indices.value(), sort_row}), false, true);
  } else {
    std::tie(uniqued, uniq_idx) = torch::_unique(sort_row, false, true);
  }

  auto new_row =
      torch::arange(uniqued.numel() - 1, -1, -1)
          .index_select(
              0, uniq_idx.slice(
                     0, n_leading_indices, n_leading_indices + row.size(-1)))
          .index_select(0, rev_sort_idx);
  return {new_row, uniqued};
}

std::tuple<c10::intrusive_ptr<SparseMatrix>, torch::optional<torch::Tensor>>
CompactCOO(
    const c10::intrusive_ptr<SparseMatrix>& mat, int64_t dim,
    torch::optional<torch::Tensor> leading_indices) {
  torch::Tensor row, col;
  auto coo = mat->COOTensors();
  if (dim == 0)
    std::tie(row, col) = coo;
  else
    std::tie(col, row) = coo;

  torch::Tensor new_row, uniqued;
  std::tie(new_row, uniqued) = CompactIndices(row, leading_indices);

  if (dim == 0) {
    auto ret = SparseMatrix::FromCOO(
        torch::stack({new_row, col}, 0), mat->value(),
        std::vector<int64_t>{uniqued.numel(), mat->shape()[1]});
    auto ret_idx = torch::optional<torch::Tensor>(uniqued.flip(-1));
    return {ret, ret_idx};
  } else {
    auto ret = SparseMatrix::FromCOO(
        torch::stack({col, new_row}, 0), mat->value(),
        std::vector<int64_t>{mat->shape()[0], uniqued.numel()});
    auto ret_idx = torch::optional<torch::Tensor>(uniqued.flip(-1));
    return {ret, ret_idx};
  }
}

std::tuple<c10::intrusive_ptr<SparseMatrix>, torch::optional<torch::Tensor>>
CompactCSC(
    const c10::intrusive_ptr<SparseMatrix>& mat, int64_t dim,
    torch::optional<torch::Tensor> leading_indices) {
  std::shared_ptr<dgl::sparse::CSR> csr;
  if (dim == 0)
    csr = mat->CSCPtr();
  else
    csr = mat->CSRPtr();

  torch::Tensor new_indices, uniqued;
  std::tie(new_indices, uniqued) =
      CompactIndices(csr->indices, leading_indices);

  if (dim == 0) {
    auto ret = SparseMatrix::FromCSC(
        csr->indptr, new_indices, mat->value(),
        std::vector<int64_t>{uniqued.numel(), mat->shape()[1]});
    auto ret_idx = torch::optional<torch::Tensor>(uniqued.flip(-1));
    return {ret, ret_idx};
  } else {
    auto ret = SparseMatrix::FromCSR(
        csr->indptr, new_indices, mat->value(),
        std::vector<int64_t>{mat->shape()[0], uniqued.numel()});
    auto ret_idx = torch::optional<torch::Tensor>(uniqued.flip(-1));
    return {ret, ret_idx};
  }
}

std::tuple<c10::intrusive_ptr<SparseMatrix>, torch::optional<torch::Tensor>>
Compact(
    const c10::intrusive_ptr<SparseMatrix>& mat, int64_t dim,
    torch::optional<torch::Tensor> leading_indices) {
  if (dim == 0) {
    if (mat->HasCSC())
      return CompactCSC(mat, dim, leading_indices);
    else
      return CompactCOO(mat, dim, leading_indices);
  } else {
    if (mat->HasCSR())
      return CompactCSC(mat, dim, leading_indices);
    else
      return CompactCOO(mat, dim, leading_indices);
  }
}

}  // namespace sparse
}  // namespace dgl
