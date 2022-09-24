/*!
 *  Copyright (c) 2022 by Contributors
 * \file array/cpu/coo_remove_if.cc
 * \brief COO matrix remove entries CPU implementation
 */
#include <dgl/array.h>
#include <utility>
#include <vector>
#include "array_utils.h"

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType, typename DType>
COOMatrix COORemoveIf(COOMatrix coo, NDArray values, DType criteria) {
  const IdType* row = coo.row.Ptr<IdType>();
  const IdType* col = coo.col.Ptr<IdType>();
  const IdType* data = COOHasData(coo) ? coo.data.Ptr<IdType>() : nullptr;
  const DType* val = values.Ptr<DType>();
  const auto idtype = coo.row->dtype;
  const auto ctx = coo.row->ctx;
  const int64_t nnz = coo.row->shape[0];
  IdArray new_row = IdArray::Empty({nnz}, idtype, ctx);
  IdArray new_col = IdArray::Empty({nnz}, idtype, ctx);
  IdArray new_eid = IdArray::Empty({nnz}, idtype, ctx);
  IdType* new_row_data = new_row.Ptr<IdType>();
  IdType* new_col_data = new_col.Ptr<IdType>();
  IdType* new_eid_data = new_eid.Ptr<IdType>();

  int64_t j = 0;
  for (int64_t i = 0; i < nnz; ++i) {
    if (val[i] != criteria) {
      new_row_data[j] = row[i];
      new_col_data[j] = col[i];
      new_eid_data[j] = data ? data[j] : j;
      ++j;
    }
  }
  return COOMatrix(
      coo.num_rows,
      coo.num_cols,
      new_row.CreateView({j}, idtype, 0),
      new_col.CreateView({j}, idtype, 0),
      new_eid.CreateView({j}, idtype, 0));
}

template COOMatrix COORemoveIf<kDGLCPU, int32_t, int8_t>(COOMatrix, NDArray, int8_t);
template COOMatrix COORemoveIf<kDGLCPU, int32_t, uint8_t>(COOMatrix, NDArray, uint8_t);
template COOMatrix COORemoveIf<kDGLCPU, int32_t, float>(COOMatrix, NDArray, float);
template COOMatrix COORemoveIf<kDGLCPU, int32_t, double>(COOMatrix, NDArray, double);
template COOMatrix COORemoveIf<kDGLCPU, int64_t, int8_t>(COOMatrix, NDArray, int8_t);
template COOMatrix COORemoveIf<kDGLCPU, int64_t, uint8_t>(COOMatrix, NDArray, uint8_t);
template COOMatrix COORemoveIf<kDGLCPU, int64_t, float>(COOMatrix, NDArray, float);
template COOMatrix COORemoveIf<kDGLCPU, int64_t, double>(COOMatrix, NDArray, double);

};  // namespace impl
};  // namespace aten
};  // namespace dgl
