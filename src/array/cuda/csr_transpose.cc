/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/csr_transpose.cc
 * \brief CSR transpose (convert to CSC)
 */
#include <dgl/array.h>
#include "../../runtime/cuda/cuda_common.h"

namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {

template <DLDeviceType XPU, typename IdType>
CSRMatrix CSRTranspose(CSRMatrix csr) {
  CHECK(sizeof(IdType) == 4) << "CUDA CSR2CSC does not support int64.";
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, thr_entry->stream));

  NDArray indptr = csr.indptr, indices = csr.indices, data = csr.data;
  const int64_t nnz = indices->shape[0];
  const auto& ctx = indptr->ctx;
  const auto bits = indptr->dtype.bits;
  if (aten::IsNullArray(data))
    data = aten::Range(0, nnz, bits, ctx);
  const int32_t* indptr_ptr = static_cast<int32_t*>(indptr->data);
  const int32_t* indices_ptr = static_cast<int32_t*>(indices->data);
  const void* data_ptr = data->data;

  NDArray t_indptr = aten::NewIdArray(csr.num_cols + 1, ctx, bits);
  NDArray t_indices = aten::NewIdArray(nnz, ctx, bits);
  NDArray t_data = aten::NewIdArray(nnz, ctx, bits);
  int32_t* t_indptr_ptr = static_cast<int32_t*>(t_indptr->data);
  int32_t* t_indices_ptr = static_cast<int32_t*>(t_indices->data);
  void* t_data_ptr = t_data->data;

#if __CUDA_API_VERSION >= 10010
  auto device = runtime::DeviceAPI::Get(csr.indptr->ctx);
  // workspace
  size_t workspace_size;
  CUSPARSE_CALL(cusparseCsr2cscEx2_bufferSize(
      thr_entry->cusparse_handle,
      csr.num_rows, csr.num_cols, nnz,
      data_ptr, indptr_ptr, indices_ptr,
      t_data_ptr, t_indptr_ptr, t_indices_ptr,
      CUDA_R_32F,
      CUSPARSE_ACTION_NUMERIC,
      CUSPARSE_INDEX_BASE_ZERO,
      CUSPARSE_CSR2CSC_ALG1,  // see cusparse doc for reference
      &workspace_size));
  void* workspace = device->AllocWorkspace(ctx, workspace_size);
  CUSPARSE_CALL(cusparseCsr2cscEx2(
      thr_entry->cusparse_handle,
      csr.num_rows, csr.num_cols, nnz,
      data_ptr, indptr_ptr, indices_ptr,
      t_data_ptr, t_indptr_ptr, t_indices_ptr,
      CUSPARSE_ACTION_NUMERIC,
      CUSPARSE_INDEX_BASE_ZERO,
      CUSPARSE_CSR2CSC_ALG1,  // see cusparse doc for reference
      workspace));
  device->FreeWorkspace(ctx, workspace);
#else
  CUSPARSE_CALL(cusparseScsr2csc(
      thr_entry->cusparse_handle,
      csr.num_rows, csr.num_cols, nnz,
      static_cast<const float*>(data_ptr), indptr_ptr, indices_ptr,
      static_cast<float*>(t_data_ptr), t_indices_ptr, t_indptr_ptr,
      CUSPARSE_ACTION_NUMERIC,
      CUSPARSE_INDEX_BASE_ZERO));
#endif

  return CSRMatrix(csr.num_cols, csr.num_rows,
                   t_indptr, t_indices, t_data,
                   false);
}

template CSRMatrix CSRTranspose<kDLGPU, int32_t>(CSRMatrix csr);
template CSRMatrix CSRTranspose<kDLGPU, int64_t>(CSRMatrix csr);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
