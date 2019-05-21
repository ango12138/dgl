/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/binary_reduce_sum.cu
 * \brief CUDA kernels for binary reduce sum
 */
#include <dgl/runtime/device_api.h>

#include "../../runtime/cuda/cuda_common.h"
#include "./binary_reduce_impl.cuh"
#include "./backward_binary_reduce_impl.cuh"
#include "../utils.h"

using minigun::Csr;
using minigun::advance::RuntimeConfig;

namespace dgl {
namespace kernel {
namespace cuda {
// specialization for cusparse

template <typename DType>
cusparseStatus_t Xcsrmm2(cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const DType* alpha, const cusparseMatDescr_t descrA,
    const DType* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const DType* B, int ldb, const DType* beta, DType* C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUSPARSE_STATUS_EXECUTION_FAILED;
}

template <>
cusparseStatus_t Xcsrmm2<float>(cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const float* alpha, const cusparseMatDescr_t descrA,
    const float* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const float* B, int ldb, const float* beta, float* C, int ldc) {
  return cusparseScsrmm2(handle, transA, transB, m, n, k, nnz,
      alpha, descrA, csrValA, csrRowPtrA, csrColIndA,
      B, ldb, beta, C, ldc);
}

template <>
cusparseStatus_t Xcsrmm2<double>(cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const double* alpha, const cusparseMatDescr_t descrA,
    const double* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const double* B, int ldb, const double* beta, double* C, int ldc) {
  return cusparseDcsrmm2(handle, transA, transB, m, n, k, nnz,
      alpha, descrA, csrValA, csrRowPtrA, csrColIndA,
      B, ldb, beta, C, ldc);
}

template <typename DType>
cublasStatus_t Xgeam(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const DType* alpha, const DType* A, int lda,
    const DType* beta, const DType* B, int ldb,
    DType* C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUBLAS_STATUS_EXECUTION_FAILED;
}

template <>
cublasStatus_t Xgeam<float>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const float* alpha, const float* A, int lda,
    const float* beta, const float* B, int ldb,
    float* C, int ldc) {
  return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda,
      beta, B, ldb, C, ldc);
}

template <>
cublasStatus_t Xgeam<double>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const double* alpha, const double* A, int lda,
    const double* beta, const double* B, int ldb,
    double* C, int ldc) {
  return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda,
      beta, B, ldb, C, ldc);
}

template <typename DType>
void CusparseCsrmm2(const RuntimeConfig& rtcfg, const Csr& csr,
    const DType* B_data, DType* C_data, int x_length) {
  const int m = csr.row_offsets.length - 1;
  const int k = csr.row_offsets.length - 1;
  const int n = x_length;
  const int nnz = csr.column_indices.length;
  const DType alpha = 1.0;
  const DType beta = 0.0;
  // device
  auto device = runtime::DeviceAPI::Get(rtcfg.ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, rtcfg.stream));
  // allocate matrix for temporary transposed output
  DType* trans_out = static_cast<DType*>(device->AllocWorkspace(rtcfg.ctx, k * n * sizeof(DType)));
  // all one data array
  DType* valptr = static_cast<DType*>(device->AllocWorkspace(rtcfg.ctx, nnz * sizeof(DType)));
  utils::Fill<kDLGPU>(rtcfg.ctx, valptr, nnz, static_cast<DType>(1.));
  cusparseMatDescr_t descr;
  CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
  CUSPARSE_CALL(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_CALL(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
  CUSPARSE_CALL(Xcsrmm2<DType>(
      thr_entry->cusparse_handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE,
      m, n, k, nnz, &alpha,
      descr, valptr, csr.row_offsets.data, csr.column_indices.data,
      B_data, n, &beta, trans_out, m));
  device->FreeWorkspace(rtcfg.ctx, valptr);
  // transpose the output matrix
  if (!thr_entry->cublas_handle) {
    CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
  }
  CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle, rtcfg.stream));
  CUBLAS_CALL(Xgeam<DType>(
      thr_entry->cublas_handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n, m,
      &alpha, trans_out, m,
      &beta, nullptr, n,
      C_data, n));
  device->FreeWorkspace(rtcfg.ctx, trans_out);
}

// forward

template <typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void FallbackCallBinaryReduce(
    const minigun::advance::RuntimeConfig& rtcfg,
    const minigun::Csr& csr,
    const minigun::Csr& rev_csr,
    GData<DType>* gdata) {
  using minigun::IntArray1D;
  typedef FunctorsTempl<DType, LeftSelector,
                        RightSelector, BinaryOp, Reducer>
          Functors;
  typedef BinaryReduce<DType, Functors> UDF;
  // TODO(minjie): allocator
  minigun::advance::Advance<kDLGPU, AdvanceConfig, GData<DType>, UDF>(
        rtcfg, csr, gdata, IntArray1D());
}

template <int Mode, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void FallbackCallBackwardBinaryReduce(
    const minigun::advance::RuntimeConfig& rtcfg,
    const minigun::Csr& csr, const minigun::Csr& rev_csr,
    BackwardGData<DType>* gdata) {
  using minigun::IntArray1D;
  typedef BackwardFunctorsTempl<DType, LeftSelector,
                        RightSelector, BinaryOp, Reducer>
          Functors;
  typedef BackwardBinaryReduce<Mode, DType, Functors> UDF;
  // TODO(minjie): allocator
  minigun::advance::Advance<kDLGPU, AdvanceConfig, BackwardGData<DType>, UDF>(
        rtcfg, rev_csr, gdata, IntArray1D());
}
}  // namespace cuda

template <>
void CallBinaryReduce<kDLGPU, float, SelectSrc, SelectEdge,
                      BinaryUseLhs<float>, ReduceSum<kDLGPU, float>>(
    const RuntimeConfig& rtcfg,
    const Csr& csr, const Csr& rev_csr,
    GData<float>* gdata) {
  if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
    cuda::FallbackCallBinaryReduce<float, SelectSrc, SelectEdge,
      BinaryUseLhs<float>, ReduceSum<kDLGPU, float>>(rtcfg, csr, rev_csr, gdata);
  } else {
    cuda::CusparseCsrmm2(rtcfg, rev_csr, gdata->lhs_data, gdata->out_data, gdata->x_length);
  }
}

template <>
void CallBinaryReduce<kDLGPU, double, SelectSrc, SelectEdge,
                      BinaryUseLhs<double>, ReduceSum<kDLGPU, double>>(
    const RuntimeConfig& rtcfg,
    const Csr& csr, const Csr& rev_csr,
    GData<double>* gdata) {
  if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
    cuda::FallbackCallBinaryReduce<double, SelectSrc, SelectEdge,
      BinaryUseLhs<double>, ReduceSum<kDLGPU, double>>(rtcfg, csr, rev_csr, gdata);
  } else {
    cuda::CusparseCsrmm2(rtcfg, rev_csr, gdata->lhs_data, gdata->out_data, gdata->x_length);
  }
}

// backward

template <>
void CallBackwardBinaryReduce<kDLGPU, binary_op::kGradLhs, float,
                              SelectDst, SelectEdge,
                              BinaryUseLhs<float>, ReduceSum<kDLGPU, float>>(
    const RuntimeConfig& rtcfg,
    const Csr& csr, const Csr& rev_csr,
    BackwardGData<float>* gdata) {
  if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
    cuda::FallbackCallBackwardBinaryReduce<binary_op::kGradLhs, float, SelectDst, SelectEdge,
      BinaryUseLhs<float>, ReduceSum<kDLGPU, float>>(rtcfg, csr, rev_csr, gdata);
  } else {
    cuda::CusparseCsrmm2(rtcfg, csr, gdata->grad_out_data, gdata->grad_lhs_data, gdata->x_length);
  }
}

template <>
void CallBackwardBinaryReduce<kDLGPU, binary_op::kGradLhs, double,
                              SelectDst, SelectEdge,
                              BinaryUseLhs<double>, ReduceSum<kDLGPU, double>>(
    const RuntimeConfig& rtcfg,
    const Csr& csr, const Csr& rev_csr,
    BackwardGData<double>* gdata) {
  if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
    cuda::FallbackCallBackwardBinaryReduce<binary_op::kGradLhs, double, SelectDst, SelectEdge,
      BinaryUseLhs<double>, ReduceSum<kDLGPU, double>>(rtcfg, csr, rev_csr, gdata);
  } else {
    cuda::CusparseCsrmm2(rtcfg, csr, gdata->grad_out_data, gdata->grad_lhs_data, gdata->x_length);
  }
}

// generate definitions

#define REDUCER ReduceSum
#define XPU kDLGPU

EVAL(GEN_DTYPE, GEN_TARGET, GEN_BINARY_OP, GEN_DEFINE)
EVAL(GEN_BACKWARD_MODE, GEN_DTYPE, GEN_TARGET, GEN_BINARY_OP, GEN_BACKWARD_DEFINE)

}  // namespace kernel
}  // namespace dgl
