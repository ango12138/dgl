/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/gather_mm.cu
 * \brief GatherMM C APIs and definitions.
 */
#include <dgl/array.h>
#include <algorithm>  // std::swap
#include "./utils.h"
#include "./functor.cuh"
#include "./atomic.cuh"

namespace dgl {
using namespace cuda;
namespace aten {

namespace {

/*! \brief Call cuBLAS geam API for transpose operation for float and double. */
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

/*! \brief Call cuBLAS GEMM API for dense matmul operation for float and double. */
template <typename DType>
cublasStatus_t cublasGemm(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k,
    const DType* alpha, const DType* A, int lda,
    const DType* B, int ldb, const DType* beta,
    DType* C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUBLAS_STATUS_EXECUTION_FAILED;
}

template <>
cublasStatus_t cublasGemm<float>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k,
    const float* alpha, const float* A, int lda,
    const float* B, int ldb, const float* beta,
    float* C, int ldc) {
  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda,
      B, ldb, beta, C, ldc);
}

template <>
cublasStatus_t cublasGemm<double>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k,
    const double* alpha, const double* A, int lda,
    const double* B, int ldb, const double* beta,
    double* C, int ldc) {
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda,
      B, ldb, beta, C, ldc);
}

/*
 * \brief Tranpose the input matrix.
 * \param row number of rows of input matrix.
 * \param col number of columns of input matrix.
 */
template <typename DType>
void _Transpose(cublasHandle_t handle,
                const DType* in, DType* out,
                int row, int col) {
  DType alpha = 1., beta = 0.;
  CUBLAS_CALL(Xgeam<DType>(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      row, col,
      &alpha, in, col,
      &beta, nullptr, row,
      out, row));
}

}  // namespace

/* \Note
   Idea 1: tranpose B and compute dot product of each row vector of A
   and column vector of B. Reuse in A.
   Idea 2: multiply 1 element of A with a row of B and compute partial
   result of the output
*/

/* Implementation of Idea 2
  \Note One warp is assigned to process one row of A. Each WARP sequentially
  multiplies one element of A and a row of B to compute partial result of the
  output. A is loaded in shared memory in a coalesced way. Output matrix is
  loaded in registers. B should get benefit from L2 cache.
*/

namespace cuda {

template <typename Idx, typename DType>
__global__ void gatherMMUnsortedEKernel(
    const DType* __restrict__ A,
    const DType* __restrict__ B,
    DType* __restrict__ C,
    const Idx* __restrict__ etype,
    int64_t num_rows,
    int64_t in_len, int64_t out_len) {
    unsigned int tId = threadIdx.x;
    unsigned int laneId = tId & 31;
    unsigned int gId = (blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int warpId = gId >> 5;
    unsigned int row = warpId;
    if (row < num_rows) {
        unsigned int local_row = row & 3;  // hardcoded for TB size 128 (4 warps)
        const int sh_h_tile = 64;
        __shared__ DType sh_H[4 * sh_h_tile];
        int h_tile = sh_h_tile;
        for (unsigned int k_start = 0; k_start < in_len; k_start += 64) {
            if ((in_len - k_start) < h_tile) h_tile = in_len - k_start;
            /* Load A in shared mem in a coalesced way */
            for (unsigned int l = laneId; l < h_tile; l += 32)
                sh_H[local_row * sh_h_tile + l] = A[row * in_len + (k_start + l)];
            __syncwarp();

            int B_offset = etype[row] * in_len * out_len;  // assume all weights are of same dim
            for (unsigned int outloop = 0; outloop < out_len; outloop +=32) {
                DType out_reg = 0;  // thread private
                unsigned int l = laneId;
                if (l < out_len) {
                    /* iterate over elements of a row of A */
                    for (unsigned int i = 0; i < h_tile; i++) {
                        DType h_val =  sh_H[local_row * sh_h_tile + i];
                        /* iterate over elements of a row of B in parallel */
                        out_reg += h_val * B[B_offset + ((i + k_start) * out_len + (outloop + l))];
                    }
                    C[row * out_len + (outloop + l)] += out_reg;
                }
            }
        }
    }
}

template <typename Idx, typename DType>
__global__ void gatherMMUnsortedEKernel_scatter(
    const DType* __restrict__ A,
    const DType* __restrict__ B,
    DType* __restrict__ C,
    const Idx* __restrict__ etype,
    int64_t num_rows,
    int64_t in_len, int64_t out_len) {
    unsigned int tId = threadIdx.x;
    unsigned int laneId = tId & 31;
    unsigned int gId = (blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int warpId = gId >> 5;
    unsigned int row = warpId;
    if (row < num_rows) {
        unsigned int local_row = row & 3;  // hardcoded for TB size 128 (4 warps)
        const int sh_a_tile = 64;
        __shared__ DType sh_A[4 * sh_a_tile];
        int a_tile = sh_a_tile;
        for (unsigned int k_start = 0; k_start < in_len; k_start += 64) {
            if ((in_len - k_start) < a_tile) a_tile = in_len - k_start;
            /* Load A in shared mem in a coalesced way */
            for (unsigned int l = laneId; l < a_tile; l += 32)
                sh_A[local_row * sh_a_tile + l] = A[row * in_len + (k_start + l)];
            __syncwarp();

            int C_offset = etype[row] * in_len * out_len;
            for (unsigned int outloop = 0; outloop < out_len; outloop +=32) {
                DType out_reg = 0;  // thread private
                unsigned int l = laneId;
                if (l < out_len) {
                    DType b_val = B[row * out_len + (outloop + l)];
                    /* iterate over elements of a row of A */
                    for (unsigned int i = 0; i < a_tile; i++) {
                        DType a_val = sh_A[local_row * sh_a_tile + i];
                        Idx C_idx = C_offset + ((i + k_start) * out_len + (outloop + l));
                        atomicAdd(reinterpret_cast<float*>(&C[C_idx]),
                            static_cast<float>(a_val * b_val));
                    }
                }
            }
        }
    }
}


/* \brief Implementation of GatherMM operator for un-sorted input matrix A.
 * Each edge looks up the weight matrix according to it's edge type.
 */
template <int XPU, typename IdType, int bits>
void gatherMM(const NDArray A,
              const NDArray B,
              NDArray C,
              const NDArray idx_b, int num_rel,
              bool a_trans, bool b_trans) {
    SWITCH_BITS(bits, DType, {
        auto device = runtime::DeviceAPI::Get(A->ctx);
        auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        const DType *A_data = A.Ptr<DType>();
        const DType *B_data = B.Ptr<DType>();
        int out_len = B->shape[1];  // cols of B
        int in_len = A->shape[1];  // cols of A
        int64_t A_offset = 0, B_offset = 0, C_offset = 0;
        if (!thr_entry->cublas_handle)
            CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
        CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle,
            thr_entry->stream));
        DType* B_trans_data = nullptr;
        if (b_trans) {
            in_len = B->shape[0]/num_rel;
            B_trans_data = static_cast<DType*>(device->AllocWorkspace \
                (B->ctx, B->shape[0] * B->shape[1] * sizeof(DType)));
            // tranpose B per relation
            for (int rel = 0; rel < num_rel; ++rel) {
                _Transpose(thr_entry->cublas_handle, B_data + B_offset,
                    B_trans_data + B_offset, in_len, out_len);
                B_offset += in_len * out_len;
            }
            std::swap(in_len, out_len);
        }
        IdType tot_num_rows = A->shape[0];
        const int ntx = 128;
        const int warp_size = 32;
        const int nbx =  ((tot_num_rows * warp_size + ntx - 1) / ntx);
        const dim3 nblks(nbx);
        const dim3 nthrs(ntx);
        if (a_trans) {
            CUDA_KERNEL_CALL((gatherMMUnsortedEKernel_scatter<IdType, DType>),
                nblks, nthrs, 0, thr_entry->stream,
                static_cast<DType*>(A->data),
                static_cast<DType*>(B->data),
                static_cast<DType*>(C->data),
                static_cast<IdType*>(idx_b->data),
                tot_num_rows,
                in_len, out_len);
        } else {
            CUDA_KERNEL_CALL((gatherMMUnsortedEKernel<IdType, DType>),
                nblks, nthrs, 0, thr_entry->stream,
                static_cast<DType*>(A->data),
                (b_trans) ? B_trans_data : static_cast<DType*>(B->data),
                static_cast<DType*>(C->data),
                static_cast<IdType*>(idx_b->data),
                tot_num_rows,
                in_len, out_len);
        }
        if (b_trans)
            device->FreeWorkspace(B->ctx, B_trans_data);
    });
}

}  //namespace cuda

/* \brief Implementation of GatherMM operator where input matrix A is sorted
 * according to relation types. Each relation type calls cuBLAS GEMM operator
 * sequentially.
 */
template <int XPU, typename IdType, int bits>
void segment_mm(const NDArray A,
              const NDArray B,
              NDArray C,
              const NDArray seglen_A,
              bool a_trans, bool b_trans) {
    SWITCH_BITS(bits, DType, {
        auto device = runtime::DeviceAPI::Get(A->ctx);
        const DType *A_data = A.Ptr<DType>();
        const DType *B_data = B.Ptr<DType>();
        const IdType* seglen_A_data = seglen_A.Ptr<IdType>();
        DType *C_data = C.Ptr<DType>();
        int64_t A_offset = 0, B_offset = 0, C_offset = 0;
        int64_t m, n, k;
        int64_t num_rel = seglen_A.NumElements();
        DType alpha = 1., beta = 0.;

        auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        if (!thr_entry->cublas_handle)
            CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
        CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle,
            thr_entry->stream));

        for (int etype = 0; etype < num_rel; ++etype) {
            IdType B_dim1 = B->shape[0] / num_rel;
            assert((a_trans) ? seglen_A_data[etype] : A->shape[1] ==  \
                (b_trans) ? B->shape[1] : B_dim1);
            m = seglen_A_data[etype];  // rows of A
            n = B->shape[1];  // cols of B
            k = A->shape[1];  // cols of A == rows of B
            int ldb = n, lda = k, ldc = n;
            cublasOperation_t transB = CUBLAS_OP_N;
            cublasOperation_t transA = CUBLAS_OP_N;
            if (a_trans) {
                transA = CUBLAS_OP_T;
                ldb = n, lda = k, ldc = n;
                std::swap(m, k);
            }
            if (b_trans) {
                transB = CUBLAS_OP_T;
                k = B_dim1;
                ldb = n, lda = n, ldc = k;
                std::swap(n, k);
            }
            CUBLAS_CALL(cublasGemm<DType>(
                thr_entry->cublas_handle,
                transB,
                transA,
                n, m, k,
                &alpha,
                B_data + B_offset, ldb,
                A_data + A_offset, lda,
                &beta,
                C_data + C_offset, ldc));
            A_offset += m * k;
            B_offset += k * n;
            C_offset += m * n;
        }
    });
}

/* \brief Implementation of GatherMM operator where input matrix A is sorted
 * according to relation types. Each relation type calls cuBLAS GEMM operator
 * sequentially.
 */
template <int XPU, typename IdType, int bits>
void segment_mm_explicitTranspose(const NDArray A,
              const NDArray B,
              NDArray C,
              const NDArray seglen_A,
              bool a_trans, bool b_trans) {
    SWITCH_BITS(bits, DType, {
        auto device = runtime::DeviceAPI::Get(A->ctx);
        const DType *A_data = A.Ptr<DType>();
        const DType *B_data = B.Ptr<DType>();
        const IdType* seglen_A_data = seglen_A.Ptr<IdType>();
        DType *C_data = C.Ptr<DType>();
        int64_t A_offset = 0, B_offset = 0, C_offset = 0;
        int64_t m, n, k;
        int64_t num_rel = seglen_A.NumElements();
        DType alpha = 1., beta = 0.;

        auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        if (!thr_entry->cublas_handle)
            CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
        CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle,
            thr_entry->stream));

        for (int etype = 0; etype < num_rel; ++etype) {
            IdType B_dim1 = B->shape[0] / num_rel;
            assert((a_trans) ? seglen_A_data[etype] : A->shape[1] ==  \
                (b_trans) ? B->shape[1] : B_dim1);
            m = seglen_A_data[etype];  // rows of A
            n = B->shape[1];  // cols of B
            k = A->shape[1];  // cols of A == rows of B

            DType* A_trans_data = nullptr;
            DType* B_trans_data = nullptr;
            if (a_trans) {
                A_trans_data = static_cast<DType*>(device->AllocWorkspace \
                    (A->ctx, m * k * sizeof(DType)));
                _Transpose(thr_entry->cublas_handle, A_data + A_offset, A_trans_data, m, k);
            }
            if (b_trans) {
                IdType tmp_k = B_dim1;
                B_trans_data = static_cast<DType*>(device->AllocWorkspace \
                    (B->ctx, n * tmp_k * sizeof(DType)));
                _Transpose(thr_entry->cublas_handle, B_data + B_offset, B_trans_data, tmp_k, n);
            }
            if (a_trans || b_trans) {
                if (a_trans)
                    std::swap(m, k);
                if (b_trans)  {
                    k = B_dim1;
                    std::swap(n, k);
                }
            }
            int ldb = n, lda = k, ldc = n;
            CUBLAS_CALL(cublasGemm<DType>(
                thr_entry->cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n, m, k,
                &alpha,
                (b_trans) ? B_trans_data : B_data + B_offset, ldb,
                (a_trans) ? A_trans_data : A_data + A_offset, lda,
                &beta,
                C_data + C_offset, ldc));
            if (a_trans)
                device->FreeWorkspace(A->ctx, A_trans_data);
            if (b_trans)
                device->FreeWorkspace(B->ctx, B_trans_data);
            A_offset += m * k;
            B_offset += k * n;
            C_offset += m * n;
        }
    });
}

/*!
 * \brief Implementation of Gather_mm operator. The input matrix A is
 *        expected to be sorted according to relation type.
 * \param A The input dense matrix of dimension m x k
 * \param B The input dense matrix of dimension k x n
 * \param C The output dense matrix of dimension m x n
 * \param idx_a : The input vector to gather left hand operand on
 * \param idx_b : The input vector to gather right hand operand on
 * \param a_trans Matrix A to be transposed
 * \param b_trans Matrix B to be transposed
 */
template <int XPU, typename IdType, int bits>
void gatherMM(const NDArray A,
          const NDArray B,
          NDArray C,
          const int num_rel,
          const NDArray idx_a,
          const NDArray idx_b,
          bool a_trans, bool b_trans) {
    cuda::gatherMM<XPU, IdType, bits>(A, B, C, idx_b, num_rel, a_trans, b_trans);
}

/*!
 * \brief Implementation of Gather_mm operator. The input matrix A is
 *        expected to be sorted according to relation type.
 * \param A The input dense matrix of dimension m x k
 * \param B The input dense matrix of dimension k x n
 * \param C The output dense matrix of dimension m x n
 * \param seglen_A The input vector of size R. Each element
 *        is the length of segments of input ``A``
 * \param a_trans Matrix A to be transposed
 * \param b_trans Matrix B to be transposed
 */
template <int XPU, typename IdType, int bits>
void segmentMM(const NDArray A,
          const NDArray B,
          NDArray C,
          const NDArray seglen_A,
          bool a_trans, bool b_trans) {
    segment_mm<XPU, IdType, bits>(A, B, C, seglen_A, a_trans, b_trans);
}


template void gatherMM<kDLGPU, int32_t, 16>(
    const NDArray A, const NDArray B, NDArray C, const int num_rel,
    const NDArray idx_a, const NDArray idx_b, bool a_trans, bool b_trans);
template void gatherMM<kDLGPU, int64_t, 16>(
    const NDArray A, const NDArray B, NDArray C, const int num_rel,
    const NDArray idx_a, const NDArray idx_b, bool a_trans, bool b_trans);
template void gatherMM<kDLGPU, int32_t, 32>(
    const NDArray A, const NDArray B, NDArray C, const int num_rel,
    const NDArray idx_a, const NDArray idx_b, bool a_trans, bool b_trans);
template void gatherMM<kDLGPU, int64_t, 32>(
    const NDArray A, const NDArray B, NDArray C, const int num_rel,
    const NDArray idx_a, const NDArray idx_b, bool a_trans, bool b_trans);
template void gatherMM<kDLGPU, int32_t, 64>(
    const NDArray A, const NDArray B, NDArray C, const int num_rel,
    const NDArray idx_a, const NDArray idx_b, bool a_trans, bool b_trans);
template void gatherMM<kDLGPU, int64_t, 64>(
    const NDArray A, const NDArray B, NDArray C, const int num_rel,
    const NDArray idx_a, const NDArray idx_b, bool a_trans, bool b_trans);

template void segmentMM<kDLGPU, int32_t, 16>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);
template void segmentMM<kDLGPU, int64_t, 16>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);
template void segmentMM<kDLGPU, int32_t, 32>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);
template void segmentMM<kDLGPU, int64_t, 32>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);
template void segmentMM<kDLGPU, int32_t, 64>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);
template void segmentMM<kDLGPU, int64_t, 64>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);

}  // namespace aten
}  // namespace dgl
