/*!
 *  Copyright (c) 2020 by Contributors
 * \file kernel/cuda/sddmm.cu
 * \brief SDDMM C APIs and definitions.
 */
#include <dgl/array.h>
#include "./sddmm.cuh"
#include "./functor2.cuh"

namespace dgl {
namespace kernel {

#define SWITCH_OP(op, Op, ...)                                      \
  do {                                                              \
    if ((op) == "add") {                                            \
      typedef dgl::kernel::cuda::binary::Add<DType> Op;             \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "sub") {                                     \
      typedef dgl::kernel::cuda::binary::Sub<DType> Op;             \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "mul") {                                     \
      typedef dgl::kernel::cuda::binary::Mul<DType> Op;             \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "div") {                                     \
      typedef dgl::kernel::cuda::binary::Div<DType> Op;             \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "copy_u") {                                  \
      typedef dgl::kernel::cuda::binary::CopyU<DType> Op;           \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "copy_e") {                                  \
      typedef dgl::kernel::cuda::binary::CopyE<DType> Op;           \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "dot") {                                     \
      typedef dgl::kernel::cuda::binary::Dot<DType> Op;             \
      { __VA_ARGS__ }                                               \
    } else {                                                        \
      LOG(FATAL) << "Unsupported SpMM/SDDMM binary operator: " << op;     \
    }                                                               \
  } while (0)


template <int XPU, typename IdType, typename DType>
void SDDMMCsr(const std::string& op,
              const BcastOff& bcast,
              const aten::CSRMatrix& csr,
              NDArray ufeat,
              NDArray vfeat,
              NDArray out) {
  SWITCH_OP(op, Op, {
    cuda::SDDMMCsr<IdType, DType, Op>(bcast, csr, ufeat, vfeat, out);
  });
}

template <int XPU, typename IdType, typename DType>
void SDDMMCoo(const std::string& op,
              const BcastOff& bcast,
              const aten::COOMatrix& coo,
              NDArray ufeat,
              NDArray vfeat,
              NDArray out) {
  SWITCH_OP(op, Op, {
    cuda::SDDMMCoo<IdType, DType, Op>(bcast, coo, ufeat, vfeat, out);
  });
}

template void SDDMMCsr<kDLGPU, int32_t, float>(
    const std::string& op, const BcastOff& bcast, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray vfeat, NDArray out);
template void SDDMMCsr<kDLGPU, int64_t, float>(
    const std::string& op, const BcastOff& bcast, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray vfeat, NDArray out);
template void SDDMMCsr<kDLGPU, int32_t, double>(
    const std::string& op, const BcastOff& bcast, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray vfeat, NDArray out);
template void SDDMMCsr<kDLGPU, int64_t, double>(
    const std::string& op, const BcastOff& bcast, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray vfeat, NDArray out);

template void SDDMMCoo<kDLGPU, int32_t, float>(
    const std::string& op, const BcastOff& bcast, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray vfeat, NDArray out);
template void SDDMMCoo<kDLGPU, int64_t, float>(
    const std::string& op, const BcastOff& bcast, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray vfeat, NDArray out);
template void SDDMMCoo<kDLGPU, int32_t, double>(
    const std::string& op, const BcastOff& bcast, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray vfeat, NDArray out);
template void SDDMMCoo<kDLGPU, int64_t, double>(
    const std::string& op, const BcastOff& bcast, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray vfeat, NDArray out);

}  // namespace kernel
}  // namespace dgl
