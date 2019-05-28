/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cpu/binary_reduce_none.cc
 * \brief CPU kernels for binary reduce none
 */
#include "./binary_reduce_impl.h"
#include "./backward_binary_reduce_impl.h"

namespace dgl {
namespace kernel {

#define REDUCER ReduceNone
#define XPU kDLCPU

EVAL(GEN_DTYPE, GEN_TARGET, GEN_BINARY_OP, GEN_DEFINE);
EVAL(GEN_BACKWARD_MODE, GEN_DTYPE, GEN_TARGET, GEN_BINARY_OP, GEN_BACKWARD_DEFINE);

}  // namespace kernel
}  // namespace dgl
