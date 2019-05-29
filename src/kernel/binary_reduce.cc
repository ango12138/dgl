/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/binary_reduce.cc
 * \brief Binary reduce C APIs and definitions.
 */
#include "./binary_reduce.h"
#include "./common.h"
#include "./binary_reduce_impl_decl.h"
#include "./utils.h"
#include "../c_api_common.h"

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLArgValue;
using dgl::runtime::DGLRetValue;
using dgl::runtime::PackedFunc;
using dgl::runtime::NDArray;

namespace dgl {
namespace kernel {
namespace {

std::string ShapeString(NDArray nd) {
  std::ostringstream oss;
  oss << "(";
  for (int i = 1; i < nd->ndim; ++i) {
    oss << nd->shape[i];
    if (i != nd->ndim - 1) {
      oss << ",";
    }
  }
  oss << ")";
  return oss.str();
}

std::vector<int64_t> ComputeStride(const std::vector<int64_t>& shape) {
  std::vector<int64_t> ret(shape.size(), 1);
  for (int i = shape.size() - 2; i >= 0; --i) {
    ret[i] = ret[i+1] * shape[i+1];
  }
  return ret;
}

bool IsValidBinaryOpShape(NDArray lhs, NDArray rhs) {
  if (lhs->ndim != rhs->ndim) {
    return false;
  }
  for (int i = 1; i < lhs->ndim; ++i) {
    if (lhs->shape[i] != rhs->shape[i]) {
      return false;
    }
  }
  return true;
}

bool HasBcast(NDArray lhs, NDArray rhs) {
  if (lhs->ndim != rhs->ndim) {
    return true;
  }
  for (int i = 1; i < lhs->ndim; ++i) {
    if (lhs->shape[i] != rhs->shape[i]) {
      return true;
    }
  }
  return false;
}

BcastInfo CalcBcastInfo(NDArray lhs, NDArray rhs) {
  BcastInfo ret;
  const int max_ndim = std::max(lhs->ndim, rhs->ndim) - 1;
  int64_t accum = 0;
  for (int j = 0; j < max_ndim; ++j) {
    const int dl = (lhs->ndim - 1 - j < 1)? 1 : lhs->shape[lhs->ndim - 1 - j];
    const int dr = (rhs->ndim - 1 - j < 1)? 1 : rhs->shape[rhs->ndim - 1 - j];
    if (dl != dr) {
      if (dl != 1 && dr != 1) {
        LOG(FATAL) << "Invalid broadcasting between feature shapes "
          << ShapeString(lhs) << " and " << ShapeString(rhs);
      }
      if (accum != 0) {
        ret.lhs_shape.push_back(accum);
        ret.rhs_shape.push_back(accum);
        ret.out_shape.push_back(accum);
        accum = 0;
      }
      ret.lhs_shape.push_back(dl);
      ret.rhs_shape.push_back(dr);
      ret.out_shape.push_back(std::max(dl, dr));
    } else {
      if (accum == 0) {
        accum = dl;
      } else {
        accum *= dl;
      }
    }
    ret.real_out_shape.push_back(std::max(dl, dr));
  }
  if (accum != 0) {
    ret.lhs_shape.push_back(accum);
    ret.rhs_shape.push_back(accum);
    ret.out_shape.push_back(accum);
    accum = 0;
  }
  std::reverse(ret.real_out_shape.begin(), ret.real_out_shape.end());
  std::reverse(ret.lhs_shape.begin(), ret.lhs_shape.end());
  std::reverse(ret.rhs_shape.begin(), ret.rhs_shape.end());
  std::reverse(ret.out_shape.begin(), ret.out_shape.end());
  // stride
  ret.lhs_stride = ComputeStride(ret.lhs_shape);
  ret.rhs_stride = ComputeStride(ret.rhs_shape);
  ret.out_stride = ComputeStride(ret.out_shape);
  return ret;
}

std::string IdArrayToStr(IdArray arr) {
  int64_t len = arr->shape[0];
  std::ostringstream oss;
  oss << "[";
  if (arr->dtype.bits == 32) {
    int32_t* data = static_cast<int32_t*>(arr->data);
    for (int64_t i = 0; i < len; ++i) {
      oss << data[i] << " ";
    }
  } else {
    int64_t* data = static_cast<int64_t*>(arr->data);
    for (int64_t i = 0; i < len; ++i) {
      oss << data[i] << " ";
    }
  }
  oss << "]";
  return oss.str();
}

}  // namespace


std::vector<int64_t> InferBinaryFeatureShape(
    NDArray lhs,
    NDArray rhs) {
  return CalcBcastInfo(lhs, rhs).real_out_shape;
}

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelInferBinaryFeatureShape")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NDArray lhs = args[0];
    NDArray rhs = args[1];
    const auto& shape = InferBinaryFeatureShape(lhs, rhs);
    const int64_t len = shape.size();
    NDArray ret = NDArray::Empty(
        {len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
    int64_t* ret_data = static_cast<int64_t*>(ret->data);
    std::copy(shape.begin(), shape.end(), ret_data);
    *rv = ret;
  });

void BinaryOpReduce(
    const std::string& reducer,
    const std::string& op,
    const ImmutableGraph* graph,
    binary_op::Target lhs, binary_op::Target rhs,
    NDArray lhs_data, NDArray rhs_data,
    NDArray out_data,
    NDArray lhs_mapping, NDArray rhs_mapping,
    NDArray out_mapping) {
  // sanity check
  const auto& ctx = graph->Context();
  CHECK_EQ(ctx, lhs_data->ctx) << "Expected device context " << ctx
    << ". But got " << lhs_data->ctx << " for lhs_data.";
  CHECK_EQ(ctx, rhs_data->ctx) << "Expected device context " << ctx
    << ". But got " << rhs_data->ctx << " for rhs_data.";
  CHECK_EQ(ctx, out_data->ctx) << "Expected device context " << ctx
    << ". But got " << out_data->ctx << " for out_data.";
  if (!utils::IsNoneArray(lhs_mapping)) {
    CHECK_EQ(ctx, lhs_mapping->ctx) << "Expected device context " << ctx
      << ". But got " << lhs_mapping->ctx << " for rhs_data.";
  }
  if (!utils::IsNoneArray(rhs_mapping)) {
    CHECK_EQ(ctx, rhs_mapping->ctx) << "Expected device context " << ctx
      << ". But got " << rhs_mapping->ctx << " for rhs_data.";
  }
  if (!utils::IsNoneArray(out_mapping)) {
    CHECK_EQ(ctx, out_mapping->ctx) << "Expected device context " << ctx
      << ". But got " << out_mapping->ctx << " for rhs_data.";
  }
  // Process mapping
  if (HasBcast(lhs_data, rhs_data)) {
    BcastInfo info = CalcBcastInfo(lhs_data, rhs_data);
    DGL_XPU_SWITCH(ctx.device_type, BinaryReduceBcastImpl,
        info, reducer, op, graph,
        lhs, rhs,
        lhs_data, rhs_data, out_data,
        lhs_mapping, rhs_mapping, out_mapping);
  } else {
    CHECK(IsValidBinaryOpShape(lhs_data, rhs_data))
      << "Cannot compute binary operation between feature shapes "
      << ShapeString(lhs_data) << " and " << ShapeString(rhs_data);
    DGL_XPU_SWITCH(ctx.device_type, BinaryReduceImpl,
        reducer, op, graph,
        lhs, rhs,
        lhs_data, rhs_data, out_data,
        lhs_mapping, rhs_mapping, out_mapping);
  }
}

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelBinaryOpReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    std::string op = args[1];
    GraphHandle ghdl = args[2];
    int lhs = args[3];
    int rhs = args[4];
    NDArray lhs_data = args[5];
    NDArray rhs_data = args[6];
    NDArray out_data = args[7];
    NDArray lhs_mapping = args[8];
    NDArray rhs_mapping = args[9];
    NDArray out_mapping = args[10];

    GraphInterface* gptr = static_cast<GraphInterface*>(ghdl);
    const ImmutableGraph* igptr = dynamic_cast<ImmutableGraph*>(gptr);
    CHECK(igptr) << "Invalid graph object argument. Must be an immutable graph.";
    BinaryOpReduce(reducer, op, igptr,
        static_cast<binary_op::Target>(lhs), static_cast<binary_op::Target>(rhs),
        lhs_data, rhs_data, out_data,
        lhs_mapping, rhs_mapping, out_mapping);
  });

void BackwardLhsBinaryOpReduce(
    const std::string& reducer,
    const std::string& op,
    const ImmutableGraph* graph,
    binary_op::Target lhs, binary_op::Target rhs,
    NDArray lhs_mapping,
    NDArray rhs_mapping,
    NDArray out_mapping,
    NDArray lhs_data,
    NDArray rhs_data,
    NDArray out_data,
    NDArray grad_out_data,
    NDArray grad_lhs_data) {
  // sanity check
  const auto& ctx = graph->Context();
  CHECK_EQ(ctx, lhs_data->ctx) << "Expected device context " << ctx
    << ". But got " << lhs_data->ctx << " for lhs_data.";
  CHECK_EQ(ctx, rhs_data->ctx) << "Expected device context " << ctx
    << ". But got " << rhs_data->ctx << " for rhs_data.";
  CHECK_EQ(ctx, out_data->ctx) << "Expected device context " << ctx
    << ". But got " << out_data->ctx << " for out_data.";
  CHECK_EQ(ctx, grad_out_data->ctx) << "Expected device context " << ctx
    << ". But got " << grad_out_data->ctx << " for grad_out_data.";
  CHECK_EQ(ctx, grad_lhs_data->ctx) << "Expected device context " << ctx
    << ". But got " << grad_lhs_data->ctx << " for grad_lhs_data.";
  if (!utils::IsNoneArray(lhs_mapping)) {
    CHECK_EQ(ctx, lhs_mapping->ctx) << "Expected device context " << ctx
      << ". But got " << lhs_mapping->ctx << " for rhs_data.";
  }
  if (!utils::IsNoneArray(rhs_mapping)) {
    CHECK_EQ(ctx, rhs_mapping->ctx) << "Expected device context " << ctx
      << ". But got " << rhs_mapping->ctx << " for rhs_data.";
  }
  if (!utils::IsNoneArray(out_mapping)) {
    CHECK_EQ(ctx, out_mapping->ctx) << "Expected device context " << ctx
      << ". But got " << out_mapping->ctx << " for rhs_data.";
  }
  if (HasBcast(lhs_data, rhs_data)) {
    BcastInfo info = CalcBcastInfo(lhs_data, rhs_data);
    DGL_XPU_SWITCH(ctx.device_type, BackwardBinaryReduceBcastImpl,
        info, reducer, op, graph,
        lhs, rhs,
        lhs_mapping, rhs_mapping, out_mapping,
        lhs_data, rhs_data, out_data, grad_out_data,
        grad_lhs_data, utils::NoneArray());
  } else {
    DGL_XPU_SWITCH(ctx.device_type, BackwardBinaryReduceImpl,
        reducer, op, graph,
        lhs, rhs,
        lhs_mapping, rhs_mapping, out_mapping,
        lhs_data, rhs_data, out_data, grad_out_data,
        grad_lhs_data, utils::NoneArray());
  }
}

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelBackwardLhsBinaryOpReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    std::string op = args[1];
    GraphHandle ghdl = args[2];
    int lhs = args[3];
    int rhs = args[4];
    NDArray lhs_mapping = args[5];
    NDArray rhs_mapping = args[6];
    NDArray out_mapping = args[7];
    NDArray lhs_data = args[8];
    NDArray rhs_data = args[9];
    NDArray out_data = args[10];
    NDArray grad_out_data = args[11];
    NDArray grad_lhs_data = args[12];

    GraphInterface* gptr = static_cast<GraphInterface*>(ghdl);
    const ImmutableGraph* igptr = dynamic_cast<ImmutableGraph*>(gptr);
    CHECK(igptr) << "Invalid graph object argument. Must be an immutable graph.";
    BackwardLhsBinaryOpReduce(
        reducer, op, igptr,
        static_cast<binary_op::Target>(lhs), static_cast<binary_op::Target>(rhs),
        lhs_mapping, rhs_mapping, out_mapping,
        lhs_data, rhs_data, out_data, grad_out_data,
        grad_lhs_data);
  });

void BackwardRhsBinaryOpReduce(
    const std::string& reducer,
    const std::string& op,
    const ImmutableGraph* graph,
    binary_op::Target lhs, binary_op::Target rhs,
    NDArray lhs_mapping,
    NDArray rhs_mapping,
    NDArray out_mapping,
    NDArray lhs_data,
    NDArray rhs_data,
    NDArray out_data,
    NDArray grad_out_data,
    NDArray grad_rhs_data) {
  // sanity check
  const auto& ctx = graph->Context();
  CHECK_EQ(ctx, lhs_data->ctx) << "Expected device context " << ctx
    << ". But got " << lhs_data->ctx << " for lhs_data.";
  CHECK_EQ(ctx, rhs_data->ctx) << "Expected device context " << ctx
    << ". But got " << rhs_data->ctx << " for rhs_data.";
  CHECK_EQ(ctx, out_data->ctx) << "Expected device context " << ctx
    << ". But got " << out_data->ctx << " for out_data.";
  CHECK_EQ(ctx, grad_out_data->ctx) << "Expected device context " << ctx
    << ". But got " << grad_out_data->ctx << " for grad_out_data.";
  CHECK_EQ(ctx, grad_rhs_data->ctx) << "Expected device context " << ctx
    << ". But got " << grad_rhs_data->ctx << " for grad_rhs_data.";
  if (!utils::IsNoneArray(lhs_mapping)) {
    CHECK_EQ(ctx, lhs_mapping->ctx) << "Expected device context " << ctx
      << ". But got " << lhs_mapping->ctx << " for rhs_data.";
  }
  if (!utils::IsNoneArray(rhs_mapping)) {
    CHECK_EQ(ctx, rhs_mapping->ctx) << "Expected device context " << ctx
      << ". But got " << rhs_mapping->ctx << " for rhs_data.";
  }
  if (!utils::IsNoneArray(out_mapping)) {
    CHECK_EQ(ctx, out_mapping->ctx) << "Expected device context " << ctx
      << ". But got " << out_mapping->ctx << " for rhs_data.";
  }
  if (HasBcast(lhs_data, rhs_data)) {
    BcastInfo info = CalcBcastInfo(lhs_data, rhs_data);
    DGL_XPU_SWITCH(ctx.device_type, BackwardBinaryReduceBcastImpl,
        info, reducer, op, graph,
        lhs, rhs,
        lhs_mapping, rhs_mapping, out_mapping,
        lhs_data, rhs_data, out_data, grad_out_data,
        utils::NoneArray(), grad_rhs_data);
  } else {
    DGL_XPU_SWITCH(ctx.device_type, BackwardBinaryReduceImpl,
        reducer, op, graph,
        lhs, rhs,
        lhs_mapping, rhs_mapping, out_mapping,
        lhs_data, rhs_data, out_data, grad_out_data,
        utils::NoneArray(), grad_rhs_data);
  }
}

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelBackwardRhsBinaryOpReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    std::string op = args[1];
    GraphHandle ghdl = args[2];
    int lhs = args[3];
    int rhs = args[4];
    NDArray lhs_mapping = args[5];
    NDArray rhs_mapping = args[6];
    NDArray out_mapping = args[7];
    NDArray lhs_data = args[8];
    NDArray rhs_data = args[9];
    NDArray out_data = args[10];
    NDArray grad_out_data = args[11];
    NDArray grad_rhs_data = args[12];

    GraphInterface* gptr = static_cast<GraphInterface*>(ghdl);
    const ImmutableGraph* igptr = dynamic_cast<ImmutableGraph*>(gptr);
    CHECK(igptr) << "Invalid graph object argument. Must be an immutable graph.";
    BackwardRhsBinaryOpReduce(
        reducer, op, igptr,
        static_cast<binary_op::Target>(lhs), static_cast<binary_op::Target>(rhs),
        lhs_mapping, rhs_mapping, out_mapping,
        lhs_data, rhs_data, out_data, grad_out_data,
        grad_rhs_data);
  });

void CopyReduce(
    const std::string& reducer,
    const ImmutableGraph* graph,
    binary_op::Target target,
    NDArray in_data, NDArray out_data,
    NDArray in_mapping, NDArray out_mapping) {
  // sanity check
  const auto& ctx = graph->Context();
  CHECK_EQ(ctx, in_data->ctx) << "Expected device context " << ctx
    << ". But got " << in_data->ctx << " for in_data.";
  CHECK_EQ(ctx, out_data->ctx) << "Expected device context " << ctx
    << ". But got " << out_data->ctx << " for out_data.";
  if (!utils::IsNoneArray(in_mapping)) {
    CHECK_EQ(ctx, in_mapping->ctx) << "Expected device context " << ctx
      << ". But got " << in_mapping->ctx << " for rhs_data.";
  }
  if (!utils::IsNoneArray(out_mapping)) {
    CHECK_EQ(ctx, out_mapping->ctx) << "Expected device context " << ctx
      << ". But got " << out_mapping->ctx << " for rhs_data.";
  }
  DGL_XPU_SWITCH(ctx.device_type, BinaryReduceImpl,
      reducer, binary_op::kUseLhs, graph,
      target, binary_op::kDst /* any value != target could do */,
      in_data, utils::NoneArray(), out_data,
      in_mapping, utils::NoneArray(), out_mapping);
}

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelCopyReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    GraphHandle ghdl = args[1];
    int target = args[2];
    NDArray in_data = args[3];
    NDArray out_data = args[4];
    NDArray in_mapping = args[5];
    NDArray out_mapping = args[6];

    GraphInterface* gptr = static_cast<GraphInterface*>(ghdl);
    const ImmutableGraph* igptr = dynamic_cast<ImmutableGraph*>(gptr);
    CHECK(igptr) << "Invalid graph object argument. Must be an immutable graph.";
    CopyReduce(reducer, igptr,
        static_cast<binary_op::Target>(target),
        in_data, out_data,
        in_mapping, out_mapping);
  });

void BackwardCopyReduce(
    const std::string& reducer,
    const ImmutableGraph* graph,
    binary_op::Target target,
    NDArray in_mapping,
    NDArray out_mapping,
    NDArray in_data,
    NDArray out_data,
    NDArray grad_out_data,
    NDArray grad_in_data) {
  // sanity check
  const auto& ctx = graph->Context();
  CHECK_EQ(ctx, in_data->ctx) << "Expected device context " << ctx
    << ". But got " << in_data->ctx << " for in_data.";
  CHECK_EQ(ctx, out_data->ctx) << "Expected device context " << ctx
    << ". But got " << out_data->ctx << " for out_data.";
  CHECK_EQ(ctx, grad_out_data->ctx) << "Expected device context " << ctx
    << ". But got " << grad_out_data->ctx << " for grad_out_data.";
  CHECK_EQ(ctx, grad_in_data->ctx) << "Expected device context " << ctx
    << ". But got " << grad_in_data->ctx << " for grad_in_data.";
  if (!utils::IsNoneArray(in_mapping)) {
    CHECK_EQ(ctx, in_mapping->ctx) << "Expected device context " << ctx
      << ". But got " << in_mapping->ctx << " for rhs_data.";
  }
  if (!utils::IsNoneArray(out_mapping)) {
    CHECK_EQ(ctx, out_mapping->ctx) << "Expected device context " << ctx
      << ". But got " << out_mapping->ctx << " for rhs_data.";
  }
  DGL_XPU_SWITCH(ctx.device_type, BackwardBinaryReduceImpl,
      reducer, binary_op::kUseLhs, graph,
      target, binary_op::kDst /* any value != target could do */,
      in_mapping, utils::NoneArray(), out_mapping,
      in_data, utils::NoneArray(), out_data, grad_out_data,
      grad_in_data, utils::NoneArray());
}

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelBackwardCopyReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    GraphHandle ghdl = args[1];
    int target = args[2];
    NDArray in_data = args[3];
    NDArray out_data = args[4];
    NDArray grad_out_data = args[5];
    NDArray grad_in_data = args[6];
    NDArray in_mapping = args[7];
    NDArray out_mapping = args[8];

    GraphInterface* gptr = static_cast<GraphInterface*>(ghdl);
    const ImmutableGraph* igptr = dynamic_cast<ImmutableGraph*>(gptr);
    CHECK(igptr) << "Invalid graph object argument. Must be an immutable graph.";
    BackwardCopyReduce(
        reducer, igptr, static_cast<binary_op::Target>(target),
        in_mapping, out_mapping,
        in_data, out_data, grad_out_data,
        grad_in_data);
  });

}  // namespace kernel
}  // namespace dgl
