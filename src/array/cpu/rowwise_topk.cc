/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/rowwise_topk.cc
 * \brief rowwise topk
 */
#include <numeric>
#include <algorithm>
#include "./rowwise_pick.h"

namespace dgl {
namespace aten {
namespace impl {
namespace {

template <typename IdxType, typename FloatType>
inline PickFn<IdxType> GetTopkPickFn(int64_t k, FloatArray weight, bool ascending) {
  const FloatType* wdata = static_cast<FloatType*>(weight->data);
  PickFn<IdxType> pick_fn = [k, ascending, wdata]
    (IdxType rowid, IdxType off, IdxType len,
     const IdxType* col, const IdxType* data,
     IdxType* out_idx) {
      std::function<bool(IdxType, IdxType)> compare_fn;
      if (ascending) {
        if (data) {
          compare_fn = [wdata, data] (IdxType i, IdxType j) {
              return wdata[data[i]] < wdata[data[j]];
            };
        } else {
          compare_fn = [wdata, data] (IdxType i, IdxType j) {
              return wdata[i] < wdata[j];
            };
        }
      } else {
        if (data) {
          compare_fn = [wdata, data] (IdxType i, IdxType j) {
              return wdata[data[i]] > wdata[data[j]];
            };
        } else {
          compare_fn = [wdata, data] (IdxType i, IdxType j) {
              return wdata[i] > wdata[j];
            };
        }
      }

      std::vector<IdxType> idx(len);
      std::iota(idx.begin(), idx.end(), off);
      std::sort(idx.begin(), idx.end(), compare_fn);
      for (int64_t j = 0; j < k; ++j) {
        out_idx[j] = idx[j];
      }
    };

  return pick_fn;
}

}  // namespace

template <DLDeviceType XPU, typename IdxType, typename FloatType>
COOMatrix CSRRowWiseTopk(
    CSRMatrix mat, IdArray rows, int64_t k, FloatArray weight, bool ascending) {
  auto pick_fn = GetTopkPickFn<IdxType, FloatType>(k, weight, ascending);
  return CSRRowWisePick(mat, rows, k, false, pick_fn);
}

template COOMatrix CSRRowWiseTopk<kDLCPU, int32_t, float>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseTopk<kDLCPU, int64_t, float>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseTopk<kDLCPU, int32_t, double>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseTopk<kDLCPU, int64_t, double>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);

template <DLDeviceType XPU, typename IdxType, typename FloatType>
COOMatrix COORowWiseTopk(
    COOMatrix mat, IdArray rows, int64_t k, FloatArray weight, bool ascending) {
  auto pick_fn = GetTopkPickFn<IdxType, FloatType>(k, weight, ascending);
  return COORowWisePick(mat, rows, k, false, pick_fn);
}

template COOMatrix COORowWiseTopk<kDLCPU, int32_t, float>(
    COOMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix COORowWiseTopk<kDLCPU, int64_t, float>(
    COOMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix COORowWiseTopk<kDLCPU, int32_t, double>(
    COOMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix COORowWiseTopk<kDLCPU, int64_t, double>(
    COOMatrix, IdArray, int64_t, FloatArray, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
