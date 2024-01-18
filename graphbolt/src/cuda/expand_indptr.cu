/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/expand_indptr.cu
 * @brief ExpandIndptr operator implementation on CUDA.
 */
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/cub.cuh>
#include <limits>

#include "./common.h"

namespace graphbolt {
namespace ops {

template <typename indices_t, typename nodes_t>
struct RepeatIndex {
  const nodes_t* nodes;
  __host__ __device__ auto operator()(indices_t i) {
    return thrust::make_constant_iterator(nodes ? nodes[i] : i);
  }
};

template <typename indptr_t, typename indices_t>
struct OutputBufferIndexer {
  const indptr_t* indptr;
  indices_t* buffer;
  __host__ __device__ auto operator()(int64_t i) { return buffer + indptr[i]; }
};

template <typename indptr_t>
struct AdjacentDifference {
  const indptr_t* indptr;
  __host__ __device__ auto operator()(int64_t i) {
    return indptr[i + 1] - indptr[i];
  }
};

torch::Tensor ExpandIndptrImpl(
    torch::Tensor indptr, torch::ScalarType output_dtype,
    torch::optional<int64_t> num_edges, torch::optional<torch::Tensor> nodes) {
  if (!num_edges.has_value()) {
    num_edges = AT_DISPATCH_INTEGRAL_TYPES(
        indptr.scalar_type(), "ExpandIndptrIndptr[-1]", ([&]() -> int64_t {
          auto indptr_ptr = indptr.data_ptr<scalar_t>();
          auto num_edges = cuda::CopyScalar{indptr_ptr + indptr.size(0) - 1};
          return static_cast<scalar_t>(num_edges);
        }));
  }
  auto csc_rows =
      torch::empty(num_edges.value(), indptr.options().dtype(output_dtype));

  AT_DISPATCH_INTEGRAL_TYPES(
      indptr.scalar_type(), "ExpandIndptrIndptr", ([&] {
        using indptr_t = scalar_t;
        auto indptr_ptr = indptr.data_ptr<indptr_t>();
        AT_DISPATCH_INTEGRAL_TYPES(
            output_dtype, "ExpandIndptrIndices", ([&] {
              using indices_t = scalar_t;
              auto csc_rows_ptr = csc_rows.data_ptr<indices_t>();

              auto nodes_dtype =
                  nodes ? nodes.value().scalar_type() : output_dtype;
              AT_DISPATCH_INTEGRAL_TYPES(
                  nodes_dtype, "ExpandIndptrNodes", ([&] {
                    using nodes_t = scalar_t;
                    auto nodes_ptr =
                        nodes ? nodes.value().data_ptr<nodes_t>() : nullptr;

                    thrust::counting_iterator<int64_t> iota(0);
                    auto input_buffer = thrust::make_transform_iterator(
                        iota, RepeatIndex<indices_t, nodes_t>{nodes_ptr});
                    auto output_buffer = thrust::make_transform_iterator(
                        iota, OutputBufferIndexer<indptr_t, indices_t>{
                                  indptr_ptr, csc_rows_ptr});
                    auto buffer_sizes = thrust::make_transform_iterator(
                        iota, AdjacentDifference<indptr_t>{indptr_ptr});

                    const auto num_rows = indptr.size(0) - 1;
                    constexpr int64_t max_copy_at_once =
                        std::numeric_limits<int32_t>::max();
                    for (int64_t i = 0; i < num_rows; i += max_copy_at_once) {
                      CUB_CALL(
                          DeviceCopy::Batched, input_buffer + i,
                          output_buffer + i, buffer_sizes + i,
                          std::min(num_rows - i, max_copy_at_once));
                    }
                  }));
            }));
      }));
  return csc_rows;
}

}  // namespace ops
}  // namespace graphbolt