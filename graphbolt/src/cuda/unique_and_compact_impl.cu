/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/unique_and_compact_impl.cu
 * @brief Unique and compact operator implementation on CUDA.
 */
#include <c10/cuda/CUDAStream.h>
#include <graphbolt/cuda_ops.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>

#include <cub/cub.cuh>

#include "./common.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

template <typename scalar_t>
struct EqualityFunc {
  const scalar_t* sorted_order;
  const scalar_t* found_locations;
  const scalar_t* searched_items;
  __host__ __device__ auto operator()(int64_t i) {
    return sorted_order[found_locations[i]] == searched_items[i];
  }
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> UniqueAndCompact(
    const torch::Tensor src_ids, const torch::Tensor dst_ids,
    const torch::Tensor unique_dst_ids, int num_bits) {
  TORCH_CHECK(
      src_ids.scalar_type() == dst_ids.scalar_type() &&
          dst_ids.scalar_type() == unique_dst_ids.scalar_type(),
      "Dtypes of tensors passed to UniqueAndCompact need to be identical.");
  auto allocator = cuda::GetAllocator();
  auto stream = cuda::GetCurrentStream();
  const auto exec_policy = thrust::cuda::par_nosync(allocator).on(stream);
  return AT_DISPATCH_INTEGRAL_TYPES(
      src_ids.scalar_type(), "unique_and_compact", ([&] {
        auto src_ids_ptr = src_ids.data_ptr<scalar_t>();
        auto dst_ids_ptr = dst_ids.data_ptr<scalar_t>();
        auto unique_dst_ids_ptr = unique_dst_ids.data_ptr<scalar_t>();

        // If the given num_bits argument is not in the reasonable range,
        // we recompute it to speedup the expensive sort operations.
        if (num_bits <= 0 || num_bits > sizeof(scalar_t) * 8) {
          auto max_id_src = torch::empty(
              1, c10::TensorOptions()
                     .dtype(src_ids.scalar_type())
                     .pinned_memory(true));
          auto max_id_src_ptr = max_id_src.data_ptr<scalar_t>();
          size_t workspace_size;
          cub::DeviceReduce::Max(
              nullptr, workspace_size, src_ids_ptr, max_id_src_ptr,
              src_ids.size(0));
          auto temp = allocator.AllocateStorage<char>(workspace_size);
          cub::DeviceReduce::Max(
              temp.get(), workspace_size, src_ids_ptr, max_id_src_ptr,
              src_ids.size(0));
          auto max_id_dst = thrust::reduce(
              exec_policy, unique_dst_ids_ptr,
              unique_dst_ids_ptr + unique_dst_ids.size(0),
              static_cast<scalar_t>(0), thrust::maximum<scalar_t>{});
          // thrust::reduce syncs, so we can also safely access max_id_src_ptr
          num_bits =
              cuda::NumberOfBits(std::max(max_id_src_ptr[0], max_id_dst) + 1);
        }

        // Sort the unique_dst_ids tensor.
        auto sorted_unique_dst_ids =
            Sort<false>(unique_dst_ids_ptr, unique_dst_ids.size(0), num_bits);
        auto sorted_unique_dst_ids_ptr =
            sorted_unique_dst_ids.data_ptr<scalar_t>();

        // Mark dst nodes in the src_ids tensor.
        auto is_dst = allocator.AllocateStorage<bool>(src_ids.size(0));
        thrust::binary_search(
            exec_policy, sorted_unique_dst_ids_ptr,
            sorted_unique_dst_ids_ptr + unique_dst_ids.size(0), src_ids_ptr,
            src_ids_ptr + src_ids.size(0), is_dst.get());

        // Filter the non-dst nodes in the src_ids tensor, hence only_src.
        auto only_src =
            torch::empty(src_ids.size(0), sorted_unique_dst_ids.options());
        {
          auto only_src_size =
              thrust::remove_copy_if(
                  exec_policy, src_ids_ptr, src_ids_ptr + src_ids.size(0),
                  is_dst.get(), only_src.data_ptr<scalar_t>(),
                  thrust::identity<bool>{}) -
              only_src.data_ptr<scalar_t>();
          only_src = only_src.slice(0, 0, only_src_size);
        }

        // Sort the only_src tensor so that we can unique it with Encode
        // operation later.
        auto sorted_only_src = Sort<false>(
            only_src.data_ptr<scalar_t>(), only_src.size(0), num_bits);

        auto unique_only_src =
            torch::empty(only_src.size(0), src_ids.options());
        auto unique_only_src_ptr = unique_only_src.data_ptr<scalar_t>();
        auto unique_only_src_cnt = allocator.AllocateStorage<scalar_t>(1);

        {  // Compute the unique operation on the only_src tensor.
          size_t workspace_size;
          CUDA_CALL(cub::DeviceRunLengthEncode::Encode(
              nullptr, workspace_size, sorted_only_src.data_ptr<scalar_t>(),
              unique_only_src_ptr, cub::DiscardOutputIterator{},
              unique_only_src_cnt.get(), only_src.size(0), stream));
          auto temp = allocator.AllocateStorage<char>(workspace_size);
          CUDA_CALL(cub::DeviceRunLengthEncode::Encode(
              temp.get(), workspace_size, sorted_only_src.data_ptr<scalar_t>(),
              unique_only_src_ptr, cub::DiscardOutputIterator{},
              unique_only_src_cnt.get(), only_src.size(0), stream));

          auto unique_only_src_size =
              cuda::CopyScalar(unique_only_src_cnt.get());
          unique_only_src = unique_only_src.slice(
              0, 0, static_cast<scalar_t>(unique_only_src_size));
        }

        auto real_order = torch::cat({unique_dst_ids, unique_only_src});
        // Sort here so that binary search can be used to lookup new_ids.
        auto [sorted_order, new_ids] = Sort(real_order, num_bits);
        auto sorted_order_ptr = sorted_order.data_ptr<scalar_t>();
        auto new_ids_ptr = new_ids.data_ptr<int64_t>();
        // Holds the found locations of the src and dst ids in the sorted_order.
        // Later is used to lookup the new ids of the src_ids and dst_ids
        // tensors.
        auto new_src_ids_loc =
            allocator.AllocateStorage<scalar_t>(src_ids.size(0));
        auto new_dst_ids_loc =
            allocator.AllocateStorage<scalar_t>(dst_ids.size(0));
        thrust::lower_bound(
            exec_policy, sorted_order_ptr,
            sorted_order_ptr + sorted_order.size(0), src_ids_ptr,
            src_ids_ptr + src_ids.size(0), new_src_ids_loc.get());
        thrust::lower_bound(
            exec_policy, sorted_order_ptr,
            sorted_order_ptr + sorted_order.size(0), dst_ids_ptr,
            dst_ids_ptr + dst_ids.size(0), new_dst_ids_loc.get());
        {  // Check if unique_dst_ids includes all dst_ids.
          thrust::counting_iterator<int64_t> iota(0);
          auto equal_it = thrust::make_transform_iterator(
              iota, EqualityFunc<scalar_t>{
                        sorted_order_ptr, new_dst_ids_loc.get(), dst_ids_ptr});
          auto all_exist = thrust::all_of(
              exec_policy, equal_it, equal_it + dst_ids.size(0),
              thrust::identity<bool>());
          if (!all_exist) {
            throw std::out_of_range("Some ids not found.");
          }
        }

        // Finally, lookup the new compact ids of the src and dst tensors via
        // gather operations.
        auto new_src_ids = torch::empty_like(src_ids);
        auto new_dst_ids = torch::empty_like(dst_ids);
        thrust::gather(
            exec_policy, new_src_ids_loc.get(),
            new_src_ids_loc.get() + src_ids.size(0),
            new_ids.data_ptr<int64_t>(), new_src_ids.data_ptr<scalar_t>());
        thrust::gather(
            exec_policy, new_dst_ids_loc.get(),
            new_dst_ids_loc.get() + dst_ids.size(0),
            new_ids.data_ptr<int64_t>(), new_dst_ids.data_ptr<scalar_t>());
        return std::make_tuple(real_order, new_src_ids, new_dst_ids);
      }));
}

}  // namespace ops
}  // namespace graphbolt
