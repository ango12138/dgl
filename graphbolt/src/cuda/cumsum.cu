/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/cumsum.cu
 * @brief Cumsum operators implementation on CUDA.
 */
#include <cub/cub.cuh>

#include "./common.h"

namespace graphbolt {
namespace ops {

torch::Tensor ExclusiveCumSum(torch::Tensor indptr) {
  auto allocator = cuda::GetAllocator();
  auto stream = cuda::GetCurrentStream();
  auto result = torch::empty_like(indptr);

  AT_DISPATCH_INTEGRAL_TYPES(
      indptr.scalar_type(), "ExclusiveCumSum", ([&] {
        size_t tmp_storage_size = 0;
        cub::DeviceScan::ExclusiveSum(
            nullptr, tmp_storage_size, indptr.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(), indptr.size(0), stream);
        auto tmp_storage = allocator.AllocateStorage<char>(tmp_storage_size);
        cub::DeviceScan::ExclusiveSum(
            tmp_storage.get(), tmp_storage_size, indptr.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(), indptr.size(0), stream);
      }));
  return result;
}

}  // namespace ops
}  // namespace graphbolt