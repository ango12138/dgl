/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file utils.h
 * @brief CUDA utilities.
 */

#ifndef GRAPHBOLT_CUDA_UTILS_H_
#define GRAPHBOLT_CUDA_UTILS_H_

namespace graphbolt {
namespace cuda {

// The cache line size of GPU.
#define GPU_CACHE_LINE_SIZE 128
// The max number of threads per block.
#define CUDA_MAX_NUM_THREADS 1024

/**
 * @brief Calculate the number of threads needed given the size of the dimension
 * to be processed.
 *
 * It finds the largest power of two that is less than or equal to the minimum
 * of size and CUDA_MAX_NUM_THREADS.
 */
inline int FindNumThreads(int size) {
  int ret = 1;
  while ((ret << 1) <= std::min(size, CUDA_MAX_NUM_THREADS)) {
    ret <<= 1;
  }
  return ret;
}

template <typename T>
int _NumberOfBits(const T& range) {
  if (range <= 1) {
    // ranges of 0 or 1 require no bits to store
    return 0;
  }

  int bits = 1;
  const auto urange = static_cast<std::make_unsigned_t<T>>(range);
  while (bits < static_cast<int>(sizeof(T) * 8) && (1ull << bits) < urange) {
    ++bits;
  }

  return bits;
}

}  // namespace cuda
}  // namespace graphbolt

#endif  // GRAPHBOLT_CUDA_UTILS_H_
