/*!
 *  Copyright (c) 2020 by Contributors
 * \file dgl/aten/array_ops.h
 * \brief Common array operations required by DGL.
 *
 * Note that this is not meant for a full support of array library such as ATen.
 * Only a limited set of operators required by DGL are implemented.
 */
#ifndef DGL_ATEN_ARRAY_OPS_H_
#define DGL_ATEN_ARRAY_OPS_H_

#include <utility>
#include <tuple>
#include "./types.h"

namespace dgl {
namespace aten {

//////////////////////////////////////////////////////////////////////
// ID array
//////////////////////////////////////////////////////////////////////

/*! \return A special array to represent null. */
inline NDArray NullArray() {
  return NDArray::Empty({0}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
}

/*!
 * \return Whether the input array is a null array.
 */
inline bool IsNullArray(NDArray array) {
  return array->shape[0] == 0;
}

/*!
 * \brief Create a new id array with given length
 * \param length The array length
 * \param ctx The array context
 * \param nbits The number of integer bits
 * \return id array
 */
IdArray NewIdArray(int64_t length,
                   DLContext ctx = DLContext{kDLCPU, 0},
                   uint8_t nbits = 64);

/*!
 * \brief Create a new id array using the given vector data
 * \param vec The vector data
 * \param nbits The integer bits of the returned array
 * \param ctx The array context
 * \return the id array
 */
template <typename T>
IdArray VecToIdArray(const std::vector<T>& vec,
                     uint8_t nbits = 64,
                     DLContext ctx = DLContext{kDLCPU, 0});

/*!
 * \brief Return an array representing a 1D range.
 * \param low Lower bound (inclusive).
 * \param high Higher bound (exclusive).
 * \param nbits result array's bits (32 or 64)
 * \param ctx Device context
 * \return range array
 */
IdArray Range(int64_t low, int64_t high, uint8_t nbits, DLContext ctx);

/*!
 * \brief Return an array full of the given value
 * \param val The value to fill.
 * \param length Number of elements.
 * \param nbits result array's bits (32 or 64)
 * \param ctx Device context
 * \return the result array
 */
IdArray Full(int64_t val, int64_t length, uint8_t nbits, DLContext ctx);

/*! \brief Create a deep copy of the given array */
IdArray Clone(IdArray arr);

/*! \brief Convert the idarray to the given bit width */
IdArray AsNumBits(IdArray arr, uint8_t bits);

/*! \brief Arithmetic functions */
IdArray Add(IdArray lhs, IdArray rhs);
IdArray Sub(IdArray lhs, IdArray rhs);
IdArray Mul(IdArray lhs, IdArray rhs);
IdArray Div(IdArray lhs, IdArray rhs);

IdArray Add(IdArray lhs, dgl_id_t rhs);
IdArray Sub(IdArray lhs, dgl_id_t rhs);
IdArray Mul(IdArray lhs, dgl_id_t rhs);
IdArray Div(IdArray lhs, dgl_id_t rhs);

IdArray Add(dgl_id_t lhs, IdArray rhs);
IdArray Sub(dgl_id_t lhs, IdArray rhs);
IdArray Mul(dgl_id_t lhs, IdArray rhs);
IdArray Div(dgl_id_t lhs, IdArray rhs);

BoolArray LT(IdArray lhs, dgl_id_t rhs);

/*! \brief Stack two arrays (of len L) into a 2*L length array */
IdArray HStack(IdArray arr1, IdArray arr2);

/*!
 * \brief Return the data under the index. In numpy notation, A[I]
 * \tparam ValueType The type of return value.
 */
template<typename ValueType>
ValueType IndexSelect(NDArray array, uint64_t index);
NDArray IndexSelect(NDArray array, IdArray index);

/*!
 * \brief Permute the elements of an array according to given indices.
 *
 * Equivalent to:
 *
 * <code>
 *     result = np.zeros_like(array)
 *     result[indices] = array
 * </code>
 */
NDArray Scatter(NDArray array, IdArray indices);

/*!
 * \brief Repeat each element a number of times.  Equivalent to np.repeat(array, repeats)
 * \param array A 1D vector
 * \param repeats A 1D integer vector for number of times to repeat for each element in
 *                \c array.  Must have the same shape as \c array.
 */
NDArray Repeat(NDArray array, IdArray repeats);

/*!
 * \brief Relabel the given ids to consecutive ids.
 *
 * Relabeling is done inplace. The mapping is created from the union
 * of the give arrays.
 *
 * \param arrays The id arrays to relabel.
 * \return mapping array M from new id to old id.
 */
IdArray Relabel_(const std::vector<IdArray>& arrays);

/*!\brief Return whether the array is a valid 1D int array*/
inline bool IsValidIdArray(const dgl::runtime::NDArray& arr) {
  return arr->ndim == 1 && arr->dtype.code == kDLInt;
}

/*!
 * \brief Packs a tensor containing padded sequences of variable length.
 *
 * Similar to \c pack_padded_sequence in PyTorch, except that
 *
 * 1. The length for each sequence (before padding) is inferred as the number
 *    of elements before the first occurrence of \c pad_value.
 * 2. It does not sort the sequences by length.
 * 3. Along with the tensor containing the packed sequence, it returns both the
 *    length, as well as the offsets to the packed tensor, of each sequence.
 *
 * \param array The tensor containing sequences padded to the same length
 * \param pad_value The padding value
 * \return A triplet of packed tensor, the length tensor, and the offset tensor
 *
 * \note Example: consider the following array with padding value -1:
 *
 * <code>
 *     [[1, 2, -1, -1],
 *      [3, 4,  5, -1]]
 * </code>
 *
 * The packed tensor would be [1, 2, 3, 4, 5].
 *
 * The length tensor would be [2, 3], i.e. the length of each sequence before padding.
 *
 * The offset tensor would be [0, 2], i.e. the offset to the packed tensor for each
 * sequence (before padding)
 */
template<typename ValueType>
std::tuple<NDArray, IdArray, IdArray> Pack(NDArray array, ValueType pad_value);

/*!
 * \brief Batch-slice a 1D or 2D array, and then pack the list of sliced arrays
 * by concatenation.
 *
 * If a 2D array is given, then the function is equivalent to:
 *
 * <code>
 *     def ConcatSlices(array, lengths):
 *         slices = [array[i, :l] for i, l in enumerate(lengths)]
 *         packed = np.concatenate(slices)
 *         offsets = np.cumsum([0] + lengths[:-1])
 *         return packed, offsets
 * </code>
 *
 * If a 1D array is given, then the function is equivalent to
 *
 * <code>
 *     def ConcatSlices(array, lengths):
 *         slices = [array[:l] for l in lengths]
 *         packed = np.concatenate(slices)
 *         offsets = np.cumsum([0] + lengths[:-1])
 *         return packed, offsets
 * </code>
 *
 * \param array A 1D or 2D tensor for slicing
 * \param lengths A 1D tensor indicating the number of elements to slice
 * \return The tensor with packed slices along with the offsets.
 */
std::pair<NDArray, IdArray> ConcatSlices(NDArray array, IdArray lengths);

// inline implementations
template <typename T>
IdArray VecToIdArray(const std::vector<T>& vec,
                     uint8_t nbits,
                     DLContext ctx) {
  IdArray ret = NewIdArray(vec.size(), DLContext{kDLCPU, 0}, nbits);
  if (nbits == 32) {
    std::copy(vec.begin(), vec.end(), static_cast<int32_t*>(ret->data));
  } else if (nbits == 64) {
    std::copy(vec.begin(), vec.end(), static_cast<int64_t*>(ret->data));
  } else {
    LOG(FATAL) << "Only int32 or int64 is supported.";
  }
  return ret.CopyTo(ctx);
}

}  // namespace aten
}  // namespace dgl

#endif  // DGL_ATEN_ARRAY_OPS_H_
