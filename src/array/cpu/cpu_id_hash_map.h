/**
 *  Copyright (c) 2019 by Contributors
 * @file dgl/cpu_id_hash_map.h
 * @brief Class about CPU id hash map
 */

#ifndef DGL_ARRAY_CPU_CPU_ID_HASH_MAP_H_
#define DGL_ARRAY_CPU_CPU_ID_HASH_MAP_H_

#include <dgl/aten/types.h>

#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#define CAS(ptr, oldval, newval, ret)                         \
  do {                                                        \
    if (sizeof(newval) == 32) {                               \
      *ret = _InterlockedCompareExchange(                     \
        reinterpret_cast<LONG*>(ptr), newval, oldval);        \
    } else if (sizeof(newval) == 64) {                        \
      *ret = _InterlockedCompareExchange64(                   \
        reinterpret_cast<LONGLONG*>(ptr), newval, oldval);    \
    } else {                                                  \
      LOG(FATAL) << "ID can only be int32 or int64";          \
    }                                                         \
  } while (0)
#elif __GNUC__
#if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 1)
#error "requires GCC 4.1 or greater"
#endif
#define CAS(ptr, oldval, newval, ret) \
  *ret = __sync_val_compare_and_swap(ptr, oldval, newval)
#else
#error "CAS not supported on this platform"
#endif

namespace dgl {
namespace aten {

template <typename IdType>
class CpuIdHashMap {
 public:
  struct Mapping {
      IdType key;
      IdType value;
  };

  CpuIdHashMap();

  CpuIdHashMap(const CpuIdHashMap& other) = delete;
  CpuIdHashMap& operator=(const CpuIdHashMap& other) = delete;

  size_t Init(IdArray ids, IdArray unique_ids);

  void Map(IdArray ids, IdType default_val, IdArray new_ids) const;

  ~CpuIdHashMap();

  // Return the new id of the given id. If the given id is not contained
  // in the hash map, returns the default_val instead.
  IdType map(IdType id, IdType default_val) const;

  size_t fillInIds(size_t num_ids,
    const IdType* ids_data, IdArray unique_ids);

  void next(IdType* pos, IdType* delta) const;

  void insert_cas(IdType id, std::vector<int16_t>* valid, size_t index);

  // Key must exist.
  void set_value(IdType k, IdType v);

  bool attempt_insert_at(int64_t pos, IdType key,
    std::vector<int16_t>* valid, size_t index);

 private:
  static constexpr IdType kEmptyKey = static_cast<IdType>(-1);
  static constexpr int grain_size = 1024;

  Mapping* _hmap;
  IdType _mask;
};

}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_CPU_ID_HASH_MAP_H_
