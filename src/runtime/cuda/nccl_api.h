/*!
 *  Copyright (c) 2021 by Contributors
 * \file nccl_api.h
 * \brief Wrapper around NCCL routines. 
 */

#include "nccl.h"

#include <dgl/runtime/object.h>

namespace dgl {
namespace runtime {
namespace cuda {

class NCCLUniqueId : public runtime::Object {
 public:
  NCCLUniqueId();

  static constexpr const char* _type_key = "cuda.NCCLUniqueId";
  DGL_DECLARE_OBJECT_TYPE_INFO(NCCLUniqueId, Object);

  ncclUniqueId Get() const;

 private:
  ncclUniqueId id_;
};

DGL_DEFINE_OBJECT_REF(NCCLUniqueIdRef, NCCLUniqueId);

class NCCLCommunicator : public runtime::Object {
 public:
  NCCLCommunicator(
      int size,
      int rank,
      ncclUniqueId id);

  ~NCCLCommunicator();

  // disable copying
  NCCLCommunicator(const NCCLCommunicator& other) = delete;
  NCCLCommunicator& operator=(
      const NCCLCommunicator& other);

  ncclComm_t Get();

  /**
   * @brief Perform an all-to-all communication.
   *
   * @param send The continous array of data to send.
   * @param size The size of data to send to each rank.
   * @param recv The continous array of data to recieve.
   * @param type The type of data to send.
   * @param stream The stream to operate on.
   */
  void AllToAll(
      const void * send,
      int64_t size,
      void * recv,
      ncclDataType_t type,
      cudaStream_t stream);

  /**
   * @brief Perform an all-to-all variable sized communication.
   *
   * @param send The arrays of data to send.
   * @param send_size The size of each array to send.
   * @param recv The arrays of data to recieve.
   * @param recv_size The size of each array to recieve.
   * @param type The type of data to send.
   * @param stream The stream to operate on.
   */
  void AllToAllV(
      const void * const * const send,
      const int64_t * send_size,
      void * const * const recv,
      const int64_t * recv_size,
      ncclDataType_t type,
      cudaStream_t stream);

  /**
   * @brief Perform an all-to-all with sparse data (idx and value pairs). By
   * necessity, the sizes of each message are variable.
   *
   * @tparam IdType The type of index.
   * @tparam DType The type of value.
   * @param send_idx The set of indexes to send on the device.
   * @param send_value The set of values to send on the device.
   * @param send_prefix The exclusive prefix sum of elements to send on the
   * host.
   * @param recv_idx The set of indexes to recieve on the device.
   * @param recv_value The set of values to recieve on the device.
   * @param recv_prefix The exclusive prefix sum of the number of elements to
   * recieve on the host.
   * @param stream The stream to communicate on.
   */
  template<typename IdType, typename DType>
  void SparseAllToAll(
          const IdType * send_idx,
          const DType * send_value,
          const int64_t * send_prefix,
          IdType * recv_idx,
          DType * recv_value,
          const int64_t * recv_prefix,
          cudaStream_t stream);

  static constexpr const char* _type_key = "cuda.NCCLCommunicator";
  DGL_DECLARE_OBJECT_TYPE_INFO(NCCLCommunicator, Object);

 private:
  ncclComm_t comm_;
  int size_;
  int rank_;
};

DGL_DEFINE_OBJECT_REF(NCCLCommunicatorRef, NCCLCommunicator);

}
}
}
