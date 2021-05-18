"""API creating NCCL communicators."""

from .. import backend as F
from .._ffi.function import _init_api

_COMM_MODES_MAP = {
    'remainder': 0
}

class UniqueId(object):
    """ Class for allowing python code to create and communicate NCCL Unique
        IDs, needed for creating communicators.
    """
    def __init__(self, id_str=None):
        """ Create an object reference the current NCCL unique id.
        """
        if id_str:
            if isinstance(id_str, bytes):
                id_str = id_str.decode('utf-8')
            self._handle = _CAPI_DGLNCCLUniqueIdFromString(id_str)
        else:
            self._handle = _CAPI_DGLNCCLGetUniqueId()

    def get(self):
        """ Get the C-handle for this object.
        """
        return self._handle

    def __str__(self):
        return _CAPI_DGLNCCLUniqueIdToString(self._handle)

    def __repr__(self):
        return "UniqueId[{}]".format(str(self))

    def __eq__(self, other):
        return str(self) == str(other)


class Communicator(object):
    """ High-level wrapper for NCCL communication.
    """
    def __init__(self, size, rank, unique_id):
        """ Create a new NCCL communicator.

            Parameters
            ----------
            size : int
                The number of processes in the communicator.
            rank : int
                The rank of the current process in the communicator.
            unique_id : NCCLUniqueId
                The unique id of the root process (rank=0).
        """
        assert rank < size, "The rank of a process must be less than the " \
            "size of the communicator."
        self._handle = _CAPI_DGLNCCLCreateComm(size, rank, unique_id.get())
        self._rank = rank
        self._size = size

    def sparse_all_to_all_push(self, idx, value, partition):
        """ Perform an all-to-all-v operation, where by all processors send out
            a set of indices and corresponding values.

            Parameters
            ----------
            idx : tensor
                The 1D set of indices to send to other processors.
            value : tensor
                The multi-dimension set of values to send to other processors.
                The 0th dimension must match that of `idx`.
            partition : NDArrayPartition
                The object containing information for assigning indices to
                processors.

            Returns
            -------
            tensor
                The 1D tensor of the recieved indices.
            tensor
                The set of recieved values.

        """
        out_idx, out_value = _CAPI_DGLNCCLSparseAllToAllPush(
            self.get(), F.zerocopy_to_dgl_ndarray(idx),
            F.zerocopy_to_dgl_ndarray(value),
            partition.get())
        return (F.zerocopy_from_dgl_ndarray(out_idx),
                F.zerocopy_from_dgl_ndarray(out_value))

    def sparse_all_to_all_pull(self, req_idx, value, partition):
        """ Perform an all-to-all-v operation, where by all processors request
            the values corresponding to ther set of indices.

            Parameters
            ----------
            req_idx : IdArray
                The set of indices this processor is requesting.
            value : NDArray
                The multi-dimension set of values that can be requested from
                this processor.
            partition : NDArrayPartition
                The object containing information for assigning indices to
                processors.

            Returns
            -------
            tensor
                The set of recieved values, corresponding to `req_idx`.

        """
        out_value = _CAPI_DGLNCCLSparseAllToAllPull(
            self.get(), F.zerocopy_to_dgl_ndarray(req_idx),
            F.zerocopy_to_dgl_ndarray(value),
            partition.get())
        return F.zerocopy_from_dgl_ndarray(out_value)

    def get(self):
        """ Get the C-Handle for this object.
        """
        return self._handle

    def rank(self):
        """ Get the rank of this process in this communicator.

            Returns
            -------
            int
                The rank of this process.
        """
        return self._rank

    def size(self):
        """ Get the size of this communicator.

            Returns
            -------
            int
                The number of processes in this communicator.
        """
        return self._size

_init_api("dgl.cuda.nccl")
