"""Utility functions related to pinned memory tensors."""

from .. import backend as F
from .._ffi.function import _init_api

def pin_memory_inplace(tensor):
    """Register the tensor into pinned memory in-place (i.e. without copying)."""
    F.to_dgl_nd(tensor).pin_memory_()

def unpin_memory_inplace(tensor):
    """Unregister the tensor from pinned memory in-place (i.e. without copying)."""
    F.to_dgl_nd(tensor).unpin_memory_()

def gather_pinned_tensor_rows(tensor, rows):
    """Directly gather rows from a CPU tensor given an indices array on CUDA devices,
    and returns the result on the same CUDA device without copying.

    Parameters
    ----------
    tensor : Tensor
        The tensor.  Must be in pinned memory.
    rows : Tensor
        The rows to gather.  Must be a CUDA tensor.

    Returns
    -------
    Tensor
        The result with the same device as :attr:`rows`.
    """
    return F.from_dgl_nd(_CAPI_DGLIndexSelectCPUFromGPU(F.to_dgl_nd(tensor), F.to_dgl_nd(rows)))

def scatter_pinned_tensor_rows(dest, rows, source):
    """Directly scatter rows from a GPU tensor given an indices array on CUDA devices,
    to a pinned tensor on the CPU.

    Parameters
    ----------
    dest : Tensor
        The tensor on the CPU to scatter rows to. Must be in pinned memory.
    rows : Tensor
        The rows to scatter. Must be a CUDA tensor with unique entries.
    source : Tensor
        The tensor on the GPU to scatter rows from.
    """
    _CAPI_DGLIndexScatterGPUToCPU(F.to_dgl_nd(dest), F.to_dgl_nd(rows),
        F.to_dgl_nd(source))


_init_api("dgl.ndarray.uvm", __name__)
