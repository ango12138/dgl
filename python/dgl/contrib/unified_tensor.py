"""Unified Tensor."""
from .. import backend as F
from .._ffi.function import _init_api
from .. import utils


class UnifiedTensor: #UnifiedTensor
    '''Class for storing unified tensor.

    Parameters
    ----------
    input : Tensor
        Tensor which we want to convert into the
        unified tensor.
    device : device
        Device to create the mapping of the unified tensor.
    '''

    def __init__(self, input, device):
        if F.device_type(device) != 'cuda':
            raise ValueError("Target device must be a cuda device")

        self._input = input
        self._array = F.zerocopy_to_dgl_ndarray(self._input)
        self._device = device

        self._array.pin_memory_(utils.to_dgl_context(self._device))

    def __len__(self):
        return len(self._array)

    def __repr__(self):
        return self._input.__repr__()

    def __getitem__(self, key):
        return self._input[key]

    def __setitem__(self, key, val):
        self._input[key] = val

    def __del__(self):
        self._array.unpin_memory_(utils.to_dgl_context(self._device))
        self._array = None
        self._input = None

    def gather_row(self, index):
        '''Gather the rows designated by the index. Performs
        a direct GPU to CPU access using the unified virtual
        memory (UVM) capability of CUDA.

        Parameters
        ----------
        index : Tensor
            Tensor which contains the row indicies
        '''
        return F.zerocopy_from_dgl_ndarray(
                _CAPI_DGLIndexSelectCPUFromGPU(self._array,
                            F.zerocopy_to_dgl_ndarray(index)))

    @property
    def shape(self):
        """Shape of this tensor"""
        return self._array.shape

    @property
    def dtype(self):
        """Type of this tensor"""
        return self._array.dtype

    @property
    def device(self):
        """Device of this tensor"""
        return self._device

_init_api("dgl.ndarray.uvm", __name__)