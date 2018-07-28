from __future__ import absolute_import

import torch
import scipy.sparse

Tensor = torch.Tensor
SparseTensor = scipy.sparse.spmatrix
sum = torch.sum
max = torch.max

def asnumpy(a):
    return a.cpu().numpy()

def concatenate(tensors, axis=0):
    return torch.concatenate(tensors, axis)

def packable(tensors):
    return all(isinstance(x, torch.Tensor) and \
               x.dtype == tensors[0].dtype and \
               x.shape[1:] == tensors[0].shape[1:] for x in tensors)

def pack(tensors):
    return torch.cat(tensors)

def unpackable(x):
    return isinstance(x, torch.Tensor) and x.numel() > 0

def unpack(x):
    return torch.split(x, 1)

def shape(x):
    return x.shape

def expand_dims(x, axis):
    return x.unsqueeze(axis)

def prod(x, axis=None, keepdims=None):
    args = ([axis] if axis else []) + ([keepdims] if keepdims else []) 
    return torch.prod(x, *args)

def item(x):
    return x.item()

def isinteger(x):
    return x.dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64]

def isin(x, y):
    assert x.device == y.device
    assert x.dtype == y.dtype
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    return (x[None, :] == y[:, None]).any(-1)

def dtype(x):
    return x.dtype

def astype(x, dtype):
    return x.type(dtype)

bool = torch.uint8
ones = torch.ones
unique = torch.unique

def gather_row(data, row_index):
    return torch.index_select(data, 0, row_index)

def scatter_row(data, row_index, value):
    return data.index_copy(0, row_index, value)
