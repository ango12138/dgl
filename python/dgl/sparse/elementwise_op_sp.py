"""DGL elementwise operators for sparse matrix module."""
from typing import Union
from numbers import Number

import torch

from .diag_matrix import DiagMatrix
from .sparse_matrix import SparseMatrix, val_like
from .utils import is_scalar


def spsp_add(A, B):
    """Invoke C++ sparse library for addition"""
    return SparseMatrix(
        torch.ops.dgl_sparse.spsp_add(A.c_sparse_matrix, B.c_sparse_matrix)
    )

# Since these functions are never exposed but instead used for implementing
# the builtin operators, we can return NotImplemented to (1) raise TypeError
# for invalid value types, (2) not handling DiagMatrix in SparseMatrix
# functions but instead delegate it to DiagMatrix.__radd__ etc, (3) allow
# others to implement their own class that operates with SparseMatrix objects.
# See also:
# https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

def sp_add(A: SparseMatrix, B: SparseMatrix) -> SparseMatrix:
    """Elementwise addition

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    B : SparseMatrix
        The other sparse matrix

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------

    >>> row = torch.tensor([1, 0, 2])
    >>> col = torch.tensor([0, 3, 2])
    >>> val = torch.tensor([10, 20, 30])
    >>> A = from_coo(row, col, val, shape=(3, 4))
    >>> A + A
    SparseMatrix(indices=tensor([[0, 1, 2],
            [3, 0, 2]]),
    values=tensor([40, 20, 60]),
    shape=(3, 4), nnz=3)
    """
    return spsp_add(A, B) if isinstance(B, SparseMatrix) else NotImplemented


def sp_sub(A: SparseMatrix, B: SparseMatrix) -> SparseMatrix:
    """Elementwise subtraction

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    B : SparseMatrix
        The other sparse matrix

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------

    >>> row = torch.tensor([1, 0, 2])
    >>> col = torch.tensor([0, 3, 2])
    >>> val = torch.tensor([10, 20, 30])
    >>> val2 = torch.tensor([5, 10, 15])
    >>> A = from_coo(row, col, val, shape=(3, 4))
    >>> B = from_coo(row, col, val2, shape=(3, 4))
    >>> A - B
    SparseMatrix(indices=tensor([[0, 1, 2],
            [3, 0, 2]]),
    values=tensor([10, 5, 15]),
    shape=(3, 4), nnz=3)
    """
    return A + (-B)


def sp_mul(
    A: SparseMatrix, B: Union[Number, torch.Tensor]
) -> SparseMatrix:
    """Elementwise multiplication

    Parameters
    ----------
    A : SparseMatrix
        First operand
    B : Number or torch.Tensor
        Second operand.  Must be a scalar.

    Returns
    -------
    SparseMatrix
        Result of A * B

    Examples
    --------

    >>> row = torch.tensor([1, 0, 2])
    >>> col = torch.tensor([0, 3, 2])
    >>> val = torch.tensor([1, 2, 3])
    >>> A = from_coo(row, col, val, shape=(3, 4))

    >>> A * 2
    SparseMatrix(indices=tensor([[1, 0, 2],
            [0, 3, 2]]),
    values=tensor([2, 4, 6]),
    shape=(3, 4), nnz=3)

    >>> 2 * A
    SparseMatrix(indices=tensor([[1, 0, 2],
            [0, 3, 2]]),
    values=tensor([2, 4, 6]),
    shape=(3, 4), nnz=3)
    """
    return val_like(A, A.val * B) if is_scalar(B) else NotImplemented


def sp_div(
    A: SparseMatrix, B: Union[Number, torch.Tensor]
) -> SparseMatrix:
    """Elementwise division

    Parameters
    ----------
    A : SparseMatrix
        First operand
    B : Number or torch.Tensor
        Second operand.  Must be a scalar.

    Returns
    -------
    SparseMatrix
        Result of A / B

    Examples
    --------

    >>> row = torch.tensor([1, 0, 2])
    >>> col = torch.tensor([0, 3, 2])
    >>> val = torch.tensor([1, 2, 3])
    >>> A = from_coo(row, col, val, shape=(3, 4))

    >>> A / 2
    SparseMatrix(indices=tensor([[1, 0, 2],
            [0, 3, 2]]),
    values=tensor([0.5000, 1.0000, 1.5000]),
    shape=(3, 4), nnz=3)
    """
    return val_like(A, A.val / B) if is_scalar(B) else NotImplemented


def sp_power(
    A: SparseMatrix, scalar: Union[Number, torch.Tensor]
) -> SparseMatrix:
    """Take the power of each nonzero element and return a sparse matrix with
    the result.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    scalar : Number or torch.Tensor
        Exponent.  Must be a scalar.

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    >>> row = torch.tensor([1, 0, 2])
    >>> col = torch.tensor([0, 3, 2])
    >>> val = torch.tensor([10, 20, 30])
    >>> A = from_coo(row, col, val)
    >>> A ** 2
    SparseMatrix(indices=tensor([[1, 0, 2],
            [0, 3, 2]]),
    values=tensor([100, 400, 900]),
    shape=(3, 4), nnz=3)
    """
    return val_like(A, A.val ** scalar) if is_scalar(scalar) else NotImplemented


SparseMatrix.__add__ = sp_add
SparseMatrix.__sub__ = sp_sub
SparseMatrix.__mul__ = sp_mul
SparseMatrix.__rmul__ = sp_mul
SparseMatrix.__truediv__ = sp_div
SparseMatrix.__pow__ = sp_power
