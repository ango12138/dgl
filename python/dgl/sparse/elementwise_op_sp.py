"""DGL elementwise operators for sparse matrix module."""
from typing import Union

import torch

from .sparse_matrix import SparseMatrix, val_like


def spsp_add(A, B):
    """Invoke C++ sparse library for addition"""
    return SparseMatrix(
        torch.ops.dgl_sparse.spsp_add(A.c_sparse_matrix, B.c_sparse_matrix)
    )


def sp_add(A: SparseMatrix, B: SparseMatrix) -> SparseMatrix:
    """Elementwise addition

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    B : SparseMatrix
        Sparse matrix

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
    # Python falls back to B.__radd__ then TypeError when NotImplemented is
    # returned.
    return spsp_add(A, B) if isinstance(B, SparseMatrix) else NotImplemented


def sp_mul(A: SparseMatrix, B: Union[SparseMatrix, float, int]) -> SparseMatrix:
    """Elementwise multiplication

    Parameters
    ----------
    A : SparseMatrix
        First operand
    B : SparseMatrix or float or int
        Second operand

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
    if isinstance(B, (float, int)):
        return val_like(A, A.val * B)
    elif isinstance(B, SparseMatrix):
        return val_like(B, A * B.val)
    # Python falls back to B.__rmul__(A) then TypeError when NotImplemented is
    # returned.
    # So this also handles the case of scalar * SparseMatrix since we set
    # SparseMatrix.__rmul__ to be the same as SparseMatrix.__mul__.
    return NotImplemented


def sp_power(A: SparseMatrix, scalar: Union[float, int]) -> SparseMatrix:
    """Take the power of each nonzero element and return a sparse matrix with
    the result.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    scalar : float or int
        Exponent

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
    # Python falls back to scalar.__rpow__ then TypeError when NotImplemented
    # is returned.
    return (
        val_like(A, A.val**scalar)
        if isinstance(scalar, (float, int))
        else NotImplemented
    )


SparseMatrix.__add__ = sp_add
SparseMatrix.__mul__ = sp_mul
SparseMatrix.__rmul__ = sp_mul
SparseMatrix.__pow__ = sp_power
