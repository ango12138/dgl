"""Built-in reducer function."""
# pylint: disable=redefined-builtin
from __future__ import absolute_import

import sys

from .base import BuiltinFunction, TargetCode
from ..runtime import ir
from ..runtime.ir import var


class ReduceFunction(BuiltinFunction):
    """Base builtin reduce function class."""

    def _invoke(self, graph, edge_frame, out_size, edge_map=None,
                 out_map=None):
        """Symbolic computation of this builtin function to create
        runtime.executor
        """
        raise NotImplementedError

    @property
    def name(self):
        """Return the name of this builtin function."""
        raise NotImplementedError


class SimpleReduceFunction(ReduceFunction):
    """Builtin reduce function that aggregates a single field into another
    single field."""
    def __init__(self, name, msg_field, out_field):
        self._name = name
        self.msg_field = msg_field
        self.out_field = out_field

    def _invoke(self, graph, edge_frame, out_size, edge_map=None,
                 out_map=None):
        """Symbolic execution of this builtin function"""
        reducer = self._name
        graph = var.GRAPH(graph)
        edge_map = var.MAP(edge_map)
        out_map = var.MAP(out_map)
        edge_data = ir.READ_COL(edge_frame, var.STR(self.msg_field))
        return ir.COPY_REDUCE(reducer, graph, TargetCode.EDGE, edge_data,
                              out_size, edge_map, out_map)

    @property
    def name(self):
        return self._name


###############################################################################
# Generate all following reducer functions:
# sum, max, min, prod

def _gen_reduce_builtin(reducer):
    docstring = """Builtin reduce function that aggregates messages by {0}.

    Parameters
    ----------
    msg : str
        The message field.
    out : str
        The output node feature field.
    Examples
    --------
    >>> import dgl
    >>> reduce_func = dgl.function.{0}('m', 'h')

    The above example is equivalent to the following user defined function
    (if using PyTorch):

    >>> import torch
    >>> def reduce_func(nodes):
    >>>     return {{'h': torch.{0}(nodes.mailbox['m'], dim=1)}}
    """.format(reducer)

    def func(msg, out):
        return SimpleReduceFunction(reducer, msg, out)
    func.__name__ = reducer
    func.__doc__ = docstring
    return func


__all__ = []

for reducer in ["max", "min", "sum", "prod"]:
    func = _gen_reduce_builtin(reducer)
    setattr(sys.modules[__name__], reducer, func)
    __all__.append(reducer)
