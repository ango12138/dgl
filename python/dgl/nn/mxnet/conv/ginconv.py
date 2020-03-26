"""MXNet Module for Graph Isomorphism Network layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import mxnet as mx
from mxnet.gluon import nn

from .... import function as fn
from ....utils import expand_as_pair


class GINConv(nn.Block):
    r"""Graph Isomorphism Network layer from paper `How Powerful are Graph
    Neural Networks? <https://arxiv.org/pdf/1810.00826.pdf>`__.

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula.
    aggregator_type : str
        Aggregator type to use (``sum``, ``max`` or ``mean``).
    init_eps : float, optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter.
    """
    def __init__(self,
                 apply_func,
                 aggregator_type,
                 init_eps=0,
                 learn_eps=False):
        super(GINConv, self).__init__()
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))

        with self.name_scope():
            self.apply_func = apply_func
            self.eps = self.params.get('eps',
                                       shape=(1,),
                                       grad_req='write' if learn_eps else 'null',
                                       init=mx.init.Constant(init_eps))

    def forward(self, graph, feat):
        r"""Compute Graph Isomorphism Network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray
            If a mxnet.NDArray is given, the input feature of shape :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of
            nodes.
            If a pair of mxnet.NDArray is given, the pair must contain two tensors of
            shape :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input dimensionality requirement of ``apply_func``.

        Returns
        -------
        mxnet.NDArray
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output dimensionality of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as input dimensionality.
        """
        graph = graph.local_var()
        feat_src, feat_dst = expand_as_pair(feat)
        graph.srcdata['h'] = feat_src
        graph.update_all(fn.copy_u('h', 'm'), self._reducer('m', 'neigh'))
        rst = (1 + self.eps.data(feat_dst.context)) * feat_dst + graph.dstdata['neigh']
        if self.apply_func is not None:
            rst = self.apply_func(rst)
        return rst
