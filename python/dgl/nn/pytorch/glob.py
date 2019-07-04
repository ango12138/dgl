"""Torch modules for graph global pooling."""
# pylint: disable= no-member, arguments-differ
import torch as th
import torch.nn as nn
from torch.nn import init

from ... import function as fn, BatchedDGLGraph
from ...utils import get_ndata_name, get_edata_name
from ...batched_graph import sum_nodes, mean_nodes, max_nodes, broadcast_nodes, softmax_nodes, topk_nodes


class SumPooling(nn.Module):
    r"""Apply sum pooling over the graph.
    """
    _feat_name = '_gpool_feat'
    def __init__(self):
        super(SumPooling, self).__init__()

    def forward(self, feat, graph):
        _feat_name = get_ndata_name(graph, self._feat_name)
        graph.ndata[_feat_name] = feat
        readout = sum_nodes(graph, _feat_name)
        graph.ndata.pop(_feat_name)
        return readout


class AvgPooling(nn.Module):
    r"""Apply average pooling over the graph.
    """
    _feat_name = '_gpool_avg'
    def __init__(self):
        super(AvgPooling, self).__init__()

    def forward(self, feat, graph):
        _feat_name = get_ndata_name(graph, self._feat_name)
        graph.ndata[_feat_name] = feat
        readout = mean_nodes(graph, _feat_name)
        graph.ndata.pop(_feat_name)
        return readout


class MaxPooling(nn.Module):
    r"""Apply max pooling over the graph.
    """
    _feat_name = '_gpool_max'
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, feat, graph):
        _feat_name = get_ndata_name(graph, self._feat_name)
        graph.ndata[_feat_name] = feat
        readout = max_nodes(graph, _feat_name)
        graph.ndata.pop(_feat_name)
        return readout


class SortPooling(nn.Module):
    r"""Apply sort pooling (from paper "An End-to-End Deep Learning Architecture
    for Graph Classification") over the graph.
    """
    _feat_name = '_gpool_sort'
    def __init__(self, k):
        super(SortPooling, self).__init__()
        self.k = k

    def forward(self, feat, graph):
        # Sort the feature of each node in ascending order.
        feat, _ = feat.sort(dim=-1)
        graph.ndata[self._feat_name] = feat
        # Sort nodes according to the their last features.
        ret = topk_nodes(graph, self._feat_name, self.k).view(-1, self.k * feat.shape[-1])
        g.ndata.pop(self._feat_name)
        return ret


class GlobAttnPooling(nn.Module):
    r"""Apply global attention pooling over the graph.
    """
    _gate_name = '_gpool_attn_gate'
    _readout_name = '_gpool_attn_readout'
    def __init__(self, gate_nn, nn=None):
        super(GlobAttnPooling, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        # TODO(zihao): reset parameters of gate_nn and nn
        pass

    def forward(self, feat, graph):
        feat = feat.unsqueeze(-1) if feat.dim() == 1 else feat
        gate = self.gate_nn(feat)
        feat = self.nn(feat) if self.nn else feat

        feat_name = get_ndata_name(graph, self.gate_name)
        graph.ndata[feat_name] = gate
        gate = softmax_nodes(graph, feat_name)
        graph.ndata.pop(feat_name)

        feat_name = get_ndata_name(graph, self.readout_name)
        graph.ndata[feat_name] = feat * gate
        readout = sum_nodes(graph, feat_name)
        graph.ndata.pop(feat_name)

        return readout


class Set2Set(nn.Module):
    r"""Apply Set2Set (from paper "Order Matters: Sequence to sequence for sets") over the graph.
    """
    _score_name = '_gpool_s2s_score'
    _readout_name = '_gpool_s2s_readout'
    def __init__(self, input_dim, n_iters, n_layers):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers= n_layers
        self.lstm = th.nn.LSTM(self.output_dim, self.input_dim, n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        # TODO(zihao): finish this
        pass

    def forward(self, feat, graph):
        batch_size = 1
        if isinstance(graph, BatchedDGLGraph):
            batch_size = graph.batch_size

        h = (feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
             feat.new_zeros((self.n_layers, batch_size, self.input_dim)))
        q_star = feat.new_zeros(batch_size, self.output_dim)

        for i in range(self.n_iters):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.input_dim)

            score = (feat * broadcast_nodes(graph, q)).sum(dim=-1, keepdim=True)
            feat_name = get_ndata_name(graph, self._score_name)
            graph.ndata[feat_name] = score
            score = softmax_nodes(graph, feat_name)
            graph.ndata.pop(feat_name)

            feat_name = get_ndata_name(graph, self._readout_name)
            graph.ndata[feat_name] = feat * score
            readout = sum_nodes(graph, feat_name)
            graph.ndata.pop(feat_name)

            q_star = th.cat([q, readout], dim=-1)

        return q_star

    def extra_repr(self):
        """Set the extra representation of the module.
        which will come into effect when printing the model.
        """
        summary = 'input_dim={input_dim}, out_dim={out_dim}' +\
            'n_iters={n_iters}, n_layers={n_layers}'
        return summary.format(**self.__dict__)
