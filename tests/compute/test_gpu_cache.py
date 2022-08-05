import numpy as np
import networkx as nx
import unittest
import scipy.sparse as ssp
import pytest

import dgl
import backend as F
from test_utils import parametrize_idtype

D = 5

def generate_graph(idtype, grad=False, add_data=True):
    g = dgl.DGLGraph().to(F.ctx(), dtype=idtype)
    g.add_nodes(10)
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edge(0, i)
        g.add_edge(i, 9)
    # add a back flow from 9 to 0
    g.add_edge(9, 0)
    if add_data:
        ncol = F.randn((10, D))
        ecol = F.randn((17, D))
        if grad:
            ncol = F.attach_grad(ncol)
            ecol = F.attach_grad(ecol)
        g.ndata['h'] = ncol
        g.edata['l'] = ecol
    return g

@unittest.skipIf(not F.gpu_ctx(), reason='only necessary with GPU')
@parametrize_idtype
def test_gpu_cache(idtype):
    g = generate_graph(idtype)
    cache = dgl.contrib.GpuCache(5, D, idtype)
    h = g.ndata['h']

    t = 5
    keys = F.arange(0, t, dtype=idtype)
    values, m_idx, m_keys = cache.query(keys)
    m_values = h[F.tensor(m_keys, F.int64)]
    values[F.tensor(m_idx, F.int64)] = m_values
    cache.replace(m_keys, m_values)

    keys = F.arange(3, 8, dtype=idtype)
    values, m_idx, m_keys = cache.query(keys)
    assert m_keys.shape[0] == 3 and m_idx.shape[0] == 3
    m_values = h[F.tensor(m_keys, F.int64)]
    values[F.tensor(m_idx, F.int64)] = m_values
    assert (values != h[F.tensor(keys, F.int64)]).sum().item() == 0
    cache.replace(m_keys, m_values)

if __name__ == '__main__':
    test_gpu_cache(F.int64)
    test_gpu_cache(F.int32)
