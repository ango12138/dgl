# TODO(gaiyu): DFS

import random
import sys
import time

import dgl
import dgl.backend as F
import dgl.ndarray as nd
import dgl.utils as utils
import igraph
import networkx as nx
import numpy as np
import torch as th
import scipy.sparse as sp

g = dgl.DGLGraph()
n = int(sys.argv[2])
a = sp.random(n, n, 10 / n, data_rvs=lambda n: np.ones(n))
print(a.todense())
g.from_scipy_sparse_matrix(a)

ig = igraph.Graph(directed=True)
ig.add_vertices(n)
ig.add_edges(list(zip(a.row, a.col)))

src = random.choice(range(n))

t0 = time.time()
fs = getattr(g._graph, sys.argv[1])
layers_cpp = fs([src], out=True, step=False)
t_cpp = time.time() - t0

t0 = time.time()
v, delimeter, _ = ig.bfs(src)
t_igraph = time.time() - t0

print(t_cpp, t_igraph)
layers_igraph = [v[i : j] for i, j in zip(delimeter[:-1], delimeter[1:])]
toset = lambda x: set(utils.toindex(x).tousertensor().numpy())
print(layers_cpp)
print(layers_igraph)
assert len(layers_cpp) == len(layers_igraph)
assert all(toset(x) == set(y) for x, y in zip(layers_cpp, layers_igraph))
