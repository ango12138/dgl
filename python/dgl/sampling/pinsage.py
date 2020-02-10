"""PinSAGE sampler & related functions and classes"""

import numpy as np

from .. import backend as F
from .. import convert
from .. import transform
from .. import function as fn
from .randomwalks import random_walk


class GenericPinSAGESampler(object):
    """PinSAGE-like sampler extended to any heterographs, given a metapath.

    Given a heterogeneous graph, PinSAGE neighbor sampler would generate a homogeneous
    graph where the neighbors of each node are the most commonly visited nodes of the
    same type by random walk with restarts.  The random walk with restarts are based
    on a given metapath, which should have the same beginning and ending node type.

    Parameters
    ----------
    G : DGLHeteroGraph
        The bidirectional bipartite graph.

        The graph should only have two node types: ``ntype`` and ``other_type``.
        The graph should only have two edge types, one connecting from ``ntype`` to
        ``other_type``, and another connecting from ``other_type`` to ``ntype``.

        PinSAGE works on a bidirectional bipartite graph where for each edge
        going from node u to node v, there exists an edge going from node v to node u.
    ntype : str
        The node type for which the graph would be constructed on.
    metapath : list[str]
        The metapath.
    random_walk_length : int
        The maximum number of steps of random walk with restarts.

        Note that here we consider traversing from ``ntype`` to ``other_type`` then back
        to ``ntype`` as a single step (i.e. a single step consists of two hops).

        Usually considered a hyperparameter.
    random_walk_restart_prob : int
        Restart probability of random walk with restarts.

        Note that the random walks only would halt on node type ``ntype``, and would
        never halt on ``other_type``.

        Usually considered a hyperparameter.
    num_random_walks : int
        Number of random walks to try for each seed node.

        Usually considered a hyperparameter.
    num_neighbors : int
        Number of neighbors to select for each seed.
    weight_column : str, default "weights"
        The weight of each neighbor, stored as an edge feature.

    Inputs
    ------
    seed_nodes : Tensor
        A tensor of seed node IDs of node type ``ntype``.

    Outputs
    -------
    g : DGLHeteroGraph
        A homogeneous graph constructed by selecting neighbors for each seed node according
        to PinSAGE algorithm.

    Examples
    --------
    See examples in :any:`PinSAGESampler`.
    """
    def __init__(self, G, ntype, metapath, random_walk_length, random_walk_restart_prob,
                 num_random_walks, num_neighbors, weight_column='weights'):
        self.G = G
        self.ntype = ntype
        self.weight_column = weight_column
        self.num_random_walks = num_random_walks
        self.num_neighbors = num_neighbors
        self.random_walk_length = random_walk_length

        self.metapath = metapath * random_walk_length
        restart_prob = np.tile(np.array([random_walk_restart_prob, 0]), random_walk_length)
        restart_prob[0] = 0     # allow at least one step
        self.restart_prob = F.zerocopy_from_numpy(restart_prob)

    # pylint: disable=no-member
    def __call__(self, seed_nodes):
        seed_nodes = F.repeat(seed_nodes, self.num_random_walks, 0)
        paths, _ = random_walk(
            self.G, seed_nodes, metapath=self.metapath, restart_prob=self.restart_prob)
        src = F.reshape(paths[:, 2::2], (-1,))
        dst = F.repeat(paths[:, 0], self.random_walk_length, 0)

        src_mask = (src != -1)
        src = F.boolean_mask(src, src_mask)
        dst = F.boolean_mask(dst, src_mask)

        neighbor_graph = convert.graph(
            (src, dst), card=self.G.number_of_nodes(self.ntype), ntype=self.ntype)
        neighbor_graph = transform.to_simple(neighbor_graph, return_counts=self.weight_column)
        neighbor_graph = transform.select_topk(
            neighbor_graph, self.weight_column, self.num_neighbors)

        # normalize weights
        with neighbor_graph.local_scope():
            neighbor_graph.edata[self.weight_column] = F.astype(
                neighbor_graph.edata[self.weight_column], F.float32)
            neighbor_graph.update_all(
                fn.copy_e(self.weight_column, 'm'),
                fn.sum('m', 's'))
            neighbor_graph.apply_edges(
                fn.e_div_v(self.weight_column, 's', 'w'))
            new_weights = neighbor_graph.edata['w']
        neighbor_graph.edata[self.weight_column] = new_weights

        return neighbor_graph


class PinSAGESampler(GenericPinSAGESampler):
    """PinSAGE neighbor sampler.

    Given a bidirectional bipartite graph, PinSAGE neighbor sampler would generate
    a homogeneous graph where the neighbors of each node are the most commonly visited
    nodes of the same type by random walk with restarts.

    Parameters
    ----------
    G : DGLHeteroGraph
        The bidirectional bipartite graph.

        The graph should only have two node types: ``ntype`` and ``other_type``.
        The graph should only have two edge types, one connecting from ``ntype`` to
        ``other_type``, and another connecting from ``other_type`` to ``ntype``.

        PinSAGE works on a bidirectional bipartite graph where for each edge
        going from node u to node v, there exists an edge going from node v to node u.
    ntype : str
        The node type for which the graph would be constructed on.
    other_type : str
        The other node type.
    random_walk_length : int
        The maximum number of steps of random walk with restarts.

        Note that here we consider traversing from ``ntype`` to ``other_type`` then back
        to ``ntype`` as a single step (i.e. a single step consists of two hops).

        Usually considered a hyperparameter.
    random_walk_restart_prob : int
        Restart probability of random walk with restarts.

        Note that the random walks only would halt on node type ``ntype``, and would
        never halt on ``other_type``.

        Usually considered a hyperparameter.
    num_random_walks : int
        Number of random walks to try for each seed node.

        Usually considered a hyperparameter.
    num_neighbors : int
        Number of neighbors to select for each seed.
    weight_column : str, default "weights"
        The weight of each neighbor, stored as an edge feature.

    Inputs
    ------
    seed_nodes : Tensor
        A tensor of seed node IDs of node type ``ntype``.

    Outputs
    -------
    g : DGLHeteroGraph
        A homogeneous graph constructed by selecting neighbors for each seed node according
        to PinSAGE algorithm.

    Examples
    --------
    Generate a random bidirectional bipartite graph with 3000 "A" nodes and 5000 "B" nodes.
    >>> g = scipy.sparse.random(3000, 5000, 0.003)
    >>> G = dgl.heterograph({
    ...     ('A', 'AB', 'B'): g,
    ...     ('B', 'BA', 'A'): g.T})

    Then we create a PinSAGE neighbor sampler that samples a graph of node type "A".  Each
    node would have (a maximum of) 10 neighbors.
    >>> sampler = dgl.sampling.PinSAGESampler(G, 'A', 'B', 3, 0.5, 200, 10)

    This is how we select the neighbors for node #0, #1 and #2 of type "A" according to
    PinSAGE algorithm:
    >>> seeds = torch.LongTensor([0, 1, 2])
    >>> frontier = sampler(seeds)
    >>> frontier.all_edges(form='uv')
    (tensor([ 230,    0,  802,   47,   50, 1639, 1533,  406, 2110, 2687, 2408, 2823,
                0,  972, 1230, 1658, 2373, 1289, 1745, 2918, 1818, 1951, 1191, 1089,
             1282,  566, 2541, 1505, 1022,  812]),
     tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2]))

    For an end-to-end example of PinSAGE model, including sampling on multiple layers
    and computing with the sampled graphs, please refer to [TODO]

    References
    ----------
    Graph Convolutional Neural Networks for Web-Scale Recommender Systems
        Ying et al., 2018, https://arxiv.org/abs/1806.01973
    """
    def __init__(self, G, ntype, other_type, random_walk_length, random_walk_restart_prob,
                 num_random_walks, num_neighbors, weight_column='weights'):
        metagraph = G.metagraph
        fw_etype = list(metagraph[ntype][other_type])[0]
        bw_etype = list(metagraph[other_type][ntype])[0]
        super().__init__(G, ntype, [fw_etype, bw_etype], random_walk_length,
                         random_walk_restart_prob, num_random_walks, num_neighbors,
                         weight_column=weight_column)
