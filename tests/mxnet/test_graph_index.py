import os
os.environ['DGLBACKEND'] = 'mxnet'
import networkx as nx
import mxnet as mx
import numpy as np
import scipy as sp
import dgl
from dgl.graph_index import map_to_subgraph_nid, GraphIndex, create_graph_index
from dgl import utils

def generate_from_networkx():
    edges = [[2, 3], [2, 5], [3, 0], [1, 0], [4, 3], [4, 5]]
    nx_graph = nx.DiGraph()
    nx_graph.add_edges_from(edges)
    g = create_graph_index(nx_graph)
    ig = create_graph_index(nx_graph, readonly=True)
    return g, ig

def generate_from_edgelist():
    edges = [[2, 3], [2, 5], [3, 0], [6, 10], [10, 3], [10, 15]]
    g = create_graph_index(edges)
    ig = create_graph_index(edges, readonly=True)
    return g, ig

def generate_rand_graph(n):
    arr = (sp.sparse.random(n, n, density=0.1, format='coo') != 0).astype(np.int64)
    g = create_graph_index(arr)
    ig = create_graph_index(arr, readonly=True)
    return g, ig

def check_graph_equal(g1, g2):
    adj1 = g1.adjacency_matrix(False, mx.cpu())[0] != 0
    adj2 = g2.adjacency_matrix(False, mx.cpu())[0] != 0
    assert mx.nd.sum(adj1 - adj2).asnumpy() == 0

def test_graph_gen():
    g, ig = generate_from_edgelist()
    check_graph_equal(g, ig)
    g, ig = generate_rand_graph(10)
    check_graph_equal(g, ig)

def sort_edges(edges):
    edges = [e.tousertensor() for e in edges]
    idx = mx.nd.argsort(edges[2])
    return (edges[0][idx], edges[1][idx], edges[2][idx])

def check_basics(g, ig):
    assert g.number_of_nodes() == ig.number_of_nodes()
    assert g.number_of_edges() == ig.number_of_edges()

    edges = g.edges(True)
    iedges = ig.edges(True)
    assert np.all(edges[0].tousertensor().asnumpy() == iedges[0].tousertensor().asnumpy())
    assert np.all(edges[1].tousertensor().asnumpy() == iedges[1].tousertensor().asnumpy())
    assert np.all(edges[2].tousertensor().asnumpy() == iedges[2].tousertensor().asnumpy())

    for i in range(g.number_of_nodes()):
        assert g.has_node(i) == ig.has_node(i)

    for i in range(g.number_of_nodes()):
        assert mx.nd.sum(g.predecessors(i).tousertensor()).asnumpy() == mx.nd.sum(ig.predecessors(i).tousertensor()).asnumpy()
        assert mx.nd.sum(g.successors(i).tousertensor()).asnumpy() == mx.nd.sum(ig.successors(i).tousertensor()).asnumpy()

    randv = np.random.randint(0, g.number_of_nodes(), 10)
    randv = utils.toindex(randv)
    in_src1, in_dst1, in_eids1 = sort_edges(g.in_edges(randv))
    in_src2, in_dst2, in_eids2 = sort_edges(ig.in_edges(randv))
    nnz = in_src2.shape[0]
    assert mx.nd.sum(in_src1 == in_src2).asnumpy() == nnz
    assert mx.nd.sum(in_dst1 == in_dst2).asnumpy() == nnz
    assert mx.nd.sum(in_eids1 == in_eids2).asnumpy() == nnz

    out_src1, out_dst1, out_eids1 = sort_edges(g.out_edges(randv))
    out_src2, out_dst2, out_eids2 = sort_edges(ig.out_edges(randv))
    nnz = out_dst2.shape[0]
    assert mx.nd.sum(out_dst1 == out_dst2).asnumpy() == nnz
    assert mx.nd.sum(out_src1 == out_src2).asnumpy() == nnz
    assert mx.nd.sum(out_eids1 == out_eids2).asnumpy() == nnz

    num_v = len(randv)
    assert mx.nd.sum(g.in_degrees(randv).tousertensor() == ig.in_degrees(randv).tousertensor()).asnumpy() == num_v
    assert mx.nd.sum(g.out_degrees(randv).tousertensor() == ig.out_degrees(randv).tousertensor()).asnumpy() == num_v
    randv = randv.tousertensor()
    for v in randv.asnumpy():
        assert g.in_degree(v) == ig.in_degree(v)
        assert g.out_degree(v) == ig.out_degree(v)

    for u in randv.asnumpy():
        for v in randv.asnumpy():
            if len(g.edge_id(u, v)) == 1:
                assert g.edge_id(u, v).tonumpy() == ig.edge_id(u, v).tonumpy()
            assert g.has_edge_between(u, v) == ig.has_edge_between(u, v)
    randv = utils.toindex(randv)
    ids = g.edge_ids(randv, randv)[2].tonumpy()
    assert sum(ig.edge_ids(randv, randv)[2].tonumpy() == ids) == len(ids)
    assert sum(g.has_edges_between(randv, randv).tonumpy() == ig.has_edges_between(randv, randv).tonumpy()) == len(randv)


def test_basics():
    g, ig = generate_from_edgelist()
    check_basics(g, ig)
    g, ig = generate_from_networkx()
    check_basics(g, ig)
    g, ig = generate_rand_graph(100)
    check_basics(g, ig)

def test_node_subgraph():
    num_vertices = 100
    g, ig = generate_rand_graph(num_vertices)

    # node_subgraph
    randv1 = np.random.randint(0, num_vertices, 20)
    randv = np.unique(randv1)
    subg = g.node_subgraph(utils.toindex(randv))
    subig = ig.node_subgraph(utils.toindex(randv))
    check_basics(subg, subig)
    check_graph_equal(subg, subig)
    assert mx.nd.sum(map_to_subgraph_nid(subg, utils.toindex(randv1[0:10])).tousertensor()
            == map_to_subgraph_nid(subig, utils.toindex(randv1[0:10])).tousertensor()) == 10

    # node_subgraphs
    randvs = []
    subgs = []
    for i in range(4):
        randv = np.unique(np.random.randint(0, num_vertices, 20))
        randvs.append(utils.toindex(randv))
        subgs.append(g.node_subgraph(utils.toindex(randv)))
    subigs= ig.node_subgraphs(randvs)
    for i in range(4):
        check_basics(subg, subig)
        check_graph_equal(subgs[i], subigs[i])

def test_create_graph():
    elist = [(1, 2), (0, 1), (0, 2)]
    ig = dgl.DGLGraph(elist, readonly=True)
    g = dgl.DGLGraph(elist, readonly=False)
    for edge in elist:
        assert g.edge_id(edge[0], edge[1]) == ig.edge_id(edge[0], edge[1])

    data = [1, 2, 3]
    rows = [1, 0, 0]
    cols = [2, 1, 2]
    mat = sp.sparse.coo_matrix((data, (rows, cols)))
    g = dgl.DGLGraph(mat, readonly=False)
    ig = dgl.DGLGraph(mat, readonly=True)
    for edge in elist:
        assert g.edge_id(edge[0], edge[1]) == ig.edge_id(edge[0], edge[1])

if __name__ == '__main__':
    test_basics()
    test_graph_gen()
    test_node_subgraph()
    test_create_graph()
