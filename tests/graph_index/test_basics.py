from dgl import DGLError
from dgl.utils import toindex
from dgl.graph_index import create_graph_index
import networkx as nx

def test_edge_id():
    gi = create_graph_index()

    gi.add_nodes(4)
    gi.add_edge(0, 1)
    eid = gi.edge_id(0, 1).tolist()
    assert len(eid) == 1
    assert eid[0] == 0
    assert not gi.is_multigraph()

    # multiedges
    gi.add_edge(0, 1)
    eid = gi.edge_id(0, 1).tolist()
    assert len(eid) == 2
    assert eid[0] == 0
    assert eid[1] == 1
    assert gi.is_multigraph()

    gi.add_edges(toindex([0, 1, 1, 2]), toindex([2, 2, 2, 3]))
    src, dst, eid = gi.edge_ids(toindex([0, 0, 2, 1]), toindex([2, 1, 3, 2]))
    eid_answer = [2, 0, 1, 5, 3, 4]
    assert len(eid) == 6
    assert all(e == ea for e, ea in zip(eid, eid_answer))

    # find edges
    src, dst, eid = gi.find_edges(toindex([1, 3, 5]))
    assert len(src) == len(dst) == len(eid) == 3
    assert src[0] == 0 and src[1] == 1 and src[2] == 2
    assert dst[0] == 1 and dst[1] == 2 and dst[2] == 3
    assert eid[0] == 1 and eid[1] == 3 and eid[2] == 5

    # source broadcasting
    src, dst, eid = gi.edge_ids(toindex([0]), toindex([1, 2]))
    eid_answer = [0, 1, 2]
    assert len(eid) == 3
    assert all(e == ea for e, ea in zip(eid, eid_answer))

    # destination broadcasting
    src, dst, eid = gi.edge_ids(toindex([1, 0]), toindex([2]))
    eid_answer = [3, 4, 2]
    assert len(eid) == 3
    assert all(e == ea for e, ea in zip(eid, eid_answer))

    gi.clear()
    assert not gi.is_multigraph()
    # the following assumes that grabbing nonexistent edge will throw an error
    try:
        gi.edge_id(0, 1)
        fail = True
    except DGLError:
        fail = False
    finally:
        assert not fail

    gi.add_nodes(4)
    gi.add_edge(0, 1)
    eid = gi.edge_id(0, 1).tolist()
    assert len(eid) == 1
    assert eid[0] == 0

def test_nx():
    gi = create_graph_index()

    gi.add_nodes(2)
    gi.add_edge(0, 1)
    nxg = gi.to_networkx()
    assert not nxg.is_multigraph()
    gi.add_edge(0, 1)
    nxg = gi.to_networkx()
    assert nxg.is_multigraph()

    nxg = nx.Graph()
    nxg.add_edge(0, 1)
    gi.from_networkx(nxg)
    assert not gi.is_multigraph()

    nxg = nx.MultiGraph()
    nxg.add_edge(0, 1)
    nxg.add_edge(0, 1)
    gi.from_networkx(nxg)
    assert gi.is_multigraph()

def test_predsucc():
    gi = create_graph_index()

    gi.add_nodes(4)
    gi.add_edge(0, 1)
    gi.add_edge(0, 1)
    gi.add_edge(0, 2)
    gi.add_edge(2, 0)
    gi.add_edge(3, 0)
    gi.add_edge(0, 0)
    gi.add_edge(0, 0)

    pred = gi.predecessors(0)
    assert len(pred) == 3
    assert 2 in pred
    assert 3 in pred
    assert 0 in pred

    succ = gi.successors(0)
    assert len(succ) == 3
    assert 1 in succ
    assert 2 in succ
    assert 0 in succ


if __name__ == '__main__':
    test_edge_id()
    test_nx()
