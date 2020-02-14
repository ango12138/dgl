import dgl
import backend as F
import numpy as np
import unittest

def check_random_walk(g, metapath, traces, ntypes, prob=None):
    traces = F.asnumpy(traces)
    ntypes = F.asnumpy(ntypes)
    for j in range(traces.shape[1] - 1):
        assert ntypes[j] == g.get_ntype_id(g.to_canonical_etype(metapath[j])[0])
        assert ntypes[j + 1] == g.get_ntype_id(g.to_canonical_etype(metapath[j])[2])

    for i in range(traces.shape[0]):
        for j in range(traces.shape[1] - 1):
            assert g.has_edge_between(
                traces[i, j], traces[i, j+1], etype=metapath[j])
            if prob is not None and prob in g.edges[metapath[j]].data:
                p = F.asnumpy(g.edges[metapath[j]].data['p'])
                eids = g.edge_id(traces[i, j], traces[i, j+1], etype=metapath[j])
                assert p[eids] != 0

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU random walk not implemented")
def test_random_walk():
    g1 = dgl.heterograph({
        ('user', 'follow', 'user'): [(0, 1), (1, 2), (2, 0)]
        })
    g2 = dgl.heterograph({
        ('user', 'follow', 'user'): [(0, 1), (1, 2), (1, 3), (2, 0), (3, 0)]
        })
    g3 = dgl.heterograph({
        ('user', 'follow', 'user'): [(0, 1), (1, 2), (2, 0)],
        ('user', 'view', 'item'): [(0, 0), (1, 1), (2, 2)],
        ('item', 'viewed-by', 'user'): [(0, 0), (1, 1), (2, 2)]})
    g4 = dgl.heterograph({
        ('user', 'follow', 'user'): [(0, 1), (1, 2), (1, 3), (2, 0), (3, 0)],
        ('user', 'view', 'item'): [(0, 0), (0, 1), (1, 1), (2, 2), (3, 2), (3, 1)],
        ('item', 'viewed-by', 'user'): [(0, 0), (1, 0), (1, 1), (2, 2), (2, 3), (1, 3)]})

    g2.edata['p'] = F.tensor([3, 0, 3, 3, 3], dtype=F.float32)
    g4.edges['follow'].data['p'] = F.tensor([3, 0, 3, 3, 3], dtype=F.float32)
    g4.edges['viewed-by'].data['p'] = F.tensor([1, 1, 1, 1, 1, 1], dtype=F.float32)

    traces, ntypes = dgl.sampling.random_walk(g1, [0, 1, 2, 0, 1, 2], length=4)
    check_random_walk(g1, ['follow'] * 4, traces, ntypes)
    traces, ntypes = dgl.sampling.random_walk(g1, [0, 1, 2, 0, 1, 2], length=4, restart_prob=0.)
    check_random_walk(g1, ['follow'] * 4, traces, ntypes)
    traces, ntypes = dgl.sampling.random_walk(
        g1, [0, 1, 2, 0, 1, 2], length=4, restart_prob=F.zeros((4,), F.float32, F.cpu()))
    check_random_walk(g1, ['follow'] * 4, traces, ntypes)
    traces, ntypes = dgl.sampling.random_walk(
        g1, [0, 1, 2, 0, 1, 2], length=5,
        restart_prob=F.tensor([0, 0, 0, 0, 1], dtype=F.float32))
    check_random_walk(
        g1, ['follow'] * 4, F.slice_axis(traces, 1, 0, 5), F.slice_axis(ntypes, 0, 0, 5))
    assert (F.asnumpy(traces)[:, 5] == -1).all()

    traces, ntypes = dgl.sampling.random_walk(
        g2, [0, 1, 2, 3, 0, 1, 2, 3], length=4)
    check_random_walk(g2, ['follow'] * 4, traces, ntypes)

    traces, ntypes = dgl.sampling.random_walk(
        g2, [0, 1, 2, 3, 0, 1, 2, 3], length=4, prob='p')
    check_random_walk(g2, ['follow'] * 4, traces, ntypes, 'p')

    metapath = ['follow', 'view', 'viewed-by'] * 2
    traces, ntypes = dgl.sampling.random_walk(
        g3, [0, 1, 2, 0, 1, 2], metapath=metapath)
    check_random_walk(g3, metapath, traces, ntypes)

    metapath = ['follow', 'view', 'viewed-by'] * 2
    traces, ntypes = dgl.sampling.random_walk(
        g4, [0, 1, 2, 3, 0, 1, 2, 3], metapath=metapath)
    check_random_walk(g4, metapath, traces, ntypes)

    metapath = ['follow', 'view', 'viewed-by'] * 2
    traces, ntypes = dgl.sampling.random_walk(
        g4, [0, 1, 2, 3, 0, 1, 2, 3], metapath=metapath, prob='p')
    check_random_walk(g4, metapath, traces, ntypes, 'p')
    traces, ntypes = dgl.sampling.random_walk(
        g4, [0, 1, 2, 3, 0, 1, 2, 3], metapath=metapath, prob='p', restart_prob=0.)
    check_random_walk(g4, metapath, traces, ntypes, 'p')
    traces, ntypes = dgl.sampling.random_walk(
        g4, [0, 1, 2, 3, 0, 1, 2, 3], metapath=metapath, prob='p',
        restart_prob=F.zeros((6,), F.float32, F.cpu()))
    check_random_walk(g4, metapath, traces, ntypes, 'p')
    traces, ntypes = dgl.sampling.random_walk(
        g4, [0, 1, 2, 3, 0, 1, 2, 3], metapath=metapath + ['follow'], prob='p',
        restart_prob=F.tensor([0, 0, 0, 0, 0, 0, 1], F.float32))
    check_random_walk(g4, metapath, traces[:, :7], ntypes[:7], 'p')
    assert (F.asnumpy(traces[:, 7]) == -1).all()

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU pack traces not implemented")
def test_pack_traces():
    traces, types = (np.array(
        [[ 0,  1, -1, -1, -1, -1, -1],
         [ 0,  1,  1,  3,  0,  0,  0]], dtype='int64'),
        np.array([0, 0, 1, 0, 0, 1, 0], dtype='int64'))
    traces = F.zerocopy_from_numpy(traces)
    types = F.zerocopy_from_numpy(types)
    result = dgl.sampling.pack_traces(traces, types)
    assert F.array_equal(result[0], F.tensor([0, 1, 0, 1, 1, 3, 0, 0, 0], dtype=F.int64))
    assert F.array_equal(result[1], F.tensor([0, 0, 0, 0, 1, 0, 0, 1, 0], dtype=F.int64))
    assert F.array_equal(result[2], F.tensor([2, 7], dtype=F.int64))
    assert F.array_equal(result[3], F.tensor([0, 2], dtype=F.int64))

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU sample neighbors not implemented")
def test_sample_neighbors():
    g = dgl.graph([(1,0),(2,0),(3,0),(0,1),(2,1),(3,1),(0,2)],
            'user', 'follow')
    g.edata['prob'] = F.tensor([.5, .5, 0., .5, .5, 0., 1.], dtype=F.float32)
    g1 = dgl.bipartite([(0,0),(0,1),(1,2),(3,2)], 'user', 'play', 'game')
    g1.edata['prob'] = F.tensor([.8, .5, .5, .5], dtype=F.float32)
    g2 = dgl.bipartite([(2,0),(2,1),(2,2),(1,0),(1,3),(0,0)], 'game', 'liked-by', 'user')
    g2.edata['prob'] = F.tensor([.3, .5, .2, .5, .1, .1], dtype=F.float32)
    g3 = dgl.bipartite([(0,0),(1,0),(2,0),(3,0)], 'user', 'flips', 'coin')

    hg = dgl.hetero_from_relations([g, g1, g2, g3])

    def _test1(p, replace):
        for i in range(10):
            subg = dgl.sampling.sample_neighbors(g, [0, 1], 2, prob=p, replace=replace)
            assert subg.number_of_nodes() == 4
            assert subg.number_of_edges() == 4
            u, v = subg.edges()
            assert set(F.asnumpy(F.unique(v))) == {0, 1}
            assert F.array_equal(g.has_edges_between(u, v), F.ones((4,), dtype=F.int64))
            edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
            if not replace:
                # check no duplication
                assert len(edge_set) == 4
            if p is not None:
                assert not (3, 0) in edge_set
                assert not (3, 1) in edge_set
    _test1(None, True)   # w/ replacement, uniform
    _test1(None, False)  # w/o replacement, uniform
    _test1('prob', True)   # w/ replacement
    _test1('prob', False)  # w/o replacement

    def _test2(p, replace):  # fanout > #neighbors
        for i in range(10):
            subg = dgl.sampling.sample_neighbors(g, [0, 2], 2, prob=p, replace=replace)
            assert subg.number_of_nodes() == 4
            num_edges = 4 if replace else 3
            assert subg.number_of_edges() == num_edges
            u, v = subg.edges()
            assert set(F.asnumpy(F.unique(v))) == {0, 2}
            assert F.array_equal(g.has_edges_between(u, v), F.ones((num_edges,), dtype=F.int64))
            edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
            if not replace:
                # check no duplication
                assert len(edge_set) == num_edges
            if p is not None:
                assert not (3, 0) in edge_set
    _test2(None, True)   # w/ replacement, uniform
    _test2(None, False)  # w/o replacement, uniform
    _test2('prob', True)   # w/ replacement
    _test2('prob', False)  # w/o replacement

    def _test3(p, replace):
        for i in range(10):
            subg = dgl.sampling.sample_neighbors(hg, {'user' : [0,1], 'game' : 0}, 2, prob=p, replace=replace)
            assert len(subg.ntypes) == 3
            assert len(subg.etypes) == 4
            assert subg['follow'].number_of_edges() == 4
            assert subg['play'].number_of_edges() == 2 if replace else 1
            assert subg['liked-by'].number_of_edges() == 4 if replace else 3
            assert subg['flips'].number_of_edges() == 0

    _test3(None, True)   # w/ replacement, uniform
    _test3(None, False)  # w/o replacement, uniform
    _test3('prob', True)   # w/ replacement
    _test3('prob', False)  # w/o replacement


@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU sample neighbors not implemented")
def test_sample_neighbors_outedge():
    g = dgl.graph([(0,1),(0,2),(0,3),(1,0),(1,2),(1,3),(2,0)],
            'user', 'follow')
    g.edata['prob'] = F.tensor([.5, .5, 0., .5, .5, 0., 1.], dtype=F.float32)
    g1 = dgl.bipartite([(0,0),(1,0),(2,1),(2,3)], 'game', 'play', 'user')
    g1.edata['prob'] = F.tensor([.8, .5, .5, .5], dtype=F.float32)
    g2 = dgl.bipartite([(0,2),(1,2),(2,2),(0,1),(3,1),(0,0)], 'user', 'liked-by', 'game')
    g2.edata['prob'] = F.tensor([.3, .5, .2, .5, .1, .1], dtype=F.float32)
    g3 = dgl.bipartite([(0,0),(0,1),(0,2),(0,3)], 'coin', 'flips', 'user')

    hg = dgl.hetero_from_relations([g, g1, g2, g3])

    def _test1(p, replace):
        for i in range(10):
            subg = dgl.sampling.sample_neighbors(g, [0, 1], 2, prob=p, replace=replace, edge_dir='out')
            assert subg.number_of_nodes() == 4
            assert subg.number_of_edges() == 4
            u, v = subg.edges()
            assert set(F.asnumpy(F.unique(u))) == {0, 1}
            assert F.array_equal(g.has_edges_between(u, v), F.ones((4,), dtype=F.int64))
            edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
            if not replace:
                # check no duplication
                assert len(edge_set) == 4
            if p is not None:
                assert not (0, 3) in edge_set
                assert not (1, 3) in edge_set
    _test1(None, True)   # w/ replacement, uniform
    _test1(None, False)  # w/o replacement, uniform
    _test1('prob', True)   # w/ replacement
    _test1('prob', False)  # w/o replacement

    def _test2(p, replace):  # fanout > #neighbors
        for i in range(10):
            subg = dgl.sampling.sample_neighbors(g, [0, 2], 2, prob=p, replace=replace, edge_dir='out')
            assert subg.number_of_nodes() == 4
            num_edges = 4 if replace else 3
            assert subg.number_of_edges() == num_edges
            u, v = subg.edges()
            assert set(F.asnumpy(F.unique(u))) == {0, 2}
            assert F.array_equal(g.has_edges_between(u, v), F.ones((num_edges,), dtype=F.int64))
            edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
            if not replace:
                # check no duplication
                assert len(edge_set) == num_edges
            if p is not None:
                assert not (0, 3) in edge_set
    _test2(None, True)   # w/ replacement, uniform
    _test2(None, False)  # w/o replacement, uniform
    _test2('prob', True)   # w/ replacement
    _test2('prob', False)  # w/o replacement

    def _test3(p, replace):
        for i in range(10):
            subg = dgl.sampling.sample_neighbors(hg, {'user' : [0,1], 'game' : 0}, 2, prob=p, replace=replace, edge_dir='out')
            assert len(subg.ntypes) == 3
            assert len(subg.etypes) == 4
            assert subg['follow'].number_of_edges() == 4
            assert subg['play'].number_of_edges() == 2 if replace else 1
            assert subg['liked-by'].number_of_edges() == 4 if replace else 3
            assert subg['flips'].number_of_edges() == 0

    _test3(None, True)   # w/ replacement, uniform
    _test3(None, False)  # w/o replacement, uniform
    _test3('prob', True)   # w/ replacement
    _test3('prob', False)  # w/o replacement


@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU sample neighbors not implemented")
def test_sample_neighbors_topk():
    g = dgl.graph([(1,0),(2,0),(3,0),(0,1),(2,1),(3,1),(0,2)],
            'user', 'follow')
    g.edata['weight'] = F.tensor([.5, .3, 0., -5., 22., 0., 1.], dtype=F.float32)
    g1 = dgl.bipartite([(0,0),(0,1),(1,2),(3,2)], 'user', 'play', 'game')
    g1.edata['weight'] = F.tensor([.8, .5, .4, .5], dtype=F.float32)
    g2 = dgl.bipartite([(2,0),(2,1),(2,2),(1,0),(1,3),(0,0)], 'game', 'liked-by', 'user')
    g2.edata['weight'] = F.tensor([.3, .5, .2, .5, .1, .1], dtype=F.float32)
    g3 = dgl.bipartite([(0,0),(1,0),(2,0),(3,0)], 'user', 'flips', 'coin')
    g3.edata['weight'] = F.tensor([10, 2, 13, -1], dtype=F.float32)

    hg = dgl.hetero_from_relations([g, g1, g2, g3])

    def _test1():
        subg = dgl.sampling.sample_neighbors_topk(g, [0, 1], 2, 'weight')
        assert subg.number_of_nodes() == 4
        assert subg.number_of_edges() == 4
        u, v = subg.edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert edge_set == {(2,0),(1,0),(2,1),(3,1)}
    _test1()

    def _test2():  # k > #neighbors
        subg = dgl.sampling.sample_neighbors_topk(g, [0, 2], 2, 'weight')
        assert subg.number_of_nodes() == 4
        assert subg.number_of_edges() == 3
        u, v = subg.edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert edge_set == {(2,0),(1,0),(0,2)}
    _test2()

    def _test3():
        subg = dgl.sampling.sample_neighbors_topk(hg, {'user' : [0,1], 'game' : 0}, 2, 'weight')
        assert len(subg.ntypes) == 3
        assert len(subg.etypes) == 4
        u, v = subg['follow'].edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert edge_set == {(2,0),(1,0),(2,1),(3,1)}
        u, v = subg['play'].edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert edge_set == {(0,0)}
        u, v = subg['liked-by'].edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert edge_set == {(2,0),(2,1),(1,0)}
        assert subg['flips'].number_of_edges() == 0
    _test3()

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU sample neighbors not implemented")
def test_sample_neighbors_topk_outedge():
    g = dgl.graph([(0,1),(0,2),(0,3),(1,0),(1,2),(1,3),(2,0)],
            'user', 'follow')
    g.edata['weight'] = F.tensor([.5, .3, 0., -5., 22., 0., 1.], dtype=F.float32)
    g1 = dgl.bipartite([(0,0),(1,0),(2,1),(2,3)], 'game', 'play', 'user')
    g1.edata['weight'] = F.tensor([.8, .5, .4, .5], dtype=F.float32)
    g2 = dgl.bipartite([(0,2),(1,2),(2,2),(0,1),(3,1),(0,0)], 'user', 'liked-by', 'game')
    g2.edata['weight'] = F.tensor([.3, .5, .2, .5, .1, .1], dtype=F.float32)
    g3 = dgl.bipartite([(0,0),(0,1),(0,2),(0,3)], 'coin', 'flips', 'user')
    g3.edata['weight'] = F.tensor([10, 2, 13, -1], dtype=F.float32)

    hg = dgl.hetero_from_relations([g, g1, g2, g3])

    def _test1():
        subg = dgl.sampling.sample_neighbors_topk(g, [0, 1], 2, 'weight', edge_dir='out')
        assert subg.number_of_nodes() == 4
        assert subg.number_of_edges() == 4
        u, v = subg.edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert edge_set == {(0,2),(0,1),(1,2),(1,3)}
    _test1()

    def _test2():  # k > #neighbors
        subg = dgl.sampling.sample_neighbors_topk(g, [0, 2], 2, 'weight', edge_dir='out')
        assert subg.number_of_nodes() == 4
        assert subg.number_of_edges() == 3
        u, v = subg.edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert edge_set == {(0,2),(0,1),(2,0)}
    _test2()

    def _test3():
        subg = dgl.sampling.sample_neighbors_topk(hg, {'user' : [0,1], 'game' : 0}, 2, 'weight', edge_dir='out')
        assert len(subg.ntypes) == 3
        assert len(subg.etypes) == 4
        u, v = subg['follow'].edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert edge_set == {(0,2),(0,1),(1,2),(1,3)}
        u, v = subg['play'].edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert edge_set == {(0,0)}
        u, v = subg['liked-by'].edges()
        edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
        assert edge_set == {(0,2),(1,2),(0,1)}
        assert subg['flips'].number_of_edges() == 0
    _test3()

if __name__ == '__main__':
    test_random_walk()
    test_pack_traces()
    test_sample_neighbors()
    test_sample_neighbors_outedge()
    test_sample_neighbors_topk()
    test_sample_neighbors_topk_outedge()
