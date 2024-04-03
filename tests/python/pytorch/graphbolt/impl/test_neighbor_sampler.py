import unittest
from functools import partial

import backend as F

import dgl
import dgl.graphbolt as gb
import pytest
import torch


def get_hetero_graph():
    # COO graph:
    # [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    # [2, 4, 2, 3, 0, 1, 1, 0, 0, 1]
    # [1, 1, 1, 1, 0, 0, 0, 0, 0] - > edge type.
    # num_nodes = 5, num_n1 = 2, num_n2 = 3
    ntypes = {"n1": 0, "n2": 1, "n3": 2}
    etypes = {"n2:e1:n3": 0, "n3:e2:n2": 1}
    indptr = torch.LongTensor([0, 0, 2, 4, 6, 8, 10])
    indices = torch.LongTensor([3, 5, 3, 4, 1, 2, 2, 1, 1, 2])
    type_per_edge = torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    edge_attributes = {
        "weight": torch.FloatTensor(
            [2.5, 0, 8.4, 0, 0.4, 1.2, 2.5, 0, 8.4, 0.5]
        ),
        "mask": torch.BoolTensor([1, 0, 1, 0, 1, 1, 1, 0, 1, 1]),
    }
    node_type_offset = torch.LongTensor([0, 1, 3, 6])
    return gb.fused_csc_sampling_graph(
        indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=ntypes,
        edge_type_to_id=etypes,
        edge_attributes=edge_attributes,
    )


@unittest.skipIf(F._default_context_str != "gpu", reason="Enabled only on GPU.")
@pytest.mark.parametrize("hetero", [False, True])
@pytest.mark.parametrize("prob_name", [None, "weight", "mask"])
@pytest.mark.parametrize("sorted", [False, True])
def test_NeighborSampler_GraphFetch(hetero, prob_name, sorted):
    if sorted:
        items = torch.arange(3)
    else:
        items = torch.tensor([2, 0, 1])
    names = "seed_nodes"
    itemset = gb.ItemSet(items, names=names)
    graph = get_hetero_graph().to(F.ctx())
    if hetero:
        itemset = gb.ItemSetDict({"n3": itemset})
    else:
        graph.type_per_edge = None
    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    fanout = torch.LongTensor([2])
    datapipe = item_sampler.map(gb.SubgraphSampler._preprocess)
    datapipe = datapipe.map(
        partial(gb.NeighborSampler._prepare, graph.node_type_to_id)
    )
    sample_per_layer = gb.SamplePerLayer(
        datapipe, graph.sample_neighbors, fanout, False, prob_name
    )
    compact_per_layer = sample_per_layer.compact_per_layer(True)
    gb.seed(123)
    expected_results = list(compact_per_layer)
    datapipe = gb.FetchInsubgraphData(datapipe, sample_per_layer)
    datapipe = datapipe.wait_future()
    datapipe = gb.SamplePerLayerFromFetchedSubgraph(datapipe, sample_per_layer)
    datapipe = datapipe.compact_per_layer(True)
    gb.seed(123)
    new_results = list(datapipe)
    assert len(expected_results) == len(new_results)
    for a, b in zip(expected_results, new_results):
        assert repr(a) == repr(b)


@pytest.mark.parametrize("layer_dependency", [False, True])
@pytest.mark.parametrize("overlap_graph_fetch", [False, True])
def test_labor_dependent_minibatching(layer_dependency, overlap_graph_fetch):
    num_edges = 200
    csc_indptr = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64),
            torch.ones(num_edges + 1, dtype=torch.int64) * num_edges,
        )
    )
    indices = torch.arange(1, num_edges + 1)
    graph = gb.fused_csc_sampling_graph(
        csc_indptr.int(),
        indices.int(),
    ).to(F.ctx())
    torch.random.set_rng_state(torch.manual_seed(123).get_state())
    batch_dependency = 100
    itemset = gb.ItemSet(
        torch.zeros(batch_dependency + 1).int(), names="seed_nodes"
    )
    datapipe = gb.ItemSampler(itemset, batch_size=1).copy_to(F.ctx())
    fanouts = [5, 5]
    datapipe = datapipe.sample_layer_neighbor(
        graph,
        fanouts,
        layer_dependency=layer_dependency,
        batch_dependency=batch_dependency,
    )
    dataloader = gb.DataLoader(
        datapipe, overlap_graph_fetch=overlap_graph_fetch
    )
    res = list(dataloader)
    assert len(res) == batch_dependency + 1
    if layer_dependency:
        assert torch.equal(
            res[0].input_nodes,
            res[0].sampled_subgraphs[1].original_row_node_ids,
        )
    else:
        assert res[0].input_nodes.size(0) > res[0].sampled_subgraphs[
            1
        ].original_row_node_ids.size(0)
    delta = 0
    for i in range(batch_dependency):
        res_current = (
            res[i].sampled_subgraphs[-1].original_row_node_ids.tolist()
        )
        res_next = (
            res[i + 1].sampled_subgraphs[-1].original_row_node_ids.tolist()
        )
        intersect_len = len(set(res_current).intersection(set(res_next)))
        assert intersect_len >= fanouts[-1]
        delta += 1 + fanouts[-1] - intersect_len
    assert delta >= fanouts[-1]
