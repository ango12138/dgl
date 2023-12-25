import unittest

import backend as F
import dgl.graphbolt as gb
import pytest
import torch

from .. import gb_test_utils


@unittest.skipIf(
    F._default_context_str == "cpu",
    reason="Tests for pinned memory are only meaningful on GPU.",
)
@pytest.mark.parametrize(
    "indptr_dtype",
    [torch.int32, torch.int64],
)
@pytest.mark.parametrize(
    "indices_dtype",
    [torch.int8, torch.int16, torch.int32, torch.int64],
)
@pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("is_pinned", [False, True])
def test_index_select_csc(indptr_dtype, indices_dtype, idtype, is_pinned):
    """Original graph in COO:
    1   0   1   0   1   0
    1   0   0   1   0   1
    0   1   0   1   0   0
    0   1   0   0   1   0
    1   0   0   0   0   1
    0   0   1   0   1   0
    """
    indptr = torch.tensor([0, 3, 5, 7, 9, 12, 14], dtype=indptr_dtype)
    indices = torch.tensor(
        [0, 1, 4, 2, 3, 0, 5, 1, 2, 0, 3, 5, 1, 4], dtype=indices_dtype
    )
    index = torch.tensor([0, 5, 3], dtype=idtype)

    cpu_indptr, cpu_indices = torch.ops.graphbolt.index_select_csc(
        indptr, indices, index
    )
    if is_pinned:
        indptr = indptr.pin_memory()
        indices = indices.pin_memory()
    else:
        indptr = indptr.cuda()
        indices = indices.cuda()
    index = index.cuda()

    gpu_indptr, gpu_indices = torch.ops.graphbolt.index_select_csc(
        indptr, indices, index
    )

    assert not cpu_indptr.is_cuda
    assert not cpu_indices.is_cuda

    assert gpu_indptr.is_cuda
    assert gpu_indices.is_cuda

    assert torch.equal(cpu_indptr, gpu_indptr.cpu())
    assert torch.equal(cpu_indices, gpu_indices.cpu())


def test_InSubgraphSampler_node_pairs_homo():
    """Original graph in COO:
    1   0   1   0   1   0
    1   0   0   1   0   1
    0   1   0   1   0   0
    0   1   0   0   1   0
    1   0   0   0   0   1
    0   0   1   0   1   0
    """
    indptr = torch.LongTensor([0, 3, 5, 7, 9, 12, 14])
    indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 5, 1, 2, 0, 3, 5, 1, 4])
    graph = gb.fused_csc_sampling_graph(indptr, indices)

    seed_nodes = torch.LongTensor([0, 5, 3])
    item_set = gb.ItemSet(seed_nodes, names="seed_nodes")
    batch_size = 1
    item_sampler = gb.ItemSampler(item_set, batch_size=batch_size)

    in_subgraph_sampler = gb.InSubgraphSampler(
        item_sampler, graph, output_cscformat=False
    )

    it = iter(in_subgraph_sampler)

    def original_node_pairs(minibatch):
        sampled_subgraph = minibatch.sampled_subgraphs[0]
        _src = [
            sampled_subgraph.original_row_node_ids[id]
            for id in sampled_subgraph.node_pairs[0]
        ]
        _dst = [
            sampled_subgraph.original_column_node_ids[id]
            for id in sampled_subgraph.node_pairs[1]
        ]
        return _src, _dst

    mn = next(it)
    assert torch.equal(mn.seed_nodes, torch.LongTensor([0]))
    assert original_node_pairs(mn) == ([0, 1, 4], [0, 0, 0])

    mn = next(it)
    assert torch.equal(mn.seed_nodes, torch.LongTensor([5]))
    assert original_node_pairs(mn) == ([1, 4], [5, 5])

    mn = next(it)
    assert torch.equal(mn.seed_nodes, torch.LongTensor([3]))
    assert original_node_pairs(mn) == ([1, 2], [3, 3])


def test_InSubgraphSampler_node_pairs_hetero():
    """Original graph in COO:
    1   0   1   0   1   0
    1   0   0   1   0   1
    0   1   0   1   0   0
    0   1   0   0   1   0
    1   0   0   0   0   1
    0   0   1   0   1   0
    node_type_0: [0, 1, 2]
    node_type_1: [3, 4, 5]
    edge_type_0: node_type_0 -> node_type_0
    edge_type_1: node_type_0 -> node_type_1
    edge_type_2: node_type_1 -> node_type_0
    edge_type_3: node_type_1 -> node_type_1
    """
    ntypes = {
        "N0": 0,
        "N1": 1,
    }
    etypes = {
        "N0:R0:N0": 0,
        "N0:R1:N1": 1,
        "N1:R2:N0": 2,
        "N1:R3:N1": 3,
    }
    indptr = torch.LongTensor([0, 3, 5, 7, 9, 12, 14])
    indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 5, 1, 2, 0, 3, 5, 1, 4])
    node_type_offset = torch.LongTensor([0, 3, 6])
    type_per_edge = torch.LongTensor([0, 0, 2, 0, 2, 0, 2, 1, 1, 1, 3, 3, 1, 3])
    graph = gb.fused_csc_sampling_graph(
        csc_indptr=indptr,
        indices=indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=ntypes,
        edge_type_to_id=etypes,
    )

    item_set = gb.ItemSetDict(
        {
            "N0": gb.ItemSet(torch.LongTensor([1, 0, 2]), names="seed_nodes"),
            "N1": gb.ItemSet(torch.LongTensor([0, 2, 1]), names="seed_nodes"),
        }
    )
    batch_size = 2
    item_sampler = gb.ItemSampler(item_set, batch_size=batch_size)

    in_subgraph_sampler = gb.InSubgraphSampler(
        item_sampler, graph, output_cscformat=False
    )

    it = iter(in_subgraph_sampler)

    mn = next(it)
    assert torch.equal(mn.seed_nodes["N0"], torch.LongTensor([1, 0]))
    expected_node_pairs = {
        "N0:R0:N0": (torch.LongTensor([2, 1, 0]), torch.LongTensor([0, 1, 1])),
        "N0:R1:N1": (torch.LongTensor([]), torch.LongTensor([])),
        "N1:R2:N0": (torch.LongTensor([0, 1]), torch.LongTensor([0, 1])),
        "N1:R3:N1": (torch.LongTensor([]), torch.LongTensor([])),
    }
    for etype, pairs in mn.sampled_subgraphs[0].node_pairs.items():
        assert torch.equal(pairs[0], expected_node_pairs[etype][0])
        assert torch.equal(pairs[1], expected_node_pairs[etype][1])

    mn = next(it)
    assert mn.seed_nodes == {
        "N0": torch.LongTensor([2]),
        "N1": torch.LongTensor([0]),
    }
    expected_node_pairs = {
        "N0:R0:N0": (torch.LongTensor([1]), torch.LongTensor([0])),
        "N0:R1:N1": (torch.LongTensor([2, 0]), torch.LongTensor([0, 0])),
        "N1:R2:N0": (torch.LongTensor([1]), torch.LongTensor([0])),
        "N1:R3:N1": (torch.LongTensor([]), torch.LongTensor([])),
    }
    for etype, pairs in mn.sampled_subgraphs[0].node_pairs.items():
        assert torch.equal(pairs[0], expected_node_pairs[etype][0])
        assert torch.equal(pairs[1], expected_node_pairs[etype][1])

    mn = next(it)
    assert torch.equal(mn.seed_nodes["N1"], torch.LongTensor([2, 1]))
    expected_node_pairs = {
        "N0:R0:N0": (torch.LongTensor([]), torch.LongTensor([])),
        "N0:R1:N1": (torch.LongTensor([0, 1]), torch.LongTensor([0, 1])),
        "N1:R2:N0": (torch.LongTensor([]), torch.LongTensor([])),
        "N1:R3:N1": (torch.LongTensor([1, 2, 0]), torch.LongTensor([0, 1, 1])),
    }
    for etype, pairs in mn.sampled_subgraphs[0].node_pairs.items():
        assert torch.equal(pairs[0], expected_node_pairs[etype][0])
        assert torch.equal(pairs[1], expected_node_pairs[etype][1])


def test_InSubgraphSampler_homo():
    """Original graph in COO:
    1   0   1   0   1   0
    1   0   0   1   0   1
    0   1   0   1   0   0
    0   1   0   0   1   0
    1   0   0   0   0   1
    0   0   1   0   1   0
    """
    indptr = torch.LongTensor([0, 3, 5, 7, 9, 12, 14])
    indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 5, 1, 2, 0, 3, 5, 1, 4])
    graph = gb.fused_csc_sampling_graph(indptr, indices)

    seed_nodes = torch.LongTensor([0, 5, 3])
    item_set = gb.ItemSet(seed_nodes, names="seed_nodes")
    batch_size = 1
    item_sampler = gb.ItemSampler(item_set, batch_size=batch_size)

    in_subgraph_sampler = gb.InSubgraphSampler(
        item_sampler, graph, output_cscformat=True
    )

    it = iter(in_subgraph_sampler)

    def original_indices(minibatch):
        sampled_subgraph = minibatch.sampled_subgraphs[0]
        _indices = sampled_subgraph.original_row_node_ids[
            sampled_subgraph.node_pairs.indices
        ]
        return _indices

    mn = next(it)
    assert torch.equal(mn.seed_nodes, torch.LongTensor([0]))
    assert torch.equal(
        mn.sampled_subgraphs[0].node_pairs.indptr, torch.tensor([0, 3])
    )
    assert torch.equal(original_indices(mn), torch.tensor([0, 1, 4]))

    mn = next(it)
    assert torch.equal(mn.seed_nodes, torch.LongTensor([5]))
    assert torch.equal(
        mn.sampled_subgraphs[0].node_pairs.indptr, torch.tensor([0, 2])
    )
    assert torch.equal(original_indices(mn), torch.tensor([1, 4]))

    mn = next(it)
    assert torch.equal(mn.seed_nodes, torch.LongTensor([3]))
    assert torch.equal(
        mn.sampled_subgraphs[0].node_pairs.indptr, torch.tensor([0, 2])
    )
    assert torch.equal(original_indices(mn), torch.tensor([1, 2]))


def test_InSubgraphSampler_hetero():
    """Original graph in COO:
    1   0   1   0   1   0
    1   0   0   1   0   1
    0   1   0   1   0   0
    0   1   0   0   1   0
    1   0   0   0   0   1
    0   0   1   0   1   0
    node_type_0: [0, 1, 2]
    node_type_1: [3, 4, 5]
    edge_type_0: node_type_0 -> node_type_0
    edge_type_1: node_type_0 -> node_type_1
    edge_type_2: node_type_1 -> node_type_0
    edge_type_3: node_type_1 -> node_type_1
    """
    ntypes = {
        "N0": 0,
        "N1": 1,
    }
    etypes = {
        "N0:R0:N0": 0,
        "N0:R1:N1": 1,
        "N1:R2:N0": 2,
        "N1:R3:N1": 3,
    }
    indptr = torch.LongTensor([0, 3, 5, 7, 9, 12, 14])
    indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 5, 1, 2, 0, 3, 5, 1, 4])
    node_type_offset = torch.LongTensor([0, 3, 6])
    type_per_edge = torch.LongTensor([0, 0, 2, 0, 2, 0, 2, 1, 1, 1, 3, 3, 1, 3])
    graph = gb.fused_csc_sampling_graph(
        csc_indptr=indptr,
        indices=indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=ntypes,
        edge_type_to_id=etypes,
    )

    item_set = gb.ItemSetDict(
        {
            "N0": gb.ItemSet(torch.LongTensor([1, 0, 2]), names="seed_nodes"),
            "N1": gb.ItemSet(torch.LongTensor([0, 2, 1]), names="seed_nodes"),
        }
    )
    batch_size = 2
    item_sampler = gb.ItemSampler(item_set, batch_size=batch_size)

    in_subgraph_sampler = gb.InSubgraphSampler(
        item_sampler, graph, output_cscformat=True
    )

    it = iter(in_subgraph_sampler)

    mn = next(it)
    assert torch.equal(mn.seed_nodes["N0"], torch.LongTensor([1, 0]))
    expected_node_pairs = {
        "N0:R0:N0": gb.CSCFormatBase(
            indptr=torch.LongTensor([0, 1, 3]),
            indices=torch.LongTensor([2, 1, 0]),
        ),
        "N0:R1:N1": gb.CSCFormatBase(
            indptr=torch.LongTensor([0]), indices=torch.LongTensor([])
        ),
        "N1:R2:N0": gb.CSCFormatBase(
            indptr=torch.LongTensor([0, 1, 2]), indices=torch.LongTensor([0, 1])
        ),
        "N1:R3:N1": gb.CSCFormatBase(
            indptr=torch.LongTensor([0]), indices=torch.LongTensor([])
        ),
    }
    for etype, pairs in mn.sampled_subgraphs[0].node_pairs.items():
        assert torch.equal(pairs.indices, expected_node_pairs[etype].indices)
        assert torch.equal(pairs.indptr, expected_node_pairs[etype].indptr)

    mn = next(it)
    assert mn.seed_nodes == {
        "N0": torch.LongTensor([2]),
        "N1": torch.LongTensor([0]),
    }
    expected_node_pairs = {
        "N0:R0:N0": gb.CSCFormatBase(
            indptr=torch.LongTensor([0, 1]), indices=torch.LongTensor([1])
        ),
        "N0:R1:N1": gb.CSCFormatBase(
            indptr=torch.LongTensor([0, 2]), indices=torch.LongTensor([2, 0])
        ),
        "N1:R2:N0": gb.CSCFormatBase(
            indptr=torch.LongTensor([0, 1]), indices=torch.LongTensor([1])
        ),
        "N1:R3:N1": gb.CSCFormatBase(
            indptr=torch.LongTensor([0, 0]), indices=torch.LongTensor([])
        ),
    }
    for etype, pairs in mn.sampled_subgraphs[0].node_pairs.items():
        assert torch.equal(pairs.indices, expected_node_pairs[etype].indices)
        assert torch.equal(pairs.indptr, expected_node_pairs[etype].indptr)

    mn = next(it)
    assert torch.equal(mn.seed_nodes["N1"], torch.LongTensor([2, 1]))
    expected_node_pairs = {
        "N0:R0:N0": gb.CSCFormatBase(
            indptr=torch.LongTensor([0]), indices=torch.LongTensor([])
        ),
        "N0:R1:N1": gb.CSCFormatBase(
            indptr=torch.LongTensor([0, 1, 2]), indices=torch.LongTensor([0, 1])
        ),
        "N1:R2:N0": gb.CSCFormatBase(
            indptr=torch.LongTensor([0]), indices=torch.LongTensor([])
        ),
        "N1:R3:N1": gb.CSCFormatBase(
            indptr=torch.LongTensor([0, 1, 3]),
            indices=torch.LongTensor([1, 2, 0]),
        ),
    }
    for etype, pairs in mn.sampled_subgraphs[0].node_pairs.items():
        assert torch.equal(pairs.indices, expected_node_pairs[etype].indices)
        assert torch.equal(pairs.indptr, expected_node_pairs[etype].indptr)
