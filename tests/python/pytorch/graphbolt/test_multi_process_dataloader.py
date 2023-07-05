import backend as F
import dgl
import dgl.graphbolt
import gb_test_utils
import torch


def test_DataLoader():
    N = 40
    B = 4
    itemset = dgl.graphbolt.ItemSet(torch.arange(N))
    # TODO(BarclayII): temporarily using DGLGraph.  Should test using
    # GraphBolt's storage as well once issue #5953 is resolved.
    graph = dgl.add_reverse_edges(dgl.rand_graph(200, 6000))
    features = dgl.graphbolt.feature_store.TorchBasedFeatureStore(
        torch.randn(200, 4)
    )
    labels = dgl.graphbolt.feature_store.TorchBasedFeatureStore(
        torch.randint(0, 10, (200,))
    )

    def sampler_func(data):
        seeds = data
        sampler = dgl.dataloading.NeighborSampler([2, 2])
        return sampler.sample(graph, seeds)

    def fetch_func(data):
        input_nodes, output_nodes, adjs = data
        input_features = features.read(input_nodes)
        output_labels = labels.read(output_nodes)
        return input_features, output_labels, adjs

    minibatch_sampler = dgl.graphbolt.MinibatchSampler(itemset, batch_size=B)
    subgraph_sampler = dgl.graphbolt.SubgraphSampler(
        minibatch_sampler,
        sampler_func,
    )
    feature_fetcher = dgl.graphbolt.FeatureFetcher(subgraph_sampler, fetch_func)
    device_transferrer = dgl.graphbolt.CopyTo(feature_fetcher, F.ctx())

    dataloader = dgl.graphbolt.MultiProcessDataLoader(
        device_transferrer,
        num_workers=4,
    )
    assert len(list(dataloader)) == N // B
