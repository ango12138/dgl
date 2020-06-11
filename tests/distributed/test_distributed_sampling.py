import dgl
import unittest
import os
from dgl.data import CitationGraphDataset
from dgl.distributed.sampling import sample_neighbors
from dgl.distributed import partition_graph, load_partition, GraphPartitionBook
import sys
import multiprocessing as mp
import numpy as np
import backend as F
import time
from utils import get_local_usable_addr


def myexcepthook(exctype, value, traceback):
    for p in mp.active_children():
        p.terminate()
    # raise Exception("11111111111")


class MochDistGraph:

    def __init__(self, partition_book, num_nodes):
        self.partition_book = partition_book
        self.total_num_nodes = num_nodes

    def get_partition_book(self):
        return self.partition_book


class MockServerState:

    def __init__(self, g, rank, partition_book):
        self.rank = rank
        self.partition_book = partition_book
        self.hgraph = dgl.as_heterograph(g)

    @property
    def graph(self):
        return self.hgraph

    @property
    def total_num_nodes(self):
        return self.hgraph.number_of_nodes()

    @property
    def total_num_edges(self):
        return self.hgraph.number_of_edges()


def start_server(rank):
    import dgl
    part_g, node_feats, edge_feats, meta = load_partition(
        '/tmp/test.json', rank)
    num_nodes, num_edges, node_map, edge_map, num_partitions = meta
    gpb = GraphPartitionBook(part_id=rank,
                             num_parts=num_partitions,
                             node_map=node_map,
                             edge_map=edge_map,
                             part_graph=part_g)
    server_state = MockServerState(part_g, rank, gpb)
    dgl.distributed.start_server(server_id=rank,
                                 ip_config='rpc_ip_config.txt',
                                 num_clients=1,
                                 server_state=server_state)


def start_client(rank):
    import dgl
    dgl.distributed.connect_to_server(ip_config='rpc_ip_config.txt')

    part_g, node_feats, edge_feats, meta = load_partition(
        '/tmp/test.json', rank)
    num_nodes, num_edges, node_map, edge_map, num_partitions = meta
    gpb = GraphPartitionBook(part_id=rank,
                             num_parts=num_partitions,
                             node_map=node_map,
                             edge_map=edge_map,
                             part_graph=part_g)

    g = MochDistGraph(gpb, num_nodes)
    print("Pre sample")
    results = sample_neighbors(g, [0, 10, 99], 3)
    print("after sample")
    dgl.distributed.shutdown_servers()
    dgl.distributed.finalize_client()
    return results


@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
def test_rpc_sampling():
    num_server = 3
    ip_config = open("rpc_ip_config.txt", "w")
    ip_addr = get_local_usable_addr()
    ip_config.write('%s 3\n' % ip_addr)
    ip_config.close()
    # sys.excepthook = myexcepthook
    # partition graph
    g = CitationGraphDataset("cora")[0]
    g.readonly()
    num_parts = num_server
    num_hops = 2

    partition_graph(g, 'test', num_parts, '/tmp',
                    num_hops=num_hops, part_method='metis')

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(i,))
        p.start()
        # time.sleep(1)
        pserver_list.append(p)

    sampled_graph = start_client(0)
    print("Done sampling")
    for p in pserver_list:
        p.join()

    src, dst = sampled_graph.edges()
    assert sampled_graph.number_of_nodes() == g.number_of_nodes()
    assert np.all(F.asnumpy(g.has_edges_between(src, dst).bool()))
    eids = g.edge_ids(src, dst)
    assert np.array_equal(
        F.asnumpy(sampled_graph.edata[dgl.EID]), F.asnumpy(eids))


if __name__ == "__main__":
    test_rpc_sampling()
