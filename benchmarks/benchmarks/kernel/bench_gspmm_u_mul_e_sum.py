import time
import dgl
import torch

from .. import utils

def calc_gflops(graph, feat_size, num_heads, time):
    return round(2 * graph.num_edges() * feat_size / 1000000000 / time, 2)  # count both mul and add

@utils.benchmark('flops', timeout=600)
@utils.parametrize('graph', ['ogbn-arxiv', 'reddit', 'ogbn-proteins'])
@utils.parametrize('feat_size', [4, 32, 256])
@utils.parametrize('num_heads', [0, 1, 4])
def track_flops(graph, feat_size, num_heads):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph, format='csc').to(device)
    if num_heads == 0:
        x = torch.randn(graph.num_nodes(), feat_size, device=device)
        w = torch.randn(graph.num_edges(), feat_size, device=device)
    else:
        x = torch.randn(graph.num_nodes(), num_heads, feat_size // num_heads, device=device)
        w = torch.randn(graph.num_edges(), num_heads, 1, device=device)

    # dry run
    for i in range(3):
        y = dgl.ops.u_mul_e_sum(graph, x, w)

    # timing
    accum = 0.
    for i in range(10):
        with utils.TorchOpTimer(device) as timer:
            y = dgl.ops.u_mul_e_sum(graph, x, w)
        accum += timer.time

    return calc_gflops(graph, feat_size, num_heads, accum / 10)
