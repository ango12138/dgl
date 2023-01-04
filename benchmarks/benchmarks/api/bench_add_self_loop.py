import time

import numpy as np
import torch

import dgl
import dgl.function as fn

from .. import utils


@utils.benchmark("time")
@utils.parametrize("graph_name", ["cora", "livejournal"])
@utils.parametrize("format", ["coo"])
def track_time(graph_name, format):
    print(f"---------------~~~~ {dgl.__version__}, {torch.__version__}")
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, format)
    graph = graph.to(device)

    # dry run
    for i in range(3):
        g = graph.add_self_loop()

    # timing

    with utils.Timer() as t:
        for i in range(3):
            edges = graph.add_self_loop()

    return t.elapsed_secs / 3
