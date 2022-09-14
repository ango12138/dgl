import argparse

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy

import dgl
from dgl.data.rdf import AIFBDataset
from dgl.contrib.cugraph.nn.conv.relgraphconv_ops import RelGraphConvOps

class RGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, h_dim)
        # two-layer RGCN
        self.conv1 = RelGraphConvOps(h_dim, h_dim, num_rels,
                                     regularizer='basis', num_bases=num_bases, self_loop=False)
        self.conv2 = RelGraphConvOps(h_dim, out_dim, num_rels,
                                     regularizer='basis', num_bases=num_bases, self_loop=False)

    def forward(self, g):
        x = self.emb.weight
        h = F.relu(self.conv1(g, x, g.edata[dgl.ETYPE], norm=g.edata['norm']))
        h = self.conv2(g, h, g.edata[dgl.ETYPE], norm=g.edata['norm'])
        return h

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN for entity classification')
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose data information.")
    args = parser.parse_args()

    # load and preprocess dataset
    data = AIFBDataset()
    device = th.device('cuda')
    hg = data[0]
    hg = hg.to(device)

    if args.verbose:
        print(f'# nodes: {hg.num_nodes()}')
        print(f'# destination nodes: {hg.num_dst_nodes()}')
        print(f'# source nodes: {hg.num_src_nodes()}')
        print(f'# edges: {hg.num_edges()}')
        print(f'# Node types: {len(hg.ntypes)}')
        print(f'# Canonical edge types: {len(hg.etypes)}')
        print(f'# Unique edge type names: {len(set(hg.etypes))}')

    num_rels = len(hg.canonical_etypes)
    category = data.predict_category
    labels = hg.nodes[category].data.pop('labels')
    train_mask = hg.nodes[category].data.pop('train_mask')
    test_mask = hg.nodes[category].data.pop('test_mask')
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    # calculate normalization weight for each edge, and find target category and node id
    for cetype in hg.canonical_etypes:
        hg.edges[cetype].data['norm'] = dgl.norm_by_dst(hg, cetype).unsqueeze(1)
    category_id = hg.ntypes.index(category)
    g = dgl.to_homogeneous(hg, edata=['norm'])
    node_ids = th.arange(g.num_nodes()).to(device)
    target_idx = node_ids[g.ndata[dgl.NTYPE] == category_id]  # target node index

    # cugraph-ops requires node/edge type to be int32
    g.ndata['ntype'] = g.ndata.pop(dgl.NTYPE).type(th.int32)
    g.edata[dgl.ETYPE] = g.edata[dgl.ETYPE].type(th.int32)

    # create model
    h_dim = 16  # hidden feature dim
    num_classes = data.num_classes
    num_bases = 10
    model = RGCN(g.num_nodes(), h_dim, num_classes, num_rels, num_bases)
    model = model.to(device)

    optimizer = th.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training
    model.train()
    for epoch in range(50):
        logits = model(g)
        logits = logits[target_idx]
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        train_acc = accuracy(logits[train_idx].argmax(dim=1), labels[train_idx]).item()
        print("Epoch {:05d} | Loss {:.4f} | Train Accuracy {:.4f} ".format(epoch, loss.item(), train_acc))

    # evaluation
    model.eval()
    with th.no_grad():
        logits = model(g)
    logits = logits[target_idx]
    test_acc = accuracy(logits[test_idx].argmax(dim=1), labels[test_idx]).item()
    print("Test Accuracy: {:.4f}".format(test_acc))
