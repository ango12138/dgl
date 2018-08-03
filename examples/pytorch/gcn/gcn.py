"""
Semi-Supervised Classification with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1609.02907
Code: https://github.com/tkipf/gcn
"""
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from ..data import load_cora, load_citeseer, load_pubmed

def gcn_msg(self, src, edge):
    return src

def gcn_reduce(self, node, msgs):
    return sum(msgs)

class NodeUpdateModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None):
        super(NodeUpdateModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node, accum):
        h = self.linear(accum)
        if self.activation:
            h = self.activation(h)
        return h

class GCN(nn.Module):
    def __init__(self,
                 nx_graph,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = DGLGraph(nx_graph)
        self.dropout = dropout
        # input layer
        self.layers = nn.ModuleList([NodeUpdateModule(in_feats, n_hidden, activation)])
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(NodeUpdateModule(n_hidden, n_hidden, activation))
        # output layer
        self.layers.append(NodeUpdateModule(n_hidden, n_classes))

    def forward(self, features):
        for n in self.g.nodes():
            print(n)
        assert False
        self.g.set_n_repr(features)
        for layer in self.layers:
            # apply dropout
            if self.dropout:
                val = F.dropout(self.g.get_n_repr(), p=self.dropout)
                self.g.set_n_repr(val)
            self.g.update_all(gcn_msg, gcn_reduce, layer)
        return self.g.get_n_repr()

def main(args):
    # load and preprocess dataset
    if args.dataset == 'cora':
        data = load_cora()
    elif args.dataset == 'citeseer':
        data = load_citeseer()
    elif args.dataset == 'pubmed':
        data = load_pubmed()
    else:
        raise RuntimeError('Error dataset: {}'.format(args.dataset))
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.ByteTensor(data.train_mask)
    in_feats = features.size(1)
    n_classes = data.num_labels

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        mask = mask.cuda()

    # create GCN model
    model = GCN(data.graph,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)

    if cuda:
        model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # initialize graph
    t0 = time.time()
    for epoch in range(args.n_epochs):
        # forward
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask])
        print("epoch {} loss: {}".format(epoch, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('%f seconds/epoch' % ((time.time() - t0) / args.n_epochs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, required=True,
            help="dataset")
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=10,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    args = parser.parse_args()
    print(args)

    main(args)
