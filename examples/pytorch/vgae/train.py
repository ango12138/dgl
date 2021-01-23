import argparse
import os
import time

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn.functional as F

from input_data import load_data
import model
from preprocess import mask_test_edges, mask_test_edges_dgl, sparse_to_tuple, preprocess_graph

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description='Variant Graph Auto Encoder')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--epochs', '-e', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--hidden1', '-h1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', '-h2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--datasrc', '-s', type=str, default='website',
                    help='Dataset download from dgl Dataset or website.')
parser.add_argument('--dataset', '-d', type=str, default='cora', help='Dataset string.')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU id to use.')
args = parser.parse_args()


# roc_means = []
# ap_means = []

def compute_loss_para(adj):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def get_scores(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def dgl_main():
    # Load from DGL dataset
    if args.dataset == 'cora':
        dataset = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        dataset = PubmedGraphDataset()
    else:
        raise NotImplementedError
    graph = dataset[0]

    # check device
    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")

    # Extract node features
    feats = graph.ndata.pop('feat').to(device)
    in_dim = feats.shape[-1]

    graph = graph.to(device)

    # generate input
    adj_orig = graph.adjacency_matrix().to_dense().to(device)

    # build test set with 10% positive links
    train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_dgl(graph, adj_orig)

    # create train graph
    train_edge_idx = torch.tensor(train_edge_idx).to(device)
    train_graph = dgl.edge_subgraph(graph, train_edge_idx, preserve_nodes=True)
    train_graph = train_graph.to(device)
    adj = train_graph.adjacency_matrix().to_dense().to(device)

    # compute loss parameters
    weight_tensor, norm = compute_loss_para(adj)

    # create model
    vgae_model = model.VGAEModel(in_dim, args.hidden1, args.hidden2)
    vgae_model = vgae_model.to(device)

    # create training component
    optimizer = torch.optim.Adam(vgae_model.parameters(), lr=args.learning_rate)
    print('Total Parameters:', sum([p.nelement() for p in vgae_model.parameters()]))

    # create training epoch
    for epoch in range(args.epochs):
        t = time.time()

        # Training and validation using a full graph
        vgae_model.train()

        logits = vgae_model.forward(train_graph, feats)

        # compute loss
        loss = norm * F.binary_cross_entropy(logits.view(-1), adj.view(-1), weight=weight_tensor)
        kl_divergence = 0.5 / logits.size(0) * (
                1 + 2 * vgae_model.log_std - vgae_model.mean ** 2 - torch.exp(vgae_model.log_std) ** 2).sum(
            1).mean()
        loss -= kl_divergence

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = get_acc(logits, adj)

        val_roc, val_ap = get_scores(val_edges, val_edges_false, logits)

        # Print out performance
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), "train_acc=",
              "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc), "val_ap=", "{:.5f}".format(val_ap),
              "time=", "{:.5f}".format(time.time() - t))

    test_roc, test_ap = get_scores(test_edges, test_edges_false, logits)
    # roc_means.append(test_roc)
    # ap_means.append(test_ap)
    print("End of training!", "test_roc=", "{:.5f}".format(test_roc), "test_ap=", "{:.5f}".format(test_ap))


def web_main():
    adj, features = load_data(args.dataset)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    features = sparse_to_tuple(features.tocoo())

    # Create Model
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                        torch.FloatTensor(adj_norm[1]),
                                        torch.Size(adj_norm[2]))
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                         torch.FloatTensor(adj_label[1]),
                                         torch.Size(adj_label[2]))
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                        torch.FloatTensor(features[1]),
                                        torch.Size(features[2]))

    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight

    # row = adj.tocoo().row
    # col = adj.tocoo().col
    # row = torch.tensor(row)
    # col = torch.tensor(col)
    # graph = dgl.graph((row, col))

    graph = dgl.from_scipy(adj)
    graph.add_self_loop()

    features = features.to_dense()
    in_dim = features.shape[-1]

    vgae_model = model.VGAEModel(in_dim, args.hidden1, args.hidden2)
    # create training component
    optimizer = torch.optim.Adam(vgae_model.parameters(), lr=args.learning_rate)
    print('Total Parameters:', sum([p.nelement() for p in vgae_model.parameters()]))

    def get_scores(edges_pos, edges_neg, adj_rec):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        preds = []
        pos = []
        for e in edges_pos:
            # print(e)
            # print(adj_rec[e[0], e[1]])
            preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    def get_acc(adj_rec, adj_label):
        labels_all = adj_label.to_dense().view(-1).long()
        preds_all = (adj_rec > 0.5).view(-1).long()
        accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
        return accuracy

    # create training epoch
    for epoch in range(args.epochs):
        t = time.time()

        # Training and validation using a full graph
        vgae_model.train()

        logits = vgae_model.forward(graph, features)

        # compute loss
        loss = norm * F.binary_cross_entropy(logits.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
        kl_divergence = 0.5 / logits.size(0) * (
                1 + 2 * vgae_model.log_std - vgae_model.mean ** 2 - torch.exp(vgae_model.log_std) ** 2).sum(
            1).mean()
        loss -= kl_divergence

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = get_acc(logits, adj_label)

        val_roc, val_ap = get_scores(val_edges, val_edges_false, logits)

        # Print out performance
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), "train_acc=",
              "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc), "val_ap=", "{:.5f}".format(val_ap),
              "time=", "{:.5f}".format(time.time() - t))

    test_roc, test_ap = get_scores(test_edges, test_edges_false, logits)
    print("End of training!", "test_roc=", "{:.5f}".format(test_roc), "test_ap=", "{:.5f}".format(test_ap))


# if __name__ == '__main__':
#     for i in range(10):
#         main()
#
#     roc_mean = np.mean(roc_means)
#     roc_std = np.std(roc_means, ddof=1)
#     ap_mean = np.mean(ap_means)
#     ap_std = np.std(ap_means, ddof=1)
#     print("roc_mean=", "{:.5f}".format(roc_mean), "roc_std=", "{:.5f}".format(roc_std), "ap_mean=",
#           "{:.5f}".format(ap_mean), "ap_std=", "{:.5f}".format(ap_std))

if __name__ == '__main__':
    if args.datasrc == 'dgl':
        dgl_main()
    elif args.datasrc == 'website':
        web_main()
