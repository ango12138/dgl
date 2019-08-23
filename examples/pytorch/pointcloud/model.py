import torch
import torch.nn as nn
import torch.nn.functional as F
from edgeconv import EdgeConv
from factory import NearestNeighborGraph

class Model(nn.Module):
    def __init__(self, k, feature_dims, emb_dims, output_classes, input_dims=3,
                 dropout_prob=0.5):
        super(Model, self).__init__()

        self.nng = NearestNeighborGraph(k)
        self.conv = nn.ModuleList()

        self.num_layers = len(feature_dims)
        for i in range(self.num_layers):
            self.conv.append(EdgeConv(
                feature_dims[i - 1] if i > 0 else input_dims,
                feature_dims[i]))

        self.proj = nn.Linear(sum(feature_dims), emb_dims[0])

        self.embs = nn.ModuleList()
        self.bn_embs = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.num_embs = len(emb_dims) - 1
        for i in range(1, self.num_embs + 1):
            self.embs.append(nn.Linear(
                # * 2 because of concatenation of max- and mean-pooling
                emb_dims[i - 1] if i > 1 else (emb_dims[i - 1] * 2),
                emb_dims[i]))
            self.bn_embs.append(nn.BatchNorm1d(emb_dims[i]))
            self.dropouts.append(nn.Dropout(dropout_prob))

        self.proj_output = nn.Linear(emb_dims[-1], output_classes)

    def forward(self, x):
        hs = []
        batch_size, n_points, x_dims = x.shape
        h = x

        for i in range(self.num_layers):
            g = self.nng(h)
            # Although the official implementation includes batch norm
            # within EdgeConv, I choose to omit it for a number of reasons:
            #
            # (1) When the point clouds within each batch do not have the
            #     same number of points, batch norm would not work.
            #
            # (2) Even if the point clouds always have the same number of
            #     points, the points may as well be shuffled even with the
            #     same (type of) object (and the official implementation
            #     *does* shuffle the points of the same example for each
            #     epoch).
            #
            #     For example, the first point of a point cloud of an
            #     airplane does not always necessarily reside at its nose.
            #
            #     In this case, the learned statistics of each position
            #     by batch norm is not as meaningful as those learned from
            #     images.
            h = self.conv[i](g, h)
            h = F.leaky_relu(h, 0.2)
            hs.append(h.view(batch_size, n_points, -1))

        h = torch.cat(hs, 2)
        h = self.proj(h)
        h_max, _ = torch.max(h, 1)
        h_avg = torch.mean(h, 1)
        h = torch.cat([h_max, h_avg], 1)

        for i in range(self.num_embs):
            h = self.embs[i](h)
            h = self.bn_embs[i](h)
            h = F.leaky_relu(h, 0.2)
            h = self.dropouts[i](h)

        h = self.proj_output(h)
        return h


def compute_loss(logits, y, eps=0.2):
    num_classes = logits.shape[1]
    one_hot = torch.zeros_like(logits).scatter_(1, y.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)
    log_prob = F.log_softmax(logits, 1)
    loss = -(one_hot * log_prob).sum(1).mean()
    return loss
