"""Torch Module for PGExplainer"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

__all__ = ["PGExplainer"]


class PGExplainer(nn.Module):
    r"""PGExplainer from `Parameterized Explainer for Graph Neural Network
    <https://arxiv.org/pdf/2011.04573>`

    PGExplainer adopts a deep neural network (explanation network) to parameterize the generation
    process of explanations, which enables it to explain multiple instances
    collectively. PGExplainer models the underlying structure as edge
    distributions, from which the explanatory graph is sampled.

    Parameters
    ----------
    model : nn.Module
        The GNN model to explain that tackles multiclass graph classification

        * Its forward function must have the form
          :attr:`forward(self, graph, nfeat, embed, edge_weight)`.
        * The output of its forward function is the logits if embed=False else
          return the intermediate node embeddings.
    num_features : int
        Node embedding size used by :attr:`model`.
    epochs : int, optional
        Number of epochs to train the explanation network. Default: 10.
    lr : float, optional
        Learning rate to train the explanation network. Default: 0.01.
    coff_budget : float, optional
        Size regularization to constrain the explanation size. Default: 0.01.
    coff_connect : float, optional
        Entropy regularization to constrain the connectivity of explanation. Default: 5e-4.
    init_tmp : float, optional
        The temperature at the first epoch. Default: 5.0.
    final_tmp : float, optional
        The temperature at the final epoch. Default: 1.0.
    sample_bias : float, optional
        Some members of a population are systematically more likely to be selected
        in a sample than others. Default: 0.0.
    """

    def __init__(
        self,
        model,
        num_features,
        epochs=10,
        lr=0.01,
        coff_budget=0.01,
        coff_connect=5e-4,
        init_tmp=5.0,
        final_tmp=1.0,
        sample_bias=0.0,
    ):
        super(PGExplainer, self).__init__()

        self.model = model
        self.num_features = num_features * 2

        # training hyperparameters for PGExplainer
        self.epochs = epochs
        self.lr = lr
        self.coff_budget = coff_budget
        self.coff_connect = coff_connect
        self.init_tmp = init_tmp
        self.final_tmp = final_tmp
        self.sample_bias = sample_bias

        self.init_bias = 0.0

        # Explanation network in PGExplainer
        self.elayers = nn.ModuleList()
        self.elayers.append(
            nn.Sequential(nn.Linear(self.num_features, 64), nn.ReLU())
        )
        self.elayers.append(nn.Linear(64, 1))

    def set_masks(self, graph, feat, edge_mask=None):
        r"""Set the edge mask that play a crucial role to explain the
        prediction made by the GNN for a graph. Initialize learnable edge
        mask if it is None.

        Parameters
        ----------
        graph : DGLGraph
            A homogeneous graph.
        feat : Tensor
            The input feature of shape :math:`(N, D)`. :math:`N` is the
            number of nodes, and :math:`D` is the feature size.
        edge_mask : Tensor, optional
            Learned importance mask of the edges in the graph, which is a tensor
            of shape :math:`(E)`, where :math:`E` is the number of edges in the
            graph. The values are within range :math:`(0, 1)`. The higher,
            the more important. Default: None.
        """
        num_nodes, _ = feat.shape
        num_edges = graph.num_edges()

        init_bias = self.init_bias
        std = nn.init.calculate_gain("relu") * math.sqrt(2.0 / (2 * num_nodes))

        if edge_mask is None:
            self.edge_mask = torch.randn(num_edges) * std + init_bias
        else:
            self.edge_mask = edge_mask

        self.edge_mask = self.edge_mask.to(graph.device)

    def clear_masks(self):
        r"""Clear the edge mask that play a crucial role to explain the
        prediction made by the GNN for a graph.
        """
        self.edge_mask = None

    def loss(self, prob, ori_pred):
        r"""The loss function that is used to learn the edge
        distribution.

        Parameters
        ----------
        prob:  Tensor
            Tensor contains a set of probabilities for each possible
            class label of some model.
        ori_pred: int
            Integer representing the original prediction.

        Returns
        -------
        float
            The function returns the sum of the three loss components,
            which is a scalar tensor representing the total loss.
        """
        logit = prob[ori_pred]
        # 1e-6 added to logit to avoid taking the logarithm of zero
        logit += 1e-6
        # computing the cross-entropy loss for a single prediction
        pred_loss = -torch.log(logit)

        # size
        edge_mask = self.sparse_mask_values
        if self.coff_budget <= 0:
            size_loss = self.coff_budget * torch.sum(edge_mask)
        else:
            size_loss = self.coff_budget * F.relu(
                torch.sum(edge_mask) - self.coff_budget
            )

        # entropy
        scale = 0.99
        edge_mask = self.edge_mask * (2 * scale - 1.0) + (1.0 - scale)
        mask_ent = -edge_mask * torch.log(edge_mask) - (
            1 - edge_mask
        ) * torch.log(1 - edge_mask)
        mask_ent_loss = self.coff_connect * torch.mean(mask_ent)

        loss = pred_loss + size_loss + mask_ent_loss

        return loss

    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        r"""Sample from the instantiation of concrete distribution when training.

        Parameters
        ----------
        log_alpha : Tensor
            A tensor representing the log of the prior probability of activating the gate.
        beta : float, optional
            Controls the degree of randomness in the gate's output.
        training : bool, optional
            Indicates whether the gate is being used during training or evaluation.

        Returns
        -------
        Tensor
            If training is set to True, the output is a tensor of probabilities that
            represent the probability of activating the gate for each input element.
            If training is set to False, the output is also a tensor of probabilities,
            but they are determined solely by the log_alpha values, without adding any
            random noise.
        """
        if training:
            bias = self.sample_bias
            random_noise = torch.rand(log_alpha.size()).to(log_alpha.device)
            random_noise = bias + (1 - 2 * bias) * random_noise
            gate_inputs = torch.log(random_noise) - torch.log(
                1.0 - random_noise
            )
            gate_inputs = (gate_inputs + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)

        return gate_inputs

    def train_step(self, dataset, func_extract_feat):
        r"""Training the explanation network by gradient descent(GD)
        using Adam optimizer

        Parameters
        ----------
        dataset : dgl.data
            The dataset to train the importance edge mask.
        func_extract_feat : func
            A function that extracts the node embeddings for each individual graphs.
        """
        graph, _ = dataset[0]
        self.model = self.model.to(graph.device)
        self.elayers = self.elayers.to(graph.device)

        optimizer = Adam(self.elayers.parameters(), lr=self.lr)

        ori_pred_dict = {}

        self.model.eval()
        with torch.no_grad():
            for idx, (g, _) in enumerate(dataset):
                feat = func_extract_feat(g)

                logits = self.model(g, feat)
                ori_pred_dict[idx] = logits.argmax(-1).data

        # train the mask generator
        for epoch in range(self.epochs):
            loss = 0.0
            pred_list = []

            tmp = float(
                self.init_tmp
                * np.power(self.final_tmp / self.init_tmp, epoch / self.epochs)
            )

            self.elayers.train()
            optimizer.zero_grad()

            for idx, (g, _) in enumerate(dataset):
                feat = func_extract_feat(g)

                prob, edge_mask = self.explain_graph(
                    g, feat, tmp=tmp, training=True
                )

                self.edge_mask = edge_mask

                loss_tmp = self.loss(prob.unsqueeze(dim=0), ori_pred_dict[idx])
                loss_tmp.backward()

                loss += loss_tmp.item()
                pred_label = prob.argmax(-1).item()
                pred_list.append(pred_label)

            optimizer.step()
            print(f"Epoch: {epoch} | Loss: {loss}")

    def explain_graph(self, graph, feat, tmp=1.0, training=False):
        r"""Learn and return an edge mask that plays a crucial role to
        explain the prediction made by the GNN for a graph. Also, return
        the prediction made with the edges chosen based on the edge mask.

        Parameters
        ----------
        graph : DGLGraph
            A homogeneous graph.
        feat : Tensor
            The input feature of shape :math:`(N, D)`. :math:`N` is the
            number of nodes, and :math:`D` is the feature size.
        tmp : float
            The temperature parameter fed to the sampling procedure.
        training : bool
            We indicate we want to train the explanation network if
            set to True. If set to False, we indicate that we want
            to find the edges of the graph that contribute most to
            the graph explanations.

        Returns
        -------
        Tensor, Tensor
            The classification probability for graph with edge mask,
            the edge weights where higher edge weights indicate that
            they contribute to the explanation.

        Examples
        --------

        >>> import torch as th
        >>> import torch.nn as nn
        >>> import dgl
        >>> from dgl.data import GINDataset
        >>> from dgl.dataloading import GraphDataLoader
        >>> from dgl.nn import GraphConv, PGExplainer

        >>> # Define the model
        >>> class Model(nn.Module):
        ...     def __init__(self, in_feats, out_feats):
        ...         super().__init__()
        ...         self.conv = GraphConv(in_feats, out_feats)
        ...         self.fc = nn.Linear(out_feats, 1)
        ...         nn.init.xavier_uniform_(self.fc.weight)
        ...
        ...     def forward(self, g, h, embed=False, edge_weight=None):
        ...         h = self.conv(g, h, edge_weight=edge_weight)
        ...         if not embed:
        ...             g.ndata['h'] = h
        ...             hg = dgl.mean_nodes(g, 'h')
        ...             return th.sigmoid(self.fc(hg))
        ...         else:
        ...             return h

        >>> # Load dataset
        >>> data = GINDataset('MUTAG', self_loop=True)
        >>> dataloader = GraphDataLoader(data, batch_size=64, shuffle=True)

        >>> # Train the model
        >>> feat_size = data[0][0].ndata['attr'].shape[1]
        >>> model = Model(feat_size, data.gclasses)
        >>> criterion = nn.BCELoss()
        >>> optimizer = th.optim.Adam(model.parameters(), lr=1e-2)
        >>> for bg, labels in dataloader:
        ...     preds = model(bg, bg.ndata['attr'])
        ...     loss = criterion(preds.squeeze(1).float(), labels.float())
        ...     optimizer.zero_grad()
        ...     loss.backward()
        ...     optimizer.step()

        >>> # Initialize the explainer
        >>> explainer = PGExplainer(model, data.gclasses)
        >>> explainer.train_step(data, lambda g: g.ndata["attr"])

        >>> # Explain the prediction for graph 0
        >>> graph, l = data[0]
        >>> graph_feat = graph.ndata.pop("attr")
        >>> probs, edge_weight = explainer.explain_graph(graph, graph_feat)
        """
        self.model = self.model.to(graph.device)
        self.elayers = self.elayers.to(graph.device)

        embed = self.model(graph, feat, embed=True)
        embed = embed.data

        edge_idx = graph.edges()

        node_size = embed.shape[0]
        col, row = edge_idx
        col_emb = embed[col.long()]
        row_emb = embed[row.long()]
        emb = torch.cat([col_emb, row_emb], dim=-1)

        for elayer in self.elayers:
            emb = elayer(emb)
        values = emb.reshape(-1)

        values = self.concrete_sample(values, beta=tmp, training=training)
        self.sparse_mask_values = values

        mask_sparse = torch.sparse_coo_tensor(
            [edge_idx[0].tolist(), edge_idx[1].tolist()],
            values.tolist(),
            (node_size, node_size),
        )
        mask_sigmoid = mask_sparse.to_dense()
        # set the symmetric edge weights
        sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2
        edge_mask = sym_mask[edge_idx[0].long(), edge_idx[1].long()]

        # inverse the weights before sigmoid
        self.clear_masks()
        self.set_masks(graph, feat, edge_mask)

        # the model prediction with the updated edge mask
        logits = self.model(graph, feat, edge_weight=self.edge_mask)
        probs = F.softmax(logits, dim=-1)

        self.clear_masks()

        return probs, edge_mask
