"""
.. currentmodule:: dgl

DGL at a Glance
=========================

**Author**: `Minjie Wang <https://jermainewang.github.io/>`_, Quan Gan, `Jake
Zhao <https://cs.nyu.edu/~jakezhao/>`_, Zheng Zhang

The goal of this tutorial:

- Understand how DGL enables computation on a graph from a
  high level.
- Train a simple graph neural network in DGL to classify nodes in a graph.

At the end of this tutorial, we hope you get a brief feeling of how DGL works.
"""

###############################################################################
# Why DGL?
# ----------------
# DGL is designed to perform **machine learning** on **graph-structured
# data**. Specifically, DGL enables trouble-free implementation of a class of deep learning 
# models called graph neural networks (GNN). DGL is a library built on top of PyTorch.
# It provides friendly APIs for performing custom computation on graphs based on message passing.
# Through DGL, we hope to benefit both researchers
# and practitioners doing machine learning on structured data.
#
# *This tutorial assumes basic familiarity with pytorch.*

###############################################################################
# A Toy Problem: classification for Zachary's Karate Club
# ----------------------------------
#
# We illustrate how to use DGL by walking through an example to classify nodes on the well-known 
# `Zachary's Karate Club social network <https://en.wikipedia.org/wiki/Zachary%27s_karate_club>`
# The network contains 34 club members, and edges capture pairs of members that interacted outside of the club.
# The club has subsequently ``split'' into two sub-groups due to conflicts between the instructor and the club president.
# **Our task is to classify which sub-group each member joins based on the social network structure.**
#
# The graph below is a visualization of the karate club network. 
# Node 0 and 33 are the instructor and the club president, respectively.
#
# .. image:: http://historicaldataninjas.com/wp-content/uploads/2014/05/karate.jpg 
#    :height: 400px
#    :width: 500px
#    :align: center
#


###############################################################################
# Build the graph
# ---------------
# A graph is built using :class:`~dgl.DGLGraph` class. Here is how we add the 34 members
# and their interaction edges into the graph.

import dgl

def build_karate_club_graph():
    g = dgl.DGLGraph()
    # add 34 nodes into the graph; nodes are labeled from 0~33
    g.add_nodes(34)
    # all 78 edges as a list of tuples
    edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
        (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
        (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
        (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
        (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
        (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
        (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
        (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
        (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
        (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
        (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
        (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
        (33, 31), (33, 32)]
    # add edges using two list of nodes: src and dst
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # edges are directional in DGL; make them bi-directional
    g.add_edges(dst, src)

    return g

###############################################################################
# We can print out the number of nodes and edges in our newly constructed graph:

G = build_karate_club_graph()
print('We have %d nodes.' % G.number_of_nodes())
print('We have %d edges.' % G.number_of_edges())

###############################################################################
# We can also visualize the graph by converting it to a `networkx
# <https://networkx.github.io/documentation/stable/>`_ graph:

import networkx as nx
nx_G = G.to_networkx()
pos = nx.circular_layout(nx_G)
nx.draw(nx_G, pos, with_labels=True)

###############################################################################
# Assign features
# ---------------
# Graph neural networks associate features with nodes and edges for training. 
# For our classification example, we assign each node's an input feature as a one-hot vector:
# node :math:`v_i`'s  
# feature vector is :math:`[0,\ldots,1,\dots,0]`, where the :math:`i^{th}` position is one).
#
# In DGL, we can add features for all nodes at once, using a feature tensor that
# batches node features along the first dimension. This code below adds the one-hot
# feature for all nodes:

import torch

G.ndata['feat'] = torch.eye(34)


###############################################################################
# We can print out the node features to verify:

# print out node 2's input feature
print(G.nodes[2].data['feat'])

# print out node 10 and 11's input features
print(G.nodes[[10, 11]].data['feat'])

###############################################################################
# Define a Graph Convolutional Network (GCN)
# ------------------------------------------
# To solve our classification problem, we use the Graph Convolutional
# Network (GCN) developed by `Kipf and
# Welling <https://arxiv.org/abs/1609.02907>`_. At a high-level, the GCN model does 
# the following:
#
# - Each node :math:`v_i` has a feature vector :math:`h_i`.
# - Each node aggregates the feature vectors :math:`h_j` from its neighbors, performs
#   an affine and non-linear transformation to update its own feature.
#
# A graphical demonstration is displayed below.
#
# .. image:: https://s3.us-east-2.amazonaws.com/dgl.ai/tutorial/1_first/mailbox.png
#    :alt: mailbox
#    :align: center
#
# A GCN model may consist of several layers, each of which uses a separate weight tensor for the affine transformation.
#
#
# We use DGL's message passing interface to implement a GCN layer.  In DGL, in order to specify computation
# based on message passing, programmers must provide the following three components:
#
# 1. Define the message function.
#    - The message function specifies, for each edge, what message to send to the ``mailbox'' of the edge's downstream node.
# 2. Define the reduce function.
#    - The reduce function specifies, for each node, how messages received at its ``mailbox'' are aggregated.
# 3. Specify the set of nodes and edges to perform message passing.
#
# To enable gradient-based optimization, programmers should sub-class PyTorch's `nn.Module <https://pytorch.org/docs/stable/nn.html` and perform message-passing in the 
# forward function of the module.

# This is how a GCN layer is implemented:
 
import torch.nn as nn
import torch.nn.functional as F

# Define the message function to generate messages along edges
# NOTE: we ignore the GCN's normalization constant c_ij for this tutorial.
def gcn_message(edges):
    # The argument is a batch of edges.
    # This computes a (batch of) message called 'msg' using the source node's feature 'h'.
    return {'msg' : edges.src['h']}

# Define the reduce function to aggregate incoming messages at nodes
def gcn_reduce(nodes):
    # The argument is a batch of nodes.
    # This computes the new 'h' features by summing received 'msg' in each node's mailbox.
    return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}

# Define the GCNLayer module to perform message passing 
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
        # g is the graph and the inputs is the input node features
        # first set the node features
        g.ndata['h'] = inputs
        # trigger message passing on all edges 
        g.send(g.edges(), gcn_message)
        # trigger aggregation at all nodes
        g.recv(g.nodes(), gcn_reduce)
        # get the result node features
        h = g.ndata.pop('h')
        # perform linear transformation
        return self.linear(h)

###############################################################################
# We then define a neural network that contains two GCN layers. The first layer 
# transforms input features of size of 34 to a hidden size of 5. The second layer 
# transforms the hidden size of 5 to output features of size 2 which correspond to the two sub-groups
# of the karate club.

# Define a 2-layer GCN model
class Net(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(Net, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)
        return h
# input_feature_size=34, hidden_size=5, num_classes=2
net = Net(34, 5, 2)

###############################################################################
# Train the GCN model to classify nodes
# ----------------------------------------
#
# To prepare the input features and labels, we adopt a 
# semi-supervised setting. The input feature of each node is initialized by 
# one-hot encoding. Only the instructor (node 0) and the club president
# (node 33) are labeled.

inputs = torch.eye(34)
labeled_nodes = torch.tensor([0, 33])  # only the instructor and the president nodes are labeled
labels = torch.tensor([0, 1])  # their labels are different

###############################################################################
# The training loop is the same as other NN models. We (1) create an optimizer,
# (2) feed the inputs to the model, (3) calculate the loss and (4) use autograd
# to optimize the model.

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
all_logits = []
for epoch in range(30):
    logits = net(G, inputs)
    # we save the logits for visualization later
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)
    # we only compute loss for labeled nodes
    loss = F.nll_loss(logp[labeled_nodes], labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

###############################################################################
# Since the model produces an output feature of size 2 for each node, we can
# visualize by plotting the output feature in a 2D space.

import matplotlib.animation as animation
import matplotlib.pyplot as plt

def draw(i):
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'
    pos = {}
    colors = []
    for v in range(34):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    ax.axis('off')
    ax.set_title('Epoch: %d' % i)
    nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors,
            with_labels=True, node_size=300, ax=ax)

###############################################################################
# We first plot the initial guess before training. As you can see, the nodes
# are not classified correctly.

fig = plt.figure(dpi=150)
fig.clf()
ax = fig.subplots()
draw(0)  # draw the prediction of the first epoch
plt.close()

###############################################################################
# .. image:: https://s3.us-east-2.amazonaws.com/dgl.ai/tutorial/1_first/karate0.png
#    :height: 300px
#    :width: 400px
#    :align: center

###############################################################################
# The following animation shows how the model correctly predicts the community
# after a series of training epochs.

ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=200)

###############################################################################
# .. image:: https://s3.us-east-2.amazonaws.com/dgl.ai/tutorial/1_first/karate.gif
#    :height: 300px
#    :width: 400px
#    :align: center

###############################################################################
# Next steps
# ----------
# In the :doc:`next tutorial <2_basics>`, we will go through some more basics
# of DGL, such as reading and writing node/edge features.
