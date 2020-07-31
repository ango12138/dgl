import torch
import torch.nn as nn
import os
import dgl


def cuda(x):
    if torch.cuda.is_available() and not os.getenv('NOCUDA', None):
        return x.to(torch.device('cuda'))   # works for both DGLGraph and tensor
    else:
        return tensor


class GRUUpdate(nn.Module):
    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size

        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)

    def update_zm(self, node):
        src_x = node.data['src_x']
        s = node.data['s']
        rm = node.data['accum_rm']
        z = torch.sigmoid(self.W_z(torch.cat([src_x, s], 1)))
        m = torch.tanh(self.W_h(torch.cat([src_x, rm], 1)))
        m = (1 - z) * s + z * m
        return {'m': m, 'z': z}

    def update_r(self, node, zm=None):
        dst_x = node.data['dst_x']
        m = node.data['m'] if zm is None else zm['m']
        r_1 = self.W_r(dst_x)
        r_2 = self.U_r(m)
        r = torch.sigmoid(r_1 + r_2)
        return {'r': r, 'rm': r * m}

    def forward(self, node):
        dic = self.update_zm(node)
        dic.update(self.update_r(node, zm=dic))
        return dic

def tocpu(g):
    src, dst = g.edges()
    src = src.cpu()
    dst = dst.cpu()
    return dgl.graph((src, dst), num_nodes=g.number_of_nodes())

def line_graph(g, backtracking=True, shared=False):
    g2 = tocpu(g)
    g2 = dgl.line_graph(g2, backtracking, shared)
    g2 = g2.to(g.device)
    g2.ndata.update(g.edata)
    return g2
