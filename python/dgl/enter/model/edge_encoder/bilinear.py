import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearPredictor(nn.Module):
    def __init__(self,
                 in_size: int = -1,
                 out_size: int = 1,
                 hidden_size: int = 32,
                 num_layers: int = 1,
                 bias: bool = False):
        super(BilinearPredictor, self).__init__()
        self.bilinear = nn.Bilinear(in_size, in_size, hidden_size, bias=bias)
        lins_list = []
        for _ in range(num_layers-2):
            lins_list.append(nn.Linear(hidden_size, hidden_size, bias=bias))
            lins_list.append(nn.ReLU())
        lins_list.append(nn.Linear(hidden_size, out_size, bias=bias))
        self.linear = nn.Sequential(*lins_list)

    def forward(self, h_src, h_dst):
        h = self.bilinear(h_src, h_dst)
        h = self.linear(h)
        h = torch.sigmoid(h)
        return h
