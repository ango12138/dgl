import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as FN
from .. import randomwalk
from ..utils import cuda
from .layers import ScaledEmbedding

def create_embeddings(n_nodes, n_features):
    return nn.Parameter(torch.randn(n_nodes, n_features))

def mix_embeddings(h, ndata, emb, proj):
    '''Combine node-specific trainable embedding ``h`` with categorical inputs
    (projected by ``emb``) and numeric inputs (projected by ``proj``).
    '''
    e = []
    for key in emb.keys():
        value = ndata[key]
        result = emb[key](cuda(value))
        if result.dim() == 3:
            result = result.mean(1)     # bag of words
        e.append(result)
    for key in proj.keys():
        value = ndata[key]
        e.append(proj[key](cuda(value)))
    return (cuda(h) if h is not None else 0) + torch.stack(e, 0).sum(0)

def get_embeddings(h, nodeset):
    return h[nodeset]

def put_embeddings(h, nodeset, new_embeddings):
    n_nodes = nodeset.shape[0]
    n_features = h.shape[1]
    return h.scatter(0, nodeset[:, None].expand(n_nodes, n_features), new_embeddings)

def safediv(a, b):
    b = torch.where(b == 0, torch.ones_like(b), b)
    return a / b

def init_weight(w, func_name, nonlinearity):
    getattr(nn.init, func_name)(w, gain=nn.init.calculate_gain(nonlinearity))

def init_bias(w):
    nn.init.constant_(w, 0)

class PinSageConv(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super(PinSageConv, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.Q = nn.Linear(in_features, hidden_features)
        self.W = nn.Linear(in_features + hidden_features, out_features)
        self.V = nn.Linear(in_features, hidden_features)

        init_weight(self.Q.weight, 'xavier_uniform_', 'leaky_relu')
        init_weight(self.W.weight, 'xavier_uniform_', 'leaky_relu')
        init_weight(self.V.weight, 'xavier_uniform_', 'leaky_relu')
        init_bias(self.Q.bias)
        init_bias(self.W.bias)
        init_bias(self.V.bias)

    def forward(self, nodes):
        h_agg = safediv(nodes.data['h_agg'], nodes.data['w'][:, None])
        h = nodes.data['h']
        h_concat = torch.cat([h, h_agg], 1)
        # HSelf
        #h_new = F.leaky_relu(self.W(h_concat) + self.V(nodes.data['h_x']))
        h_new = F.leaky_relu(self.W(h_concat))
        h_new = safediv(h_new, h_new.norm(dim=1, keepdim=True))
        return {'h': h_new}

    def project(self, h):
        return F.leaky_relu(self.Q(h))


class PinSage(nn.Module):
    def __init__(self, feature_sizes, use_feature=False, G=None, emb={}, proj={}):
        super(PinSage, self).__init__()
        self.in_features = feature_sizes[0]
        self.out_features = feature_sizes[-1]
        self.n_layers = len(feature_sizes) - 1

        self.convs = nn.ModuleList()
        for i in range(self.n_layers):
            self.convs.append(PinSageConv(
                feature_sizes[i], feature_sizes[i+1], feature_sizes[i+1]))

        self.use_feature = use_feature

        if use_feature:
            self.emb = nn.ModuleDict()
            self.proj = nn.ModuleDict()

            for key, scheme in G.node_attr_schemes().items():
                if scheme.dtype == torch.int64:
                    self.emb[key] = emb.get(key, ScaledEmbedding(
                            G.ndata[key].max().item() + 1,
                            self.in_features,
                            padding_idx=0))
                elif scheme.dtype == torch.float32:
                    w = nn.Linear(scheme.shape[0], self.in_features)
                    init_weight(w.weight, 'xavier_uniform_', 'leaky_relu')
                    init_bias(w.bias)
                    self.proj[key] = proj.get(key, nn.Sequential(w, nn.LeakyReLU()))

    msg = [FN.copy_src('h_q', 'h_w'),
           FN.copy_src('deg', 'w')]
    red = [FN.sum('h_w', 'h_agg'), FN.sum('w', 'w')]

    
    def forward(self, nf, h_emb):
        '''
        nf: NodeFlow.
        '''
        if self.n_layers == 0:
            nid = nf.layer_parent_nid(-1)
            h = h_emb(cuda(nid + 1)) if h_emb is not None else None
            if self.use_feature:
                nf.layers[-1].data['h'] = mix_embeddings(
                        h, nf.layers[-1].data, self.emb, self.proj)
            else:
                nf.layers[-1].data['h'] = cuda(h)
            return nf.layers[-1].data['h']

        nids = [nf.layer_parent_nid(i) for i in range(nf.num_layers)]
        nids = torch.cat(nids)
        h = h_emb(cuda(nids + 1)) if h_emb is not None else None
        idx = 0
        for i in range(nf.num_layers):
            hi = h[idx:nf.layer_size(i) + idx]
            idx += nf.layer_size(i)
            if self.use_feature:
                nf.layers[i].data['h_x'] = mix_embeddings(
                        hi, nf.layers[i].data, self.emb, self.proj)
            else:
                nf.layers[i].data['h_x'] = cuda(hi)
            nf.layers[i].data['deg'] = cuda(torch.ones(nf.layer_size(i)))
        nf.layers[0].data['h'] = nf.layers[0].data['h_x']

        for i in range(nf.num_blocks):
            parent_nid_1 = nf.layer_parent_nid(i + 1)
            parent_nid_0 = nf.layer_parent_nid(i)
            result = nf.layers[i].data['h'][nf.map_from_parent_nid(i, parent_nid_1)]
            nf.layers[i + 1].data['h'] = result
            result = self.convs[i].project(nf.layers[i].data['h'])
            nf.layers[i].data['h_q'] = result
            nf.block_compute(i, self.msg, self.red, self.convs[i])

        result = nf.layers[-1].data['h']
        assert (result != result).sum() == 0
        return result

