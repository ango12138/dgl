"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn
"""

import argparse
import itertools
import numpy as np
import time, gc
import tqdm
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import dgl
import dgl.function as fn
from dgl import DGLGraph
import tqdm

from dgl.data.knowledge_graph import load_data
from dgl.nn import RelGraphConv
from model import RelGraphEmbedLayer
from utils import thread_wrapped_func

class LinkPredict(nn.Module):
    def __init__(self,
                 device,
                 h_dim,
                 num_rels,
                 edge_rels,
                 num_bases=-1,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=True,
                 low_mem=False,
                 relation_regularizer = "bdd"):
        super(LinkPredict, self).__init__()
        self.device = th.device(device if device >= 0 else 'cpu')
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.edge_rels = edge_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.relation_regularizer = relation_regularizer

        self.w_relation = nn.Parameter(th.Tensor(num_rels, h_dim).to(self.device))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

        self.layers = nn.ModuleList()
        # h2h
        for idx in range(self.num_hidden_layers):
            act = F.relu if idx < self.num_hidden_layers - 1 else None
            self.layers.append(RelGraphConv(
                self.h_dim, self.h_dim, self.edge_rels, self.relation_regularizer,
                self.num_bases, activation=act, self_loop=self.use_self_loop,
                low_mem=low_mem, dropout=self.dropout))

    def forward(self, p_blocks, p_feats, n_blocks=None, n_feats=None, p_g=None, n_g=None, sample='neighbor'):
        p_h = p_feats

        # full subgraph
        if sample == 'path':
            p_blocks = p_blocks.int()
            p_blocks = p_blocks.to(self.device)
            for layer in self.layers:
                p_h = layer(p_blocks, p_h, p_blocks.edata['etype'], p_blocks.edata['norm'])
            return p_h

        for layer, block in zip(self.layers, p_blocks):
            block = block.int()
            block = block.to(self.device)
            p_h = layer(block, p_h, block.edata['etype'], block.edata['norm'])

        if n_blocks is None:
            return p_h

        n_h = n_feats
        for layer, block in zip(self.layers, n_blocks):
            block = block.int()
            block = block.to(self.device)
            n_h = layer(block, n_h, block.edata['etype'], block.edata['norm'])

        head, tail, eid = p_g.all_edges(form='all')
        p_head_emb = p_h[head]
        p_tail_emb = p_h[tail]
        rids = p_g.edata['etype']
        r_emb = self.w_relation[rids]
        head, tail = n_g.all_edges(form='uv')
        n_head_emb = n_h[head]
        n_tail_emb = n_h[tail]
        return (p_head_emb, p_tail_emb, r_emb, n_head_emb, n_tail_emb, p_h, n_h)

    def inference(self, g, in_feats, batch_size, device):
        """
        Inference with the RGCN model on full neighbor
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        x = in_feats
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.h_dim)

            sampler = NodeSampler(g, [None])
            dataloader = DataLoader(dataset=th.arange(g.number_of_nodes()),
                                    batch_size=batch_size,
                                    collate_fn=sampler.sample_nodes,
                                    shuffle=False,
                                    pin_memory=True,
                                    drop_last=False,
                                    num_workers=0)
            for seeds, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]
                block = block.int().to(device)
                idx = block.srcdata[dgl.NID]
                h = layer(block, x[idx].to(device), block.edata['etype'], block.edata['norm'])
                y[seeds] = h.cpu()

            x = y
        return y

def regularization_loss(n_emb, r_emb):
    return th.mean(n_emb.pow(2)) + \
            th.mean(r_emb.pow(2))

def calc_pos_score(h_emb, t_emb, r_emb):
    # DistMult
    score = th.sum(h_emb * r_emb * t_emb, dim=-1)
    return score

def calc_neg_tail_score(heads, tails, r_emb, num_chunks, chunk_size, neg_sample_size, device=None):
    hidden_dim = heads.shape[1]
    r = r_emb

    if device is not None:
        r = r.to(device)
        heads = heads.to(device)
        tails = tails.to(device)
    tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
    tails = th.transpose(tails, 1, 2)
    tmp = (heads * r).reshape(num_chunks, chunk_size, hidden_dim)
    return th.bmm(tmp, tails)

def calc_neg_head_score(heads, tails, r_emb, num_chunks, chunk_size, neg_sample_size, device=None):
    hidden_dim = tails.shape[1]
    r = r_emb
    if device is not None:
        r = r.to(device)
        heads = heads.to(device)
        tails = tails.to(device)
    heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
    heads = th.transpose(heads, 1, 2)
    tmp = (tails * r).reshape(num_chunks, chunk_size, hidden_dim)
    return th.bmm(tmp, heads)

def get_loss(h_emb, t_emb, nh_emb, nt_emb, r_emb,
    num_chunks, chunk_size, neg_sample_size):
    # triplets is a list of data samples (positive and negative)
    # each row in the triplets is a 3-tuple of (source, relation, destination)
    pos_score = calc_pos_score(h_emb, t_emb, r_emb).view(-1, 1)
    t_neg_score = calc_neg_tail_score(h_emb, nt_emb, r_emb, num_chunks, chunk_size, neg_sample_size)
    h_neg_score = calc_neg_head_score(nh_emb, t_emb, r_emb, num_chunks, chunk_size, neg_sample_size)
    #pos_score = F.logsigmoid(pos_score)
    h_neg_score = h_neg_score.view(-1, 1)
    t_neg_score = t_neg_score.view(-1, 1)
    #h_neg_score = F.logsigmoid(-h_neg_score).mean(dim=1)
    #t_neg_score = F.logsigmoid(-t_neg_score).mean(dim=1)

    score = th.cat([pos_score, h_neg_score, t_neg_score])
    label = th.cat([th.full((pos_score.shape), 1),
                    th.full((h_neg_score.shape), 0),
                    th.full((t_neg_score.shape), 0)]).to(score.device)
    predict_loss = F.binary_cross_entropy_with_logits(score, label)
    #pos_score = pos_score.mean()
    #h_neg_score = h_neg_score.mean()
    #t_neg_score = t_neg_score.mean()
    #predict_loss = -(2 * pos_score + h_neg_score + t_neg_score) / 2

    return predict_loss

class NodeSampler:
    def __init__(self, g, fanouts):
        self.g = g
        self.fanouts = fanouts

    def sample_nodes(self, seeds):
        seeds = th.tensor(seeds).long()
        seeds = seeds.squeeze()

        curr = seeds
        blocks = []
        for fanout in self.fanouts:
            if fanout is None or fanout == -1:
                frontier = dgl.in_subgraph(self.g, curr)
            else:
                frontier = dgl.sampling.sample_neighbors(self.g, curr, fanout)
            etypes = self.g.edata['etype'][frontier.edata[dgl.EID]]
            norm = self.g.edata['norm'][frontier.edata[dgl.EID]]
            block = dgl.to_block(frontier, curr)
            block.srcdata['ntype'] = self.g.ndata['ntype'][block.srcdata[dgl.NID]]
            block.srcdata['type_id'] = self.g.ndata['type_id'][block.srcdata[dgl.NID]]
            block.edata['etype'] = etypes[block.edata[dgl.EID]]
            block.edata['norm'] = norm[block.edata[dgl.EID]]
            curr = block.srcdata[dgl.NID]
            blocks.insert(0, block.int())

        return seeds, blocks

class LinkPathSampler:
    def __init__(self, g, num_rel):
        self.g = g
        self.num_rel = num_rel

    def sample_blocks(self, seeds):
        pseeds = th.tensor(seeds).long()
        pseeds = pseeds.squeeze()
        bsize = pseeds.shape[0]

        g = self.g
        subg = g.edge_subgraph(pseeds)
        p_u, p_v = subg.edges()
        p_rel = subg.edata['etype']
        org_nid = subg.ndata[dgl.NID]
        subg.ndata['type_id'] = g.ndata['type_id'][subg.ndata[dgl.NID]]

        # only half of the edges will be used as graph structure
        pos_idx = np.random.choice(bsize, size=bsize//2, replace=False)
        pos_idx = th.tensor(pos_idx).long()
        subg = dgl.remove_edges(subg, pos_idx)
        rels = subg.edata['etype']

        subg = dgl.add_reverse_edges(subg, copy_ndata=True)
        # dgl.NID and dgl.EID are not copy automatically
        subg.ndata['old_nid'] = org_nid
        subg.edata['etype'] = th.cat([rels, rels + self.num_rel])

        # calculate norm for compact_subg
        in_deg = subg.in_degrees(range(subg.number_of_nodes())).float()
        norm = 1.0 / in_deg
        norm[th.isinf(norm)] = 0
        subg.ndata['norm'] = norm.long()
        subg.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
        subg.edata['norm'] = subg.edata['norm'].unsqueeze(1)
        return (bsize, subg, p_u, p_v, p_rel)

class LinkNeighborSampler:
    def __init__(self, g, neg_edges, num_edges, num_neg, fanouts,
        is_train=True, keep_pos_edges=False):
        self.g = g
        self.neg_edges = neg_edges
        self.num_edges = num_edges
        self.num_neg = num_neg
        self.fanouts = fanouts
        self.is_train = is_train
        self.keep_pos_edges = keep_pos_edges

    def sample_blocks(self, seeds):
        pseeds = th.tensor(seeds).long()
        bsize = pseeds.shape[0]
        if self.num_neg is not None:
            nseeds = th.randint(self.num_edges, (self.num_neg,))
            nseeds = self.neg_edges[nseeds]
        else:
            nseeds = th.randint(self.num_edges, (bsize,))
            nseeds = self.neg_edges[nseeds]

        g = self.g
        fanouts = self.fanouts
        assert len(g.canonical_etypes) == 1
        p_g = g.edge_subgraph(pseeds)
        n_g = g.edge_subgraph(nseeds)

        pg_seed = p_g.ndata[dgl.NID]
        ng_seed = n_g.ndata[dgl.NID]

        p_blocks = []
        n_blocks = []
        p_curr = pg_seed
        n_curr = ng_seed
        for i, fanout in enumerate(fanouts):
            if fanout is None:
                p_frontier = dgl.in_subgraph(g, p_curr)
                n_frontier = dgl.in_subgraph(g, n_curr)
            else:
                p_frontier = dgl.sampling.sample_neighbors(g, p_curr, fanout)
                n_frontier = dgl.sampling.sample_neighbors(g, n_curr, fanout)

            '''
            if self.keep_pos_edges is False and \
                self.is_train and i == 0:
                old_frontier = p_frontier
                p_frontier = dgl.remove_edges(old_frontier, pseeds)
            '''
            p_etypes = g.edata['etype'][p_frontier.edata[dgl.EID]]
            # print(p_etypes)
            n_etypes = g.edata['etype'][n_frontier.edata[dgl.EID]]
            p_norm = g.edata['norm'][p_frontier.edata[dgl.EID]]
            n_norm = g.edata['norm'][n_frontier.edata[dgl.EID]]
            p_block = dgl.to_block(p_frontier, p_curr)
            n_block = dgl.to_block(n_frontier, n_curr)
            p_block.srcdata['ntype'] = g.ndata['ntype'][p_block.srcdata[dgl.NID]]
            n_block.srcdata['ntype'] = g.ndata['ntype'][n_block.srcdata[dgl.NID]]
            p_block.srcdata['type_id'] = g.ndata['type_id'][p_block.srcdata[dgl.NID]]
            n_block.srcdata['type_id'] = g.ndata['type_id'][n_block.srcdata[dgl.NID]]
            p_block.edata['etype'] = p_etypes
            n_block.edata['etype'] = n_etypes
            p_block.edata['norm'] = p_norm
            n_block.edata['norm'] = n_norm
            p_curr = p_block.srcdata[dgl.NID]
            n_curr = n_block.srcdata[dgl.NID]
            p_blocks.insert(0, p_block.int())
            n_blocks.insert(0, n_block.int())

        return (bsize, p_g, n_g, p_blocks, n_blocks)

def fullgraph_emb(g, embed_layer, model, node_feats, dim_size, device):
    in_feats = th.zeros(g.number_of_nodes(), dim_size)

    for i in range((g.number_of_nodes() + 1023) // 1024):
        idx = th.arange(start=i * 1024,
                        end=(i+1) * 1024 \
                            if (i+1) * 1024 < g.number_of_nodes() \
                            else g.number_of_nodes())
        in_feats[idx] = embed_layer(idx,
                                    g.ndata['ntype'][idx],
                                    g.ndata['type_id'][idx],
                                    node_feats).cpu()

    emb = model.inference(g, in_feats, 1024, device)
    return emb

def fullgraph_eval(train_g, g, embed_layer, model, device, node_feats,
    dim_size, pos_eids, neg_eids, neg_cnt, queue, filterred_test):
    model.eval()
    embed_layer.eval()
    t0 = time.time()
    logs = []

    with th.no_grad():
        embs = fullgraph_emb(train_g, embed_layer, model, node_feats, dim_size, device)
        mrr = 0
        mr = 0
        hit1 = 0
        hit3 = 0
        hit10 = 0
        pos_batch_size = 1024
        pos_cnt = pos_eids.shape[0]
        total_cnt = 0

        u, v = g.find_edges(pos_eids)
        # randomly select neg_cnt edges and corrupt them int neg heads and neg tails
        if neg_cnt > 0:
            total_neg_seeds = th.randint(neg_eids.shape[0], (neg_cnt * ((pos_cnt // pos_batch_size) + 1),))
            print(total_neg_seeds)
        # treat all nodes in head or tail as neg nodes
        else:
            n_u, n_v = g.find_edges(neg_eids)
            n_u = th.unique(n_u)
            n_v = th.unique(n_v)

        for p_i in range(int((pos_cnt + pos_batch_size - 1) // pos_batch_size)):
            print("Eval {}-{}".format(p_i * pos_batch_size,
                                      (p_i + 1) * pos_batch_size \
                                      if (p_i + 1) * pos_batch_size < pos_cnt \
                                      else pos_cnt))
            eids = pos_eids[p_i * pos_batch_size : \
                            (p_i + 1) * pos_batch_size \
                            if (p_i + 1) * pos_batch_size < pos_cnt \
                            else pos_cnt]
            su = u[p_i * pos_batch_size : \
                   (p_i + 1) * pos_batch_size \
                   if (p_i + 1) * pos_batch_size < pos_cnt \
                   else pos_cnt]
            sv = v[p_i * pos_batch_size : \
                   (p_i + 1) * pos_batch_size \
                   if (p_i + 1) * pos_batch_size < pos_cnt \
                   else pos_cnt]

            eids_t = g.edata['etype'][eids].to(device)
            phead_emb = embs[su].to(device)
            ptail_emb = embs[sv].to(device)
            if queue is None:
                rel_emb = model.w_relation[eids_t]
            else:
                rel_emb = model.module.w_relation[eids_t]

            pos_score = calc_pos_score(phead_emb, ptail_emb, rel_emb)
            #pos_score = F.logsigmoid(pos_score).reshape(phead_emb.shape[0], -1)

            if neg_cnt > 0:
                n_eids = neg_eids[total_neg_seeds[p_i * neg_cnt:(p_i + 1) * neg_cnt]]
                n_u, n_v = g.find_edges(n_eids)

            neg_batch_size = 10000
            head_neg_cnt = n_u.shape[0]
            tail_neg_cnt = n_v.shape[0]
            t_neg_score = []
            h_neg_score = []
            for n_i in range(int((head_neg_cnt + neg_batch_size - 1)//neg_batch_size)):
                sn_u = n_u[n_i * neg_batch_size : \
                           (n_i + 1) * neg_batch_size \
                           if (n_i + 1) * neg_batch_size < head_neg_cnt
                           else head_neg_cnt]
                nhead_emb = embs[sn_u].to(device)
                h_neg_score.append(calc_neg_head_score(nhead_emb,
                                                       ptail_emb,
                                                       rel_emb,
                                                       1,
                                                       ptail_emb.shape[0],
                                                       nhead_emb.shape[0]).reshape(-1, nhead_emb.shape[0]))

            for n_i in range(int((tail_neg_cnt + neg_batch_size - 1)//neg_batch_size)):
                sn_v = n_v[n_i * neg_batch_size : \
                           (n_i + 1) * neg_batch_size \
                           if (n_i + 1) * neg_batch_size < tail_neg_cnt
                           else tail_neg_cnt]
                ntail_emb = embs[sn_v].to(device)
                t_neg_score.append(calc_neg_tail_score(phead_emb,
                                                       ntail_emb,
                                                       rel_emb,
                                                       1,
                                                       phead_emb.shape[0],
                                                       ntail_emb.shape[0]).reshape(-1, ntail_emb.shape[0]))
            t_neg_score = th.cat(t_neg_score, dim=1)
            h_neg_score = th.cat(h_neg_score, dim=1)

            if filterred_test:
                # exclude false neg edges
                for idx in range(eids.shape[0]):
                    tail_pos = g.has_edges_between(th.full((n_v.shape[0],), su[idx], dtype=th.long), n_v)
                    head_pos = g.has_edges_between(n_u, th.full((n_u.shape[0],), sv[idx], dtype=th.long))
                    etype = g.edata['etype'][eids[idx]]

                    loc = tail_pos == 1
                    u_ec = su[idx]
                    n_v_ec = n_v[loc]
                    false_neg_comp = th.full((n_v_ec.shape[0],), 0, device=device)
                    for idx_ec in range(n_v_ec.shape[0]):
                        sn_v_ec = n_v_ec[idx_ec]
                        _, _, eid_ec = g.edge_ids(u_ec.unsqueeze(dim=0), sn_v_ec.unsqueeze(dim=0), return_uv=True)
                        etype_ec = g.edata['etype'][eid_ec]
                        loc_ec = etype_ec == etype
                        eid_ec = eid_ec[loc_ec]
                        # has edge
                        if eid_ec.shape[0] > 0:
                            false_neg_comp[idx_ec] = -th.abs(pos_score[idx])
                    t_neg_score[idx][loc] += false_neg_comp

                    loc = head_pos == 1
                    n_u_ec = n_u[loc]
                    v_ec = sv[idx]
                    false_neg_comp = th.full((n_u_ec.shape[0],), 0, device=device)
                    for idx_ec in range(n_u_ec.shape[0]):
                        sn_u_ec = n_u_ec[idx_ec]
                        _, _, eid_ec = g.edge_ids(sn_u_ec.unsqueeze(dim=0), v_ec.unsqueeze(dim=0), return_uv=True)
                        etype_ec = g.edata['etype'][eid_ec]
                        loc_ec = etype_ec == etype
                        eid_ec = eid_ec[loc_ec]
                        # has edge
                        if eid_ec.shape[0] > 0:
                            false_neg_comp[idx_ec] = -th.abs(pos_score[idx])
                    h_neg_score[idx][loc] += false_neg_comp

            pos_score = pos_score.view(-1,1)
            # perturb object
            scores = th.cat([pos_score, h_neg_score], dim=1)
            scores = th.sigmoid(scores)
            _, indices = th.sort(scores, dim=1, descending=True)
            #rankings = th.sum(neg_scores >= pos_scores, dim=1) + 1
            indices = th.nonzero(indices == 0)
            rankings = indices[:, 1].view(-1) + 1
            rankings = rankings.cpu().detach().numpy()
            for ranking in rankings:
                logs.append({
                    'MRR': 1.0 / ranking,
                    'MR': float(ranking),
                    'HITS@1': 1.0 if ranking <= 1 else 0.0,
                    'HITS@3': 1.0 if ranking <= 3 else 0.0,
                    'HITS@10': 1.0 if ranking <= 10 else 0.0
                })

            # perturb subject
            scores = th.cat([pos_score, t_neg_score], dim=1)
            scores = th.sigmoid(scores)
            _, indices = th.sort(scores, dim=1, descending=True)
            #rankings = th.sum(neg_scores >= pos_scores, dim=1) + 1
            indices = th.nonzero(indices == 0)
            rankings = indices[:, 1].view(-1) + 1
            rankings = rankings.cpu().detach().numpy()
            for ranking in rankings:
                logs.append({
                    'MRR': 1.0 / ranking,
                    'MR': float(ranking),
                    'HITS@1': 1.0 if ranking <= 1 else 0.0,
                    'HITS@3': 1.0 if ranking <= 3 else 0.0,
                    'HITS@10': 1.0 if ranking <= 10 else 0.0
                })

    if queue is not None:
        queue.put(logs)
    else:
        metrics = {}
        if len(logs) > 0:
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        for k, v in metrics.items():
            print('Average {}: {}'.format(k, v))
    t1 = time.time()
    print("Full eval {} exmpales takes {} seconds".format(pos_eids.shape[0], t1 - t0))

@thread_wrapped_func
def run(proc_id, n_gpus, args, devices, dataset, pos_seeds, neg_seeds, queue=None):
    dev_id = devices[proc_id]
    train_g, valid_g, test_g, num_rels, edge_rels = dataset
    train_seeds, valid_seeds, test_seeds = pos_seeds
    train_neg_seeds, valid_neg_seeds, test_neg_seeds = neg_seeds

    batch_size = args.batch_size
    chunk_size = args.chunk_size
    fanouts = [args.fanout if args.fanout > 0 else None] * args.n_layers

    num_train_edges = train_neg_seeds.shape[0]
    node_tids = test_g.ndata['ntype']
    num_of_ntype = int(th.max(node_tids).numpy()) + 1

    # check cuda
    use_cuda = dev_id >= 0
    if use_cuda:
        th.cuda.set_device(dev_id)
        node_tids = node_tids.to(dev_id)

    # build dataloader for training, validation
    if args.sampler == 'neighbor':
        train_sampler = LinkNeighborSampler(train_g,
                                        train_neg_seeds,
                                        num_train_edges,
                                        num_neg=None,
                                        fanouts=fanouts)
    elif args.sampler == 'path':
        train_sampler = LinkPathSampler(train_g,
                                        num_rel=num_rels)
    else:
        assert 'Unsupported sampler {}'.format(args.sampler)
    dataloader = DataLoader(dataset=train_seeds,
                            batch_size=batch_size,
                            collate_fn=train_sampler.sample_blocks,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            num_workers=args.num_workers)

    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=dev_id)

    embed_layer = RelGraphEmbedLayer(dev_id,
                                     test_g.number_of_nodes(),
                                     node_tids,
                                     num_of_ntype,
                                     [None] * num_of_ntype,
                                     args.n_hidden)

    model = LinkPredict(dev_id,
                        args.n_hidden,
                        num_rels,
                        edge_rels,
                        num_bases=args.n_bases,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        use_self_loop=args.use_self_loop,
                        low_mem=args.low_mem,
                        relation_regularizer=args.relation_regularizer)
    if dev_id >= 0:
        model.cuda(dev_id)
        if args.mix_cpu_gpu is False:
            embed_layer.cuda(dev_id)
            embed_layer.node_embeds.cuda(dev_id)

    if n_gpus > 1:
        embed_layer = DistributedDataParallel(embed_layer, device_ids=[dev_id], output_device=dev_id)
        model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id, find_unused_parameters=True)

    # optimizer
    all_params = itertools.chain(model.parameters(), embed_layer.parameters())
    optimizer = th.optim.Adam(all_params, lr=args.lr)
    node_feats = [None] * num_of_ntype

    print("start training...")
    for epoch in range(args.n_epochs):
        model.train()
        if epoch > 1:
            t0 = time.time()
        for i, sample_data in enumerate(dataloader):
            if args.sampler == 'neighbor':
                bsize, p_g, n_g, p_blocks, n_blocks = sample_data
                p_feats = embed_layer(p_blocks[0].srcdata[dgl.NID],
                                      p_blocks[0].srcdata['ntype'],
                                      p_blocks[0].srcdata['type_id'],
                                      node_feats)
                n_feats = embed_layer(n_blocks[0].srcdata[dgl.NID],
                                      n_blocks[0].srcdata['ntype'],
                                      n_blocks[0].srcdata['type_id'],
                                      node_feats)

                p_head_emb, p_tail_emb, r_emb, n_head_emb, n_tail_emb, p_emb, n_emb = \
                    model(p_blocks, p_feats, n_blocks, n_feats, p_g, n_g)
            else:
                bsize, g, p_u, p_v, rids = sample_data
                in_feats = embed_layer(g.ndata['old_nid'],
                                       g.ndata['ntype'],
                                       g.ndata['type_id'],
                                       node_feats)
                mb_feats = model(g, in_feats, sample=args.sampler)
                p_head_emb = mb_feats[p_u]
                p_tail_emb = mb_feats[p_v]

                nh_idx = th.randint(low=0, high=mb_feats.shape[0], size=(p_head_emb.shape[0],))
                nt_idx = th.randint(low=0, high=mb_feats.shape[0], size=(p_tail_emb.shape[0],))
                n_head_emb = mb_feats[nh_idx]
                n_tail_emb = mb_feats[nt_idx]
                r_emb = model.w_relation[rids]

            pred_loss = get_loss(p_head_emb,
                                p_tail_emb,
                                n_head_emb,
                                n_tail_emb,
                                r_emb,
                                int(batch_size / chunk_size),
                                chunk_size,
                                chunk_size)

            if args.sampler == 'neighbor':
                if queue is None:
                    reg_loss = regularization_loss(th.cat([p_feats, n_feats]), model.w_relation)
                else:
                    reg_loss = regularization_loss(th.cat([p_feats, n_feats]), model.module.w_relation)
            else:
                if queue is None:
                    reg_loss = regularization_loss(mb_feats, model.w_relation)
                else:
                    reg_loss = regularization_loss(mb_feats, model.module.w_relation)
            loss = pred_loss + args.regularization_coef * reg_loss

            print("pos {}, reg_loss {}".format(pred_loss.detach(),
                                               args.regularization_coef * reg_loss.detach()))
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            th.nn.utils.clip_grad_norm_(embed_layer.parameters(), args.grad_norm)
            optimizer.step()
            t1 = time.time()

            print("Epoch {}, Iter {}, Loss:{}".format(epoch, i, loss.detach()))

        if epoch > 1:
            dur = t1 - t0
            print("Epoch {} takes {} seconds".format(epoch, dur))

        if epoch > 1 and epoch % args.evaluate_every == 0:
            # We use multi-gpu evaluation to speed things up
            fullgraph_eval(train_g,
                        valid_g,
                        embed_layer,
                        model,
                        dev_id,
                        node_feats,
                        args.n_hidden,
                        valid_seeds,
                        valid_neg_seeds,
                        args.valid_neg_cnt,
                        queue,
                        False)
            if proc_id == 0 and queue is not None:
                logs = []
                for i in range(n_gpus):
                    log = queue.get()
                    logs = logs + log

                metrics = {}
                for metric in logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
                print("-------------- Eval result --------------")
                for k, v in metrics.items():
                    print('Eval average {} : {}'.format(k, v))
                print("-----------------------------------------")

        if n_gpus > 1:
            th.distributed.barrier()

    print()
    if args.model_path is not None:
        th.save(model.state_dict(), args.model_path)

    # We use multi-gpu testing to speed things up
    fullgraph_eval(train_g,
                   test_g,
                   embed_layer,
                   model,
                   dev_id,
                   node_feats,
                   args.n_hidden,
                   test_seeds,
                   test_neg_seeds,
                   args.test_neg_cnt,
                   queue,
                   not args.no_test_filter)
    if proc_id == 0 and queue is not None:
        logs = []
        for i in range(n_gpus):
            log = queue.get()
            logs = logs + log

        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        print("-------------- Test result --------------")
        for k, v in metrics.items():
            print('Test average {} : {}'.format(k, v))
        print("-----------------------------------------")

def main(args, devices):
    # load graph data
    dataset = load_data(args.dataset)
    g = dataset[0]
    # wn18, fb15k has only one node type.
    g.ndata['type_id'] = th.arange(g.number_of_nodes())
    num_nodes = dataset.num_nodes
    train_mask = g.edata.pop('train_mask')
    valid_mask = g.edata.pop('val_mask')
    test_mask = g.edata.pop('test_mask')

    train_g = g.edge_subgraph(train_mask, preserve_nodes=True)
    valid_g = g.edge_subgraph((train_mask | valid_mask), preserve_nodes=True)
    test_g  = g

    # train_g
    train_seed_mask = train_g.edata.pop('train_edge_mask')
    train_seeds = th.nonzero(train_seed_mask).squeeze()
    train_g.edata.pop('valid_edge_mask')
    train_g.edata.pop('test_edge_mask')

    # valid_g
    t_seed_mask = valid_g.edata.pop('train_edge_mask')
    val_seed_mask = valid_g.edata.pop('valid_edge_mask')
    valid_seeds = th.nonzero(val_seed_mask).squeeze()
    valid_neg_seeds = th.nonzero(t_seed_mask | val_seed_mask).squeeze()
    valid_g.edata.pop('test_edge_mask')

    # test_g
    t_seed_mask = test_g.edata.pop('train_edge_mask')
    v_seed_mask = test_g.edata.pop('valid_edge_mask')
    test_seed_mask = test_g.edata.pop('test_edge_mask')
    test_seeds = th.nonzero(test_seed_mask).squeeze()
    test_neg_seeds = th.nonzero(t_seed_mask | v_seed_mask | test_seed_mask).squeeze()
    num_rels = dataset.num_rels
    edge_rels = num_rels * 2 # we add reverse edges

    print("Train pos edges #{}".format(train_seeds.shape[0]))
    print("Valid pos edges #{}".format(valid_seeds.shape[0]))
    print("Test pos edges #{}".format(test_seeds.shape[0]))
    print("Valid neg edges #{}".format(valid_neg_seeds.shape[0]))
    print("Test neg edges #{}".format(test_neg_seeds.shape[0]))

    train_shuffle = th.randperm(train_seeds.shape[0])
    train_seeds = train_seeds[train_shuffle]

    u, v, eid = train_g.all_edges(form='all')
    # calculate norm for each edge type and store in edge
    if args.global_norm:
        _, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = th.ones(v.shape[0]) / degrees
        norm = norm.unsqueeze(1)
        train_g.edata['norm'] = norm.long()
    else:
        train_g.edata['norm'] = th.full((eid.shape[0],1), 1)
        for rel_id in range(edge_rels):
            idx = (train_g.edata['etype'] == rel_id)
            u_r = u[idx]
            v_r = v[idx]
            _, inverse_index, count = th.unique(v_r, return_inverse=True, return_counts=True)
            degrees = count[inverse_index]
            norm = th.ones(v_r.shape[0]) / degrees
            norm = norm.unsqueeze(1)
            train_g.edata['norm'][idx] = norm.long()
    train_g.edata['norm'].share_memory_()
    train_g.edata['etype'].share_memory_()
    train_g.ndata['ntype'].share_memory_()

    # get valid set
    u, v, eid = valid_g.all_edges(form='all')
    # calculate norm for each edge type and store in edge
    if args.global_norm:
        _, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = th.ones(v.shape[0]) / degrees
        norm = norm.unsqueeze(1)
        valid_g.edata['norm'] = norm.long()
    else:
        valid_g.edata['norm'] = th.full((eid.shape[0],1), 1).long()
        for rel_id in range(edge_rels):
            idx = (valid_g.edata['etype'] == rel_id)
            u_r = u[idx]
            v_r = v[idx]
            _, inverse_index, count = th.unique(v_r, return_inverse=True, return_counts=True)
            degrees = count[inverse_index]
            norm = th.ones(v_r.shape[0]) / degrees
            norm = norm.unsqueeze(1)
            valid_g.edata['norm'][idx] = norm.long()
    valid_g.edata['norm'].share_memory_()
    valid_g.edata['etype'].share_memory_()
    valid_g.ndata['ntype'].share_memory_()

    # get test set
    u, v, eid = test_g.all_edges(form='all')
    # calculate norm for each edge type and store in edge
    if args.global_norm:
        _, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = th.ones(v.shape[0]) / degrees
        norm = norm.unsqueeze(1)
        test_g.edata['norm'] = norm.long()
    else:
        test_g.edata['norm'] = th.full((eid.shape[0],1), 1).long()
        for rel_id in range(edge_rels):
            idx = (test_g.edata['etype'] == rel_id)
            u_r = u[idx]
            v_r = v[idx]
            _, inverse_index, count = th.unique(v_r, return_inverse=True, return_counts=True)
            degrees = count[inverse_index]
            norm = th.ones(v_r.shape[0]) / degrees
            norm = norm.unsqueeze(1)
            test_g.edata['norm'][idx] = norm.long()
    test_g.edata['norm'].share_memory_()
    test_g.edata['etype'].share_memory_()
    test_g.ndata['ntype'].share_memory_()

    train_g.create_format_()
    valid_g.create_format_()
    test_g.create_format_()

    n_gpus = len(devices)
    # cpu
    if devices[0] == -1:
        run(0, 0, args, ['cpu'], (train_g, valid_g, test_g, num_rels, edge_rels),
            (train_seeds, valid_seeds, test_seeds),
            (train_seeds, valid_neg_seeds, test_neg_seeds))
    # gpu
    elif n_gpus == 1:
        run(0, n_gpus, args, devices, (train_g, valid_g, test_g, num_rels, edge_rels),
            (train_seeds, valid_seeds, test_seeds),
            (train_seeds, valid_neg_seeds, test_neg_seeds))
    # multi gpu
    else:
        queue = mp.Queue(n_gpus)
        procs = []
        num_train_seeds = train_seeds.shape[0]
        num_valid_seeds = valid_seeds.shape[0]
        num_test_seeds = test_seeds.shape[0]
        tseeds_per_proc = num_train_seeds // n_gpus
        vseeds_per_proc = num_valid_seeds // n_gpus
        tstseeds_per_proc = num_test_seeds // n_gpus
        for proc_id in range(n_gpus):
            # we have multi-gpu for training, evaluation and testing
            # so split trian set, valid set and test set into num-of-gpu parts.
            proc_train_seeds = train_seeds[proc_id * tseeds_per_proc :
                                           (proc_id + 1) * tseeds_per_proc \
                                           if (proc_id + 1) * tseeds_per_proc < num_train_seeds \
                                           else num_train_seeds]
            proc_valid_seeds = valid_seeds[proc_id * vseeds_per_proc :
                                           (proc_id + 1) * vseeds_per_proc \
                                           if (proc_id + 1) * vseeds_per_proc < num_valid_seeds \
                                           else num_valid_seeds]
            proc_test_seeds = test_seeds[proc_id * tstseeds_per_proc :
                                         (proc_id + 1) * tstseeds_per_proc \
                                         if (proc_id + 1) * tstseeds_per_proc < num_test_seeds \
                                         else num_test_seeds]

            p = mp.Process(target=run, args=(proc_id, n_gpus, args, devices,
                           (train_g, valid_g, test_g, num_rels, edge_rels),
                           (proc_train_seeds, proc_valid_seeds, proc_test_seeds),
                           (train_seeds, valid_neg_seeds, test_neg_seeds),
                           queue))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

def config():
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=200,
            help="number of hidden units")
    parser.add_argument("--gpu", type=str, default='0',
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=10,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=20,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--model_path", type=str, default=None,
            help='path for save the model')
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    parser.add_argument("--batch-size", type=int, default=1024,
            help="Mini-batch size.")
    parser.add_argument("--chunk-size", type=int, default=32,
            help="Negative sample chunk size. In each chunk, positive pairs will share negative edges")
    parser.add_argument("--fanout", type=int, default=10,
            help="Fan-out of neighbor sampling.")
    parser.add_argument("--global-norm", default=False, action='store_true',
            help="Whether we use normalization of global graph or bipartite relation graph")
    parser.add_argument("--regularization-coef", type=float, default=0.001,
            help="Regularization Coeffiency.")
    parser.add_argument("--grad-norm", type=float, default=1.0,
            help="norm to clip gradient to")
    parser.add_argument("--valid-neg-cnt", type=int, default=1000,
            help="Validation negative sample cnt.")
    parser.add_argument("--test-neg-cnt", type=int, default=-1,
            help="Test negative sample cnt.")
    parser.add_argument("--num-workers", type=int, default=0,
            help="Number of workers for dataloader.")
    parser.add_argument("--low-mem", default=False, action='store_true',
            help="Whether use low mem RelGraphCov")
    parser.add_argument("--mix-cpu-gpu", default=False, action='store_true',
            help="Whether store node embeddins in cpu")
    parser.add_argument("--no-test-filter", default=False, action='store_true',
            help="Whether we do filterred evaluation in test")
    parser.add_argument("--relation-regularizer", default='basis',
            help="Relation weight regularizer")
    parser.add_argument("--sampler", type=str, default='neighbor',
            help="subgraph sampler")
    parser.add_argument("--evaluate-every", type=int, default=10,
            help="perform evaluation every n epochs")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = config()
    devices = list(map(int, args.gpu.split(',')))
    print(args)
    assert args.batch_size > 0
    main(args, devices)