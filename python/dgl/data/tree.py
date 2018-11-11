"""Tree-structured data.

Including:
    - Stanford Sentiment Treebank
"""
from __future__ import absolute_import

from collections import namedtuple, OrderedDict
from nltk.tree import Tree
from nltk.corpus.reader import BracketParseCorpusReader
import networkx as nx

import numpy as np
import os
import dgl
import dgl.backend as F
from dgl.data.utils import download, extract_archive, get_download_dir

_urls = {
    'sst' : 'https://www.dropbox.com/s/w0b4fka64096wqz/sst.zip?dl=1',
}

SSTBatch = namedtuple('SSTBatch', ['graph', 'nid_with_word', 'wordid', 'label'])

class SST(object):
    """Stanford Sentiment Treebank dataset.

    Each sample is the constituency tree of a sentence. The leaf nodes
    represent words. The word is a int value stored in the "x" feature field.
    The non-leaf node has a special value PAD_WORD.
    Each node also has a sentiment annotation: 5 classes (very negative,
    negative, neutral, positive and very positive). The sentiment label is a
    int value stored in the "y" feature field.

    .. note::
        This dataset class is compatible with pytorch's Dataset class.

    .. note::
        All the samples will be loaded and preprocessed in the memory first.
    
    Parameters
    ----------
    mode : str, optional
        Can be 'train', 'val', 'test'. Which data file to use.
    vocab_file : str, optional
        Optional vocabulary file.
    """
    PAD_WORD=-1  # special pad word id
    UNK_WORD=-1  # out-of-vocabulary word id
    def __init__(self, mode='train', vocab_file=None):
        self.mode = mode
        self.dir = get_download_dir()
        self.zip_file_path='{}/sst.zip'.format(self.dir)
        self.pretrained_file = 'glove.840B.300d.txt' if mode == 'train' else ''
        self.pretrained_emb = None
        self.vocab_file = '{}/sst/vocab.txt'.format(self.dir) if vocab_file is None else vocab_file
        download(_urls['sst'], path=self.zip_file_path)
        extract_archive(self.zip_file_path, '{}/sst'.format(self.dir))
        self.trees = []
        self.num_classes = 5
        print('Preprocessing...')
        self._load()
        print('Dataset creation finished. #Trees:', len(self.trees))

    def _load(self):
        # load vocab file
        self.vocab = OrderedDict()
        with open(self.vocab_file) as vf:
            for line in vf.readlines():
                line = line.strip()
                self.vocab[line] = len(self.vocab)

        # filter glove
        if self.pretrained_file != '' and os.path.exists(self.pretrained_file):
            glove_emb = {}
            with open(self.pretrained_file, 'r') as pf:
                for line in pf.readlines():
                    sp = line.split(' ')
                    if sp[0].lower() in self.vocab:
                        glove_emb[sp[0].lower()] = np.array([float(x) for x in sp[1:]])
        files = ['{}.txt'.format(self.mode)]
        corpus = BracketParseCorpusReader('{}/sst'.format(self.dir), files)
        sents = corpus.parsed_sents(files[0])

        #initialize with glove
        pretrained_emb = []
        fail_cnt = 0
        for line in self.vocab.keys():
            if self.pretrained_file != '' and os.path.exists(self.pretrained_file):
                if not line.lower() in glove_emb:
                    fail_cnt += 1
                pretrained_emb.append(glove_emb.get(line.lower(), np.random.uniform(-0.05, 0.05, 300)))

        if self.pretrained_file != '' and os.path.exists(self.pretrained_file):
            self.pretrained_emb = F.tensor(np.stack(pretrained_emb, 0))
            print('Miss word in GloVe {0:.4f}'.format(1.0*fail_cnt/len(self.pretrained_emb)))
        # build trees
        for sent in sents:
            self.trees.append(self._build_tree(sent))


    def _build_tree(self, root):
        g = nx.DiGraph()
        def _rec_build(nid, node):
            for child in node:
                cid = g.number_of_nodes()
                if isinstance(child[0], str) or isinstance(child[0], bytes):
                    # leaf node
                    word = self.vocab.get(child[0].lower(), SST.UNK_WORD)
                    g.add_node(cid, x=word, y=int(child.label()))
                else:
                    g.add_node(cid, x=SST.PAD_WORD, y=int(child.label()))
                    _rec_build(cid, child)
                g.add_edge(cid, nid)
        # add root
        g.add_node(0, x=SST.PAD_WORD, y=int(root.label()))
        _rec_build(0, root)
        ret = dgl.DGLGraph()
        ret.from_networkx(g, node_attrs=['x', 'y'])
        return ret

    def __getitem__(self, idx):
        return self.trees[idx]

    def __len__(self):
        return len(self.trees)

    @property 
    def num_vocabs(self):
        return len(self.vocab)

    @staticmethod
    def batcher(batch):
        nid_with_word = []
        wordid = []
        label = []
        gnid = 0
        for tree in batch:
            for nid in range(tree.number_of_nodes()):
                if tree.nodes[nid].data['x'] != SST.PAD_WORD:
                    nid_with_word.append(gnid)
                    wordid.append(tree.nodes[nid].data['x'])
                label.append(tree.nodes[nid].data['y'])
                gnid += 1
        batch_trees = dgl.batch(batch)
        return SSTBatch(graph=batch_trees,
                        nid_with_word=F.tensor(nid_with_word, dtype=F.int64),
                        wordid=F.tensor(wordid, dtype=F.int64),
                        label=F.tensor(label, dtype=F.int64))
