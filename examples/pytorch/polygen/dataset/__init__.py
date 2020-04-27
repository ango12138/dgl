from .vertexgraph import *
from .facegraph import *
from .fields import *
from .utils import prepare_dataset
from .preprocess_mesh import preprocess as preprocess_mesh_obj
import os
import random

class ClassificationDataset(object):
    "Dataset class for classification task."
    def __init__(self):
        raise NotImplementedError

class ShapeNetVertexDataset(object):
    '''
    Dataset class for ShapeNet Vertex.
    '''
    COORD_BIN = 32
    INIT_BIN = COORD_BIN
    EOS_BIN = COORD_BIN + 1
    PAD_BIN = COORD_BIN + 2
    MAX_VERT_LENGTH = 98
    MAX_LENGTH = MAX_VERT_LENGTH * 3 + 1
    
    def __init__(self, dataset_list_file ='all_file_list.txt'):
        dataset_list_dir = '/home/ubuntu/data/new/ShapeNetCore.v2/'
        dataset_list_path = os.path.join(dataset_list_dir, dataset_list_file)
        # train
        with open(dataset_list_path+'.train', 'r') as f:
            self.train_dataset_list = f.readlines()
        # test
        with open(dataset_list_path+'.test', 'r') as f:
            self.test_dataset_list = f.readlines()
 
        with open(dataset_list_path, 'r') as f:
            self.dataset_list = f.readlines()
        self.pad_id = self.PAD_BIN
        # We don't need field since we handle them in preprocess
        # self.tgt_field = Field(np.range(COORD_BIN+2),
        #                       preprocessing=lambda seq: [self.INIT_INDEX] + seq + [self.EOS_INDEX],
        #                       postprocessing=strip_func)
    
    @property
    def vocab_size(self):
        return self.COORD_BIN+3 

    def __call__(self, graph_pool, mode='train', batch_size=32, k=1,
                 device='cpu', dev_rank=0, ndev=1):
        '''
        Create a batched graph correspond to the mini-batch of the dataset.
        args:
            graph_pool: a GraphPool object for accelerating.
            mode: train/valid/test
            batch_size: batch size
            k: beam size(only required for test)
            device: str or torch.device
            dev_rank: rank (id) of current device
            ndev: number of devices
        '''
        if mode == 'train':
            dataset_list = self.train_dataset_list
        else:
            dataset_list = self.test_dataset_list

        n = len(dataset_list)
        # make sure all devices have the same number of batch
        n = n // ndev * ndev

        # XXX: partition then shuffle may not be equivalent to shuffle then
        # partition
        order = list(range(dev_rank, n, ndev))
        np.random.shuffle(order)

        tgt_buf = []

        for idx in order:
            obj_file = dataset_list[idx].strip()
            verts, faces = preprocess_mesh_obj(obj_file)
            # Flattern verts, order Y(up), X(front), Z(right)
            reordered_verts = np.zeros_like(verts)
            reordered_verts[:,0] = verts[:,1]
            reordered_verts[:,1] = verts[:,0]
            reordered_verts[:,2] = verts[:,2]
            flattern_verts = [self.INIT_BIN] + reordered_verts.flatten().astype(np.int64).tolist() + [self.EOS_BIN]
            # exp
            if len(flattern_verts) > self.MAX_LENGTH:
                continue
                flattern_verts = flattern_verts[:self.MAX_LENGTH-1] + [self.EOS_BIN]
            tgt_buf.append(flattern_verts)
            if len(tgt_buf) == batch_size:
                if mode == 'infer':
                    yield graph_pool.beam(tgt_buf, self.INIT_BIN, self.MAX_LENGTH, k, device=device)
                else:
                    yield graph_pool(tgt_buf, device=device)
                tgt_buf = []

        if len(tgt_buf) != 0:
            # NOTE: currently if tgt_buf == 0, will have bug.
            # Issue: https://github.com/dmlc/dgl/issues/1475
            if len(tgt_buf) > 1:
                if mode == 'infer':
                    yield graph_pool.beam(self.INIT_BIN, self.MAX_LENGTH, k, device=device)
                else:
                    yield graph_pool(tgt_buf, device=device)

    def get_sequence(self, batch):
        "return a list of sequence from a list of index arrays"
        return batch


class ShapeNetFaceDataset(object):
    '''
    Dataset class for ShapeNet Vertex.
    '''
    COORD_BIN = 32
    INIT_BIN = COORD_BIN
    EOS_BIN = COORD_BIN + 1
    PAD_BIN = COORD_BIN + 2
    #MAX_VERT_LENGTH = 133
    MAX_VERT_LENGTH = 78
    START_FACE_VERT_IDX = 0
    STOP_FACE_VERT_IDX = 1
    FACE_VERT_OFFSET = STOP_FACE_VERT_IDX + 1
    MAX_FACE_LENGTH = (800 + 2) // 3
    
    def __init__(self, dataset_list_file ='all_file_list.txt'):
    #def __init__(self, dataset_list_file ='table_chair.txt'):
        dataset_list_dir = '/home/ubuntu/data/new/ShapeNetCore.v2/'
        dataset_list_path = os.path.join(dataset_list_dir, dataset_list_file)
        # train
        with open(dataset_list_path+'.train', 'r') as f:
        #with open(dataset_list_path, 'r') as f:
            self.train_dataset_list = f.readlines()
        # test
        with open(dataset_list_path+'.test', 'r') as f:
        #with open(dataset_list_path, 'r') as f:
            self.test_dataset_list = f.readlines()
 
        self.pad_id = self.PAD_BIN
        # We don't need field since we handle them in preprocess
        # self.tgt_field = Field(np.range(COORD_BIN+2),
        #                       preprocessing=lambda seq: [self.INIT_INDEX] + seq + [self.EOS_INDEX],
        #                       postprocessing=strip_func)
   

    def random_batch(self, graph_pool, mode='train', batch_size=32, k=1,
                     device='cpu', dev_rank=0, ndev=1):
        '''
        Return a random batch, reorder every time

        '''
        tmp_iter = self.__call__(graph_pool, mode=mode, batch_size=batch_size, k=k, device=device,
                                 dev_rank=dev_rank, ndev=ndev)
        for i, g in enumerate(tmp_iter):
            return g

    def __call__(self, graph_pool, mode='train', batch_size=32, k=1,
                 device='cpu', dev_rank=0, ndev=1):
        '''
        Create a batched graph correspond to the mini-batch of the dataset.
        args:
            graph_pool: a GraphPool object for accelerating.
            mode: train/valid/test
            batch_size: batch size
            k: beam size(only required for test)
            device: str or torch.device
            dev_rank: rank (id) of current device
            ndev: number of devices
        '''
        if mode == 'train':
            dataset_list = self.train_dataset_list
        else:
            dataset_list = self.test_dataset_list

        n = len(dataset_list)
        # make sure all devices have the same number of batch
        n = n // ndev * ndev

        # XXX: partition then shuffle may not be equivalent to shuffle then
        # partition
        order = list(range(dev_rank, n, ndev))
        #if mode == 'train':
        np.random.shuffle(order)

        src_buf, tgt_buf = [], []

        for idx in order:
            #same_idx = 0
            #obj_file = dataset_list[same_idx].strip()
            obj_file = dataset_list[idx].strip()
            verts, faces = preprocess_mesh_obj(obj_file)
            # Flattern verts, order Y(up), X(front), Z(right)
            reordered_verts = np.zeros_like(verts)
            reordered_verts[:,0] = verts[:,1]
            reordered_verts[:,1] = verts[:,0]
            reordered_verts[:,2] = verts[:,2]
            # pad start and end bin at the front
            st_end_verts = np.array([[self.INIT_BIN] * 3, [self.EOS_BIN] * 3])
            full_verts = np.concatenate([st_end_verts, reordered_verts], axis=0).astype(np.int64)
            # verts
            if full_verts.shape[0] > self.MAX_VERT_LENGTH + 2:
                continue
            # NOTES: we are padding for now. but the padding can be applied in final decoder recv phase
            #padded_verts = (np.ones(self.MAX_VERT_LENGTH + 2, 3) * self.PAD_BIN).astype(np.int64)
            #padded_verts[:full_verts.shape[0], :] = full_verts
            src_buf.append(full_verts)

            # faces go to 800
            # START_FACE_BIN is num_of_node + 1, STOP_FACE_BIN is num_of_node + 2
            # TODO: does it matter to put the START/STOP at the beginning of the sequence or the end?
            # I suppose doesn't matter. Let's first try append at the beginning.
            faces += self.FACE_VERT_OFFSET
            flattern_faces = [self.START_FACE_VERT_IDX] + faces.flatten().astype(np.int64).tolist() + [self.STOP_FACE_VERT_IDX]
            # -1 for considering the FACE_EOS_BIN
            if faces.shape[0] > self.MAX_FACE_LENGTH:
                continue

            tgt_buf.append(flattern_faces)
            if len(tgt_buf) == batch_size:
                if mode == 'test':
                    #yield graph_pool.beam(self.sos_id, self.MAX_LENGTH, k, device=device)
                    yield graph_pool(src_buf, tgt_buf, device=device)
                else:
                    yield graph_pool(src_buf, tgt_buf, device=device)
                src_buf, tgt_buf = [], []
        
        if len(tgt_buf) != 0:
            # NOTE: currently if tgt_buf == 0, will have bug.
            # Issue: https://github.com/dmlc/dgl/issues/1475
            if len(tgt_buf) > 1:
                if mode == 'test':
                    #yield graph_pool.beam(self.sos_id, self.MAX_LENGTH, k, device=device)
                    yield graph_pool(src_buf, tgt_buf, device=device)
                else:
                    yield graph_pool(src_buf, tgt_buf, device=device)

    def get_sequence(self, batch):
        "return a list of sequence from a list of index arrays"
        return batch


class TranslationDataset(object):
    '''
    Dataset class for translation task.
    By default, the source language shares the same vocabulary with the target language.
    '''
    INIT_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    PAD_TOKEN = '<pad>'
    MAX_LENGTH = 50
    def __init__(self, path, exts, train='train', valid='valid', test='test', vocab='vocab.txt', replace_oov=None):
        vocab_path = os.path.join(path, vocab)
        self.src = {}
        self.tgt = {}
        with open(os.path.join(path, train + '.' + exts[0]), 'r', encoding='utf-8') as f:
            self.src['train'] = f.readlines()
        with open(os.path.join(path, train + '.' + exts[1]), 'r', encoding='utf-8') as f:
            self.tgt['train'] = f.readlines()
        with open(os.path.join(path, valid + '.' + exts[0]), 'r', encoding='utf-8') as f:
            self.src['valid'] = f.readlines()
        with open(os.path.join(path, valid + '.' + exts[1]), 'r', encoding='utf-8') as f:
            self.tgt['valid'] = f.readlines()
        with open(os.path.join(path, test + '.' + exts[0]), 'r', encoding='utf-8') as f:
            self.src['test'] = f.readlines()
        with open(os.path.join(path, test + '.' + exts[1]), 'r', encoding='utf-8') as f:
            self.tgt['test'] = f.readlines()

        if not os.path.exists(vocab_path):
            self._make_vocab(vocab_path)

        vocab = Vocab(init_token=self.INIT_TOKEN,
                      eos_token=self.EOS_TOKEN,
                      pad_token=self.PAD_TOKEN,
                      unk_token=replace_oov)
        vocab.load(vocab_path)
        self.vocab = vocab
        strip_func = lambda x: x[:self.MAX_LENGTH]
        self.src_field = Field(vocab,
                               preprocessing=None,
                               postprocessing=strip_func)
        self.tgt_field = Field(vocab,
                               preprocessing=lambda seq: [self.INIT_TOKEN] + seq + [self.EOS_TOKEN],
                               postprocessing=strip_func)

    def get_seq_by_id(self, idx, mode='train', field='src'):
        "get raw sequence in dataset by specifying index, mode(train/valid/test), field(src/tgt)"
        if field == 'src':
            return self.src[mode][idx].strip().split()
        else:
            return [self.INIT_TOKEN] + self.tgt[mode][idx].strip().split() + [self.EOS_TOKEN]

    def _make_vocab(self, path, thres=2):
        word_dict = {}
        for mode in ['train', 'valid', 'test']:
            for line in self.src[mode] + self.tgt[mode]:
                for token in line.strip().split():
                    if token not in word_dict:
                        word_dict[token] = 0
                    else:
                        word_dict[token] += 1

        with open(path, 'w') as f:
            for k, v in word_dict.items():
                if v > 2:
                    print(k, file=f)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad_id(self):
        return self.vocab[self.PAD_TOKEN]

    @property
    def sos_id(self):
        return self.vocab[self.INIT_TOKEN]

    @property
    def eos_id(self):
        return self.vocab[self.EOS_TOKEN]

    def __call__(self, graph_pool, mode='train', batch_size=32, k=1,
                 device='cpu', dev_rank=0, ndev=1):
        '''
        Create a batched graph correspond to the mini-batch of the dataset.
        args:
            graph_pool: a GraphPool object for accelerating.
            mode: train/valid/test
            batch_size: batch size
            k: beam size(only required for test)
            device: str or torch.device
            dev_rank: rank (id) of current device
            ndev: number of devices
        '''
        src_data, tgt_data = self.src[mode], self.tgt[mode]
        n = len(src_data)
        # make sure all devices have the same number of batch
        n = n // ndev * ndev

        # XXX: partition then shuffle may not be equivalent to shuffle then
        # partition
        order = list(range(dev_rank, n, ndev))
        if mode == 'train':
            random.shuffle(order)

        src_buf, tgt_buf = [], []

        for idx in order:
            src_sample = self.src_field(
                src_data[idx].strip().split())
            tgt_sample = self.tgt_field(
                tgt_data[idx].strip().split())
            src_buf.append(src_sample)
            tgt_buf.append(tgt_sample)
            if len(src_buf) == batch_size:
                if mode == 'test':
                    yield graph_pool.beam(src_buf, self.sos_id, self.MAX_LENGTH, k, device=device)
                else:
                    yield graph_pool(src_buf, tgt_buf, device=device)
                src_buf, tgt_buf = [], []

        if len(src_buf) != 0:
            if mode == 'test':
                yield graph_pool.beam(src_buf, self.sos_id, self.MAX_LENGTH, k, device=device)
            else:
                yield graph_pool(src_buf, tgt_buf, device=device)

    def get_sequence(self, batch):
        "return a list of sequence from a list of index arrays"
        ret = []
        for seq in batch:
            try:
                l = seq.index(self.eos_id)
            except:
                l = len(seq)
            ret.append(' '.join(self.vocab[token] for token in seq[:l] if not token in filter_list))
        return ret

def get_dataset(dataset):
    "we wrapped a set of datasets as example"
    prepare_dataset(dataset)
    if dataset == 'babi':
        raise NotImplementedError
    elif dataset == 'copy' or dataset == 'sort':
        return TranslationDataset(
            'data/{}'.format(dataset),
            ('in', 'out'),
            train='train',
            valid='valid',
            test='test',
        )
    elif dataset == 'vertex':
        return ShapeNetVertexDataset()
    elif dataset == 'face':
        return ShapeNetFaceDataset()
    elif dataset == 'multi30k':
        return TranslationDataset(
            'data/multi30k',
            ('en.atok', 'de.atok'),
            train='train',
            valid='val',
            test='test2016',
            replace_oov='<unk>'
        )
    elif dataset == 'wmt14':
        return TranslationDataset(
            'data/wmt14',
            ('en', 'de'),
            train='train.tok.clean.bpe.32000',
            valid='newstest2013.tok.bpe.32000',
            test='newstest2014.tok.bpe.32000.ende',
            vocab='vocab.bpe.32000')
    else:
        raise KeyError()
