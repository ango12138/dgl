from scipy import io
import numpy as np
import os
import datetime

from .dgl_dataset import DGLBuiltinDataset
from .utils import get_download_dir, download, extract_archive, loadtxt, makedirs
from ..utils import retry_method_with_fix
from ..graph import DGLGraph


class GDELTDataset(DGLBuiltinDataset):
    r"""GDELT dataset for event-based temporal graph

    The Global Database of Events, Language, and Tone (GDELT) dataset.
    Event data consists of coded interactions between socio-political actors
    (i.e., cooperative or hostile actions between individuals, groups, sectors
    and nation states). Events are automatically identified and extracted
    from news articles by the BBN ACCENT event coder. These events are
    essentially triples consisting of a source actor, an event type (according
    to the CAMEO taxonomy of events), and a target actor.

    Offical website:
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/28075

    Statistics
    ----------

    Examples
    ----------
    >>> data = GDELTDataset()
    >>>
    """
    _url = {
        'train': 'https://github.com/INK-USC/RENet/raw/master/data/GDELT/train.txt',
        'valid': 'https://github.com/INK-USC/RENet/raw/master/data/GDELT/valid.txt',
        'test': 'https://github.com/INK-USC/RENet/raw/master/data/GDELT/test.txt',
    }
    _sha1_str = {
        'train': '',
        'valid': '',
        'test': ''
    }

    def __init__(self, mode='train', raw_dir=None, force_reload=False, verbose=False):
        mode = mode.lower()
        assert mode in self._url, "Mode not valid."
        self.dir = get_download_dir()
        self.mode = mode
        super(GDELTDataset, self).__init__(name='GDELT',
                                           url=self._url[mode],
                                           raw_dir=raw_dir,
                                           force_reload=force_reload,
                                           verbose=verbose)

    def download(self):
        makedirs(self.save_path)
        file_path = os.path.join(self.save_path, self.mode + '.txt')
        download(self.url, path=file_path)

    def process(self):
        file_path = os.path.join(self.raw_path, self.mode + '.txt')
        self._data = loadtxt(file_path, delimiter='\t').astype(np.int64)

        # The source code is not released, but the paper indicates there're
        # totally 137 samples. The cutoff below has exactly 137 samples.
        self._time_index = np.floor(self._data[:, 3] / 15).astype(np.int64)
        self._start_time = self._time_index[self._time_index != -1].min()
        self._end_time = self._time_index.max()


class GDELT(object):
    """
    The Global Database of Events, Language, and Tone (GDELT) dataset.
    This contains events happend all over the world (ie every protest held anywhere
     in Russia on a given day is collapsed to a single entry).

    This Dataset consists of
    events collected from 1/1/2018 to 1/31/2018 (15 minutes time granularity).

    Reference:
    - `Recurrent Event Network for Reasoning over Temporal
    Knowledge Graphs <https://arxiv.org/abs/1904.05530>`_
    - `The Global Database of Events, Language, and Tone (GDELT) <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/28075>`_


    Parameters
    ------------
    mode: str
      Load train/valid/test data. Has to be one of ['train', 'valid', 'test']

    """
    _url = {
        'train': 'https://github.com/INK-USC/RENet/raw/master/data/GDELT/train.txt',
        'valid': 'https://github.com/INK-USC/RENet/raw/master/data/GDELT/valid.txt',
        'test': 'https://github.com/INK-USC/RENet/raw/master/data/GDELT/test.txt',
    }

    def __init__(self, mode):
        assert mode.lower() in self._url, "Mode not valid"
        self.dir = get_download_dir()
        self.mode = mode
        # self.graphs = []
        train_data = loadtxt(os.path.join(
            self.dir, 'GDELT', 'train.txt'), delimiter='\t').astype(np.int64)
        if self.mode == 'train':
            self._load(train_data)
        elif self.mode == 'valid':
            val_data = loadtxt(os.path.join(
                self.dir, 'GDELT', 'valid.txt'), delimiter='\t').astype(np.int64)
            train_data[:, 3] = -1
            self._load(np.concatenate([train_data, val_data], axis=0))
        elif self.mode == 'test':
            val_data = loadtxt(os.path.join(
                self.dir, 'GDELT', 'valid.txt'), delimiter='\t').astype(np.int64)
            test_data = loadtxt(os.path.join(
                self.dir, 'GDELT', 'test.txt'), delimiter='\t').astype(np.int64)
            train_data[:, 3] = -1
            val_data[:, 3] = -1
            self._load(np.concatenate(
                [train_data, val_data, test_data], axis=0))

    def _download(self):
        for dname in self._url:
            dpath = os.path.join(
                self.dir, 'GDELT', self._url[dname.lower()].split('/')[-1])
            download(self._url[dname.lower()], path=dpath)

    @retry_method_with_fix(_download)
    def _load(self, data):
        # The source code is not released, but the paper indicates there're
        # totally 137 samples. The cutoff below has exactly 137 samples.
        self.data = data
        self.time_index = np.floor(data[:, 3]/15).astype(np.int64)
        self.start_time = self.time_index[self.time_index != -1].min()
        self.end_time = self.time_index.max()

    def __getitem__(self, idx):
        if idx >= len(self) or idx < 0:
            raise IndexError("Index out of range")
        i = idx + self.start_time
        g = DGLGraph()
        g.add_nodes(self.num_nodes)
        row_mask = self.time_index <= i
        edges = self.data[row_mask][:, [0, 2]]
        rate = self.data[row_mask][:, 1]
        g.add_edges(edges[:, 0], edges[:, 1])
        g.edata['rel_type'] = rate.reshape(-1, 1)
        return g

    def __len__(self):
        return self.end_time - self.start_time + 1

    @property
    def num_nodes(self):
        return 23033

    @property
    def is_temporal(self):
        return True
