import os
import torch as th
import torch.nn as nn


class AminerDataset:
    """
    The FullFreeBase dataset is adapted from OpenKE.
    The dataset is divided into train, valid, test set with ratio of 90%, 5%, 5%.
    Triples in train.txt, valid.txt, test.txt is encoded as entity2id.txt and relation2id.txt.
    """
    def __init__(self, path):

        self.url = 'https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/aminer.zip'

        if not os.path.exists(os.path.join(path, 'aminer')):
            print('File not found. Downloading from', self.url)
            self._download_and_extract(path, 'aminer.zip')

    def _download_and_extract(self, path, filename):
        import shutil, zipfile, zlib
        from tqdm import tqdm
        import requests

        fn = os.path.join(path, filename)

        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)
        f_remote = requests.get(self.url, stream=True)
        assert f_remote.status_code == 200, 'fail to open {}'.format(self.url)
        with open(fn, 'wb') as writer:
            for chunk in tqdm(f_remote.iter_content(chunk_size=1024*1024*3)):
                writer.write(chunk)
        print('Download finished. Unzipping the file...')

        with zipfile.ZipFile(fn) as zf:
            zf.extractall(path)
        print('Unzip finished.')
        self.fn = fn

#if __name__ == '__main__':
#    fb = FullFBDataset('/Users/ziqiaom/Desktop/dataloader/data')