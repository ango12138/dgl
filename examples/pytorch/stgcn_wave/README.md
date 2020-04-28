Spatio-Temporal Graph Convolutional Networks
============

- Paper link: [arXiv](https://arxiv.org/pdf/1709.04875v4.pdf)
- Author's code repo: https://github.com/VeritasYin/STGCN_IJCAI-18.
Dependencies
------------
- PyTorch 1.1.0+
- sklearn
- dgl



How to run
----------
please get METR_LA dataset from [this Google drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX).
and [this Github repo](https://github.com/chnsh/DCRNN_PyTorch)

An experiment in default settings can be run with

```bash
python main.py
```

An experiment on the METR_LA dataset in customized settings can be run with
```bash
python main.py --lr --seed --disable-cuda --batch_size  --epochs
```

Results
-------

```bash
python main.py
```
METR_LA MAE: ~5.76
