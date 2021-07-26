#!/bin/bash

set -e

# . /opt/conda/etc/profile.d/conda.sh
# conda activate pytorch-ci
# Default building only with cpu
DEVICE=${DGL_BENCH_DEVICE:-cpu}

pip install -r /asv/torch_gpu_pip.txt
pip install pandas rdflib ogb

# build
if [[ $DEVICE == "cpu" ]]; then
   pip install dgl==0.7.0 -f https://data.dgl.ai/wheels/repo.html
else
   pip install dgl-cu111==0.7.0 -f https://data.dgl.ai/wheels/repo.html
fi
# mkdir -p build
# pushd build
# cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DBUILD_TORCH=ON $CMAKE_VARS ..
# make -j
# popd

# conda deactivate
