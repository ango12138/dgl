#!/bin/bash
set -e
. /opt/conda/etc/profile.d/conda.sh

if [ $# -ne 1 ]; then
    echo "Device argument required, can be cpu or gpu"
    exit -1
fi

CMAKE_VARS="-DBUILD_CPP_TEST=ON -DUSE_OPENMP=ON -DBUILD_TORCH=ON"
# This is a semicolon-separated list of Python interpreters containing PyTorch.
# The value here is for CI.  Replace it with your own or comment this whole
# statement for default Python interpreter.
CMAKE_VARS="$CMAKE_VARS -DTORCH_PYTHON_INTERPS=/opt/conda/envs/pytorch-ci/bin/python"

if [ "$1" == "gpu" ]; then
    CMAKE_VARS="-DUSE_CUDA=ON $CMAKE_VARS"
fi

if [ -d build ]; then
    rm -rf build
fi

CCACHE_BASEDIR=$PWD

mkdir build

rm -rf _download

if ! command -v ccache &> /dev/null
then
    echo "Didn't find ccache"
else
    ccache -s
    CMAKE_VARS=" -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache $CMAKE_VARS"
fi

pushd build
echo $CMAKE_VARS
cmake $CMAKE_VARS ..
make -j
popd

pushd python
for backend in pytorch mxnet tensorflow
do 
conda activate "${backend}-ci"
rm -rf build *.egg-info dist
pip uninstall -y dgl
# test install
python3 setup.py install
# test inplace build (for cython)
python3 setup.py build_ext --inplace
done
popd
