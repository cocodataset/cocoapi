## System Requirements
- Ubuntu 16.04
- Python >= 3.5

## Setup with Singularity
We recommend the users to adopt singualrity as the default environment manager to minimize the efforts of configuration.
Singularity is a virtual environment manager like virtualenv, but in the system-level.

#### Install Singularity >= 2.6
```bash
# install dependency
sudo apt update
sudo apt install build-essential python libarchive-dev

# install singularity
wget https://github.com/sylabs/singularity/releases/download/2.6.1/singularity-2.6.1.tar.gz
tar xzfv singularity-2.6.1.tar.gz
cd singularity-2.6.1
./configure --prefix=/usr/local
make
sudo make install
```

#### Download singularity image for SimpleDet
```bash
wget https://simpledet-model.oss-cn-beijing.aliyuncs.com/simpledet.img
```

#### Invoke simpledet shell
Here we need to map the working directory into singularity shell, note that **symlink to files outside the working directory will not work** since singularity has its own filesystem. Thus we recommend users to map the whole data storage into singularity by replacing $WORKDIR by something like `/data` or `/mnt/`.

```bash
sudo singularity shell --no-home --nv -s /usr/bin/zsh --bind $WORKDIR /path/to/simpledet.img
```

## Setup from Scratch
#### Install CUDA, cuDNN and NCCL

#### Install cocotools
```bash
# Install a patched cocotools for python3
git clone https://github.com/RogerChern/cocoapi
cd cocoapi/PythonAPI
python3 setup.py install
```

#### Install MXNet
```bash
# Install dependency
sudo apt-get update
sudo apt-get install -y build-essential git
sudo apt-get install -y libopenblas-dev
```

```bash
# Specify simpledet directory
export SIMPLEDET_DIR=/path/to/simpledet
export COCOAPI_DIR = /path/to/cocoapi

git clone https://github.com/apache/incubator-mxnet mxnet
cd mxnet
git checkout 1.3.1
git submodule init
git submodule update:
echo "USE_OPENCV = 0" >> ./config.mk
echo "USE_BLAS = openblas" >> ./config.mk
echo "USE_CUDA = 1" >> ./config.mk
echo "USE_CUDA_PATH = /usr/local/cuda" >> ./config.mk
echo "USE_CUDNN = 1" >> ./config.mk
echo "USE_NCCL = 1" >> ./config.mk
echo "USE_DIST_KVSTORE = 1" >> ./config.mk
cp -r $SIMPLEDET_DIR/operator_cxx/* src/operator/
mkdir -p src/coco_api
cp -r $COCOAPI_DIR/common src/coco_api/
make -j
cd python
python3 setup.py install
```
