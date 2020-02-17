FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
#Ref: https://github.com/chainer/chainer/blob/master/docker/python3/Dockerfile
#     https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && apt-get clean
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  software-properties-common
RUN add-apt-repository ppa:jeff250/opencv

RUN apt-get update && apt-get install -y --no-install-recommends \
  cmake \
  curl \
  git \
  libgtk2.0-0 \
  libjpeg-dev \
  liblmdb-dev \
  libopenblas-dev \
  libopencv-contrib-dev \
  libopencv-dev \
  libopencv-nonfree-dev \
  libopenmpi-dev \
  libpng-dev \
  libsm6 \
  libxext6 \
  ninja-build \
  openmpi-bin \
  openmpi-doc \
  pkg-config \
  rsync \
  unzip \
  vim \
  wget \
  zlib1g \
  zlib1g-dev \
  imagemagick && \
  rm -rf /var/lib/apt/lists/*

# Install hdf5
# https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16?u=piojanu
RUN wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz -O ~/hdf5.tar.gz && \
  tar -xvzf ~/hdf5.tar.gz -C ~/ && cd ~/hdf5-1.10.5 && \
  ./configure --enable-parallel --enable-shared --prefix=/usr/local/hdf5 && \
  make -j 4 && make install -j 4 && \
  cd ~ && rm -rf ~/hdf5-1.10.5 && rm ~/hdf5.tar.gz

# Install conda
# https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/Dockerfile
# Python 3.5.2 - https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh
# Python 3.6.0 - https://repo.continuum.io/miniconda/Miniconda3-4.3.14-Linux-x86_64.sh

RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.3.14-Linux-x86_64.sh -O ~/miniconda.sh && \
  /bin/bash ~/miniconda.sh -b -p /opt/conda && \
  rm ~/miniconda.sh

# Install libraries
RUN /opt/conda/bin/conda install -c anaconda protobuf
RUN /opt/conda/bin/pip install --upgrade pip
RUN /opt/conda/bin/pip install -i https://mirrors.aliyun.com/pypi/simple --upgrade --ignore-installed setuptools
RUN /opt/conda/bin/pip install -i https://mirrors.aliyun.com/pypi/simple numpy scipy scikit-image matplotlib pytz PyYAML Pillow tqdm protobuf ninja lmdb addict fire rdp opencv-python typing mpi4py
RUN /opt/conda/bin/pip install -i https://mirrors.aliyun.com/pypi/simple torch torchvision pytorch-ignite tensorboardX
RUN /opt/conda/bin/pip install git+https://github.com/rbgirshick/yacs
RUN /opt/conda/bin/pip install git+https://github.com/Cadene/pretrained-models.pytorch
RUN export CC=mpicc && export HDF5_MPI="ON" && export HDF5_DIR=/usr/local/hdf5 && /opt/conda/bin/pip install -i https://mirrors.aliyun.com/pypi/simple --no-binary=h5py h5py
RUN /opt/conda/bin/conda clean -y -all

# gpu tracing
ENV LD_LIBRARY_PATH .:/usr/local/hdf5/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH /usr/local/hdf5/bin:/usr/local/cuda/bin:/opt/conda/bin:$PATH
ENV CPATH /usr/local/cuda/include:$CPATH
