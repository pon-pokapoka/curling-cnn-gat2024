FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04

SHELL ["/bin/bash", "-c"]

ARG BOOST_VERSION=1.83
RUN apt update && apt install -y\
    cmake\
    gdb\
    git\
    libboost${BOOST_VERSION}-dev\
    libboost-date-time${BOOST_VERSION}-dev\
    libboost-regex${BOOST_VERSION}-dev\
#     libcusparselt0\
#     libcusparselt-dev\
    libicu-dev\
    ninja-build\
    unzip\
    wget\
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_VERSION=2.4.1
ARG CUDA_VERSION=124
RUN wget https://download.pytorch.org/libtorch/cu${CUDA_VERSION}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcu${CUDA_VERSION}.zip\
    && unzip libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}+cu${CUDA_VERSION}.zip -d /usr/local\
    && rm libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}+cu${CUDA_VERSION}.zip

