FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]
RUN apt update && apt install -y\
    cmake\
    gdb\
    git\
    libboost1.74-dev\
    libboost-date-time1.74-dev\
    libboost-regex1.74-dev\
    libcusparselt0\
    libcusparselt-dev\
    libicu-dev\
    ninja-build\
    unzip\
    wget\
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_VERSION=2.2.1
ARG CUDA_VERSION=121
RUN wget https://download.pytorch.org/libtorch/cu${CUDA_VERSION}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcu${CUDA_VERSION}.zip\
    && unzip libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}+cu${CUDA_VERSION}.zip -d /usr/local\
    && rm libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}+cu${CUDA_VERSION}.zip
