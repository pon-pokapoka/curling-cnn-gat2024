FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]
RUN apt update && apt install -y cmake git libboost1.74-dev libboost-date-time1.74-dev libboost-regex1.74-dev libicu-dev ninja-build wget

ARG TORCH_VERSION=2.2.0
ARG CUDA_VERSION=118
RUN apt install -y unzip\
    && wget https://download.pytorch.org/libtorch/cu${CUDA_VERSION}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcu${CUDA_VERSION}.zip\
    && unzip libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}+cu${CUDA_VERSION}.zip -d /usr/local\
    && rm libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}+cu${CUDA_VERSION}.zip
