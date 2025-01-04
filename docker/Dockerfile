# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu20.04
ENV TZ=TR \
    DEBIAN_FRONTEND=noninteractive \
    PARALLEL_HOME=/home/root/ParallelProcessing \
    PARALLEL_GIT_URL=https://github.com/CengizPoyraz/ParallelProcessing.git
WORKDIR $PARALLEL_HOME
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt upgrade -y
RUN apt install -y software-properties-common libomp-dev build-essential openmpi-bin openmpi-doc libopenmpi-dev libthrust-dev git curl nano

RUN git clone $PARALLEL_GIT_URL $PARALLEL_HOME


