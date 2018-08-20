FROM ubuntu:16.04

ARG HOME=/root
ARG BUILD_CORES_NUMBER=8
ARG http_proxy
ARG https_proxy
ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}

# nGraph dependencies
RUN apt-get -y update && \
    apt-get -y install git build-essential cmake clang-3.9 git curl zlib1g zlib1g-dev libtinfo-dev && \
    apt-get clean autoclean && \
    apt-get autoremove -y

# Python dependencies
RUN apt-get -y install python python3 \
                       python-pip python3-pip \
                       python-dev python3-dev \
                       python-virtualenv && \
    apt-get clean autoclean && \
    apt-get autoremove -y

RUN pip install --upgrade pip setuptools wheel && \
    pip3 install --upgrade pip setuptools wheel

# ONNX dependencies
RUN apt-get -y install protobuf-compiler libprotobuf-dev && \
    apt-get clean autoclean && \
    apt-get autoremove -y

# Install nGraph
WORKDIR ${HOME}
RUN git clone https://github.com/NervanaSystems/ngraph.git && \
    mkdir ngraph/build && \
    cd build && \
    cmake ../ -DNGRAPH_USE_PREBUILT_LLVM=TRUE && \
    make -j ${BUILD_CORES_NUMBER} && \
    make install

# Build nGraph Wheel
WORKDIR ${HOME}/ngraph/python
ENV PYBIND_HEADERS_PATH ${HOME}/ngraph/python/pybind11
ENV NGRAPH_CPP_BUILD_PATH ${HOME}/ngraph_dist
RUN git clone --recursive -b allow-nonconstructible-holders https://github.com/jagerman/pybind11.git && \
    python3 setup.py bdist_wheel

# Test nGraph-ONNX
WORKDIR ${HOME}
RUN git clone https://github.com/NervanaSystems/ngraph-onnx.git && \
    pip install tox
WORKDIR ${HOME}/ngraph-onnx
