FROM ubuntu:16.04

ENV LAST_UPDATED 2018-07-02
ENV LC_CTYPE=C.UTF-8

# Define proxies
ARG http_proxy
ARG https_proxy
ENV http_proxy=${http_proxy} https_proxy=${https_proxy} HTTP_PROXY=${http_proxy} HTTPS_PROXY=${https_proxy}

RUN apt-get -y update --fix-missing
RUN apt-get -y install build-essential git cmake

# Python dependencies
RUN apt-get -y install python python3
RUN apt-get -y install python-dev python3-dev
RUN apt-get -y install python-pip python3-pip
RUN pip install --upgrade pip setuptools wheel
RUN pip3 install --upgrade pip setuptools wheel

# nGraph dependencies
RUN apt-get -y install git build-essential cmake clang-3.9 git curl zlib1g zlib1g-dev libtinfo-dev

# Install nGraph in /root/ngraph
WORKDIR /sources
RUN git clone https://github.com/NervanaSystems/ngraph.git
WORKDIR /sources/ngraph
RUN git checkout 6ccfbeb52a465e48122ad981247010ce14d24f16
WORKDIR /sources/ngraph/build
RUN cmake -DNGRAPH_USE_PREBUILT_LLVM=TRUE -DNGRAPH_UNIT_TEST_ENABLE=FALSE ..
RUN make -j 18
RUN make install

# Build nGraph Wheel
WORKDIR /sources/ngraph/python
RUN git clone --recursive -b allow-nonconstructible-holders https://github.com/jagerman/pybind11.git
ENV PYBIND_HEADERS_PATH /sources/ngraph/python/pybind11
ENV NGRAPH_CPP_BUILD_PATH /root/ngraph_dist
RUN python3 setup.py bdist_wheel

# ONNX dependencies
RUN apt-get -y install protobuf-compiler libprotobuf-dev
RUN pip3 install -U numpy
RUN pip3 install git+https://github.com/onnx/onnx@410530e8c6ad7e7b66644a247dace2de11a724e3#egg=onnx

# Install ngraph-onnx
RUN pip3 install -U `find /sources/ngraph/python/dist/ -name 'ngraph*.whl'`
RUN pip3 install git+https://github.com/NervanaSystems/ngraph-onnx/@6ad1f498b9eb64333088632fb3f398ace192af52

WORKDIR /root
COPY scripts /root/scripts
