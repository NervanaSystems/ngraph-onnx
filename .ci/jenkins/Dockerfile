FROM ubuntu:16.04

ARG http_proxy
ARG https_proxy
ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}

RUN apt-get -y update

# nGraph dependencies
RUN apt-get -y install git build-essential cmake clang-3.9 git curl zlib1g zlib1g-dev libtinfo-dev

# Python dependencies
RUN apt-get -y install python python3
RUN apt-get -y install python-pip python3-pip
RUN apt-get -y install python-dev python3-dev
RUN apt-get -y install python-virtualenv
RUN pip install --upgrade pip setuptools wheel
RUN pip3 install --upgrade pip setuptools wheel

# ONNX dependencies
RUN apt-get -y install protobuf-compiler libprotobuf-dev

# Install nGraph in /root/ngraph
WORKDIR /root
RUN git clone https://github.com/NervanaSystems/ngraph.git
RUN mkdir /root/ngraph/build
WORKDIR /root/ngraph/build
RUN cmake ../ -DNGRAPH_USE_PREBUILT_LLVM=TRUE
RUN make -j 8
RUN make install

# Build nGraph Wheel
WORKDIR /root/ngraph/python
RUN git clone --recursive -b allow-nonconstructible-holders https://github.com/jagerman/pybind11.git
ENV PYBIND_HEADERS_PATH /root/ngraph/python/pybind11
ENV NGRAPH_CPP_BUILD_PATH /root/ngraph_dist
RUN python3 setup.py bdist_wheel

# Test nGraph-ONNX
WORKDIR /~/
RUN git clone https://github.com/NervanaSystems/ngraph-onnx.git
WORKDIR /~/ngraph-onnx
RUN pip install tox
CMD TOX_INSTALL_NGRAPH_FROM=`find /~/ngraph/python/dist/ -name 'ngraph*.whl'` tox
