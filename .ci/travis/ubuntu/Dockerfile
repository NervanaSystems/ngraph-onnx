FROM ubuntu:16.04

RUN apt-get -y update

# nGraph dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  cmake \
  clang-3.9 \
  git \
  curl \
  zlib1g \
  zlib1g-dev \
  libtinfo-dev \
  unzip \
  autoconf \
  automake \
  libtool && \
  apt-get clean autoclean && apt-get autoremove -y

# Python dependencies
RUN apt-get update && apt-get install -y \
  python3 \
  python3-dev \
  python3-pip \
  python-virtualenv && \
  apt-get clean autoclean && apt-get autoremove -y

RUN pip3 install --upgrade pip setuptools wheel

# ONNX dependencies
RUN apt-get -y install protobuf-compiler libprotobuf-dev

# Install nGraph in /root/ngraph
WORKDIR /root
RUN git clone https://github.com/NervanaSystems/ngraph.git
RUN mkdir /root/ngraph/build
WORKDIR /root/ngraph/build
RUN cmake ../ -DNGRAPH_CPU_ENABLE=FALSE -DNGRAPH_USE_PREBUILT_LLVM=TRUE -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE -DCMAKE_INSTALL_PREFIX=/root/ngraph_dist
RUN make -j"$(nproc)"
RUN make install

# Build nGraph Wheel
WORKDIR /root/ngraph/python
RUN git clone --recursive -b allow-nonconstructible-holders https://github.com/jagerman/pybind11.git
ENV NGRAPH_ONNX_IMPORT_ENABLE TRUE
ENV PYBIND_HEADERS_PATH /root/ngraph/python/pybind11
ENV NGRAPH_CPP_BUILD_PATH /root/ngraph_dist
RUN python3 setup.py bdist_wheel

# Test nGraph-ONNX
COPY . /root/ngraph-onnx
WORKDIR /root/ngraph-onnx
RUN pip install tox
CMD NGRAPH_BACKEND=INTERPRETER TOX_INSTALL_NGRAPH_FROM=`find /root/ngraph/python/dist/ -name 'ngraph*.whl'` tox
