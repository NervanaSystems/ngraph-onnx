# ******************************************************************************
# Copyright 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

FROM ubuntu:16.04

ARG http_proxy
ARG https_proxy
ENV http_proxy=${http_proxy} https_proxy=${https_proxy} HTTP_PROXY=${http_proxy} HTTPS_PROXY=${https_proxy}

# nGraph dependencies
RUN apt-get -y update --fix-missing && \
    apt-get -y install git build-essential cmake clang-3.9 git curl zlib1g zlib1g-dev libtinfo-dev \
                       python python3 python-pip python3-pip python-dev python3-dev python-virtualenv && \
    apt -y autoremove && \
    apt clean all

WORKDIR /root
RUN git clone https://github.com/google/protobuf.git
WORKDIR /root/protobuf
RUN git submodule update --init --recursive
RUN ./autogen.sh
RUN ./configure
RUN make
RUN make check
RUN make install
RUN ldconfig # refresh shared library cache.
WORKDIR /root
RUN git clone https://github.com/NervanaSystems/ngraph.git
RUN mkdir /root/ngraph/build
WORKDIR /root/ngraph/build
RUN cmake ../ -DNGRAPH_USE_PREBUILT_LLVM=TRUE
RUN make -j 8
RUN make install

RUN pip install --upgrade pip setuptools wheel && pip3 install --upgrade pip setuptools wheel

RUN git clone --recursive -b allow-nonconstructible-holders https://github.com/jagerman/pybind11.git /root/pybind11
