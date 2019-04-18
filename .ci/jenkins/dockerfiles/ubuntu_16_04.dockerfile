FROM ubuntu:16.04

ARG HOME=/root
ARG http_proxy
ARG https_proxy
ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}

# nGraph dependencies
RUN apt-get update && apt-get -y --no-install-recommends install \
        build-essential=12.1ubuntu2 \
        cmake=3.5.1-1ubuntu3 \
        clang-3.9=1:3.9.1-4ubuntu3~16.04.2 \
        git=1:2.7.4-0ubuntu1.6 \
        curl=7.47.0-1ubuntu2.12 \
        zlib1g=1:1.2.8.dfsg-2ubuntu4.1 \
        zlib1g-dev=1:1.2.8.dfsg-2ubuntu4.1 \
        libtinfo-dev=6.0+20160213-1ubuntu1 \
        unzip=6.0-20ubuntu1 \
        autoconf=2.69-9 \
        automake=1:1.15-4ubuntu1 \
        libtool=2.4.6-0.1 && \
  apt-get clean autoclean && \
  apt-get autoremove -y

# Python dependencies
RUN apt-get -y --no-install-recommends install \
        python3=3.5.1-3 \
        python3-pip=8.1.1-2ubuntu0.4 \
        python3-dev=3.5.1-3 \
        python-virtualenv=15.0.1+ds-3ubuntu1 && \
    apt-get clean autoclean && \
    apt-get autoremove -y

RUN pip3 install --upgrade pip==19.0.3 \
        setuptools==41.0.0 \
        wheel==0.33.1

# ONNX dependencies
RUN apt-get -y --no-install-recommends install \
        protobuf-compiler=2.6.1-1.3 \
        libprotobuf-dev=2.6.1-1.3 && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Install tox
RUN pip3 install tox==3.9.0

# Build nGraph master
ARG NGRAPH_CACHE_DIR=/cache

WORKDIR /root
RUN git clone https://github.com/NervanaSystems/ngraph.git
WORKDIR /root/ngraph
RUN mkdir -p ./build
WORKDIR /root/ngraph/build
RUN cmake ../ -DNGRAPH_TOOLS_ENABLE=FALSE -DNGRAPH_UNIT_TEST_ENABLE=FALSE -DNGRAPH_USE_PREBUILT_LLVM=TRUE -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE && \
    make -j "$(lscpu --parse=CORE | grep -v '#' | sort | uniq | wc -l)"

# Store built nGraph
RUN mkdir -p ${NGRAPH_CACHE_DIR} && \
    cp -Rf /root/ngraph/build ${NGRAPH_CACHE_DIR}/

# Cleanup remaining sources
RUN rm -rf /root/ngraph
