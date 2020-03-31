FROM ubuntu:18.04

ARG http_proxy
ARG https_proxy
ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}

# nGraph dependencies
RUN apt-get update && apt-get -y --no-install-recommends install \
        libcurl4-openssl-dev=7.58.0-2ubuntu3.8 \
        pkg-config=0.29.1-0ubuntu2 \
        build-essential=12.4ubuntu1 \
        clang-3.9=1:3.9.1-19ubuntu1 \
        git=1:2.17.1-1ubuntu0.6 \
        curl=7.58.0-2ubuntu3.8 \
        wget=1.19.4-1ubuntu2.2 \
        zlib1g-dev=1:1.2.11.dfsg-0ubuntu2 \
        libtinfo-dev=6.1-1ubuntu1.18.04 \
        unzip=6.0-21ubuntu1 \
        autoconf=2.69-11 \
        automake=1:1.15.1-3ubuntu2 \
        ocl-icd-opencl-dev=2.2.11-1ubuntu1 \
        libtool=2.4.6-2 && \
  apt-get clean autoclean && \
  apt-get autoremove -y

RUN wget https://www.cmake.org/files/v3.13/cmake-3.13.3.tar.gz --no-check-certificate && \
    tar xf cmake-3.13.3.tar.gz && \
    (cd cmake-3.13.3 && ./bootstrap --system-curl --parallel=$(nproc --all) && make --jobs=$(nproc --all) && make install) && \
    rm -rf cmake-3.13.3 cmake-3.13.3.tar.gz

# install the iGPU drivers copied into the container from the build context
ARG opencl_url="https://github.com/intel/compute-runtime/releases/download"
ARG opencl_version="19.29.13530"
ARG igc_version="1.0.10-2306"
ARG gmmlib_version="19.2.3"
WORKDIR /tmp/intel-opencl
RUN wget --no-check-certificate ${opencl_url}/${opencl_version}/intel-gmmlib_${gmmlib_version}_amd64.deb && \
    wget --no-check-certificate ${opencl_url}/${opencl_version}/intel-igc-core_${igc_version}_amd64.deb && \
    wget --no-check-certificate ${opencl_url}/${opencl_version}/intel-igc-opencl_${igc_version}_amd64.deb && \
    wget --no-check-certificate ${opencl_url}/${opencl_version}/intel-opencl_${opencl_version}_amd64.deb && \
    wget --no-check-certificate ${opencl_url}/${opencl_version}/intel-ocloc_${opencl_version}_amd64.deb && \
    dpkg -i *.deb && rm -rf *.deb

# Python dependencies
RUN apt-get -y --no-install-recommends install \
        python3=3.6.7-1~18.04 \
        python3-pip=9.0.1-2.3~ubuntu1.18.04.1 \
        python3-dev=3.6.7-1~18.04 \
        python-virtualenv=15.1.0+ds-1.1 && \
    apt-get clean autoclean && \
    apt-get autoremove -y

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash

RUN pip3 install --upgrade pip==19.0.3 \
        setuptools==41.0.0 \
        wheel==0.33.1

# ONNX dependencies
RUN apt-get -y --no-install-recommends install \
        git-lfs=2.10.0 \
        protobuf-compiler=3.0.0-9.1ubuntu1 \
        libprotobuf-dev=3.0.0-9.1ubuntu1 && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Install tox
RUN pip3 install tox
