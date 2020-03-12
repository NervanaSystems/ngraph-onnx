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
        git=1:2.7.4-0ubuntu1.7 \
        curl=7.47.0-1ubuntu2.14 \
        wget=1.17.1-1ubuntu1.5 \
        zlib1g=1:1.2.8.dfsg-2ubuntu4.3 \
        zlib1g-dev=1:1.2.8.dfsg-2ubuntu4.3 \
        libtinfo-dev=6.0+20160213-1ubuntu1 \
        unzip=6.0-20ubuntu1 \
        autoconf=2.69-9 \
        automake=1:1.15-4ubuntu1 \
        ocl-icd-opencl-dev \
        libtool=2.4.6-0.1 && \
  apt-get clean autoclean && \
  apt-get autoremove -y

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
RUN pip3 install tox
