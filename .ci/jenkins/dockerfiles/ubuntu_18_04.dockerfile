FROM ubuntu:18.04

ARG HOME=/root
ARG http_proxy
ARG https_proxy
ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}

# nGraph dependencies
RUN apt-get update && apt-get -y --no-install-recommends install \
        build-essential \
        cmake \
        clang-3.9  \
        git \
        curl  \
        wget  \
        zlib1g \
        zlib1g-dev \
        libtinfo-dev  \
        unzip \
        autoconf \
        automake \
        ocl-icd-opencl-dev \
        libtool  && \
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
        python3 \
        python3-pip \
        python3-dev  \
        python-virtualenv && \
    apt-get clean autoclean && \
    apt-get autoremove -y

RUN pip3 install --upgrade \
        setuptools \
        wheel 

# ONNX dependencies
RUN apt-get -y --no-install-recommends install \
        protobuf-compiler \
        libprotobuf-dev && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Inference Engine dependencies
RUN apt-get update && apt-get install -y \
        libssl-dev \
        ca-certificates \
        libboost-regex-dev \
        gcc-multilib \
        g++-multilib \
        libgtk2.0-dev \
        pkg-config \
        libcairo2-dev \
        libpango1.0-dev \
        libglib2.0-dev \
        libgtk2.0-dev \
        libswscale-dev \
        libavcodec-dev \
        libavformat-dev \
        libusb-1.0-0-dev && \
        apt-get clean autoclean && apt-get autoremove -y

# Install tox
RUN pip3 install tox
