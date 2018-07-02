FROM ubuntu:16.04

ENV LAST_UPDATED 2018-07-02
ENV LC_CTYPE=C.UTF-8

# Define proxies
ARG http_proxy
ARG https_proxy
ENV http_proxy=${http_proxy} https_proxy=${https_proxy} HTTP_PROXY=${http_proxy} HTTPS_PROXY=${https_proxy}

RUN apt-get -y update --fix-missing
RUN apt-get -y install build-essential git cmake swig gcc-4.8 g++-4.8 automake
RUN apt-get -y install locales clang-format clang-3.8 llvm-3.8 libclang-3.8-dev

# Python dependencies
RUN apt-get -y install python python-dev python-pip  python-opencv python-matplotlib
RUN pip install --upgrade pip setuptools wheel numpy

# PaddlePaddle dependencies
RUN apt install -y --allow-downgrades \
    protobuf-compiler libprotobuf-dev openssh-server bison \
    wget unzip unrar tar xz-utils bzip2 gzip coreutils ntp \
    curl sed grep graphviz libjpeg-dev zlib1g-dev  \
    liblapack-dev liblapacke-dev \
    net-tools libtool ccache && \
    apt clean -y

# Install PaddlePaddle
RUN pip install numpy wheel requests nltk rarfile common scipy Pillow matplotlib recordio protobuf
WORKDIR /root/
RUN git clone https://github.com/PaddlePaddle/Paddle.git
WORKDIR /root/Paddle/
RUN git checkout 94a741d
RUN WITH_GPU=OFF WITH_TESTING=OFF bash paddle/scripts/paddle_build.sh build
RUN pip install build/python/dist/paddlepaddle-0.12.0-cp27-cp27mu-linux_x86_64.whl

COPY scripts /root/scripts
COPY models /root/models
WORKDIR /root/models
RUN tar xzf paddle-model.tar.gz

WORKDIR /root/
