FROM ubuntu:16.04

ENV LAST_UPDATED 2018-07-02
ENV LC_CTYPE=C.UTF-8

# Define proxies
ARG http_proxy
ARG https_proxy
ENV http_proxy=${http_proxy} https_proxy=${https_proxy} HTTP_PROXY=${http_proxy} HTTPS_PROXY=${https_proxy}

RUN apt-get -y update --fix-missing
RUN apt-get -y install wget git
RUN apt-get -y install build-essential git cmake openmpi-bin

# Python dependencies
RUN apt-get -y install python python3
RUN apt-get -y install python-dev python3-dev
RUN apt-get -y install python-pip python3-pip
RUN pip install --upgrade pip setuptools wheel
RUN pip3 install --upgrade pip setuptools wheel

# ONNX dependencies
RUN apt-get -y install protobuf-compiler libprotobuf-dev
RUN pip3 install git+https://github.com/onnx/onnx@410530e8c6ad7e7b66644a247dace2de11a724e3#egg=onnx
RUN pip3 install cntk==2.5.1

# Download models
WORKDIR /root/cntk_models
RUN wget -O ResNet50_ImageNet_CNTK.model https://www.cntk.ai/Models/CNTK_Pretrained/ResNet50_ImageNet_CNTK.model

RUN pip3 install scikit-image

COPY scripts /root/scripts
WORKDIR /root/
