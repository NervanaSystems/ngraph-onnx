FROM  onnx/onnx-docker:cpu

ENV LAST_UPDATED 2018-07-02
ENV LC_CTYPE=C.UTF-8

# Define proxies
ARG http_proxy
ARG https_proxy
ENV http_proxy=${http_proxy} https_proxy=${https_proxy} HTTP_PROXY=${http_proxy} HTTPS_PROXY=${https_proxy}

RUN apt-get -y update --fix-missing
RUN apt-get -y install wget
RUN pip install --upgrade pip

WORKDIR /root/caffe2_models/resnet50
RUN wget -O init_net.pb https://s3.amazonaws.com/download.caffe2.ai/models/resnet50/init_net.pb
RUN wget -O predict_net.pb https://s3.amazonaws.com/download.caffe2.ai/models/resnet50/predict_net.pb

WORKDIR /root/scripts
COPY scripts /root/scripts
WORKDIR /root
