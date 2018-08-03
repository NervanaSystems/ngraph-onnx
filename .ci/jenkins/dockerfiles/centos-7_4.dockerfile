FROM centos:7.4.1708

ARG HOME=/root
ARG BUILD_CORES_NUMBER=8
ARG http_proxy
ARG https_proxy
ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}

RUN yum -y update && \
    yum -y install diffutils gcc-c++ git make ncurses-devel ncurses-libs patch perl-Data-Dumper wget zlib-devel && \
    yum clean all

# Installing Cmake 3.4
WORKDIR ${HOME}
RUN wget https://cmake.org/files/v3.4/cmake-3.4.3.tar.gz --no-check-certificate && \
    tar -xzvf cmake-3.4.3.tar.gz && \
    cd cmake-3.4.3 && \
    ./bootstrap && \
    make -j ${BUILD_CORES_NUMBER} && \
    make install && \
    cd ${HOME} && \
    rm -rf cmake-*

# Install nGraph in /root/ngraph
RUN git clone https://github.com/NervanaSystems/ngraph.git && \
    mkdir ngraph/build && \
    cd build && \
    cmake ../ -DNGRAPH_USE_PREBUILT_LLVM=TRUE && \
    make -j ${BUILD_CORES_NUMBER} && \
    make install
