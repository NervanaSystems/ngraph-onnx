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

ARG BASE_IMAGE=base_ngraph-onnx
FROM $BASE_IMAGE

# Install nGraph in /~/ngraph
WORKDIR /root
RUN git clone https://github.com/NervanaSystems/ngraph.git && mkdir /root/ngraph/build
WORKDIR /root/ngraph/build
RUN cmake ../ -DNGRAPH_USE_PREBUILT_LLVM=TRUE
RUN make -j 8
RUN make install

# Build nGraph Wheel
WORKDIR /root/ngraph/python
RUN mv /root/pybind11 /root/ngraph/python/pybind11
ENV PYBIND_HEADERS_PATH=/root/ngraph/python/pybind11 NGRAPH_CPP_BUILD_PATH=/root/ngraph_dist
RUN python3 setup.py bdist_wheel

# Test nGraph-ONNX
COPY . /root/ngraph-onnx
WORKDIR /root/ngraph-onnx
RUN pip install tox
CMD TOX_INSTALL_NGRAPH_FROM=`find /root/ngraph/python/dist/ -name 'ngraph*.whl'` tox