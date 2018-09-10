#!/bin/bash
# INTEL CONFIDENTIAL
# Copyright 2018 Intel Corporation All Rights Reserved.
# The source code contained or described herein and all documents related to the
# source code ("Material") are owned by Intel Corporation or its suppliers or
# licensors. Title to the Material remains with Intel Corporation or its
# suppliers and licensors. The Material may contain trade secrets and proprietary
# and confidential information of Intel Corporation and its suppliers and
# licensors, and is protected by worldwide copyright and trade secret laws and
# treaty provisions. No part of the Material may be used, copied, reproduced,
# modified, published, uploaded, posted, transmitted, distributed, or disclosed
# in any way without Intel's prior express written permission.
# No license under any patent, copyright, trade secret or other intellectual
# property right is granted to or conferred upon you by disclosure or delivery of
# the Materials, either expressly, by implication, inducement, estoppel or
# otherwise. Any license under such intellectual property rights must be express
# and approved by Intel in writing.

# Install nGraph in /root/ngraph
cd /home
if [ -e ./ngraph ]; then
    cd ./ngraph
    # If ngraph repo is up to date, and wheel exist - no need to rebuild it so exit
    if [[ $(git pull) == *"Already up-to-date"* && -n $(find /home/ngraph/python/dist/ -name 'ngraph*.whl') ]]; then
        exit 0
    else
    # Remove old wheel
    NGRAPH_WHL = $(find /home/ngraph/python/dist/ -name 'ngraph*.whl' -printf '%Ts\t%p\n' | sort -nr | cut -f2 | head -n1)
    rm ${NGRAPH_WHL}
    fi
else
    git clone https://github.com/NervanaSystems/ngraph.git
    cd ./ngraph
fi
mkdir -p ./build
cd ./build
cmake ../ -DNGRAPH_USE_PREBUILT_LLVM=TRUE -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE -DCMAKE_INSTALL_PREFIX=/home/ngraph_dist
make -j $(lscpu --parse=CORE | grep -v '#' | sort | uniq | wc -l)
make install

# Build nGraph wheel
cd /home/ngraph/python
if [ ! -d ./pybind11 ]; then
    git clone --recursive -b allow-nonconstructible-holders https://github.com/jagerman/pybind11.git
fi

export PYBIND_HEADERS_PATH="/home/ngraph/python/pybind11"
export NGRAPH_CPP_BUILD_PATH="/home/ngraph_dist"
python3 setup.py bdist_wheel

# Copy Onnx models
if [ -d /home/onnx_models/.onnx ]; then
    rsync -avhz /home/onnx_models/.onnx /root/
fi
