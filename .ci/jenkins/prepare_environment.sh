#!/bin/bash
# INTEL CONFIDENTIAL
# Copyright 2018-2020 Intel Corporation
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

set -x
set -e

function build_open_vino() {
    set -x
    local directory="$1"
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Debug \
                -DENABLE_VALIDATION_SET=OFF \
                -DENABLE_VPU=OFF \
                -DENABLE_DLIA=OFF \
                -DENABLE_GNA=OFF \
                -DENABLE_CPPLINT=OFF \
                -DENABLE_TESTS=OFF \
                -DENABLE_BEH_TESTS=OFF \
                -DENABLE_FUNCTIONAL_TESTS=OFF \
                -DENABLE_MKL_DNN=ON \
                -DENABLE_CLDNN=OFF \
                -DENABLE_PROFILING_ITT=OFF \
                -DENABLE_SAMPLES=OFF \
                -DENABLE_SPEECH_DEMO=OFF \
                -DENABLE_PYTHON=ON \
                -DPYTHON_EXECUTABLE=`which python3` \
                -DNGRAPH_ONNX_IMPORT_ENABLE=ON \
                -DNGRAPH_IE_ENABLE=ON \
                -DNGRAPH_INTERPRETER_ENABLE=ON \
                -DNGRAPH_DEBUG_ENABLE=OFF \
                -DNGRAPH_DYNAMIC_COMPONENTS_ENABLE=ON \
                -DCMAKE_INSTALL_PREFIX=${directory}/openvino_dist"

    cd "${directory}/openvino"
    
    mkdir -p ./build
    cd ./build
    cmake ${CMAKE_ARGS} ..  || return 1
    make -j $(lscpu --parse=CORE | grep -v '#' | sort | uniq | wc -l) || return 1
    make install || return 1

    cd "${directory}/openvino/ngraph/python"
    if [ ! -d ./pybind11 ]; then
        git clone --recursive https://github.com/pybind/pybind11.git
    fi
    virtualenv -p `which python3` venv
    . venv/bin/activate
    rm -f "${directory}/openvino/ngraph/python/dist/ngraph*.whl"
    rm -rf "${directory}/openvino/ngraph/python/*.so ${directory}/openvino/ngraph/python/build"
    export PYBIND_HEADERS_PATH="${directory}/openvino/ngraph/python/pybind11"
    export NGRAPH_CPP_BUILD_PATH="${directory}/openvino_dist"
    export NGRAPH_ONNX_IMPORT_ENABLE="TRUE"
    mv "${directory}/ngraph-onnx/.ci/jenkins/setup.py" .
    python3 setup.py develop || return 1
    return 0
}

function main() {
    NUM_PARAMETERS="1"
    if [ $# -lt "${NUM_PARAMETERS}" ]; then
        echo "ERROR: Expected at least ${NUM_PARAMETERS} parameter got $#"
        exit 1
    fi

    PATTERN='[-a-zA-Z0-9_]*='
    for i in "$@"
    do
        case $i in
            --build-dir=*)
                BUILD_DIR="${i//${PATTERN}/}"
                ;;
            *)
                echo "Parameter $i not recognized."
                exit 1
                ;;
        esac
    done

    BUILD_OV_CALL="build_open_vino \"${BUILD_DIR}\""

    eval "${BUILD_OV_CALL}"

}

if [[ ${BASH_SOURCE[0]} == "${0}" ]]; then
    main "${@}"
fi
