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

function build_ngraph() {
    set -x
    local directory="$1"
    local backends="$2"
    CMAKE_ARGS="-DNGRAPH_TOOLS_ENABLE=FALSE -DNGRAPH_WARNINGS_AS_ERRORS=TRUE -DCMAKE_BUILD_TYPE=Release -DNGRAPH_UNIT_TEST_ENABLE=FALSE -DNGRAPH_USE_PREBUILT_LLVM=TRUE -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE -DCMAKE_INSTALL_PREFIX=${directory}/ngraph_dist"
    cd "${directory}/ngraph"

    # CMAKE args for nGraph backends
    if [[ ${backends} == *"igpu"* ]]; then
        echo "Building nGraph for Intel GPU."
        CMAKE_ARGS="${CMAKE_ARGS} -DNGRAPH_INTERPRETER_ENABLE=TRUE"
    fi
    if [[ ${backends} == *"interpreter"* ]]; then
        echo "Building nGraph for INTERPRETER backend."
        CMAKE_ARGS="${CMAKE_ARGS} -DNGRAPH_INTELGPU_ENABLE=TRUE"
    fi

    cd "${directory}/ngraph"
    mkdir -p ./build
    cd ./build
    cmake ${CMAKE_ARGS} ..  || return 1
    make -j $(lscpu --parse=CORE | grep -v '#' | sort | uniq | wc -l) || return 1
    make install || return 1
    cd "${directory}/ngraph/python"
    if [ ! -d ./pybind11 ]; then
        git clone --recursive https://github.com/pybind/pybind11.git
    fi
    rm -f "${directory}/ngraph/python/dist/ngraph*.whl"
    rm -rf "${directory}/ngraph/python/*.so ${directory}/ngraph/python/build"
    export PYBIND_HEADERS_PATH="${directory}/ngraph/python/pybind11"
    export NGRAPH_CPP_BUILD_PATH="${directory}/ngraph_dist"
    export NGRAPH_ONNX_IMPORT_ENABLE="TRUE"
    python3 setup.py bdist_wheel || return 1
    # Clean build artifacts
    # rm -rf "${directory}/ngraph_dist"
    return 0
}

function build_dldt() {
    set -x
    local directory="$1"
    CMAKE_ARGS="-DNGRAPH_CPU_ENABLE=TRUE -DNGRAPH_LIBRARY_OUTPUT_DIRECTORY=${directory}/dldt_dist \
                -DNGRAPH_COMPONENT_PREFIX=deployment_tools/ngraph/ -DNGRAPH_USE_PREBUILT_LLVM=TRUE \
                -DNGRAPH_TOOLS_ENABLE=TRUE -DNGRAPH_WARNINGS_AS_ERRORS=TRUE -DNGRAPH_UNIT_TEST_ENABLE=FALSE \
                -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=OFF -DENABLE_RPATH=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
                -DENABLE_PERFORMANCE_TESTS=ON -DENABLE_TESTS=ON -DNGRAPH_INTERPRETER_ENABLE=ON -DNGRAPH_DEBUG_ENABLE=OFF \
                -DENABLE_SAMPLES=OFF -DENABLE_FUNCTIONAL_TESTS=ON -DENABLE_MODELS=OFF -DENABLE_PRIVATE_MODELS=OFF \
                -DENABLE_GNA=OFF -DENABLE_VPU=OFF -DENABLE_SANITIZER=OFF -DENABLE_MYRIAD=OFF -DENABLE_MKL_DNN=ON \
                -DENABLE_CLDNN=OFF -DENABLE_VALIDATION_SET=OFF -DPYTHON_EXECUTABLE=`which python` \
                -DNGRAPH_ONNX_IMPORT_ENABLE=ON -DNGRAPH_UNIT_TEST_OPENVINO_ENABLE=TRUE -DNGRAPH_IE_ENABLE=ON \
                -DCMAKE_INSTALL_PREFIX=${directory}/dldt_dist -DNGRAPH_DYNAMIC_COMPONENTS_ENABLE=ON"
    cd "${directory}/dldt/ngraph"
    git checkout rblaczkowski/updated-ie-enabled

    cd "${directory}/dldt"
    
    # CMAKE args for nGraph backends
    mkdir -p ./build
    cd ./build
    cmake ${CMAKE_ARGS} ..  || return 1
    make -j $(lscpu --parse=CORE | grep -v '#' | sort | uniq | wc -l) || return 1

    make install || return 1

    cd "${directory}/dldt/ngraph/python"

    if [ ! -d ./pybind11 ]; then
        git clone --recursive https://github.com/pybind/pybind11.git
    fi

    rm -f "${directory}/dldt/ngraph/python/dist/ngraph*.whl"
    rm -rf "${directory}/dldt/ngraph/python/*.so ${directory}/dldt/ngraph/python/build"
    export PYBIND_HEADERS_PATH="${directory}/dldt/ngraph/python/pybind11"
    export NGRAPH_CPP_BUILD_PATH="${directory}/deployment_tools/ngraph/"
    export NGRAPH_ONNX_IMPORT_ENABLE="TRUE"
    python3 setup.py bdist_wheel || return 1
    return 0
}

function main() {
    # By default copy stored nGraph master and use it to build PR branch
    BACKENDS="cpu"

    NUM_PARAMETERS="2"
    if [ $# -lt "${NUM_PARAMETERS}" ]; then
        echo "ERROR: Expected at least ${NUM_PARAMETERS} parameter got $#"
        exit 1
    fi

    PATTERN='[-a-zA-Z0-9_]*='
    for i in "$@"
    do
        case $i in
            --backends=*)
                BACKENDS="${i//${PATTERN}/}"
                ;;
            --build-dir=*)
                BUILD_DIR="${i//${PATTERN}/}"
                ;;
            *)
                echo "Parameter $i not recognized."
                exit 1
                ;;
        esac
    done

    BUILD_NGRAPH_CALL="build_ngraph \"${BUILD_DIR}\" \"${BACKENDS}\""
    BUILD_DLDT_CALL="build_dldt \"${BUILD_DIR}\""

    # eval "${BUILD_NGRAPH_CALL}"
    eval "${BUILD_DLDT_CALL}"

}

if [[ ${BASH_SOURCE[0]} == "${0}" ]]; then
    main "${@}"
fi
