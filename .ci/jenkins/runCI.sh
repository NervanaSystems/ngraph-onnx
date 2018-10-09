#!/bin/bash

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

CI_PATH="$(pwd)"
CI_ROOT=".ci/jenkins"
REPO_ROOT="${CI_PATH%$CI_ROOT}"
DOCKER_CONTAINER="ngraph-onnx_ci"
ENVPREP_ARGS="--rebuild-ngraph"

# Function run() builds image with requirements needed to build ngraph and run onnx tests, runs container and executes tox tests
function run() {
    set -x
    cd ./dockerfiles
    docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy -f=./ubuntu-16_04.dockerfile -t ngraph-onnx:ubuntu-16_04 .

    cd "${CI_PATH}"
    mkdir -p ${CI_PATH}/ONNX_CI
    if [[ -z $(docker ps -a | grep -i "${DOCKER_CONTAINER}") ]]; 
    then 
        docker run -h "$(hostname)" --privileged --name "${DOCKER_CONTAINER}" -v "${REPO_ROOT}":/root -d ngraph-onnx:ubuntu-16_04 tail -f /dev/null
        docker exec "${DOCKER_CONTAINER}" bash -c "/root/${CI_ROOT}/prepare_environment.sh ${ENVPREP_ARGS}"
    fi

    NGRAPH_WHL=$(docker exec ${DOCKER_CONTAINER} find /root/ngraph/python/dist/ -name "ngraph*.whl")
    docker exec -e TOX_INSTALL_NGRAPH_FROM="${NGRAPH_WHL}" "${DOCKER_CONTAINER}" tox -c /root

    echo "========== FOLLOWING ITEMS WERE CREATED DURING SCRIPT EXECUTION =========="
    echo "Docker image: ngraph-onnx:ubuntu-16_04"
    echo "Docker container: ${DOCKER_CONTAINER}"
    echo "Directory: ${CI_PATH}/ONNX_CI"
    echo "Multiple files generated during tox execution"
    echo ""
    echo "TO REMOVE THEM RUN THIS SCRIPT WITH PARAMETER: --cleanup"
}

# Function cleanup() removes items created during script execution
function cleanup() {
    set -x
    HOME_FILES=$(docker exec "${DOCKER_CONTAINER}" bash -c 'rm -rf /home/$(find /home/ -user root)')
    for f in ${HOME_FILES}; 
    do
        rm -rf $f
    done
    rm -rf ${CI_PATH}/ONNX_CI
    ROOT_FILES=$(docker exec "${DOCKER_CONTAINER}" bash -c 'rm -rf /root/$(find /root/ -user root)')
    for f in ${ROOT_FILES}; 
    do
        rm -rf $f
    done
    docker rm -f "${DOCKER_CONTAINER}"
    docker rmi --force ngraph-onnx:ubuntu-16_04
}

PATTERN='[-a-zA-Z0-9_]*='
for i in "$@"
do
    case $i in
        --help*)
            printf "Following parameters are available:
    
            --help  displays this message
            --cleanup  removes docker image, container and files created during script execution
            --ngraph-commit nGraph commit SHA to run tox tests on
            "
            exit 0
        ;;
        --cleanup*)
            cleanup
            exit 0
        ;;
        --ngraph-commit*)
            SHA=`echo $i | sed "s/${PATTERN}//"`
            ENVPREP_ARGS="${ENVPREP_ARGS} --ngraph-commit ${SHA}"
        ;;
    esac
done

run