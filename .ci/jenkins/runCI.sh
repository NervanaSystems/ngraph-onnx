#!/bin/bash

# ******************************************************************************
# Copyright 2018-2019 Intel Corporation
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

CI_PATH="$( cd "$(dirname "$0")" ; pwd -P )"
CI_ROOT=".ci/jenkins"
REPO_ROOT="${CI_PATH%$CI_ROOT}"
DOCKER_CONTAINER="ngraph-onnx_ci_repro"

function clone_ngraph() {
    local sha="$1"

    set -x
    cd "${REPO_ROOT}"
    git clone https://github.com/NervanaSystems/ngraph
    if [ -n "${sha}" ]; then
        cd ./ngraph
        git reset --hard "${sha}"
    fi
}

function build_image() {
    local image_name="$1"

    cd "${CI_PATH}"
    ./utils/docker.sh build \
                    --image_name="${image_name}" \
                    --dockerfile_path="../dockerfiles/ubuntu-16_04.dockerfile" || return 1
}

function start_container() {
    local image_name="$1"

    ${CI_ROOT}/utils/docker.sh start \
                            --image_name="${image_name}" \
                            --container_name=${DOCKER_CONTAINER} \
                            --volumes="-v ${CI_PATH}:/home -v ${REPO_ROOT}:/root"
}

function prepare_environment() {
    docker cp ${CI_PATH}/utils/docker.sh ${DOCKER_CONTAINER}:/home
    docker exec ${DOCKER_CONTAINER} bash -c "/root/${CI_ROOT}/prepare_environment.sh"
}

function run_tox_tests() {
    NGRAPH_WHL=\$(docker exec ${DOCKER_CONTAINER} find /root/ngraph/python/dist/ -name 'ngraph*.whl')
    docker exec -e TOX_INSTALL_NGRAPH_FROM=\${NGRAPH_WHL} ${DOCKER_CONTAINER} tox -c /root
}

# Function cleanup() removes items created during script execution
function cleanup() {
    set -x

    docker exec "${DOCKER_CONTAINER}" bash -c "rm -rf /home"
    docker exec "${DOCKER_CONTAINER}" bash -c "rm -rf /root/ngraph /root/ngraph_dist /root/.tox /root/.onnx /root/__pycache__ /root/ngraph_onnx.egg-info /root/cpu_codegen"
    docker exec "${DOCKER_CONTAINER}" bash -c 'rm -rf $(find /root/ -user root)'
    docker rm -f "${DOCKER_CONTAINER}"
    docker rmi "${DOCKER_IMAGE}"
}

function print_help() {
    printf "Following parameters are available:

            --help                  - displays this message
            --cleanup               - removes docker image, container and files created during script execution
            --docker-image          - Docker image name used in CI (script will build image with provided name if not already present)
            [--rebuild-image]       - forces image rebuild
            [--ngraph-commit]       - nGraph commit SHA to run tox tests on
            "
}

function main() {
    REBUILD_IMAGE="FALSE"

    PATTERN='[-a-zA-Z0-9_]*='
    for i in "$@"
    do
        case $i in
            --help*)
                print_help
                exit 0
            ;;
            --cleanup*)
                cleanup
                exit 0
            ;;
            --docker-image=*)
                DOCKER_IMAGE=`echo $i | sed "s/${PATTERN}//"`
            ;;
            --rebuild-image)
                REBUILD_IMAGE="TRUE"
            ;;
            --ngraph-commit=*)
                SHA=`echo $i | sed "s/${PATTERN}//"`
            ;;
        esac
    done

    if [ -z $(docker ps -a | grep -o "^[a-z0-9]\+\s\+${DOCKER_CONTAINER}\s\+") ]; then
        RERUN="TRUE"
        echo "=========================== !!! ATTENTION !!! ============================"
        echo "Docker container ${DOCKER_CONTAINER} is present (may be stopped)."
        echo "Script will rerun tox tests without rebuilding the nGraph!"
        echo "To start from scratch remove the container. To do so execute the command:"
        echo "    docker rm -f ${DOCKER_CONTAINER}"
    fi

    if [ -z "${RERUN}" ]; then

        if [ -z "${DOCKER_IMAGE}" ]; then
            echo "No Docker image name provided!"
            print_help
            exit 1
        fi

        clone_ngraph "${SHA}"

        if [ -z $(docker images | grep -o "^${DOCKER_IMAGE}\s\+") ]; then
            REBUILD_IMAGE="TRUE"
        fi

        if [[ "${REBUILD_IMAGE}" == *"TRUE"* ]]; then
            build_image "${DOCKER_IMAGE}"
        fi

        start_container "${DOCKER_IMAGE}"
        prepare_environment
    fi

    run_tox_tests

    echo "========== FOLLOWING ITEMS WERE CREATED DURING SCRIPT EXECUTION =========="
    echo "Docker image: ${DOCKER_IMAGE}"
    echo "Docker container: ${DOCKER_CONTAINER}"
    echo "Multiple files generated during tox execution"
    echo ""
    echo "TO REMOVE THEM RUN THIS SCRIPT WITH PARAMETER: --cleanup"
}

if [[ ${BASH_SOURCE[0]} == "${0}" ]]; then
    main "${@}"
fi