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

# This is to set workdir to ngraph-onnx no matter where this sh is started from
SCRIPT=$(readlink -f "$0")
BASEDIR=$(dirname "$SCRIPT")
cd $BASEDIR && cd ../..

print_help()
{
    echo Available commands - please use one at a time:
    echo --create-base-img  Creates Base Image unless there is one already
    echo --create-test-img  Creates Test Image unless there is one already
    echo --run-test-img     Runs Test Image which automatically executes tox
    echo --remove-test-img  Removes Test Image
    echo --remove-base-img  Removes Base Image
    echo --help             Prints this help

}


create_base_img()
{
    if docker images --format "{{.Repository}}:{{.Tag}}" | grep -m 1 base_ngraph-onnx; then
        echo Base image found so no need to build it again
        return $?
    else
        echo No base image found so need to create one
        echo -----------------------Build Base image------------------------------------
        docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy -t base_ngraph-onnx -f .ci/jenkins/base_ngraph-onnx.dockerfile .
        return $?
    fi
}


create_test_img()
{
    if docker images --format "{{.Repository}}:{{.Tag}}" | grep -m 1 test_ngraph-onnx; then
        echo Test image found so no need to build it again
        return $?
    else
        echo -----------------------Build Test image-----------------------------------
        BASE_IMAGE=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep -m 1 base_ngraph-onnx)
        docker build --build-arg BASE_IMAGE=$BASE_IMAGE -t test_ngraph-onnx -f .ci/jenkins/test_ngraph-onnx.dockerfile .
        return $?
    fi
}

run_test_img()
{
    echo ------------------------Run test image------------------------------------
    if docker ps -a | grep ngraph-onnx_jenkins > /dev/null; then
        docker rm ngraph-onnx_jenkins --force
    fi
    docker run --name ngraph-onnx_jenkins test_ngraph-onnx
    return $?
}

remove_test_image()
{
    echo ---------------------Remove test image------------------------------------
    docker rm ngraph-onnx_jenkins --force && docker rmi test_ngraph-onnx --force
    return $?
}

remove_base_image()
{
    echo ---------------------Remove base image------------------------------------
    docker rmi base_ngraph-onnx --force
    return $?
}

#Main
if [ "$#" -eq 0 ]; then
    if create_base_img && create_test_img && run_test_img; then
        echo Testing returned success
    else
        echo Something went wrong
    fi

    echo Do you want me to remove the test image? y/n
    read decission
    if [ $decission = "y" ] || [ $decission = "yes" ]; then
        remove_test_image
        echo Would you like to remove base image as well? y/n
        read decission
        if [ $decission = "y" ] || [ $decission = "yes" ]; then
            remove_base_image
        fi
    fi

else
    case $1 in
    --create-base-img)
        create_base_img
        ;;
    --create-test-img)
        create_test_img
        ;;
    --run-test-img)
        run_test_img
        ;;
    --remove-test-img)
        remove_test_image
        ;;
    --remove-base-img)
        remove_base_image
        ;;
    --help)
        print_help
        ;;
    *)
        print_help
        ;;
    esac
fi
