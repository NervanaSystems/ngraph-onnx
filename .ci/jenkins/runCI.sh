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

SCRIPT=$(readlink -f "$0")
BASEDIR=$(dirname "$SCRIPT")

# This is to set workdir to ngraph-onnx no matter where this sh is started from
cd $BASEDIR && cd ../..

if docker images --format "{{.Repository}}:{{.Tag}}" | grep -m 1 base_ngraph-onnx; then
   echo Base image found so proceeding to build ngraph and onnx
else
   echo No base image found so need to create one
   docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy -t base_ngraph-onnx -f .ci/jenkins/base_ngraph-onnx.dockerfile .
fi

echo -----------------------Build Test image-----------------------------------
docker build --build-arg BASE_IMAGE=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep -m 1 base_ngraph-onnx) -t test_ngraph-onnx -f .ci/jenkins/test_ngraph-onnx.dockerfile .

echo ------------------------Run test image------------------------------------
docker run --name ngraph-onnx_jenkins test_ngraph-onnx

TEST_RES=$?

echo ------------------------Cleanup Docker------------------------------------
docker rm ngraph-onnx_jenkins --force
docker rmi test_ngraph-onnx --force

exit $TEST_RES