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

#SCRIPT=$(readlink -f "$0")
#BASEDIR=$(dirname "$SCRIPT")

#cd $BASEDIR && cd ../..

#docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy -t test_ngraph -f .ci/jenkins/Dockerfile .
docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy -t test_ngraph .

docker run test_ngraph

docker rmi test_ngraph --force
