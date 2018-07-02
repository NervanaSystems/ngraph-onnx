#!/usr/bin/env bash
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

set -e

# -----------------------------------------------------
# print utility colors
# -----------------------------------------------------
END='\033[0m'
RED='\033[00;31m'
GREEN='\033[00;32m'
BLUE='\033[00;34m'
LBLUE='\033[01;34m'

function error() {
   echo -e "${RED}[ERROR]: \"$1\"${END}"
}

function info() {
  echo -e "${GREEN}[INFO]: \"$1\"${END}"
}

function blue() {
  echo -e "${LBLUE} $1 ${END}"
}

function help() {
  blue "Usage:"
  blue "run_benchmarks.sh [-s|--size N]"
  blue "-s|--size N         The number of data to run inference on."
}

#--------------------------------------------------------------------------
#           CONSTANTS
#--------------------------------------------------------------------------

DATASET_SIZE=100

DOCKER_MAP_MODELS_DIR="-v `pwd`/models:/root/models"
DOCKER_MAP_RESULTS_DIR="-v `pwd`/results/:/root/results"

#--------------------------------------------------------------------------
#   Commandline arguments parsing
#--------------------------------------------------------------------------

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -s|--size)
        DATASET_SIZE="$2"
        shift # past argument
        shift # past value
    ;;
    -h|--help)
        help
        exit 0
    ;;
    *)    # unknown option
        POSITIONAL+=("$1") # save it in an array for later
        shift # past argument
    ;;
esac
done

DATASET_ARGS+=" --set_size ${DATASET_SIZE}"
#set -- "${POSITIONAL[@]}" # restore positional parameters

info "Using randomly generated data."

#--------------------------------------------------------------------------
#           BENCHMARKS
#--------------------------------------------------------------------------

info "--------------------------------------------------------------------------"
info "Running Pytorch native benchmark"
info "--------------------------------------------------------------------------"

docker build -t bmark_pytorch_native -f pytorch_native/Dockerfile .

docker run -it ${DOCKER_MAP_MODELS_DIR} bmark_pytorch_native \
            python3 scripts/pytorch_native.py --export_onnx_model

PYTORCH_THREADS=0
# Load data in separate subprocesses when loading real data.
if [[ -n "${IMAGE_NET_PATH+x}" ]]; then
    PYTORCH_THREADS=2
fi

info "docker run -it ${DOCKER_MAP_RESULTS_DIR} --env OMP_NUM_THREADS=1  bmark_pytorch_native \
            python3 scripts/pytorch_native.py -j ${PYTORCH_THREADS} -b 1 ${DATASET_ARGS}"
docker run -it ${DOCKER_MAP_RESULTS_DIR} --env OMP_NUM_THREADS=1  bmark_pytorch_native \
            python3 scripts/pytorch_native.py -j ${PYTORCH_THREADS} -b 1 ${DATASET_ARGS}

info "--------------------------------------------------------------------------"
info "Running nGraph inference benchmark on PyTorch model"
info "--------------------------------------------------------------------------"

docker build -t ngraph_cpu -f ngraph_cpu/Dockerfile .

info "docker run -it ${DOCKER_MAP_RESULTS_DIR} ${DOCKER_MAP_MODELS_DIR} ngraph_cpu \
       python3 scripts/test_ngraph.py ${DATASET_ARGS} --output_file bmark_pytorch_ngraph_cpu \
       models/pytorch_resnet50.onnx CPU"

docker run -it ${DOCKER_MAP_RESULTS_DIR} ${DOCKER_MAP_MODELS_DIR} ngraph_cpu \
       python3 scripts/test_ngraph.py ${DATASET_ARGS} --output_file bmark_pytorch_ngraph_cpu \
       models/pytorch_resnet50.onnx CPU

info "--------------------------------------------------------------------------"
info "Running Caffe2 native benchmark"
info "--------------------------------------------------------------------------"

# caffe2 comes with python 2.X as a default.
docker build -t bmark_caffe2_native -f caffe2_native/Dockerfile .

docker run -it ${DOCKER_MAP_MODELS_DIR} bmark_caffe2_native \
            python scripts/caffe2_native.py --export_onnx_model

info "docker run -it ${DOCKER_MAP_RESULTS_DIR} bmark_caffe2_native \
            python scripts/caffe2_native.py ${DATASET_ARGS}"
docker run -it ${DOCKER_MAP_RESULTS_DIR} bmark_caffe2_native \
            python scripts/caffe2_native.py ${DATASET_ARGS}


info "--------------------------------------------------------------------------"
info "Running nGraph inference benchmark on Caffe2 model"
info "--------------------------------------------------------------------------"

info "docker run -it ${DOCKER_MAP_RESULTS_DIR} ${DOCKER_MAP_MODELS_DIR} ngraph_cpu \
       python3 scripts/test_ngraph.py ${DATASET_ARGS} --output_file bmark_caffe2_ngraph_cpu \
       models/caffe2_resnet50.onnx CPU"

docker run -it ${DOCKER_MAP_RESULTS_DIR} ${DOCKER_MAP_MODELS_DIR} ngraph_cpu \
       python3 scripts/test_ngraph.py ${DATASET_ARGS} --output_file bmark_caffe2_ngraph_cpu \
       models/caffe2_resnet50.onnx CPU

info "--------------------------------------------------------------------------"
info "Running CNTK native benchmark"
info "--------------------------------------------------------------------------"

docker build -t bmark_cntk_native -f cntk_native/Dockerfile .

info "docker run -it ${DOCKER_MAP_RESULTS_DIR} bmark_cntk_native \
            python3 scripts/cntk_native.py ${DATASET_ARGS}"

docker run -it ${DOCKER_MAP_RESULTS_DIR} bmark_cntk_native \
            python3 scripts/cntk_native.py ${DATASET_ARGS}

info "--------------------------------------------------------------------------"
info "Running nGraph inference benchmark on CNTK model"
info "--------------------------------------------------------------------------"

info "docker run -it ${DOCKER_MAP_RESULTS_DIR} -v `pwd`/../models/cntk_ResNet50_ImageNet:/root/models  ngraph_cpu \
    python3 scripts/test_ngraph.py ${DATASET_ARGS} --output_file bmark_cntk_ngraph_models/model.onnx CPU"

docker run -it ${DOCKER_MAP_RESULTS_DIR} ${DOCKER_MAP_MODELS_DIR} ngraph_cpu \
    python3 scripts/test_ngraph.py ${DATASET_ARGS} --output_file bmark_cntk_ngraph_cpu models/cntk_resnet50.onnx CPU

info "--------------------------------------------------------------------------"
info "Running PaddlePaddle native benchmark"
info "--------------------------------------------------------------------------"

docker build -t bmark_paddlepaddle_native -f paddle_native/Dockerfile .

info ${DATASET_ARGS}

# run benchmark on randomly generated data
docker run -it ${DOCKER_MAP_RESULTS_DIR} bmark_paddlepaddle_native \
            python scripts/paddlepaddle_native.py --batch_size 1 --iterations ${DATASET_SIZE} --device CPU --use_mkldnn --infer_model_path models/pass0_mkldnn/ --data_set imagenet

info "--------------------------------------------------------------------------"
info "Running nGraph inference benchmark on PaddlePaddle model"
info "--------------------------------------------------------------------------"

docker build -t ngraph_cpu -f ngraph_cpu/Dockerfile .

info "docker run -it ${DOCKER_MAP_RESULTS_DIR} ${DOCKER_MAP_MODELS_DIR} ngraph_cpu \
    python3 scripts/test_ngraph.py ${DATASET_ARGS} --output_file bmark_PaddlePaddle_ngraph_cpu models/PaddlePaddle_resnet50.onnx CPU"

docker run -it ${DOCKER_MAP_RESULTS_DIR} ${DOCKER_MAP_MODELS_DIR} ngraph_cpu \
    python3 scripts/test_ngraph.py ${DATASET_ARGS} --output_file bmark_paddlepaddle_ngraph_cpu models/paddlepaddle_resnet50.onnx CPU

docker run -it ${DOCKER_MAP_RESULTS_DIR} ngraph_cpu \
    python3 scripts/print_results.py
