#!/usr/bin/env bash

set -e

# -----------------------------------------------------
# print utility colors
# -----------------------------------------------------
END='\033[0m'
RED='\033[00;31m'
GREEN='\033[00;32m'

function error() {
   echo -e "${RED}[ERROR]: \"$1\"${END}"
}

function info() {
  echo -e "${GREEN}[INFO]: \"$1\"${END}"
}

# -----------------------------------------------------
# Commandline arguments parsing
# -----------------------------------------------------

if [[ $# -lt 1 ]]; then
  error "This script needs at least one argument equal to path to directory containing tox.ini"
  exit 1
fi

if [[ -n $1 ]]; then
  info "toxinidir: $1"
  TOXINIDIR=$1
fi

# -----------------------------------------------------
# Check for copyright notice existence
# -----------------------------------------------------

FILE_NAMES=( $(find ${TOXINIDIR}/ngraph_onnx/onnx_importer/ ${TOXINIDIR}/tests/ -name '*.py' -not -size 0 -print) )

for (( i=0; i<${#FILE_NAMES[@]}; i++ ))
do
    file_name=${FILE_NAMES[i]}
    grep -L -q -E '#.*Copyright.*Intel\ Corporation.*$' ${file_name}
    if [[ $? -ne 0 ]]; then
        error "Error: copyright notice not found in file: ${file_name}"
        fail=1
    fi
done

if [[ -n $fail ]]; then
    info "There was at least one file without copyright notice found!"
    exit 1
fi
