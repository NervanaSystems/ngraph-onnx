#!/usr/bin/env bash

set -e

# -----------------------------------------------------
# print utility colors
# -----------------------------------------------------
END='\033[0m'
RED='\033[00;31m'

function error() {
   echo -e "${RED}[ERROR]: \"$1\"${END}"
}

# -----------------------------------------------------
# Check for copyright notice existence
# -----------------------------------------------------

FILE_NAMES=( $(find `pwd`/ngraph_onnx/onnx_importer/ `pwd`/tests/ -name '*.py' -not -size 0 -print) )

for (( i=0; i<${#FILE_NAMES[@]}; i++ ))
do
    file_name=${FILE_NAMES[i]}
    if ! grep -L -q -E '#.*Copyright.*Intel\ Corporation.*$' ${file_name}; then
        error "Error: copyright notice not found in file: ${file_name}";
        fail=1;
    fi;
done
if [ -n ${fail} ]; then
    exit 1;
fi
