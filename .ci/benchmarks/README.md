# Using benchmarking script
Run test_ngraph.py to execute nGraph benchmark using ONNX model.

## Prerequisites
Follow [build](https://github.com/NervanaSystems/ngraph-onnx/blob/master/BUILDING.md) instructions to install nGraph-ONNX from sources.

## Running the script
Usage: `python3 test_ngraph.py [ARGUMENT] MODEL_PATH BACKEND`
Runs benchmark, using ONNX model located in `MODEL_PATH`, on a specified `BACKEND`.

Possible backends:
`cpu`
`gpu`
`intelgpu`

Required arguments:</br>
`--set_size`            size of generated input dataset</br>
`--output_file`         results output file name</br>

Optional arguments:</br>
`--print-freq`, `-p`    output print frequency (default: 10)</br>
`--batch_size`          size of a input batch (default: 1)</br>

Example execution command:</br>
`python3 test_ngraph.py --print-freq 1 --batch_size 1 --set_size 100 --output_file onnx_time ./vgg19.onnx intelgpu`
