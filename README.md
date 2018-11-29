# ngraph-onnx [![Build Status](https://travis-ci.org/NervanaSystems/ngraph-onnx.svg?branch=master)](https://travis-ci.org/NervanaSystems/ngraph-onnx/branches)

nGraph Backend for ONNX.

This repository contains tools to run [ONNX](http://onnx.ai/) models using the [Intel® nGraph™ library](https://github.com/NervanaSystems/ngraph) as a backend.

## Installation

### Prerequisites

Python 3.4 or higher is required.

####  Protocol Buffers

You will need Protocol Buffers `v.2.6.1` or higher installed on your system to use ONNX.

On Ubuntu, for example you can install protobuf using:

    # apt install protobuf-compiler libprotobuf-dev

And on Mac OS you can install protobuf using Homebrew:

    $ brew install protobuf


You can verify whether you have version `>=2.6.1` installed using the command:

    $ protoc --version
    libprotoc 3.4.0


#### nGraph

The other requirement is of course nGraph and nGraph's Python bindings.
You can follow these instructions to build an nGraph Python wheel containing both.

##### nGraph build process on Ubuntu 16.04

Prepare System:

    # apt update
    # apt install python3 python3-pip python3-dev
    # apt install build-essential cmake curl clang-3.9 git zlib1g zlib1g-dev libtinfo-dev

Clone nGraph's `v0.10.0-rc.5` tag, build and install it into `$HOME/ngraph_dist`:

    $ git clone -b 'v0.10.0-rc.5' --single-branch --depth 1 https://github.com/NervanaSystems/ngraph.git
    $ mkdir ngraph/build
    $ cd ngraph/build
    $ cmake ../ -DNGRAPH_USE_PREBUILT_LLVM=TRUE -DCMAKE_INSTALL_PREFIX=$HOME/ngraph_dist
    $ make
    $ make install

Build Python package (Binary wheel) for nGraph:

    $ cd ngraph/python
    $ git clone --recursive -b allow-nonconstructible-holders https://github.com/jagerman/pybind11.git
    $ export PYBIND_HEADERS_PATH=$PWD/pybind11
    $ export NGRAPH_CPP_BUILD_PATH=$HOME/ngraph_dist
    $ python3 setup.py bdist_wheel

For additional information how to build nGraph Python bindings see:

https://github.com/NervanaSystems/ngraph/blob/master/python/README.md

Once the Python binary wheel file (`ngraph-*.whl`) is prepared you can install it using pip.

For example:

    (your_venv) $ pip install -U dist/ngraph-0.10.0-cp35-cp35m-linux_x86_64.whl

You can check that nGraph is properly installed in your Python shell:

```python
>>> import ngraph as ng
>>> ng.abs([[1, 2, 3], [4, 5, 6]])
<Abs: 'Abs_1' ([2, 3])>
```

If you don't see any errors, nGraph should be installed correctly.


### Installing ngraph-onnx

You can install ngraph-onnx using pip:

     (your_venv) $ pip install git+https://github.com/NervanaSystems/ngraph-onnx/@v0.10.0-rc.5


## Usage example

### Importing an ONNX model

You can download models from the [ONNX model zoo](https://github.com/onnx/models). For example ResNet-50:

```
$ wget https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz
$ tar -xzvf resnet50.tar.gz
```

Use the following Python commands to convert the downloaded model to an nGraph model:

```python
# Import ONNX and load an ONNX file from disk
>>> import onnx
>>> onnx_protobuf = onnx.load('resnet50/model.onnx')

# Convert ONNX model to an ngraph model
>>> from ngraph_onnx.onnx_importer.importer import import_onnx_model
>>> ng_models = import_onnx_model(onnx_protobuf)

# The importer returns a list of ngraph models for every ONNX graph output:
>>> print(ng_models)
[{
    'name': 'gpu_0/softmax_1',
    'output': <Softmax: 'gpu_0/softmax_1' ([1, 1000])>,
    'inputs': [<Parameter: 'gpu_0/data_0' ([1, 3, 224, 224], float)>]
}]
```

The `output` field contains the nGraph node corresponding to the output node in the imported ONNX computational graph.
The `inputs` list contains all input parameters for the computation which generates the output.

### Running a computation

After importing the ONNX model, you can use it to generate and call a computation function.

```python
# Using an ngraph runtime (CPU backend) create a callable computation
>>> import ngraph as ng
>>> ng_model = ng_models[0]
>>> runtime = ng.runtime(backend_name='CPU')
>>> resnet = runtime.computation(ng_model['output'], *ng_model['inputs'])

# Load an image (or create a mock as in this example)
>>> import numpy as np
>>> picture = np.ones([1, 3, 224, 224], dtype=np.float32)

# Run computation on the picture:
>>> resnet(picture)
array([[2.16105225e-04, 5.58412459e-04, 9.70510737e-05, 5.76671700e-05,
        1.81550844e-04, 3.28226888e-04, 3.09511415e-05, 1.93187807e-04,
        ...
```

### Unsupported ONNX operations

* ArgMax
* ArgMin
* GRU
* Gather
* GlobalLpPool
* Hardmax
* InstanceNormalization
* LSTM
* LpNormalization
* LpPool
* MaxRoiPool
* RNN
* RandomNormal
* RandomNormalLike
* RandomUniform
* RandomUniformLike
* SpaceToDepth
* Tile
* TopK

All other operators except experimental ones are supported. Refer to ONNX docs for the complete
[operator list](https://github.com/onnx/onnx/blob/master/docs/Operators.md).
