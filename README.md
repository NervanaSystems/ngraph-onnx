# ngraph-onnx [![Build Status](https://travis-ci.org/NervanaSystems/ngraph-onnx.svg?branch=master)](https://travis-ci.org/NervanaSystems/ngraph-onnx/branches)

nGraph Backend for ONNX.

This repository contains tools to run [ONNX](http://onnx.ai/) models using the [Intel® nGraph™ library](https://github.com/NervanaSystems/ngraph) as a backend.

## Installation

### Prerequisites

Python 3.4 or higher is required.

####  Protocol Buffers

You will need Protocol Buffers `v.2.6.1` or higher installed on your system to use ONNX.

On Ubuntu, for exmaple you can install protobuf using:

    # apt install protobuf-compiler libprotobuf-dev

And on Mac OS you can install protobuf using Homebrew:

    $ brew install protobuf


You can verify whether you have version `>=2.6.1` installed using the command:

    $ protoc --version
    libprotoc 3.4.0


#### nGraph

The other requirement is of course nGraph and nGraph's Python bindings.
You can follow these instructions to build an ngraph Python wheel containing both.

##### nGraph build process on Ubuntu 16.04

Prepare System:

    # apt update
    # apt install python3 python3-pip python3-dev
    # apt install build-essential cmake curl clang-3.9 git zlib1g zlib1g-dev libtinfo-dev

Build ngraph and install it into `$HOME/ngraph_dist`:

    $ git clone https://github.com/NervanaSystems/ngraph.git
    $ mkdir ngraph/build
    $ cd ngraph/build
    $ cmake ../ -DNGRAPH_USE_PREBUILT_LLVM=TRUE
    $ make -j 8
    $ make install

Build Python package (Binary wheel) for ngraph:

    $ cd ngraph/python
    $ git clone --recursive -b allow-nonconstructible-holders https://github.com/jagerman/pybind11.git
    $ export PYBIND_HEADERS_PATH=$PWD/pybind11
    $ export NGRAPH_CPP_BUILD_PATH=$HOME/ngraph_dist
    $ python3 setup.py bdist_wheel

For additional information how to build ngraph Python bindings see:

https://github.com/NervanaSystems/ngraph/blob/master/python/README.md

Once the Python binary wheel file (`ngraph-*.whl`) is prepared you can install it using pip.

For example:

    (your_venv) $ pip install -U dist/ngraph-0.2.0-cp35-cp35m-linux_x86_64.whl

You can check that ngraph is properly installed in your Python shell:

```python
>>> import ngraph as ng
>>> ng.abs([[1, 2, 3], [4, 5, 6]])
<Abs: 'Abs_1' ([2, 3])>
```

If you don't see any errors, ngraph should be installed correctly.


### Installing ngraph-onnx

You can install ngraph-onnx using pip:

     (your_venv) $ pip install git+https://github.com/NervanaSystems/ngraph-onnx/


## Usage example

### Importing the ONNX model

```python
# Import ONNX and load an ONNX file from disk
>>> import onnx
>>> onnx_protobuf = onnx.load('/path/to/ResNet20_CIFAR10_model.onnx')

# Convert ONNX model to an ngraph model
>>> from ngraph_onnx.onnx_importer.importer import import_onnx_model
>>> ng_models = import_onnx_model(onnx_protobuf)

# The importer returns a list of ngraph models for every ONNX graph output:
>>> print(ng_models)
[{
    'name': 'Plus5475_Output_0',
    'output': <Add: 'Add_1972' ([1, 10])>,
    'inputs': [<Parameter: 'Parameter_1104' ([1, 3, 32, 32], float)>]
 }]
```

The `output` field contains the ngraph node corrsponding to the output node in the imported ONNX computational graph.
The `inputs` list contains all input parameters for the computation which generates the output.

### Running a computation

After importing the ONNX model, you can use it to generate and call a computation function.

```python
# Using an ngraph runtime (CPU backend) create a callable computation
>>> import ngraph as ng
>>> ng_model = ng_models[0]
>>> runtime = ng.runtime(manager_name='CPU')
>>> resnet = runtime.computation(ng_model['output'], *ng_model['inputs'])

# Load an image (or create a mock as in this example)
>>> import numpy as np
>>> picture = np.ones([1, 3, 32, 32])

# Run computation on the picture:
>>> resnet(picture)
array([[ 1.312082 , -1.6729496,  4.2079577,  1.4012241, -3.5463796,
         2.3433776,  1.7799224, -1.6155214,  0.0777044, -4.2944093]],
      dtype=float32)
```

### Supported ONNX operations

* Abs
* Add
* And
* AveragePool
* BatchNormalization
* Ceil
* Constant
* Conv
* Div
* Dot
* Elu
* Equal
* Exp
* Floor
* Gemm
* Greater
* HardSigmoid
* LeakyRelu
* Less
* Log
* LogSoftmax
* MatMul
* Max
* MaxPool
* Mean
* Min
* Mul
* Neg
* Not
* Or
* PRelu
* Reciprocal
* ReduceLogSumExp
* ReduceMax
* ReduceMean
* ReduceMin
* ReduceProd
* ReduceSum
* Relu
* Reshape
* Selu
* Sigmoid
* Softmax
* Softplus
* Softsign
* Sqrt
* Sub
* Sum
* Tanh
* Xor

Refer to ONNX docs for the complete
[operator list](https://github.com/onnx/onnx/blob/master/docs/Operators.md).
