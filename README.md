# ngraph-onnx [![Build Status](https://travis-ci.org/NervanaSystems/ngraph-onnx.svg?branch=master)](https://travis-ci.org/NervanaSystems/ngraph-onnx/branches)

nGraph Backend for ONNX. this is tes

This repository contains tools to run [ONNX][onnx] models using the [Intel nGraph library][ngraph_github] as a backend.

## Installation

nGraph and nGraph-ONNX are available as binary wheels you can install from PyPI.

nGraph binary wheels are currently tested on Ubuntu 16.04, if you're using a different system, you may want to [build][building] nGraph-ONNX from sources.

### Prerequisites

Python 3.4 or higher is required. 

    # apt update
    # apt install python3 python-virtualenv

### Using a virtualenv (optional)

You may wish to use a virutualenv for your installation.

    $ virtualenv -p $(which python3) venv
    $ source venv/bin/activate
    (venv) $

### Installing

    (venv) $ pip install ngraph-core
    (venv) $ pip install ngraph-onnx

## Usage example

### Importing an ONNX model

You can download models from the [ONNX model zoo][onnx_model_zoo]. For example ResNet-50:

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
[operator list][onnx_operators].

[onnx]: http://onnx.ai/
[onnx_model_zoo]: https://github.com/onnx/models
[onnx_operators]: https://github.com/onnx/onnx/blob/master/docs/Operators.md
[ngraph_github]: https://github.com/NervanaSystems/ngraph
[building]: https://github.com/NervanaSystems/ngraph-onnx/blob/master/BUILDING.md
