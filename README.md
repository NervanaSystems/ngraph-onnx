# DISCONTINUATION OF PROJECT #
This project will no longer be maintained by Intel.
This project has been identified as having known security escapes.
Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.
Intel no longer accepts patches to this project.
# ngraph-onnx [![Build Status](https://travis-ci.org/NervanaSystems/ngraph-onnx.svg?branch=master)](https://travis-ci.org/NervanaSystems/ngraph-onnx/branches)

nGraph Backend for ONNX.

This repository contains tools to run [ONNX][onnx] models using the [Intel nGraph library][ngraph_github] as a backend.

## Installation

Follow our [build][building] instructions to install nGraph-ONNX from sources.

<!-- @TODO: Restore pip installation section when new wheels are on PyPI

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
-->

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
>>> ng_function = import_onnx_model(onnx_protobuf)

# The importer returns a list of ngraph models for every ONNX graph output:
>>> print(ng_function)
<Function: 'resnet50' ([1, 1000])>
```

This creates an nGraph `Function` object, which can be used to execute a computation on a chosen backend.

### Running a computation

After importing an ONNX model, you will have an nGraph `Function` object. 
Now you can create an nGraph `Runtime` backend and use it to compile your `Function` to a backend-specific `Computation` object.
Finally, you can execute your model by calling the created `Computation` object with input data.

```python
# Using an ngraph runtime (CPU backend) create a callable computation object
>>> import ngraph as ng
>>> runtime = ng.runtime(backend_name='CPU')
>>> resnet_on_cpu = runtime.computation(ng_function)

# Load an image (or create a mock as in this example)
>>> import numpy as np
>>> picture = np.ones([1, 3, 224, 224], dtype=np.float32)

# Run computation on the picture:
>>> resnet_on_cpu(picture)
[array([[2.16105007e-04, 5.58412226e-04, 9.70510227e-05, 5.76671446e-05,
         7.45318757e-05, 4.80892748e-04, 5.67404088e-04, 9.48728994e-05,
         ...
```

[onnx]: http://onnx.ai/
[onnx_model_zoo]: https://github.com/onnx/models
[ngraph_github]: https://github.com/NervanaSystems/ngraph
[building]: https://github.com/NervanaSystems/ngraph-onnx/blob/master/BUILDING.md
