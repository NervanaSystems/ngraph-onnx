# Building nGraph and nGraph-ONNX

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
    # apt install python3 python3-pip python3-dev python-virtualenv
    # apt install build-essential cmake curl clang-3.9 git zlib1g zlib1g-dev libtinfo-dev unzip autoconf automake libtool

Clone nGraph's `v0.14.0-rc.1` tag, build and install it into `$HOME/ngraph_dist`:

    $ cd # Change directory to where you would like to clone nGraph sources
    $ git clone -b 'v0.14.0-rc.1' --single-branch --depth 1 https://github.com/NervanaSystems/ngraph.git
    $ mkdir ngraph/build
    $ cd ngraph/build
    $ cmake ../ -DCMAKE_INSTALL_PREFIX=$HOME/ngraph_dist -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE -DNGRAPH_USE_PREBUILT_LLVM=TRUE 
    $ make
    $ make install

Prepare a Python virtual environment for nGraph (recommended):

    $ mkdir -p ~/.virtualenvs && cd ~/.virtualenvs
    $ virtualenv -p $(which python3) nGraph
    $ source nGraph/bin/activate
    (nGraph) $ 

Build Python package (Binary wheel) for nGraph:

    (nGraph) $ cd # Change directory to where you would like to clone nGraph sources
    (nGraph) $ cd ngraph/python
    (nGraph) $ git clone --recursive https://github.com/jagerman/pybind11.git
    (nGraph) $ export PYBIND_HEADERS_PATH=$PWD/pybind11
    (nGraph) $ export NGRAPH_CPP_BUILD_PATH=$HOME/ngraph_dist
    (nGraph) $ export NGRAPH_ONNX_IMPORT_ENABLE=TRUE
    (nGraph) $ pip install numpy
    (nGraph) $ python setup.py bdist_wheel

For additional information how to build nGraph Python bindings see:

https://github.com/NervanaSystems/ngraph/blob/master/python/README.md

Once the Python binary wheel file (`ngraph-*.whl`) is prepared you can install it using pip.

For example:

    (nGraph) $ pip install -U dist/ngraph_core-0.0.0.dev0-cp35-cp35m-linux_x86_64.whl

You can check that nGraph is properly installed in your Python shell:

```python
>>> import ngraph as ng
>>> ng.abs([[1, 2, 3], [4, 5, 6]])
<Abs: 'Abs_1' ([2, 3])>
```

Additionally check that nGraph and nGraph's Python wheel were both build with the `NGRAPH_ONNX_IMPORT_ENABLE` option:

```python
from ngraph.impl import onnx_import
```

If you don't see any errors, nGraph should be installed correctly.


### Installing ngraph-onnx

You can install ngraph-onnx using pip:

     (nGraph) $ pip install git+https://github.com/NervanaSystems/ngraph-onnx/@v0.14.0-rc1

