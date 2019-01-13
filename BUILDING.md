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
    # apt install python3 python3-pip python3-dev
    # apt install build-essential cmake curl clang-3.9 git zlib1g zlib1g-dev libtinfo-dev

Clone nGraph's `v0.10.0` tag, build and install it into `$HOME/ngraph_dist`:

    $ git clone -b 'v0.10.0' --single-branch --depth 1 https://github.com/NervanaSystems/ngraph.git
    $ mkdir ngraph/build
    $ cd ngraph/build
    $ cmake ../ -DNGRAPH_USE_PREBUILT_LLVM=TRUE -DCMAKE_INSTALL_PREFIX=$HOME/ngraph_dist -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE
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

     (your_venv) $ pip install git+https://github.com/NervanaSystems/ngraph-onnx/@v0.10.0

