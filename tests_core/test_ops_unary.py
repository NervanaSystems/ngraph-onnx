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

from __future__ import print_function, division

import pytest
import onnx
import onnx.mapping
import numpy as np

from tests_core.utils import run_model, run_node, get_node_model, get_runtime
from onnx.helper import make_node, make_graph, make_tensor_value_info, make_model
from ngraph_onnx.core_importer.importer import import_onnx_model
from ngraph_onnx.core_importer.utils.types import np_dtype_to_tensor_type_name
from ngraph.exceptions import NgraphTypeError


@pytest.mark.xfail(reason='Refactoring to nGraph core importer.')
@pytest.mark.parametrize('input_data', [
    np.array([-4, 0, 5, -10]),
    np.array([[-4, 0, 5, -10], [-4, 0, 5, -10]]),
    np.array([[[1, 2], [-3, 4]], [[1, -2], [3, 4]]]),
])
def test_abs(input_data):
    expected_output = np.abs(input_data)
    node = onnx.helper.make_node('Abs', inputs=['x'], outputs=['y'])
    ng_results = run_node(node, [input_data])
    assert np.array_equal(ng_results, [expected_output])


@pytest.mark.xfail(reason='Refactoring to nGraph core importer.')
@pytest.mark.parametrize('input_data', [
    np.array([4, 0, 5, 10]),
    np.array([[4, 0, 5, 10], [4, 0, 5, 10]]),
    np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]),
])
def test_sqrt(input_data):
    input_data = input_data.astype(np.float32)
    expected_output = np.sqrt(input_data)
    node = onnx.helper.make_node('Sqrt', inputs=['x'], outputs=['y'])
    ng_results = run_node(node, [input_data])
    assert np.allclose(ng_results, [expected_output])


@pytest.mark.xfail(reason='Refactoring to nGraph core importer.')
@pytest.mark.parametrize('input_data', [
    np.array([4, 0, 5, 10]),
    np.array([[4, 0, 5, 10], [4, 0, 5, 10]]),
    np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]),
])
def test_exp(input_data):
    input_data = input_data.astype(np.float32)
    expected_output = np.exp(input_data)
    node = onnx.helper.make_node('Exp', inputs=['x'], outputs=['y'])
    ng_results = run_node(node, [input_data])
    assert np.allclose(ng_results, [expected_output])


@pytest.mark.xfail(reason='Refactoring to nGraph core importer.')
@pytest.mark.parametrize('input_data', [
    np.array([4, 2, 5, 10]),
    np.array([[4, 1, 5, 10], [4, 2, 5, 10]]),
    np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]),
])
def test_log(input_data):
    input_data = input_data.astype(np.float32)
    expected_output = np.log(input_data)
    node = onnx.helper.make_node('Log', inputs=['x'], outputs=['y'])
    ng_results = run_node(node, [input_data])
    assert np.allclose(ng_results, [expected_output])


@pytest.mark.xfail(reason='Refactoring to nGraph core importer.')
@pytest.mark.parametrize('input_data', [
    np.array([-4, 0, 5, -10]),
    np.array([[-4, 0, 5, -10], [-4, 0, 5, -10]]),
    np.array([[[1, 2], [-3, 4]], [[1, -2], [3, 4]]]),
])
def test_neg(input_data):
    expected_output = np.negative(input_data)
    node = onnx.helper.make_node('Neg', inputs=['x'], outputs=['y'])
    ng_results = run_node(node, [input_data])
    assert np.array_equal(ng_results, [expected_output])


@pytest.mark.xfail(reason='Refactoring to nGraph core importer.')
@pytest.mark.parametrize('input_data', [
    np.array([-4.2, 0.43, 5.99, -10.01]),
    np.array([[-4.5, 0.99, 5.01, -10.00], [-4.5, 0.5, 5.1, 10.01]]),
    np.array([[[1, 2], [-3, 4]], [[1, -2], [3, 4]]]) / 6,
])
def test_floor(input_data):
    expected_output = np.floor(input_data)
    node = onnx.helper.make_node('Floor', inputs=['x'], outputs=['y'])
    ng_results = run_node(node, [input_data])
    assert np.array_equal(ng_results, [expected_output])


@pytest.mark.xfail(reason='Refactoring to nGraph core importer.')
@pytest.mark.parametrize('input_data', [
    np.array([-4.2, 0, 5.99, -10.01]),
    np.array([[-4.5, 0.99, 5.01, -10.00], [-4.5, 0.5, 5.1, 10.01]]),
    np.array([[[1, 2], [-3, 4]], [[1, -2], [3, 4]]]) / 6,
])
def test_ceil(input_data):
    expected_output = np.ceil(input_data)
    node = onnx.helper.make_node('Ceil', inputs=['x'], outputs=['y'])
    ng_results = run_node(node, [input_data])
    assert np.array_equal(ng_results, [expected_output])


@pytest.mark.xfail(reason='Refactoring to nGraph core importer.')
@pytest.mark.parametrize('min_value, max_value', [
    (np.finfo(np.float32).min, np.finfo(np.float32).max),
    (-0.5, 0.5),
    (0., np.finfo(np.float32).max),
])
def test_clip(min_value, max_value):
    np.random.seed(133391)
    data = (np.float32(-100.) +
            np.random.randn(3, 4, 5).astype(np.float32) * np.float32(200.))

    node = onnx.helper.make_node('Clip', inputs=['x'], outputs=['y'],
                                 min=float(min_value), max=float(max_value))
    expected = np.clip(data, min_value, max_value)
    ng_results = run_node(node, [data])
    assert np.allclose(ng_results, [expected])


@pytest.mark.xfail(reason='Refactoring to nGraph core importer.')
def test_clip_default():
    np.random.seed(133391)
    data = -100. + np.random.randn(3, 4, 5).astype(np.float32) * 200.0

    node = onnx.helper.make_node('Clip', inputs=['x'], outputs=['y'], min=0.)
    expected = np.clip(data, np.float32(0.), np.finfo(np.float32).max)
    ng_results = run_node(node, [data])
    assert np.allclose(ng_results, [expected])

    node = onnx.helper.make_node('Clip', inputs=['x'], outputs=['y'], max=0.)
    expected = np.clip(data, np.finfo(np.float32).min, np.float32(0.))
    ng_results = run_node(node, [data])
    assert np.allclose(ng_results, [expected])


@pytest.mark.xfail(reason='Refactoring to nGraph core importer.')
@pytest.mark.parametrize('input_data', [
    np.array([-4.2, 1, 5.99, -10.01]),
    np.array([[-4.5, 0.99, 5.01, -10.00], [-4.5, 0.5, 5.1, 10.01]]),
    np.array([[[1, 2], [-3, 4]], [[1, -2], [3, 4]]]) / 6,
])
def test_reciprocal(input_data):
    expected_output = np.reciprocal(input_data)
    node = onnx.helper.make_node('Reciprocal', inputs=['x'], outputs=['y'])
    ng_results = run_node(node, [input_data])
    assert np.allclose(ng_results, [expected_output])


# -> NGRAPH-1839
@pytest.mark.xfail(reason='Need nGraph support for ArgMin/ArgMax')
@pytest.mark.parametrize('axis, dim1, dim2', [
    (0, 1, 60),
    (1, 3, 20),
    (2, 12, 5),
])
def test_hardmax(axis, dim1, dim2):
    def hardmax_2d(data):
        return np.eye(data.shape[1], dtype=data.dtype)[np.argmax(data, axis=1)]

    np.random.seed(133391)
    data = np.random.rand(3, 4, 5).astype(np.float32)
    expected = hardmax_2d(data.reshape(dim1, dim2)).reshape(3, 4, 5)
    node = onnx.helper.make_node('Hardmax', inputs=['x'], outputs=['y'], axis=axis)
    ng_results = run_node(node, [data])
    assert np.allclose(ng_results, [expected])


# -> NGRAPH-1839
@pytest.mark.xfail(reason='Need nGraph support for ArgMin/ArgMax')
def test_hardmax_special_cases():
    def hardmax_2d(data):
        return np.eye(data.shape[1], dtype=data.dtype)[np.argmax(data, axis=1)]

    np.random.seed(133391)
    data = np.random.rand(3, 4, 5).astype(np.float32)

    # default axis=1
    expected = hardmax_2d(data.reshape(3, 20)).reshape(3, 4, 5)
    node = onnx.helper.make_node('Hardmax', inputs=['x'], outputs=['y'])
    ng_results = run_node(node, [data])
    assert np.allclose(ng_results, [expected])

    with pytest.raises(ValueError):
        node = onnx.helper.make_node('Hardmax', inputs=['x'], outputs=['y'], axis=-1)
        ng_results = run_node(node, [data])

    with pytest.raises(ValueError):
        node = onnx.helper.make_node('Hardmax', inputs=['x'], outputs=['y'], axis=3)
        ng_results = run_node(node, [data])

    # For multiple occurrences of the maximal values, the first occurrence is selected
    # for one-hot output
    data = np.array([[3, 3, 3, 1]]).astype(np.float32)
    expected = np.array([[1, 0, 0, 0]]).astype(np.float32)
    node = onnx.helper.make_node('Hardmax', inputs=['x'], outputs=['y'])
    ng_results = run_node(node, [data])
    assert np.allclose(ng_results, [expected])


@pytest.mark.xfail(reason='Refactoring to nGraph core importer.')
def test_hardsigmoid():
    def hardsigmoid(data, alpha=float(0.2), beta=float(0.5)):
        return np.clip(alpha * data + beta, 0, 1)

    np.random.seed(133391)
    alpha = np.random.rand()
    beta = np.random.rand()
    data = np.random.rand(3, 4, 5).astype(np.float32)

    expected = hardsigmoid(data, alpha, beta)
    node = onnx.helper.make_node('HardSigmoid', inputs=['x'], outputs=['y'], alpha=alpha,
                                 beta=beta)
    ng_results = run_node(node, [data])
    assert np.allclose(ng_results, [expected])

    expected = hardsigmoid(data)
    node = onnx.helper.make_node('HardSigmoid', inputs=['x'], outputs=['y'])
    ng_results = run_node(node, [data])
    assert np.allclose(ng_results, [expected])


@pytest.mark.xfail(reason='Refactoring to nGraph core importer.')
def test_softmax():
    def softmax_2d(x):
        max_x = np.max(x, axis=1).reshape((-1, 1))
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

    np.random.seed(133391)
    data = np.random.randn(3, 4, 5).astype(np.float32)

    node = onnx.helper.make_node('Softmax', inputs=['x'], outputs=['y'], axis=0)
    expected = softmax_2d(data.reshape(1, 60)).reshape(3, 4, 5)
    ng_results = run_node(node, [data])
    assert np.allclose(ng_results, [expected])

    node = onnx.helper.make_node('Softmax', inputs=['x'], outputs=['y'], axis=1)
    expected = softmax_2d(data.reshape(3, 20)).reshape(3, 4, 5)
    ng_results = run_node(node, [data])
    assert np.allclose(ng_results, [expected])

    # default axis is 1
    node = onnx.helper.make_node('Softmax', inputs=['x'], outputs=['y'])
    ng_results = run_node(node, [data])
    assert np.allclose(ng_results, [expected])

    node = onnx.helper.make_node('Softmax', inputs=['x'], outputs=['y'], axis=2)
    expected = softmax_2d(data.reshape(12, 5)).reshape(3, 4, 5)
    ng_results = run_node(node, [data])
    assert np.allclose(ng_results, [expected])

    node = onnx.helper.make_node('Softmax', inputs=['x'], outputs=['y'], axis=-1)
    expected = softmax_2d(data.reshape(12, 5)).reshape(3, 4, 5)
    ng_results = run_node(node, [data])
    assert np.allclose(ng_results, [expected])

    with pytest.raises(ValueError):
        node = onnx.helper.make_node('Softmax', inputs=['x'], outputs=['y'], axis=3)
        ng_results = run_node(node, [data])


@pytest.mark.xfail(reason='Refactoring to nGraph core importer.')
def test_logsoftmax():
    def logsoftmax_2d(x):
        max_x = np.max(x, axis=1).reshape((-1, 1))
        exp_x = np.exp(x - max_x)
        return x - max_x - np.log(np.sum(exp_x, axis=1).reshape((-1, 1)))

    np.random.seed(133391)
    data = np.random.randn(3, 4, 5).astype(np.float32)

    node = onnx.helper.make_node('LogSoftmax', inputs=['x'], outputs=['y'], axis=0)
    expected = logsoftmax_2d(data.reshape(1, 60)).reshape(3, 4, 5)
    ng_results = run_node(node, [data])
    assert np.allclose(ng_results, [expected])

    node = onnx.helper.make_node('LogSoftmax', inputs=['x'], outputs=['y'], axis=1)
    expected = logsoftmax_2d(data.reshape(3, 20)).reshape(3, 4, 5)
    ng_results = run_node(node, [data])
    assert np.allclose(ng_results, [expected])

    # default axis is 1
    node = onnx.helper.make_node('LogSoftmax', inputs=['x'], outputs=['y'])
    ng_results = run_node(node, [data])
    assert np.allclose(ng_results, [expected])

    node = onnx.helper.make_node('LogSoftmax', inputs=['x'], outputs=['y'], axis=2)
    expected = logsoftmax_2d(data.reshape(12, 5)).reshape(3, 4, 5)
    ng_results = run_node(node, [data])
    assert np.allclose(ng_results, [expected])

    with pytest.raises(ValueError):
        node = onnx.helper.make_node('LogSoftmax', inputs=['x'], outputs=['y'], axis=-1)
        ng_results = run_node(node, [data])

    with pytest.raises(ValueError):
        node = onnx.helper.make_node('LogSoftmax', inputs=['x'], outputs=['y'], axis=3)
        ng_results = run_node(node, [data])


@pytest.mark.xfail(reason='Refactoring to nGraph core importer.')
def test_softplus():
    def softplus(x):
        return np.log(np.exp(x) + 1)

    np.random.seed(133391)
    data = np.random.randn(3, 4, 5).astype(np.float32)

    node = onnx.helper.make_node('Softplus', inputs=['x'], outputs=['y'])
    expected = softplus(data)
    ng_results = run_node(node, [data])
    assert np.allclose(ng_results, [expected])


@pytest.mark.xfail(reason='Refactoring to nGraph core importer.')
def test_softsign():
    def softsign(x):
        return x / (1 + np.abs(x))

    np.random.seed(133391)
    data = np.random.randn(3, 4, 5).astype(np.float32)

    node = onnx.helper.make_node('Softsign', inputs=['x'], outputs=['y'])
    expected = softsign(data)
    ng_results = run_node(node, [data])
    assert np.allclose(ng_results, [expected])


@pytest.mark.xfail(reason='Refactoring to nGraph core importer.')
def test_identity():
    np.random.seed(133391)
    shape = [2, 4]
    input_data = np.random.randn(*shape).astype(np.float32)

    identity_node = make_node('Identity', inputs=['x'], outputs=['y'])
    ng_results = run_node(identity_node, [input_data])
    assert np.array_equal(ng_results, [input_data])

    node1 = make_node('Add', inputs=['A', 'B'], outputs=['add1'], name='add_node1')
    node2 = make_node('Identity', inputs=['add1'], outputs=['identity1'], name='identity_node1')
    node3 = make_node('Abs', inputs=['identity1'], outputs=['Y'], name='abs_node1')

    graph = make_graph([node1, node2, node3], 'test_graph',
                       [make_tensor_value_info('A', onnx.TensorProto.FLOAT, shape),
                        make_tensor_value_info('B', onnx.TensorProto.FLOAT, shape)],
                       [make_tensor_value_info('Y', onnx.TensorProto.FLOAT, shape)])
    model = make_model(graph, producer_name='ngraph ONNX Importer')
    ng_model_function = import_onnx_model(model)[0]
    runtime = get_runtime()
    computation = runtime.computation(ng_model_function)
    ng_results = computation(input_data, input_data)
    expected_result = np.abs(input_data + input_data)

    assert np.array_equal(ng_results, expected_result)


@pytest.mark.xfail(reason='Refactoring to nGraph core importer.')
@pytest.mark.parametrize('val_type, input_data', [
    (np.dtype(bool), np.zeros((2, 2), dtype=int)),
])
def test_cast_to_bool(val_type, input_data):
    expected = np.array(input_data, dtype=val_type)

    model = get_node_model('Cast', input_data, opset=6,
                           to=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[val_type])
    result = run_model(model, [input_data])
    assert np.allclose(result, expected)

    model = get_node_model('Cast', input_data, opset=5, to=np_dtype_to_tensor_type_name(val_type))
    result = run_model(model, [input_data])
    assert np.allclose(result, expected)


@pytest.mark.xfail(reason='Refactoring to nGraph core importer.')
@pytest.mark.parametrize('val_type, range_start, range_end, in_dtype', [
    (np.dtype(np.float32), -8, 8, np.dtype(np.int32)),
    (np.dtype(np.float64), -16383, 16383, np.dtype(np.int64)),
])
def test_cast_to_float(val_type, range_start, range_end, in_dtype):
    np.random.seed(133391)
    input_data = np.random.randint(range_start, range_end, size=(2, 2), dtype=in_dtype)
    expected = np.array(input_data, dtype=val_type)

    model = get_node_model('Cast', input_data, opset=6,
                           to=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[val_type])
    result = run_model(model, [input_data])
    assert np.allclose(result, expected)

    model = get_node_model('Cast', input_data, opset=5, to=np_dtype_to_tensor_type_name(in_dtype))
    result = run_model(model, [input_data])
    assert np.allclose(result, expected)


@pytest.mark.xfail(reason='Refactoring to nGraph core importer.')
@pytest.mark.parametrize('val_type', [
    np.dtype(np.int8),
    np.dtype(np.int16),
    np.dtype(np.int32),
    np.dtype(np.int64),
])
def test_cast_to_int(val_type):
    np.random.seed(133391)
    input_data = np.ceil(-8 + np.random.rand(2, 3, 4) * 16)
    expected = np.array(input_data, dtype=val_type)

    model = get_node_model('Cast', input_data, opset=6,
                           to=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[val_type])
    result = run_model(model, [input_data])
    assert np.allclose(result, expected)

    model = get_node_model('Cast', input_data, opset=5, to=np_dtype_to_tensor_type_name(val_type))
    result = run_model(model, [input_data])
    assert np.allclose(result, expected)


@pytest.mark.xfail(reason='Refactoring to nGraph core importer.')
@pytest.mark.parametrize('val_type', [
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.uint32),
    np.dtype(np.uint64),
])
def test_cast_to_uint(val_type):
    np.random.seed(133391)
    input_data = np.ceil(np.random.rand(2, 3, 4) * 16)
    expected = np.array(input_data, dtype=val_type)

    model = get_node_model('Cast', input_data, opset=6,
                           to=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[val_type])
    result = run_model(model, [input_data])
    assert np.allclose(result, expected)

    model = get_node_model('Cast', input_data, opset=5, to=np_dtype_to_tensor_type_name(val_type))
    result = run_model(model, [input_data])
    assert np.allclose(result, expected)


@pytest.mark.xfail(reason='Refactoring to nGraph core importer.')
def test_cast_errors():
    np.random.seed(133391)
    input_data = np.ceil(np.random.rand(2, 3, 4) * 16)

    # missing 'to' attribute
    node = onnx.helper.make_node('Cast', inputs=['A'], outputs=['B'])
    input_tensors = [make_tensor_value_info(name, onnx.TensorProto.FLOAT, value.shape)
                     for name, value in zip(node.input, [input_data])]
    output_tensors = [make_tensor_value_info(name, onnx.TensorProto.FLOAT16, value.shape)
                      for name, value in zip(node.output, ())]  # type: ignore

    graph = make_graph([node], 'compute_graph', input_tensors, output_tensors)
    model = make_model(graph, producer_name='NgraphBackend')
    with pytest.raises(ValueError):
        import_onnx_model(model)[0]

    # unsupported data type representation
    node = onnx.helper.make_node('Cast', inputs=['A'], outputs=['B'], to=1.2345)
    input_tensors = [make_tensor_value_info(name, onnx.TensorProto.FLOAT, value.shape)
                     for name, value in zip(node.input, [input_data])]
    output_tensors = [make_tensor_value_info(name, onnx.TensorProto.INT32, value.shape)
                      for name, value in zip(node.output, ())]  # type: ignore

    graph = make_graph([node], 'compute_graph', input_tensors, output_tensors)
    model = make_model(graph, producer_name='NgraphBackend')
    with pytest.raises(ValueError):
        import_onnx_model(model)[0]

    # unsupported input tensor data type:
    node = onnx.helper.make_node('Cast', inputs=['A'], outputs=['B'], to=onnx.TensorProto.INT32)
    input_tensors = [make_tensor_value_info(name, onnx.TensorProto.COMPLEX64, value.shape)
                     for name, value in zip(node.input, [input_data])]
    output_tensors = [make_tensor_value_info(name, onnx.TensorProto.INT32, value.shape)
                      for name, value in zip(node.output, ())]  # type: ignore

    graph = make_graph([node], 'compute_graph', input_tensors, output_tensors)
    model = make_model(graph, producer_name='NgraphBackend')
    with pytest.raises((ValueError, NgraphTypeError)):
        import_onnx_model(model)[0]

    # unsupported output tensor data type:
    node = onnx.helper.make_node('Cast', inputs=['A'], outputs=['B'],
                                 to=onnx.TensorProto.COMPLEX128)
    input_tensors = [make_tensor_value_info(name, onnx.TensorProto.FLOAT, value.shape)
                     for name, value in zip(node.input, [input_data])]
    output_tensors = [make_tensor_value_info(name, onnx.TensorProto.COMPLEX128, value.shape)
                      for name, value in zip(node.output, ())]  # type: ignore

    graph = make_graph([node], 'compute_graph', input_tensors, output_tensors)
    model = make_model(graph, producer_name='NgraphBackend')
    with pytest.raises(ValueError):
        import_onnx_model(model)[0]


@pytest.mark.parametrize('value_type', [
    np.float32,
    np.float64,
])
def test_constant(value_type):
    values = np.random.randn(5, 5).astype(value_type)
    node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['values'],
        value=onnx.helper.make_tensor(
            name='const_tensor',
            data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(value_type)],
            dims=values.shape,
            vals=values.flatten()))

    ng_results = run_node(node, [])
    assert np.allclose(ng_results, [values])


# https://github.com/onnx/onnx/issues/1190
@pytest.mark.xfail(reason='/onnx/helper.py:152: TypeError: <value> has type '
                          'numpy.float16, but expected one of: int, long')
def test_constant_err():
    values = np.random.randn(5, 5).astype(np.float16)
    node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['values'],
        value=onnx.helper.make_tensor(
            name='const_tensor',
            data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(np.float16)],
            dims=values.shape,
            vals=values.flatten()))

    ng_results = run_node(node, [])
    assert np.allclose(ng_results, [values])
