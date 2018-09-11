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
import numpy as np

from onnx.helper import make_node, make_graph, make_tensor_value_info, make_model
from ngraph_onnx.core_importer.backend import NgraphBackend


@pytest.fixture()
def _get_data_shapes():
    a_shape = [5, 4]
    b_shape = [5, 4]
    c_shape = [4, 2]
    d_shape = [5, 2]
    out_shape = [5, 2]
    return a_shape, b_shape, c_shape, d_shape, out_shape


def _get_simple_model(a_shape, b_shape, c_shape, d_shape, out_shape):
    node1 = make_node('Add', inputs=['A', 'B'], outputs=['add1'], name='add_node')
    node2 = make_node('Abs', inputs=['add1'], outputs=['abs1'], name='abs_node')
    node3 = make_node('Gemm', inputs=['abs1', 'C', 'D'], outputs=['Y'], name='gemm_node')

    graph = make_graph([node1, node2, node3], 'test_graph',
                       [make_tensor_value_info('A', onnx.TensorProto.FLOAT, a_shape),
                        make_tensor_value_info('B', onnx.TensorProto.FLOAT, b_shape),
                        make_tensor_value_info('C', onnx.TensorProto.FLOAT, c_shape),
                        make_tensor_value_info('D', onnx.TensorProto.FLOAT, d_shape)],
                       [make_tensor_value_info('Y', onnx.TensorProto.FLOAT, out_shape)])
    model = make_model(graph, producer_name='ngraph ONNX Importer')
    return model


def _get_input_data(*inputs):
    np.random.seed(133391)
    data = [np.random.randn(*inpt).astype(np.float32) for inpt in inputs]
    return tuple(data)


@pytest.config.nnp_skip(reason='Do not run on NNP')
@pytest.config.gpu_skip(reason='Do not run on GPU')
@pytest.config.cpu_skip(reason='Do not run on CPU')
def test_supports_ngraph_device_interpreter():
    assert NgraphBackend.supports_ngraph_device('INTERPRETER')


@pytest.config.nnp_skip(reason='Do not run on NNP')
@pytest.config.gpu_skip(reason='Do not run on GPU')
@pytest.config.interpreter_skip(reason='Do not run on INTERPRETER')
def test_supports_ngraph_device_cpu():
    assert NgraphBackend.supports_ngraph_device('CPU')


@pytest.config.cpu_skip(reason='Do not run on CPU')
@pytest.config.gpu_skip(reason='Do not run on GPU')
@pytest.config.interpreter_skip(reason='Do not run on INTERPRETER')
def test_supports_ngraph_device_nnp():
    assert NgraphBackend.supports_ngraph_device('NNP')


@pytest.config.nnp_skip(reason='Do not run on NNP')
@pytest.config.cpu_skip(reason='Do not run on CPU')
@pytest.config.interpreter_skip(reason='Do not run on INTERPRETER')
def test_supports_ngraph_device_gpu():
    assert NgraphBackend.supports_ngraph_device('GPU')


@pytest.config.nnp_skip(reason='Do not run on NNP')
@pytest.config.cpu_skip(reason='Do not run on CPU')
@pytest.config.interpreter_skip(reason='Do not run on INTERPRETER')
def test_supports_device_gpu():
    assert NgraphBackend.supports_device('CUDA')


def test_run_model():
    a_shape, b_shape, c_shape, d_shape, out_shape = _get_data_shapes()
    input_a, input_b, input_c, input_d = _get_input_data(a_shape, b_shape, c_shape, d_shape)

    model = _get_simple_model(a_shape, b_shape, c_shape, d_shape, out_shape)

    ng_results = NgraphBackend.run_model(model, [input_a, input_b, input_c, input_d])
    expected = np.dot(np.abs(input_a + input_b), input_c) + input_d

    assert np.allclose(ng_results, [expected])


def test_run_node():
    input_data = _get_input_data([2, 3, 4, 5])
    node = onnx.helper.make_node('Abs', inputs=['x'], outputs=['y'])
    ng_results = NgraphBackend.run_node(node, input_data)
    expected = np.abs(input_data)
    assert np.array_equal(ng_results, expected)


def test_prepare():
    a_shape, b_shape, c_shape, d_shape, out_shape = _get_data_shapes()
    model = _get_simple_model(a_shape, b_shape, c_shape, d_shape, out_shape)
    backend = NgraphBackend.prepare(model)

    for idx in range(10):
        input_a, input_b, input_c, input_d = _get_input_data(a_shape, b_shape, c_shape, d_shape)
        ng_results = backend.run([input_a, input_b, input_c, input_d])
        expected = np.dot(np.abs(input_a + input_b), input_c) + input_d
        assert np.allclose(ng_results, [expected])
