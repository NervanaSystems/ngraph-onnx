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

from tests.utils import convert_and_calculate


@pytest.mark.parametrize('input_data', [
    np.array([-4, 0, 5, -10]),
    np.array([[-4, 0, 5, -10], [-4, 0, 5, -10]]),
    np.array([[[1, 2], [-3, 4]], [[1, -2], [3, 4]]]),
])
def test_abs(input_data):
    expected_output = np.abs(input_data)
    node = onnx.helper.make_node('Abs', inputs=['x'], outputs=['y'])
    ng_results = convert_and_calculate(node, [input_data], [expected_output])
    assert np.array_equal(ng_results, [expected_output])


@pytest.mark.parametrize('input_data', [
    np.array([4, 0, 5, 10]),
    np.array([[4, 0, 5, 10], [4, 0, 5, 10]]),
    np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]),
])
def test_sqrt(input_data):
    expected_output = np.sqrt(input_data)
    node = onnx.helper.make_node('Sqrt', inputs=['x'], outputs=['y'])
    ng_results = convert_and_calculate(node, [input_data], [expected_output])
    assert np.allclose(ng_results, [expected_output])


@pytest.mark.parametrize('input_data', [
    np.array([4, 0, 5, 10]),
    np.array([[4, 0, 5, 10], [4, 0, 5, 10]]),
    np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]),
])
def test_exp(input_data):
    expected_output = np.exp(input_data)
    node = onnx.helper.make_node('Exp', inputs=['x'], outputs=['y'])
    ng_results = convert_and_calculate(node, [input_data], [expected_output])
    assert np.allclose(ng_results, [expected_output])


@pytest.mark.parametrize('input_data', [
    np.array([4, 2, 5, 10]),
    np.array([[4, 1, 5, 10], [4, 2, 5, 10]]),
    np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]),
])
def test_log(input_data):
    expected_output = np.log(input_data)
    node = onnx.helper.make_node('Log', inputs=['x'], outputs=['y'])
    ng_results = convert_and_calculate(node, [input_data], [expected_output])
    assert np.allclose(ng_results, [expected_output])


@pytest.mark.parametrize('input_data', [
    np.array([-4, 0, 5, -10]),
    np.array([[-4, 0, 5, -10], [-4, 0, 5, -10]]),
    np.array([[[1, 2], [-3, 4]], [[1, -2], [3, 4]]]),
])
def test_neg(input_data):
    expected_output = np.negative(input_data)
    node = onnx.helper.make_node('Neg', inputs=['x'], outputs=['y'])
    ng_results = convert_and_calculate(node, [input_data], [expected_output])
    assert np.array_equal(ng_results, [expected_output])


@pytest.mark.parametrize('input_data', [
    np.array([-4.2, 0.43, 5.99, -10.01]),
    np.array([[-4.5, 0.99, 5.01, -10.00], [-4.5, 0.5, 5.1, 10.01]]),
    np.array([[[1, 2], [-3, 4]], [[1, -2], [3, 4]]]) / 6,
])
def test_floor(input_data):
    expected_output = np.floor(input_data)
    node = onnx.helper.make_node('Floor', inputs=['x'], outputs=['y'])
    ng_results = convert_and_calculate(node, [input_data], [expected_output])
    assert np.array_equal(ng_results, [expected_output])


@pytest.mark.parametrize('input_data', [
    np.array([-4.2, 0, 5.99, -10.01]),
    np.array([[-4.5, 0.99, 5.01, -10.00], [-4.5, 0.5, 5.1, 10.01]]),
    np.array([[[1, 2], [-3, 4]], [[1, -2], [3, 4]]]) / 6,
])
def test_ceil(input_data):
    expected_output = np.ceil(input_data)
    node = onnx.helper.make_node('Ceil', inputs=['x'], outputs=['y'])
    ng_results = convert_and_calculate(node, [input_data], [expected_output])
    assert np.array_equal(ng_results, [expected_output])


@pytest.mark.parametrize('input_data', [
    np.array([-4.2, 1, 5.99, -10.01]),
    np.array([[-4.5, 0.99, 5.01, -10.00], [-4.5, 0.5, 5.1, 10.01]]),
    np.array([[[1, 2], [-3, 4]], [[1, -2], [3, 4]]]) / 6,
])
def test_reciprocal(input_data):
    expected_output = np.reciprocal(input_data)
    node = onnx.helper.make_node('Reciprocal', inputs=['x'], outputs=['y'])
    ng_results = convert_and_calculate(node, [input_data], [expected_output])
    assert np.allclose(ng_results, [expected_output])


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
    ng_results = convert_and_calculate(node, [data], [expected])
    assert np.allclose(ng_results, [expected])

    expected = hardsigmoid(data)
    node = onnx.helper.make_node('HardSigmoid', inputs=['x'], outputs=['y'])
    ng_results = convert_and_calculate(node, [data], [expected])
    assert np.allclose(ng_results, [expected])


def test_softmax():
    def softmax_2d(x):
        max_x = np.max(x, axis=1).reshape((-1, 1))
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

    np.random.seed(133391)
    data = np.random.randn(3, 4, 5).astype(np.float32)

    node = onnx.helper.make_node('Softmax', inputs=['x'], outputs=['y'], axis=0)
    expected = softmax_2d(data.reshape(1, 60)).reshape(3, 4, 5)
    ng_results = convert_and_calculate(node, [data], [expected])
    assert np.allclose(ng_results, [expected])

    node = onnx.helper.make_node('Softmax', inputs=['x'], outputs=['y'], axis=1)
    expected = softmax_2d(data.reshape(3, 20)).reshape(3, 4, 5)
    ng_results = convert_and_calculate(node, [data], [expected])
    assert np.allclose(ng_results, [expected])

    # default axis is 1
    node = onnx.helper.make_node('Softmax', inputs=['x'], outputs=['y'])
    ng_results = convert_and_calculate(node, [data], [expected])
    assert np.allclose(ng_results, [expected])

    node = onnx.helper.make_node('Softmax', inputs=['x'], outputs=['y'], axis=2)
    expected = softmax_2d(data.reshape(12, 5)).reshape(3, 4, 5)
    ng_results = convert_and_calculate(node, [data], [expected])
    assert np.allclose(ng_results, [expected])

    node = onnx.helper.make_node('Softmax', inputs=['x'], outputs=['y'], axis=-1)
    expected = softmax_2d(data.reshape(12, 5)).reshape(3, 4, 5)
    ng_results = convert_and_calculate(node, [data], [expected])
    assert np.allclose(ng_results, [expected])

    with pytest.raises(ValueError):
        node = onnx.helper.make_node('Softmax', inputs=['x'], outputs=['y'], axis=3)
        ng_results = convert_and_calculate(node, [data], [expected])


def test_logsoftmax():
    def logsoftmax_2d(x):
        max_x = np.max(x, axis=1).reshape((-1, 1))
        exp_x = np.exp(x - max_x)
        return x - max_x - np.log(np.sum(exp_x, axis=1).reshape((-1, 1)))

    np.random.seed(133391)
    data = np.random.randn(3, 4, 5).astype(np.float32)

    node = onnx.helper.make_node('LogSoftmax', inputs=['x'], outputs=['y'], axis=0)
    expected = logsoftmax_2d(data.reshape(1, 60)).reshape(3, 4, 5)
    ng_results = convert_and_calculate(node, [data], [expected])
    assert np.allclose(ng_results, [expected])

    node = onnx.helper.make_node('LogSoftmax', inputs=['x'], outputs=['y'], axis=1)
    expected = logsoftmax_2d(data.reshape(3, 20)).reshape(3, 4, 5)
    ng_results = convert_and_calculate(node, [data], [expected])
    assert np.allclose(ng_results, [expected])

    # default axis is 1
    node = onnx.helper.make_node('LogSoftmax', inputs=['x'], outputs=['y'])
    ng_results = convert_and_calculate(node, [data], [expected])
    assert np.allclose(ng_results, [expected])

    node = onnx.helper.make_node('LogSoftmax', inputs=['x'], outputs=['y'], axis=2)
    expected = logsoftmax_2d(data.reshape(12, 5)).reshape(3, 4, 5)
    ng_results = convert_and_calculate(node, [data], [expected])
    assert np.allclose(ng_results, [expected])

    with pytest.raises(ValueError):
        node = onnx.helper.make_node('LogSoftmax', inputs=['x'], outputs=['y'], axis=-1)
        ng_results = convert_and_calculate(node, [data], [expected])

    with pytest.raises(ValueError):
        node = onnx.helper.make_node('LogSoftmax', inputs=['x'], outputs=['y'], axis=3)
        ng_results = convert_and_calculate(node, [data], [expected])
