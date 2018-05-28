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

from tests.utils import run_node


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


@pytest.mark.parametrize('input_data', [
    np.array([4, 0, 5, 10]),
    np.array([[4, 0, 5, 10], [4, 0, 5, 10]]),
    np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]),
])
def test_sqrt(input_data):
    expected_output = np.sqrt(input_data)
    node = onnx.helper.make_node('Sqrt', inputs=['x'], outputs=['y'])
    ng_results = run_node(node, [input_data])
    assert np.allclose(ng_results, [expected_output])


@pytest.mark.parametrize('input_data', [
    np.array([4, 0, 5, 10]),
    np.array([[4, 0, 5, 10], [4, 0, 5, 10]]),
    np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]),
])
def test_exp(input_data):
    expected_output = np.exp(input_data)
    node = onnx.helper.make_node('Exp', inputs=['x'], outputs=['y'])
    ng_results = run_node(node, [input_data])
    assert np.allclose(ng_results, [expected_output])


@pytest.mark.parametrize('input_data', [
    np.array([4, 2, 5, 10]),
    np.array([[4, 1, 5, 10], [4, 2, 5, 10]]),
    np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]),
])
def test_log(input_data):
    expected_output = np.log(input_data)
    node = onnx.helper.make_node('Log', inputs=['x'], outputs=['y'])
    ng_results = run_node(node, [input_data])
    assert np.allclose(ng_results, [expected_output])


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


@pytest.mark.parametrize('min_value, max_value', [
    (np.finfo(np.float32).min, np.finfo(np.float32).max),
    (0., np.finfo(np.float32).max),
    (-0.5, 0.5),
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


def test_softplus():
    def softplus(x):
        return np.log(np.exp(x) + 1)

    np.random.seed(133391)
    data = np.random.randn(3, 4, 5).astype(np.float32)

    node = onnx.helper.make_node('Softplus', inputs=['x'], outputs=['y'])
    expected = softplus(data)
    ng_results = run_node(node, [data])
    assert np.allclose(ng_results, [expected])


def test_softsign():
    def softsign(x):
        return x / (1 + np.abs(x))

    np.random.seed(133391)
    data = np.random.randn(3, 4, 5).astype(np.float32)

    node = onnx.helper.make_node('Softsign', inputs=['x'], outputs=['y'])
    expected = softsign(data)
    ng_results = run_node(node, [data])
    assert np.allclose(ng_results, [expected])
