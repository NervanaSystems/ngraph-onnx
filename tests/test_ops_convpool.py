# ----------------------------------------------------------------------------
# Copyright 2018 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import print_function, division

import onnx
import pytest

import ngraph_api as ng
import numpy as np
from onnx.helper import make_node, make_graph, make_tensor_value_info, make_model
from ngraph_onnx.onnx_importer.importer import import_onnx_model
from tests.utils import convert_and_calculate


@pytest.fixture
def ndarray_1x1x4x4():
    return np.array([[11, 12, 13, 14],
                     [15, 16, 17, 18],
                     [19, 20, 21, 22],
                     [23, 24, 25, 26]], dtype=np.float32).reshape(1, 1, 4, 4)


def make_onnx_model_for_conv_op(x_shape, weights_shape, transpose=False, **attributes):
    output_shape = ()  # We don't need output shape to be accurate for these tests

    if transpose:
        node_op = 'ConvTranspose'
    else:
        node_op = 'Conv'

    node = make_node(node_op, ['X', 'weight'], ['Y'], name='test_node', **attributes)
    graph = make_graph([node], 'test_graph',
                       [make_tensor_value_info('X', onnx.TensorProto.FLOAT, x_shape),
                        make_tensor_value_info('weight', onnx.TensorProto.FLOAT, weights_shape)],
                       [make_tensor_value_info('Y', onnx.TensorProto.FLOAT, output_shape)])
    model = make_model(graph, producer_name='ngraph ONNXImporter')
    return model


def import_and_compute_conv(x, weights, transpose=False, **attributes):
    x, weights = np.array(x), np.array(weights)
    onnx_model = make_onnx_model_for_conv_op(x.shape, weights.shape,
                                             transpose=transpose, **attributes)
    ng_model = import_onnx_model(onnx_model)[0]
    computation = get_transformer().computation(ng_model['output'], *ng_model['inputs'])
    return computation(x, weights)


def get_transformer():
    return ng.runtime()


def test_2d_conv():
    # x should have shape N(batch) x C x H x W
    input_x = np.array([
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.]], dtype=np.float32).reshape(1, 1, 9, 9)

    # filter weights should have shape M x C x kH x kW
    input_filter = np.array([
        [1., 0., -1.],
        [2., 0., -2.],
        [1., 0., -1.]], dtype=np.float32).reshape(1, 1, 3, 3)

    # convolution with padding=1 should produce 9 x 9 output:
    result = import_and_compute_conv(input_x, input_filter, pads=(1, 1, 1, 1), strides=(1, 1))
    assert np.array_equal(result,
                          np.array([[[[0., -15., -15., 15., 15., 0., 0., 0., 0.],
                                      [0., -20., -20., 20., 20., 0., 0., 0., 0.],
                                      [0., -20., -20., 20., 20., 0., 0., 0., 0.],
                                      [0., -20., -20., 20., 20., 0., 0., 0., 0.],
                                      [0., -20., -20., 20., 20., 0., 0., 0., 0.],
                                      [0., -20., -20., 20., 20., 0., 0., 0., 0.],
                                      [0., -20., -20., 20., 20., 0., 0., 0., 0.],
                                      [0., -20., -20., 20., 20., 0., 0., 0., 0.],
                                      [0., -15., -15., 15., 15., 0., 0., 0., 0.]]]],
                                   dtype=np.float32))

    # convolution with padding=0 should produce 7 x 7 output:
    result = import_and_compute_conv(input_x, input_filter, pads=(0, 0, 0, 0), strides=(1, 1))
    assert np.array_equal(result,
                          np.array([[[[-20, -20, 20, 20, 0, 0, 0],
                                      [-20, -20, 20, 20, 0, 0, 0],
                                      [-20, -20, 20, 20, 0, 0, 0],
                                      [-20, -20, 20, 20, 0, 0, 0],
                                      [-20, -20, 20, 20, 0, 0, 0],
                                      [-20, -20, 20, 20, 0, 0, 0],
                                      [-20, -20, 20, 20, 0, 0, 0]]]],
                                   dtype=np.float32))

    # convolution with strides=2 should produce 4 x 4 output:
    result = import_and_compute_conv(input_x, input_filter, pads=(0, 0, 0, 0), strides=(2, 2))
    assert np.array_equal(result,
                          np.array([[[[-20., 20., 0., 0.],
                                      [-20., 20., 0., 0.],
                                      [-20., 20., 0., 0.],
                                      [-20., 20., 0., 0.]]]],
                                   dtype=np.float32))

    # convolution with dilations=2 should produce 5 x 5 output:
    result = import_and_compute_conv(input_x, input_filter, dilations=(2, 2))
    assert np.array_equal(result,
                          np.array([[[[0, 0, 20, 20, 0],
                                      [0, 0, 20, 20, 0],
                                      [0, 0, 20, 20, 0],
                                      [0, 0, 20, 20, 0],
                                      [0, 0, 20, 20, 0]]]],
                                   dtype=np.float32))


@pytest.mark.skip(reason='Needs refactoring to ngraph++')
def test_3d_conv():
    # x should have shape N(batch) x C x H x W x D
    input_x = np.array([
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.]], dtype=np.float32).reshape(1, 1, 9, 9, 1)
    input_x = np.broadcast_to(input_x, (1, 1, 9, 9, 4))

    # filter weights should have shape M x C x kH x kW x kD
    input_filter = np.array([
        [1., 0., -1.],
        [2., 0., -2.],
        [1., 0., -1.]], dtype=np.float32).reshape(1, 1, 3, 3, 1)
    input_filter = np.broadcast_to(input_filter, (1, 1, 3, 3, 3))

    # convolution with padding=0 should produce 7 x 7 x 2 output:
    result = import_and_compute_conv(input_x, input_filter,
                                     pads=(0, 0, 0, 0, 0, 0), strides=(1, 1, 1))

    assert np.array_equal(np.moveaxis(result.squeeze(), (0, 1, 2), (1, 2, 0)),
                          np.array([[[-60., -60., 60., 60., 0., 0., 0.],
                                     [-60., -60., 60., 60., 0., 0., 0.],
                                     [-60., -60., 60., 60., 0., 0., 0.],
                                     [-60., -60., 60., 60., 0., 0., 0.],
                                     [-60., -60., 60., 60., 0., 0., 0.],
                                     [-60., -60., 60., 60., 0., 0., 0.],
                                     [-60., -60., 60., 60., 0., 0., 0.]],

                                    [[-60., -60., 60., 60., 0., 0., 0.],
                                     [-60., -60., 60., 60., 0., 0., 0.],
                                     [-60., -60., 60., 60., 0., 0., 0.],
                                     [-60., -60., 60., 60., 0., 0., 0.],
                                     [-60., -60., 60., 60., 0., 0., 0.],
                                     [-60., -60., 60., 60., 0., 0., 0.],
                                     [-60., -60., 60., 60., 0., 0., 0.]]],
                                   dtype=np.float32))


@pytest.mark.skip(reason='Needs refactoring to ngraph++')
def test_2d_conv_transpose():
    # x should have shape N(batch) x C x H x W
    input_x = np.array(
        [[0., -15., -15., 15., 15., 0., 0., 0., 0.],
         [0., -20., -20., 20., 20., 0., 0., 0., 0.],
         [0., -20., -20., 20., 20., 0., 0., 0., 0.],
         [0., -20., -20., 20., 20., 0., 0., 0., 0.],
         [0., -20., -20., 20., 20., 0., 0., 0., 0.],
         [0., -20., -20., 20., 20., 0., 0., 0., 0.],
         [0., -20., -20., 20., 20., 0., 0., 0., 0.],
         [0., -20., -20., 20., 20., 0., 0., 0., 0.],
         [0., -15., -15., 15., 15., 0., 0., 0., 0.]], dtype=np.float32).reshape(1, 1, 9, 9)

    # filter weights should have shape M x C x kH x kW
    input_filter = np.array([
        [1., 0., -1.],
        [2., 0., -2.],
        [1., 0., -1.]], dtype=np.float32).reshape(1, 1, 3, 3)

    # deconvolution with padding=1 should produce 9 x 9 output:
    result = import_and_compute_conv(input_x, input_filter, transpose=True,
                                     pads=(1, 1, 1, 1), strides=(1, 1))

    assert np.array_equal(result.reshape(9, 9),
                          np.array([[-50., -50., 100., 100., -50., -50., 0., 0., 0.],
                                    [-75., -75., 150., 150., -75., -75., 0., 0., 0.],
                                    [-80., -80., 160., 160., -80., -80., 0., 0., 0.],
                                    [-80., -80., 160., 160., -80., -80., 0., 0., 0.],
                                    [-80., -80., 160., 160., -80., -80., 0., 0., 0.],
                                    [-80., -80., 160., 160., -80., -80., 0., 0., 0.],
                                    [-80., -80., 160., 160., -80., -80., 0., 0., 0.],
                                    [-75., -75., 150., 150., -75., -75., 0., 0., 0.],
                                    [-50., -50., 100., 100., -50., -50., 0., 0., 0.]],
                                   dtype=np.float32))


@pytest.mark.skip(reason='Needs refactoring to ngraph++')
def test_padding():
    node = onnx.helper.make_node('Pad', inputs=['x'], outputs=['y'], pads=[1, 1, 1, 1])
    x = np.ones((2, 2), dtype=np.float32)
    y = np.pad(x, pad_width=1, mode='constant')

    ng_results = convert_and_calculate(node, [x], [y])
    assert np.array_equal(ng_results, [y])

    node = onnx.helper.make_node('Pad', inputs=['x'], outputs=['y'],
                                 mode='constant', pads=[0, 0, 1, 3, 0, 0, 2, 4])
    x = np.random.randn(1, 3, 4, 5).astype(np.float32)
    y = np.pad(x, pad_width=((0, 0), (0, 0), (1, 2), (3, 4)), mode='constant')

    ng_results = convert_and_calculate(node, [x], [y])
    assert np.array_equal(ng_results, [y])


def test_pool_average(ndarray_1x1x4x4):
    x = ndarray_1x1x4x4

    node = onnx.helper.make_node('AveragePool', inputs=['x'], outputs=['y'],
                                 kernel_shape=(2, 2), strides=(2, 2))
    y = np.array([[13.5, 15.5],
                  [21.5, 23.5]], dtype=np.float32).reshape(1, 1, 2, 2)
    ng_results = convert_and_calculate(node, [x], [y])
    assert np.array_equal(ng_results, [y])

    node = onnx.helper.make_node('AveragePool', inputs=['x'], outputs=['y'],
                                 kernel_shape=(2, 2), strides=(2, 2), pads=(1, 1, 1, 1))
    y = np.array([[11, 12.5, 14],
                  [17, 18.5, 20],
                  [23, 24.5, 26]], dtype=np.float32).reshape(1, 1, 3, 3)
    ng_results = convert_and_calculate(node, [x], [y])
    assert np.array_equal(ng_results, [y])


@pytest.mark.skip(reason='Needs refactoring to ngraph++')
def test_pool_average_3d(ndarray_1x1x4x4):
    x = np.broadcast_to(ndarray_1x1x4x4, (1, 1, 4, 4, 4))

    node = onnx.helper.make_node('AveragePool', inputs=['x'], outputs=['y'],
                                 kernel_shape=(2, 2, 2), strides=(2, 2, 2))
    y = np.array([[[13.5, 15.5],
                   [21.5, 23.5]],

                  [[13.5, 15.5],
                   [21.5, 23.5]]], dtype=np.float32).reshape(1, 1, 2, 2, 2)
    ng_results = convert_and_calculate(node, [x], [y])

    assert np.array_equal(ng_results, [y])


def test_pool_max(ndarray_1x1x4x4):
    node = onnx.helper.make_node('MaxPool', inputs=['x'], outputs=['y'],
                                 kernel_shape=(2, 2), strides=(2, 2))

    x = ndarray_1x1x4x4
    y = np.array([[16, 18],
                  [24, 26]], dtype=np.float32).reshape(1, 1, 2, 2)

    ng_results = convert_and_calculate(node, [x], [y])
    assert np.array_equal(ng_results, [y])


@pytest.mark.skip(reason='Needs refactoring to ngraph++')
def test_pool_global_max(ndarray_1x1x4x4):
    node = onnx.helper.make_node('GlobalMaxPool', inputs=['x'], outputs=['y'])

    x = ndarray_1x1x4x4
    y = np.array([26], dtype=np.float32).reshape(1, 1, 1, 1)

    ng_results = convert_and_calculate(node, [x], [y])
    assert np.array_equal(ng_results, [y])


@pytest.mark.skip(reason='Needs refactoring to ngraph++')
def test_pool_global_average(ndarray_1x1x4x4):
    node = onnx.helper.make_node('GlobalAveragePool', inputs=['x'], outputs=['y'])

    x = ndarray_1x1x4x4
    y = np.array([18.5], dtype=np.float32).reshape(1, 1, 1, 1)

    ng_results = convert_and_calculate(node, [x], [y])
    assert np.array_equal(ng_results, [y])


@pytest.mark.skip(reason='Needs refactoring to ngraph++')
def test_pool_global_average_3d(ndarray_1x1x4x4):
    x = np.broadcast_to(ndarray_1x1x4x4, (1, 1, 4, 4, 4))

    node = onnx.helper.make_node('GlobalAveragePool', inputs=['x'], outputs=['y'])
    y = np.array([18.5], dtype=np.float32).reshape(1, 1, 1, 1, 1)
    ng_results = convert_and_calculate(node, [x], [y])
    assert np.array_equal(ng_results, [y])
