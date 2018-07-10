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

import onnx
from onnx.helper import make_node, make_graph, make_tensor_value_info, make_model

import numpy as np
import pytest

from ngraph_onnx.onnx_importer.model_wrappers import NodeWrapper, GraphWrapper, ModelWrapper
from ngraph_onnx.onnx_importer.utils.conv import get_pads
from ngraph_onnx.onnx_importer.utils.matmul import has_matmul_compatible_shapes
from ngraph_onnx.onnx_importer.utils.numeric_limits import NumericLimits


def test_get_pads():
    def wrap_node(node):
        graph = make_graph([node], 'test_graph',
                           [make_tensor_value_info('X', onnx.TensorProto.FLOAT, (1, 1, 1, 1)),
                            make_tensor_value_info('Y', onnx.TensorProto.FLOAT, (1, 1, 1, 1))],
                           [make_tensor_value_info('Z', onnx.TensorProto.FLOAT, ())])
        model = ModelWrapper(make_model(graph))
        return NodeWrapper(node, GraphWrapper(graph, model))

    node = wrap_node(make_node('Conv', ['X', 'Y'], ['Z'], pads=(2, 3, 2, 3)))
    assert get_pads(node) == ([2, 3], [2, 3])

    node = wrap_node(make_node('Conv', ['X', 'Y'], ['Z'], pads=(1, 1, 2, 4)))
    assert get_pads(node) == ([1, 1], [2, 4])

    node = wrap_node(make_node('Conv', ['X', 'Y'], ['Z'], auto_pad='VALID', kernel_shape=(5, 5)))
    assert get_pads(node) == ([0, 0], [0, 0])

    node = wrap_node(make_node('Conv', ['X', 'Y'], ['Z'],
                               auto_pad='SAME_UPPER', kernel_shape=(5, 5)))
    assert get_pads(node) == ([2, 2], [2, 2])

    node = wrap_node(make_node('Conv', ['X', 'Y'], ['Z'],
                               auto_pad='SAME_UPPER', kernel_shape=(6, 6)))
    assert get_pads(node) == ([2, 2], [3, 3])


def test_matmul_compatible_shapes():
    # --------- Positive cases ---------
    # matrix @ matrix
    shape_a = [2, 3]
    shape_b = [3, 2]
    assert has_matmul_compatible_shapes(shape_a, shape_b)

    # matrix @ vector
    shape_a = [2, 4]
    shape_b = [4]
    assert has_matmul_compatible_shapes(shape_a, shape_b)

    # vector @ matrix
    shape_a = [3]
    shape_b = [3, 2]
    assert has_matmul_compatible_shapes(shape_a, shape_b)

    # vector @ vector
    shape_a = [4]
    shape_b = [4]
    assert has_matmul_compatible_shapes(shape_a, shape_b)

    # --------- Broadcasting cases ---------
    shape_a = [4, 1, 2, 3]
    shape_b = [3, 2]
    assert has_matmul_compatible_shapes(shape_a, shape_b)

    shape_a = [3, 2, 4]
    shape_b = [1, 1, 4, 10]
    assert has_matmul_compatible_shapes(shape_a, shape_b)

    # --------- Negative cases ---------
    # TODO


@pytest.mark.parametrize('data_type', [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    int,
])
def test_numeric_limits_integral(data_type):
    np_type_min = np.iinfo(data_type).min
    np_type_max = np.iinfo(data_type).max
    type_min = NumericLimits.min(data_type)
    type_max = NumericLimits.max(data_type)

    assert np.equal(np_type_min, type_min)
    assert np.equal(np_type_max, type_max)


@pytest.mark.parametrize('data_type', [
    np.float32,
    np.float64,
    float,
])
def test_numeric_limits_floating_point(data_type):
    np_type_min = np.finfo(data_type).min
    np_type_max = np.finfo(data_type).max
    type_min = NumericLimits.min(data_type)
    type_max = NumericLimits.max(data_type)

    assert np.equal(np_type_min, type_min)
    assert np.equal(np_type_max, type_max)


@pytest.mark.parametrize('data_type', [
    str,
    bool,
])
def test_numeric_limits_errors(data_type):

    with pytest.raises(ValueError):
        NumericLimits.max(data_type)

    with pytest.raises(ValueError):
        NumericLimits.min(data_type)
