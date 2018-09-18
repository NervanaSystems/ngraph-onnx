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

import numpy as np
import onnx
import pytest

from tests_core.utils import run_node


@pytest.mark.parametrize('onnx_op, numpy_func, data_type', [
    ('And', np.logical_and, np.bool),
    ('Or', np.logical_or, np.bool),
    ('Xor', np.logical_xor, np.bool),
    ('Equal', np.equal, np.int32),
    ('Greater', np.greater, np.int32),
    ('Less', np.less, np.int32),
])
def test_logical(onnx_op, numpy_func, data_type):
    node = onnx.helper.make_node(onnx_op, inputs=['A', 'B'], outputs=['C'], broadcast=1)

    input_a = np.array([[0, 1, -1], [0, 1, -1], [0, 1, -1]]).astype(data_type)
    input_b = np.array([[0, 0, 0], [1, 1, 1], [-1, -1, -1]]).astype(data_type)
    expected_output = numpy_func(input_a, input_b)
    ng_results = run_node(node, [input_a, input_b], opset_version=4)
    assert np.array_equal(ng_results, [expected_output])

    input_a = np.array([[0, 1, -1], [0, 1, -1], [0, 1, -1]]).astype(data_type)
    input_b = np.array(1).astype(data_type)
    expected_output = numpy_func(input_a, input_b)
    ng_results = run_node(node, [input_a, input_b], opset_version=4)
    assert np.array_equal(ng_results, [expected_output])


def test_logical_not():
    input_data = np.array([[0, 1, -1], [0, 1, -1], [0, 1, -1]])
    expected_output = np.logical_not(input_data)

    node = onnx.helper.make_node('Not', inputs=['X'], outputs=['Y'])
    ng_results = run_node(node, [input_data])
    assert np.array_equal(ng_results, [expected_output])
