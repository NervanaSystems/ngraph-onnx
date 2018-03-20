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

from tests.utils import convert_and_calculate, all_arrays_equal


def test_reshape():
    data = np.arange(2560).reshape(16, 4, 4, 10)
    node = onnx.helper.make_node('Reshape', inputs=['x'], outputs=['y'], shape=(256, 10))
    expected_output = data.reshape(256, 10)

    ng_results = convert_and_calculate(node, [data], [expected_output])
    assert np.array_equal(ng_results, [expected_output])


@pytest.mark.parametrize('axis,expected_output', [
    (0, np.arange(120).reshape(1, 120)),
    (1, np.arange(120).reshape(2, 60)),
    (2, np.arange(120).reshape(6, 20)),
    (3, np.arange(120).reshape(24, 5)),
    (4, np.arange(120).reshape(120, 1)),
])
def test_flatten(axis, expected_output):
    data = np.arange(120).reshape(2, 3, 4, 5)
    node = onnx.helper.make_node('Flatten', inputs=['x'], outputs=['y'], axis=axis)
    ng_results = convert_and_calculate(node, [data], [expected_output])
    assert np.array_equal(ng_results, [expected_output])


def test_flatten_exception():
    data = np.arange(120).reshape(2, 3, 4, 5)
    node = onnx.helper.make_node('Flatten', inputs=['x'], outputs=['y'], axis=5)

    with pytest.raises(ValueError):
        convert_and_calculate(node, [data], [data])


def test_transpose():
    data = np.arange(120).reshape(2, 3, 4, 5)

    node = onnx.helper.make_node('Transpose', inputs=['x'], outputs=['y'])
    expected_output = data.T
    ng_results = convert_and_calculate(node, [data], [expected_output])
    assert np.array_equal(ng_results, [expected_output])

    node = onnx.helper.make_node('Transpose', inputs=['x'], outputs=['y'], perm=(3, 1, 0, 2))
    expected_output = np.transpose(data, axes=(3, 1, 0, 2))
    ng_results = convert_and_calculate(node, [data], [expected_output])
    assert np.array_equal(ng_results, [expected_output])


def test_slice():
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

    expected_output = np.array([[5, 6, 7]])
    node = onnx.helper.make_node('Slice', inputs=['x'], outputs=['y'],
                                 axes=[0, 1], starts=[1, 0], ends=[2, 3])
    ng_results = convert_and_calculate(node, [data], [expected_output])
    assert np.array_equal(ng_results, [expected_output])

    expected_output = np.array([[2, 3, 4]])
    node = onnx.helper.make_node('Slice', inputs=['x'], outputs=['y'], starts=[0, 1],
                                 ends=[-1, 1000])
    ng_results = convert_and_calculate(node, [data], [expected_output])
    assert np.array_equal(ng_results, [expected_output])

    node = onnx.helper.make_node('Slice', inputs=['x'], outputs=['y'], axes=[0, 1],
                                 starts=[0, 0], ends=[3, 10])
    data = np.random.randn(20, 10, 5).astype(np.float32)
    expected_output = data[0:3, 0:10]
    ng_results = convert_and_calculate(node, [data], [expected_output])
    assert np.array_equal(ng_results, [expected_output])

    # default axes
    node = onnx.helper.make_node('Slice', inputs=['x'], outputs=['y'], starts=[0, 0, 3],
                                 ends=[20, 10, 4])
    data = np.random.randn(20, 10, 5).astype(np.float32)
    expected_output = data[:, :, 3:4]
    ng_results = convert_and_calculate(node, [data], [expected_output])
    assert np.array_equal(ng_results, [expected_output])

    # end out of bounds
    node = onnx.helper.make_node('Slice', inputs=['x'], outputs=['y'], axes=[1], starts=[1],
                                 ends=[1000])
    data = np.random.randn(20, 10, 5).astype(np.float32)
    expected_output = data[:, 1:1000]
    ng_results = convert_and_calculate(node, [data], [expected_output])
    assert np.array_equal(ng_results, [expected_output])

    # negative value
    node = onnx.helper.make_node('Slice', inputs=['x'], outputs=['y'], axes=[1], starts=[0],
                                 ends=[-1])
    data = np.random.randn(20, 10, 5).astype(np.float32)
    expected_output = data[:, 0:-1]
    ng_results = convert_and_calculate(node, [data], [expected_output])
    assert np.array_equal(ng_results, [expected_output])

    # start ouf of bounds
    node = onnx.helper.make_node('Slice', inputs=['x'], outputs=['y'], axes=[1], starts=[1000],
                                 ends=[1000])
    data = np.random.randn(20, 10, 5).astype(np.float32)
    expected_output = data[:, 1000:1000]
    ng_results = convert_and_calculate(node, [data], [expected_output])
    assert np.array_equal(ng_results, [expected_output])


def test_concat():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]])
    expected_output = np.concatenate((a, b), axis=0)
    node = onnx.helper.make_node('Concat', inputs=['x', 'y'], outputs=['z'], axis=0)
    ng_results = convert_and_calculate(node, [a, b], [expected_output])
    assert np.array_equal(ng_results, [expected_output])

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]]).T
    expected_output = np.concatenate((a, b), axis=1)
    node = onnx.helper.make_node('Concat', inputs=['x', 'y'], outputs=['z'], axis=1)
    ng_results = convert_and_calculate(node, [a, b], [expected_output])
    assert np.array_equal(ng_results, [expected_output])

    test_cases = {
        '1d': ([1, 2],
               [3, 4]),
        '2d': ([[1, 2], [3, 4]],
               [[5, 6], [7, 8]]),
        '3d': ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
               [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]),
    }

    for test_case, values in test_cases.items():
        values = [np.asarray(v) for v in values]
        for i in range(len(values[0].shape)):
            in_args = ['value' + str(k) for k in range(len(values))]
            node = onnx.helper.make_node(
                'Concat',
                inputs=[s for s in in_args],
                outputs=['output'],
                axis=i,
            )
            expected_output = np.concatenate(values, i)
            ng_results = convert_and_calculate(node, [v for v in values], [expected_output])
            assert np.array_equal(ng_results, [expected_output])


@pytest.mark.skip(reason='Needs refactoring to ngraph++')
def test_squeeze():
    data = np.arange(6).reshape(1, 2, 3, 1)
    expected_output = data.reshape(2, 3)

    node = onnx.helper.make_node('Squeeze', inputs=['x'], outputs=['y'], axes=[0, 3])
    ng_results = convert_and_calculate(node, [data], [expected_output])
    assert np.array_equal(ng_results, [expected_output])


@pytest.mark.skip(reason='Needs refactoring to ngraph++')
@pytest.mark.parametrize('node,expected_output', [
    # Split into 2 equal parts along axis=0
    (onnx.helper.make_node('Split', inputs=['x'], outputs=['y', 'z'], axis=0),
     [np.array([[0, 1, 2, 3]]), np.array([[4, 5, 6, 7]])]),

    # Split into 2 equal parts along axis=1
    (onnx.helper.make_node('Split', inputs=['x'], outputs=['a', 'b'], axis=1),
     [np.array([[0, 1], [4, 5]]), np.array([[2, 3], [6, 7]])]),

    # Split into 4 equal parts along axis=1
    (onnx.helper.make_node('Split', inputs=['x'], outputs=['a', 'b', 'c', 'd'], axis=1),
     [np.array([[0], [4]]), np.array([[1], [5]]), np.array([[2], [6]]), np.array([[3], [7]])]),

    # Split into 2 unequal parts along axis=1
    (onnx.helper.make_node('Split', inputs=['x'], outputs=['a', 'b'], axis=1, split=(3, 1)),
     [np.array([[0, 1, 2], [4, 5, 6]]), np.array([[3], [7]])]),
])
def test_split(node, expected_output):
    data = np.arange(8).reshape(2, 4)
    ng_results = convert_and_calculate(node, [data], expected_output)
    assert all_arrays_equal(ng_results, expected_output)
