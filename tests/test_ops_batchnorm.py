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

import numpy as np

from tests.utils import convert_and_calculate


def make_batch_norm_node(**node_attributes):
    return onnx.helper.make_node('BatchNormalization', inputs=['X', 'scale', 'B', 'mean', 'var'],
                                 outputs=['Y'], **node_attributes)


def test_batch_norm_test_mode():
    data = np.arange(48).reshape(1, 3, 4, 4)
    scale = np.ones((3,)).astype(np.float32)  # Gamma
    bias = np.zeros((3,)).astype(np.float32)  # Beta
    mean = np.mean(data, axis=(0, 2, 3))
    var = np.var(data, axis=(0, 2, 3))

    expected_output = np.array(
        [[[[-1.62694025, -1.41001487, -1.19308949, -0.97616416],
           [-0.75923878, -0.54231346, -0.32538807, -0.10846269],
           [0.10846269, 0.32538807, 0.54231334, 0.75923872],
           [0.9761641, 1.19308949, 1.41001487, 1.62694025]],

          [[-1.62694049, -1.41001511, -1.19308972, -0.97616434],
           [-0.7592392, -0.54231358, -0.32538843, -0.10846281],
           [0.10846233, 0.32538795, 0.5423131, 0.75923872],
           [0.97616386, 1.19308949, 1.41001463, 1.62694025]],

          [[-1.62694025, -1.41001511, -1.19308949, -0.97616434],
           [-0.75923872, -0.54231358, -0.32538795, -0.10846233],
           [0.10846233, 0.32538795, 0.54231358, 0.7592392],
           [0.97616386, 1.19308949, 1.41001511, 1.62694073]]]], dtype=np.float32)

    node = make_batch_norm_node(is_test=1, spatial=1)
    result = convert_and_calculate(node, [data, scale, bias, mean, var], [expected_output])
    assert np.isclose(result, expected_output).all()

    scale = np.broadcast_to(0.1, (3,)).astype(np.float32)  # Gamma
    bias = np.broadcast_to(1, (3,)).astype(np.float32)  # Beta

    expected_output = np.array(
        [[[[0.83730596, 0.85899848, 0.88069105, 0.90238357],
           [0.92407608, 0.94576865, 0.96746117, 0.98915374],
           [1.01084626, 1.03253877, 1.05423129, 1.07592392],
           [1.09761643, 1.11930895, 1.14100146, 1.16269398]],

          [[0.83730596, 0.85899854, 0.88069105, 0.90238357],
           [0.92407608, 0.94576865, 0.96746117, 0.98915374],
           [1.01084626, 1.03253877, 1.05423141, 1.07592392],
           [1.09761643, 1.11930895, 1.14100146, 1.16269398]],

          [[0.83730596, 0.85899848, 0.88069105, 0.90238357],
           [0.92407614, 0.94576865, 0.96746117, 0.98915374],
           [1.01084626, 1.03253877, 1.05423141, 1.07592392],
           [1.09761643, 1.11930895, 1.14100146, 1.16269398]]]], dtype=np.float32)

    node = make_batch_norm_node(is_test=1, spatial=1)
    result = convert_and_calculate(node, [data, scale, bias, mean, var], [expected_output])
    assert np.isclose(result, expected_output).all()
