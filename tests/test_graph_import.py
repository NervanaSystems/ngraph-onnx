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
import ngraph as ng

from tests.utils import get_runtime
from onnx.helper import make_node, make_graph, make_tensor_value_info, make_model

from ngraph_onnx.onnx_importer.importer import import_onnx_model


def test_simple_graph():
    node1 = make_node('Add', ['A', 'B'], ['X'], name='add_node1')
    node2 = make_node('Add', ['X', 'C'], ['Y'], name='add_node2')
    graph = make_graph([node1, node2], 'test_graph',
                       [make_tensor_value_info('A', onnx.TensorProto.FLOAT, [1]),
                        make_tensor_value_info('B', onnx.TensorProto.FLOAT, [1]),
                        make_tensor_value_info('C', onnx.TensorProto.FLOAT, [1])],
                       [make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1])])
    model = make_model(graph, producer_name='ngraph ONNXImporter')

    ng_model = import_onnx_model(model)[0]

    runtime = get_runtime()
    computation = runtime.computation(ng_model['output'], *ng_model['inputs'])
    assert np.array_equal(computation(1, 2, 3), np.array([6.0], dtype=np.float32))
    assert np.array_equal(computation(4, 5, 6), np.array([15.0], dtype=np.float32))


def test_bad_data_shape():
    A = ng.parameter(shape=[2, 2], name='A', dtype=np.float32)
    B = ng.parameter(shape=[2, 2], name='B')
    model = (A + B)
    runtime = ng.runtime(backend_name='CPU')
    computation = runtime.computation(model, A, B)

    value_a = np.array([[1, 2]], dtype=np.float32)
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    try:
        computation(value_a, value_b)
    except ValueError:
        return

    assert False, 'Bad shape uncaught!!!'
