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
import pytest

import ngraph as ng

from onnx.helper import make_node, make_graph, make_tensor_value_info, make_model
from ngraph_onnx.onnx_importer.backend import NgraphBackend
from string import ascii_uppercase


def get_runtime():
    return ng.runtime(backend_name=pytest.config.getoption('backend', default='CPU'))


def run_node(onnx_node, data_inputs):
    # type: (onnx.NodeProto, List[np.ndarray]) -> List[np.ndarray]
    """
    Convert ONNX node to ngraph node and perform computation on input data.

    :param onnx_node: ONNX NodeProto describing a computation node
    :param data_inputs: list of numpy ndarrays with input data
    :return: list of numpy ndarrays with computed output
    """
    NgraphBackend.backend_name = pytest.config.getoption('backend', default='CPU')
    if NgraphBackend.supports_ngraph_device(NgraphBackend.backend_name):
        return NgraphBackend.run_node(onnx_node, data_inputs)
    else:
        raise RuntimeError('The requested nGraph backend <' + NgraphBackend.backend_name +
                           '> is not supported!')


def run_model(onnx_model, data_inputs):
    # type: (onnx.ModelProto, List[np.ndarray]) -> List[np.ndarray]
    """
    Convert ONNX model to an ngraph model and perform computation on input data.

    :param onnx_model: ONNX ModelProto describing an ONNX model
    :param data_inputs: list of numpy ndarrays with input data
    :return: list of numpy ndarrays with computed output
    """
    NgraphBackend.backend_name = pytest.config.getoption('backend', default='CPU')
    if NgraphBackend.supports_ngraph_device(NgraphBackend.backend_name):
        return NgraphBackend.run_model(onnx_model, data_inputs)
    else:
        raise RuntimeError('The requested nGraph backend <' + NgraphBackend.backend_name +
                           '> is not supported!')


def get_node_model(op_type, *input_data, opset=1, num_outputs=1, **node_attributes):
    # type: (str, Any, Optional[int], Optional[int], Optional[Any]) -> onnx.ModelProto
    """Generate model with single requested node.

    :param op_type: The ONNX node operation.
    :param input_data: Optional list of input arguments for node.
    :param opset: The ONNX operation set version to use. Default to 4.
    :param num_outputs: The number of node outputs.
    :param node_attributes: Optional dictionary of node attributes.
    :return: Generated model with single node for requested ONNX operation.
    """
    node_inputs = [np.array(data) for data in input_data]
    num_inputs = len(node_inputs)
    node_input_names = [ascii_uppercase[idx] for idx in range(num_inputs)]
    node_output_names = [ascii_uppercase[num_inputs + idx] for idx in range(num_outputs)]
    onnx_node = make_node(op_type, node_input_names, node_output_names, **node_attributes)

    input_tensors = [make_tensor_value_info(name, onnx.TensorProto.FLOAT, value.shape)
                     for name, value in zip(onnx_node.input, node_inputs)]
    output_tensors = [make_tensor_value_info(name, onnx.TensorProto.FLOAT, value.shape)
                      for name, value in zip(onnx_node.output, ())]  # type: ignore

    graph = make_graph([onnx_node], 'compute_graph', input_tensors, output_tensors)
    model = make_model(graph, producer_name='NgraphBackend')
    model.opset_import[0].version = opset
    return model


def all_arrays_equal(first_list, second_list):
    # type: (Iterable[np.ndarray], Iterable[np.ndarray]) -> bool
    """
    Check that all numpy ndarrays in `first_list` are equal to all numpy ndarrays in `second_list`.

    :param first_list: iterable containing numpy ndarray objects
    :param second_list: another iterable containing numpy ndarray objects
    :return: True if all ndarrays are equal, otherwise False
    """
    return all(map(lambda pair: np.array_equal(*pair), zip(first_list, second_list)))
