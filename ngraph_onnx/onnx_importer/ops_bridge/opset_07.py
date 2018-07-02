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

import ngraph as ng

from ngraph_onnx.onnx_importer.ops_bridge.opset_06 import *  # noqa
from ngraph_onnx.onnx_importer.utils.binary import numpy_style_broadcast_for_binary_operation


# OP STATUS
# BatchNormalization-7  supported by opset 6
# Dropout-7             supported by opset 6
# GRU-7                 TODO
# Gemm-7                supported by opset 6
# LSTM-7                TODO
# Multinomial-7         TODO
# PRelu-7               supported by opset 6
# RNN-7                 TODO
# Upsample-7            TODO


def Acos(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply f(x) = acos(x) to the input tensor elementwise."""
    return ng.acos(ng_inputs[0])


def Add(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform element-wise binary addition with numpy-style broadcasting."""
    left, right = numpy_style_broadcast_for_binary_operation(onnx_node, ng_inputs)
    return ng.add(left, right)


def And(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform the `and` logical operation elementwise on two input tensors with numpy-style broadcasting."""
    left, right = numpy_style_broadcast_for_binary_operation(onnx_node, ng_inputs)
    left = ng.convert(ng.not_equal(left, 0), int)
    right = ng.convert(ng.not_equal(right, 0), int)
    return ng.convert(left * right, bool)


def Asin(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply f(x) = asin(x) to the input tensor elementwise."""
    return ng.asin(ng_inputs[0])


def Atan(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply f(x) = atan(x) to the input tensor elementwise."""
    return ng.atan(ng_inputs[0])


def Cos(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply f(x) = cos(x) to the input tensor elementwise."""
    return ng.cos(ng_inputs[0])


def Div(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform element-wise binary division with numpy-style broadcasting."""
    left, right = numpy_style_broadcast_for_binary_operation(onnx_node, ng_inputs)
    return ng.divide(left, right)


def Equal(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform the `equal` logical operation elementwise on two input tensors with numpy-style broadcasting."""
    left, right = numpy_style_broadcast_for_binary_operation(onnx_node, ng_inputs)
    return ng.equal(left, right)


def Greater(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform the `greater` logical operation elementwise on two input tensors with numpy-style broadcasting."""
    left, right = numpy_style_broadcast_for_binary_operation(onnx_node, ng_inputs)
    return ng.greater(left, right)


def Less(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform the `less` logical operation elementwise on two input tensors with numpy-style broadcasting."""
    left, right = numpy_style_broadcast_for_binary_operation(onnx_node, ng_inputs)
    return ng.less(left, right)


def Mul(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform element-wise binary multiplication with numpy-style broadcasting."""
    left, right = numpy_style_broadcast_for_binary_operation(onnx_node, ng_inputs)
    return ng.multiply(left, right)


def Or(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform the `or` logical operation elementwise on two input tensors with numpy-style broadcasting."""
    left, right = numpy_style_broadcast_for_binary_operation(onnx_node, ng_inputs)
    left = ng.convert(ng.not_equal(left, 0), int)
    right = ng.convert(ng.not_equal(right, 0), int)
    return (left + right) > 0


def Pow(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform element-wise binary power with numpy-style broadcasting."""
    base, exponent = numpy_style_broadcast_for_binary_operation(onnx_node, ng_inputs)
    return ng.power(base, exponent)


def Sin(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply f(x) = sin(x) to the input tensor elementwise."""
    return ng.sin(ng_inputs[0])


def Sub(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform element-wise binary subtraction with numpy-style broadcasting."""
    left, right = numpy_style_broadcast_for_binary_operation(onnx_node, ng_inputs)
    return ng.subtract(left, right)


def Tan(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply f(x) = tan(x) to the input tensor elementwise."""
    return ng.tan(ng_inputs[0])


def Xor(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform the `xor` logical operation elementwise on two input tensors with numpy-style broadcasting."""
    left, right = numpy_style_broadcast_for_binary_operation(onnx_node, ng_inputs)
    not_left = ng.convert(ng.equal(left, 0), int)
    left = ng.convert(ng.not_equal(left, 0), int)
    right = ng.convert(ng.not_equal(right, 0), int)
    not_right = ng.convert(ng.equal(right, 0), int)

    return ((not_left * right) + (not_right * left)) > 0
