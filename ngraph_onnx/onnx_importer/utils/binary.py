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

from __future__ import division
from __future__ import print_function

import logging

import ngraph_api as ng

from typing import List

from ngraph_onnx import TYPE_CHECKING

from ngraph_onnx.onnx_importer.utils.decorators import function_deprecated
from pyngraph import Node as NgraphNode

if TYPE_CHECKING:
    from ngraph_onnx.onnx_importer.model_wrappers import NodeWrapper

logger = logging.getLogger(__name__)


def broadcast_for_binary_operation(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """
    Cast shape of the right operand to make ops compatible for an element-wise binary operation.

    Casting is based on `broadcast` and `axis` attributes of an ONNX node.

    :param onnx_node: wrapped ONNX node
    :param ng_inputs: left and right operand
    :return: left and right operand after broadcasting
    """
    left = ng_inputs[0]
    right = ng_inputs[1]

    dimensions_identical = left.shape == right.shape
    if dimensions_identical:
        return left, right

    broadcast = onnx_node.get_attribute_value('broadcast', 0)
    if not broadcast:
        logger.warning('%s node (%s): operands have different dimensions, and "broadcast"'
                       ' attribute is not set. ', onnx_node.op_type, onnx_node.name)
        return left, right

    start_axis = onnx_node.get_attribute_value('axis')  # start of mutually equal shape
    right = ng.broadcast(right, left.shape, start_axis)
    return left, right


@function_deprecated
def cast_axes_for_matmul(ng_input_left, ng_input_right):  # type: ignore
    """
    Prepare two ngraph tensors for matrix multiplication by casting axes.

    Matching axes will be cast to enable matrix @ matrix or vector @ matrix dot multiply.

    :param ng_input_left: first input to matrix multiplication
    :param ng_input_right: second input to matrix multiplication
    :return: tuple with the first and second input tensor with axes cast for matrix multiplication
    """
    left, right = ng_input_left, ng_input_right
    left_num_axes = len(left.axes)
    right_num_axes = len(right.axes)

    if left_num_axes == right_num_axes == 1:
        # vector @ vector
        # cast to axes: i, icast_axes_for_matmul
        assert left.shape.lengths == right.shape.lengths, \
            'Vector lengths must be equal for multiplication.'
        if left.shape != right.shape:
            right = ng.cast_axes(right, axes=left.axes)

    elif left_num_axes == 1:
        # vector @ matrix
        # cast to axes: i, ...ij
        if left.axes[0] != right.axes[-2]:
            left = ng.cast_axes(left, axes=right.axes[-2])

    elif right_num_axes == 1:
        # matrix @ vector
        # cast to axes: ...i, i
        if left.axes[-1] != right.axes[0]:
            right = ng.cast_axes(right, axes=left.axes[-1])

    else:
        # matrix @ matrix
        # cast to axes: ...ij, ...jk
        right_axes = [ng.make_axis(name='DOT_{}'.format(i), length=axis.length)
                      for i, axis in enumerate(right.shape)]
        right_axes[-2] = left.axes[-1]
        right = ng.cast_axes(right, axes=right_axes)

    return left, right
