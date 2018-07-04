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

import ngraph as ng

from typing import List, Optional, Tuple

from ngraph.exceptions import UserInputError
from ngraph.utils.types import TensorShape

from ngraph_onnx import TYPE_CHECKING

from ngraph.impl import Node as NgraphNode

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

    dimensions_identical = list(left.shape) == list(right.shape)
    if dimensions_identical:
        return left, right

    broadcast = onnx_node.get_attribute_value('broadcast', 0)
    if not broadcast:
        logger.warning('%s node (%s): operands have different dimensions, and "broadcast"'
                       ' attribute is not set. ', onnx_node.op_type, onnx_node.name)
        return left, right

    start_axis = onnx_node.get_attribute_value('axis')  # start of mutually equal shape
    right = ng.broadcast_to(right, left.shape, start_axis)
    return left, right


def numpy_style_broadcast_for_binary_operation(onnx_node, ng_inputs):
    # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """
    Cast shape of two nodes to make them compatible for an element-wise binary operation.

    :param onnx_node: a wrapped ONNX node
    :param ng_inputs: left and right node (inputs of the binary op)
    :return: left and right node after broadcasting
    """
    left = ng_inputs[0]
    right = ng_inputs[1]

    dimensions_identical = list(left.shape) == list(right.shape)
    if dimensions_identical:
        return left, right

    try:
        output_shape, left_full_shape, right_full_shape = numpy_style_broadcast_output_shape(left.shape, right.shape)
    except UserInputError:
        raise UserInputError('%s node (%s): Unable to broadcast shapes %s and %s.',
                             onnx_node.op_type, onnx_node.name, left.shape, right.shape)

    if list(right.shape) != output_shape:
        one_pos = [i for i, dim in enumerate(right_full_shape) if dim == 1]
        right = ng.reshape(right, [dim for dim in right.shape if dim != 1])  # Squeeze
        right = ng.broadcast(right, output_shape, broadcast_axes=one_pos)

    if list(left.shape) != output_shape:
        one_pos = [i for i, dim in enumerate(left_full_shape) if dim == 1]
        left = ng.reshape(left, [dim for dim in left.shape if dim != 1])
        left = ng.broadcast(left, output_shape, broadcast_axes=one_pos)

    return left, right


def numpy_style_broadcast_output_shape(shape_a, shape_b):
    # type: (TensorShape, TensorShape) -> Tuple[TensorShape, TensorShape, TensorShape]
    """Calculate output shape of numpy-style broadcast operation.

    :param shape_a: shape of first input tensor
    :param shape_b: shape of the second input tensor
    :return: shape of the output tensor, full shape of input tensors
    """
    output_shape = []  # type: List[int]

    shape_a = list(shape_a)
    shape_b = list(shape_b)
    rank_a = len(shape_a)
    rank_b = len(shape_b)
    max_rank = max(rank_a, rank_b)

    # left-pad A's shape with 1s until it also has p dimensions
    if rank_a < max_rank:
        for idx in range(max_rank - rank_a):
            shape_a.insert(0, 1)

    # left-pad B's shape with 1s until is also has p dimensions
    elif rank_b < max_rank:
        for idx in range(max_rank - rank_b):
            shape_b.insert(0, 1)

    for idx in range(max_rank - 1, -1, -1):
        idx_dim_a = shape_a[idx]
        idx_dim_b = shape_b[idx]
        if idx_dim_a != 1 and idx_dim_b != 1 and idx_dim_a != idx_dim_b:
            raise UserInputError('Shapes %s and %s are incompatible for broadcasting.', shape_a, shape_b)
        output_shape.insert(0, max(idx_dim_a, idx_dim_b))

    return output_shape, shape_a, shape_b
