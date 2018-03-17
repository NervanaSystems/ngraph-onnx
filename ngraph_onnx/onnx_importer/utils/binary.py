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

from typing import List

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
