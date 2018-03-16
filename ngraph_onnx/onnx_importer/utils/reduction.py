# ----------------------------------------------------------------------------
# Copyright 2018 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import division
from __future__ import print_function

import ngraph_api as ng

from typing import Callable, Iterable, List, Optional
from ngraph_onnx import TYPE_CHECKING
from pyngraph import Node as NgraphNode
if TYPE_CHECKING:
    from ngraph_onnx.onnx_importer.model_wrappers import NodeWrapper


def get_reduction_axes(onnx_node, ng_input):  # type: (NodeWrapper, NgraphNode) -> List[int]
    """Create a list of axes to be used in a reduction operation."""
    reduction_axes = onnx_node.get_attribute_value('axes')
    if reduction_axes is None:
        reduction_axes = list(range(len(ng_input.shape)))

    return reduction_axes


def make_reduction_op(ng_op_type, onnx_node, ng_input):
    # type: (Callable, NodeWrapper, NgraphNode) -> NgraphNode
    """
    Create an ngraph Op node for a reduction operation (min, max, sum, etc.).

    :param ng_op_type: an ngraph reduction factory function such as ng.max, etc.
    :param onnx_node: wrapped ONNX node
    :param ng_input: ngraph Op to be used as input to the reduction node
    """
    reduction_axes = get_reduction_axes(onnx_node, ng_input)
    op_node = ng_op_type(ng_input, reduction_axes)

    if onnx_node.get_attribute_value('keepdims', default=1):
        output_shape = list(ng_input.shape)
        # flatten reduced axes
        for idx in reduction_axes:
            output_shape[idx] = 1
        op_node = ng.reshape(op_node, list(range(len(op_node.shape))), output_shape)

    return op_node
