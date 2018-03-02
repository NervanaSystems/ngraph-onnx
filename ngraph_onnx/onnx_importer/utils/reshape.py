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

import numpy as np

from typing import List

from pyngraph import Node as NgraphNode

import ngraph_api as ng


def transpose(node):  # type: (NgraphNode) -> NgraphNode
    """Return transposed tensor.

    :param node: Input tensor we want to transpose
    """
    axes_order = list(range(len(node.shape)))
    axes_order.reverse()
    out_shape = list(node.shape)
    out_shape.reverse()
    node = ng.reshape(node, axes_order, out_shape)
    return node


def infer_dimensions(node_name, input_shape, output_shape):
    # type: (str, List[int], List[int]) -> List[int]
    """Infer `output_shape` dimension values.

    :param node_name: The input node name.
    :param input_shape: The input data shape.
    :param output_shape: The requested output shape for the input node data.
    """
    # Check whether there are dimensions equal to -1 in output_shape. There may be at most one
    # such case. Its value is then inferred from the size of the tensor and the remaining
    # dimensions.
    if output_shape.count(-1) > 1:
        raise ng.exceptions.UserInputError('Reshape node (%s): more than one dimension is set to '
                                           '(-1). Only one dimension value can be inferred.',
                                           node_name)
    elif -1 in output_shape:
        idx = output_shape.index(-1)
        output_shape[idx] = 1
        output_shape[idx] = int(np.product(input_shape) / np.product(output_shape))

    # If an output dimension is equal to zero its actual value is copied from the input shape
    # argument.
    for idx, dim in enumerate(output_shape):
        if dim == 0:
            try:
                output_shape[idx] = input_shape[idx]
            except IndexError as e:
                raise ng.exceptions.UserInputError('Reshape node (%s): can not copy dimension '
                                                   'from the shape argument since requested index '
                                                   'is out of range.', node_name)
    return output_shape


def flatten_innermost_empty_dims(node):  # type: (NgraphNode) -> NgraphNode
    """Flatten input shape if there is at least one innermost dimension equal to one.

    node(shape: 1,2,3,1,1,1) -> node(shape: 1,2,3)
    node(shape: 1,2,3) -> node(shape: 1,2,3)
    node(shape: 1) -> node(shape: 1)

    :param node: The input node whose data we want to flatten.
    """
    shape = node.shape
    if len(shape) < 2:
        return node

    input_order = list(range(len(node.shape)))

    if shape[-1] == 1:
        output_shape = list(shape)
        while len(output_shape) > 1 and output_shape[-1] == 1:
            output_shape.pop()
        return ng.reshape(node, input_order, output_shape)
    else:
        return node
