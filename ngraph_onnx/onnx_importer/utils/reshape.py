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

import numpy as np

from typing import List

from ngraph.impl import Node as NgraphNode

import ngraph as ng


def reorder_axes(node, axes_order):  # type: (NgraphNode, List[int]) -> NgraphNode
    """Permute axes according to specified axes_order parameter.

    :param node: The node which axes we want to permute.
    :param axes_order: The permutation of node tensor axes.
    :return: New node with permuted axes.
    """
    out_shape = list(node.shape)
    if axes_order is None:
        axes_order = list(range(len(node.shape)))
    elif len(axes_order) != len(node.shape):
        raise ng.exceptions.UserInputError('Node (%s): provided axes count is different than '
                                           'input tensor rank.', node.name)
    else:
        for idx, axis in enumerate(axes_order):
            try:
                out_shape[idx] = node.shape[axis]
            except IndexError as e:
                raise ng.exceptions.UserInputError('Node (%s): provided axes indices are out '
                                                   'of range.', node.name)
    return ng.reshape(node, axes_order, out_shape)


def transpose(node):  # type: (NgraphNode) -> NgraphNode
    """Return transposed tensor (with axes in reversed order).

    :param node: Input tensor we want to transpose
    :return: New node with reversed dimensions.
    """
    axes_order = list(range(len(node.shape)))
    axes_order.reverse()
    return reorder_axes(node, axes_order)


def infer_dimensions(node_name, input_shape, output_shape):
    # type: (str, List[int], List[int]) -> List[int]
    """Infer `output_shape` dimension values.

    :param node_name: The input node name.
    :param input_shape: The input data shape.
    :param output_shape: The requested output shape for the input node data.
    """
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

    return output_shape


def flatten_innermost_empty_dims(node):  # type: (NgraphNode) -> NgraphNode
    """Flatten input shape if there is at least one innermost dimension equal to one.

    node(shape: 1,2,3,1,1,1) -> node(shape: 1,2,3)
    node(shape: 1,2,3) -> node(shape: 1,2,3)
    node(shape: 1) -> node(shape: 1)

    :param node: The input node whose data we want to flatten.
    """
    shape = list(node.shape)
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


def get_valid_array_idx(idx, last_idx):  # type: (int, int) -> int
    """Return valid array index value within range [0, last_idx].

    Negative values are interpreted such that it represent number of elements before the array end.
    If `idx` is greater than `last_idx` then it is interpreted as `last_idx` value.

    :param idx: The value of index we would like to get.
    :param last_idx: The maximum available index value.
    :return: Valid index value.
    """
    if idx >= 0:
        return min(idx, last_idx)
    else:
        return max(0, last_idx + idx)


def make_slice_op(node, axes, starts, ends):
    # type: (NgraphNode, List[int], List[int], List[int]) -> NgraphNode
    """Perform slice operation on provided node.

    :param node: The node we want to slice.
    :param axes: The list of axes on which we perform slicing.
    :param starts: The start index (inclusive) of slice for each sliced axis respectively.
    :param ends: The end index (exclusive) of slice for each sliced axis respectively.
    :return: The new node representing sliced portion of input node data.
    """
    lower_bounds = [0] * len(node.shape)
    upper_bounds = list(node.shape)

    for idx, axe in enumerate(axes):
        lower_bounds[axe] = get_valid_array_idx(starts[idx], node.shape[axe])
        upper_bounds[axe] = get_valid_array_idx(ends[idx], node.shape[axe])

    return ng.slice(node, lower_bounds, upper_bounds)
