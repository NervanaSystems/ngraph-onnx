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

from math import floor, ceil
from copy import copy
from typing import Tuple, List

from ngraph_onnx import TYPE_CHECKING

from ngraph.impl import Node as NgraphNode

import ngraph as ng

from ngraph.utils.types import get_dtype

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ngraph_onnx.onnx_importer.model_wrappers import NodeWrapper


def get_pads(onnx_node, kernel_shape=None):
    # type: (NodeWrapper, List[int]) -> Tuple[List[int], List[int]]
    """
    Get padding values for the operation described by an ONNX node.

    If `auto_pad` attribute is specified as SAME_UPPER or SAME_LOWER, or VALID values are
    calculated. Otherwise values are taken from the `pads` attribute.

    `pads` value should follow [x1_begin, x2_begin..., x1_end, x2_end,...]

    :param onnx_node: wrapped ONNX node for Conv or Pool operation
    :return: tuple of numbers of pixels to pad (height, width, depth)
    """
    auto_pad = onnx_node.get_attribute_value('auto_pad')
    pads = onnx_node.get_attribute_value('pads', ())  # Padding along each axis
    if(kernel_shape is None):
        kernel_shape = get_kernel_shape(onnx_node)

    if len(pads) == 0:
        pads = [0] * len(kernel_shape)

    # Attribute 'auto_pad' is deprecated, but is currently used by CNTK.
    if auto_pad == 'VALID':
        pads = [0, 0] * len(kernel_shape)

    elif auto_pad == 'SAME_UPPER' or auto_pad == 'SAME_LOWER':
        # SAME_UPPER or SAME_LOWER mean pad the input so that the output size match the input.
        # In case of odd number add the extra padding at the end for SAME_UPPER and at the
        # beginning for SAME_LOWER.
        def pad_value(kernel_dim):  # type: (int) -> float
            return (kernel_dim - 1.0) / 2.0

        pads_starts = [floor(pad_value(dim)) if auto_pad == 'SAME_UPPER' else
                       ceil(pad_value(dim)) for dim in kernel_shape]
        pads_ends = [ceil(pad_value(dim)) if auto_pad == 'SAME_UPPER' else
                     floor(pad_value(dim)) for dim in kernel_shape]
        pads = pads_starts + pads_ends

    if len(pads) <= 3:
        padding_above = pads
        padding_below = pads
    else:
        padding_above = pads[:len(pads) // 2]
        padding_below = pads[len(pads) // 2:]

    return padding_above, padding_below


def get_kernel_shape(onnx_node):  # type: (NodeWrapper) -> List[int]
    """
    Get shape of kernel (filter) in pixels.

    :param onnx_node: wrapped ONNX node for Conv or Pool operation
    :return: tuple of numbers representing kernel shape (height, width, depth)
    """
    kernel_shape = onnx_node.get_attribute_value('kernel_shape', ())

    if len(kernel_shape) == 0:
        kernel_shape = [1, 1]

    return kernel_shape


def get_strides(onnx_node, kernel_shape=None):  # type: (NodeWrapper, List[int]) -> List[int]
    """
    Get number of pixels to stride operation by in each direction.

    :param onnx_node: wrapped ONNX node for Conv or Pool operation
    :return: tuple of numbers of pixels to stride by (height, width, depth)
    """
    strides = onnx_node.get_attribute_value('strides', ())  # stride along each axis
    if kernel_shape is None:
        kernel_shape = get_kernel_shape(onnx_node)

    if len(strides) == 0:
        strides = [1] * len(kernel_shape)

    return strides


def get_dilations(onnx_node):  # type: (NodeWrapper) -> List[int]
    """
    Get number of pixels for filter dilation in each direction.

    :param onnx_node: wrapped ONNX node for Conv or Pool operation
    :return: tuple of numbers of pixels for filter dilation (height, width, depth)
    """
    dilations = onnx_node.get_attribute_value('dilations', ())  # dilation along each axis
    kernel_shape = get_kernel_shape(onnx_node)

    if len(dilations) == 0:
        dilations = [1] * len(kernel_shape)

    return dilations


def make_convolution_op(onnx_node, ng_inputs):
    # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """
    Create an ngraph convolution Op based on an ONNX node.

    :param onnx_node: wrapped ONNX node for Conv of ConvTranspose op
    :param ng_inputs: ngraph TensorOp input tensors
    :return: ngraph Op for convolution or deconvolution
    """
    if len(ng_inputs) == 3:
        data, weights, bias = ng_inputs
    elif len(ng_inputs) == 2:
        data, weights = ng_inputs
        bias = ng.constant(0, dtype=get_dtype(data.get_element_type()))
    else:
        raise ValueError('Conv node (%s): unexpected number of input values: %d.',
                         onnx_node.name, len(ng_inputs))

    groups = onnx_node.get_attribute_value('group', 1)

    strides = get_strides(onnx_node)
    dilation = get_dilations(onnx_node)
    padding_below, padding_above = get_pads(onnx_node)
    if groups != 1:
        # Split one convolution op to N ops where N is the number of groups and concat results after computation.
        # reference: https://github.com/NervanaSystems/ngraph-mxnet/blob/fdd692/src/ngraph/ngraph_emitter.cc#L822-L856
        data_shape = list(data.shape)
        weights_shape = list(weights.shape)
        convolutions_nodes = []

        # initial bounds for splice
        data_lower_part = len(data_shape) * [0]
        data_upper_part = copy(data_shape)

        weights_lower_part = len(weights_shape) * [0]
        weights_upper_part = copy(weights_shape)

        for group in range(groups):
            # update bounds for splice
            data_lower_part[1] = group * int((data_shape[1] / groups))
            data_upper_part[1] = (group + 1) * int((data_shape[1] / groups))

            sliced_data = ng.slice(data, data_lower_part, data_upper_part)

            # update bounds for splice
            weights_lower_part[0] = group * int((weights_shape[0] / groups))
            weights_upper_part[0] = max((group + 1) * int((weights_shape[0] / groups)), 1)

            sliced_weights = ng.slice(weights, weights_lower_part, weights_upper_part)
            convolutions_nodes.append(ng.convolution(sliced_data, sliced_weights, strides,
                                                     dilation, padding_below, padding_above))
        conv = ng.concat(convolutions_nodes, axis=1)
    else:
        conv = ng.convolution(data, weights, strides, dilation, padding_below, padding_above)
    if len(bias.shape) > 0:
        return conv + ng.broadcast_to(bias, conv.shape, 1)
    else:
        return conv
