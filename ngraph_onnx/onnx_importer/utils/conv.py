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

import logging

log = logging.getLogger(__file__)

from math import floor, ceil
from typing import Tuple, List, Dict, TYPE_CHECKING

from pyngraph import Node as NgraphNode
import ngraph_api as ng

from ngraph_onnx.onnx_importer.utils.axes import reorder_axes
from ngraph_onnx.onnx_importer.utils.decorators import function_deprecated
from ngraph_onnx.onnx_importer.utils.misc import verify_symmetric_padding
from ngraph_onnx.onnx_importer.utils.utils_pos_axes import cast_to_pos_axes

if TYPE_CHECKING:
    from ngraph_onnx.onnx_importer.model_wrappers import NodeWrapper

def get_pads(onnx_node: 'NodeWrapper') -> Tuple[int, int, int]:  # flake8: noqa
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
    kernel_shape = get_kernel_shape(onnx_node)

    if len(pads) == 0:
        pads = [0] * len(kernel_shape)

    # Attribute 'auto_pad' is deprecated, but is currently used by CNTK
    if auto_pad:
        if auto_pad == 'VALID':
            pads = [0, 0] * len(kernel_shape)

        else:
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

    verify_symmetric_padding(onnx_node, pads)

    return pads


def get_kernel_shape(onnx_node):  # type:  (NodeWrapper) -> Tuple[int, int, int]
    """
    Get shape of kernel (filter) in pixels.

    :param onnx_node: wrapped ONNX node for Conv or Pool operation
    :return: tuple of numbers representing kernel shape (height, width, depth)
    """
    kernel_shape = onnx_node.get_attribute_value('kernel_shape', ())

    if len(kernel_shape) == 0:
        kernel_shape = [1, 1]

    return kernel_shape


def get_strides(onnx_node):  # type: (NodeWrapper) -> Tuple[int, int, int]
    """
    Get number of pixels to stride operation by in each direction.

    :param onnx_node: wrapped ONNX node for Conv or Pool operation
    :return: tuple of numbers of pixels to stride by (height, width, depth)
    """
    strides = onnx_node.get_attribute_value('strides', ())  # stride along each axis
    kernel_shape = get_kernel_shape(onnx_node)

    if len(strides) == 0:
        strides = [1] * len(kernel_shape)

    return strides


def get_dilations(onnx_node):  # type: (NodeWrapper) -> Tuple[int, int, int]
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


def get_conv_params(onnx_node):  # type: (NodeWrapper) -> Dict
    """
    Parse ONNX Conv operation attributes and produce an ngraph compatible conv_params dict.

    :param onnx_node: wrapped ONNX node for Conv or ConvTranspose operation
    :return: dict of conv_params for ng.convolution
    """
    pads = get_pads(onnx_node)
    strides = get_strides(onnx_node)
    dilations = get_dilations(onnx_node)

    return {'pads': pads, 'strides': strides, 'dilations': dilations}


def make_convolution_op(onnx_node, ng_inputs, transpose=False):
    # type: (NodeWrapper, List[NgraphNode], bool) -> NgraphNode
    """
    Create an ngraph convolution Op based on an ONNX node.

    :param onnx_node: wrapped ONNX node for Conv of ConvTranspose op
    :param ng_inputs: ngraph TensorOp input tensors
    :return: ngraph Op for convolution or deconvolution
    """
    if len(ng_inputs) == 3:
        x, weights, bias = ng_inputs
    elif len(ng_inputs) == 2:
        x, weights = ng_inputs
        bias = ng.constant(0)
    else:
        raise ValueError('Conv node (%s): unexpected number of input values: %d.',
                         onnx_node.name, len(ng_inputs))

    groups = onnx_node.get_attribute_value('group', 1)
    if groups != 1:
        log.warning('Conv node (%s): `group` attribute value %d is not supported.',
                    onnx_node.name, groups)

    # Prepare ngraph convolution operation
    conv_params = get_conv_params(onnx_node)

    strides = conv_params['strides']
    dilation = conv_params['dilations']
    padding = conv_params['pads']

    if len(padding) <= 3:
        padding_above = padding
        padding_bellow = padding
    else:
        padding_above = padding[:len(padding) // 2]
        padding_bellow = padding[len(padding) // 2:]

    conv = ng.convolution(x, weights, strides, dilation, padding_above, padding_bellow)

    return conv
