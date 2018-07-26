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

from typing import List

from ngraph_onnx import TYPE_CHECKING

from ngraph.impl import Node as NgraphNode
import ngraph as ng
import logging

from ngraph_onnx.onnx_importer.utils.conv import get_pads, get_strides, get_kernel_shape

if TYPE_CHECKING:
    from ngraph_onnx.onnx_importer.model_wrappers import NodeWrapper

log = logging.getLogger(__name__)


def get_op_type(onnx_node):  # type: (NodeWrapper) -> str
    """
    Parse ONNX pooling operation attributes and produce an ngraph compatible pool_params dict.

    :param onnx_node: wrapped ONNX node for a pooling operation op
    :return: type of poling op
    """
    if onnx_node.op_type in ['AveragePool', 'GlobalAveragePool']:
        pooling_op = 'avg'
    elif onnx_node.op_type in ['MaxPool', 'GlobalMaxPool']:
        pooling_op = 'max'
    else:
        raise NotImplementedError('%s node (%s): Unsupported pooling type.',
                                  onnx_node.op_type, onnx_node.name)

    return pooling_op


def reduce_extra_dims(spatial_dims_count, param_shape, onnx_node):
    # type: (int, List[int], NodeWrapper) -> List[int]
    """Remove extra dimensions from input parameter shape.

    :param spatial_dims_count: The number of current node spatial dimensions.
    :param param_shape: Parameter shape.
    :param onnx_node: The currently processed node.
    """
    # We assume data are in [D1,...,DN] format
    # (https://github.com/onnx/onnx/blob/master/docs/Operators.md#attributes-4),
    # In case when there is more dimensions than we expected actually we don't know which part of
    # them is correct. Thus we assume here the correct part is the one with innermost dimensions and
    # we inform the user about this situation.
    if len(param_shape) > spatial_dims_count:
        log.warning('{} node ({}) Parameter shape size is bigger than spatial dimensions count. '
                    'Reducing outermost dimensions!'.format(onnx_node.op_type, onnx_node.name))
        param_shape = param_shape[-spatial_dims_count:]
    return param_shape


def make_pooling_op(onnx_node, ng_inputs, kernel_shape=None):
    # type: (NodeWrapper, List[NgraphNode], List[int]) -> NgraphNode
    """
    Create an ngraph pooling Op based on an ONNX node.

    :param onnx_node: wrapped ONNX node for a pooling op
    :param ng_inputs: ngraph TensorOp input tensors
    :param kernel_shape: kernel shape for this op
    :return: ngraph pooling op
    """
    x = ng_inputs[0]

    op_type = get_op_type(onnx_node)

    # We assume data are in [D1,...,DN] format thus we subtract [N,C] dimensions.
    spatial_dims = len(x.shape) - 2  # get spatial dimensions

    if kernel_shape is None:
        kernel_shape = get_kernel_shape(onnx_node)
    kernel_shape = reduce_extra_dims(spatial_dims, kernel_shape, onnx_node)

    strides = get_strides(onnx_node, kernel_shape)
    padding_below, padding_above = get_pads(onnx_node, kernel_shape)

    strides = reduce_extra_dims(spatial_dims, strides, onnx_node)
    padding_above = reduce_extra_dims(spatial_dims, padding_above, onnx_node)
    padding_below = reduce_extra_dims(spatial_dims, padding_below, onnx_node)

    include_pad = onnx_node.get_attribute_value('count_include_pad', 0) != 0

    if op_type == 'avg':
        ng_op = ng.avg_pool(x, kernel_shape, strides, padding_below, padding_above,
                            include_padding=include_pad)
    elif op_type == 'max':
        ng_op = ng.max_pool(x, kernel_shape, strides, padding_below, padding_above)
    else:
        raise NotImplementedError('%s node (%s): Unsupported pooling type.',
                                  onnx_node.op_type, onnx_node.name)
    return ng_op


def make_global_pooling_op(onnx_node, ng_inputs):
    # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """
    Create a ngraph global pooling operation.

    This is equivalent to pooling with kernel size equal to the spatial dimension of input tensor.

    :param onnx_node: wrapped ONNX node for a pooling op
    :param ng_inputs: ngraph TensorOp input tensors
    :return: ngraph pooling op
    """
    kernel_shape = list(ng_inputs[0].shape)

    return make_pooling_op(onnx_node, ng_inputs, kernel_shape)
