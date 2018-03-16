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

from typing import Dict, List

from ngraph_onnx import TYPE_CHECKING

from pyngraph import Node as NgraphNode
import ngraph_api as ng
import logging

from ngraph_onnx.onnx_importer.utils.conv import get_pads, get_strides, get_kernel_shape
from ngraph_onnx.onnx_importer.utils.decorators import function_deprecated

if TYPE_CHECKING:
    from ngraph_onnx.onnx_importer.model_wrappers import NodeWrapper

log = logging.getLogger(__file__)


def get_op_type(onnx_node):  # type: (NodeWrapper) -> str
    """
    Parse ONNX pooling operation attributes and produce an ngraph compatible pool_params dict.

    :param onnx_node: wrapped ONNX node for a pooling operation op
    :return: dict of pool_params for ng.pooling
    """
    if onnx_node.op_type in ['AveragePool', 'GlobalAveragePool']:
        pooling_op = 'avg'
    elif onnx_node.op_type in ['MaxPool', 'GlobalMaxPool']:
        pooling_op = 'max'
    else:
        raise NotImplementedError('%s node (%s): Unsupported pooling type.',
                                  onnx_node.op_type, onnx_node.name)

    return pooling_op


@function_deprecated
def make_pool_output_axes(input_tensor, pool_params):  # type: ignore
    """
    Prepare axes for the output of an ng.convolution operation.

    :param input_tensor: ngraph tensor with pooling input data
    :param pool_params: dict of pool_params for ng.pooling
    :return: ngraph Axes compatible with pooling operation
    """
    number_output_channels = input_tensor.axes[0].length
    mini_batch_size = input_tensor.axes[-1].length

    input_d, input_h, input_w = input_tensor.axes.lengths[1:4]  # axes order C, D, H, W, N

    params = pool_params
    output_d = int((input_d + 2 * params['pad_d'] - params['T']) / params['str_d']) + 1
    output_h = int((input_h + 2 * params['pad_h'] - params['R']) / params['str_h']) + 1
    output_w = int((input_w + 2 * params['pad_w'] - params['S']) / params['str_w']) + 1

    output_axes = ng.make_axes(axes=(
        ng.make_axis(name='C', docstring='channels', length=int(number_output_channels)),
        ng.make_axis(name='D', docstring='depth', length=int(output_d)),
        ng.make_axis(name='H', docstring='height', length=int(output_h)),
        ng.make_axis(name='W', docstring='width', length=int(output_w)),
        ng.make_axis(name='N', docstring='mini-batch size', length=int(mini_batch_size)),
    ))
    return output_axes


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


def make_pooling_op(onnx_node, ng_inputs, custom_pool_params=None):
    # type: (NodeWrapper, List[NgraphNode], Dict) -> NgraphNode
    """
    Create an ngraph pooling Op based on an ONNX node.

    :param onnx_node: wrapped ONNX node for a pooling op
    :param ng_inputs: ngraph TensorOp input tensors
    :param custom_pool_params: optional pool_params overriding values based on onnx_node
    :return: ngraph pooling op
    """
    x = ng_inputs[0]

    strides = get_strides(onnx_node)
    padding_above, padding_below = get_pads(onnx_node)
    kernel_shape = get_kernel_shape(onnx_node)
    op_type = get_op_type(onnx_node)

    # We assume data are in [D1,...,DN] format thus we subtract [N,C] dimensions.
    spatial_dims = len(x.shape) - 2  # get spatial dimensions

    strides = reduce_extra_dims(spatial_dims, strides, onnx_node)
    padding_above = reduce_extra_dims(spatial_dims, padding_above, onnx_node)
    padding_below = reduce_extra_dims(spatial_dims, padding_below, onnx_node)
    kernel_shape = reduce_extra_dims(spatial_dims, kernel_shape, onnx_node)

    if op_type == 'avg':
        ng_op = ng.avg_pool(x, kernel_shape, strides, padding_above, padding_below, False)
    elif op_type == 'max':
        ng_op = ng.max_pool(x, kernel_shape, strides, padding_above, padding_below)
    else:
        raise NotImplementedError('%s node (%s): Unsupported pooling type.',
                                  onnx_node.op_type, onnx_node.name)
    return ng_op


@function_deprecated
def make_global_pooling_op(onnx_node, ng_inputs):
    # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """
    Create a ngraph global pooling operation.

    This is equivalent to pooling with kernel size equal to the spatial dimension of input tensor.

    :param onnx_node: wrapped ONNX node for a pooling op
    :param ng_inputs: ngraph TensorOp input tensors
    :return: ngraph pooling op
    """
    x = ng_inputs[0]

    if len(x.axes) == 4:  # ONNX input axes order NCHW
        _, _, kernel_h, kernel_w = x.axes.lengths
        pool_params = {'R': kernel_h, 'S': kernel_w}
    elif len(x.axes) == 5:  # ONNX input axes order NCHWD
        _, _, kernel_h, kernel_w, kernel_d = x.axes.lengths
        pool_params = {'R': kernel_h, 'S': kernel_w, 'T': kernel_d}
    else:
        raise NotImplementedError('%s node (%s): only 2D and 3D pooling ops are supported.',
                                  onnx_node.op_type, onnx_node.name)

    return make_pooling_op(onnx_node, ng_inputs, custom_pool_params=pool_params)
