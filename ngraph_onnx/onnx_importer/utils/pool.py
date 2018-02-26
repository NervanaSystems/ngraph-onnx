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

from typing import Tuple, Dict, List, TYPE_CHECKING

from pyngraph import Node as NgraphNode
import ngraph_api as ng

from ngraph_onnx.onnx_importer.utils.axes import reorder_axes
from ngraph_onnx.onnx_importer.utils.conv import get_pads, get_strides
from ngraph_onnx.onnx_importer.utils.decorators import function_deprecated

if TYPE_CHECKING:
    from ngraph_onnx.onnx_importer.model_wrappers import NodeWrapper


def get_kernel_shape(onnx_node):  # type:  (NodeWrapper) -> Tuple[int, int, int]
    """
    Get shape of kernel (filter) in pixels.

    :param onnx_node: wrapped ONNX node for Conv or Pool operation
    :return: tuple of numbers representing kernel shape (height, width, depth)
    """
    ker_h, ker_w, ker_d = 1, 1, 1  # default values
    kernel_shape = onnx_node.get_attribute_value('kernel_shape', ())

    if len(kernel_shape) == 2:  # ONNX input axes order NCHW
        ker_h, ker_w = kernel_shape
    elif len(kernel_shape) == 3:  # ONNX input axes order NCHWD
        ker_h, ker_w, ker_d = kernel_shape

    return ker_h, ker_w, ker_d


def get_pool_params(onnx_node):  # type: (NodeWrapper) -> Dict
    """
    Parse ONNX pooling operation attributes and produce an ngraph compatible pool_params dict.

    :param onnx_node: wrapped ONNX node for a pooling operation op
    :return: dict of pool_params for ng.pooling
    """
    pad_h, pad_w, pad_d = get_pads(onnx_node)
    str_h, str_w, str_d = get_strides(onnx_node)
    ker_h, ker_w, ker_d = get_kernel_shape(onnx_node)
    pad_c, str_c, ker_c = 0, 1, 1  # default values for channel

    if onnx_node.op_type in ['AveragePool', 'GlobalAveragePool']:
        pooling_op = 'avg'
    elif onnx_node.op_type in ['MaxPool', 'GlobalMaxPool']:
        pooling_op = 'max'
    else:
        raise NotImplementedError('%s node (%s): Unsupported pooling type.',
                                  onnx_node.op_type, onnx_node.name)

    return {'pad_d': pad_d, 'pad_h': pad_h, 'pad_w': pad_w, 'pad_c': pad_c,
            'str_d': str_d, 'str_h': str_h, 'str_w': str_w, 'str_c': str_c,
            'J': ker_c, 'R': ker_h, 'S': ker_w, 'T': ker_d, 'op': pooling_op}


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

    pool_params = get_pool_params(onnx_node)
    if custom_pool_params:
        pool_params.update(custom_pool_params)

    strides = [pool_params['str_h'], pool_params['str_w']]
    padding_above = [pool_params['pad_h'], pool_params['pad_w']]
    kernel_shape = get_kernel_shape(onnx_node)[0:2]
    type = pool_params['op']

    if type == 'avg':
        ng_op = ng.avg_pool(x, kernel_shape, strides, padding_above, padding_above, False)
    elif type == 'max':
        ng_op = ng.max_pool(x, kernel_shape, strides, padding_above, padding_above)

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
