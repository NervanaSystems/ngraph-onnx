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

from __future__ import print_function
from __future__ import division

import logging
from string import ascii_letters
from typing import Tuple, List, TYPE_CHECKING

from functools import reduce

from pyngraph import Node as NgraphNode
import ngraph_api as ng

from ngraph_onnx.onnx_importer.utils.axes import reorder_axes, reshape_workaround, \
    rename_axes
from ngraph_onnx.onnx_importer.utils.decorators import refactoring_required
from ngraph_onnx.onnx_importer.utils.misc import split_pads_into_pairs
from ngraph_onnx.onnx_importer.utils.pool import make_pooling_op, make_global_pooling_op
from ngraph_onnx.onnx_importer.utils.reduction import make_reduction_op
from ngraph_onnx.onnx_importer.utils.binary import cast_axes_for_binary_broadcast, \
    cast_axes_for_matmul
from ngraph_onnx.onnx_importer.utils.conv import make_convolution_op
from ngraph_onnx.onnx_importer.utils.utils_pos_axes import cast_to_pos_axes

if TYPE_CHECKING:
    from ngraph_onnx.onnx_importer.model_wrappers import NodeWrapper

logger = logging.getLogger(__name__)


def make_ng_nodes(onnx_node):  # type: (NodeWrapper) -> Tuple[NgraphNode]
    """Create ngraph output Ops for an ONNX node."""
    op_type = onnx_node.op_type

    try:
        ng_node_factory = globals()[op_type]
    except KeyError:
        raise NotImplementedError('Unknown operation: %s', op_type)

    ng_inputs = onnx_node.get_ng_inputs()
    ng_outputs = ng_node_factory(onnx_node, ng_inputs)

    if type(ng_outputs) != tuple:
        ng_outputs = (ng_outputs,)

    return ng_outputs


# Unary Ops
def Abs(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply f(x) = abs(x) to the input tensor elementwise."""
    return ng.absolute(ng_inputs[0])


def Ceil(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply f(x) = ceil(x) to the input tensor elementwise."""
    return ng.ceiling(ng_inputs[0])


def Exp(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate the exponential of the input tensor elementwise."""
    return ng.exp(ng_inputs[0])


def Floor(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply f(x) = floor(x) to the input tensor elementwise."""
    return ng.floor(ng_inputs[0])


def Log(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate the natural log of the input tensor elementwise."""
    return ng.log(ng_inputs[0])


def Neg(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply f(x) = -x to the input tensor elementwise (each element has flipped sign)."""
    return ng.negative(ng_inputs[0])


@refactoring_required
def Reciprocal(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply f(x) = 1/x to the input tensor elementwise."""
    return ng.reciprocal(ng_inputs[0])


def Sqrt(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply f(x) = x^0.5 (square root) to the input tensor elementwise."""
    return ng.sqrt(ng_inputs[0])


@refactoring_required
def Sigmoid(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply the sigmoid function, f(x) = 1 / (1 + exp(-x)) to the input tensor elementwise."""
    return ng.sigmoid(ng_inputs[0])


@refactoring_required
def Tanh(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate the hyperbolic tangent of the input tensor elementwise."""
    return ng.tanh(ng_inputs[0])


@refactoring_required
def Relu(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply the Relu function, f(x) = max(0, x) to the input tensor elementwise."""
    return ng.maximum(ng_inputs[0], 0.)


@refactoring_required
def LeakyRelu(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply the Leaky Relu function to the input tensor elementwise.

    f(x) = alpha * x for x < 0, f(x) = x for x >= 0
    """
    alpha = onnx_node.get_attribute_value('alpha', 0.01)
    if not 0 <= alpha <= 1:
        logger.warning('LeakyRelu node (%s): alpha value should be in range (0,1), but is: %s',
                       onnx_node.name, alpha)
    return ng.maximum(alpha * ng_inputs[0], ng_inputs[0])


@refactoring_required
def PRelu(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply the Parametric Relu function to the input tensor elementwise.

    f(x) = slope * x for x < 0, f(x) = x for x >= 0
    The slope parameter is passed to the node as its second input.
    """
    x, slope = ng_inputs
    x = ng.broadcast(x, x.axes + slope.axes)
    slope = ng.broadcast(slope, axes=x.axes)
    return ng.maximum(slope * x, x)


@refactoring_required
def Selu(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply the scaled exponential linear unit function to the input tensor elementwise.

    f(x) = gamma * (alpha * exp(x) - alpha) for x <= 0, f(x) = gamma * x for x > 0
    """
    x = ng_inputs[0]
    alpha = onnx_node.get_attribute_value('alpha', 1.6732)
    gamma = onnx_node.get_attribute_value('gamma', 1.0507)

    return gamma * (ng.maximum(x, 0) + alpha * (ng.exp(-ng.maximum(-x, 0)) - 1))


@refactoring_required
def Elu(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply the exponential linear unit function to the input tensor elementwise.

    f(x) = alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0
    """
    x = ng_inputs[0]
    alpha = onnx_node.get_attribute_value('alpha', 1)

    if not alpha < 0:
        logger.warning('Elu node (%s): alpha value should be positive, but is: %s',
                       onnx_node.name, alpha)

    return ng.maximum(x, 0) + alpha * (ng.exp(-ng.maximum(-x, 0)) - 1)


@refactoring_required
def Softplus(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply Softplus function, f(x) = ln(exp(x) + 1) to the input tensor elementwise."""
    return ng.log((ng.exp(ng_inputs[0]) + 1))


# Reduction Ops
@refactoring_required
def ReduceSum(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the sum of the input tensor's elements along the provided axes."""
    return make_reduction_op(ng.sum, onnx_node, ng_inputs[0])


@refactoring_required
def ReduceMax(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the maximum value of the input tensor's elements along the provided axes."""
    return make_reduction_op(ng.max, onnx_node, ng_inputs[0])


@refactoring_required
def ReduceMin(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the minimum value of the input tensor's elements along the provided axes."""
    return make_reduction_op(ng.min, onnx_node, ng_inputs[0])


@refactoring_required
def ReduceLogSumExp(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the log sum exponent of the input tensor's element' along the provided axes."""
    op = ng.exp(ng_inputs[0])
    op = make_reduction_op(ng.sum, onnx_node, op)
    op = ng.log(op)
    return op


@refactoring_required
def ReduceMean(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the mean value of the input tensor's elements along the provided axes."""
    return make_reduction_op(ng.mean, onnx_node, ng_inputs[0])


@refactoring_required
def ReduceProd(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the product of the input tensor's elements along the provided axes."""
    return make_reduction_op(ng.prod, onnx_node, ng_inputs[0])


@refactoring_required
def ArgMin(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the indices of the min elements of the input tensor along the provided axes."""
    return make_reduction_op(ng.argmin, onnx_node, ng_inputs[0])


@refactoring_required
def ArgMax(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the indices of the max elements of the input tensor along the provided axes."""
    return make_reduction_op(ng.argmax, onnx_node, ng_inputs[0])


# Binary Ops
def Add(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform element-wise binary addition."""
    left, right = ng_inputs
    return left + right


@refactoring_required
def Sub(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform element-wise binary subtraction."""
    left, right = cast_axes_for_binary_broadcast(onnx_node, ng_inputs)
    return ng.subtract(left, right)


@refactoring_required
def Mul(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform element-wise binary multiplication."""
    left, right = cast_axes_for_binary_broadcast(onnx_node, ng_inputs)
    return ng.multiply(left, right)


def Div(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform element-wise binary division."""
    left, right = cast_axes_for_binary_broadcast(onnx_node, ng_inputs)
    return ng.divide(left, right)


# Logical ops
@refactoring_required
def Equal(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform the `equal` logical operation elementwise on two input tensors."""
    left, right = cast_axes_for_binary_broadcast(onnx_node, ng_inputs)
    return ng.equal(left, right)


@refactoring_required
def Less(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform the `less` logical operation elementwise on two input tensors."""
    left, right = cast_axes_for_binary_broadcast(onnx_node, ng_inputs)
    return ng.less(left, right)


@refactoring_required
def Greater(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform the `greater` logical operation elementwise on two input tensors."""
    left, right = cast_axes_for_binary_broadcast(onnx_node, ng_inputs)
    return ng.greater(left, right)


@refactoring_required
def And(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform the `and` logical operation elementwise on two input tensors."""
    left, right = cast_axes_for_binary_broadcast(onnx_node, ng_inputs)
    left = ng.not_equal(left, 0)
    right = ng.not_equal(right, 0)
    return left * right


@refactoring_required
def Or(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform the `or` logical operation elementwise on two input tensors."""
    left, right = cast_axes_for_binary_broadcast(onnx_node, ng_inputs)
    left = ng.not_equal(left, 0)
    right = ng.not_equal(right, 0)
    return (left + right) > 0


@refactoring_required
def Xor(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform the `xor` logical operation elementwise on two input tensors."""
    left, right = cast_axes_for_binary_broadcast(onnx_node, ng_inputs)
    left = ng.not_equal(left, 0)
    right = ng.not_equal(right, 0)
    return (left + right) % 2


@refactoring_required
def Not(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Return the negation of the input tensor elementwise."""
    return ng.equal(ng_inputs[0] + 1, 1)


# Variadic Ops
@refactoring_required
def Sum(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate element-wise sum of the input tensors."""
    return reduce(ng.add, ng_inputs)


@refactoring_required
def Min(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate element-wise min of the input tensors."""
    return reduce(ng.minimum, ng_inputs)


@refactoring_required
def Max(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate element-wise max of the input tensors."""
    return reduce(ng.maximum, ng_inputs)


@refactoring_required
def Mean(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate element-wise mean of the input tensors."""
    return reduce(ng.add, ng_inputs) / len(ng_inputs)


# Matrix multiplication
@refactoring_required
def Dot(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate matrix product, similar to numpy.dot."""
    logger.warning('Dot node (%s): Dot operation is deprecated, use MatMul.', onnx_node.name)
    return MatMul(onnx_node, ng_inputs)


@refactoring_required
def MatMul(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate matrix product, similar to numpy.matmul."""
    left, right = cast_axes_for_matmul(*ng_inputs)
    return cast_to_pos_axes(ng.dot(left, right))


@refactoring_required
def Gemm(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate general matrix multiplication Y = alpha * (A @ B) + beta * C."""
    input_a, input_b, input_c = ng_inputs
    alpha = onnx_node.get_attribute_value('alpha', 1)  # Scalar multiplier for A @ B
    beta = onnx_node.get_attribute_value('beta', 1)  # Scalar multiplier for input tensor C
    broadcast = onnx_node.get_attribute_value('broadcast', 1)  # Should C be broadcast?
    trans_a = onnx_node.get_attribute_value('transA', False)  # Should A be transposed?
    trans_b = onnx_node.get_attribute_value('transB', False)  # Should B be transposed?

    if not broadcast:
        logger.warning('Gemm node (%s): import does not support broadcast value %s',
                       onnx_node.name, broadcast)

    if trans_a:
        input_a = ng.Transpose(input_a)

    if trans_b:
        input_b = ng.Transpose(input_b)

    input_a, input_b = cast_axes_for_matmul(input_a, input_b)
    a_dot_b = ng.dot(input_a, input_b)
    a_dot_b = cast_to_pos_axes(a_dot_b)
    return alpha * a_dot_b + beta * input_c


# Convolution ops
@refactoring_required
def Conv(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate output of a convolution operation based on an input tensor and a filter."""
    return cast_to_pos_axes(make_convolution_op(onnx_node, ng_inputs))


@refactoring_required
def ConvTranspose(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate output of a transpose convolution operation based on an input tensor and a filter."""
    return cast_to_pos_axes(make_convolution_op(onnx_node, ng_inputs, transpose=True))


@refactoring_required
def Pad(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Add padding to the input tensor."""
    pads = onnx_node.get_attribute_value('pads')
    constant = 'constant'
    mode = onnx_node.get_attribute_value('mode', constant)  # 'constant', 'reflect' or 'edge'
    value = onnx_node.get_attribute_value('value', 0)

    if mode != constant or value != 0:
        raise NotImplementedError('Pad node (%s): only constant padding with value=0 '
                                  'is supported.', onnx_node.name)

    # Split paddings into pairs for each axis
    pads = [pad for pad in split_pads_into_pairs(pads)]
    return cast_to_pos_axes(ng.pad(ng_inputs[0], pads))


# Pooling
@refactoring_required
def AveragePool(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply average pooling across the the tensor."""
    return cast_to_pos_axes(make_pooling_op(onnx_node, ng_inputs))


@refactoring_required
def MaxPool(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply max pooling across the the tensor."""
    return cast_to_pos_axes(make_pooling_op(onnx_node, ng_inputs))


@refactoring_required
def GlobalMaxPool(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Equivalent to MaxPool with kernel size equal to spatial dimensions of input tensor."""
    return cast_to_pos_axes(make_global_pooling_op(onnx_node, ng_inputs))


@refactoring_required
def GlobalAveragePool(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Equivalent to AveragePool with kernel size equal to spatial dimensions of input tensor."""
    return cast_to_pos_axes(make_global_pooling_op(onnx_node, ng_inputs))


# Reshape ops
@refactoring_required
def Flatten(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Flatten the input tensor into a 2D matrix."""
    data = ng_inputs[0]
    axis = onnx_node.get_attribute_value('axis', 1)

    if axis < 0 or axis > len(data.axes):
        raise ValueError('Flatten node (%s): %d is not a valid value for `axis`.',
                         onnx_node.name, axis)

    return cast_to_pos_axes(ng.flatten_at(data, axis))


@refactoring_required
def Transpose(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Transpose the input tensor similar to numpy.transpose.

    By default, reverse the dimensions, but if `perm` attribute is specified
    permute the axes according to the values given.
    """
    data = ng_inputs[0]
    permute_axes = onnx_node.get_attribute_value('perm')

    if permute_axes:
        input_template = ''.join([ascii_letters[i] for i in range(len(data.axes))])
        output_template = ''.join([ascii_letters[i] for i in permute_axes])
        ng_op = reorder_axes(data, input_template, output_template)
    else:
        ng_op = ng.Transpose(data)

    return cast_to_pos_axes(ng_op)


@refactoring_required
def Slice(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Produce a slice of the input tensor along multiple axes."""
    x = ng_inputs[0]

    starts = onnx_node.get_attribute_value('starts')
    ends = onnx_node.get_attribute_value('ends')
    if not (starts and ends and len(starts) == len(ends)):
        raise ValueError('Slice node (%s): attributes `starts` and `ends` must be set '
                         'and of equal length.', onnx_node.name)

    axes = onnx_node.get_attribute_value('axes', list(range(len(starts))))
    slices_count = max(len(axes), *starts)
    if slices_count > len(x.axes):
        raise ValueError('Slice node (%s): specifies %d slices, there are only %d input axes.',
                         onnx_node.name, slices_count, len(x.axes))

    slices = [slice(starts[axes.index(axis_number)], ends[axes.index(axis_number)])
              if (axis_number in axes) else slice(None) for axis_number in range(len(x.axes))]

    return cast_to_pos_axes(ng.tensor_slice(x, slices))


@refactoring_required
def Concat(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Concatenate a list of tensors into a single tensor."""
    axis = onnx_node.get_attribute_value('axis', 0)

    if len(ng_inputs) < 2:
        raise ValueError('Concat node (%s): requires at least 2 inputs, %d given.',
                         onnx_node.name, len(ng_inputs))

    unique_input_ranks = {len(node.axes) for node in ng_inputs}
    if len(unique_input_ranks) != 1:
        raise ValueError('Concat node (%s): input tensors must be of equal rank.', onnx_node.name)

    if axis >= unique_input_ranks.pop():
        raise ValueError('Concat node (%s): `axis` attribute is out of range.', onnx_node.name)

    ng_axis = ng_inputs[0].axes[axis]
    return ng.concat_along_axis(ng_inputs, ng_axis)


@refactoring_required
def Squeeze(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Remove single-dimensional entries from the shape of a tensor."""
    data = ng_inputs[0]
    axes_to_squeeze = onnx_node.get_attribute_value('axes')

    if max(axes_to_squeeze) >= len(data.axes):
        raise ValueError('Squeeze node (%s): `axes` attribute value %d is out of range.',
                         onnx_node.name, max(axes_to_squeeze))

    slices = [0 if index in axes_to_squeeze else
              slice(None) for index, axis in enumerate(data.axes)]

    return ng.tensor_slice(data, slices)


@refactoring_required
def Reshape(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Reshape the input tensor similar to numpy.reshape."""
    data = ng_inputs[0]
    shape = onnx_node.get_attribute_value('shape', data.axes.lengths)

    # This is code we want to use, but cannot due to a bug:
    # https://github.com/NervanaSystems/private-ngraph/issues/2372
    """
    new_axes = ng.make_axes([ng.make_axis(length=length) for length in shape])
    x = ng.flatten(data)
    x = ng.cast_axes(x, new_axes.flatten())
    x = ng.unflatten(x)
    return cast_to_pos_axes(x)
    """
    return reshape_workaround(data, shape)


@refactoring_required
def Split(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> Tuple[NgraphNode]
    """Split a tensor into a list of tensors."""
    data = ng_inputs[0]
    count_outputs = len(onnx_node.get_output_names())
    axis_to_split = onnx_node.get_attribute_value('axis')
    if axis_to_split < 0:
        axis_to_split = len(data.axes) + axis_to_split
    len_axis_to_split = data.axes[axis_to_split].length
    len_parts = onnx_node.get_attribute_value('split')

    if not len_parts:
        if len_axis_to_split % count_outputs:
            raise ValueError('Split node (%s): Tensor cannot be split into %d equal parts, along '
                             'axis of length %d', onnx_node.name, count_outputs, len_axis_to_split)
        len_parts = [int(len_axis_to_split / count_outputs)] * count_outputs

    outputs = []
    start_index = 0
    for len_part in len_parts:
        end_index = start_index + len_part
        output_axes = [ng.make_axis(length=len_part, name=data.axes[i].name) if i == axis_to_split
                       else data.axes[i] for i in range(len(data.axes))]
        slices = [slice(start_index, end_index) if i == axis_to_split else
                  slice(None) for i in range(len(data.axes))]
        outputs.append(ng.tensor_slice(data, slices, axes=ng.make_axes(output_axes)))
        start_index = end_index

    return tuple(outputs)  # type: ignore


# Misc
@refactoring_required
def Constant(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Produce a constant tensor."""
    value_tensor = onnx_node.get_attribute_value('value')
    return cast_to_pos_axes(ng.constant(value_tensor.to_array()))


@refactoring_required
def Softmax(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute softmax normalized values for each layer in the batch of the given input."""
    input_ = ng_inputs[0]
    axis = onnx_node.get_attribute_value('axis', 1)
    return ng.softmax(input_, normalization_axes=input_.axes[axis])


@refactoring_required
def BatchNormalization(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Carry out batch normalization."""
    x, scale, bias, mean, var = ng_inputs

    is_test = onnx_node.get_attribute_value('is_test', 1)
    spatial = onnx_node.get_attribute_value('spatial', 1)
    epsilon = onnx_node.get_attribute_value('epsilon', 1e-3)
    # @TODO: Implement learning mode support
    # momentum = onnx_node.get_attribute_value('momentum', 0.99)

    if not is_test:
        raise NotImplementedError('BatchNormalization node (%s): only `is_test` mode is currently '
                                  'supported.', onnx_node.name)
    if not spatial:
        raise NotImplementedError('BatchNormalization node (%s): only `spatial` mode is currently '
                                  'supported.', onnx_node.name)

    if len(x.axes) == 5:
        x = rename_axes(x, 'NCHWD')
    else:
        x = rename_axes(x, 'NCHW')

    mean = rename_axes(mean, 'C')
    scale = rename_axes(scale, 'C')
    bias = rename_axes(bias, 'C')
    var = rename_axes(var, 'C')

    ng_op = ng.unflatten(scale * ((x - mean) * ng.reciprocal(ng.sqrt(var + epsilon))) + bias)

    return cast_to_pos_axes(ng_op)
