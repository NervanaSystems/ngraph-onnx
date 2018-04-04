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

from __future__ import print_function
from __future__ import division

import logging
from typing import Tuple, List

import numpy as np
from functools import reduce
from ngraph.utils.types import get_dtype
from ngraph_onnx import TYPE_CHECKING

from ngraph.impl import Node as NgraphNode
import ngraph as ng

from ngraph_onnx.onnx_importer.utils.binary import broadcast_for_binary_operation
from ngraph_onnx.onnx_importer.utils.conv import make_convolution_op
from ngraph_onnx.onnx_importer.utils.decorators import refactoring_required
from ngraph_onnx.onnx_importer.utils.matmul import has_matmul_compatible_shapes
from ngraph_onnx.onnx_importer.utils.misc import split_pads_into_pairs
from ngraph_onnx.onnx_importer.utils.pool import make_pooling_op, make_global_pooling_op
from ngraph_onnx.onnx_importer.utils.reduction import make_reduction_op, get_reduction_axes
from ngraph_onnx.onnx_importer.utils.reshape import transpose, infer_dimensions, \
    flatten_innermost_empty_dims, reorder_axes, make_slice_op

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


def Reciprocal(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply f(x) = 1/x to the input tensor elementwise."""
    return 1 / ng_inputs[0]


def Sqrt(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply f(x) = x^0.5 (square root) to the input tensor elementwise."""
    return ng.sqrt(ng_inputs[0])


def Sigmoid(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply the sigmoid function, f(x) = 1 / (1 + exp(-x)) to the input tensor elementwise."""
    return 1 / (1 + ng.exp(ng.negative(ng_inputs[0])))


def Tanh(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate the hyperbolic tangent of the input tensor elementwise."""
    return ng.tanh(ng_inputs[0])


def Relu(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply the Relu function, f(x) = max(0, x) to the input tensor elementwise."""
    return ng.maximum(ng_inputs[0], 0)


def LeakyRelu(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply the Leaky Relu function to the input tensor elementwise.

    f(x) = alpha * x for x < 0, f(x) = x for x >= 0
    """
    alpha = onnx_node.get_attribute_value('alpha', 0.01)
    if not 0 <= alpha <= 1:
        logger.warning('LeakyRelu node (%s): alpha value should be in range (0,1), but is: %s',
                       onnx_node.name, alpha)
    return ng.maximum(alpha * ng_inputs[0], ng_inputs[0])


def PRelu(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply the Parametric Relu function to the input tensor elementwise.

    f(x) = slope * x for x < 0, f(x) = x for x >= 0
    The slope parameter is passed to the node as its second input.
    """
    x, slope = ng_inputs
    return ng.maximum(slope * x, x)


def Selu(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply the scaled exponential linear unit function to the input tensor elementwise.

    f(x) = gamma * (alpha * exp(x) - alpha) for x <= 0, f(x) = gamma * x for x > 0
    """
    x = ng_inputs[0]
    alpha = onnx_node.get_attribute_value('alpha', 1.6732)
    gamma = onnx_node.get_attribute_value('gamma', 1.0507)

    return (gamma * (ng.maximum(x, 0) + alpha * (ng.exp(ng.negative(ng.maximum(ng.negative(x), 0))) - 1)))


def Elu(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply the exponential linear unit function to the input tensor elementwise.

    f(x) = alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0
    """
    x = ng_inputs[0]
    alpha = onnx_node.get_attribute_value('alpha', 1)

    if not alpha < 0:
        logger.warning('Elu node (%s): alpha value should be positive, but is: %s',
                       onnx_node.name, alpha)

    return (ng.maximum(x, 0) + alpha * (ng.exp(ng.negative(ng.maximum(ng.negative(x), 0))) - 1))


@refactoring_required
def Softplus(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply Softplus function, f(x) = ln(exp(x) + 1) to the input tensor elementwise."""
    return ng.log((ng.exp(ng_inputs[0]) + 1))


# Reduction Ops
def ReduceSum(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the sum of the input tensor's elements along the provided axes.

    The output tensor has the same rank as the input if Node attribute keepdims equals 1.
    If keepdims equals 0, then the output tensor have the reduced dimension pruned.
    """
    return make_reduction_op(ng.sum, onnx_node, ng_inputs[0])


def ReduceMax(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the maximum value of the input tensor's elements along the provided axes."""
    return make_reduction_op(ng.max, onnx_node, ng_inputs[0])


def ReduceMin(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the minimum value of the input tensor's elements along the provided axes."""
    return make_reduction_op(ng.min, onnx_node, ng_inputs[0])


def ReduceLogSumExp(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the log sum exponent of the input tensor's element' along the provided axes."""
    op = ng.exp(ng_inputs[0])
    op = make_reduction_op(ng.sum, onnx_node, op)
    op = ng.log(op)
    return op


def ReduceMean(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the mean value of the input tensor's elements along the provided axes."""
    input_shape = list(ng_inputs[0].shape)
    sum_node = make_reduction_op(ng.sum, onnx_node, ng_inputs[0])
    reduction_axes = get_reduction_axes(onnx_node, ng_inputs[0])
    avg_elem_count = np.prod([input_shape[x] for x in reduction_axes])
    const_node = ng.broadcast(ng.constant(avg_elem_count, get_dtype(sum_node.get_element_type())),
                              sum_node.shape)
    return ng.divide(sum_node, const_node)


def ReduceProd(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the product of the input tensor's elements along the provided axes."""
    return make_reduction_op(ng.prod, onnx_node, ng_inputs[0])


@refactoring_required
def ArgMin(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the indices of the min elements of the input tensor along the provided axes."""
    return None  # tmp
    # return make_reduction_op(ng.argmin, onnx_node, ng_inputs[0])


@refactoring_required
def ArgMax(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the indices of the max elements of the input tensor along the provided axes."""
    return None  # tmp
    # return make_reduction_op(ng.argmax, onnx_node, ng_inputs[0])


# Binary Ops
def Add(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform element-wise binary addition."""
    left, right = broadcast_for_binary_operation(onnx_node, ng_inputs)
    return ng.add(left, right)


def Sub(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform element-wise binary subtraction."""
    left, right = broadcast_for_binary_operation(onnx_node, ng_inputs)
    return ng.subtract(left, right)


def Mul(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform element-wise binary multiplication."""
    left, right = broadcast_for_binary_operation(onnx_node, ng_inputs)
    return ng.multiply(left, right)


def Div(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform element-wise binary division."""
    left, right = broadcast_for_binary_operation(onnx_node, ng_inputs)
    return ng.divide(left, right)


# Logical ops
def Equal(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform the `equal` logical operation elementwise on two input tensors."""
    left, right = broadcast_for_binary_operation(onnx_node, ng_inputs)
    return ng.equal(left, right)


def Less(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform the `less` logical operation elementwise on two input tensors."""
    left, right = broadcast_for_binary_operation(onnx_node, ng_inputs)
    return ng.less(left, right)


def Greater(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform the `greater` logical operation elementwise on two input tensors."""
    left, right = broadcast_for_binary_operation(onnx_node, ng_inputs)
    return ng.greater(left, right)


def And(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform the `and` logical operation elementwise on two input tensors."""
    left, right = broadcast_for_binary_operation(onnx_node, ng_inputs)
    left = ng.convert(ng.not_equal(left, 0), int)
    right = ng.convert(ng.not_equal(right, 0), int)
    return ng.convert(left * right, bool)


def Or(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform the `or` logical operation elementwise on two input tensors."""
    left, right = broadcast_for_binary_operation(onnx_node, ng_inputs)
    left = ng.convert(ng.not_equal(left, 0), int)
    right = ng.convert(ng.not_equal(right, 0), int)
    return (left + right) > 0


def Xor(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform the `xor` logical operation elementwise on two input tensors."""
    left, right = broadcast_for_binary_operation(onnx_node, ng_inputs)
    not_left = ng.convert(ng.equal(left, 0), int)
    left = ng.convert(ng.not_equal(left, 0), int)
    right = ng.convert(ng.not_equal(right, 0), int)
    not_right = ng.convert(ng.equal(right, 0), int)

    return ((not_left * right) + (not_right * left)) > 0


def Not(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Return the negation of the input tensor elementwise."""
    data = ng.convert(ng.not_equal(ng_inputs[0], 0), bool)
    return ng.logical_not(data)


# Variadic Ops
def Sum(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate element-wise sum of the input tensors."""
    return reduce(ng.add, ng_inputs)


def Min(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate element-wise min of the input tensors."""
    return reduce(ng.minimum, ng_inputs)


def Max(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate element-wise max of the input tensors."""
    return reduce(ng.maximum, ng_inputs)


def Mean(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate element-wise mean of the input tensors."""
    sum_node = reduce(ng.add, ng_inputs)
    count_array = np.full(sum_node.shape, len(ng_inputs),
                          dtype=get_dtype(sum_node.get_element_type()))
    return sum_node / ng.constant(count_array)


# Matrix multiplication
@refactoring_required
def Dot(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate matrix product, similar to numpy.dot."""
    logger.warning('Dot node (%s): Dot operation is deprecated, use MatMul.', onnx_node.name)
    return MatMul(onnx_node, ng_inputs)


def MatMul(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate matrix product, similar to numpy.matmul."""
    left, right = ng_inputs
    return ng.dot(left, right)


def Gemm(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate general matrix multiplication Y = alpha * (A @ B) + beta * C."""
    input_a, input_b, input_c = ng_inputs
    alpha = onnx_node.get_attribute_value('alpha', 1)  # Scalar multiplier for A @ B
    beta = onnx_node.get_attribute_value('beta', 1)  # Scalar multiplier for input tensor C
    broadcast = onnx_node.get_attribute_value('broadcast', 1)  # Should C be broadcast?
    trans_a = onnx_node.get_attribute_value('transA', False)  # Should A be transposed?
    trans_b = onnx_node.get_attribute_value('transB', False)  # Should B be transposed?

    if trans_a:
        input_a = transpose(input_a)
    if trans_b:
        input_b = transpose(input_b)

    # onnx-tensorflow: https://github.com/onnx/onnx-tensorflow/
    #  blob/17075f44c9071600beccfc62c92b22d1cd957bfd/onnx_tf/backend.py#L711
    # They have hardcoded flatten input `A` before transposition.
    #
    # Firstly, we check whether input data have incompatible shapes and then try flatten input data.
    if not has_matmul_compatible_shapes(input_a.shape, input_b.shape):
        input_a = flatten_innermost_empty_dims(input_a)
        input_b = flatten_innermost_empty_dims(input_b)
        if not has_matmul_compatible_shapes(input_a.shape, input_b.shape):
            raise ValueError('Gemm node (%s): input "A" and "B" data shapes are incompatible to '
                             'multiply with each other.', onnx_node.name)

    a_dot_b = ng.dot(input_a, input_b)

    if not broadcast and input_c.shape != a_dot_b.shape:
        raise ValueError('Gemm node (%s): input data shapes are incompatible and broadcast '
                         ' was not requested!', onnx_node.name)

    return alpha * a_dot_b + beta * input_c


# Convolution ops
def Conv(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate output of a convolution operation based on an input tensor and a filter."""
    return make_convolution_op(onnx_node, ng_inputs)


@refactoring_required
def ConvTranspose(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate output of a transpose convolution operation based on an input tensor and a filter."""
    return None  # tmp
    # return cast_to_pos_axes(make_convolution_op(onnx_node, ng_inputs, transpose=True))


def Pad(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Add padding to the input tensor."""
    pads = onnx_node.get_attribute_value('pads')
    constant = 'constant'
    mode = onnx_node.get_attribute_value('mode', constant)  # 'constant', 'reflect' or 'edge'
    value = onnx_node.get_attribute_value('value', 0)

    if mode != constant:
        raise NotImplementedError('Pad node (%s): only constant padding is supported.', onnx_node.name)

    # Split paddings into pairs for each axis
    pading_below, pading_above = split_pads_into_pairs(pads)
    return ng.pad(ng_inputs[0], ng.constant(value,
                  dtype=get_dtype(ng_inputs[0].get_element_type())), pading_below, pading_above)


# Pooling
def AveragePool(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply average pooling across the the tensor."""
    return make_pooling_op(onnx_node, ng_inputs)


def MaxPool(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply max pooling across the the tensor."""
    return make_pooling_op(onnx_node, ng_inputs)


def GlobalMaxPool(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Equivalent to MaxPool with kernel size equal to spatial dimensions of input tensor."""
    return make_global_pooling_op(onnx_node, ng_inputs)


def GlobalAveragePool(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Equivalent to AveragePool with kernel size equal to spatial dimensions of input tensor."""
    return make_global_pooling_op(onnx_node, ng_inputs)


# Reshape ops
def Flatten(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Flatten the input tensor into a 2D matrix.

    Flattening happens at axis specified by 'axis' attribute.
    First dimension of output tensor is the product of [d_0, ... d_{axis-1}] dimensions of input tensor.
    The last dimension is the product of the rest of input tensor dimensions: [d_{axis}, ..., d_n]
    """
    input_node = ng_inputs[0]
    axis = onnx_node.get_attribute_value('axis', 1)
    input_shape = list(input_node.shape)

    if axis < 0 or axis > len(input_shape):
        raise ValueError('Flatten node (%s): %d is not a valid value for `axis`.',
                         onnx_node.name, axis)

    first_dim = 1
    last_dim = 1

    for index in range(len(input_shape)):
        last_dim = last_dim * input_shape[index]
        if index < axis:
            first_dim = last_dim

    last_dim = int(last_dim / first_dim)
    # the order in which we iterate over input tensor dimensions while reshaping it.
    input_order = list(range(len(input_shape)))
    output_shape = [first_dim, last_dim]

    return ng.reshape(input_node, input_order, output_shape)


def Transpose(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Transpose the input tensor similar to numpy.transpose.

    By default, reverse the dimensions, but if `perm` attribute is specified
    permute the axes according to the values given.
    """
    input_node = ng_inputs[0]
    permute_axes = onnx_node.get_attribute_value('perm')
    if permute_axes is None:
        return transpose(input_node)
    else:
        return reorder_axes(input_node, permute_axes)


def Slice(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Produce a slice of the input tensor along multiple axes."""
    input_node = ng_inputs[0]

    starts = onnx_node.get_attribute_value('starts')
    ends = onnx_node.get_attribute_value('ends')
    if not (starts and ends and len(starts) == len(ends)):
        raise ValueError('Slice node (%s): attributes `starts` and `ends` must be set '
                         'and of equal length.', onnx_node.name)

    axes = onnx_node.get_attribute_value('axes')
    if axes is None:
        axes = list(range(len(starts)))
    else:
        for axis in axes:
            if axis < 0 or axis > len(input_node.shape) - 1:
                raise ValueError('Slice node (%s): specified axes are out of node\' dimensions '
                                 'bounds', onnx_node.name)

    return make_slice_op(input_node, axes, starts, ends)


def Concat(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Concatenate a list of tensors into a single tensor."""
    axis = onnx_node.get_attribute_value('axis')
    if axis is None:
        raise ValueError('Concat node (%s): requires "axis" attribute', onnx_node.name)

    if len(ng_inputs) < 2:
        raise ValueError('Concat node (%s): requires at least 2 inputs, %d given.',
                         onnx_node.name, len(ng_inputs))

    unique_input_ranks = {len(node.shape) for node in ng_inputs}
    if len(unique_input_ranks) != 1:
        raise ValueError('Concat node (%s): input tensors must be of equal rank.', onnx_node.name)

    if axis >= unique_input_ranks.pop() or axis < 0:
        raise ValueError('Concat node (%s): `axis` attribute is out of range.', onnx_node.name)

    return ng.concat(ng_inputs, axis)


def Squeeze(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Remove single-dimensional entries from the shape of a tensor."""
    data = ng_inputs[0]
    axes_to_squeeze = onnx_node.get_attribute_value('axes')
    if axes_to_squeeze is None:
        raise ValueError('Squeeze node (%s): the "axes" attribute is mandatory.', onnx_node.name)

    for axis in axes_to_squeeze:
        if axis < 0 or axis >= len(data.shape):
            raise ValueError('Squeeze node (%s): `axes` attribute value %d is out of range.',
                             onnx_node.name, axis)
        if data.shape[axis] > 1:
            raise ValueError('Squeeze node (%s): can only remove single-dimensional axes: '
                             'shape[%d] = %d', onnx_node.name, axis, data.shape[axis])

    input_order = list(range(len(data.shape)))
    out_shape = [data.shape[i] for i in range(len(data.shape)) if i not in axes_to_squeeze]

    return ng.reshape(data, input_order, out_shape)


def Unsqueeze(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Insert single-dimensional entries to the shape of a tensor.

    :param onnx_node: The ONNX node we create operation for.
    :param ng_inputs: nGraph node which provide data.
    :return: nGraph node with applied unsqueeze operation on it's data.
    """
    data = ng_inputs[0]
    axes = onnx_node.get_attribute_value('axes')
    if axes is None:
        raise ValueError('Unsqueeze node (%s): the "axes" attribute is mandatory.', onnx_node.name)

    input_order = list(range(len(data.shape)))
    out_shape = list(data.shape)
    axes.sort()
    for axis in axes:
        # this condition forbids adding new dimensions greater than len(out_shape), i.e:
        # if we have input tensor of shape (3,4,5) and we provide 'axes' attribute with value
        # [10], then such input is considered invalid.
        if axis < 0 or axis > len(out_shape):
            raise ValueError('Unsqueeze node (%s): `axes` attribute value %d is out of range.',
                             onnx_node.name, axis)
        out_shape.insert(axis, 1)

    return ng.reshape(data, input_order, out_shape)


def Reshape(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Reshape the input tensor similar to numpy.reshape.

    At most one dimension of the new shape can be -1. In this case, the value is inferred from
    the size of the tensor and the remaining dimensions. A dimension could also be 0, in which
    case the actual dimension value is going to be copied from the shape argument.
    """
    data = ng_inputs[0]
    output_shape = onnx_node.get_attribute_value('shape', data.shape)

    if output_shape == data.shape:
        return data

    input_order = list(range(len(data.shape)))
    output_shape = infer_dimensions(onnx_node.name, data.shape, output_shape)

    return ng.reshape(data, input_order, output_shape)


def Split(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> Tuple[NgraphNode, ...]
    """Split a tensor into a list of tensors."""
    data = ng_inputs[0]
    count_outputs = len(onnx_node.get_output_names())
    axis_to_split = onnx_node.get_attribute_value('axis', 0)

    if axis_to_split < 0 or axis_to_split >= len(data.shape):
        raise ValueError('Split node (%s) provided split axis is out of input tensor dimensions'
                         ' range.', onnx_node.name)

    len_axis_to_split = data.shape[axis_to_split]
    len_parts = onnx_node.get_attribute_value('split')

    if len_parts is None:
        if len_axis_to_split % count_outputs:
            raise ValueError('Split node (%s): Tensor cannot be split into %d equal parts, along '
                             'axis of length %d', onnx_node.name, count_outputs, len_axis_to_split)
        len_parts = [int(len_axis_to_split / count_outputs)] * count_outputs
    elif sum(len_parts) != len_axis_to_split:
        raise ValueError('Split node (%s): provided lengths of split parts does not sum up to '
                         'length of axis we split on: %d != %d', onnx_node.name, sum(len_parts),
                         len_axis_to_split)

    outputs = []
    start_index = 0

    for len_part in len_parts:
        end_index = start_index + len_part
        outputs.append(make_slice_op(data, [axis_to_split], [start_index], [end_index]))
        start_index = end_index

    return tuple(outputs)


# Misc
def Constant(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Produce a constant tensor."""
    value_tensor = onnx_node.get_attribute_value('value')
    return ng.constant(value_tensor.to_array())


def Softmax(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute softmax normalized values for each layer in the batch of the given input."""
    input_ = ng_inputs[0]
    axis = onnx_node.get_attribute_value('axis', 1)
    if axis == -1:  # Use last dimension
        axis = len(input_.shape) - 1
    return ng.softmax(input_, range(axis, len(input_.shape)))


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

    mean = ng.broadcast(mean, x.shape, axis=1)
    scale = ng.broadcast(scale, x.shape, axis=1)
    var = ng.broadcast(var, x.shape, axis=1)
    bias = ng.broadcast(bias, x.shape, axis=1)
    epsilon = ng.broadcast(ng.constant(epsilon, dtype=get_dtype(x.get_element_type())),
                           x.shape, axis=1)
    return (scale * ((x - mean) * (1 / (ng.sqrt(var + epsilon)))) + bias)
