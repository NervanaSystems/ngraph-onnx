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

from ngraph_onnx.onnx_importer.utils.binary import broadcast_for_binary_operation, \
    numpy_style_broadcast_for_binary_operation
from ngraph_onnx.onnx_importer.utils.conv import make_convolution_op, get_strides, get_dilations, \
    get_pads
from ngraph_onnx.onnx_importer.utils.matmul import reshape_for_matmul
from ngraph_onnx.onnx_importer.utils.types import onnx_tensor_type_to_numpy_type
from ngraph_onnx.onnx_importer.utils.misc import split_pads_into_pairs
from ngraph_onnx.onnx_importer.utils.pool import make_pooling_op, make_global_pooling_op
from ngraph_onnx.onnx_importer.utils.reduction import make_reduction_op, get_reduction_axes
from ngraph_onnx.onnx_importer.utils.reshape import transpose, infer_dimensions, \
    reorder_axes, make_slice_op, flatten
from ngraph_onnx.onnx_importer.utils.numeric_limits import NumericLimits
from ngraph_onnx.onnx_importer.utils.types import get_bool_nodes

if TYPE_CHECKING:
    from ngraph_onnx.onnx_importer.model_wrappers import NodeWrapper

logger = logging.getLogger(__name__)

# OP STATUS
# ArgMax-1                    TODO
# ArgMin-1                    TODO
# ConvTranspose-1             TODO
# GRU-1                       TODO
# Gather-1                    TODO
# GlobalLpPool-1              TODO
# Hardmax-1                   TODO
# InstanceNormalization-1     TODO
# LRN-1                       TODO
# LSTM-1                      TODO
# LpNormalization-1           TODO
# LpPool-1                    TODO
# MaxRoiPool-1                TODO
# RNN-1                       TODO
# RandomNormal-1              TODO
# RandomNormalLike-1          TODO
# RandomUniform-1             TODO
# RandomUniformLike-1         TODO
# SpaceToDepth-1              TODO
# Tile-1                      TODO
# TopK-1                      TODO
# Upsample-1                  TODO

# EXPERIMENTAL OPS
# Affine-1                    TODO
# ATen-1                      TODO
# ConstantFill-1              TODO
# Crop-1                      TODO
# GRUUnit-1                   TODO
# GivenTensorFill-1           TODO
# If-1                        TODO
# ImageScaler-1               TODO
# Loop-1                      TODO
# LoopIndexTensor-1           TODO
# MeanVarianceNormalization-1 TODO
# ParametricSoftplus-1        TODO
# Scale-1                     TODO
# ScaledTanh-1                TODO


# Unary Ops
def Abs(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply f(x) = abs(x) to the input tensor elementwise."""
    return ng.absolute(ng_inputs[0])


def Ceil(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply f(x) = ceil(x) to the input tensor elementwise."""
    return ng.ceiling(ng_inputs[0])


def Cast(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Limit input tensor values within specified interval."""
    data = ng_inputs[0]
    cast_to_type = onnx_node.get_attribute_value('to')
    if cast_to_type is None:
        raise ValueError('Cast node (%s): \'to\' attribute is required.')

    input_tensor_type = get_dtype(data.get_element_type())
    new_type = onnx_tensor_type_to_numpy_type(cast_to_type)
    unsupported_types = [
        onnx_tensor_type_to_numpy_type('COMPLEX64'),
        onnx_tensor_type_to_numpy_type('COMPLEX128'),
    ]

    if input_tensor_type in unsupported_types:
        raise ValueError('Cast node (%s): input tensor data type (%s) is not supported.',
                         onnx_node.name, str(input_tensor_type))
    if new_type in unsupported_types:
        raise ValueError('Cast node (%s): casting to type (%s) is not supported.',
                         onnx_node.name, str(new_type))

    return ng.convert(data, new_type)


def Clip(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Limit input tensor values within specified interval."""
    data = ng_inputs[0]
    data_elem_dtype = get_dtype(data.get_element_type())
    max_value = onnx_node.get_attribute_value('max', np.finfo(data_elem_dtype).max)
    min_value = onnx_node.get_attribute_value('min', np.finfo(data_elem_dtype).min)

    return ng.minimum(ng.maximum(data, ng.constant(min_value, data_elem_dtype)),
                      ng.constant(max_value, data_elem_dtype))


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
    return ng.relu(ng_inputs[0])


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
    if len(slope.shape) == 0:
        return ng.maximum(slope * x, x)
    elif slope.shape[0] == 1:
        slope = ng.broadcast_to(slope, [x.shape[0], 1])
        slope = ng.reshape(slope, [x.shape[0]])
        return ng.maximum(ng.broadcast_to(slope, x.shape, 0) * x, x)
    else:
        return ng.maximum(ng.broadcast_to(slope, x.shape, 1) * x, x)


def ThresholdedRelu(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply the Thresholded Relu function to the input tensor elementwise.

    f(x) = 0 for x <= alpha, f(x) = x for x > alpha
    """
    x = ng_inputs[0]
    alpha = onnx_node.get_attribute_value('alpha', 1.0)
    x_map = ng.convert(ng.greater(x, alpha), get_dtype(x.get_element_type()))
    x = x * x_map
    return x


def Selu(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply the scaled exponential linear unit function to the input tensor elementwise.

    f(x) = gamma * (alpha * exp(x) - alpha) for x <= 0, f(x) = gamma * x for x > 0
    """
    x = ng_inputs[0]
    alpha = onnx_node.get_attribute_value('alpha', 1.67326319217681884765625)
    gamma = onnx_node.get_attribute_value('gamma', 1.05070102214813232421875)

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


def Softmax(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute softmax normalized values for each layer in the batch of the given input.

    :param onnx_node: The ONNX node representing this operation.
    :param ng_inputs: The input tensors.
    :return: The tensor with applied Softmax operation.
    """
    data = ng_inputs[0]
    axis = onnx_node.get_attribute_value('axis', 1)
    # negative values are interpreted as i-th index from the end.
    if axis < 0:
        axis = len(data.shape) + axis
    if axis < 0 or axis >= len(data.shape):
        raise ValueError('Softmax node (%s): provided axis attribute is out of input tensor'
                         ' dimensions range.', onnx_node.name)
    return ng.softmax(data, range(axis, len(data.shape)))


def LogSoftmax(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute logarithm of softmax values for each layer in the batch of the given input.

    :param onnx_node: The ONNX node representing this operation.
    :param ng_inputs: The input tensors.
    :return: The tensor with applied LogSoftmax operation.
    """
    data = ng_inputs[0]
    axis = onnx_node.get_attribute_value('axis', 1)
    if axis < 0 or axis >= len(data.shape):
        raise ValueError('LogSoftmax node (%s): provided axis attribute is out of input tensor'
                         ' dimensions range.', onnx_node.name)
    return ng.log(ng.softmax(data, range(axis, len(data.shape))))


def Identity(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Identity operator returning input tensor.

    :param onnx_node: The ONNX node representing this operation.
    :param ng_inputs: The input tensors.
    :return: The input tensor.
    """
    return ng_inputs[0]


def HardSigmoid(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply f(x) = max(0, min(1, alpha * x + beta)) function to tensor element-wise.

    :param onnx_node: The ONNX node representing this operation.
    :param ng_inputs: The input tensors.
    :return: The tensor with applied HardSigmoid operation.
    """
    data = ng_inputs[0]
    data_type = get_dtype(data.get_element_type()).type
    alpha = onnx_node.get_attribute_value('alpha', float(0.2))
    beta = onnx_node.get_attribute_value('beta', float(0.5))
    return ng.maximum(data_type(0), ng.minimum(data_type(1), alpha * data + beta))


def Softplus(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply Softplus function, f(x) = ln(exp(x) + 1) to the input tensor element-wise.

    :param onnx_node: The ONNX node representing this operation.
    :param ng_inputs: The input tensors.
    :return: The tensor with applied Softplus operation.
    """
    return ng.log((ng.exp(ng_inputs[0]) + 1))


def Softsign(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Apply Softsign function, f(x) = x / (1 + |x|) to the input tensor element-wise.

    :param onnx_node: The ONNX node representing this operation.
    :param ng_inputs: The input tensors.
    :return: The tensor with applied Softsign operation.
    """
    return ng_inputs[0] / (1 + ng.abs(ng_inputs[0]))


# Reduction Ops
def ReduceSum(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the sum of the input tensor's elements along the provided axes.

    The output tensor has the same rank as the input if Node attribute keepdims equals 1.
    If keepdims equals 0, then the output tensor have the reduced dimension pruned.
    """
    return make_reduction_op(ng.sum, onnx_node, ng_inputs[0])


def ReduceSumSquare(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the sum square of the input tensor's element along the provided axes.

    :param onnx_node: The ONNX node representing this operation.
    :param ng_inputs: The input tensors.
    :return: The tensor with applied ReduceSumSquare operation.
    """
    square_node = ng_inputs[0] * ng_inputs[0]
    return make_reduction_op(ng.sum, onnx_node, square_node)


def ReduceMax(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the maximum value of the input tensor's elements along the provided axes."""
    return make_reduction_op(ng.max, onnx_node, ng_inputs[0])


def ReduceMin(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the minimum value of the input tensor's elements along the provided axes."""
    return make_reduction_op(ng.min, onnx_node, ng_inputs[0])


def ReduceLogSum(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the log sum of the input tensor's element along the provided axes.

    :param onnx_node: The ONNX node representing this operation.
    :param ng_inputs: The input tensors.
    :return: The tensor with applied ReduceLogSum operation.
    """
    sum_node = make_reduction_op(ng.sum, onnx_node, ng_inputs[0])
    return ng.log(sum_node)


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
    const_node = ng.broadcast_to(ng.constant(avg_elem_count, get_dtype(sum_node.get_element_type())),
                                 sum_node.shape)
    return ng.divide(sum_node, const_node)


def ReduceProd(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the product of the input tensor's elements along the provided axes.

    :param onnx_node: The ONNX node representing this operation.
    :param ng_inputs: The input tensors.
    :return: The tensor with applied ReduceProd operation.
    """
    return make_reduction_op(ng.prod, onnx_node, ng_inputs[0])


def ReduceL1(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the L1 norm of the input tensor's element along the provided axes.

    :param onnx_node: The ONNX node representing this operation.
    :param ng_inputs: The input tensors.
    :return: The tensor with applied ReduceL1 operation.
    """
    abs_node = ng.abs(ng_inputs[0])
    return make_reduction_op(ng.sum, onnx_node, abs_node)


def ReduceL2(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Compute the L2 norm of the input tensor's element along the provided axes.

    :param onnx_node: The ONNX node representing this operation.
    :param ng_inputs: The input tensors.
    :return: The tensor with applied ReduceL2 operation.
    """
    square_node = ng_inputs[0] * ng_inputs[0]
    sum_node = make_reduction_op(ng.sum, onnx_node, square_node)
    return ng.sqrt(sum_node)


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


def Pow(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform element-wise binary power."""
    base, exponent = broadcast_for_binary_operation(onnx_node, ng_inputs)
    return ng.power(base, exponent)


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
    left, right = get_bool_nodes(broadcast_for_binary_operation(onnx_node, ng_inputs))
    return ng.logical_and(left, right)


def Or(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform the `or` logical operation elementwise on two input tensors."""
    left, right = get_bool_nodes(broadcast_for_binary_operation(onnx_node, ng_inputs))
    return ng.logical_or(left, right)


def Xor(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform the `xor` logical operation elementwise on two input tensors."""
    left, right = get_bool_nodes(broadcast_for_binary_operation(onnx_node, ng_inputs))
    return ng.logical_or(ng.logical_and(left, ng.logical_not(right)),
                         ng.logical_and(ng.logical_not(left), right))


def Not(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Return the negation of the input tensor elementwise."""
    return ng.logical_not(ng.convert(ng_inputs[0], bool))


# Variadic Ops
def Sum(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate element-wise sum of the input tensors."""
    initial_value_node = ng.constant(0, get_dtype(ng_inputs[0].get_element_type()))
    return reduce(ng.add, ng_inputs, initial_value_node)


def Min(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate element-wise min of the input tensors."""
    np_dtype = get_dtype(ng_inputs[0].get_element_type())
    initial_value_node = ng.constant(NumericLimits.max(np_dtype), np_dtype)
    return reduce(ng.minimum, ng_inputs, initial_value_node)


def Max(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate element-wise max of the input tensors."""
    np_dtype = get_dtype(ng_inputs[0].get_element_type())
    initial_value_node = ng.constant(NumericLimits.min(np_dtype), np_dtype)
    return reduce(ng.maximum, ng_inputs, initial_value_node)


def Mean(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate element-wise mean of the input tensors."""
    initial_value_node = ng.constant(0, get_dtype(ng_inputs[0].get_element_type()))
    sum_node = reduce(ng.add, ng_inputs, initial_value_node)
    count_array = np.full(sum_node.shape, len(ng_inputs),
                          dtype=get_dtype(sum_node.get_element_type()))
    return sum_node / ng.constant(count_array)


# Matrix multiplication
def MatMul(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate matrix product, similar to numpy.matmul."""
    left, right = ng_inputs
    return ng.dot(left, right)


def Gemm(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate general matrix multiplication Y = alpha * (A @ B) + beta * C.

    Support is currently limited to 2D matrices only. Higher dimensional tensors will
    be flattened to 2D before multiplication.
    """
    input_a, input_b, input_c = ng_inputs
    alpha = onnx_node.get_attribute_value('alpha', 1)  # Scalar multiplier for A @ B
    beta = onnx_node.get_attribute_value('beta', 1)  # Scalar multiplier for input tensor C
    trans_a = onnx_node.get_attribute_value('transA', False)  # Should A be transposed?
    trans_b = onnx_node.get_attribute_value('transB', False)  # Should B be transposed?

    if trans_a:
        input_a = transpose(input_a)
    if trans_b:
        input_b = transpose(input_b)

    input_a, input_b = reshape_for_matmul(onnx_node, input_a, input_b)

    a_dot_b = ng.dot(input_a, input_b)

    if alpha != 1:
        a_dot_b = alpha * a_dot_b

    if beta != 1:
        input_c = beta * input_c

    _, input_c = numpy_style_broadcast_for_binary_operation(onnx_node, [a_dot_b, input_c])
    return a_dot_b + input_c


def Dropout(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Dropout [inference only].

    For inference Dropout is a simple data pass through.
    """
    return ng_inputs[0]


# Convolution ops
def Conv(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate output of a convolution operation based on an input tensor and a filter."""
    return make_convolution_op(onnx_node, ng_inputs)


def ConvTranspose(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Calculate convolution transpose."""
    if len(ng_inputs) == 3:
        data, weights, bias = ng_inputs
    elif len(ng_inputs) == 2:
        data, weights = ng_inputs
        bias = ng.constant(0, dtype=get_dtype(data.get_element_type()))

    strides = get_strides(onnx_node)
    dilation = get_dilations(onnx_node)
    padding_below, padding_above = get_pads(onnx_node)

    output_padding = onnx_node.get_attribute_value('output_padding')
    if output_padding is None:
        raise ValueError('ConvTranspose node (s%): output_padding attribute is required.', onnx_node.name)

    data_shape = list(data.shape)
    weights_shape = list(weights.shape)

    num_spatial_dims = len(data.shape) - 2
    data_dilation_strides = [1, 1]

    data_batch_shape = [1] * (num_spatial_dims + 2)
    data_batch_shape[0] = data_shape[0]
    data_batch_shape[1] = weights_shape[1]

    for i in range(num_spatial_dims):
        # Calculating spatial dims of data output shape for ngraph conv backprop op
        # | pb + s(ds-1) + op - d(ws-1)+1 |
        # | ----------------------------- | + 1
        # |_            dds              _|
        #
        # d   - dilation
        # ds  - data shape
        # dds - data dilation strides
        # op  - putput padding
        # pb  - padding below
        # s   - strides
        # ws  - weights shape
        data_batch_shape[i + 2] = (
            (
                padding_below[i]
                + ((data_shape[i + 2] - 1) * strides[i] + 1)
                + output_padding[i]
            )
            - ((weights_shape[i + 2] - 1) * dilation[i] + 1)
            + 1
        ) // data_dilation_strides[i] + 1

    transconv = ng.convolution_backprop_data(data_batch_shape,
                                             weights,
                                             data,
                                             strides,
                                             dilation,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides)
    if len(bias.shape) > 0:
        return transconv + ng.broadcast_to(bias, transconv.shape, 1)
    else:
        return transconv


def Pad(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Add padding to the input tensor."""
    data = ng_inputs[0]
    # Oprator set version 1
    paddings = onnx_node.get_attribute_value('paddings')
    # Operator set version >= 2
    pads = onnx_node.get_attribute_value('pads')

    pads = pads if pads is not None else paddings
    if pads is None:
        raise ValueError('Pad node (s%): pads attribute is required.', onnx_node.name)

    constant = 'constant'
    mode = onnx_node.get_attribute_value('mode', constant)  # 'constant', 'reflect' or 'edge'
    value = onnx_node.get_attribute_value('value', 0.)

    if len(pads) != 2 * len(data.shape):
        raise ValueError('Pad node (%s): \'pads rank (%d) should be double of input tensor '
                         'rank (%d).', onnx_node.name, len(pads), len(data.shape))

    # Operator set version 1 accepts only positive values, while operator set version 2 use negative
    # values to remove pads elements. Here we check only for latter case.
    if any(map(lambda x: x < 0, pads)):
        raise NotImplementedError('Pad node (%s): removing padding elements is not supported yet.',
                                  onnx_node.name)
    if mode != constant:
        raise NotImplementedError('Pad node (%s): only constant padding is supported.',
                                  onnx_node.name)

    # Split paddings into pairs for each axis
    pading_below, pading_above = split_pads_into_pairs(pads)
    return ng.pad(data, ng.constant(value,
                  dtype=get_dtype(data.get_element_type())), pading_below, pading_above)


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

    return flatten(input_node, axis)


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
        return ng_inputs[0]

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

    out_shape = [data.shape[i] for i in range(len(data.shape)) if i not in axes_to_squeeze]
    return ng.reshape(data, out_shape)


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

    return ng.reshape(data, out_shape)


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

    output_shape = infer_dimensions(onnx_node.name, data.shape, output_shape)
    return ng.reshape(data, output_shape)


def Split(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> Tuple[NgraphNode, ...]
    """Split a tensor into a list of tensors."""
    data = ng_inputs[0]
    count_outputs = len(onnx_node.get_output_names())
    axis_to_split = onnx_node.get_attribute_value('axis', 0)

    if axis_to_split < 0:
        # Cover Python negative indexing
        axis_to_split = len(data.shape) + axis_to_split
    elif axis_to_split >= len(data.shape):
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


def DepthToSpace(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Rearranges (permutes) input tensor data from depth into blocks of spatial data.

    Values from the depth dimension (assuming NCHW layout) are moved in spatial blocks to the
    height and width dimensions.

    :param onnx_node: The ONNX node representing this operation.
    :param ng_inputs: The input tensors.
    :return: Tensor with shape [N, C/(blocksize * blocksize), H * blocksize, W * blocksize].
    """
    data = ng_inputs[0]
    block_size = onnx_node.get_attribute_value('blocksize')
    if block_size is None:
        raise ValueError('DepthToSpace node (%s): missing required attribute \"blocksize\"',
                         onnx_node.name)
    # Set default values to each dimension to be able to work with 3D or 4D data.
    n, c, h, w = 1, 1, 1, 1
    if len(data.shape) == 4:
        n, c, h, w, = data.shape
    elif len(data.shape) == 3:
        c, h, w = data.shape
    else:
        raise ValueError('DepthToSpace node (%s): the provided tensor shape (%s) is not supported',
                         onnx_node.name, str(data.shape))
    # First we have to disperse the data from depth channel, then rearrange them so as appropriate
    # chunks of data where close to their destination place. Finally squeeze data from
    # respective dimensions.
    flat_node = ng.reshape(data, [n, block_size, block_size, c // (block_size ** 2), h, w])
    flat_node = reorder_axes(flat_node, [0, 3, 4, 1, 5, 2])
    return ng.reshape(flat_node, [n, c // (block_size ** 2), h * block_size, w * block_size])


# Misc
def Constant(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Produce a constant tensor."""
    value_tensor = onnx_node.get_attribute_value('value')
    return ng.constant(value_tensor.to_array())


def BatchNormalization(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Carry out batch normalization."""
    x, scale, bias, mean, var = ng_inputs

    is_test = onnx_node.get_attribute_value('is_test', 1)
    spatial = onnx_node.get_attribute_value('spatial', 1)
    epsilon = onnx_node.get_attribute_value('epsilon', 1e-5)

    # @TODO: Implement learning mode support
    # momentum = onnx_node.get_attribute_value('momentum', 0.99)

    if not is_test:
        raise NotImplementedError('BatchNormalization node (%s): only `is_test` mode is currently '
                                  'supported.', onnx_node.name)
    if not spatial:
        raise NotImplementedError('BatchNormalization node (%s): only `spatial` mode is currently '
                                  'supported.', onnx_node.name)

    return ng.batch_norm(epsilon, scale, bias, x, mean, var)


def LRN(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Carry out Local Region Normalization.

    :param onnx_node: The ONNX node representation of LRN.
    :param ng_inputs: The input data node.
    :return: LRN output node.
    """
    data = ng_inputs[0]

    alpha = onnx_node.get_attribute_value('alpha', 1e-4)
    beta = onnx_node.get_attribute_value('beta', 0.75)
    bias = onnx_node.get_attribute_value('bias', 1.0)
    size = onnx_node.get_attribute_value('size')

    if size is None:
        raise ValueError('LRN node (%s): required `size` attribute is missing', onnx_node.name)

    return ng.lrn(data, alpha, beta, bias, size)


def Shape(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Return input shape."""
    # Dtype int64 is required for ONNX unit tests.
    return ng.constant(ng_inputs[0].shape, dtype=np.int64)


def Size(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Return input size."""
    # Dtype int64 is required for ONNX unit tests.
    return ng.constant(flatten(ng_inputs[0], 0).shape[1], dtype=np.int64)
