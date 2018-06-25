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
from typing import Tuple

from ngraph.utils.types import TensorShape
from ngraph.impl import Node as NgraphNode

from ngraph_onnx import TYPE_CHECKING
from ngraph_onnx.onnx_importer.utils.reshape import flatten
from ngraph_onnx.onnx_importer.utils.binary import numpy_style_broadcast_output_shape

if TYPE_CHECKING:
    from ngraph_onnx.onnx_importer.model_wrappers import NodeWrapper


def _is_matrix(shape):  # type: (TensorShape) -> bool
    """Check if tensor is a 2D matrix.

    :param shape: The shape of tensor to check.
    """
    return len(shape) == 2


def _is_vector(shape):  # type: (TensorShape) -> bool
    """Check if tensor is a 1D vector.

    :param shape: The shape of tensor to check.
    """
    return len(shape) == 1


def has_matmul_compatible_shapes(shape_a, shape_b):  # noqa: C901
    # type: (TensorShape, TensorShape) -> bool
    # FIXME: C901 function too complex
    """Check wheter input tensors have compatible shapes to multiply A @ B.

    Shape requirements are defined by NumPy.matmul function. They boil down to:
    * If both arguments shapes are <= 2D standard matrix multiplication rules apply.
    * If at least one argument rank is greater than 2, it is treated as stack of matrices and,
        *  matrix multiplication takes place in last two dimensions.
        *  standard Python broadcasting rules apply to stack of matrices, where matrices are
            treated as elements.

    There is a great article with clear description of broadcasting rules:
    https://eli.thegreenplace.net/2015/broadcasting-arrays-in-numpy/#id2

    :param shape_a: First tensor we want to multiply.
    :param shape_b: Second tensor we want to multiply.
    """
    shape_a = list(shape_a)
    shape_b = list(shape_b)
    rank_a = len(shape_a)
    rank_b = len(shape_b)

    last_two_dims_a = shape_a[-2:] if rank_a > 2 else shape_a
    last_two_dims_b = shape_b[-2:] if rank_b > 2 else shape_b

    # check last two dimensions whether they are compatible and truncate at
    # most last two dimensions where multiplication takes place
    if _is_matrix(last_two_dims_a) and _is_vector(last_two_dims_b):
        if last_two_dims_a[-1] != last_two_dims_b[0]:
            return False
        else:
            shape_a = shape_a[0:-2]
            shape_b = shape_b[0:-1]
    elif _is_vector(last_two_dims_a) and _is_matrix(last_two_dims_b):
        if last_two_dims_a[0] != last_two_dims_b[0]:
            return False
        else:
            shape_a = shape_a[0:-1]
            shape_b = shape_b[0:-2]
    elif _is_vector(last_two_dims_a) and _is_vector(last_two_dims_b):
        if last_two_dims_a[0] != last_two_dims_b[0]:
            return False
        else:
            shape_a = shape_a[0:-1]
            shape_b = shape_b[0:-1]
    elif _is_matrix(last_two_dims_a) and _is_matrix(last_two_dims_b):
        if last_two_dims_a[-1] != last_two_dims_b[0]:
            return False
        else:
            shape_a = shape_a[0:-2]
            shape_b = shape_b[0:-2]

    if numpy_style_broadcast_output_shape(shape_a, shape_b) is None:
        return False

    return True


def reshape_for_matmul(onnx_node, input_a, input_b):
    # type: (NodeWrapper, NgraphNode, NgraphNode) -> Tuple[NgraphNode, NgraphNode]
    """Adjust input tensor shapes for matrix multiplication.

    This is based on an idea from onnx-tensorflow
    https://github.com/onnx/onnx-tensorflow/blob/17075f44c9071600beccfc62c92b22d1cd957bfd/onnx_tf/backend.py#L711
    They have hardcoded flatten input `A` before transposition.

    :param onnx_node: ONNX node for the matrix multiplication operation
    :param input_a: left side input node
    :param input_b: right side input node
    :return: tuple with input_a and input_b reshaped if needed
    """
    # First we check whether input data have incompatible shapes and then try flatten input data.
    if not has_matmul_compatible_shapes(input_a.shape, input_b.shape):
        input_a = flatten(input_a, 1)  # Flatten ND tensors to 2D matrices
        input_b = flatten(input_b, 1)
        if not has_matmul_compatible_shapes(input_a.shape, input_b.shape):
            raise ValueError('%s node (%s): input "A" and "B" data shapes are incompatible to '
                             'multiply with each other.', onnx_node.op_type, onnx_node.name)
    return input_a, input_b
