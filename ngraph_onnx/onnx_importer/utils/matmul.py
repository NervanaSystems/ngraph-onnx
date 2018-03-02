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

from ngraph_api.utils.types import TensorShape


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
    # FIXME: C901 too comples function
    """Check wheter input tensors have compatible shapes to multiply A @ B.

    Shape requirements are defined by NumPy.matmul function. They boil down to:
    * If both arguments shapes are <= 2D standard matrix multiplication rules apply.
    * If at least one argument rank is greater than 2, it is treated as stack of matrices and,
        *  matrix multiplication takes place in last two dimensions.
        *  standard Python broadcasting rules apply to stack of matrices, where matrices are
            treated as elements.

    There is a great article with clear description of broadcasting rules:
    https://eli.thegreenplace.net/2015/broadcasting-arrays-in-numpy/#id2

    :param node_a: First tensor we want to multiply.
    :param node_b: Second tensor we want to multiply.
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

    # update ranks
    rank_a = len(shape_a)
    rank_b = len(shape_b)
    max_dim = max(rank_a, rank_b)

    # left-pad A's shape with 1s until it also has p dimensions
    if rank_a < max_dim:
        for idx in range(max_dim - rank_a):
            shape_a.insert(0, 1)
    # left-pad B's shape with 1s until is also has p dimensions
    elif rank_b < max_dim:
        for idx in range(max_dim - rank_b):
            shape_b.insert(0, 1)

    for idx in range(max_dim - 1, -1, -1):
        idx_dim_a = shape_a[idx]
        idx_dim_b = shape_b[idx]
        if idx_dim_a != 1 and idx_dim_b != 1 and idx_dim_a != idx_dim_b:
            return False

    return True
