from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE, TENSOR_TYPE_TO_NP_TYPE
from onnx import TensorProto

from typing import Any


def onnx_tensor_type_to_numpy_type(data_type):  # type: (Any) -> np.dtype
    """Return ONNX TensorProto type mapped into numpy dtype.

    :param data_type: The type we want to convert from.
    :return: Converted numpy dtype.
    """
    if type(data_type) is int:
        return TENSOR_TYPE_TO_NP_TYPE[data_type]
    elif type(data_type) is str:
        return TENSOR_TYPE_TO_NP_TYPE[TensorProto.DataType.Value(data_type)]
    else:
        raise ValueError('Unsupported data type representation (%s).', str(type(data_type)))


def np_dtype_to_tensor_type_name(data_type):  # type: (np.dtype) -> str
    """Return TensorProto type name respective to provided numpy dtype.

    :param data_type: Numpy dtype we want to convert.
    :return: String representation of TensorProto type name.
    """
    return TensorProto.DataType.Name(NP_TYPE_TO_TENSOR_TYPE[data_type])
