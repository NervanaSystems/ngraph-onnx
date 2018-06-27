from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
import onnx.mapping

from typing import Any

TENSOR_TYPE_STR_TO_NP_TYPE = {
    'UINT8': np.dtype('uint8'),
    'UINT16': np.dtype('uint16'),
    'UINT32': np.dtype('uint32'),
    'UINT64': np.dtype('uint64'),
    'INT8': np.dtype('int8'),
    'INT16': np.dtype('int16'),
    'INT32': np.dtype('int32'),
    'INT64': np.dtype('int64'),
    'FLOAT16': np.dtype('float16'),
    'FLOAT': np.dtype('float32'),
    'DOUBLE': np.dtype('float64'),
    'COMPLEX64': np.dtype('complex64'),
    'COMPLEX128': np.dtype('complex128'),
    'BOOL': np.dtype('bool'),
}

NP_TYPE_TO_TENSOR_TYPE_STR = {v: k for k, v in TENSOR_TYPE_STR_TO_NP_TYPE.items()}


def onnx_tensor_type_to_numpy_type(data_type):
    # type: (Any) -> np.dtype
    """Return ONNX TensorProto type mapped into numpy dtype.

    :param data_type: The type we want to convert from.
    :return: Converted numpy dtype.
    """
    if type(data_type) is int:
        return onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[data_type]
    elif type(data_type) is str:
        return TENSOR_TYPE_STR_TO_NP_TYPE[data_type]
    else:
        raise ValueError('Unsupported data type representation (%s).', str(type(data_type)))
