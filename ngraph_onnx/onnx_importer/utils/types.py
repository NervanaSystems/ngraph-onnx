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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE, TENSOR_TYPE_TO_NP_TYPE
from onnx import TensorProto

import ngraph as ng
from ngraph.impl import Node as NgraphNode
from ngraph.impl import Type as NgraphType
from ngraph.utils.types import get_dtype

from typing import Any, Tuple
logger = logging.getLogger(__name__)


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


def np_dtype_to_tensor_type(data_type):  # type: (np.type) -> int
    """Return TensorProto type for provided numpy dtype.

    :param data_type: Numpy data type object.
    :return: TensorProto.DataType enum value for corresponding type.
    """
    return NP_TYPE_TO_TENSOR_TYPE[data_type]


def get_bool_nodes(nodes):   # type: (Tuple[NgraphNode, ...]) -> Tuple[NgraphNode, ...]
    """Convert each input node to bool data type if necessary.

    :param nodes: Input nodes to be converted.
    :return: Converted nodes.
    """
    bool_nodes = []
    for node in nodes:
        if not node.get_element_type() == NgraphType.boolean:
            bool_nodes.append(ng.convert(node, bool))
            logger.warning('Converting node of type: <{}> to bool.'.format(get_dtype(
                node.get_element_type())))
        else:
            bool_nodes.append(node)

    return tuple(bool_nodes)
