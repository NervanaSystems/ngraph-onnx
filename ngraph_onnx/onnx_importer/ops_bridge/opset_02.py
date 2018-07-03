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

# TODO: GlobalLpPool-2
# TODO: LpPool-2
# Split-2               supported by opset_01.Split

from __future__ import print_function
from __future__ import division

from typing import Tuple, List

import ngraph as ng

from ngraph_onnx import TYPE_CHECKING
from ngraph.utils.types import get_dtype
from ngraph.impl import Node as NgraphNode
from ngraph_onnx.onnx_importer.utils.misc import split_pads_into_pairs

if TYPE_CHECKING:
    from ngraph_onnx.onnx_importer.model_wrappers import NodeWrapper

from ngraph_onnx.onnx_importer.ops_bridge.opset_01 import *  # noqa


def Pad(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Add padding to the input tensor."""
    pads = onnx_node.get_attribute_value('pads')

    if pads is None:
        raise ValueError('Pad node (s%): pads attribute is required.', onnx_node.name)

    constant = 'constant'
    mode = onnx_node.get_attribute_value('mode', constant)  # 'constant', 'reflect' or 'edge'
    value = onnx_node.get_attribute_value('value', 0.)

    if len(pads) != 2 * len(ng_inputs[0].shape):
        raise ValueError('Pad node (%s): \'pads rank (%d) should be double of input tensor '
                         'rank (%d).', onnx_node.name, len(pads), len(ng_inputs[0].shape))

    # check for negative pads values
    if any(map(lambda x: x < 0, pads)):
        raise NotImplementedError('Pad node (%s): removing padding elements is not supported yet.',
                                  onnx_node.name)
    if mode != constant:
        raise NotImplementedError('Pad node (%s): only constant padding is supported.', onnx_node.name)

    # Split pads into pairs for each axis
    pading_below, pading_above = split_pads_into_pairs(pads)
    return ng.pad(ng_inputs[0], ng.constant(value,
                  dtype=get_dtype(ng_inputs[0].get_element_type())), pading_below, pading_above)
