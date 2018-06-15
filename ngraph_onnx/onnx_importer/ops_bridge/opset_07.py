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

import ngraph as ng

from ngraph_onnx.onnx_importer.ops_bridge.opset_06 import *  # noqa
from ngraph_onnx.onnx_importer.utils.binary import numpy_style_broadcast_for_binary_operation


def Add(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Perform element-wise binary addition with numpy-style broadcasting."""
    left, right = numpy_style_broadcast_for_binary_operation(onnx_node, ng_inputs)
    return ng.add(left, right)
