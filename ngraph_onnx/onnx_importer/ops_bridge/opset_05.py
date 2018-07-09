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

from ngraph_onnx.onnx_importer.ops_bridge.opset_04 import Reshape as Reshape_v4
from ngraph_onnx.onnx_importer.ops_bridge.opset_04 import *  # noqa

import logging
logger = logging.getLogger(__name__)


def Reshape(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Reshape the input tensor similar to numpy.reshape.

    At most one dimension of the new shape can be -1. In this case, the value is inferred from
    the size of the tensor and the remaining dimensions. A dimension could also be 0, in which
    case the actual dimension value is going to be copied from the shape argument.
    """
    logger.warning('Reshape node (%s) - dynamic output shape is not fully supported yet',
                   onnx_node.name)
    return Reshape_v4(onnx_node, ng_inputs)
