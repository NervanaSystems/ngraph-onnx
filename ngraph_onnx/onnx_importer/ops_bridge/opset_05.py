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
from ngraph_onnx.onnx_importer.ops_bridge.opset_04 import *  # noqa

import ngraph as ng
from ngraph_onnx.onnx_importer.utils.reshape import infer_dimensions


def Reshape(onnx_node, ng_inputs):  # type: (NodeWrapper, List[NgraphNode]) -> NgraphNode
    """Reshape the input tensor similar to numpy.reshape.

    At most one dimension of the new shape can be -1. In this case, the value is inferred from
    the size of the tensor and the remaining dimensions. A dimension could also be 0, in which
    case the actual dimension value is going to be copied from the shape argument.
    """
    data = ng_inputs[0]
    output_shape = ng_inputs[1]
    # Be input data type agnostic as long as it has correct interface.
    if hasattr(output_shape, 'get_data'):
        output_shape = output_shape.get_data().tolist()
    else:
        raise NotImplementedError('Reshape node (%s) doesn\'t support shape input of other type '
                                  'than Constant.', onnx_node.name)

    if output_shape == data.shape:
        return data

    output_shape = infer_dimensions(onnx_node.name, data.shape, output_shape)
    return ng.reshape(data, output_shape)
