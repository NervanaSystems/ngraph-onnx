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

from pyngraph import Node as NgraphNode

import ngraph_api as ng


def transpose(node):  # type: (NgraphNode) -> NgraphNode
    """Return transposed tensor.

    :param node: Input tensor we want to transpose
    """
    axes_order = list(range(len(node.shape)))
    axes_order.reverse()
    out_shape = list(node.shape)
    out_shape.reverse()
    node = ng.reshape(node, axes_order, out_shape)
    return node
