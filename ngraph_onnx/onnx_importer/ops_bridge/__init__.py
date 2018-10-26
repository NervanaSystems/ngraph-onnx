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

from __future__ import print_function
from __future__ import division

import logging
from importlib import import_module
from types import ModuleType
from typing import Tuple

from ngraph_onnx import TYPE_CHECKING

from ngraph.impl import Node as NgraphNode

if TYPE_CHECKING:
    from ngraph_onnx.onnx_importer.model_wrappers import NodeWrapper

logger = logging.getLogger(__name__)


def make_ng_nodes(node_factory, onnx_node):  # type: (ModuleType, NodeWrapper) -> Tuple[NgraphNode]
    """Create ngraph output Ops for an ONNX node."""
    op_type = onnx_node.op_type

    try:
        ng_node_factory_function = getattr(node_factory, op_type)
    except AttributeError:
        raise NotImplementedError('Unknown operation: %s', op_type)

    ng_inputs = onnx_node.get_ng_inputs()
    ng_outputs = ng_node_factory_function(onnx_node, ng_inputs)

    if type(ng_outputs) != tuple:
        ng_outputs = (ng_outputs,)

    return ng_outputs


def get_node_factory(opset_version: int = None) -> ModuleType:
    """Import the factory module which converts ops to nGraph nodes.

    :param opset_version: specify `ai.onnx` operator set version
    :return: module with factory functions for all ops in op set
    """
    fallback_module = import_module('ngraph_onnx.onnx_importer.ops_bridge.opset_latest')
    if opset_version is None:
        return fallback_module

    opset_module_name = 'ngraph_onnx.onnx_importer.ops_bridge.opset_{:02d}'.format(opset_version)
    try:
        return import_module(opset_module_name)
    except ImportError:
        latest_opset = fallback_module.LATEST_SUPPORTED_OPSET_VERSION  # type: ignore
        logger.warning('ONNX `ai.onnx` opset version %s is not supported. '
                       'Falling back to latest supported version: %s',
                       opset_version, latest_opset)
        return fallback_module
