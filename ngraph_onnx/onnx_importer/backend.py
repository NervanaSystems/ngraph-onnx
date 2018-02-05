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
"""
ONNX Backend implementation.

See ONNX documentation for details:
https://github.com/onnx/onnx/blob/master/docs/Implementing%20an%20ONNX%20backend.md
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import ngraph_api as ng
import onnx

from onnx.helper import make_tensor_value_info, make_graph, make_model
from onnx.backend.base import Backend, BackendRep
from typing import Dict, List

from ngraph_onnx.onnx_importer.importer import import_onnx_model


class NgraphBackend(Backend):
    """Takes an ONNX model with inputs, perform a computation, and then return the output."""

    @classmethod
    def prepare(cls, onnx_model, device='CPU', **kwargs):
        # type: (onnx.ModelProto, str, Dict) -> NgraphBackendRep
        """Prepare backend representation of ONNX model."""
        super(NgraphBackend, cls).prepare(onnx_model, device, **kwargs)
        ng_model = import_onnx_model(onnx_model)[0]
        return NgraphBackendRep(ng_model, device)

    @classmethod
    def supports_device(cls, device):  # type: (str) -> bool
        """Check whether the backend supports a particular device."""
        return device == 'CPU'

    @classmethod
    def run_model(cls, onnx_model, inputs, device='CPU', **kwargs):
        # type: (onnx.ModelProto, List[np.ndarray], str, Dict) -> List[np.ndarray]
        """Prepare and run a computation on an ONNX model."""
        return cls.prepare(onnx_model, device, **kwargs).run(inputs)

    @classmethod
    def run_node(cls, onnx_node, inputs, device='CPU'):
        # type: (onnx.NodeProto, List[np.ndarray], str) -> List[np.ndarray]
        """Prepare and run a computation on an ONNX node."""
        input_tensors = [make_tensor_value_info(name, onnx.TensorProto.FLOAT, value.shape)
                         for name, value in zip(onnx_node.input, inputs)]
        output_tensors = [make_tensor_value_info(name, onnx.TensorProto.FLOAT, value.shape)
                          for name, value in zip(onnx_node.output, ())]  # type: ignore

        graph = make_graph([onnx_node], 'compute_graph', input_tensors, output_tensors)
        model = make_model(graph, producer_name='NgraphBackend')
        return cls.prepare(model).run(inputs)


class NgraphBackendRep(BackendRep):
    """A handle which Backend returns after preparing to execute a model repeatedly."""

    def __init__(self, ng_model, device='CPU'):  # type: (Dict, str) -> None
        super(NgraphBackendRep, self).__init__()
        self.device = device
        self.model = ng_model
        self.runtime = ng.runtime()
        self.computation = self.runtime.computation(ng_model['output'], *ng_model['inputs'])

    def run(self, inputs, **kwargs):  # type: (List[np.ndarray], Dict) -> List[np.ndarray]
        """Execute computation on the backend representation of model."""
        outputs = self.computation(*inputs)
        return [outputs]
