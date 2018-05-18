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
import ngraph as ng
import onnx

from onnx.helper import make_tensor_value_info, make_graph, make_model
from onnx.backend.base import Backend, BackendRep
from typing import Dict, List, Optional

from ngraph_onnx.onnx_importer.importer import import_onnx_model


class NgraphBackend(Backend):
    """Takes an ONNX model with inputs, perform a computation, and then return the output."""

    _ngraph_supported_devices = []  # type: List[str]
    # The backend (nGraph) to be used instead of hardcoded by ONNX test Runner.
    backend_name = None  # type: str

    _ngraph_onnx_device_map = [
        # (<ngraph_dev_name>, <onnx_dev_name>)
        ('CPU', 'CPU'),
        ('GPU', 'CUDA'),
        ('INTERPRETER', 'CPU'),
        ('ARGON', 'CPU'),
        ('NNP', 'CPU'),
    ]

    @classmethod
    def prepare(cls, onnx_model, device='CPU', **kwargs):
        # type: (onnx.ModelProto, str, Dict) -> NgraphBackendRep
        """Prepare backend representation of ONNX model."""
        super(NgraphBackend, cls).prepare(onnx_model, device, **kwargs)
        ng_model = import_onnx_model(onnx_model)
        return NgraphBackendRep(ng_model, cls.backend_name)

    @classmethod
    def _get_supported_devices(cls):  # type: () -> None
        cls._ngraph_supported_devices = ng.impl.runtime.Backend.get_registered_devices()

    @classmethod
    def _get_onnx_device_name(cls, ngraph_device_name):  # type: (str) -> Optional[str]
        return next((onnx_device for (ng_device, onnx_device) in cls._ngraph_onnx_device_map
                     if ngraph_device_name == ng_device), None)

    @classmethod
    def supports_device(cls, device):  # type: (str) -> bool
        """Check whether the backend supports a particular device."""
        if len(cls._ngraph_supported_devices) == 0:
            cls._get_supported_devices()
        onnx_devcie_name = cls._get_onnx_device_name(cls.backend_name)
        # Unknown device - there is no mapping for respective onnx device.
        if not onnx_devcie_name:
            return False
        # Check whether:
        # 1. The device is requested backend to run tests on.
        # 2. Current nGraph version supports requested backend.
        return (onnx_devcie_name == device and
                cls.backend_name in cls._ngraph_supported_devices)

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
        return cls.prepare(model, device).run(inputs)


class NgraphBackendRep(BackendRep):
    """A handle which Backend returns after preparing to execute a model repeatedly."""

    def __init__(self, ng_model, device='CPU'):  # type: (List[Dict], str) -> None
        super(NgraphBackendRep, self).__init__()
        self.device = device
        self.model = ng_model
        self.runtime = ng.runtime(backend_name=self.device)
        self.computations = [self.runtime.computation(model['output'], *model['inputs']) for
                             model in ng_model]

    def run(self, inputs, **kwargs):  # type: (List[np.ndarray], Dict) -> List[np.ndarray]
        """Execute computation on the backend representation of model."""
        return [computation(*inputs) for computation in self.computations]
