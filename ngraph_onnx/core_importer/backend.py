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
from cachetools.func import lru_cache

from onnx.helper import make_tensor_value_info, make_graph, make_model
from onnx.backend.base import Backend, BackendRep
from typing import Any, Dict, List, Optional, Sequence, Text, Tuple

from ngraph.impl import Function
from ngraph_onnx.core_importer.importer import import_onnx_model
from ngraph_onnx.core_importer.utils.types import np_dtype_to_tensor_type


class NgraphBackend(Backend):
    """Takes an ONNX model with inputs, perform a computation, and then return the output."""

    # The requested (nGraph) backend to be used instead of hardcoded by ONNX test Runner.
    backend_name = 'CPU'  # type: str

    _ngraph_onnx_device_map = [
        # (<ngraph_backend_name>, <onnx_device_name>)
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
        ng_model_functions = import_onnx_model(onnx_model)
        return NgraphBackendRep(ng_model_functions, cls.backend_name)

    @classmethod
    def _get_onnx_device_name(cls, ngraph_device_name):  # type: (str) -> Optional[str]
        return next((onnx_device for (ng_device, onnx_device) in cls._ngraph_onnx_device_map
                     if ngraph_device_name == ng_device), None)

    @classmethod
    @lru_cache(maxsize=16)
    def supports_ngraph_device(cls, ngraph_device_name):  # type: (str) -> bool
        """Check whether particular nGraph device is supported by current nGraph library.

        :param ngraph_device_name: Name of nGraph device.
        :return: True if current nGraph library supports ngraph_device_name.
        """
        try:
            ng.runtime(backend_name=ngraph_device_name)
        except RuntimeError as e:
            # Catch error raised when backend isn't available:
            # 'Backend {ngraph_device_name} not found in registered backends'
            if str(ngraph_device_name) in str(e) and 'not found' in str(e):
                return False
            else:
                raise e
        return True

    @classmethod
    def supports_device(cls, onnx_device_name):  # type: (str) -> bool
        """Check whether the requested nGraph backend supports a particular ONNX device.

        During running ONNX backend tests this function is called on each item of ONNX defined
        devices list. Currently this list is hardcoded and contains only two entries:
         ('CPU', 'CUDA'). In order to check whether the requested nGraph backend stored as
         NgraphBackend class variable we have to map its name into ONNX device namespace and then
         verify whether the current version of nGraph library supports it.

        :param onnx_device_name: One of ONNX defined devices.
        :return: True if ONNX device is supported, otherwise False.
        """
        requested_backend_name_mapped_to_onnx_device = cls._get_onnx_device_name(cls.backend_name)
        # Check whether:
        # 1. There is mapping between onnx_device_name and requested nGraph backend to run tests on.
        # 2. Current nGraph version supports requested backend.
        return (onnx_device_name == requested_backend_name_mapped_to_onnx_device
                and cls.supports_ngraph_device(cls.backend_name))

    @classmethod
    def run_model(cls, onnx_model, inputs, device='CPU', **kwargs):
        # type: (onnx.ModelProto, List[np.ndarray], str, Dict) -> List[np.ndarray]
        """Prepare and run a computation on an ONNX model."""
        return cls.prepare(onnx_model, device, **kwargs).run(inputs)

    @classmethod
    def run_node(cls,
                 node,  # type: onnx.NodeProto
                 inputs,  # type: List[np.ndarray]
                 device='CPU',  # type: Text
                 outputs_info=None,  # type: Optional[Sequence[Tuple[np.dtype, Tuple[int, ...]]]]
                 **kwargs  # type: Any
                 ):  # type: (...) -> List[np.ndarray]
        """Prepare and run a computation on an ONNX node."""
        # default values for input/output tensors
        input_tensor_types = [np_dtype_to_tensor_type(node_input.dtype) for node_input in inputs]
        output_tensor_types = [onnx.TensorProto.FLOAT for idx in range(len(node.output))]
        output_tensor_shapes = [()]  # type: List[Tuple[int, ...]]

        if outputs_info is not None:
            output_tensor_types = [np_dtype_to_tensor_type(dtype) for (dtype, shape) in
                                   outputs_info]
            output_tensor_shapes = [shape for (dtype, shape) in outputs_info]

        input_tensors = [make_tensor_value_info(name, tensor_type, value.shape)
                         for name, value, tensor_type in zip(node.input, inputs,
                                                             input_tensor_types)]
        output_tensors = [make_tensor_value_info(name, tensor_type, shape)
                          for name, shape, tensor_type in zip(node.output, output_tensor_shapes,
                                                              output_tensor_types)]

        graph = make_graph([node], 'compute_graph', input_tensors, output_tensors)
        model = make_model(graph, producer_name='NgraphBackend')
        if 'opset_version' in kwargs:
            model.opset_import[0].version = kwargs['opset_version']
        return cls.prepare(model, device).run(inputs)


class NgraphBackendRep(BackendRep):
    """A handle which Backend returns after preparing to execute a model repeatedly."""

    def __init__(self, ng_model_functions, device='CPU'):  # type: (List[Function], str) -> None
        super(NgraphBackendRep, self).__init__()
        self.device = self._get_ngraph_device_name(device)
        self.ng_model_functions = ng_model_functions
        self.runtime = ng.runtime(backend_name=self.device)
        self.computations = [self.runtime.computation(ng_function) for ng_function in ng_model_functions]

    def run(self, inputs, **kwargs):  # type: (List[np.ndarray], Dict) -> List[np.ndarray]
        """Execute computation on the backend representation of model."""
        return [computation(*inputs) for computation in self.computations]

    def _get_ngraph_device_name(self, onnx_device):  # type: (str) -> str
        return 'GPU' if onnx_device == 'CUDA' else onnx_device
