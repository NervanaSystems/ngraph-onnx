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

import onnx.backend.test
import pytest

from ngraph_onnx.onnx_importer.backend import NgraphBackend

# Set backend device name to be used instead of hardcoded by ONNX BackendTest class ones.
NgraphBackend.backend_name = pytest.config.getoption('backend', default='CPU')

# This is a pytest magic variable to load extra plugins
# Uncomment the line below to enable the ONNX compatibility report
# pytest_plugins = 'onnx.backend.test.report',

# import all test cases at global scope to make them visible to python.unittest
backend_test = onnx.backend.test.BackendTest(NgraphBackend, __name__)


# Need refactoring to ngraph++
backend_test.exclude('test_gather')
backend_test.exclude('test_hardmax')
backend_test.exclude('test_pow')
backend_test.exclude('test_top_k')
backend_test.exclude('test_Upsample_nearest_scale_2d')

backend_test.exclude('test_depthtospace_cpu')
backend_test.exclude('test_depthtospace_example_cpu')

backend_test.exclude('test_split_equal_parts')
backend_test.exclude('test_split_variable_parts')
backend_test.exclude('test_upsample')

backend_test.exclude('test_Embedding')
backend_test.exclude('test_Linear')
backend_test.exclude('test_MaxPool1d')
backend_test.exclude('test_PixelShuffle')
backend_test.exclude('test_ReflectionPad2d')
backend_test.exclude('test_ReplicationPad2d')
backend_test.exclude('test_ZeroPad')
backend_test.exclude('test_constant_pad')
backend_test.exclude('test_edge_pad')
backend_test.exclude('test_matmul_3d')
backend_test.exclude('test_matmul_4d')
backend_test.exclude('test_default_axes')

# Pad tests
backend_test.exclude('test_edge_pad_cpu')
backend_test.exclude('test_reflect_pad_cpu')
backend_test.exclude('test_ReflectionPad2d_cpu')
backend_test.exclude('test_ReplicationPad2d_cpu')

# Casting tests
backend_test.exclude('test_cast_FLOAT_to_FLOAT16_cpu')
backend_test.exclude('test_cast_FLOAT_to_DOUBLE_cpu')
backend_test.exclude('test_cast_FLOAT16_to_FLOAT_cpu')
backend_test.exclude('test_cast_FLOAT16_to_DOUBLE_cpu')
backend_test.exclude('test_cast_DOUBLE_to_FLOAT_cpu')
backend_test.exclude('test_cast_DOUBLE_to_FLOAT16_cpu')

# BatchNorm with shapes to other than 4
backend_test.exclude('test_BatchNorm1d_3d_input')
backend_test.exclude('test_BatchNorm3d')
backend_test.exclude('test_BatchNorm3d_momentum')

# big models tests
# Passing
backend_test.exclude('test_resnet50')
backend_test.exclude('test_vgg19')
backend_test.exclude('test_squeezenet')

# Failing

# Validation Error: Input index 3 must be set to consumed for operator BatchNormalization
backend_test.exclude('test_densenet121')
backend_test.exclude('test_inception_v2')
backend_test.exclude('test_shufflenet')

# Validation Error: Input size 2 not in range [min=1, max=1].
backend_test.exclude('test_inception_v1')

# NotImplementedError: 'LRN'
backend_test.exclude('test_bvlc_alexnet')
backend_test.exclude('test_vgg16')


globals().update(backend_test.enable_report().test_cases)
