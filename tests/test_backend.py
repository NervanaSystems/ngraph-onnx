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

from ngraph_onnx.onnx_importer.backend import NgraphBackend

# This is a pytest magic variable to load extra plugins
# Uncomment the line below to enable the ONNX compatibility report
# pytest_plugins = 'onnx.backend.test.report',

# import all test cases at global scope to make them visible to python.unittest
backend_test = onnx.backend.test.BackendTest(NgraphBackend, __name__)


# Need refactoring to ngraph++
# Casting tests -> NC5-159
backend_test.exclude('test_cast_DOUBLE_to_FLOAT16_cpu')
backend_test.exclude('test_cast_DOUBLE_to_FLOAT_cpu')
backend_test.exclude('test_cast_FLOAT16_to_DOUBLE_cpu')
backend_test.exclude('test_cast_FLOAT16_to_FLOAT_cpu')
backend_test.exclude('test_cast_FLOAT_to_DOUBLE_cpu')
backend_test.exclude('test_cast_FLOAT_to_FLOAT16_cpu')

# Gather tests -> NC5-160
backend_test.exclude('test_Embedding')
backend_test.exclude('test_Embedding_sparse')
backend_test.exclude('test_gather_0')
backend_test.exclude('test_gather_1')

# Hardmax tests -> NGRAPH-1839
backend_test.exclude('test_hardmax_axis_0')
backend_test.exclude('test_hardmax_axis_1')
backend_test.exclude('test_hardmax_axis_2')
backend_test.exclude('test_hardmax_default_axis')
backend_test.exclude('test_hardmax_example')
backend_test.exclude('test_hardmax_one_hot')

# Matmul tests -> NGRAPH-1838
backend_test.exclude('test_matmul_3d')
backend_test.exclude('test_matmul_4d')

# Misc tests
backend_test.exclude('test_top_k')  # -> NC5-161
backend_test.exclude('test_Upsample_nearest_scale_2d')  # -> NC5-162

# Pad tests -> NGRAPH-1505
backend_test.exclude('test_edge_pad_cpu')
backend_test.exclude('test_reflect_pad_cpu')
backend_test.exclude('test_ReflectionPad2d_cpu')
backend_test.exclude('test_ReplicationPad2d_cpu')

# Pow tests -> NC5-163
backend_test.exclude('test_pow_bcast_axis0')
backend_test.exclude('test_pow_bcast')
backend_test.exclude('test_pow')
backend_test.exclude('test_pow_example')

# Passing topologies:
# backend_test.exclude('test_resnet50')
# backend_test.exclude('test_vgg19')
# backend_test.exclude('test_squeezenet')

# Failing topologies:

# Validation Error: Input index 3 must be set to consumed for operator BatchNormalization
backend_test.exclude('test_densenet121')
backend_test.exclude('test_inception_v2')
backend_test.exclude('test_shufflenet')

# Validation Error: Input size 2 not in range [min=1, max=1].
backend_test.exclude('test_inception_v1')

# NotImplementedError: 'LRN' -> NGRAPH-1731
backend_test.exclude('test_bvlc_alexnet')
backend_test.exclude('test_vgg16')


globals().update(backend_test.enable_report().test_cases)
