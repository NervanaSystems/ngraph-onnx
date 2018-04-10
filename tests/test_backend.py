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
backend_test.exclude('test_clip')
backend_test.exclude('test_gather')
backend_test.exclude('test_hardmax')
backend_test.exclude('test_max_one_input')
backend_test.exclude('test_min_one_input')
backend_test.exclude('test_pow')
backend_test.exclude('test_shape')
backend_test.exclude('test_size')
backend_test.exclude('test_sum_one_input')
backend_test.exclude('test_top_k')
backend_test.exclude('test_Upsample_nearest_scale_2d')
backend_test.exclude('test_operator_chunk')
backend_test.exclude('test_operator_clip')
backend_test.exclude('test_operator_flatten')
backend_test.exclude('test_operator_maxpool')
backend_test.exclude('test_operator_permute2')
backend_test.exclude('test_operator_view')
backend_test.exclude('test_depthtospace_cpu')
backend_test.exclude('test_depthtospace_example_cpu')


backend_test.exclude('test_Embedding')
backend_test.exclude('test_GLU')
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

# Reshape tests
backend_test.exclude('test_reshape_extended_dims_cpu')
backend_test.exclude('test_reshape_negative_dim_cpu')
backend_test.exclude('test_reshape_one_dim_cpu')
backend_test.exclude('test_reshape_reduced_dims_cpu')
backend_test.exclude('test_reshape_reordered_dims_cpu')

# Casting tests
backend_test.exclude('test_cast_FLOAT_to_FLOAT16_cpu')
backend_test.exclude('test_cast_FLOAT_to_DOUBLE_cpu')
backend_test.exclude('test_cast_FLOAT16_to_FLOAT_cpu')
backend_test.exclude('test_cast_FLOAT16_to_DOUBLE_cpu')
backend_test.exclude('test_cast_DOUBLE_to_FLOAT_cpu')
backend_test.exclude('test_cast_DOUBLE_to_FLOAT16_cpu')

# Convolution tests
backend_test.exclude('test_Conv2d_depthwise_cpu')
backend_test.exclude('test_Conv2d_depthwise_padded_cpu')
backend_test.exclude('test_Conv2d_depthwise_strided_cpu')
backend_test.exclude('test_Conv2d_depthwise_with_multiplier_cpu')
backend_test.exclude('test_Conv2d_groups_cpu')
backend_test.exclude('test_Conv2d_groups_thnn_cpu')
backend_test.exclude('test_Conv3d_groups_cpu')

# big models tests
# passing
backend_test.exclude('test_resnet50')

# failing
backend_test.exclude('test_bvlc_alexnet')
backend_test.exclude('test_densenet121')
backend_test.exclude('test_inception_v1')
backend_test.exclude('test_inception_v2')
backend_test.exclude('test_shufflenet')
backend_test.exclude('test_squeezenet')
backend_test.exclude('test_vgg16')
backend_test.exclude('test_vgg19')

globals().update(backend_test.enable_report().test_cases)
