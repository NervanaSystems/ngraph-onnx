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

from ngraph_onnx.core_importer.backend import NgraphBackend

# Set backend device name to be used instead of hardcoded by ONNX BackendTest class ones.
selected_backend_name = pytest.config.getoption('backend', default='CPU')
NgraphBackend.backend_name = selected_backend_name

# This is a pytest magic variable to load extra plugins
# Uncomment the line below to enable the ONNX compatibility report
# pytest_plugins = 'onnx.backend.test.report',

# import all test cases at global scope to make them visible to python.unittest
backend_test = onnx.backend.test.BackendTest(NgraphBackend, __name__)

# MaxPool Indices -> NGRAPH-3131
backend_test.exclude('test_maxpool_with_argmax')

# ArgMin/ArgMax -> NC-316
backend_test.exclude('test_argmax')
backend_test.exclude('test_argmin')

# ConvTranspose -> NC-319
backend_test.exclude('test_ConvTranspose2d')
backend_test.exclude('test_convtranspose')
backend_test.exclude('test_operator_convtranspose')

# Padding modes -> NC-322
backend_test.exclude('test_ConstantPad2d')
backend_test.exclude('test_constant_pad')
backend_test.exclude('test_ZeroPad2d')
backend_test.exclude('test_ReflectionPad2d')
backend_test.exclude('test_ReplicationPad2d')
backend_test.exclude('test_edge_pad')
backend_test.exclude('test_reflect_pad')

# RNN -> NC-323
backend_test.exclude('test_simple_rnn')
backend_test.exclude('test_rnn')
backend_test.exclude('test_operator_rnn')

# LSTM -> NC-324
backend_test.exclude('test_lstm')
backend_test.exclude('test_operator_lstm')

# GRU -> NC-325
backend_test.exclude('test_gru')

# Depth to space -> NC-326
backend_test.exclude('test_depthtospace')

# Top-K -> NC-327
backend_test.exclude('test_top_k')

# MeanVarianceNormalization -> NC-328
backend_test.exclude('test_mvn')

# Tests which fail on the CPU backend -> NC-330
if selected_backend_name == 'CPU':
    backend_test.exclude('test_Conv3d_dilated')
    backend_test.exclude('test_Conv3d_dilated_strided')
    backend_test.exclude('test_GLU_dim')

# Big model tests (see test_zoo_models.py):
backend_test.exclude('test_bvlc_alexnet')
backend_test.exclude('test_densenet121')
backend_test.exclude('test_inception_v1')
backend_test.exclude('test_inception_v2')
backend_test.exclude('test_resnet50')
backend_test.exclude('test_shufflenet')
backend_test.exclude('test_squeezenet')
backend_test.exclude('test_vgg19')
backend_test.exclude('test_zfnet512')

# Tests which fail or are very slow on the INTERPRETER backend
if selected_backend_name == 'INTERPRETER':
    backend_test.exclude('test_operator_conv_cpu')
    backend_test.exclude('test_slice_start_out_of_bounds_cpu')

OnnxBackendNodeModelTest = None
OnnxBackendSimpleModelTest = None
OnnxBackendPyTorchOperatorModelTest = None
OnnxBackendPyTorchConvertedModelTest = None
globals().update(backend_test.enable_report().test_cases)


# Non-linear ops -> NC-320
pytest.mark.xfail(OnnxBackendNodeModelTest.test_prelu_example_cpu)

# Matmul ops -> NC5-314
pytest.mark.xfail(OnnxBackendNodeModelTest.test_gemm_broadcast_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_matmul_3d_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_matmul_4d_cpu)

# Trigonometric ops -> NC-317
pytest.mark.xfail(OnnxBackendNodeModelTest.test_acos_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_acos_example_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_asin_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_asin_example_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_atan_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_atan_example_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_cos_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_cos_example_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_sin_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_sin_example_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_tan_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_tan_example_cpu)

# PyTorch Operator tests -> NC-329
pytest.mark.xfail(OnnxBackendPyTorchOperatorModelTest.test_operator_mm_cpu)
pytest.mark.xfail(OnnxBackendPyTorchOperatorModelTest.test_operator_pad_cpu)
pytest.mark.xfail(OnnxBackendPyTorchOperatorModelTest.test_operator_repeat_cpu)
pytest.mark.xfail(OnnxBackendPyTorchOperatorModelTest.test_operator_repeat_dim_overflow_cpu)
pytest.mark.xfail(OnnxBackendPyTorchOperatorModelTest.test_operator_symbolic_override_cpu)

# Reshape ops -> NC-321
pytest.mark.xfail(OnnxBackendNodeModelTest.test_expand_dim_changed_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_expand_dim_unchanged_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_gather_0_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_gather_1_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_reshape_extended_dims_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_reshape_negative_dim_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_reshape_one_dim_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_reshape_reduced_dims_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_reshape_reordered_dims_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_size_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_size_example_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_tile_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_tile_precomputed_cpu)
pytest.mark.xfail(OnnxBackendSimpleModelTest.test_expand_shape_model1_cpu)
pytest.mark.xfail(OnnxBackendSimpleModelTest.test_expand_shape_model2_cpu)
pytest.mark.xfail(OnnxBackendSimpleModelTest.test_expand_shape_model3_cpu)
pytest.mark.xfail(OnnxBackendSimpleModelTest.test_expand_shape_model4_cpu)
pytest.mark.xfail(OnnxBackendPyTorchConvertedModelTest.test_Embedding_cpu)
pytest.mark.xfail(OnnxBackendPyTorchConvertedModelTest.test_Embedding_sparse_cpu)

# Other tests
pytest.mark.xfail(OnnxBackendNodeModelTest.test_cast_DOUBLE_to_FLOAT16_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_cast_FLOAT_to_FLOAT16_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_axis_0_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_axis_1_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_axis_2_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_default_axis_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_example_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_one_hot_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_instancenorm_epsilon_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_instancenorm_example_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_upsample_nearest_cpu)
