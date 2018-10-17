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


# Pooling ops -> NC5-313
backend_test.exclude('test_AvgPool1d')
backend_test.exclude('test_globalaveragepool')
backend_test.exclude('test_globalmaxpool')
backend_test.exclude('test_maxpool')
backend_test.exclude('test_MaxPool1d')
backend_test.exclude('test_averagepool')
backend_test.exclude('test_operator_maxpool')

# Matmul ops -> NC5-314
backend_test.exclude('test_matmul_3d')
backend_test.exclude('test_matmul_4d')
backend_test.exclude('test_gemm_broadcast')
backend_test.exclude('test_Linear')
backend_test.exclude('test_operator_mm')

# Convolution ops -> NC-315
backend_test.exclude('test_Conv1d')
backend_test.exclude('test_Conv2d')
backend_test.exclude('test_Conv3d')
backend_test.exclude('test_operator_conv')

# ArgMin/ArgMax -> NC-316
backend_test.exclude('test_argmax')
backend_test.exclude('test_argmin')

# Trigonometric ops -> NC-317
backend_test.exclude('test_acos')
backend_test.exclude('test_asin')
backend_test.exclude('test_atan')
backend_test.exclude('test_cos')
backend_test.exclude('test_sin')
backend_test.exclude('test_tan')

# ConvTranspose -> NC-319
backend_test.exclude('test_ConvTranspose2d')
backend_test.exclude('test_convtranspose')
backend_test.exclude('test_operator_convtranspose')

# Non-linear ops -> NC-320
backend_test.exclude('test_SELU')
backend_test.exclude('test_prelu_example')
backend_test.exclude('test_selu')
backend_test.exclude('test_operator_selu')

# Reshape ops -> NC-321
backend_test.exclude('test_reshape')
backend_test.exclude('test_expand')
backend_test.exclude('test_gather')
backend_test.exclude('test_Embedding')
backend_test.exclude('test_tile')
backend_test.exclude('test_size')
backend_test.exclude('test_squeeze_cpu')
backend_test.exclude('test_operator_flatten')

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

# PyTorch Operator tests -> NC-329
backend_test.exclude('test_operator_add')
backend_test.exclude('test_operator_basic')
backend_test.exclude('test_operator_chunk')
backend_test.exclude('test_operator_clip')
backend_test.exclude('test_operator_concat2')
backend_test.exclude('test_operator_exp')
backend_test.exclude('test_operator_index')
backend_test.exclude('test_operator_max')
backend_test.exclude('test_operator_min')
backend_test.exclude('test_operator_non_float_params')
backend_test.exclude('test_operator_pad')
backend_test.exclude('test_operator_params')
backend_test.exclude('test_operator_permute2')
backend_test.exclude('test_operator_pow')
backend_test.exclude('test_operator_reduced')
backend_test.exclude('test_operator_repeat')
backend_test.exclude('test_operator_sqrt')
backend_test.exclude('test_operator_symbolic_override')
backend_test.exclude('test_operator_transpose')
backend_test.exclude('test_operator_view')

# Other tests
backend_test.exclude('test_GLU')
backend_test.exclude('test_Softmin')
backend_test.exclude('test_hardmax')
backend_test.exclude('test_PixelShuffle')
backend_test.exclude('test_PoissonNLLLLoss_no_reduce')
backend_test.exclude('test_cast_DOUBLE_to_FLOAT16')
backend_test.exclude('test_cast_FLOAT_to_FLOAT16')
backend_test.exclude('test_instancenorm')
backend_test.exclude('test_upsample_nearest')

# Tests which fail on the CPU backend -> NC-330
if selected_backend_name == 'CPU':
    backend_test.exclude('test_Conv3d_dilated')
    backend_test.exclude('test_Conv3d_dilated_strided')
    backend_test.exclude('test_GLU_dim')

# Big model tests:
#backend_test.exclude('test_bvlc_alexnet')
backend_test.exclude('test_densenet121')
#backend_test.exclude('test_inception_v1')
#backend_test.exclude('test_inception_v2')
#backend_test.exclude('test_resnet50')
#backend_test.exclude('test_shufflenet')
backend_test.exclude('test_squeezenet')
#backend_test.exclude('test_vgg19')
#backend_test.exclude('test_zfnet512')

# Tests which fail or are very slow on the INTERPRETER backend
if selected_backend_name == 'INTERPRETER':
    backend_test.exclude('test_densenet121_cpu')
    backend_test.exclude('test_inception_v2_cpu')
    backend_test.exclude('test_resnet50_cpu')
    backend_test.exclude('test_squeezenet_cpu')
    backend_test.exclude('test_vgg19_cpu')
    backend_test.exclude('test_shufflenet_cpu')
    backend_test.exclude('test_bvlc_alexnet_cpu')
    backend_test.exclude('test_inception_v1_cpu')
    backend_test.exclude('test_zfnet512_cpu')
    backend_test.exclude('test_operator_conv_cpu')
    backend_test.exclude('test_slice_start_out_of_bounds_cpu')

globals().update(backend_test.enable_report().test_cases)
