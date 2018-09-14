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


backend_test.exclude('test_AvgPool1d')
backend_test.exclude('test_BatchNorm')
backend_test.exclude('test_ConstantPad2d')
backend_test.exclude('test_Conv1d')
backend_test.exclude('test_Conv2d')
backend_test.exclude('test_Conv3d')
backend_test.exclude('test_ConvTranspose2d')
backend_test.exclude('test_ELU')
backend_test.exclude('test_Embedding')
backend_test.exclude('test_GLU')
backend_test.exclude('test_LeakyReLU')
backend_test.exclude('test_Linear')
backend_test.exclude('test_LogSoftmax')
backend_test.exclude('test_MaxPool1d')
backend_test.exclude('test_PReLU')
backend_test.exclude('test_PixelShuffle')
backend_test.exclude('test_PoissonNLLLLoss_no_reduce')
backend_test.exclude('test_ReflectionPad2d')
backend_test.exclude('test_ReplicationPad2d')
backend_test.exclude('test_SELU')
backend_test.exclude('test_Sigmoid')
backend_test.exclude('test_Softmin')
backend_test.exclude('test_Softplus')
backend_test.exclude('test_Softsign')
backend_test.exclude('test_Tanh')
backend_test.exclude('test_ZeroPad2d')
backend_test.exclude('test_abs')
backend_test.exclude('test_acos')
backend_test.exclude('test_argmax')
backend_test.exclude('test_argmin')
backend_test.exclude('test_asin')
backend_test.exclude('test_atan')
backend_test.exclude('test_averagepool')
backend_test.exclude('test_cast')
backend_test.exclude('test_ceil')
backend_test.exclude('test_clip')
backend_test.exclude('test_constant_pad')
backend_test.exclude('test_convtranspose')
backend_test.exclude('test_cos')
backend_test.exclude('test_depthtospace')
backend_test.exclude('test_div_bcast')
backend_test.exclude('test_dropout')
backend_test.exclude('test_edge_pad')
backend_test.exclude('test_elu')
backend_test.exclude('test_exp')
backend_test.exclude('test_floor')
backend_test.exclude('test_gather')
backend_test.exclude('test_gemm_broadcast')
backend_test.exclude('test_globalaveragepool')
backend_test.exclude('test_globalmaxpool')
backend_test.exclude('test_gru')
backend_test.exclude('test_hardmax')
backend_test.exclude('test_hardsigmoid')
backend_test.exclude('test_identity')
backend_test.exclude('test_instancenorm')
backend_test.exclude('test_leakyrelu')
backend_test.exclude('test_log')
backend_test.exclude('test_lrn')
backend_test.exclude('test_lstm')
backend_test.exclude('test_matmul_3d')
backend_test.exclude('test_matmul_4d')
backend_test.exclude('test_maxpool')
backend_test.exclude('test_mul_bcast')
backend_test.exclude('test_mvn')
backend_test.exclude('test_neg')
backend_test.exclude('test_operator')
backend_test.exclude('test_prelu')
backend_test.exclude('test_reciprocal')
backend_test.exclude('test_reflect_pad')
backend_test.exclude('test_reshape')
backend_test.exclude('test_rnn')
backend_test.exclude('test_selu')
backend_test.exclude('test_shape')
backend_test.exclude('test_sigmoid')
backend_test.exclude('test_simple_rnn')
backend_test.exclude('test_sin')
backend_test.exclude('test_size')
backend_test.exclude('test_slice')
backend_test.exclude('test_softplus')
backend_test.exclude('test_softsign')
backend_test.exclude('test_sqrt')
backend_test.exclude('test_squeeze')
backend_test.exclude('test_sub_bcast')
backend_test.exclude('test_tan')
backend_test.exclude('test_tanh')
backend_test.exclude('test_thresholdedrelu')
backend_test.exclude('test_tile')
backend_test.exclude('test_top_k')
backend_test.exclude('test_upsample_nearest')


# Big model tests:
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
