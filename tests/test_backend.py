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

# Big model tests
# Passing
# backend_test.exclude('test_densenet121_cpu')
# backend_test.exclude('test_inception_v2_cpu')
# backend_test.exclude('test_resnet50_cpu')
# backend_test.exclude('test_squeezenet_cpu')
# backend_test.exclude('test_vgg19_cpu')

# Failing
# Missing op 'LRN' -> NGRAPH-1731
backend_test.exclude('test_bvlc_alexnet_cpu')
backend_test.exclude('test_inception_v1_cpu')
backend_test.exclude('test_zfnet512_cpu')

# UserInputError msg='Node (%s): provided axes count is different than input tensor rank.'
backend_test.exclude('test_shufflenet_cpu')

# Ops
# Missing ops
# Missing op 'Acos'
backend_test.exclude('test_acos_cpu')
backend_test.exclude('test_acos_example_cpu')

# Missing op 'Asin'
backend_test.exclude('test_asin_cpu')
backend_test.exclude('test_asin_example_cpu')

# Missing op 'Atan'
backend_test.exclude('test_atan_cpu')
backend_test.exclude('test_atan_example_cpu')

# Missing op 'Cast' -> NC5-159
backend_test.exclude('test_cast_DOUBLE_to_FLOAT16_cpu')
backend_test.exclude('test_cast_DOUBLE_to_FLOAT_cpu')
backend_test.exclude('test_cast_FLOAT_to_DOUBLE_cpu')
backend_test.exclude('test_cast_FLOAT_to_FLOAT16_cpu')

# Missing op 'ConvTranspose' -> NGRAPH-1840
backend_test.exclude('test_convtranspose_1d_cpu')
backend_test.exclude('test_convtranspose_3d_cpu')
backend_test.exclude('test_convtranspose_cpu')
backend_test.exclude('test_convtranspose_output_shape_cpu')
backend_test.exclude('test_convtranspose_pad_cpu')
backend_test.exclude('test_convtranspose_pads_cpu')
backend_test.exclude('test_ConvTranspose2d_cpu')
backend_test.exclude('test_ConvTranspose2d_no_bias_cpu')
backend_test.exclude('test_operator_convtranspose_cpu')

# Missing op 'Cos'
backend_test.exclude('test_cos_cpu')
backend_test.exclude('test_cos_example_cpu')

# Missing op 'Gather' -> NC5-160
backend_test.exclude('test_Embedding_cpu')
backend_test.exclude('test_Embedding_sparse_cpu')
backend_test.exclude('test_gather_0_cpu')
backend_test.exclude('test_gather_1_cpu')
backend_test.exclude('test_operator_lstm_cpu')
backend_test.exclude('test_operator_rnn_cpu')
backend_test.exclude('test_operator_rnn_single_layer_cpu')

# Missing op 'GRU' -> NC5-177
backend_test.exclude('test_gru_defaults_cpu')
backend_test.exclude('test_gru_seq_length_cpu')
backend_test.exclude('test_gru_with_initial_bias_cpu')

# Missing op 'Hardmax' -> NGRAPH-1839
backend_test.exclude('test_hardmax_axis_0_cpu')
backend_test.exclude('test_hardmax_axis_1_cpu')
backend_test.exclude('test_hardmax_axis_2_cpu')
backend_test.exclude('test_hardmax_default_axis_cpu')
backend_test.exclude('test_hardmax_example_cpu')
backend_test.exclude('test_hardmax_one_hot_cpu')

# Missing op 'Identity' -> NC5-178
backend_test.exclude('test_identity_cpu')

# Missing op 'InstanceNormalization' -> NC5-179
backend_test.exclude('test_instancenorm_epsilon_cpu')
backend_test.exclude('test_instancenorm_example_cpu')
backend_test.exclude('test_operator_symbolic_override_cpu')

# Missing op 'LRN' -> NGRAPH-1731
backend_test.exclude('test_lrn_cpu')
backend_test.exclude('test_lrn_default_cpu')

# Missing op 'LSTM' -> NC5-180
backend_test.exclude('test_lstm_defaults_cpu')
backend_test.exclude('test_lstm_with_initial_bias_cpu')
backend_test.exclude('test_lstm_with_peepholes_cpu')

# Missing op 'Pow' -> NC5-163
backend_test.exclude('test_operator_pow_cpu')
backend_test.exclude('test_pow_cpu')
backend_test.exclude('test_pow_bcast_array_cpu')
backend_test.exclude('test_pow_bcast_scalar_cpu')
backend_test.exclude('test_pow_example_cpu')

# Missing op 'ReduceL1' -> NC5-181
backend_test.exclude('test_reduce_l1_default_axes_keepdims_example_cpu')
backend_test.exclude('test_reduce_l1_default_axes_keepdims_random_cpu')
backend_test.exclude('test_reduce_l1_do_not_keepdims_example_cpu')
backend_test.exclude('test_reduce_l1_do_not_keepdims_random_cpu')
backend_test.exclude('test_reduce_l1_keep_dims_example_cpu')
backend_test.exclude('test_reduce_l1_keep_dims_random_cpu')

# Missing op 'ReduceL2' -> NC5-181
backend_test.exclude('test_reduce_l2_default_axes_keepdims_example_cpu')
backend_test.exclude('test_reduce_l2_default_axes_keepdims_random_cpu')
backend_test.exclude('test_reduce_l2_do_not_keepdims_example_cpu')
backend_test.exclude('test_reduce_l2_do_not_keepdims_random_cpu')
backend_test.exclude('test_reduce_l2_keep_dims_example_cpu')
backend_test.exclude('test_reduce_l2_keep_dims_random_cpu')

# Missing op 'ReduceLogSum' -> NC5-181
backend_test.exclude('test_reduce_log_sum_asc_axes_cpu')
backend_test.exclude('test_reduce_log_sum_cpu')
backend_test.exclude('test_reduce_log_sum_default_cpu')
backend_test.exclude('test_reduce_log_sum_desc_axes_cpu')

# Missing op 'ReduceSumSquare' -> NC5-181
backend_test.exclude('test_reduce_sum_square_default_axes_keepdims_example_cpu')
backend_test.exclude('test_reduce_sum_square_default_axes_keepdims_random_cpu')
backend_test.exclude('test_reduce_sum_square_do_not_keepdims_example_cpu')
backend_test.exclude('test_reduce_sum_square_do_not_keepdims_random_cpu')
backend_test.exclude('test_reduce_sum_square_keepdims_example_cpu')
backend_test.exclude('test_reduce_sum_square_keepdims_random_cpu')

# Missing op 'RNN' -> NC5-182
backend_test.exclude('test_rnn_seq_length_cpu')
backend_test.exclude('test_simple_rnn_defaults_cpu')
backend_test.exclude('test_simple_rnn_with_initial_bias_cpu')

# Missing op 'Sin'
backend_test.exclude('test_sin_cpu')
backend_test.exclude('test_sin_example_cpu')

# Missing op 'Tan'
backend_test.exclude('test_tan_cpu')
backend_test.exclude('test_tan_example_cpu')

# Missing op 'Tile' -> NGRAPH-1909
backend_test.exclude('test_operator_repeat_cpu')
backend_test.exclude('test_operator_repeat_dim_overflow_cpu')
backend_test.exclude('test_tile_cpu')
backend_test.exclude('test_tile_precomputed_cpu')

# Missing op 'TopK' -> NC5-161 -> NGRAPH-1910
backend_test.exclude('test_top_k_cpu')

# Missing op 'Upsample' -> NGRAPH-1841
backend_test.exclude('test_upsample_nearest_cpu')
backend_test.exclude('test_Upsample_nearest_2d_cpu')
backend_test.exclude('test_Upsample_nearest_scale_2d_cpu')
backend_test.exclude('test_Upsample_nearest_tuple_2d_cpu')

# Runtime problems
# RuntimeError msg='Broadcast arg, shape, and axes are incompatible'
backend_test.exclude('test_gemm_broadcast_cpu')
backend_test.exclude('test_operator_add_size1_broadcast_cpu')
backend_test.exclude('test_operator_add_size1_singleton_broadcast_cpu')
backend_test.exclude('test_operator_mm_cpu')
backend_test.exclude('test_prelu_broadcast_cpu')

# RuntimeError msg='Convolution padding-below rank does not match number of spatial dimensions.'
backend_test.exclude('test_Conv1d_cpu')
backend_test.exclude('test_Conv1d_dilated_cpu')
backend_test.exclude('test_Conv1d_groups_cpu')
backend_test.exclude('test_Conv1d_pad1_cpu')
backend_test.exclude('test_Conv1d_pad1size1_cpu')
backend_test.exclude('test_Conv1d_pad2_cpu')
backend_test.exclude('test_Conv1d_pad2size1_cpu')
backend_test.exclude('test_Conv1d_stride_cpu')

# RuntimeError msg='Dot axes do not have same length' -> NGRAPH-1838
backend_test.exclude('test_matmul_3d_cpu')
backend_test.exclude('test_matmul_4d_cpu')

# Other problems
# AssertionError msg='Not equal to tolerance' -> NGRAPH-1733
backend_test.exclude('test_reshape_extended_dims_cpu')
backend_test.exclude('test_reshape_negative_dim_cpu')
backend_test.exclude('test_reshape_one_dim_cpu')
backend_test.exclude('test_reshape_reduced_dims_cpu')
backend_test.exclude('test_reshape_reordered_dims_cpu')

backend_test.exclude('test_batchnorm_example_cpu')

# IndexError msg='list assignment index out of range'
backend_test.exclude('test_prelu_example_cpu')

# NG_TYPE_ERROR msg='Unidentified data type %s' param='float16'
backend_test.exclude('test_cast_FLOAT16_to_DOUBLE_cpu')
backend_test.exclude('test_cast_FLOAT16_to_FLOAT_cpu')

# NotImplementedError msg='Pad node (%s): only constant padding is supported.' -> NGRAPH-1505
backend_test.exclude('test_edge_pad_cpu')
backend_test.exclude('test_operator_pad_cpu')
backend_test.exclude('test_reflect_pad_cpu')
backend_test.exclude('test_ReflectionPad2d_cpu')
backend_test.exclude('test_ReplicationPad2d_cpu')

# UserInputError msg='Node (%s): provided axes count is different than input tensor rank.'
backend_test.exclude('test_PixelShuffle_cpu')

# ValueError msg='LogSoftmax node (%s): provided axis attribute is out of input tensor dimensions range.'
backend_test.exclude('test_log_softmax_lastdim_cpu')


globals().update(backend_test.enable_report().test_cases)
