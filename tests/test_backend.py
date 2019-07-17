# ******************************************************************************
# Copyright 2018-2019 Intel Corporation
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

import tests.utils

# Set backend device name to be used instead of hardcoded by ONNX BackendTest class ones.
selected_backend_name = tests.utils.BACKEND_NAME
NgraphBackend.backend_name = selected_backend_name

# This is a pytest magic variable to load extra plugins
# Uncomment the line below to enable the ONNX compatibility report
# pytest_plugins = 'onnx.backend.test.report',

# import all test cases at global scope to make them visible to python.unittest
backend_test = onnx.backend.test.BackendTest(NgraphBackend, __name__)

# MaxPool Indices -> NGRAPH-3131
backend_test.exclude('test_maxpool_with_argmax')

# RNN -> NC-323
backend_test.exclude('test_simple_rnn')
backend_test.exclude('test_rnn')
backend_test.exclude('test_operator_rnn')

# PyTorch LSTM operator -> NGONNX-373
backend_test.exclude('test_operator_lstm')

# GRU -> NC-325
backend_test.exclude('test_gru')

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

OnnxBackendNodeModelTest = None
OnnxBackendSimpleModelTest = None
OnnxBackendPyTorchOperatorModelTest = None
OnnxBackendPyTorchConvertedModelTest = None
globals().update(backend_test.enable_report().test_cases)

# PyTorch Operator tests -> NC-329
pytest.mark.xfail(OnnxBackendPyTorchOperatorModelTest.test_operator_repeat_cpu)
pytest.mark.xfail(OnnxBackendPyTorchOperatorModelTest.test_operator_repeat_dim_overflow_cpu)

# Dynamic Expand -> NGONNX-367
pytest.mark.xfail(OnnxBackendNodeModelTest.test_expand_dim_changed_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_expand_dim_unchanged_cpu)
pytest.mark.xfail(OnnxBackendSimpleModelTest.test_expand_shape_model1_cpu)
pytest.mark.xfail(OnnxBackendSimpleModelTest.test_expand_shape_model2_cpu)
pytest.mark.xfail(OnnxBackendSimpleModelTest.test_expand_shape_model3_cpu)
pytest.mark.xfail(OnnxBackendSimpleModelTest.test_expand_shape_model4_cpu)

# Dynamic Reshape -> NGONNX-357
pytest.mark.xfail(OnnxBackendNodeModelTest.test_reshape_extended_dims_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_reshape_negative_dim_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_reshape_one_dim_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_reshape_reduced_dims_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_reshape_reordered_dims_cpu)

# Dynamic Tile -> NGONNX-368
pytest.mark.xfail(OnnxBackendNodeModelTest.test_tile_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_tile_precomputed_cpu)

# Cast -> NGONNX-427
pytest.mark.xfail(OnnxBackendNodeModelTest.test_cast_FLOAT_to_STRING_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_cast_STRING_to_FLOAT_cpu)

# Scan -> NGONNX-433
pytest.mark.xfail(OnnxBackendNodeModelTest.test_scan9_sum_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_scan_sum_cpu)

# Compress -> NGONNX-438
pytest.mark.xfail(OnnxBackendNodeModelTest.test_compress_default_axis_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_compress_0_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_compress_1_cpu)

# Isnan -> NGONNX-440
pytest.mark.xfail(OnnxBackendNodeModelTest.test_isnan_cpu)

# Constant of Shape -> NGONNX-445
pytest.mark.xfail(OnnxBackendNodeModelTest.test_constantofshape_float_ones_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_constantofshape_int_zeros_cpu)

# Scatter -> NGONNX-446
pytest.mark.xfail(OnnxBackendNodeModelTest.test_scatter_with_axis_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_scatter_without_axis_cpu)

# Max unpool -> NGONNX-447
pytest.mark.xfail(OnnxBackendNodeModelTest.test_maxunpool_export_with_output_shape_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_maxunpool_export_without_output_shape_cpu)

# OneHot -> NGONNX-453
pytest.mark.xfail(OnnxBackendNodeModelTest.test_onehot_with_axis_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_onehot_without_axis_cpu)

# TF id vectorizer -> NGONNX-471
pytest.mark.xfail(OnnxBackendNodeModelTest.test_tfidfvectorizer_tf_batch_onlybigrams_skip0_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_tfidfvectorizer_tf_batch_onlybigrams_skip5_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_tfidfvectorizer_tf_batch_uniandbigrams_skip5_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_tfidfvectorizer_tf_only_bigrams_skip0_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_tfidfvectorizer_tf_onlybigrams_levelempty_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_tfidfvectorizer_tf_onlybigrams_skip5_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_tfidfvectorizer_tf_uniandbigrams_skip5_cpu)

# Non zero -> NGONNX-472
pytest.mark.xfail(OnnxBackendNodeModelTest.test_nonzero_example_cpu)

# ConvInteger NGONNX-410
pytest.mark.xfail(OnnxBackendNodeModelTest.test_basic_convinteger_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_convinteger_with_padding_cpu)

# Quantized NGONNX-595
# Scale / zero point not a scalar
pytest.mark.xfail(OnnxBackendNodeModelTest.test_dequantizelinear_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_qlinearconv_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_qlinearmatmul_2D_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_qlinearmatmul_3D_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_quantizelinear_cpu)

# MatmulInteger NGONNX-410
pytest.mark.xfail(OnnxBackendNodeModelTest.test_matmulinteger_cpu)

# IsInf - NGONNX-528
pytest.mark.xfail(OnnxBackendNodeModelTest.test_isinf_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_isinf_negative_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_isinf_positive_cpu)

# Pooling ops NGONNX-597
pytest.mark.xfail(OnnxBackendNodeModelTest.test_maxpool_2d_ceil_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_maxpool_2d_dilations_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_averagepool_2d_ceil_cpu)

# Modulus - NGONNX-527
pytest.mark.xfail(OnnxBackendNodeModelTest.test_mod_bcast_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_mod_float_mixed_sign_example_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_mod_fmod_mixed_sign_example_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_mod_int64_mixed_sign_example_cpu)

# NonMaxSuppression - NGONNX-526
pytest.mark.xfail(OnnxBackendNodeModelTest.test_nonmaxsuppression_center_point_box_format_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_nonmaxsuppression_flipped_coordinates_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_nonmaxsuppression_identical_boxes_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_nonmaxsuppression_limit_output_size_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_nonmaxsuppression_single_box_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_nonmaxsuppression_suppress_by_IOU_and_scores_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_nonmaxsuppression_suppress_by_IOU_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_nonmaxsuppression_two_batches_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_nonmaxsuppression_two_classes_cpu)

# Resize NGONNX-598
pytest.mark.xfail(OnnxBackendNodeModelTest.test_resize_downsample_linear_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_resize_downsample_nearest_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_resize_nearest_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_resize_upsample_linear_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_resize_upsample_nearest_cpu)

# ReverseSequence - NGONNX-525
pytest.mark.xfail(OnnxBackendNodeModelTest.test_reversesequence_batch_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_reversesequence_time_cpu)

# Dynamic Slice NGONNX-522, 599
pytest.mark.xfail(OnnxBackendNodeModelTest.test_slice_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_slice_default_axes_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_slice_default_steps_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_slice_end_out_of_bounds_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_slice_neg_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_slice_neg_steps_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_slice_start_out_of_bounds_cpu)

# StrNormalizer NGONNX-600
pytest.mark.xfail(OnnxBackendNodeModelTest.test_strnormalizer_export_monday_casesensintive_lower_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_strnormalizer_export_monday_casesensintive_nochangecase_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_strnormalizer_export_monday_casesensintive_upper_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_strnormalizer_export_monday_empty_output_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_strnormalizer_export_monday_insensintive_upper_twodim_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_strnormalizer_nostopwords_nochangecase_cpu)
pytest.mark.xfail(OnnxBackendSimpleModelTest.test_strnorm_model_monday_casesensintive_lower_cpu)
pytest.mark.xfail(OnnxBackendSimpleModelTest.test_strnorm_model_monday_casesensintive_nochangecase_cpu)
pytest.mark.xfail(OnnxBackendSimpleModelTest.test_strnorm_model_monday_casesensintive_upper_cpu)
pytest.mark.xfail(OnnxBackendSimpleModelTest.test_strnorm_model_monday_empty_output_cpu)
pytest.mark.xfail(OnnxBackendSimpleModelTest.test_strnorm_model_monday_insensintive_upper_twodim_cpu)
pytest.mark.xfail(OnnxBackendSimpleModelTest.test_strnorm_model_nostopwords_nochangecase_cpu)

# RoiAlign NGONNX-601
pytest.mark.xfail(OnnxBackendNodeModelTest.test_roialign_cpu)

# NGONNX-521
pytest.mark.xfail(OnnxBackendNodeModelTest.test_top_k_cpu)

# Other tests
pytest.mark.xfail(OnnxBackendNodeModelTest.test_upsample_nearest_cpu)

# Tests which fail on the INTELGPU backend
if selected_backend_name == 'INTELGPU':
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_edge_pad_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_erf_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_gather_0_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_gather_1_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_gemm_broadcast_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_axis_0_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_axis_1_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_axis_2_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_default_axis_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_example_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_one_hot_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_maxpool_2d_same_upper_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_reflect_pad_cpu)
    pytest.mark.xfail(OnnxBackendPyTorchConvertedModelTest.test_Embedding_cpu)
    pytest.mark.xfail(OnnxBackendPyTorchConvertedModelTest.test_Embedding_sparse_cpu)
    pytest.mark.xfail(OnnxBackendPyTorchConvertedModelTest.test_ReflectionPad2d_cpu)
    pytest.mark.xfail(OnnxBackendPyTorchConvertedModelTest.test_ReplicationPad2d_cpu)
    pytest.mark.xfail(OnnxBackendPyTorchOperatorModelTest.test_operator_pad_cpu)

# Tests which fail or are very slow on the INTERPRETER backend
if selected_backend_name == 'INTERPRETER':
    backend_test.exclude('test_operator_conv_cpu')
    # Cast -> NGONNX-427
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_cast_DOUBLE_to_FLOAT16_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_cast_FLOAT_to_FLOAT16_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_cast_FLOAT16_to_DOUBLE_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_cast_FLOAT16_to_FLOAT_cpu)

if selected_backend_name == 'CPU':
    # Cast -> NGONNX-427
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_cast_DOUBLE_to_FLOAT16_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_cast_FLOAT_to_FLOAT16_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_cast_FLOAT16_to_DOUBLE_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_cast_FLOAT16_to_FLOAT_cpu)
    # Tests which fail on the CPU backend -> NC-330
    backend_test.exclude('test_Conv3d_dilated')
    backend_test.exclude('test_Conv3d_dilated_strided')

if selected_backend_name == 'PlaidML':
    pytest.mark.xfail(OnnxBackendPyTorchConvertedModelTest.test_Embedding_cpu)
    pytest.mark.xfail(OnnxBackendPyTorchConvertedModelTest.test_Embedding_sparse_cpu)
    pytest.mark.xfail(OnnxBackendPyTorchConvertedModelTest.test_ReflectionPad2d_cpu)
    pytest.mark.xfail(OnnxBackendPyTorchConvertedModelTest.test_ReplicationPad2d_cpu)
    pytest.mark.xfail(OnnxBackendPyTorchOperatorModelTest.test_operator_pad_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_clip_default_inbounds_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_clip_default_max_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_clip_default_min_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_convtranspose_output_shape_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_edge_pad_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_edge_pad_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_erf_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_gather_0_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_gather_1_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_axis_0_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_axis_1_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_axis_2_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_default_axis_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_example_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_one_hot_cpu)
    pytest.mark.xfail(OnnxBackendNodeModelTest.test_reflect_pad_cpu)
    # Test which fail on PlaidML with INTELGPU
    pytest.mark.xfail(OnnxBackendPyTorchOperatorModelTest.test_operator_pow_cpu)
