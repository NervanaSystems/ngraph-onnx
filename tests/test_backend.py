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

# MeanVarianceNormalization -> NC-328
backend_test.exclude('test_mvn')

# Tests which fail on the CPU backend -> NC-330
if selected_backend_name == 'CPU':
    backend_test.exclude('test_Conv3d_dilated')
    backend_test.exclude('test_Conv3d_dilated_strided')

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

OnnxBackendNodeModelTest = None
OnnxBackendSimpleModelTest = None
OnnxBackendPyTorchOperatorModelTest = None
OnnxBackendPyTorchConvertedModelTest = None
globals().update(backend_test.enable_report().test_cases)

# PyTorch Operator tests -> NC-329
pytest.mark.xfail(OnnxBackendPyTorchOperatorModelTest.test_operator_repeat_cpu)
pytest.mark.xfail(OnnxBackendPyTorchOperatorModelTest.test_operator_repeat_dim_overflow_cpu)
pytest.mark.xfail(OnnxBackendPyTorchOperatorModelTest.test_operator_symbolic_override_cpu)

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
pytest.mark.xfail(OnnxBackendPyTorchConvertedModelTest.test_Embedding_cpu)
pytest.mark.xfail(OnnxBackendPyTorchConvertedModelTest.test_Embedding_sparse_cpu)

# Cast -> NGONNX-427
pytest.mark.xfail(OnnxBackendNodeModelTest.test_cast_FLOAT_to_STRING_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_cast_STRING_to_FLOAT_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_cast_DOUBLE_to_FLOAT16_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_cast_FLOAT_to_FLOAT16_cpu)

# Scan -> NGONNX-433
pytest.mark.xfail(OnnxBackendNodeModelTest.test_scan9_sum_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_scan_sum_cpu)

# Compress -> NGONNX-438
pytest.mark.xfail(OnnxBackendNodeModelTest.test_compress_default_axis_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_compress_0_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_compress_1_cpu)

# Eyelike -> NGONNX-439
pytest.mark.xfail(OnnxBackendNodeModelTest.test_eyelike_populate_off_main_diagonal_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_eyelike_with_dtype_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_eyelike_without_dtype_cpu)

# Isnan -> NGONNX-440
pytest.mark.xfail(OnnxBackendNodeModelTest.test_isnan_cpu)

# Erf -> NGONNX-442
pytest.mark.xfail(OnnxBackendNodeModelTest.test_erf_cpu)

# Constant of Shape -> NGONNX-445
pytest.mark.xfail(OnnxBackendNodeModelTest.test_constantofshape_float_ones_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_constantofshape_int_zeros_cpu)

# Scatter -> NGONNX-446
pytest.mark.xfail(OnnxBackendNodeModelTest.test_scatter_with_axis_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_scatter_without_axis_cpu)

# Max unpool -> NGONNX-447
pytest.mark.xfail(OnnxBackendNodeModelTest.test_maxunpool_export_with_output_shape_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_maxunpool_export_without_output_shape_cpu)

# Shrink -> NGONNX-449
pytest.mark.xfail(OnnxBackendSimpleModelTest.test_shrink_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_shrink_hard_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_shrink_soft_cpu)

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

# Other tests
pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_axis_0_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_axis_1_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_axis_2_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_default_axis_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_example_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_hardmax_one_hot_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_instancenorm_epsilon_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_instancenorm_example_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_upsample_nearest_cpu)

# Dynamic Slice -> EXPERIMENTAL https://github.com/onnx/onnx/blob/master/docs/Operators.md#DynamicSlice
pytest.mark.xfail(OnnxBackendNodeModelTest.test_dynamic_slice_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_dynamic_slice_default_axes_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_dynamic_slice_end_out_of_bounds_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_dynamic_slice_neg_cpu)
pytest.mark.xfail(OnnxBackendNodeModelTest.test_dynamic_slice_start_out_of_bounds_cpu)
