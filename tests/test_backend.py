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
backend_test.exclude('test_concat')
backend_test.exclude('test_flatten')
backend_test.exclude('test_gather')
backend_test.exclude('test_hardmax')
backend_test.exclude('test_hardsigmoid')
backend_test.exclude('test_logsoftmax')
backend_test.exclude('test_max_one_input')
backend_test.exclude('test_min_one_input')
backend_test.exclude('test_pow')
backend_test.exclude('test_shape')
backend_test.exclude('test_size')
backend_test.exclude('test_softplus')
backend_test.exclude('test_softsign')
backend_test.exclude('test_softmax')
backend_test.exclude('test_squeeze')
backend_test.exclude('test_sum_one_input')
backend_test.exclude('test_thresholdedrelu')
backend_test.exclude('test_top_k')
backend_test.exclude('test_transpose')
backend_test.exclude('test_unsqueeze')
backend_test.exclude('test_Upsample_nearest_scale_2d')
backend_test.exclude('test_operator_chunk')
backend_test.exclude('test_operator_clip')
backend_test.exclude('test_operator_flatten')
backend_test.exclude('test_operator_maxpool')
backend_test.exclude('test_operator_permute2')
backend_test.exclude('test_operator_view')


backend_test.exclude('test_ConstantPad2d')
backend_test.exclude('test_Embedding')
backend_test.exclude('test_GLU')
backend_test.exclude('test_Linear')
backend_test.exclude('test_LogSoftmax')
backend_test.exclude('test_MaxPool1d')
backend_test.exclude('test_PReLU')
backend_test.exclude('test_PixelShuffle')
backend_test.exclude('test_ReflectionPad2d')
backend_test.exclude('test_ReplicationPad2d')
backend_test.exclude('test_Softmin')
backend_test.exclude('test_Softplus')
backend_test.exclude('test_Softsign')
backend_test.exclude('test_ZeroPad')
backend_test.exclude('test_constant_pad')
backend_test.exclude('test_edge_pad')
backend_test.exclude('test_log_softmax')
backend_test.exclude('test_matmul_3d')
backend_test.exclude('test_matmul_4d')
backend_test.exclude('test_reflect_pad')
backend_test.exclude('test_slice')
backend_test.exclude('test_default_axes')

# Convolution tests
backend_test.exclude('test_Conv2d_depthwise_cpu')
backend_test.exclude('test_Conv2d_depthwise_padded_cpu')
backend_test.exclude('test_Conv2d_depthwise_strided_cpu')
backend_test.exclude('test_Conv2d_depthwise_with_multiplier_cpu')
backend_test.exclude('test_Conv2d_groups_cpu')
backend_test.exclude('test_Conv2d_groups_thnn_cpu')
backend_test.exclude('test_Conv3d_cpu')
backend_test.exclude('test_Conv3d_dilated_cpu')
backend_test.exclude('test_Conv3d_dilated_strided_cpu')
backend_test.exclude('test_Conv3d_groups_cpu')
backend_test.exclude('test_Conv3d_no_bias_cpu')
backend_test.exclude('test_Conv3d_stride_cpu')
backend_test.exclude('test_Conv3d_stride_padding_cpu')

# big models tests
backend_test.exclude('test_bvlc_alexnet')
backend_test.exclude('test_resnet50')
backend_test.exclude('test_vgg16')
backend_test.exclude('test_vgg19')
backend_test.exclude('test_densenet121')
backend_test.exclude('test_inception_v1')
backend_test.exclude('test_inception_v2')
backend_test.exclude('test_shufflenet')
backend_test.exclude('test_squeezenet')


globals().update(backend_test.enable_report().test_cases)
