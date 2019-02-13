# ******************************************************************************
# Copyright 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import pytest

from ngraph_onnx.onnx_importer.backend import NgraphBackend
from tests.utils.model_zoo_tester import ModelZooTestRunner

_S3_DOWNLOAD_ONNX = 'https://s3.amazonaws.com/download.onnx/models/'
_S3_MODEL_ZOO = 'https://s3.amazonaws.com/onnx-model-zoo/'
_WINDOWS_NET = 'https://onnxzoo.blob.core.windows.net/models/'


zoo_models = [
    # ArcFace
    {
        'model_name': 'arcface_lresnet100e_opset8',
        'url': _S3_MODEL_ZOO + 'arcface/resnet100/resnet100.tar.gz',
    },

    # BVLC AlexNet
    {
        'model_name': 'bvlc_alexnet_opset3',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_3/bvlc_alexnet.tar.gz',
    },
    {
        'model_name': 'bvlc_alexnet_opset6',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_6/bvlc_alexnet.tar.gz',
    },
    {
        'model_name': 'bvlc_alexnet_opset7',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_7/bvlc_alexnet.tar.gz',
    },
    {
        'model_name': 'bvlc_alexnet_opset8',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_8/bvlc_alexnet.tar.gz',
    },
    {
        'model_name': 'bvlc_alexnet_opset9',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_9/bvlc_alexnet.tar.gz',
    },

    # BVLC GoogleNet
    {
        'model_name': 'bvlc_googlenet_opset3',
        'url': _S3_DOWNLOAD_ONNX + 'opset_3/bvlc_googlenet.tar.gz',
    },
    {
        'model_name': 'bvlc_googlenet_opset6',
        'url': _S3_DOWNLOAD_ONNX + 'opset_6/bvlc_googlenet.tar.gz',
    },
    {
        'model_name': 'bvlc_googlenet_opset7',
        'url': _S3_DOWNLOAD_ONNX + 'opset_7/bvlc_googlenet.tar.gz',
    },
    {
        'model_name': 'bvlc_googlenet_opset8',
        'url': _S3_DOWNLOAD_ONNX + 'opset_8/bvlc_googlenet.tar.gz',
    },
    {
        'model_name': 'bvlc_googlenet_opset9',
        'url': _S3_DOWNLOAD_ONNX + 'opset_9/bvlc_googlenet.tar.gz',
    },

    # BVLC CaffeNet
    {
        'model_name': 'bvlc_caffenet_opset3',
        'url': _S3_DOWNLOAD_ONNX + 'opset_3/bvlc_reference_caffenet.tar.gz',
    },
    {
        'model_name': 'bvlc_caffenet_opset6',
        'url': _S3_DOWNLOAD_ONNX + 'opset_6/bvlc_reference_caffenet.tar.gz',
    },
    {
        'model_name': 'bvlc_caffenet_opset7',
        'url': _S3_DOWNLOAD_ONNX + 'opset_7/bvlc_reference_caffenet.tar.gz',
    },
    {
        'model_name': 'bvlc_caffenet_opset8',
        'url': _S3_DOWNLOAD_ONNX + 'opset_8/bvlc_reference_caffenet.tar.gz',
    },
    {
        'model_name': 'bvlc_caffenet_opset9',
        'url': _S3_DOWNLOAD_ONNX + 'opset_9/bvlc_reference_caffenet.tar.gz',
    },

    # BVLC R-CNN ILSVRC13
    {
        'model_name': 'bvlc_rcnn_ilsvrc13_opset3',
        'url': _S3_DOWNLOAD_ONNX + 'opset_3/bvlc_reference_rcnn_ilsvrc13.tar.gz',
    },
    {
        'model_name': 'bvlc_rcnn_ilsvrc13_opset6',
        'url': _S3_DOWNLOAD_ONNX + 'opset_6/bvlc_reference_rcnn_ilsvrc13.tar.gz',
    },
    {
        'model_name': 'bvlc_rcnn_ilsvrc13_opset7',
        'url': _S3_DOWNLOAD_ONNX + 'opset_7/bvlc_reference_rcnn_ilsvrc13.tar.gz',
    },
    {
        'model_name': 'bvlc_rcnn_ilsvrc13_opset8',
        'url': _S3_DOWNLOAD_ONNX + 'opset_8/bvlc_reference_rcnn_ilsvrc13.tar.gz',
    },
    {
        'model_name': 'bvlc_rcnn_ilsvrc13_opset9',
        'url': _S3_DOWNLOAD_ONNX + 'opset_9/bvlc_reference_rcnn_ilsvrc13.tar.gz',
    },

    # DenseNet-121
    {
        'model_name': 'densenet121_opset3',
        'atol': 1e-07,
        'rtol': 0.002,
        'url': _S3_DOWNLOAD_ONNX + 'opset_3/densenet121.tar.gz',
    },
    {
        'model_name': 'densenet121_opset6',
        'atol': 1e-07,
        'rtol': 0.002,
        'url': _S3_DOWNLOAD_ONNX + 'opset_6/densenet121.tar.gz',
    },
    {
        'model_name': 'densenet121_opset7',
        'atol': 1e-07,
        'rtol': 0.002,
        'url': _S3_DOWNLOAD_ONNX + 'opset_7/densenet121.tar.gz',
    },
    {
        'model_name': 'densenet121_opset8',
        'atol': 1e-07,
        'rtol': 0.002,
        'url': _S3_DOWNLOAD_ONNX + 'opset_8/densenet121.tar.gz',
    },
    {
        'model_name': 'densenet121_opset9',
        'atol': 1e-07,
        'rtol': 0.002,
        'url': _S3_DOWNLOAD_ONNX + 'opset_9/densenet121.tar.gz',
    },

    # DUC
    {
        'model_name': 'duc_resnet101_hdc_opset7',
        'url': _S3_MODEL_ZOO + 'duc/ResNet101_DUC_HDC.tar.gz',
    },

    # Emotion-FERPlus
    {
        'model_name': 'emotion_ferplus_opset2',
        'url': _WINDOWS_NET + 'opset_2/emotion_ferplus/emotion_ferplus.tar.gz',
    },
    {
        'model_name': 'emotion_ferplus_opset7',
        'url': _WINDOWS_NET + 'opset_7/emotion_ferplus/emotion_ferplus.tar.gz',
    },
    {
        'model_name': 'emotion_ferplus_opset8',
        'url': _WINDOWS_NET + 'opset_8/emotion_ferplus/emotion_ferplus.tar.gz',
    },

    # Inception-v1
    {
        'model_name': 'inception_v1_opset3',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_3/inception_v1.tar.gz',
    },
    {
        'model_name': 'inception_v1_opset6',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_6/inception_v1.tar.gz',
    },
    {
        'model_name': 'inception_v1_opset7',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_7/inception_v1.tar.gz',
    },
    {
        'model_name': 'inception_v1_opset8',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_8/inception_v1.tar.gz',
    },
    {
        'model_name': 'inception_v1_opset9',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_9/inception_v1.tar.gz',
    },

    # Inception-v2
    {
        'model_name': 'inception_v2_opset3',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_3/inception_v2.tar.gz',
    },
    {
        'model_name': 'inception_v2_opset6',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_6/inception_v2.tar.gz',
    },
    {
        'model_name': 'inception_v2_opset7',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_7/inception_v2.tar.gz',
    },
    {
        'model_name': 'inception_v2_opset8',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_8/inception_v2.tar.gz',
    },
    {
        'model_name': 'inception_v2_opset9',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_9/inception_v2.tar.gz',
    },

    # MNIST
    {'model_name': 'mnist_opset1', 'url': _WINDOWS_NET + 'opset_1/mnist/mnist.tar.gz'},
    {'model_name': 'mnist_opset7', 'url': _WINDOWS_NET + 'opset_7/mnist/mnist.tar.gz'},
    {'model_name': 'mnist_opset8', 'url': _WINDOWS_NET + 'opset_8/mnist/mnist.tar.gz'},

    # Mobile Net
    {
        'model_name': 'mobilenet_opset7',
        'url': _S3_MODEL_ZOO + 'mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz',
    },

    # ResNet-50
    {
        'model_name': 'resnet50_opset3',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_3/resnet50.tar.gz',
    },
    {
        'model_name': 'resnet50_opset6',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_6/resnet50.tar.gz',
    },
    {
        'model_name': 'resnet50_opset7',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_7/resnet50.tar.gz',
    },
    {
        'model_name': 'resnet50_opset8',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_8/resnet50.tar.gz',
    },
    {
        'model_name': 'resnet50_opset9',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_9/resnet50.tar.gz',
    },
    {
        'model_name': 'resnet50_v2_opset7',
        'url': _S3_MODEL_ZOO + 'resnet/resnet50v2/resnet50v2.tar.gz',
    },

    # ShuffleNet
    {
        'model_name': 'shufflenet_opset3',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_3/shufflenet.tar.gz',
    },
    {
        'model_name': 'shufflenet_opset6',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_6/shufflenet.tar.gz',
    },
    {
        'model_name': 'shufflenet_opset7',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_7/shufflenet.tar.gz',
    },
    {
        'model_name': 'shufflenet_opset8',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_8/shufflenet.tar.gz',
    },
    {
        'model_name': 'shufflenet_opset9',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_9/shufflenet.tar.gz',
    },

    # SqueezeNet
    {
        'model_name': 'squeezenet_opset3',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_3/squeezenet.tar.gz',
    },
    {
        'model_name': 'squeezenet_opset6',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_6/squeezenet.tar.gz',
    },
    {
        'model_name': 'squeezenet_opset7',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_7/squeezenet.tar.gz',
    },
    {
        'model_name': 'squeezenet_opset8',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_8/squeezenet.tar.gz',
    },
    {
        'model_name': 'squeezenet_opset9',
        'atol': 1e-07,
        'rtol': 0.001,
        'url': _S3_DOWNLOAD_ONNX + 'opset_9/squeezenet.tar.gz',
    },

    # Tiny-YOLOv2
    {
        'model_name': 'tiny_yolov2_opset1',
        'url': _WINDOWS_NET + 'opset_1/tiny_yolov2/tiny_yolov2.tar.gz',
    },
    {
        'model_name': 'tiny_yolov2_opset7',
        'url': _WINDOWS_NET + 'opset_7/tiny_yolov2/tiny_yolov2.tar.gz',
    },
    {
        'model_name': 'tiny_yolov2_opset8',
        'url': _WINDOWS_NET + 'opset_8/tiny_yolov2/tiny_yolov2.tar.gz',
    },

    # VGG-19
    {'model_name': 'vgg19_opset3', 'url': _S3_DOWNLOAD_ONNX + 'opset_3/vgg19.tar.gz'},
    {'model_name': 'vgg19_opset6', 'url': _S3_DOWNLOAD_ONNX + 'opset_6/vgg19.tar.gz'},
    {'model_name': 'vgg19_opset7', 'url': _S3_DOWNLOAD_ONNX + 'opset_7/vgg19.tar.gz'},
    {'model_name': 'vgg19_opset8', 'url': _S3_DOWNLOAD_ONNX + 'opset_8/vgg19.tar.gz'},
    {'model_name': 'vgg19_opset9', 'url': _S3_DOWNLOAD_ONNX + 'opset_9/vgg19.tar.gz'},

    # ZFNet-512
    {
        'model_name': 'zfnet512_opset3',
        'url': _S3_DOWNLOAD_ONNX + 'opset_3/zfnet512.tar.gz',
    },
    {
        'model_name': 'zfnet512_opset6',
        'url': _S3_DOWNLOAD_ONNX + 'opset_6/zfnet512.tar.gz',
    },
    {
        'model_name': 'zfnet512_opset7',
        'url': _S3_DOWNLOAD_ONNX + 'opset_7/zfnet512.tar.gz',
    },
    {
        'model_name': 'zfnet512_opset8',
        'url': _S3_DOWNLOAD_ONNX + 'opset_8/zfnet512.tar.gz',
    },
    {
        'model_name': 'zfnet512_opset9',
        'url': _S3_DOWNLOAD_ONNX + 'opset_9/zfnet512.tar.gz',
    },
]


backend_name = pytest.config.getoption('backend', default='CPU')

if backend_name != 'INTERPRETER':
    # Set backend device name to be used instead of hardcoded by ONNX BackendTest class ones.
    NgraphBackend.backend_name = backend_name

    # import all test cases at global scope to make them visible to python.unittest
    backend_test = ModelZooTestRunner(NgraphBackend, zoo_models, __name__)
    test_cases = backend_test.test_cases['OnnxBackendZooModelTest']

    # Exclude failing tests...
    # Too long execution time.
    pytest.mark.skip(test_cases.test_duc_resnet101_hdc_opset7_cpu)

    # RuntimeError: 'Dot': Paired axes do not have same length. (OpSet 3) -> NC5-342
    pytest.mark.xfail(test_cases.test_resnet50_opset3_cpu)
    pytest.mark.xfail(test_cases.test_shufflenet_opset3_cpu)
    pytest.mark.xfail(test_cases.test_bvlc_alexnet_opset3_cpu)
    pytest.mark.xfail(test_cases.test_bvlc_caffenet_opset3_cpu)
    pytest.mark.xfail(test_cases.test_bvlc_googlenet_opset3_cpu)
    pytest.mark.xfail(test_cases.test_bvlc_rcnn_ilsvrc13_opset3_cpu)
    pytest.mark.xfail(test_cases.test_inception_v1_opset3_cpu)
    pytest.mark.xfail(test_cases.test_inception_v2_opset3_cpu)
    pytest.mark.xfail(test_cases.test_vgg19_opset3_cpu)
    pytest.mark.xfail(test_cases.test_zfnet512_opset3_cpu)

    # RuntimeError: sporadic result mismatch 0.1% -> NGONNX-346
    backend_test.exclude('test_resnet50_v2_opset7')
    backend_test.exclude('test_mobilenet_opset7')

    # RuntimeError: sporadic result mismatch 0.4% -> NGONNX-414
    backend_test.exclude('test_arcface_lresnet100e_opset8_cpu')

    # RuntimeError: unknown operation: ImageScaler
    backend_test.exclude('test_tiny_yolov2_opset7')
    backend_test.exclude('test_tiny_yolov2_opset8')

    # ONNX ValidationError
    backend_test.exclude('test_mnist_opset1')
    backend_test.exclude('test_tiny_yolov2_opset1')

    del test_cases
    globals().update(backend_test.enable_report().test_cases)
