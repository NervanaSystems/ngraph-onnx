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

import onnx  # noqa
import pytest

from ngraph_onnx.onnx_importer.backend import NgraphBackend
from tests.utils.model_zoo_tester import ModelZooTestRunner

_S3_DOWNLOAD_ONNX = 'https://s3.amazonaws.com/download.onnx/models/'
_S3_MODEL_ZOO = 'https://s3.amazonaws.com/onnx-model-zoo/'
_WINDOWS_NET = 'https://onnxzoo.blob.core.windows.net/models/'

zoo_models = {
    # ArcFace
    'arcface_lresnet100e_opset7': _S3_MODEL_ZOO + 'arcface/resnet100/resnet100.tar.gz',

    # BVLC AlexNet
    'bvlc_alexnet_opset7': _S3_DOWNLOAD_ONNX + 'opset_7/bvlc_alexnet.tar.gz',

    # BVLC GoogleNet
    'bvlc_googlenet_opset7': _S3_DOWNLOAD_ONNX + 'opset_7/bvlc_googlenet.tar.gz',

    # BVLC CaffeNet
    'bvlc_caffenet_opset7': _S3_DOWNLOAD_ONNX + 'opset_7/bvlc_reference_caffenet.tar.gz',

    # BVLC R-CNN ILSVRC13
    'bvlc_rcnn_ilsvrc13_opset7': _S3_DOWNLOAD_ONNX + 'opset_7/bvlc_reference_rcnn_ilsvrc13.tar.gz',

    # DenseNet-121
    'densenet121_opset7': _S3_DOWNLOAD_ONNX + 'opset_7/densenet121.tar.gz',

    # DUC
    'duc_resnet101_hdc_opset7': _S3_MODEL_ZOO + 'duc/ResNet101_DUC_HDC.tar.gz',

    # Emotion-FERPlus
    'emotion_ferplus_opset7': _WINDOWS_NET + 'opset_7/emotion_ferplus/emotion_ferplus.tar.gz',

    # Inception-v1
    'inception_v1_opset7': _S3_DOWNLOAD_ONNX + 'opset_7/inception_v1.tar.gz',

    # Inception-v2
    'inception_v2_opset7': _S3_DOWNLOAD_ONNX + 'opset_7/inception_v2.tar.gz',

    # MNIST
    'mnist_opset7': _WINDOWS_NET + 'opset_7/mnist/mnist.tar.gz',

    # Mobile Net
    'mobilenet_opset7': _S3_MODEL_ZOO + 'mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz',

    # ResNet-50
    'resnet50_opset7': _S3_DOWNLOAD_ONNX + 'opset_7/resnet50.tar.gz',

    # ResNet V2
    'resnet50_v2_opset7': _S3_MODEL_ZOO + 'resnet/resnet50v2/resnet50v2.tar.gz',

    # ShuffleNet
    'shufflenet_opset7': _S3_DOWNLOAD_ONNX + 'opset_7/shufflenet.tar.gz',

    # SqueezeNet
    'squeezenet_opset7': _S3_DOWNLOAD_ONNX + 'opset_7/squeezenet.tar.gz',

    # Tiny-YOLOv2
    'tiny_yolov2_opset7': _WINDOWS_NET + 'opset_7/tiny_yolov2/tiny_yolov2.tar.gz',

    # VGG-19
    'vgg19_opset7': _S3_DOWNLOAD_ONNX + 'opset_7/vgg19.tar.gz',

    # ZFNet-512
    'zfnet512_opset7': _S3_DOWNLOAD_ONNX + 'opset_7/zfnet512.tar.gz',
}
backend_name = pytest.config.getoption('backend', default='CPU')

if backend_name != 'INTERPRETER':
    # Set backend device name to be used instead of hardcoded by ONNX BackendTest class ones.
    NgraphBackend.backend_name = backend_name

    # import all test cases at global scope to make them visible to python.unittest
    backend_test = ModelZooTestRunner(NgraphBackend, zoo_models, __name__)
    test_cases = backend_test.test_cases['OnnxBackendZooModelTest']

    # Exclude failing tests...
    # RuntimeError: BatchNormalization: only 'spatial' mode is supported.
    pytest.mark.xfail(test_cases.test_duc_resnet101_hdc_opset7_cpu)

    # RuntimeError: Subtract: Argument element types are inconsistent. -> NC5-345
    pytest.mark.xfail(test_cases.test_arcface_lresnet100e_opset7_cpu)

    # RuntimeError: sporadic result mismatch 0.1% -> NC5-346
    pytest.mark.xfail(test_cases.test_resnet50_v2_opset7_cpu)
    pytest.mark.xfail(test_cases.test_mobilenet_opset7_cpu)

    # RuntimeError: unknown operation: ImageScaler
    backend_test.exclude('test_tiny_yolov2_opset7')

    del test_cases
    globals().update(backend_test.enable_report().test_cases)
