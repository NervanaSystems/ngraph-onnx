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

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import argparse
import time
import os

import onnx
from caffe2.proto.caffe2_pb2 import NetDef
from caffe2.python import core, workspace

from utils import AverageMeter, save_results, generate_data
# ------------------------------------------------------------------------------
#  CONFIGURATION
# ------------------------------------------------------------------------------

CAFFE_MODELS_DIR = "~/caffe2_models"

# What model are we using?
#    Format below is the model's: <folder, input image size>
MODEL_NAME = 'resnet50'

# codes - these help decypher the output and source from a list from ImageNet's object codes
#    to provide an result like "tabby cat" or "lemon" depending on what's in the picture
#   you submit to the CNN.
# codes = 'https://gist.githubusercontent.com/aaronmarkham/cd3a6b6ac071eca6f7b4a6e40e6038aa/raw/' \
#          '9edb4038a37da6b5a44c3b5bc52e448ff09bfe5b/alexnet_codes'

WARM_UP_SIZE = 10

# ------------------------------------------------------------------------------
#  COMMAND LINE ARGUMENTS
# ------------------------------------------------------------------------------


parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference benchmark')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N',
                    help='mini-batch size (default: 1)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--set_size', dest='size', type=int, help='Cardinality of data set.')
parser.add_argument('--export_onnx_model', help='Export caffe2 model to ONNX.',
                    action='store_true')

# ------------------------------------------------------------------------------
#  UTILITIES
# ------------------------------------------------------------------------------


class Caffe2NetModel(object):

    def __init__(self, model_name, models_dir):
        self.init_net_pb, self.predict_net_pb = self._load_model(model_name,
                                                                 os.path.expanduser(models_dir))
        # define blobs for input
        workspace.FeedBlob("gpu_0/data", np.random.rand(1, 3, 224, 224).astype(np.float32))

        workspace.RunNetOnce(self.init_net_pb)
        self.predict_net = core.Net(self.predict_net_pb)
        workspace.CreateNet(self.predict_net)

    def _load_model(self, model_name, models_dir):
        init_net_path = os.path.join(models_dir, model_name, 'init_net.pb')
        predict_net_path = os.path.join(models_dir, model_name, 'predict_net.pb')

        if not os.path.exists(init_net_path):
            print("WARNING: " + init_net_path + " not found!")
        else:
            if not os.path.exists(predict_net_path):
                print("WARNING: " + predict_net_path + " not found!")
            else:
                print("All needed files found!")

        # Read the contents of the input protobufs into local variables
        init_net = NetDef()
        predict_net = NetDef()
        with open(init_net_path, 'rb') as f:
            init_net.ParseFromString(f.read())
        with open(predict_net_path, 'rb') as f:
            predict_net.ParseFromString(f.read())

        return init_net, predict_net

    def run(self, data):
        workspace.FeedBlob('gpu_0/data', data)
        workspace.RunNet(self.predict_net.Proto().name)
        return np.asarray(workspace.FetchBlob('gpu_0/softmax'))


def save_onnx_model(model, dest_dir):
    # import caffe2.python.onnx.frontend as c2_onnx_frontend
    import onnx_caffe2.frontend as c2_onnx_frontend

    dest_path = os.path.join(os.path.expanduser(dest_dir), 'caffe2_resnet50.onnx')

    data_type = onnx.TensorProto.FLOAT
    data_shape = (1, 3, 224, 224)
    value_info = {
        'gpu_0/data': (data_type, data_shape)
    }

    onnx_model = c2_onnx_frontend.caffe2_net_to_onnx_model(
        model.predict_net_pb,
        model.init_net_pb,
        value_info,
    )
    onnx.checker.check_model(onnx_model)

    with open(dest_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
        

def benchmark_inference(dataset, model):
    batch_sys_time = AverageMeter()

    print('Test: [{<evaluated images>}/{overall imgs count}]\t\n'
          'Time (sys) <current batch time>  (avg time/batch)\t')

    # warm-up
    for i in range(WARM_UP_SIZE):
        _ = model.run(dataset[0][0])

    iteration_count = 0
    while iteration_count < args.size:
        # Unix time (seconds since epoch). It is system-wide by definition
        clock_sys_start = time.time()
        for i, (img, _) in enumerate(dataset):
            _ = model.run(img)

            # measure elapsed time
            batch_sys_time.update(time.time() - clock_sys_start)
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time (sys) {batch_sys_time.val:.3f}s ({batch_sys_time.avg:.3f}s)\t'
                      ''.format(
                       i * args.batch_size, len(dataset) * args.batch_size,
                       batch_sys_time=batch_sys_time))

            iteration_count += 1
            if iteration_count == args.size:
                break

            clock_sys_start = time.time()

    return {'sys_time': batch_sys_time}


# ------------------------------------------------------------------------------
#  MAIN LOGIC
# ------------------------------------------------------------------------------


def main():
    global args
    args = parser.parse_args()

    # create model
    print("=> using pre-trained model '{}'".format(MODEL_NAME))
    model = Caffe2NetModel(MODEL_NAME, CAFFE_MODELS_DIR)

    if args.export_onnx_model:
        save_onnx_model(model, '~/models')
        return

    # Data loading code
    print('Using randomly generated data!')
    dataset = generate_data(args.size)
    dataset = [(img, idx) for idx, img in enumerate(dataset)]

    perf_metrics = benchmark_inference(dataset, model)
    save_results('/root/results/', 'bmark_caffe2_native',
                 {key: val.data for key, val in perf_metrics.items()})


if __name__ == '__main__':
    main()
