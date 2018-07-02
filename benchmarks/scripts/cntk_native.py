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

import argparse
import time
import os
import numpy as np

import onnx
import cntk

from utils import AverageMeter, save_results, generate_data

# ------------------------------------------------------------------------------
#  CONFIGURATION
# ------------------------------------------------------------------------------

CNTK_MODELS_DIR = "~/cntk_models"

MODEL_NAME = 'ResNet50_ImageNet_CNTK.model'

WARM_UP_SIZE = 10

# ------------------------------------------------------------------------------
#  COMMAND LINE ARGUMENTS
# ------------------------------------------------------------------------------


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N',
                    help='mini-batch size (default: 1)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--set_size', dest='size', type=int, help='Cardinality of data set.')


# ------------------------------------------------------------------------------
#  UTILITIES
# ------------------------------------------------------------------------------


class CNTKNetModel(object):

    def __init__(self, model_name, models_dir):
        model_path = os.path.join(os.path.expanduser(models_dir), model_name)
        self.net = cntk.Function.load(model_path, device=cntk.device.cpu())

    def run(self, data):
        labels = np.zeros(1000)
        return self.net.eval({self.net.arguments[0]: [data],
                              self.net.arguments[1]: [labels]})


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
        for i, (img) in enumerate(dataset):
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
    model = CNTKNetModel(MODEL_NAME, CNTK_MODELS_DIR)

    # Data loading code
    print('Using randomly generated data!')
    dataset = generate_data(args.size)

    perf_metrics = benchmark_inference(dataset, model)
    save_results('/root/results/', 'bmark_cntk_native',
                 {key: val.data for key, val in perf_metrics.items()})


if __name__ == '__main__':
    main()
