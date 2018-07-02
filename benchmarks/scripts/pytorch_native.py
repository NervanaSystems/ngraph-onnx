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
#
# ----------------------------------------------------------------------------
# Code in this file is based on an example from PyTorch/examples
# https://github.com/pytorch/examples/blob/master/imagenet/main.py
# distributed under the following license:
#
# BSD 3-Clause License
#
# Copyright (c) 2017,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# ----------------------------------------------------------------------------

import argparse
import os
import time

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np

from torch.autograd import Variable

import onnx
import torch.onnx

from utils import AverageMeter, save_results

model_name = 'resnet50'
WARM_UP_SIZE = 10

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--set_size', dest='size', type=int, help='Cardinality of data set.')
parser.add_argument('--export_onnx_model', help='Export pytorch model to ONNX.',
                    action='store_true')
parser.add_argument('--get_model', help='Download PyTorch ResNet50 pre-trained model.',
                    action='store_true')


def generate_data(count):
    """Return a list of torch.tensors each containing randomly generated data.

    :param count: Number of entries to generate.
    :return: A list of torch tensors.
    """
    np.random.seed(133391)
    image_height = 224
    image_width = 224
    image_channels = 3
    transform = transforms.ToTensor()
    return [(transform(np.array(np.random.randint(0, 256, (image_height, image_width,
                                                           image_channels)), dtype=np.uint8)), k)
            for k in range(count)]


def main():
    global args
    args = parser.parse_args()

    # create model
    print("=> using pre-trained model '{}'".format(model_name))
    model = models.resnet50(pretrained=True)

    if args.get_model:
        return

    if args.export_onnx_model:
        dataset = generate_data(1)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)
        onnx_input_images, _ = iter(data_loader).next()
        save_onnx_model(model, onnx_input_images, '~/models')
        return

    # Data loading code
    print('Using randomly generated data!')
    dataset = generate_data(args.size + WARM_UP_SIZE)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    perf_metrics = validate(data_loader, model)

    save_results('/root/results/', 'bmark_pytorch_native',
                 {key: val.data for key, val in perf_metrics.items()})


def validate(val_loader, model):
    batch_sys_time = AverageMeter()
    batch_proc_time = AverageMeter()
    batch_perf_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    print('Test: [{<evaluated images>}/{overall images count}]\t\n'
          'Time (sys) <current batch time>  (avg time/batch)\t'
          'Time (proc) <current batch time>  (avg time/batch)\t'
          'Time (perf) <current batch time>  (avg time/batch)')

    with torch.no_grad():
        # warm-up
        for i in range(WARM_UP_SIZE):
            img, _ = iter(val_loader).next()
            _ = model(img)

        iteration_count = 0
        while iteration_count < args.size:
            # Unix time (seconds since epoch). It is system-wide by definition
            clock_sys_start = time.time()
            #  the sum of the system and user CPU time of the current process.
            # It does not include time elapsed during sleep. It is process-wide
            clock_proc_start = time.process_time()
            # does include time elapsed during sleep and is system-wide,
            # clock with highest available resolution
            clock_perf_start = time.perf_counter()
            for i, (img, _) in enumerate(val_loader):
                # compute output
                output = model(img)

                # measure elapsed time
                batch_sys_time.update(time.time() - clock_sys_start)
                batch_proc_time.update(time.process_time() - clock_proc_start)
                batch_perf_time.update(time.perf_counter() - clock_perf_start)

                if i % args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time (sys) {batch_sys_time.val:.3f}s ({batch_sys_time.avg:.3f}s)\t'
                          'Time (proc) {batch_proc_time.val:.3f}s ({batch_proc_time.avg:.3f}s)\t'
                          'Time (perf) {batch_perf_time.val:.3f}s ({batch_perf_time.avg:.3f}s)\t'
                          ''.format(
                           i * args.batch_size, len(val_loader) * args.batch_size,
                           batch_sys_time=batch_sys_time, batch_proc_time=batch_proc_time,
                           batch_perf_time=batch_perf_time))

                iteration_count += 1
                if iteration_count == args.size:
                    break

                clock_sys_start = time.time()
                clock_proc_start = time.process_time()
                clock_perf_start = time.perf_counter()

    return {'sys_time': batch_sys_time, 'proc_time': batch_proc_time, 'perf_time': batch_perf_time}


def save_onnx_model(model, data, dest_dir):
    print('Exporting model to ONNX format')

    dest_path = os.path.join(os.path.expanduser(dest_dir), 'pytorch_resnet50.onnx')

    dummy_input = Variable(data)
    torch.onnx.export(model, dummy_input, dest_path)

    print('Validating ONNX model')
    onnx_model = onnx.load(dest_path)
    onnx.checker.check_model(onnx_model)


if __name__ == '__main__':
    main()
