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

import argparse
import numpy as np
import time
import os

import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def parse_args():
    parser = argparse.ArgumentParser('Convolution model benchmark.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='The minibatch size.')
    parser.add_argument(
        '--skip_batch_num',
        type=int,
        default=0,
        help='The number of the first minibatches to skip in statistics, for better performance test.')
    parser.add_argument(
        '--iterations',
        type=int,
        default=0,
        help='The number of minibatches to process. 0 or less: whole dataset. Greater than 0: cycle the dataset if needed.')
    parser.add_argument(
        '--data_format',
        type=str,
        default='NCHW',
        choices=['NCHW', 'NHWC'],
        help='The data data_format, now only support NCHW.')
    parser.add_argument(
        '--device',
        type=str,
        default='CPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    parser.add_argument(
        '--data_set',
        type=str,
        default='flowers',
        choices=['cifar10', 'flowers', 'imagenet'],
        help='Optional dataset for 1q.')
    parser.add_argument(
        '--profile',
        action='store_true',
        help='If set, do profiling.')
    parser.add_argument(
        '--use_mkldnn',
        action='store_true',
        help='If set, use mkldnn library for speed up.')
    parser.add_argument(
        '--infer_model_path',
        type=str,
        default='',
        help='The directory for loading inference model.')

    args = parser.parse_args()
    return args


def user_data_reader(data):
    '''
    Creates a data reader for user data.
    '''

    def data_reader():
        while True:
            for b in data:
                yield b
    return data_reader


def infer(args):
    if not os.path.exists(args.infer_model_path):
        raise IOError("Invalid inference model path!")

    if args.data_set == "cifar10":
        class_dim = 10
        if args.data_format == 'NCHW':
            dshape = [3, 32, 32]
        else:
            dshape = [32, 32, 3]
    elif args.data_set == "imagenet":
        class_dim = 1000
        if args.data_format == 'NCHW':
            dshape = [3, 224, 224]
        else:
            dshape = [224, 224, 3]
    else:
        class_dim = 102
        if args.data_format == 'NCHW':
            dshape = [3, 224, 224]
        else:
            dshape = [224, 224, 3]

    generated_data = [(np.random.rand(dshape[0] * dshape[1] * dshape[2]).
                  astype(np.float32), np.random.randint(1, class_dim))
                 for _ in range(200)]

    image = fluid.layers.data(name='data', shape=dshape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    place = fluid.CUDAPlace(0) if args.device == 'GPU' else fluid.CPUPlace()
    exe = fluid.Executor(place)
    paddle.init(use_gpu=False, use_mkldnn=args.use_mkldnn)

    # load model
    [infer_program, feed_dict,
     fetch_targets] = fluid.io.load_inference_model(args.infer_model_path, exe)

    # infer data read
    infer_reader = paddle.batch(
        user_data_reader(generated_data),
        batch_size = args.batch_size)

    iters = 0
    batch_times = []
    start = time.time()
    for data in infer_reader():
        if args.iterations > 0 and iters == args.iterations + args.skip_batch_num:
            break
        if iters == args.skip_batch_num:
            profiler.reset_profiler()
        image = np.array(map(lambda x: x[0].reshape(dshape),
                                data)).astype("float32")
        label = np.array(map(lambda x: x[1], data)).astype("int64")
        label = label.reshape([-1, 1])
        predicts = exe.run(infer_program,
                      feed={feed_dict[0]:image},
                      fetch_list=fetch_targets)


        batch_time = time.time() - start
        if iters % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time (sys) {batch_sys_time:.3f}s\t'
                  ''.format(
                   iters * args.batch_size, args.iterations * args.batch_size,
                   batch_sys_time=batch_time))
        start = time.time()
        fps = args.batch_size / batch_time
        batch_times.append(batch_time)
        iters += 1

    # Postprocess benchmark data
    latencies = batch_times[args.skip_batch_num:]
    latency_avg = np.average(latencies)
    latency_pc99 = np.percentile(latencies, 99)
    fpses = np.divide(args.batch_size, latencies)
    fps_avg = np.average(fpses)
    fps_pc99 = np.percentile(fpses, 1)

    np.savez("/root/results/bmark_paddlepaddle_native.npz", sys_time=latencies)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    if args.data_format == 'NHWC':
        raise ValueError('Only support NCHW data_format now.')
    if args.profile:
        if args.device == 'GPU':
            with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
                infer(args)
        else:
            with profiler.profiler(args.device, sorted_key='total') as cpuprof:
                infer(args)
    else:
        infer(args)
