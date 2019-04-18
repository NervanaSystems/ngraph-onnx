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

import argparse
import time
import os

from utils import AverageMeter, save_results, generate_data
from ngraph_onnx.onnx_importer.importer import import_onnx_file
import ngraph as ng

import logging
logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser(description='nGraph ImageNet inference benchmark')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='Print frequency (default: 10).')
parser.add_argument('model', help='Path to model to run benchmark on.', type=str)
parser.add_argument('backend', help='nGraph backend to run computations on.', type=str)
parser.add_argument('--batch_size', default=1, type=int, help='Size of batch.')
parser.add_argument('--set_size', dest='size', type=int, help='Size of generated data set.')
parser.add_argument('--output_file', help='Results output file name.')

WARM_UP_SIZE = 10


def evaluate(backend_name, model_path, dataset):
    ng_model = import_onnx_file(model_path)[0]
    runtime = ng.runtime(backend_name=backend_name)
    computation = runtime.computation(ng_model['output'], *ng_model['inputs'])

    batch_sys_time = AverageMeter()
    batch_proc_time = AverageMeter()
    batch_perf_time = AverageMeter()

    # warm-up
    for idx, (img, _) in enumerate(dataset):
        computation(img)
        if idx == WARM_UP_SIZE:
            break

    # Unix time (seconds since epoch). It is system-wide by definition
    clock_sys_start = time.time()
    #  the sum of the system and user CPU time of the current process.
    # It does not include time elapsed during sleep. It is process-wide
    clock_proc_start = time.process_time()
    # does include time elapsed during sleep and is system-wide,
    # clock with highest available resolution
    clock_perf_start = time.perf_counter()

    for i, (batch, _) in enumerate(dataset):
        computation(batch)

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
                   i * args.batch_size, len(dataset) * args.batch_size,
                   batch_sys_time=batch_sys_time, batch_proc_time=batch_proc_time,
                   batch_perf_time=batch_perf_time))

        clock_sys_start = time.time()
        clock_proc_start = time.process_time()
        clock_perf_start = time.perf_counter()

    return {'sys_time': batch_sys_time, 'proc_time': batch_proc_time, 'perf_time': batch_perf_time}



def main():
    global args
    args = parser.parse_args()

    dataset = generate_data(args.size)
    dataset = [(img, 0) for img in dataset]

    perf_metrics = evaluate(args.backend, args.model, dataset)
    save_results('results/', args.output_file, {key: val.data for key, val in perf_metrics.items()})


if __name__ == '__main__':
    main()

