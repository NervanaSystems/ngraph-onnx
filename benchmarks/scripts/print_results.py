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

import os
import numpy as np

data_dir = "/root/results"

paddle_data = [
    np.load(os.path.join(data_dir, 'bmark_paddlepaddle_native.npz')),
    np.load(os.path.join(data_dir, 'bmark_paddlepaddle_ngraph_cpu.npz')),
]
pytorch_data = [
    np.load(os.path.join(data_dir, 'bmark_pytorch_native.npz')),
    np.load(os.path.join(data_dir, 'bmark_pytorch_ngraph_cpu.npz')),
]
caffe2_data = [
    np.load(os.path.join(data_dir, 'bmark_caffe2_native.npz')),
    np.load(os.path.join(data_dir, 'bmark_caffe2_ngraph_cpu.npz')),
]
cntk_data = [
    np.load(os.path.join(data_dir, 'bmark_cntk_native.npz')),
    np.load(os.path.join(data_dir, 'bmark_cntk_ngraph_cpu.npz')),
]
plot_data = [[(1000*data['sys_time']).tolist() for data in paddle_data],
             [(1000*data['sys_time']).tolist() for data in pytorch_data],
             [(1000*data['sys_time']).tolist() for data in caffe2_data],
             [(1000*data['sys_time']).tolist() for data in cntk_data]]

labels = ['Paddle', 'Pytorch', 'Caffe2', 'CNTK']

print('Framework\tNative\t\tnGraph')
for (i, framework) in enumerate(plot_data):
    native_result = np.median(framework[0])
    ngraph_result = np.median(framework[1])

    print('{}\t\t{}\t{}'.format(labels[i], native_result, ngraph_result))
