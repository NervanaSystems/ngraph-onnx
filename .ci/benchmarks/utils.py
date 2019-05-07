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


def save_results(dest_dir, filename, kwargs):
    # currently it is assumed that data is an array object with timings.
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except OSError:
        pass
    np.savez(os.path.join(dest_dir, filename), **kwargs)


def generate_data(count, batch_size=1, image_channels=3, image_height=224, image_width=224):
    """Return a list of numpy ndarrays each containing randomly generated data.

    :param count: Number of entries to generate.
    :param image_height:
    :param image_width:
    :param image_channels:
    :param batch_size:
    :return: A list of torch tensors.
    """
    return [np.random.rand(batch_size, image_channels, image_height, image_width).astype(np.float32)
            for k in range(count)]


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()
        self.data = []  # type: list

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.data.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
