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
from typing import Sequence, Tuple

from ngraph_onnx import TYPE_CHECKING

if TYPE_CHECKING:
    from ngraph_onnx.onnx_importer.model_wrappers import NodeWrapper


def split_pads_into_pairs(pads):  # type: (Sequence[int]) -> Tuple[Sequence[int], ...]
    """
    Convert ONNX padding format to ngraph padding format.

    :param pads: ONNX `pads` format: [x1_begin, x2_begin..., x1_end, x2_end,...]
    :return: ngraph format: [(x1_begin, x1_end), (x2_begin, x2_end), ...]
    """
    if not pads:
        return ()

    first_end_pad_index = int(len(pads) / 2)
    begin_pads = pads[:first_end_pad_index]
    end_pads = pads[first_end_pad_index:]
    return (begin_pads, end_pads)
