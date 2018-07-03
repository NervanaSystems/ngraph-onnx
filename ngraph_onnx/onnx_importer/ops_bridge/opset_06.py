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

# Abs-6                     supported by opset_01.Abs-1
# Add-6                     supported by opset_01.Add-1
# BatchNormalization-6      supported by opset_01.BatchNormalization-1
# Cast-6                    supported by opset_01.Cast-1
# Ceil-6                    supported by opset_01.Ceil-1
# Clip-6                    supported by opset_01.Clip-1
# Div-6                     supported by opset_01.Div-1
# Dropout-6                 supported by opset_01.Dropout-1
# Elu-6                     supported by opset_01.Elu-1
# Exp-6                     supported by opset_01.Exp-1
# Floor-6                   supported by opset_01.Floor-1
# Gemm-6                    supported by opset_01.Gemm-1
# HardSigmoid-6             supported by opset_01.HardSigmoid-1
# TODO: InstanceNormalization-6
# LeakyRelu-6               supported by opset_01.LeakyRelu-1
# Log-6                     supported by opset_01.Log-1
# Max-6                     supported by opset_01.Max-1
# Mean-6                    supported by opset_01.Mean-1
# Min-6                     supported by opset_01.Min-1
# Mul-6                     supported by opset_01.Mul-1
# Neg-6                     supported by opset_01.Neg-1
# PRelu-6                   supported by opset_01.PRelu-1
# Reciprocal-6              supported by opset_01.Reciprocal-1
# Relu-6                    supported by opset_01.Relu-1
# Selu-6                    supported by opset_01.Selu-1
# Sigmoid-6                 supported by opset_01.Sigmoid-1
# Sqrt-6                    supported by opset_01.Sqrt-1
# Sub-6                     supported by opset_01.Sub-1
# Sum-6                     supported by opset_01.Sum-1
# Tanh-6                    supported by opset_01.Tanh-1
# TODO: Tile-6

from ngraph_onnx.onnx_importer.ops_bridge.opset_05 import *  # noqa
