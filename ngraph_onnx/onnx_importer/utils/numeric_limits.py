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
from __future__ import unicode_literals


import numpy as np
import numbers
from typing import Union


class NumericLimits(object):
    """Class providing interface to extract numerical limits for given data type."""

    @staticmethod
    def _get_number_limits_class(dtype):
        # type: (np.dtype) -> Union[IntegralLimits, FloatingPointLimits]
        """Return specialized class instance with limits set for given data type.

        :param dtype: The data type we want to check limits for.
        :return: The specialized class instance providing numeric limits.
        """
        data_type = dtype.type
        value = data_type(1)
        if isinstance(value, numbers.Integral):
            return IntegralLimits(data_type)
        elif isinstance(value, numbers.Real):
            return FloatingPointLimits(data_type)
        else:
            raise ValueError('NumericLimits: unsupported data type: <{}>.'.format(dtype.type))

    @staticmethod
    def _get_dtype(dtype):  # type: (Union[np.dtype, int, float]) -> np.dtype
        """Return numpy dtype object wrapping provided data type.

        :param dtype: The data type to be wrapped.
        :return: The numpy dtype object.
        """
        return dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)

    @classmethod
    def max(cls, dtype):  # type: (np.dtype) -> Union[int, float]
        """Return maximum value that can be represented in given data type.

        :param dtype: The data type we want to check maximum value for.
        :return: The maximum value.
        """
        return cls._get_number_limits_class(cls._get_dtype(dtype)).max

    @classmethod
    def min(cls, dtype):  # type: (np.dtype) -> Union[int, float]
        """Return minimum value that can be represented in given data type.

        :param dtype: The data type we want to check minimum value for.
        :return: The minimum value.
        """
        return cls._get_number_limits_class(cls._get_dtype(dtype)).min


class FloatingPointLimits(object):
    """Class providing access to numeric limits for floating point data types."""

    def __init__(self, data_type):  # type: (type) -> None
        self.data_type = data_type

    @property
    def max(self):  # type: () -> float
        """Provide maximum representable value by stored data type.

        :return: The maximum value.
        """
        return np.finfo(self.data_type).max

    @property
    def min(self):  # type: () -> float
        """Provide minimum representable value by stored data type.

        :return: The minimum value.
        """
        return np.finfo(self.data_type).min


class IntegralLimits(object):
    """Class providing access to numeric limits for integral data types."""

    def __init__(self, data_type):  # type: (type) -> None
        self.data_type = data_type

    @property
    def max(self):  # type: () -> int
        """Provide maximum representable value by stored data type.

        :return: The maximum value.
        """
        return np.iinfo(self.data_type).max

    @property
    def min(self):  # type: () -> int
        """Provide minimum representable value by stored data type.

        :return: The minimum value.
        """
        return np.iinfo(self.data_type).min
