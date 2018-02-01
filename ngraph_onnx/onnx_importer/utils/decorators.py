# ----------------------------------------------------------------------------
# Copyright 2018 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import logging
from functools import wraps
from typing import Callable


log = logging.getLogger(__file__)


def refactoring_required(function: Callable) -> Callable:
    """Mark this function as requiring refactoring of code from ngraph to ngraph++ API."""
    @wraps(function)
    def wrapper(*args, **kwds):  # type: ignore
        raise NotImplementedError('Function %s has not been refactored yet', function.__name__)
    return wrapper


def function_deprecated(function: Callable) -> Callable:
    """Mark this function as deprecated.

    The function should probably be removed during the switch from ngraph to ngraph++ API.
    """
    @wraps(function)
    def wrapper(*args, **kwds):  # type: ignore
        log.warning('Using deprecated function %s', function.__name__)
        function(*args, **kwds)
    return wrapper
