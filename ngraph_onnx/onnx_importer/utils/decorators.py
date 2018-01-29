# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
    @wraps(function)
    def wrapper(*args, **kwds):
        raise NotImplementedError('Function %s has not been refactored yet', function.__name__)
    return wrapper


def function_deprecated(function: Callable) -> Callable:
    @wraps(function)
    def wrapper(*args, **kwds):
        log.warning('Using deprecated function %s', function.__name__)
        function(*args, **kwds)
    return wrapper
