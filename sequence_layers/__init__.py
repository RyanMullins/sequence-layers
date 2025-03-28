# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Streaming Sequence Layers."""

# TODO(dthkao): Remove the module imports when users are migrated to new deps.
# Alias the module names as well, so that the redirect that exists for legacy
# module imports functions correctly.
from . import attention
from . import combinators
from . import conditioning
from . import convolution
from . import dense
from . import dsp
from . import normalization
from . import pooling
from . import position
from . import recurrent
from . import simple
from . import squeeze
from . import time_varying
from . import types

# pylint: disable=wildcard-import
# TODO(rryan): Don't wildcard import, or define __all__ explicitly.
from ..attention import *
from ..combinators import *
from ..conditioning import *
from ..convolution import *
from ..dense import *
from ..dsp import *
from ..normalization import *
from ..pooling import *
from ..position import *
from ..recurrent import *
from ..simple import *
from ..squeeze import *
from ..time_varying import *
from ..types import *
# pylint: enable=wildcard-import
