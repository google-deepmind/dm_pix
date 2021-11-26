# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Testing utilities for PIX."""

import inspect
import types
from typing import Sequence, Tuple


def get_public_functions(
    root_module: types.ModuleType) -> Sequence[Tuple[str, types.FunctionType]]:
  """Returns `(function_name, function)` for all functions of `root_module`."""
  fns = []
  for name in dir(root_module):
    o = getattr(root_module, name)
    if inspect.isfunction(o):
      fns.append((name, o))
  return fns


def get_public_symbols(
    root_module: types.ModuleType) -> Sequence[Tuple[str, types.FunctionType]]:
  """Returns `(symbol_name, symbol)` for all symbols of `root_module`."""
  fns = []
  for name in getattr(root_module, '__all__'):
    o = getattr(root_module, name)
    fns.append((name, o))
  return fns
