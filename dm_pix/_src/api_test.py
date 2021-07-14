# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for dm_pix API."""

import inspect

from absl.testing import absltest
from absl.testing import parameterized
import dm_pix as pix


def named_pix_functions():
  fns = []
  for name in dir(pix):
    o = getattr(pix, name)
    if inspect.isfunction(o):
      fns.append((name, o))
  return fns


class ApiTest(parameterized.TestCase):

  @parameterized.named_parameters(*named_pix_functions())
  def test_key_argument(self, f):
    sig = inspect.signature(f)
    param_names = tuple(sig.parameters)
    self.assertNotIn("rng", param_names,
                     "Prefer `key` to `rng` in PIX (following JAX).")
    if "key" in param_names:
      self.assertLess(
          param_names.index("key"), param_names.index("image"),
          "RNG `key` argument should be before `image` in PIX.")

  @parameterized.named_parameters(*named_pix_functions())
  def test_kwarg_only_defaults(self, f):
    argspec = inspect.getfullargspec(f)
    if f.__name__ == "rot90":
      # Special case for `k` in rot90.
      self.assertLen(argspec.defaults, 1)
      return

    self.assertEmpty(
        argspec.defaults or (),
        "Optional keyword arguments in PIX should be keyword "
        "only. Prefer `f(x, *, axis=-1)` to `f(x, axis=-1)`.")


if __name__ == "__main__":
  absltest.main()
