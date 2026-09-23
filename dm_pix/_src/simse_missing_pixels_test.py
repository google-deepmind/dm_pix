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
"""Regression tests for fitting SIMSE on paired valid pixels."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import dm_pix
import jax
import jax.numpy as jnp
import numpy as np


def _reference(a, b):
  valid = ~np.isnan(a) & ~np.isnan(b)
  a, b = a[valid], b[valid]
  scale = np.dot(a, b) / np.dot(b, b)
  return np.mean((a - scale * b) ** 2)


class SIMSEMissingPixelsTest(parameterized.TestCase):

  @parameterized.product(
      mask=("first", "second", "different", "matching", "none"),
      batched=(False, True),
      compiled=(False, True),
  )
  def test_matches_paired_least_squares(self, mask, batched, compiled):
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    a = np.linspace(0.1, 0.9, 12, dtype=dtype).reshape((2, 3, 2))
    b = np.linspace(0.9, 0.2, 12, dtype=dtype).reshape((2, 3, 2))
    if mask in ("first", "different", "matching"):
      a[0, 0, 0] = np.nan
    if mask in ("second", "matching"):
      b[0, 0, 0] = np.nan
    elif mask == "different":
      b[1, 2, 1] = np.nan
    if batched:
      a = np.stack((a, np.flip(a), np.full_like(a, 0.3)))
      b = np.stack((b, np.flip(b) * 0.4, np.full_like(b, 0.6)))
      expected = np.array([_reference(x, y) for x, y in zip(a, b)])
    else:
      expected = _reference(a, b)
    metric = functools.partial(dm_pix.simse, ignore_nans=True)
    metric = jax.jit(metric) if compiled else metric
    actual = metric(jnp.asarray(a), jnp.asarray(b))
    self.assertEqual(actual.shape, expected.shape)
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-5)
    # The optimal scale must absorb a change of units in the second image.
    np.testing.assert_allclose(
        metric(a, b * 0.4), expected, atol=1e-7, rtol=1e-5
    )

  def test_unpaired_values_do_not_affect_fit(self):
    a = jnp.array([np.nan, 0.5]).reshape(1, 2, 1)
    for ignored_value in (0.0, 0.5, 1.0):
      b = jnp.array([ignored_value, 0.5]).reshape(1, 2, 1)
      np.testing.assert_allclose(
          dm_pix.simse(a, b, ignore_nans=True), 0.0, atol=1e-8
      )

  def test_no_paired_pixels_remains_undefined(self):
    a = jnp.array([np.nan, 0.5]).reshape(1, 2, 1)
    b = jnp.array([0.5, np.nan]).reshape(1, 2, 1)
    self.assertTrue(jnp.isnan(dm_pix.simse(a, b, ignore_nans=True)))

  def test_nan_propagation_without_ignore_is_unchanged(self):
    a = jnp.array([np.nan, 0.5]).reshape(1, 2, 1)
    b = jnp.ones_like(a)
    self.assertTrue(jnp.isnan(dm_pix.simse(a, b)))

  def test_finite_input_gradients_are_unchanged(self):
    a = jnp.linspace(0.1, 0.9, 12).reshape((2, 3, 2))
    b = jnp.linspace(0.8, 0.2, 12).reshape((2, 3, 2))
    expected = jax.grad(dm_pix.simse, argnums=(0, 1))(a, b)
    actual = jax.jit(
        jax.grad(
            functools.partial(dm_pix.simse, ignore_nans=True), argnums=(0, 1)
        )
    )(a, b)
    for actual_gradient, expected_gradient in zip(actual, expected):
      np.testing.assert_allclose(
          actual_gradient, expected_gradient, atol=1e-7, rtol=1e-5
      )


if __name__ == "__main__":
  absltest.main()
