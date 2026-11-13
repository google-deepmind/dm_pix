# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""SIMSE remains well-defined for a zero-valued comparison image."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_pix._src import metrics
import jax
import jax.numpy as jnp
import numpy as np


class ZeroReferenceSimseTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("eager", False), ("jitted", True)
  )
  def test_zero_comparison_returns_mean_square(self, jitted):
    a = jnp.array([1.0, -2.0, 3.0, 4.0], dtype=jnp.float32).reshape(2, 2, 1)
    b = jnp.zeros_like(a)
    fn = jax.jit(metrics.simse) if jitted else metrics.simse
    np.testing.assert_allclose(fn(a, b), jnp.mean(a**2), rtol=1e-6)

  def test_mixed_batch_is_handled_per_image(self):
    a = jnp.array([[[[1.0]]], [[[2.0]]], [[[3.0]]]])
    b = jnp.array([[[[0.0]]], [[[1.0]]], [[[0.0]]]])
    np.testing.assert_allclose(
        jax.jit(metrics.simse)(a, b),
        jnp.array([1.0, 0.0, 9.0]),
        atol=1e-6,
    )

  def test_nan_ignoring_preserves_valid_zero_reference_pixels(self):
    a = jnp.array([1.0, 2.0, jnp.nan]).reshape(1, 3, 1)
    b = jnp.array([0.0, 0.0, jnp.nan]).reshape(1, 3, 1)
    np.testing.assert_allclose(
        jax.jit(lambda x, y: metrics.simse(x, y, ignore_nans=True))(a, b),
        2.5, rtol=1e-6,
    )

  def test_no_valid_pairs_remains_nan(self):
    a = jnp.array([jnp.nan, 1.0]).reshape(1, 2, 1)
    b = jnp.array([1.0, jnp.nan]).reshape(1, 2, 1)
    self.assertTrue(jnp.isnan(metrics.simse(a, b, ignore_nans=True)))

  def test_nonzero_comparison_remains_scale_invariant(self):
    a = jnp.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2, 1)
    b = 3.0 * a
    np.testing.assert_allclose(metrics.simse(a, b), 0.0, atol=1e-6)
    b_grad = jax.grad(lambda x: metrics.simse(a, x))(b)
    self.assertTrue(jnp.all(jnp.isfinite(b_grad)))

  def test_zero_reference_value_and_gradient_are_finite(self):
    a = jnp.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2, 1)
    b = jnp.zeros_like(a)
    value, (a_grad, b_grad) = jax.jit(
        jax.value_and_grad(metrics.simse, argnums=(0, 1))
    )(a, b)
    np.testing.assert_allclose(value, jnp.mean(a**2))
    np.testing.assert_allclose(a_grad, 2 * a / a.size)
    self.assertTrue(jnp.all(jnp.isfinite(b_grad)))


if __name__ == "__main__":
  absltest.main()
