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
"""SSIM covariance gradients must pass through zero correlation."""

from absl.testing import absltest
from absl.testing import parameterized
import dm_pix
import jax
import jax.numpy as jnp
import numpy as np


def mean_filter(x):
  return jnp.mean(x, axis=(-3, -2), keepdims=True)


def smooth_ssim(a, b):
  ma, mb = mean_filter(a), mean_filter(b)
  va, vb = mean_filter((a - ma) ** 2), mean_filter((b - mb) ** 2)
  covariance = mean_filter((a - ma) * (b - mb))
  # These fixtures have valid covariance, so no clipping is required.
  result = (
      (2 * ma * mb + 0.01**2)
      * (2 * covariance + 0.03**2)
      / ((ma**2 + mb**2 + 0.01**2) * (va + vb + 0.03**2))
  )
  return result.mean(axis=(-3, -2, -1))


def images(batch=False, channels=1):
  a = jnp.array([[0.75, 0.75], [0.25, 0.25]])[..., None]
  b = jnp.array([[0.75, 0.25], [0.75, 0.25]])[..., None]
  a, b = jnp.repeat(a, channels, -1), jnp.repeat(b, channels, -1)
  if batch:
    a, b = jnp.stack([a, a]), jnp.stack([b, b])
  return a, b


class SsimCovarianceGradientTest(parameterized.TestCase):

  @parameterized.product(
      batch=[False, True], channels=[1, 3], custom_filter=[False, True]
  )
  def test_zero_covariance_gradient_matches_independent_formula(
      self, batch, channels, custom_filter
  ):
    a, b = images(batch, channels)
    kwargs = (
        {"filter_fn": mean_filter}
        if custom_filter
        else {"filter_size": 2, "filter_sigma": 1.0}
    )
    objective = lambda x: dm_pix.ssim(a, x, **kwargs).sum()
    reference = lambda x: smooth_ssim(a, x).sum()
    expected = jax.value_and_grad(reference)(b)
    for fn in (
        jax.value_and_grad(objective),
        jax.jit(jax.value_and_grad(objective)),
    ):
      actual = fn(b)
      for value, target in zip(actual, expected):
        np.testing.assert_allclose(value, target, rtol=3e-6, atol=1e-7)
    _, tangent = jax.jvp(objective, (b,), (a - 0.5,))
    step = 0.01
    finite_diff = (
        objective(b + step * (a - 0.5)) - objective(b - step * (a - 0.5))
    ) / (2 * step)
    np.testing.assert_allclose(tangent, finite_diff, rtol=5e-4, atol=1e-6)

  def test_default_gaussian_passes_gradients_at_a_zero_second_image(self):
    a, b = images()
    b = jnp.zeros_like(b)
    actual = jax.jit(
        jax.value_and_grad(lambda x: dm_pix.ssim(a, x, filter_size=2))
    )(b)
    expected = jax.value_and_grad(lambda x: smooth_ssim(a, x))(b)
    for value, target in zip(actual, expected):
      np.testing.assert_allclose(value, target, rtol=3e-6, atol=1e-8)

  @parameterized.parameters(-0.5, 0.5)
  def test_nonzero_covariance_values_and_gradients_are_preserved(self, weight):
    a, b = images()
    b = b + weight * (a - 0.5)
    actual = jax.value_and_grad(
        lambda x: dm_pix.ssim(a, x, filter_fn=mean_filter)
    )(b)
    expected = jax.value_and_grad(lambda x: smooth_ssim(a, x))(b)
    for value, target in zip(actual, expected):
      np.testing.assert_allclose(value, target, rtol=3e-6, atol=1e-7)

  def test_correlation_parameter_can_learn_from_zero(self):
    a, b = images()
    score = lambda parameter: dm_pix.ssim(
        a, b + parameter * (a - 0.5), filter_size=2, filter_sigma=1.0
    )
    parameter = jnp.array(0.0)
    derivative = jax.jit(jax.grad(score))(parameter)
    self.assertGreater(float(derivative), 0.9)
    self.assertGreater(
        float(score(parameter + 0.1 * derivative)), float(score(parameter))
    )

  @parameterized.parameters(False, True)
  def test_map_and_identity_scores_keep_existing_values(self, negative):
    a, _ = images(batch=True)
    b = 1 - a if negative else a
    actual = dm_pix.ssim(a, b, filter_size=2, return_map=True)
    expected = smooth_ssim(a, b)
    np.testing.assert_allclose(
        actual.mean(axis=(-3, -2, -1)), expected, rtol=3e-6, atol=1e-7
    )
    self.assertEqual(actual.shape, (2, 1, 1, 1))


if __name__ == "__main__":
  absltest.main()
