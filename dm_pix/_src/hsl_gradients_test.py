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
"""Regression tests for achromatic HSL gradient handling."""

import colorsys
from absl.testing import absltest
from absl.testing import parameterized
from dm_pix._src import color_conversion
import jax
import jax.numpy as jnp
import numpy as np


class HslGradientsTest(parameterized.TestCase):

  @parameterized.product(
      dtype=(jnp.float16, jnp.bfloat16, jnp.float32),
      compiled=(False, True),
  )
  def test_greyscale_path_has_finite_lightness_gradients(self, dtype, compiled):
    def grey(level):
      return color_conversion.rgb_to_hsl(jnp.repeat(level[None], 3))

    reverse = jax.jacrev(grey)
    forward = jax.jacfwd(grey)
    if compiled:
      grey, reverse, forward = map(jax.jit, (grey, reverse, forward))
    for level in (0.0, 0.25, 0.5, 1.0):
      x = jnp.array(level, dtype=dtype)
      np.testing.assert_array_equal(grey(x), np.array([0.0, 0.0, level]))
      np.testing.assert_allclose(reverse(x), [0.0, 0.0, 1.0], atol=1e-3)
      np.testing.assert_allclose(forward(x), [0.0, 0.0, 1.0], atol=1e-3)
      jacobian = jax.jacrev(color_conversion.rgb_to_hsl)(jnp.repeat(x[None], 3))
      self.assertTrue(np.isfinite(jacobian).all())
      # The existing achromatic hue/saturation branches are constants.
      np.testing.assert_array_equal(jacobian[:2], np.zeros((2, 3)))

  @parameterized.parameters(False, True)
  def test_mixed_image_values_and_gradients(self, compiled):
    image = jnp.array(
        [[[0.0, 0.0, 0.0], [0.2, 0.6, 0.9]], [[0.5, 0.5, 0.5], [0.8, 0.3, 0.1]]]
    )
    f = color_conversion.rgb_to_hsl
    gradient = jax.grad(lambda x: jnp.sum(f(x)))
    if compiled:
      f, gradient = jax.jit(f), jax.jit(gradient)
    expected = np.array(
        [
            colorsys.rgb_to_hls(*pixel)
            for pixel in np.asarray(image).reshape(-1, 3)
        ]
    )
    expected = expected[:, [0, 2, 1]].reshape(image.shape)
    np.testing.assert_allclose(f(image), expected, rtol=2e-6, atol=2e-7)
    self.assertTrue(np.isfinite(gradient(image)).all())

  def test_chromatic_gradients_match_finite_differences(self):
    x = jnp.array([0.2, 0.6, 0.9])
    numerical = np.zeros((3, 3))
    x64 = np.asarray(x).astype(np.float64)
    for index in range(3):
      delta = np.zeros(3)
      delta[index] = 1e-5
      upper = np.array(colorsys.rgb_to_hls(*(x64 + delta)))[[0, 2, 1]]
      lower = np.array(colorsys.rgb_to_hls(*(x64 - delta)))[[0, 2, 1]]
      numerical[:, index] = (upper - lower) / (2e-5)
    np.testing.assert_allclose(
        jax.jacrev(color_conversion.rgb_to_hsl)(x),
        numerical,
        rtol=2e-5,
        atol=1e-6,
    )


if __name__ == '__main__':
  absltest.main()
