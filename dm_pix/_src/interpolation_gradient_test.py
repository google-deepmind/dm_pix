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
"""Coordinate gradients of linear interpolation at exact grid points."""

from absl.testing import absltest
from absl.testing import parameterized
import dm_pix
import jax
import jax.numpy as jnp
import numpy as np


class InterpolationGradientTest(parameterized.TestCase):

  @parameterized.product(
      ndim=(1, 2, 3), flattened=(False, True), constant=(False, True)
  )
  def test_affine_ramp_gradient(self, ndim, flattened, constant):
    shape = (5,) * ndim
    slopes = jnp.arange(1, ndim + 1, dtype=jnp.float32) / 32
    grid = jnp.indices(shape, dtype=jnp.float32)
    volume = jnp.einsum("i,i...->...", slopes, grid)
    coordinates = jnp.full((ndim,), 2.0, dtype=jnp.float32)
    fn = (
        dm_pix.flat_nd_linear_interpolate_constant
        if constant
        else dm_pix.flat_nd_linear_interpolate
    )
    kwargs = {"unflattened_vol_shape": shape} if flattened else {}
    values = volume.reshape(-1) if flattened else volume

    def interpolate(x):
      return fn(values, x, **kwargs)

    expected = jnp.dot(slopes, coordinates)
    np.testing.assert_allclose(interpolate(coordinates), expected, atol=1e-7)
    for grad in (jax.grad(interpolate), jax.jit(jax.grad(interpolate))):
      np.testing.assert_allclose(grad(coordinates), slopes, atol=1e-7)
    tangent = jnp.ones(ndim)
    _, derivative = jax.jvp(interpolate, (coordinates,), (tangent,))
    np.testing.assert_allclose(derivative, jnp.sum(slopes), atol=1e-7)

  @parameterized.parameters("nearest", "constant")
  def test_identity_affine_translation_has_gradient(self, mode):
    rows, columns = jnp.indices((5, 5), dtype=jnp.float32)
    image = (rows / 16 + columns / 8)[..., None]

    def sample(offset):
      return dm_pix.affine_transform(
          image, jnp.eye(3), offset=offset, mode=mode
      )[2, 2, 0]

    offset = jnp.zeros(3)
    for grad in (jax.grad(sample), jax.jit(jax.grad(sample))):
      np.testing.assert_allclose(grad(offset), [1 / 16, 1 / 8, 0], atol=1e-7)

  def test_clipped_coordinates_and_singleton_axes(self):
    volume = jnp.arange(5, dtype=jnp.float32)[None, :] / 4

    def fn(x):
      return dm_pix.flat_nd_linear_interpolate(volume, x)

    for coordinate, expected in (([-1.0, -1.0], 0.0), ([1.0, 6.0], 1.0)):
      coordinate = jnp.asarray(coordinate)
      np.testing.assert_allclose(fn(coordinate), expected)
      np.testing.assert_array_equal(jax.grad(fn)(coordinate), jnp.zeros(2))

  def test_integer_samples_preserve_value_gradients(self):
    volume = jnp.arange(5, dtype=jnp.float32) / 4

    def fn(values):
      return dm_pix.flat_nd_linear_interpolate(values, jnp.array([2.0]))

    np.testing.assert_array_equal(jax.grad(fn)(volume), [0, 0, 1, 0, 0])


if __name__ == "__main__":
  absltest.main()
