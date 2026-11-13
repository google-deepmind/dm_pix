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
"""Tests for random cropping validation."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_pix._src import augment
import jax
import jax.numpy as jnp
import numpy as np


class RandomCropValidationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("oversized_width", (5, 8, 1)),
      ("oversized_height", (11, 2, 1)),
      ("oversized_channels", (5, 2, 2)),
      ("negative_height", (-1, 2, 1)),
  )
  def test_rejects_invalid_size_in_any_dimension(self, crop_sizes):
    image = jnp.zeros((10, 3, 1), dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, "Crop sizes"):
      augment.random_crop(jax.random.PRNGKey(0), image, crop_sizes)

  def test_rejects_wrong_number_of_dimensions(self):
    image = jnp.zeros((10, 3, 1), dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, "number of crop_sizes"):
      augment.random_crop(jax.random.PRNGKey(0), image, (2, 2))

  def test_valid_crop_matches_jitted_output(self):
    image = jnp.arange(30, dtype=jnp.float32).reshape((10, 3, 1))
    key = jax.random.PRNGKey(12)
    crop_sizes = (5, 2, 1)
    expected = augment.random_crop(key, image, crop_sizes)
    actual = jax.jit(augment.random_crop, static_argnums=2)(
        key, image, crop_sizes
    )
    self.assertEqual(actual.shape, crop_sizes)
    np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
  absltest.main()
