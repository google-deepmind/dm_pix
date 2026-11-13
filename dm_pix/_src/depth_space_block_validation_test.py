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
"""Validate the block size for spatial-to-channel rearrangements."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_pix._src import depth_and_space
import jax
import jax.numpy as jnp
import numpy as np


class BlockSizeValidationTest(parameterized.TestCase):

  @parameterized.product(
      function=(depth_and_space.depth_to_space, depth_and_space.space_to_depth),
      block_size=(0, 1, -1, -2, 1.5, True, np.float64(2.0)),
  )
  def test_invalid_block_sizes_raise_value_error(self, function, block_size):
    image = jnp.arange(64, dtype=jnp.float32).reshape(4, 4, 4)
    with self.assertRaisesRegex(ValueError, "block_size.*integer.*>= 2"):
      function(image, block_size)

  @parameterized.parameters(
      depth_and_space.depth_to_space, depth_and_space.space_to_depth
  )
  def test_invalid_block_size_is_rejected_before_batched_vmap(self, function):
    batch = jnp.ones((2, 4, 4, 4), dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, "block_size"):
      function(batch, 0)

  def test_valid_numpy_integer_block_round_trips(self):
    image = jnp.arange(4 * 4 * 3, dtype=jnp.float32).reshape(4, 4, 3)
    block_size = np.int64(2)
    grouped = depth_and_space.space_to_depth(image, block_size)
    restored = depth_and_space.depth_to_space(grouped, block_size)
    self.assertEqual(grouped.shape, (2, 2, 12))
    np.testing.assert_array_equal(restored, image)

  def test_empty_batch_retains_batch_axis(self):
    batch = jnp.zeros((0, 4, 4, 3), dtype=jnp.float32)
    grouped = depth_and_space.space_to_depth(batch, 2)
    restored = depth_and_space.depth_to_space(grouped, 2)
    self.assertEqual(grouped.shape, (0, 2, 2, 12))
    self.assertEqual(restored.shape, batch.shape)

  def test_batched_roundtrip_is_jittable(self):
    images = jnp.arange(2 * 4 * 4 * 3, dtype=jnp.float32).reshape(2, 4, 4, 3)

    @jax.jit
    def roundtrip(batch):
      packed = depth_and_space.space_to_depth(batch, 2)
      return depth_and_space.depth_to_space(packed, 2)

    result = roundtrip(images)
    self.assertEqual(result.shape, images.shape)
    np.testing.assert_array_equal(result, images)


if __name__ == "__main__":
  absltest.main()
