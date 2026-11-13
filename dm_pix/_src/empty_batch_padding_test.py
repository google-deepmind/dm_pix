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
"""Padding and resizing images with an empty batch dimension."""

from absl.testing import absltest
from absl.testing import parameterized
import dm_pix
import jax
import jax.numpy as jnp


class EmptyBatchPaddingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("nhwc", (0, 3, 5, 2), -1, (0, 7, 9, 2)),
      ("nchw", (0, 2, 3, 5), 1, (0, 2, 7, 9)),
      ("nchw_negative_axis", (0, 2, 3, 5), -3, (0, 2, 7, 9)),
  )
  def test_pad_to_size_preserves_empty_batch(self, shape, channel_axis, expected):
    image = jnp.zeros(shape, dtype=jnp.float32)
    result = dm_pix.pad_to_size(image, 7, 9, channel_axis=channel_axis)
    self.assertEqual(result.shape, expected)

  @parameterized.named_parameters(
      ("nhwc", (0, 3, 5, 2), -1, (0, 7, 9, 2)),
      ("nchw", (0, 2, 3, 5), -3, (0, 2, 7, 9)),
  )
  def test_resize_with_crop_or_pad_preserves_empty_batch(
      self, shape, channel_axis, expected
  ):
    image = jnp.zeros(shape, dtype=jnp.float32)
    result = dm_pix.resize_with_crop_or_pad(
        image, 7, 9, channel_axis=channel_axis
    )
    self.assertEqual(result.shape, expected)

  def test_nonempty_batch_padding_is_unchanged(self):
    image = jnp.ones((2, 3, 5, 1), dtype=jnp.float32)
    result = dm_pix.pad_to_size(image, 7, 9)
    self.assertEqual(result.shape, (2, 7, 9, 1))
    self.assertEqual(float(jnp.sum(result)), 30.0)

  def test_padding_empty_batch_is_jittable(self):
    image = jnp.zeros((0, 3, 5, 2), dtype=jnp.float32)
    result = jax.jit(lambda x: dm_pix.pad_to_size(x, 7, 9))(image)
    self.assertEqual(result.shape, (0, 7, 9, 2))


if __name__ == "__main__":
  absltest.main()
