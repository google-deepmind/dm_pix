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
"""Center cropping should use the same pixel alignment as centered padding."""

from absl.testing import absltest
from absl.testing import parameterized
import dm_pix
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


class CropAlignmentTest(parameterized.TestCase):

  @parameterized.product(
      sizes=[
          ((4, 6), (3, 5)),
          ((5, 7), (2, 4)),
          ((4, 4), (3, 3)),
          ((3, 3), (2, 2)),
          ((4, 5), (6, 3)),
          ((1, 4), (1, 1)),
      ],
      batched=[False, True],
      channels_first=[False, True],
  )
  def test_center_crop_matches_integer_margin_reference(
      self, sizes, batched, channels_first
  ):
    (height, width), (out_h, out_w) = sizes
    image = np.arange(height * width * 2, dtype=np.float32).reshape(
        height, width, 2
    )
    top, left = max((height - out_h) // 2, 0), max((width - out_w) // 2, 0)
    expected = image[
        top : top + min(height, out_h), left : left + min(width, out_w)
    ]
    if batched:
      image = np.stack([image, image + 100])
      expected = np.stack([expected, expected + 100])
    if channels_first:
      image = np.moveaxis(image, -1, -3)
      expected = np.moveaxis(expected, -1, -3)
    axis = -3 if channels_first else -1
    fn = lambda x: dm_pix.center_crop(x, out_h, out_w, channel_axis=axis)
    for evaluate in (fn, jax.jit(fn)):
      np.testing.assert_array_equal(evaluate(jnp.asarray(image)), expected)

  @parameterized.parameters(
      (3, 5, 4, 6), (3, 4, 6, 7), (4, 3, 5, 6), (1, 1, 2, 2)
  )
  def test_padding_then_cropping_returns_every_original_pixel(
      self, h, w, ph, pw
  ):
    image = jnp.arange(h * w * 3, dtype=jnp.float32).reshape(h, w, 3)
    padded = dm_pix.pad_to_size(image, ph, pw)
    recovered = dm_pix.center_crop(padded, h, w)
    np.testing.assert_array_equal(recovered, image)
    gradient = jax.grad(
        lambda x: dm_pix.center_crop(dm_pix.pad_to_size(x, ph, pw), h, w).sum()
    )(image)
    np.testing.assert_array_equal(gradient, jnp.ones_like(image))

  @parameterized.parameters((4, 6, 3, 5), (4, 5, 7, 2), (7, 4, 2, 7))
  def test_resize_with_crop_or_pad_matches_tensorflow(self, h, w, th, tw):
    image = np.arange(2 * h * w * 3, dtype=np.float32).reshape(2, h, w, 3)
    expected = tf.image.resize_with_crop_or_pad(image, th, tw).numpy()
    actual = dm_pix.resize_with_crop_or_pad(jnp.asarray(image), th, tw)
    np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
  absltest.main()
