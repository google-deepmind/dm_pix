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
"""Tests for dm_pix._src.color_conversion."""

import colorsys
import enum
import functools
from typing import Sequence

from absl.testing import parameterized
import chex
from dm_pix._src import color_conversion
import jax
import jax.numpy as jnp
import jax.test_util as jtu
import numpy as np
import tensorflow as tf

_NUM_IMAGES = 100
_IMG_SHAPE = (16, 16, 3)
_FLAT_IMG_SHAPE = (_IMG_SHAPE[0] * _IMG_SHAPE[1], _IMG_SHAPE[2])
_QUANTISATIONS = (None, 16, 2)


@enum.unique
class TestImages(enum.Enum):
  """Enum classes representing random images with (low, high, num_images)."""
  RAND_FLOATS_IN_RANGE = (0., 1., _NUM_IMAGES)
  RAND_FLOATS_OUT_OF_RANGE = (-0.5, 1.5, _NUM_IMAGES)
  ALL_ONES = (1., 1., 1)
  ALL_ZEROS = (0., 0., 1)


def generate_test_images(
    low: float,
    high: float,
    num_images: int,
) -> Sequence[chex.Array]:
  images = np.random.uniform(
      low=low,
      high=high,
      size=(num_images,) + _IMG_SHAPE,
  )
  return list(images.astype(np.float32))


class ColorConversionTest(
    chex.TestCase,
    jtu.JaxTestCase,
    parameterized.TestCase,
):

  @chex.all_variants
  @parameterized.product(
      test_images=[
          TestImages.RAND_FLOATS_IN_RANGE,
          TestImages.RAND_FLOATS_OUT_OF_RANGE,
          TestImages.ALL_ONES,
          TestImages.ALL_ZEROS,
      ],
      channel_last=[True, False],
  )
  def test_hsv_to_rgb(self, test_images, channel_last):
    channel_axis = -1 if channel_last else -3
    hsv_to_rgb = self.variant(
        functools.partial(
            color_conversion.hsv_to_rgb, channel_axis=channel_axis))
    for hsv in generate_test_images(*test_images.value):
      hsv = np.clip(hsv, 0., 1.)
      rgb_tf = tf.image.hsv_to_rgb(hsv).numpy()
      if not channel_last:
        hsv = hsv.swapaxes(-1, -3)
      rgb_jax = hsv_to_rgb(hsv)
      if not channel_last:
        rgb_jax = rgb_jax.swapaxes(-1, -3)
      self.assertAllClose(rgb_jax, rgb_tf, rtol=1e-3, atol=1e-3)

  @chex.all_variants
  @parameterized.product(
      test_images=[
          TestImages.RAND_FLOATS_IN_RANGE,
          TestImages.RAND_FLOATS_OUT_OF_RANGE,
          TestImages.ALL_ONES,
          TestImages.ALL_ZEROS,
      ],
      channel_last=[True, False],
  )
  def test_rgb_to_hsv(self, test_images, channel_last):
    channel_axis = -1 if channel_last else -3
    rgb_to_hsv = self.variant(
        functools.partial(
            color_conversion.rgb_to_hsv, channel_axis=channel_axis))
    for rgb in generate_test_images(*test_images.value):
      hsv_tf = tf.image.rgb_to_hsv(rgb).numpy()
      if not channel_last:
        rgb = rgb.swapaxes(-1, -3)
      hsv_jax = rgb_to_hsv(rgb)
      if not channel_last:
        hsv_jax = hsv_jax.swapaxes(-1, -3)
      self.assertAllClose(hsv_jax, hsv_tf, rtol=1e-3, atol=1e-3)

  @chex.all_variants
  def test_vmap_roundtrip(self):
    images = generate_test_images(*TestImages.RAND_FLOATS_IN_RANGE.value)
    rgb_init = np.stack(images, axis=0)
    rgb_to_hsv = self.variant(jax.vmap(color_conversion.rgb_to_hsv))
    hsv_to_rgb = self.variant(jax.vmap(color_conversion.hsv_to_rgb))
    hsv = rgb_to_hsv(rgb_init)
    rgb_final = hsv_to_rgb(hsv)
    self.assertAllClose(rgb_init, rgb_final, rtol=1e-3, atol=1e-3)

  def test_jit_roundtrip(self):
    images = generate_test_images(*TestImages.RAND_FLOATS_IN_RANGE.value)
    rgb_init = np.stack(images, axis=0)
    hsv = jax.jit(color_conversion.rgb_to_hsv)(rgb_init)
    rgb_final = jax.jit(color_conversion.hsv_to_rgb)(hsv)
    self.assertAllClose(rgb_init, rgb_final, rtol=1e-3, atol=1e-3)

  @chex.all_variants
  @parameterized.named_parameters(
      ("black", 0, 0),
      ("gray", 0.000001, 0.999999),
      ("white", 1, 1),
  )
  def test_rgb_to_hsl_golden(self, minval, maxval):
    """Compare against colorsys.rgb_to_hls as a golden implementation."""
    key = jax.random.PRNGKey(0)
    for quantization in (None, 16, 2):
      key_rand_uni, key = jax.random.split(key)
      image_rgb = jax.random.uniform(
          key=key_rand_uni,
          shape=_FLAT_IMG_SHAPE,
          dtype=np.float32,
          minval=minval,
          maxval=maxval,
      )

      # Use quantization to probe the corners of the color cube.
      if quantization is not None:
        image_rgb = jnp.round(image_rgb * quantization) / quantization

      hsl_true = np.zeros_like(image_rgb)
      for i in range(image_rgb.shape[0]):
        h, l, s = colorsys.rgb_to_hls(*image_rgb[i, :])
        hsl_true[i, :] = [h, s, l]

      image_rgb = np.reshape(image_rgb, _IMG_SHAPE)
      hsl_true = np.reshape(hsl_true, _IMG_SHAPE)
      rgb_to_hsl = self.variant(color_conversion.rgb_to_hsl)
      self.assertAllClose(rgb_to_hsl(image_rgb), hsl_true)

  @chex.all_variants
  @parameterized.named_parameters(
      ("black", 0, 0.000001),
      ("white", 0.999999, 1),
  )
  def test_rgb_to_hsl_stable(self, minval, maxval):
    """rgb_to_hsl's output near the black+white corners should be in [0, 1]."""
    key_rand_uni = jax.random.PRNGKey(0)
    image_rgb = jax.random.uniform(
        key=key_rand_uni,
        shape=_FLAT_IMG_SHAPE,
        dtype=np.float32,
        minval=minval,
        maxval=maxval,
    )
    rgb_to_hsl = self.variant(color_conversion.rgb_to_hsl)
    hsl = rgb_to_hsl(image_rgb)
    self.assertTrue(jnp.all(jnp.isfinite(hsl)))
    self.assertLessEqual(jnp.max(hsl), 1.)
    self.assertGreaterEqual(jnp.min(hsl), 0.)

  @chex.all_variants
  def test_hsl_to_rgb_golden(self):
    """Compare against colorsys.rgb_to_hls as a golden implementation."""
    key = jax.random.PRNGKey(0)
    for quantization in _QUANTISATIONS:
      key_rand_uni, key = jax.random.split(key)
      image_hsl = (
          jax.random.uniform(key_rand_uni, _FLAT_IMG_SHAPE).astype(np.float32))

      # Use quantization to probe the corners of the color cube.
      if quantization is not None:
        image_hsl = jnp.round(image_hsl * quantization) / quantization

      rgb_true = np.zeros_like(image_hsl)
      for i in range(image_hsl.shape[0]):
        h, s, l = image_hsl[i, :]
        rgb_true[i, :] = colorsys.hls_to_rgb(h, l, s)

      rgb_true = np.reshape(rgb_true, _IMG_SHAPE)
      image_hsl = np.reshape(image_hsl, _IMG_SHAPE)
      hsl_to_rgb = self.variant(color_conversion.hsl_to_rgb)
      self.assertAllClose(hsl_to_rgb(image_hsl), rgb_true)

  @chex.all_variants
  def test_hsl_rgb_roundtrip(self):
    key = jax.random.PRNGKey(0)
    for quantization in _QUANTISATIONS:
      key_rand_uni, key = jax.random.split(key)
      image_rgb = jax.random.uniform(key_rand_uni, _IMG_SHAPE)

      # Use quantization to probe the corners of the color cube.
      if quantization is not None:
        image_rgb = jnp.round(image_rgb * quantization) / quantization

      rgb_to_hsl = self.variant(color_conversion.rgb_to_hsl)
      hsl_to_rgb = self.variant(color_conversion.hsl_to_rgb)
      self.assertAllClose(image_rgb, hsl_to_rgb(rgb_to_hsl(image_rgb)))

  @chex.all_variants
  @parameterized.product(
      test_images=[
          TestImages.RAND_FLOATS_IN_RANGE,
          TestImages.RAND_FLOATS_OUT_OF_RANGE,
          TestImages.ALL_ONES,
          TestImages.ALL_ZEROS,
      ],
      keep_dims=[True, False],
      channel_last=[True, False],
  )
  def test_grayscale(self, test_images, keep_dims, channel_last):
    channel_axis = -1 if channel_last else -3
    rgb_to_grayscale = self.variant(
        functools.partial(
            color_conversion.rgb_to_grayscale,
            keep_dims=keep_dims,
            channel_axis=channel_axis))
    for rgb in generate_test_images(*test_images.value):
      grayscale_tf = tf.image.rgb_to_grayscale(rgb).numpy()
      if not channel_last:
        rgb = rgb.swapaxes(-1, -3)
      grayscale_jax = rgb_to_grayscale(rgb)
      if not channel_last:
        grayscale_jax = grayscale_jax.swapaxes(-1, -3)
      if keep_dims:
        for i in range(_IMG_SHAPE[-1]):
          self.assertAllClose(grayscale_jax[..., [i]], grayscale_tf)
      else:
        self.assertAllClose(grayscale_jax, grayscale_tf)


if __name__ == "__main__":
  tf.test.main()
