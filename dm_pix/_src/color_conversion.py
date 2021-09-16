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
"""This module provides functions to convert color spaces.

All functions expect float-encoded images, with values in [0, 1].
"""

from typing import Tuple

import chex
import jax.numpy as jnp


def split_channels(
    image: chex.Array,
    channel_axis: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
  chex.assert_axis_dimension(image, axis=channel_axis, expected=3)
  split_axes = jnp.split(image, 3, axis=channel_axis)
  return tuple(map(lambda x: jnp.squeeze(x, axis=channel_axis), split_axes))


def rgb_to_hsv(
    image_rgb: chex.Array,
    *,
    channel_axis: int = -1,
) -> chex.Array:
  """Converts an image from RGB to HSV.

  Args:
    image_rgb: an RGB image, with float values in range [0, 1]. Behavior outside
      of these bounds is not guaranteed.
    channel_axis: the channel axis. image_rgb should have 3 layers along this
      axis.

  Returns:
    An HSV image, with float values in range [0, 1], stacked along channel_axis.
  """
  red, green, blue = split_channels(image_rgb, channel_axis)
  return jnp.stack(
      rgb_planes_to_hsv_planes(red, green, blue), axis=channel_axis)


def hsv_to_rgb(
    image_hsv: chex.Array,
    *,
    channel_axis: int = -1,
) -> chex.Array:
  """Converts an image from HSV to RGB.

  Args:
    image_hsv: an HSV image, with float values in range [0, 1]. Behavior outside
      of these bounds is not guaranteed.
    channel_axis: the channel axis. image_hsv should have 3 layers along this
      axis.

  Returns:
    An RGB image, with float values in range [0, 1], stacked along channel_axis.
  """
  hue, saturation, value = split_channels(image_hsv, channel_axis)
  return jnp.stack(
      hsv_planes_to_rgb_planes(hue, saturation, value), axis=channel_axis)


def rgb_planes_to_hsv_planes(
    red: chex.Array,
    green: chex.Array,
    blue: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
  """Converts red, green, blue color planes to hue, saturation, value planes.

  All planes should have the same shape, with float values in range [0, 1].
  Behavior outside of these bounds is not guaranteed.

  Reference implementation: http://shortn/_DjPmiAOWSQ

  Args:
    red: the red color plane.
    green: the red color plane.
    blue: the red color plane.

  Returns:
    A tuple of (hue, saturation, value) planes, as float values in range [0, 1].
  """
  value = jnp.maximum(jnp.maximum(red, green), blue)
  minimum = jnp.minimum(jnp.minimum(red, green), blue)
  range_ = value - minimum

  saturation = jnp.where(value > 0, range_ / value, 0.)
  norm = 1. / (6. * range_)

  hue = jnp.where(value == green,
                  norm * (blue - red) + 2. / 6.,
                  norm * (red - green) + 4. / 6.)
  hue = jnp.where(value == red, norm * (green - blue), hue)
  hue = jnp.where(range_ > 0, hue, 0.) + (hue < 0.)

  return hue, saturation, value


def hsv_planes_to_rgb_planes(
    hue: chex.Array,
    saturation: chex.Array,
    value: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
  """Converts hue, saturation, value planes to red, green, blue color planes.

  All planes should have the same shape, with float values in range [0, 1].
  Behavior outside of these bounds is not guaranteed.

  Reference implementation: http://shortn/_NvL2jK8F87

  Args:
    hue: the hue plane (wrapping).
    saturation: the saturation plane.
    value: the value plane.

  Returns:
    A tuple of (red, green, blue) planes, as float values in range [0, 1].
  """
  dh = (hue % 1.0) * 6.  # Wrap when hue >= 360°.
  dr = jnp.clip(jnp.abs(dh - 3.) - 1., 0., 1.)
  dg = jnp.clip(2. - jnp.abs(dh - 2.), 0., 1.)
  db = jnp.clip(2. - jnp.abs(dh - 4.), 0., 1.)
  one_minus_s = 1. - saturation

  red = value * (one_minus_s + saturation * dr)
  green = value * (one_minus_s + saturation * dg)
  blue = value * (one_minus_s + saturation * db)

  return red, green, blue


def rgb_to_hsl(
    image_rgb: chex.Array,
    *,
    channel_axis: int = -1,
) -> chex.Array:
  """Converts an image from RGB to HSL.

  Args:
    image_rgb: an RGB image, with float values in range [0, 1]. Behavior outside
      of these bounds is not guaranteed.
    channel_axis: the channel axis. image_rgb should have 3 layers along this
      axis.

  Returns:
    An HSV image, with float values in range [0, 1], stacked along channel_axis.
  """
  red, green, blue = split_channels(image_rgb, channel_axis)

  c_max = jnp.maximum(red, jnp.maximum(green, blue))
  c_min = jnp.minimum(red, jnp.minimum(green, blue))
  c_sum = c_max + c_min
  c_diff = c_max - c_min

  mask = c_min == c_max

  rc = (c_max - red) / c_diff
  gc = (c_max - green) / c_diff
  bc = (c_max - blue) / c_diff

  eps = jnp.finfo(jnp.float32).eps
  h = jnp.where(
      mask, 0,
      (jnp.where(red == c_max, bc - gc,
                 jnp.where(green == c_max, 2 + rc - bc, 4 + gc - rc)) / 6) % 1)
  s = jnp.where(mask, 0, (c_diff + eps) /
                (2 * eps + jnp.where(c_sum <= 1, c_sum, 2 - c_sum)))
  l = c_sum / 2

  return jnp.stack([h, s, l], axis=-1)


def hsl_to_rgb(
    image_hsl: chex.Array,
    *,
    channel_axis: int = -1,
) -> chex.Array:
  """Converts an image from HSL to RGB.

  Args:
    image_hsl: an HSV image, with float values in range [0, 1]. Behavior outside
      of these bounds is not guaranteed.
    channel_axis: the channel axis. image_hsv should have 3 layers along this
      axis.

  Returns:
    An RGB image, with float values in range [0, 1], stacked along channel_axis.
  """
  h, s, l = split_channels(image_hsl, channel_axis)

  m2 = jnp.where(l <= 0.5, l * (1 + s), l + s - l * s)
  m1 = 2 * l - m2

  def _f(hue):
    hue = hue % 1.0
    return jnp.where(
        hue < 1 / 6, m1 + 6 * (m2 - m1) * hue,
        jnp.where(
            hue < 0.5, m2,
            jnp.where(hue < 2 / 3, m1 + 6 * (m2 - m1) * (2 / 3 - hue), m1)))

  image_rgb = jnp.stack([_f(h + 1 / 3), _f(h), _f(h - 1 / 3)], axis=-1)
  return jnp.where(s[..., jnp.newaxis] == 0, l[..., jnp.newaxis], image_rgb)


def rgb_to_grayscale(
    image: chex.Array,
    *,
    keep_dims: bool = False,
    luma_standard="rec601",
    channel_axis: int = -1,
) -> chex.Array:
  """Converts an image to a grayscale image using the luma value.

  This is equivalent to `tf.image.rgb_to_grayscale` (when keep_channels=False).

  Args:
    image: an RGB image, given as a float tensor in [0, 1].
    keep_dims: if False (default), returns a tensor with a single channel. If
      True, will tile the resulting channel.
    luma_standard: the luma standard to use, either "rec601", "rec709" or
      "bt2001". The default rec601 corresponds to TensorFlow's.
    channel_axis: the index of the channel axis.

  Returns:
    The grayscale image.
  """
  assert luma_standard in ["rec601", "rec709", "bt2001"]
  if luma_standard == "rec601":
    # TensorFlow's default.
    rgb_weights = jnp.array([0.2989, 0.5870, 0.1140], dtype=image.dtype)
  elif luma_standard == "rec709":
    rgb_weights = jnp.array([0.2126, 0.7152, 0.0722], dtype=image.dtype)
  else:
    rgb_weights = jnp.array([0.2627, 0.6780, 0.0593], dtype=image.dtype)
  grayscale = jnp.tensordot(image, rgb_weights, axes=(channel_axis, -1))
  # Add back the channel axis.
  grayscale = jnp.expand_dims(grayscale, axis=channel_axis)
  if keep_dims:
    if channel_axis < 0:
      channel_axis += image.ndim
    reps = [(1 if axis != channel_axis else 3) for axis in range(image.ndim)]
    return jnp.tile(grayscale, reps)  # Tile to 3 along the channel axis.
  else:
    return grayscale
