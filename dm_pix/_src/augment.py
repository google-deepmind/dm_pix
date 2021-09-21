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
"""This module provides image augmentation functions.

All functions expect float-encoded images, with values in [0, 1].
Do not clip their outputs to this range to allow chaining without losing
information. The outside-of-bounds behavior is (as much as possible) similar to
that of TensorFlow.
"""

from typing import Sequence, Tuple

import chex
from dm_pix._src import color_conversion
import jax
import jax.numpy as jnp


def adjust_brightness(image: chex.Array, delta: chex.Numeric) -> chex.Array:
  """Shifts the brightness of an RGB image by a given amount.

  This is equivalent to tf.image.adjust_brightness.

  Args:
    image: an RGB image, given as a float tensor in [0, 1].
    delta: the (additive) amount to shift each channel by.

  Returns:
    The brightness-adjusted image. May be outside of the [0, 1] range.
  """
  return image + jnp.asarray(delta, image.dtype)


def adjust_contrast(
    image: chex.Array,
    factor: chex.Numeric,
    *,
    channel_axis: int = -1,
) -> chex.Array:
  """Adjusts the contrast of an RGB image by a given multiplicative amount.

  This is equivalent to `tf.image.adjust_contrast`.

  Args:
    image: an RGB image, given as a float tensor in [0, 1].
    factor: the (multiplicative) amount to adjust contrast by.
    channel_axis: the index of the channel axis.

  Returns:
    The contrast-adjusted image. May be outside of the [0, 1] range.
  """
  if _channels_last(image, channel_axis):
    spatial_axes = (-3, -2)
  else:
    spatial_axes = (-2, -1)
  mean = jnp.mean(image, axis=spatial_axes, keepdims=True)
  return jnp.asarray(factor, image.dtype) * (image - mean) + mean


def adjust_gamma(
    image: chex.Array,
    gamma: chex.Numeric,
    *,
    gain: chex.Numeric = 1.,
    assume_in_bounds: bool = False,
) -> chex.Array:
  """Adjusts the gamma of an RGB image.

  This is equivalent to `tf.image.adjust_gamma`, i.e. returns
  `gain * image ** gamma`.

  Args:
    image: an RGB image, given as a [0-1] float tensor.
    gamma: the exponent to apply.
    gain: the (multiplicative) gain to apply.
    assume_in_bounds: whether the input image should be assumed to have all
      values within [0, 1]. If False (default), the inputs will be clipped to
      that range avoid NaNs.

  Returns:
    The gamma-adjusted image.
  """
  if not assume_in_bounds:
    image = jnp.clip(image, 0., 1.)  # Clip image for safety.
  return jnp.asarray(gain, image.dtype) * (
      image**jnp.asarray(gamma, image.dtype))


def adjust_hue(
    image: chex.Array,
    delta: chex.Numeric,
    *,
    channel_axis: int = -1,
) -> chex.Array:
  """Adjusts the hue of an RGB image by a given multiplicative amount.

  This is equivalent to `tf.image.adjust_hue` when TF is running on GPU. When
  running on CPU, the results will be different if all RGB values for a pixel
  are outside of the [0, 1] range.

  Args:
    image: an RGB image, given as a [0-1] float tensor.
    delta: the (additive) angle to shift hue by.
    channel_axis: the index of the channel axis.

  Returns:
    The saturation-adjusted image.
  """
  rgb = color_conversion.split_channels(image, channel_axis)
  hue, saturation, value = color_conversion.rgb_planes_to_hsv_planes(*rgb)
  rgb_adjusted = color_conversion.hsv_planes_to_rgb_planes((hue + delta) % 1.0,
                                                           saturation, value)
  return jnp.stack(rgb_adjusted, axis=channel_axis)


def adjust_saturation(
    image: chex.Array,
    factor: chex.Numeric,
    *,
    channel_axis: int = -1,
) -> chex.Array:
  """Adjusts the saturation of an RGB image by a given multiplicative amount.

  This is equivalent to `tf.image.adjust_saturation`.

  Args:
    image: an RGB image, given as a [0-1] float tensor.
    factor: the (multiplicative) amount to adjust saturation by.
    channel_axis: the index of the channel axis.

  Returns:
    The saturation-adjusted image.
  """
  rgb = color_conversion.split_channels(image, channel_axis)
  hue, saturation, value = color_conversion.rgb_planes_to_hsv_planes(*rgb)
  factor = jnp.asarray(factor, image.dtype)
  rgb_adjusted = color_conversion.hsv_planes_to_rgb_planes(
      hue, jnp.clip(saturation * factor, 0., 1.), value)
  return jnp.stack(rgb_adjusted, axis=channel_axis)


def flip_left_right(
    image: chex.Array,
    *,
    channel_axis: int = -1,
) -> chex.Array:
  """Flips an image along the horizontal axis.

  Assumes that the image is either ...HWC or ...CHW and flips the W axis.

  Args:
    image: an RGB image, given as a float tensor in [0, 1].
    channel_axis: the index of the channel axis.

  Returns:
    The flipped image.
  """
  if _channels_last(image, channel_axis):
    flip_axis = -2  # Image is ...HWC
  else:
    flip_axis = -1  # Image is ...CHW
  return jnp.flip(image, axis=flip_axis)


def flip_up_down(
    image: chex.Array,
    *,
    channel_axis: int = -1,
) -> chex.Array:
  """Flips an image along the vertical axis.

  Assumes that the image is either ...HWC or ...CHW, and flips the H axis.

  Args:
    image: an RGB image, given as a float tensor in [0, 1].
    channel_axis: the index of the channel axis.

  Returns:
    The flipped image.
  """
  if _channels_last(image, channel_axis):
    flip_axis = -3  # Image is ...HWC
  else:
    flip_axis = -2  # Image is ...CHW
  return jnp.flip(image, axis=flip_axis)


def gaussian_blur(
    image: chex.Array,
    sigma: float,
    kernel_size: float,
    *,
    padding: str = "SAME",
    channel_axis: int = -1,
) -> chex.Array:
  """Applies gaussian blur (convolution with a Gaussian kernel).

  Args:
    image: the input image, as a [0-1] float tensor. Should have 3 or 4
      dimensions with two spatial dimensions.
    sigma: the standard deviation (in pixels) of the gaussian kernel.
    kernel_size: the size (in pixels) of the square gaussian kernel. Will be
      "rounded" to the next odd integer.
    padding: either "SAME" or "VALID", passed to the underlying convolution.
    channel_axis: the index of the channel axis.

  Returns:
    The blurred image.
  """
  chex.assert_rank(image, {3, 4})
  data_format = "NHWC" if _channels_last(image, channel_axis) else "NCHW"
  dimension_numbers = (data_format, "HWIO", data_format)
  num_channels = image.shape[channel_axis]
  radius = int(kernel_size / 2)
  kernel_size_ = 2 * radius + 1
  x = jnp.arange(-radius, radius + 1).astype(jnp.float32)
  blur_filter = jnp.exp(-x**2 / (2. * sigma**2))
  blur_filter = blur_filter / jnp.sum(blur_filter)
  blur_v = jnp.reshape(blur_filter, [kernel_size_, 1, 1, 1])
  blur_h = jnp.reshape(blur_filter, [1, kernel_size_, 1, 1])
  blur_h = jnp.tile(blur_h, [1, 1, 1, num_channels])
  blur_v = jnp.tile(blur_v, [1, 1, 1, num_channels])

  expand_batch_dim = image.ndim == 3
  if expand_batch_dim:
    image = image[jnp.newaxis, ...]
  blurred = _depthwise_conv2d(
      image,
      kernel=blur_h,
      strides=(1, 1),
      padding=padding,
      channel_axis=channel_axis,
      dimension_numbers=dimension_numbers)
  blurred = _depthwise_conv2d(
      blurred,
      kernel=blur_v,
      strides=(1, 1),
      padding=padding,
      channel_axis=channel_axis,
      dimension_numbers=dimension_numbers)
  if expand_batch_dim:
    blurred = jnp.squeeze(blurred, axis=0)
  return blurred


def rot90(
    image: chex.Array,
    k: int = 1,
    *,
    channel_axis: int = -1,
) -> chex.Array:
  """Rotates an image counter-clockwise by 90 degrees.

  This is equivalent to tf.image.rot90. Assumes that the image is either
  ...HWC or ...CHW.

  Args:
    image: an RGB image, given as a float tensor in [0, 1].
    k: the number of times the rotation is applied.
    channel_axis: the index of the channel axis.

  Returns:
    The rotated image.
  """
  if _channels_last(image, channel_axis):
    spatial_axes = (-3, -2)  # Image is ...HWC
  else:
    spatial_axes = (-2, -1)  # Image is ...CHW
  return jnp.rot90(image, k, spatial_axes)


def solarize(image: chex.Array, threshold: chex.Numeric) -> chex.Array:
  """Applies solarization to an image.

  All values above a given threshold will be inverted.

  Args:
    image: an RGB image, given as a [0-1] float tensor.
    threshold: the threshold for inversion.

  Returns:
    The solarized image.
  """
  return jnp.where(image < threshold, image, 1. - image)


def random_flip_left_right(key: chex.PRNGKey, image: chex.Array) -> chex.Array:
  """50% chance of `flip_left_right(...)` otherwise returns image unchanged."""
  coin_flip = jax.random.bernoulli(key)
  return jax.lax.cond(coin_flip, flip_left_right, lambda x: x, image)


def random_flip_up_down(key: chex.PRNGKey, image: chex.Array) -> chex.Array:
  """50% chance of `flip_up_down(...)` otherwise returns image unchanged."""
  coin_flip = jax.random.bernoulli(key)
  return jax.lax.cond(coin_flip, flip_up_down, lambda x: x, image)


def random_brightness(
    key: chex.PRNGKey,
    image: chex.Array,
    max_delta: chex.Numeric,
) -> chex.Array:
  """`adjust_brightness(...)` with random delta in `[-max_delta, max_delta)`."""
  delta = jax.random.uniform(key, (), minval=-max_delta, maxval=max_delta)
  return adjust_brightness(image, delta)


def random_hue(
    key: chex.PRNGKey,
    image: chex.Array,
    max_delta: chex.Numeric,
    *,
    channel_axis: int = -1,
) -> chex.Array:
  """`adjust_hue(...)` with random delta in `[-max_delta, max_delta)`."""
  delta = jax.random.uniform(key, (), minval=-max_delta, maxval=max_delta)
  return adjust_hue(image, delta, channel_axis=channel_axis)


def random_contrast(
    key: chex.PRNGKey,
    image: chex.Array,
    lower: chex.Numeric,
    upper: chex.Numeric,
    *,
    channel_axis: int = -1,
) -> chex.Array:
  """`adjust_contrast(...)` with random factor in `[lower, upper)`."""
  factor = jax.random.uniform(key, (), minval=lower, maxval=upper)
  return adjust_contrast(image, factor, channel_axis=channel_axis)


def random_saturation(
    key: chex.PRNGKey,
    image: chex.Array,
    lower: chex.Numeric,
    upper: chex.Numeric,
    *,
    channel_axis: int = -1,
) -> chex.Array:
  """`adjust_saturation(...)` with random factor in `[lower, upper)`."""
  factor = jax.random.uniform(key, (), minval=lower, maxval=upper)
  return adjust_saturation(image, factor, channel_axis=channel_axis)


def random_crop(
    key: chex.PRNGKey,
    image: chex.Array,
    crop_sizes: Sequence[int],
) -> chex.Array:
  """Crop images randomly to specified sizes.

  Given an input image, it crops the image to the specified `crop_sizes`. If
  `crop_sizes` are lesser than the image's sizes, the offset for cropping is
  chosen at random. To deterministically crop an image,
  please use `jax.lax.dynamic_slice` and specify offsets and crop sizes.

  Args:
    key: key for pseudo-random number generator.
    image: a JAX array which represents an image.
    crop_sizes: a sequence of integers, each of which sequentially specifies the
      crop size along the corresponding dimension of the image. Sequence length
      must be identical to the rank of the image and the crop size should not be
      greater than the corresponding image dimension.

  Returns:
    A cropped image, a JAX array whose shape is same as `crop_sizes`.
  """

  image_shape = image.shape
  assert len(image_shape) == len(crop_sizes), (
      f"Number of image dims {len(image_shape)} and number of crop_sizes "
      f"{len(crop_sizes)} do not match.")
  assert image_shape >= crop_sizes, (
      f"Crop sizes {crop_sizes} should be a subset of image size {image_shape} "
      "in each dimension .")
  random_keys = jax.random.split(key, len(crop_sizes))

  slice_starts = [
      jax.random.randint(k, (), 0, img_size - crop_size + 1)
      for k, img_size, crop_size in zip(random_keys, image_shape, crop_sizes)
  ]
  out = jax.lax.dynamic_slice(image, slice_starts, crop_sizes)

  return out


def _channels_last(image: chex.Array, channel_axis: int):
  last = channel_axis == -1 or channel_axis == (image.ndim - 1)
  if not last:
    assert channel_axis == -3 or channel_axis == range(image.ndim)[-3]
  return last


def _depthwise_conv2d(
    inputs: chex.Array,
    kernel: chex.Array,
    *,
    strides: Tuple[int, int],
    padding: str,
    channel_axis: int,
    dimension_numbers: Tuple[str, str, str],
) -> chex.Array:
  """Computes a depthwise conv2d in Jax.

  Reference implementation: http://shortn/_oEpb0c2V3l

  Args:
    inputs: an NHWC or NCHW tensor (depending on dimension_numbers), with N=1.
    kernel: a [H', W', 1, C] tensor.
    strides: optional stride for the kernel.
    padding: "SAME" or "VALID".
    channel_axis: the index of the channel axis.
    dimension_numbers: see jax.lax.conv_general_dilated.

  Returns:
    The depthwise convolution of inputs with kernel, with the same
    dimension_numbers as the input.
  """
  return jax.lax.conv_general_dilated(
      inputs,
      kernel,
      strides,
      padding,
      feature_group_count=inputs.shape[channel_axis],
      dimension_numbers=dimension_numbers)
