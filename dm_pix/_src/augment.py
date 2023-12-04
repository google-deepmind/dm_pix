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

import functools
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import chex
from dm_pix._src import color_conversion
from dm_pix._src import interpolation
import jax
import jax.numpy as jnp

# DO NOT REMOVE - Logging lib.


def adjust_brightness(image: chex.Array, delta: chex.Numeric) -> chex.Array:
  """Shifts the brightness of an RGB image by a given amount.

  This is equivalent to tf.image.adjust_brightness.

  Args:
    image: an RGB image, given as a float tensor in [0, 1].
    delta: the (additive) amount to shift each channel by.

  Returns:
    The brightness-adjusted image. May be outside of the [0, 1] range.
  """
  # DO NOT REMOVE - Logging usage.

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
  # DO NOT REMOVE - Logging usage.

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
  # DO NOT REMOVE - Logging usage.

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
  # DO NOT REMOVE - Logging usage.

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
  # DO NOT REMOVE - Logging usage.

  rgb = color_conversion.split_channels(image, channel_axis)
  hue, saturation, value = color_conversion.rgb_planes_to_hsv_planes(*rgb)
  factor = jnp.asarray(factor, image.dtype)
  rgb_adjusted = color_conversion.hsv_planes_to_rgb_planes(
      hue, jnp.clip(saturation * factor, 0., 1.), value)
  return jnp.stack(rgb_adjusted, axis=channel_axis)


def elastic_deformation(
    key: chex.PRNGKey,
    image: chex.Array,
    alpha: chex.Numeric,
    sigma: chex.Numeric,
    *,
    order: int = 1,
    mode: str = "nearest",
    cval: float = 0.,
    channel_axis: int = -1,
) -> chex.Array:
  """Applies an elastic deformation to the given image.

  Introduced by [Simard, 2003] and popularized by [Ronneberger, 2015]. Deforms
  images by moving pixels locally around using displacement fields.

  Small sigma values (< 1.) give pixelated images while higher values result
  in water like results. Alpha should be in the between x5 and x10 the value
  given for sigma for sensible results.

  Args:
    key: key: a JAX RNG key.
    image: a JAX array representing an image. Assumes that the image is
      either HWC or CHW.
    alpha: strength of the distortion field. Higher values mean that pixels are
      moved further with respect to the distortion field's direction.
    sigma: standard deviation of the gaussian kernel used to smooth the
      distortion fields.
    order: the order of the spline interpolation, default is 1. The order has
      to be in the range [0, 1]. Note that PIX interpolation will only be used
      for order=1, for other values we use `jax.scipy.ndimage.map_coordinates`.
    mode: the mode parameter determines how the input array is extended beyond
      its boundaries. Default is 'nearest'. Modes 'nearest and 'constant' use
      PIX interpolation, which is very fast on accelerators (especially on
      TPUs). For all other modes, 'wrap', 'mirror' and 'reflect', we rely
      on `jax.scipy.ndimage.map_coordinates`, which however is slow on
      accelerators, so use it with care.
    cval: value to fill past edges of input if mode is 'constant'. Default is
      0.0.
    channel_axis: the index of the channel axis.

  Returns:
    The transformed image.
  """
  # DO NOT REMOVE - Logging usage.

  chex.assert_rank(image, 3)
  if channel_axis != -1:
    image = jnp.moveaxis(image, source=channel_axis, destination=-1)
  single_channel_shape = (*image.shape[:-1], 1)
  key_i, key_j = jax.random.split(key)
  noise_i = jax.random.uniform(key_i, shape=single_channel_shape) * 2 - 1
  noise_j = jax.random.uniform(key_j, shape=single_channel_shape) * 2 - 1

  # ~3 sigma on each side of the kernel's center covers ~99.7% of the
  # probability mass. There is some fiddling for smaller values. Source:
  # https://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
  kernel_size = ((sigma - 0.8) / 0.3 + 1) / 0.5 + 1
  shift_map_i = gaussian_blur(
      image=noise_i,
      sigma=sigma,
      kernel_size=kernel_size) * alpha
  shift_map_j = gaussian_blur(
      image=noise_j,
      sigma=sigma,
      kernel_size=kernel_size) * alpha

  meshgrid = jnp.meshgrid(*[jnp.arange(size) for size in single_channel_shape],
                          indexing="ij")
  meshgrid[0] += shift_map_i
  meshgrid[1] += shift_map_j

  interpolate_function = _get_interpolate_function(
      mode=mode,
      order=order,
      cval=cval,
  )
  transformed_image = jnp.concatenate([
      interpolate_function(
          image[..., channel, jnp.newaxis], jnp.asarray(meshgrid))
      for channel in range(image.shape[-1])
  ], axis=-1)

  if channel_axis != -1:  # Set channel axis back to original index.
    transformed_image = jnp.moveaxis(
        transformed_image, source=-1, destination=channel_axis)
  return transformed_image


def center_crop(
    image: chex.Array,
    height: chex.Numeric,
    width: chex.Numeric,
    *,
    channel_axis: int = -1,
) -> chex.Array:
  """Crops an image to the given size keeping the same center of the original.

  Target height/width given can be greater than the current size of the image
  which results in being a no-op for that dimension.

  In case of odd size along any dimension the bottom/right side gets the extra
  pixel.

  Args:
    image: a JAX array representing an image. Assumes that the image is either
      ...HWC or ...CHW.
    height: target height to crop the image to.
    width: target width to crop the image to.
    channel_axis: the index of the channel axis.

  Returns:
    The cropped image(s).
  """
  # DO NOT REMOVE - Logging usage.

  chex.assert_rank(image, {3, 4})
  batch, current_height, current_width, channel = _get_dimension_values(
      image=image, channel_axis=channel_axis
  )
  center_h, center_w = current_height // 2, current_width // 2

  left = max(center_w - (width // 2), 0)
  right = min(left + width, current_width)
  top = max(center_h - (height // 2), 0)
  bottom = min(top + height, current_height)

  if _channels_last(image, channel_axis):
    start_indices = (top, left, 0)
    limit_indices = (bottom, right, channel)
  else:
    start_indices = (0, top, left)
    limit_indices = (channel, bottom, right)

  if batch is not None:  # In case batch of images is given.
    start_indices = (0, *start_indices)
    limit_indices = (batch, *limit_indices)

  return jax.lax.slice(
      image, start_indices=start_indices, limit_indices=limit_indices
  )


def pad_to_size(
    image: chex.Array,
    target_height: int,
    target_width: int,
    *,
    mode: str = "constant",
    pad_kwargs: Optional[Any] = None,
    channel_axis: int = -1,
) -> chex.Array:
  """Pads an image to the given size keeping the original image centered.

  For different padding methods and kwargs please see:
  https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.pad.html

  In case of odd size difference along any dimension the bottom/right side gets
  the extra padding pixel.

  Target size can be smaller than original size which results in a no-op for
  such dimension.

  Args:
    image: a JAX array representing an image. Assumes that the image is either
      ...HWC or ...CHW.
    target_height: target height to pad the image to.
    target_width: target width to pad the image to.
    mode: Mode for padding the images, see jax.numpy.pad for details. Default is
      `constant`.
    pad_kwargs: Keyword arguments to pass jax.numpy.pad, see documentation for
      options.
    channel_axis: the index of the channel axis.

  Returns:
    The padded image(s).
  """
  # DO NOT REMOVE - Logging usage.

  chex.assert_rank(image, {3, 4})
  batch, height, width, _ = _get_dimension_values(
      image=image, channel_axis=channel_axis
  )
  delta_width = max(target_width - width, 0)
  delta_height = max(target_height - height, 0)
  if delta_width == 0 and delta_height == 0:
    return image

  left = delta_width // 2
  right = max(target_width - (left + width), 0)
  top = delta_height // 2
  bottom = max(target_height - (top + height), 0)

  pad_width = ((top, bottom), (left, right), (0, 0))
  if batch:
    pad_width = ((0, 0), *pad_width)

  return jnp.pad(image, pad_width=pad_width, mode=mode, **pad_kwargs or {})


def resize_with_crop_or_pad(
    image: chex.Array,
    target_height: chex.Numeric,
    target_width: chex.Numeric,
    *,
    pad_mode: str = "constant",
    pad_kwargs: Optional[Any] = None,
    channel_axis: int = -1,
) -> chex.Array:
  """Crops and/or pads an image to a target width and height.

  Equivalent in functionality to tf.image.resize_with_crop_or_pad but allows for
  different padding methods as well beyond zero padding.

  Args:
    image: a JAX array representing an image. Assumes that the image is either
      ...HWC or ...CHW.
    target_height: target height to crop or pad the image to.
    target_width: target width to crop or pad the image to.
    pad_mode: mode for padding the images, see jax.numpy.pad for details.
      Default is `constant`.
    pad_kwargs: keyword arguments to pass jax.numpy.pad, see documentation for
      options.
    channel_axis: the index of the channel axis.

  Returns:
    The image(s) resized by crop or pad to the desired target size.
  """
  # DO NOT REMOVE - Logging usage.

  chex.assert_rank(image, {3, 4})
  image = center_crop(
      image,
      height=target_height,
      width=target_width,
      channel_axis=channel_axis,
  )
  return pad_to_size(
      image,
      target_height=target_height,
      target_width=target_width,
      channel_axis=channel_axis,
      mode=pad_mode,
      pad_kwargs=pad_kwargs,
  )


def flip_left_right(
    image: chex.Array,
    *,
    channel_axis: int = -1,
) -> chex.Array:
  """Flips an image along the horizontal axis.

  Assumes that the image is either ...HWC or ...CHW and flips the W axis.

  Args:
    image: a JAX array representing an image. Assumes that the image is either
      ...HWC or ...CHW.
    channel_axis: the index of the channel axis.

  Returns:
    The flipped image.
  """
  # DO NOT REMOVE - Logging usage.

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
    image: a JAX array representing an image. Assumes that the image is either
      ...HWC or ...CHW.
    channel_axis: the index of the channel axis.

  Returns:
    The flipped image.
  """
  # DO NOT REMOVE - Logging usage.

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
  # DO NOT REMOVE - Logging usage.

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
  # DO NOT REMOVE - Logging usage.

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
  # DO NOT REMOVE - Logging usage.

  return jnp.where(image < threshold, image, 1. - image)


def affine_transform(
    image: chex.Array,
    matrix: chex.Array,
    *,
    offset: Union[chex.Array, chex.Numeric] = 0.,
    order: int = 1,
    mode: str = "nearest",
    cval: float = 0.0,
) -> chex.Array:
  """Applies an affine transformation given by matrix.

  Given an output image pixel index vector o, the pixel value is determined from
  the input image at position jnp.dot(matrix, o) + offset.

  This does 'pull' (or 'backward') resampling, transforming the output space to
  the input to locate data. Affine transformations are often described in the
  'push' (or 'forward') direction, transforming input to output. If you have a
  matrix for the 'push' transformation, use its inverse (jax.numpy.linalg.inv)
  in this function.

  Args:
    image: a JAX array representing an image. Assumes that the image is
      either HWC or CHW.
    matrix: the inverse coordinate transformation matrix, mapping output
      coordinates to input coordinates. If ndim is the number of dimensions of
      input, the given matrix must have one of the following shapes:

      - (ndim, ndim): the linear transformation matrix for each output
        coordinate.
      - (ndim,): assume that the 2-D transformation matrix is diagonal, with the
        diagonal specified by the given value.
      - (ndim + 1, ndim + 1): assume that the transformation is specified using
        homogeneous coordinates [1]. In this case, any value passed to offset is
        ignored.
      - (ndim, ndim + 1): as above, but the bottom row of a homogeneous
        transformation matrix is always [0, 0, 0, 1], and may be omitted.

    offset: the offset into the array where the transform is applied. If a
      float, offset is the same for each axis. If an array, offset should
      contain one value for each axis.
    order: the order of the spline interpolation, default is 1. The order has
      to be in the range [0-1]. Note that PIX interpolation will only be used
      for order=1, for other values we use `jax.scipy.ndimage.map_coordinates`.
    mode: the mode parameter determines how the input array is extended beyond
      its boundaries. Default is 'nearest'. Modes 'nearest and 'constant' use
      PIX interpolation, which is very fast on accelerators (especially on
      TPUs). For all other modes, 'wrap', 'mirror' and 'reflect', we rely
      on `jax.scipy.ndimage.map_coordinates`, which however is slow on
      accelerators, so use it with care.
    cval: value to fill past edges of input if mode is 'constant'. Default is
      0.0.

  Returns:
    The input image transformed by the given matrix.

  Example transformations:
    Rotation:

    >>> angle = jnp.pi / 4
    >>> matrix = jnp.array([
    ...    [jnp.cos(rotation), -jnp.sin(rotation), 0],
    ...    [jnp.sin(rotation), jnp.cos(rotation), 0],
    ...    [0, 0, 1],
    ... ])
    >>> result = dm_pix.affine_transform(image=image, matrix=matrix)

    Translation can be expressed through either the matrix itself
    or the offset parameter:

    >>> matrix = jnp.array([
    ...   [1, 0, 0, 25],
    ...   [0, 1, 0, 25],
    ...   [0, 0, 1, 0],
    ... ])
    >>> result = dm_pix.affine_transform(image=image, matrix=matrix)
    >>> # Or with offset:
    >>> matrix = jnp.array([
    ...   [1, 0, 0],
    ...   [0, 1, 0],
    ...   [0, 0, 1],
    ... ])
    >>> offset = jnp.array([25, 25, 0])
    >>> result = dm_pix.affine_transform(
            image=image, matrix=matrix, offset=offset)

    Reflection:

    >>> matrix = jnp.array([
    ...   [-1, 0, 0],
    ...   [0, 1, 0],
    ...   [0, 0, 1],
    ... ])
    >>> result = dm_pix.affine_transform(image=image, matrix=matrix)

    Scale:

    >>> matrix = jnp.array([
    ...   [2, 0, 0],
    ...   [0, 1, 0],
    ...   [0, 0, 1],
    ... ])
    >>> result = dm_pix.affine_transform(image=image, matrix=matrix)

    Shear:

    >>> matrix = jnp.array([
    ...   [1, 0.5, 0],
    ...   [0.5, 1, 0],
    ...   [0, 0, 1],
    ... ])
    >>> result = dm_pix.affine_transform(image=image, matrix=matrix)

    One can also combine different transformations matrices:

    >>> matrix = rotation_matrix.dot(translation_matrix)
  """
  # DO NOT REMOVE - Logging usage.

  chex.assert_rank(image, 3)
  chex.assert_rank(matrix, {1, 2})
  chex.assert_rank(offset, {0, 1})

  if matrix.ndim == 1:
    matrix = jnp.diag(matrix)

  if matrix.shape not in [(3, 3), (4, 4), (3, 4)]:
    error_msg = (
        "Expected matrix shape must be one of (ndim, ndim), (ndim,)"
        "(ndim + 1, ndim + 1) or (ndim, ndim + 1) being ndim the image.ndim. "
        f"The affine matrix provided has shape {matrix.shape}.")
    raise ValueError(error_msg)

  meshgrid = jnp.meshgrid(*[jnp.arange(size) for size in image.shape],
                          indexing="ij")
  indices = jnp.concatenate(
      [jnp.expand_dims(x, axis=-1) for x in meshgrid], axis=-1)

  if matrix.shape == (4, 4) or matrix.shape == (3, 4):
    offset = matrix[:image.ndim, image.ndim]
    matrix = matrix[:image.ndim, :image.ndim]

  coordinates = indices @ matrix.T
  coordinates = jnp.moveaxis(coordinates, source=-1, destination=0)

  # Alter coordinates to account for offset.
  offset = jnp.full((3,), fill_value=offset)
  coordinates += jnp.reshape(a=offset, newshape=(*offset.shape, 1, 1, 1))

  interpolate_function = _get_interpolate_function(
      mode=mode,
      order=order,
      cval=cval,
  )
  return interpolate_function(image, coordinates)


def rotate(
    image: chex.Array,
    angle: float,
    *,
    order: int = 1,
    mode: str = "nearest",
    cval: float = 0.0,
) -> chex.Array:
  """Rotates an image around its center using interpolation.

  Args:
    image: a JAX array representing an image. Assumes that the image is
      either HWC or CHW.
    angle: the counter-clockwise rotation angle in units of radians.
    order: the order of the spline interpolation, default is 1. The order has
      to be in the range [0,1]. See `affine_transform` for details.
    mode: the mode parameter determines how the input array is extended beyond
      its boundaries. Default is 'nearest'. See `affine_transform` for details.
    cval: value to fill past edges of input if mode is 'constant'. Default is
      0.0.

  Returns:
    The rotated image.
  """
  # DO NOT REMOVE - Logging usage.

  # Calculate inverse transform matrix assuming clockwise rotation.
  c = jnp.cos(angle)
  s = jnp.sin(angle)
  matrix = jnp.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

  # Use the offset to place the rotation at the image center.
  image_center = (jnp.asarray(image.shape) - 1.) / 2.
  offset = image_center - matrix @ image_center

  return affine_transform(image, matrix, offset=offset, order=order, mode=mode,
                          cval=cval)


def random_flip_left_right(
    key: chex.PRNGKey,
    image: chex.Array,
    *,
    probability: chex.Numeric = 0.5,
) -> chex.Array:
  """Applies `flip_left_right` with a given probability.

  Args:
    key: a JAX RNG key.
    image: a JAX array representing an image. Assumes that the image is either
      ...HWC or ...CHW.
    probability: the probability of applying flip_left_right transform. Must be
      a value in [0, 1].

  Returns:
    A left-right flipped image if condition is met, otherwise original image.
  """
  # DO NOT REMOVE - Logging usage.

  should_transform = jax.random.bernoulli(key=key, p=probability)
  return jax.lax.cond(should_transform, flip_left_right, lambda x: x, image)


def random_flip_up_down(
    key: chex.PRNGKey,
    image: chex.Array,
    *,
    probability: chex.Numeric = 0.5,
) -> chex.Array:
  """Applies `flip_up_down` with a given probability.

  Args:
    key: a JAX RNG key.
    image: a JAX array representing an image. Assumes that the image is either
      ...HWC or ...CHW.
    probability: the probability of applying flip_up_down transform. Must be a
      value in [0, 1].

  Returns:
    An up-down flipped image if condition is met, otherwise original image.
  """
  # DO NOT REMOVE - Logging usage.

  should_transform = jax.random.bernoulli(key=key, p=probability)
  return jax.lax.cond(should_transform, flip_up_down, lambda x: x, image)


def random_brightness(
    key: chex.PRNGKey,
    image: chex.Array,
    max_delta: chex.Numeric,
) -> chex.Array:
  """`adjust_brightness(...)` with random delta in `[-max_delta, max_delta)`."""
  # DO NOT REMOVE - Logging usage.

  delta = jax.random.uniform(key, (), minval=-max_delta, maxval=max_delta)
  return adjust_brightness(image, delta)


def random_gamma(
    key: chex.PRNGKey,
    image: chex.Array,
    min_gamma: chex.Numeric,
    max_gamma: chex.Numeric,
    *,
    gain: chex.Numeric = 1,
    assume_in_bounds: bool = False,
) -> chex.Array:
  """`adjust_gamma(...)` with random gamma in [min_gamma, max_gamma)`."""
  # DO NOT REMOVE - Logging usage.

  gamma = jax.random.uniform(key, (), minval=min_gamma, maxval=max_gamma)
  return adjust_gamma(
      image, gamma, gain=gain, assume_in_bounds=assume_in_bounds)


def random_hue(
    key: chex.PRNGKey,
    image: chex.Array,
    max_delta: chex.Numeric,
    *,
    channel_axis: int = -1,
) -> chex.Array:
  """`adjust_hue(...)` with random delta in `[-max_delta, max_delta)`."""
  # DO NOT REMOVE - Logging usage.

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
  # DO NOT REMOVE - Logging usage.

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
  # DO NOT REMOVE - Logging usage.

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
  # DO NOT REMOVE - Logging usage.

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


def _get_interpolate_function(
    mode: str,
    order: int,
    cval: float = 0.,
) -> Callable[[chex.Array, chex.Array], chex.Array]:
  """Selects the interpolation function to use based on the given parameters.

  PIX interpolations are preferred given they are faster on accelerators. For
  the cases where such interpolation is not implemented by PIX we really on
  jax.scipy.ndimage.map_coordinates. See specifics below.

  Args:
    mode: the mode parameter determines how the input array is extended beyond
      its boundaries. Modes 'nearest and 'constant' use PIX interpolation, which
      is very fast on accelerators (especially on TPUs). For all other modes,
      'wrap', 'mirror' and 'reflect', we rely on
      `jax.scipy.ndimage.map_coordinates`, which however is slow on
      accelerators, so use it with care.
    order: the order of the spline interpolation. The order has to be in the
      range [0, 1]. Note that PIX interpolation will only be used for order=1,
      for other values we use `jax.scipy.ndimage.map_coordinates`.
    cval: value to fill past edges of input if mode is 'constant'.

  Returns:
    The selected interpolation function.
  """
  if mode == "nearest" and order == 1:
    interpolate_function = interpolation.flat_nd_linear_interpolate
  elif mode == "constant" and order == 1:
    interpolate_function = functools.partial(
        interpolation.flat_nd_linear_interpolate_constant, cval=cval)
  else:
    interpolate_function = functools.partial(
        jax.scipy.ndimage.map_coordinates, mode=mode, order=order, cval=cval)
  return interpolate_function


def _get_dimension_values(
    image: chex.Array,
    channel_axis: int,
) -> Tuple[Optional[int], int, int, int]:
  """Gets shape values in BHWC order.

  If single image is given B is None.

  Small utility to get dimension values regardless of channel axis and single
  image or batch of images are passed.

  Args:
    image: a JAX array representing an image. Assumes that the image is either
      ...HWC or ...CHW.
    channel_axis: channel_axis: the index of the channel axis.

  Returns:
    A tuple with the values of each dimension in order BHWC.
  """
  chex.assert_rank(image, {3, 4})
  if image.ndim == 4:
    if _channels_last(image=image, channel_axis=channel_axis):
      batch, height, width, channel = image.shape
    else:
      batch, channel, height, width = image.shape
  else:
    if _channels_last(image=image, channel_axis=channel_axis):
      batch, (height, width, channel) = None, image.shape
    else:
      batch, (channel, height, width) = None, image.shape
  return batch, height, width, channel
