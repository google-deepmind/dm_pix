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
"""This module provides image patching functionality."""

from typing import Sequence

import chex
import jax
import jax.numpy as jnp

# DO NOT REMOVE - Logging lib.


def extract_patches(
    images: chex.Array,
    sizes: Sequence[int],
    strides: Sequence[int],
    rates: Sequence[int],
    *,
    padding: str = "VALID",
) -> jnp.ndarray:
  """Extract patches from images.

  This function is a wrapper for `jax.lax.conv_general_dilated_patches`
  to conform to the same interface as `tf.image.extract_patches`, except for
  this function supports arbitrary-dimensional `images`, not only 4D as in
  `tf.image.extract_patches`.

  The function extracts patches of shape `sizes` from `images` in the same
  manner as a convolution with kernel of shape `sizes`, stride equal to
  `strides`, and the given `padding` scheme. The patches are stacked in the
  channel dimension.

  Args:
    images: input batch of images of shape [B, H, W, ..., C].
    sizes: size of the extracted patches.
      Must be [1, size_rows, size_cols, ..., 1].
    strides: how far the centers of two consecutive patches are in the images.
      Must be [1, stride_rows, stride_cols, ..., 1].
    rates: sampling rate. Must be [1, rate_rows, rate_cols, ..., 1]. This is the
      input stride, specifying how far two consecutive patch samples are in the
      input. Equivalent to extracting patches with `patch_sizes_eff =
      patch_sizes + (patch_sizes - 1) * (rates - 1)`, followed by subsampling
      them spatially by a factor of rates. This is equivalent to rate in dilated
      (a.k.a. Atrous) convolutions.
    padding: the type of padding algorithm to use.

  Returns:
    Tensor of shape
    [B, patch_rows, patch_cols, ..., size_rows * size_cols * ... * C].
  """
  # DO NOT REMOVE - Logging usage.

  ndim = images.ndim

  if len(sizes) != ndim or sizes[0] != 1 or sizes[-1] != 1:
    raise ValueError("Input `sizes` must be [1, size_rows, size_cols, ..., 1] "
                     f"and same length as `images.ndim` {ndim}. Got {sizes}.")
  if len(strides) != ndim or strides[0] != 1 or strides[-1] != 1:
    raise ValueError("Input `strides` must be [1, size_rows, size_cols, ..., 1]"
                     f"and same length as `images.ndim` {ndim}. Got {strides}.")
  if len(rates) != ndim or rates[0] != 1 or rates[-1] != 1:
    raise ValueError("Input `rates` must be [1, size_rows, size_cols, ..., 1] "
                     f"and same length as `images.ndim` {ndim}. Got {rates}.")

  channels = images.shape[-1]
  lhs_spec = out_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = tuple(range(ndim))
  patches = jax.lax.conv_general_dilated_patches(
      lhs=images,
      filter_shape=sizes[1:-1],
      window_strides=strides[1:-1],
      padding=padding,
      rhs_dilation=rates[1:-1],
      dimension_numbers=jax.lax.ConvDimensionNumbers(
          lhs_spec, rhs_spec, out_spec)
  )

  # `conv_general_dilated_patches` returns `patches` in channel-major order,
  # rearrange to match interface of `tf.image.extract_patches`.
  patches = jnp.reshape(patches, patches.shape[:-1] + (channels, -1))
  patches = jnp.moveaxis(patches, -2, -1)
  patches = jnp.reshape(patches, patches.shape[:-2] + (-1,))
  return patches
