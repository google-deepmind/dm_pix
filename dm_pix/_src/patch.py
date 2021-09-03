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


def extract_patches(
    images: chex.Array,
    sizes: Sequence[int],
    strides: Sequence[int],
    rates: Sequence[int],
    *,
    padding: str = 'VALID',
) -> jnp.ndarray:
  """Extract patches from images.

  This function is a wrapper for jax.lax.conv_general_dilated_patches
  to conform to the same interface as tf.image.extract_patches.

  The function extracts patches of shape `sizes` from `images` in the same
  manner as a convolution with kernel of shape `sizes`, stride equal to
  `strides`, and the given `padding` scheme. The patches are stacked in the
  channel dimension.

  Args:
    images: input batch of images of shape [B, H, W, C].
    sizes: size of the extracted patches. Must be [1, size_rows, size_cols, 1].
    strides: how far the centers of two consecutive patches are in the images.
      Must be [1, stride_rows, stride_cols, 1].
    rates: sampling rate. Must be [1, rate_rows, rate_cols, 1]. This is the
      input stride, specifying how far two consecutive patch samples are in the
      input. Equivalent to extracting patches with `patch_sizes_eff =
      patch_sizes + (patch_sizes - 1) * (rates - 1)`, followed by subsampling
      them spatially by a factor of rates. This is equivalent to rate in dilated
      (a.k.a. Atrous) convolutions.
    padding: the type of padding algorithm to use.

  Returns:
    Tensor of shape [B, patch_rows, patch_cols, size_rows * size_cols * C].
  """
  if len(sizes) != 4 or sizes[0] != 1 or sizes[3] != 1:
    raise ValueError('Input `sizes` must be [1, size_rows, size_cols, 1]. '
                     f'Got {sizes}.')
  if len(strides) != 4 or strides[0] != 1 or strides[3] != 1:
    raise ValueError('Input `strides` must be [1, size_rows, size_cols, 1]. '
                     f'Got {strides}.')
  if len(rates) != 4 or rates[0] != 1 or rates[3] != 1:
    raise ValueError('Input `rates` must be [1, size_rows, size_cols, 1]. '
                     f'Got {rates}.')
  if images.ndim != 4:
    raise ValueError('Rank of `images` must be 4. '
                     f'Got shape {jnp.shape(images)})')

  # Rearrange axes of images to NCHW for conv_general_dilated_patches, then
  # move channel to the final dimension.
  images = jnp.transpose(images, [0, 3, 1, 2])
  channels = images.shape[1]
  patches = jax.lax.conv_general_dilated_patches(
      lhs=images,
      filter_shape=sizes[1:-1],
      window_strides=strides[1:-1],
      padding=padding,
      rhs_dilation=rates[1:-1],
  )
  patches = jnp.transpose(patches, [0, 2, 3, 1])

  # `conv_general_dilated_patches returns patches` is channel-major order,
  # rearrange to match interface of `tf.image.extract_patches`.
  patches = jnp.reshape(patches,
                        patches.shape[:3] + (channels, sizes[1], sizes[2]))
  patches = jnp.transpose(patches, [0, 1, 2, 4, 5, 3])
  patches = jnp.reshape(patches, patches.shape[:3] + (-1,))
  return patches
