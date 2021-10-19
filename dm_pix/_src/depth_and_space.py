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
"""This module provides functions for rearranging blocks of spatial data."""

import chex
import jax
import jax.numpy as jnp


def depth_to_space(inputs: chex.Array, block_size: int) -> chex.Array:
  """Rearranges data from depth into blocks of spatial data.

  Args:
    inputs: Array of shape [H, W, C] or [N, H, W, C]. The number of channels
      (depth dimension) must be divisible by block_size ** 2.
    block_size: Size of spatial blocks >= 2.

  Returns:
    For inputs of shape [H, W, C] the output is a reshaped array of shape
      [H * B, W * B, C / (B ** 2)], where B is `block_size`. If there's a
      leading batch dimension, it stays unchanged.
  """
  chex.assert_rank(inputs, {3, 4})
  if inputs.ndim == 4:  # Batched case.
    return jax.vmap(depth_to_space, in_axes=(0, None))(inputs, block_size)

  height, width, depth = inputs.shape
  if depth % (block_size**2) != 0:
    raise ValueError(
        f'Number of channels {depth} must be divisible by block_size ** 2 {block_size**2}.'
    )
  new_depth = depth // (block_size**2)
  outputs = jnp.reshape(inputs,
                        [height, width, block_size, block_size, new_depth])
  outputs = jnp.transpose(outputs, [0, 2, 1, 3, 4])
  outputs = jnp.reshape(outputs,
                        [height * block_size, width * block_size, new_depth])
  return outputs


def space_to_depth(inputs: chex.Array, block_size: int) -> chex.Array:
  """Rearranges data from blocks of spatial data into depth.

  This is the reverse of depth_to_space.
  Args:
    inputs: Array of shape [H, W, C] or [N, H, W, C]. The height and width must
      each be divisible by block_size.
    block_size: Size of spatial blocks >= 2.

  Returns:
    For inputs of shape [H, W, C] the output is a reshaped array of shape
      [H / B, W / B, C * (B ** 2)], where B is `block_size`. If there's a
      leading batch dimension, it stays unchanged.
  """
  chex.assert_rank(inputs, {3, 4})
  if inputs.ndim == 4:  # Batched case.
    return jax.vmap(space_to_depth, in_axes=(0, None))(inputs, block_size)

  height, width, depth = inputs.shape
  if height % block_size != 0:
    raise ValueError(
        f'Height {height} must be divisible by block size {block_size}.')
  if width % block_size != 0:
    raise ValueError(
        f'Width {width} must be divisible by block size {block_size}.')
  new_depth = depth * (block_size**2)
  new_height = height // block_size
  new_width = width // block_size
  outputs = jnp.reshape(inputs,
                        [new_height, block_size, new_width, block_size, depth])
  outputs = jnp.transpose(outputs, [0, 2, 1, 3, 4])
  outputs = jnp.reshape(outputs, [new_height, new_width, new_depth])
  return outputs
