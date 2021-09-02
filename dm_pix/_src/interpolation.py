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
"""This module provides functions for interpolating ND images.

All functions expect float-encoded images, with values in [0, 1].
"""

import itertools
from typing import Optional, Sequence, Tuple

import chex
from jax import lax
import jax.numpy as jnp


def _round_half_away_from_zero(a: chex.Array) -> chex.Array:
  return a if jnp.issubdtype(a.dtype, jnp.integer) else lax.round(a)


def _make_linear_interpolation_indices_nd(
    coordinates: chex.Array,
    shape: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
  """Creates linear interpolation indices and weights for ND coordinates.

  Args:
    coordinates: An array of shape (N, M_coordinates).
    shape: The shape of the ND volume, e.g. if N=3 shape=(dim_z, dim_y, dim_x).

  Returns:
    The lower and upper indices of `coordinates` and their weights.
  """
  lower = jnp.floor(coordinates).astype(jnp.int32)
  upper = jnp.ceil(coordinates).astype(jnp.int32)
  weights = coordinates - lower

  # Expand dimensions for `shape` to allow broadcasting it to every coordinate.
  # Expansion size is equal to the number of dimensions of `coordinates` - 1.
  shape = shape.reshape(shape.shape + (1,) * (coordinates.ndim - 1))

  lower = jnp.clip(lower, 0, shape - 1)
  upper = jnp.clip(upper, 0, shape - 1)

  return lower, upper, weights


def _make_linear_interpolation_indices_flat_nd(
    coordinates: chex.Array,
    shape: Sequence[int],
) -> Tuple[chex.Array, chex.Array]:
  """Creates flat linear interpolation indices and weights for ND coordinates.

  Args:
    coordinates: An array of shape (N, M_coordinates).
    shape: The shape of the ND volume, e.g. if N=3 shape=(dim_z, dim_y, dim_x).

  Returns:
    The indices into the flattened input and their weights.
  """
  coordinates = jnp.asarray(coordinates)
  shape = jnp.asarray(shape)

  if shape.shape[0] != coordinates.shape[0]:
    raise ValueError(
        (f'{coordinates.shape[0]}-dimensional coordinates provided for '
         f'{shape.shape[0]}-dimensional input'))

  lower_nd, upper_nd, weights_nd = _make_linear_interpolation_indices_nd(
      coordinates, shape)

  # Here we want to translate e.g. a 3D-disposed indices to linear ones, since
  # we have to index on the flattened source, so:
  # flat_idx = shape[1] * shape[2] * z_idx + shape[2] * y_idx + x_idx

  # The `strides` of a `shape`-sized array tell us how many elements we have to
  # skip to move to the next position along a certain axis in that array.
  # For example, for a shape=(5,4,2) we have to skip 1 value to move to the next
  # column (3rd axis), 2 values to move to get to the same position in the next
  # row (2nd axis) and 4*2=8 values to move to get to the same position on the
  # 1st axis.
  strides = jnp.concatenate([jnp.cumprod(shape[:0:-1])[::-1], jnp.array([1])])

  # Array of 2^n rows where the ith row is the binary representation of i.
  binary_array = jnp.array(
      list(itertools.product([0, 1], repeat=shape.shape[0])))

  # Expand dimensions to allow broadcasting `strides` and `binary_array` to
  # every coordinate.
  # Expansion size is equal to the number of dimensions of `coordinates` - 1.
  strides = strides.reshape(strides.shape + (1,) * (coordinates.ndim - 1))
  binary_array = binary_array.reshape(binary_array.shape + (1,) *
                                      (coordinates.ndim - 1))

  lower_1d = lower_nd * strides
  upper_1d = upper_nd * strides

  point_weights = []
  point_indices = []

  for r in binary_array:
    # `point_indices` is defined as:
    # `jnp.matmul(binary_array, upper) + jnp.matmul(1-binary_array, lower)`
    # however, to date, that implementation turns out to be slower than the
    # equivalent following one.
    point_indices.append(jnp.sum(upper_1d * r + lower_1d * (1 - r), axis=0))
    point_weights.append(
        jnp.prod(r * weights_nd + (1 - r) * (1 - weights_nd), axis=0))
  return jnp.stack(point_indices, axis=0), jnp.stack(point_weights, axis=0)


def _linear_interpolate_using_indices_nd(
    volume: chex.Array,
    indices: chex.Array,
    weights: chex.Array,
) -> chex.Array:
  """Interpolates linearly on `volume` using `indices` and `weights`."""
  target = jnp.sum(weights * volume[indices], axis=0)
  if jnp.issubdtype(volume.dtype, jnp.integer):
    target = _round_half_away_from_zero(target)
  return target.astype(volume.dtype)


def flat_nd_linear_interpolate(
    volume: chex.Array,
    coordinates: chex.Array,
    *,
    unflattened_vol_shape: Optional[Sequence[int]] = None,
) -> chex.Array:
  """Maps the input ND volume to coordinates by linear interpolation.

  Args:
    volume: A volume (flat if `unflattened_vol_shape` is provided) where to
      query coordinates.
    coordinates: An array of shape (N, M_coordinates). Where M_coordinates can
      be M-dimensional. If M_coordinates == 1, then `coordinates.shape` can
      simply be (N,), e.g. if N=3 and M_coordinates=1, this has the form (z, y,
      x).
    unflattened_vol_shape: The shape of the `volume` before flattening. If
      provided, then `volume` must be pre-flattened.

  Returns:
    The resulting mapped coordinates. The shape of the output is `M_coordinates`
    (derived from `coordinates` by dropping the first axis).
  """
  if unflattened_vol_shape is None:
    unflattened_vol_shape = volume.shape
    volume = volume.flatten()

  indices, weights = _make_linear_interpolation_indices_flat_nd(
      coordinates, shape=unflattened_vol_shape)
  return _linear_interpolate_using_indices_nd(
      jnp.asarray(volume), indices, weights)
