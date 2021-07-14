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
"""Tests for dm_pix._src.interpolation."""

from typing import Sequence, Tuple

from absl.testing import absltest
from absl.testing import parameterized
import chex
from dm_pix._src import interpolation
import jax.numpy as jnp
import jax.test_util as jtu

_SHAPE_COORDS = ((1, 1), (1, 3), (3, 2), (4, 4), (4, 1, 4), (4, 2, 2))


def _prepare_inputs(
    shape_output_coordinates: Tuple[int]) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Returns the volume and coordinates to be used in the function under test.

  Args:
    shape_output_coordinates: [N, M] shape for the output coordinates, where N
      is a scalar that determines also the number of dimensions in the volume
      and M is either a scalar (output coordinates will be a 1D array) or a
      vector.
  """
  template_coords = jnp.array([(2, 0, 3, 2.9), (1, 0, 4.3, 1), (1, 0.5, 8.4, 1),
                               (21, 0.5, 4, 1)])

  template_volume = jnp.array(
      [[[[1.6583091, 2.0139587, 2.4636955, 0.11345804, 4.044214],
         [4.538101, 4.3030543, 0.6967968, 2.0311975, 2.5746036],
         [0.52024364, 4.767304, 2.3863382, 2.496363, 3.7334495],
         [1.367867, 4.18175, 0.38294435, 3.9395797, 2.6097183]],
        [[0.7470304, 3.8882136, 0.42186677, 1.9224191, 2.3947673],
         [4.859208, 2.7876246, 0.7796812, 3.234911, 2.0911336],
         [3.9205093, 4.027418, 2.9367173, 4.367462, 0.5682403],
         [3.32689, 0.5056447, 1.3147497, 3.549356, 0.57163835]]],
       [[[4.6056757, 3.0942523, 4.809611, 0.6062186, 4.1184435],
         [1.0862654, 1.0130441, 0.24880886, 2.9144812, 2.831624],
         [0.8990741, 4.6315174, 3.490876, 3.997823, 3.166548],
         [2.2909844, 2.1135485, 0.7603508, 1.7530066, 3.3882804]],
        [[2.2388606, 0.62632084, 0.39939642, 1.2361205, 4.4961414],
         [1.3705498, 4.6373777, 2.2974424, 2.9484348, 1.8847889],
         [4.856637, 3.4407651, 1.5632284, 0.30945182, 4.8406916],
         [4.10108, 0.44603765, 3.893259, 2.656221, 4.652004]]],
       [[[1.8670297, 4.1097646, 3.9615297, 0.9295058, 3.9903827],
         [3.3507752, 1.4316595, 4.0365667, 2.3517795, 2.7806897],
         [1.245628, 4.8092294, 3.3148618, 3.6758037, 2.4036856],
         [4.2023296, 0.6232512, 2.2606378, 2.1633143, 3.019858]],
        [[3.6607206, 0.26809275, 0.43593287, 0.3059131, 0.5254775],
         [0.27680695, 0.88441014, 4.8790736, 4.796288, 4.922847],
         [3.3822608, 2.5350225, 3.771946, 0.46694577, 4.0173407],
         [4.835033, 4.4530325, 1.4543611, 4.67758, 3.4009826]]]])

  if shape_output_coordinates[0] == 1:
    volume = template_volume[0, 0, 0, :]
  elif shape_output_coordinates[0] == 3:
    volume = template_volume[0, :, :, :]
  elif shape_output_coordinates[0] == template_volume.ndim:
    volume = template_volume
  else:
    raise ValueError("Unsupported shape_output_coordinates[0] = "
                     f"{shape_output_coordinates[0]}")

  if len(shape_output_coordinates) == 2:
    if shape_output_coordinates <= template_coords.shape:
      # Get a slice of the `template_coords`.
      coordinates = template_coords[0:shape_output_coordinates[0],
                                    0:shape_output_coordinates[1]]

      if shape_output_coordinates[1] == 1:
        # Do [[ num ]] -> [ num ] to test special case.
        coordinates = coordinates.squeeze(axis=-1)
    else:
      raise ValueError("Unsupported shape_output_coordinates[1] = "
                       f"{shape_output_coordinates[1]}")
  else:
    try:
      # In this case, try reshaping the _TEMPLATE_COORDS to the desired shape.
      coordinates = jnp.reshape(template_coords, shape_output_coordinates)
    except TypeError:
      raise ValueError(f"Unsupported shape_output_coordinates = "
                       f"{shape_output_coordinates}")

  return volume, coordinates


def _prepare_expected(shape_coordinates: Sequence[int]) -> jnp.ndarray:
  if len(shape_coordinates) == 2:
    if tuple(shape_coordinates) == (1, 1):
      out = jnp.array(2.4636955)
    elif tuple(shape_coordinates) == (1, 3):
      out = jnp.array([2.4636955, 1.6583091, 0.11345804])
    elif tuple(shape_coordinates) == (3, 2):
      out = jnp.array([2.7876246, 1.836134])
    elif shape_coordinates[0] == 4:
      out = jnp.array([4.922847, 3.128356, 3.4009826, 0.88441014])
    else:
      raise ValueError(f"Unsupported shape_coordinates = {shape_coordinates}")
  elif shape_coordinates[0] == 4:
    try:
      out = jnp.array([4.922847, 3.128356, 3.4009826, 0.88441014])
      out = jnp.reshape(out, shape_coordinates[1:])
    except TypeError:
      raise ValueError(f"Unsupported shape_coordinates = {shape_coordinates}")
  else:
    raise ValueError(f"Unsupported shape_coordinates = {shape_coordinates}")

  return out


class InterpolationTest(chex.TestCase, jtu.JaxTestCase, parameterized.TestCase):

  @chex.all_variants
  @parameterized.named_parameters(
      jtu.cases_from_list(
          dict(testcase_name=f"_{shape}_coords", shape_coordinates=shape)
          for shape in _SHAPE_COORDS))
  def test_flat_nd_linear_interpolate(self, shape_coordinates):
    volume, coords = _prepare_inputs(shape_coordinates)
    expected = _prepare_expected(shape_coordinates)

    flat_nd_linear_interpolate = self.variant(
        interpolation.flat_nd_linear_interpolate)
    self.assertAllClose(flat_nd_linear_interpolate(volume, coords), expected)
    self.assertAllClose(
        flat_nd_linear_interpolate(
            volume.flatten(), coords, unflattened_vol_shape=volume.shape),
        expected)


if __name__ == "__main__":
  absltest.main()
