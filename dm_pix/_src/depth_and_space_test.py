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
"""Tests for dm_pix._src.depth_and_space."""

from absl.testing import parameterized
import chex
from dm_pix._src import depth_and_space
import jax.test_util as jtu
import numpy as np
import tensorflow as tf


class DepthAndSpaceTest(chex.TestCase, jtu.JaxTestCase, parameterized.TestCase):

  @chex.all_variants
  @parameterized.parameters(([1, 1, 1, 9], 3), ([2, 2, 2, 8], 2))
  def test_depth_to_space(self, input_shape, block_size):
    depth_to_space_fn = self.variant(
        depth_and_space.depth_to_space, static_argnums=1)
    inputs = np.arange(np.prod(input_shape), dtype=np.int32)
    inputs = np.reshape(inputs, input_shape)
    output_tf = tf.nn.depth_to_space(inputs, block_size).numpy()
    output_jax = depth_to_space_fn(inputs, block_size)
    self.assertArraysEqual(output_tf, output_jax)

  @chex.all_variants
  @parameterized.parameters(([1, 3, 3, 1], 3), ([2, 4, 4, 2], 2))
  def test_space_to_depth(self, input_shape, block_size):
    space_to_depth_fn = self.variant(
        depth_and_space.space_to_depth, static_argnums=1)
    inputs = np.arange(np.prod(input_shape), dtype=np.int32)
    inputs = np.reshape(inputs, input_shape)
    output_tf = tf.nn.space_to_depth(inputs, block_size).numpy()
    output_jax = space_to_depth_fn(inputs, block_size)
    self.assertArraysEqual(output_tf, output_jax)


if __name__ == "__main__":
  tf.test.main()
