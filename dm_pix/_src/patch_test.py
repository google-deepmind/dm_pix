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
"""Tests for dm_pix._src.patch."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from dm_pix._src import patch
import jax.test_util as jtu
import numpy as np
import tensorflow as tf


def _create_test_images(shape):
  images = np.arange(np.prod(np.array(shape)), dtype=np.float32)
  return np.reshape(images, shape)


class PatchTest(chex.TestCase, jtu.JaxTestCase, parameterized.TestCase):

  @chex.all_variants
  @parameterized.named_parameters(
      ('padding_valid', 'VALID'),
      ('padding_same', 'SAME'),
  )
  def test_extract_patches(self, padding):
    image_shape = (2, 5, 7, 3)
    images = _create_test_images(image_shape)

    sizes = (1, 2, 3, 1)
    strides = (1, 1, 2, 1)
    rates = (1, 2, 1, 1)

    extract_patches = self.variant(
        functools.partial(patch.extract_patches, padding=padding),
        static_argnums=(1, 2, 3))
    jax_patches = extract_patches(
        images,
        sizes,
        strides,
        rates,
    )
    tf_patches = tf.image.extract_patches(
        images,
        sizes=sizes,
        strides=strides,
        rates=rates,
        padding=padding,
    )
    self.assertArraysEqual(jax_patches, tf_patches.numpy())

  @chex.all_variants
  @parameterized.product(
      ({
          'sizes': (1, 2, 3),
          'strides': (1, 1, 2, 1),
          'rates': (1, 2, 1, 1),
      }, {
          'sizes': (1, 2, 3, 1),
          'strides': (1, 1, 2),
          'rates': (1, 2, 1, 1),
      }, {
          'sizes': (1, 2, 3, 1),
          'strides': (1, 1, 2, 1),
          'rates': (1, 2, 1),
      }),
      padding=('VALID', 'SAME'),
  )
  def test_extract_patches_raises(self, sizes, strides, rates, padding):
    image_shape = (2, 5, 7, 3)
    images = _create_test_images(image_shape)

    extract_patches = self.variant(
        functools.partial(patch.extract_patches, padding=padding),
        static_argnums=(1, 2, 3))
    with self.assertRaises(ValueError):
      extract_patches(
          images,
          sizes,
          strides,
          rates,
      )


if __name__ == '__main__':
  absltest.main()
