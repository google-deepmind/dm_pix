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
"""Tests for dm_pix._src.metrics."""

import functools

from absl.testing import absltest
import chex
from dm_pix._src import metrics
import jax
import jax.test_util as jtu
import numpy as np
import tensorflow as tf


class MSETest(chex.TestCase, jtu.JaxTestCase, absltest.TestCase):

  def setUp(self):
    super().setUp()
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    self._img1 = jax.random.uniform(
        key1,
        shape=(4, 32, 32, 3),
        minval=0.,
        maxval=1.,
    )
    self._img2 = jax.random.uniform(
        key2,
        shape=(4, 32, 32, 3),
        minval=0.,
        maxval=1.,
    )

  @chex.all_variants
  def test_psnr_match(self):
    psnr = self.variant(metrics.psnr)
    values_jax = psnr(self._img1, self._img2)
    values_tf = tf.image.psnr(self._img1, self._img2, max_val=1.).numpy()
    self.assertAllClose(values_jax, values_tf, rtol=1e-3, atol=1e-3)

  @chex.all_variants
  def test_simse_invariance(self):
    simse = self.variant(metrics.simse)
    simse_jax = simse(self._img1, self._img1 * 2.0)
    self.assertAllClose(simse_jax, np.zeros(4), rtol=1e-6, atol=1e-6)


class SSIMTests(chex.TestCase, jtu.JaxTestCase, absltest.TestCase):

  @chex.all_variants
  def test_ssim_golden(self):
    """Test that the SSIM implementation matches the Tensorflow version."""

    key = jax.random.PRNGKey(0)
    for shape in ((2, 12, 12, 3), (12, 12, 3), (2, 12, 15, 3), (17, 12, 3)):
      for _ in range(4):
        (max_val_key, img0_key, img1_key, filter_size_key, filter_sigma_key,
         k1_key, k2_key, key) = jax.random.split(key, 8)
        max_val = jax.random.uniform(max_val_key, minval=0.1, maxval=3.)
        img0 = max_val * jax.random.uniform(img0_key, shape=shape)
        img1 = max_val * jax.random.uniform(img1_key, shape=shape)
        filter_size = jax.random.randint(
            filter_size_key, shape=(), minval=1, maxval=10)
        filter_sigma = jax.random.uniform(
            filter_sigma_key, shape=(), minval=0.1, maxval=10.)
        k1 = jax.random.uniform(k1_key, shape=(), minval=0.001, maxval=0.1)
        k2 = jax.random.uniform(k2_key, shape=(), minval=0.001, maxval=0.1)

        ssim_gt = tf.image.ssim(
            img0,
            img1,
            max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2).numpy()
        for return_map in [False, True]:
          ssim_fn = self.variant(
              functools.partial(
                  metrics.ssim,
                  max_val=max_val,
                  filter_size=filter_size,
                  filter_sigma=filter_sigma,
                  k1=k1,
                  k2=k2,
                  return_map=return_map,
              ))
          ssim = ssim_fn(img0, img1)
          if not return_map:
            self.assertAllClose(ssim, ssim_gt, atol=1e-5, rtol=1e-5)
          else:
            self.assertAllClose(
                np.mean(ssim, list(range(-3, 0))),
                ssim_gt,
                atol=1e-5,
                rtol=1e-5)
          self.assertLessEqual(np.max(ssim), 1.)
          self.assertGreaterEqual(np.min(ssim), -1.)

  @chex.all_variants
  def test_ssim_lowerbound(self):
    """Test the unusual corner case where SSIM is -1."""
    filter_size = 11
    grid_coords = [np.linspace(-1, 1, filter_size)] * 2
    img = np.meshgrid(*grid_coords)[0][np.newaxis, ..., np.newaxis]
    eps = 1e-5
    ssim_fn = self.variant(
        functools.partial(
            metrics.ssim,
            max_val=1.,
            filter_size=filter_size,
            filter_sigma=1.5,
            k1=eps,
            k2=eps,
        ))
    ssim = ssim_fn(img, -img)
    self.assertAllClose(ssim, -np.ones_like(ssim))


if __name__ == "__main__":
  absltest.main()
