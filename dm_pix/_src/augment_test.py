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
"""Tests for dm_pix._src.augment."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from dm_pix._src import augment
import jax
import jax.test_util as jtu
import numpy as np
import tensorflow as tf

_IMG_SHAPE = (131, 111, 3)
_RAND_FLOATS_IN_RANGE = list(
    np.random.uniform(0., 1., size=(10,) + _IMG_SHAPE).astype(np.float32))
_RAND_FLOATS_OUT_OF_RANGE = list(
    np.random.uniform(-0.5, 1.5, size=(10,) + _IMG_SHAPE).astype(np.float32))
_KERNEL_SIZE = _IMG_SHAPE[0] / 10.


class _ImageAugmentationTest(jtu.JaxTestCase, parameterized.TestCase):
  """Runs tests for the various augments with the correct arguments."""

  def _test_fn_with_random_arg(self, images_list, jax_fn, tf_fn, **kw_range):
    pass

  def _test_fn(self, images_list, jax_fn, tf_fn):
    pass

  def assertAllCloseTolerant(self, x, y):
    # Increase tolerance on TPU due to lower precision.
    tol = 1e-2 if jax.local_devices()[0].platform == "tpu" else 1e-4
    super().assertAllClose(x, y, rtol=tol, atol=tol)

  @parameterized.named_parameters(("in_range", _RAND_FLOATS_IN_RANGE),
                                  ("out_of_range", _RAND_FLOATS_OUT_OF_RANGE))
  def test_adjust_brightness(self, images_list):
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=augment.adjust_brightness,
        tf_fn=tf.image.adjust_brightness,
        delta=(-0.5, 0.5))

    key = jax.random.PRNGKey(0)
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=functools.partial(augment.random_brightness, key),
        tf_fn=None,
        max_delta=(0, 0.5))

  @parameterized.named_parameters(("in_range", _RAND_FLOATS_IN_RANGE),
                                  ("out_of_range", _RAND_FLOATS_OUT_OF_RANGE))
  def test_adjust_contrast(self, images_list):
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=augment.adjust_contrast,
        tf_fn=tf.image.adjust_contrast,
        factor=(0.5, 1.5))
    key = jax.random.PRNGKey(0)
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=functools.partial(augment.random_contrast, key, upper=1),
        tf_fn=None,
        lower=(0, 0.9))

  # Doesn't make sense outside of [0, 1].
  @parameterized.named_parameters(("in_range", _RAND_FLOATS_IN_RANGE))
  def test_adjust_gamma(self, images_list):
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=augment.adjust_gamma,
        tf_fn=tf.image.adjust_gamma,
        gamma=(0.5, 1.5))

  @parameterized.named_parameters(("in_range", _RAND_FLOATS_IN_RANGE),
                                  ("out_of_range", _RAND_FLOATS_OUT_OF_RANGE))
  def test_adjust_saturation(self, images_list):
    # tf.image.adjust_saturation has a buggy implementation when the green and
    # blue channels have very close values that don't match the red channel.
    # This is due to a rounding error in http://shortn/_ETSJsEwUj5
    # if (g - b) < 0 but small enough that (hh + 1) == 1.
    # Eg: tf.image.adjust_saturation([[[0.75, 0.0369078, 0.0369079]]], 1.0)
    #     -> [[[0.03690779, 0.03690779, 0.03690779]]]
    # Perturb the inputs slightly so that this doesn't happen.
    def perturb(rgb):
      rgb_new = np.copy(rgb)
      rgb_new[..., 1] += 0.001 * (np.abs(rgb[..., 2] - rgb[..., 1]) < 1e-3)
      return rgb_new

    images_list = list(map(perturb, images_list))
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=augment.adjust_saturation,
        tf_fn=tf.image.adjust_saturation,
        factor=(0.5, 1.5))
    key = jax.random.PRNGKey(0)
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=functools.partial(augment.random_saturation, key, upper=1),
        tf_fn=None,
        lower=(0, 0.9))

  # CPU TF uses a different hue adjustment method outside of the [0, 1] range.
  # Disable out-of-range tests.
  @parameterized.named_parameters(
      ("in_range", _RAND_FLOATS_IN_RANGE),)
  def test_adjust_hue(self, images_list):
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=augment.adjust_hue,
        tf_fn=tf.image.adjust_hue,
        delta=(-0.5, 0.5))
    key = jax.random.PRNGKey(0)
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=functools.partial(augment.random_hue, key),
        tf_fn=None,
        max_delta=(0, 0.5))

  @parameterized.named_parameters(("in_range", _RAND_FLOATS_IN_RANGE),
                                  ("out_of_range", _RAND_FLOATS_OUT_OF_RANGE))
  def test_rot90(self, images_list):
    self._test_fn(
        images_list,
        jax_fn=lambda img: augment.rot90(img, k=1),
        tf_fn=lambda img: tf.image.rot90(img, k=1))
    self._test_fn(
        images_list,
        jax_fn=lambda img: augment.rot90(img, k=2),
        tf_fn=lambda img: tf.image.rot90(img, k=2))
    self._test_fn(
        images_list,
        jax_fn=lambda img: augment.rot90(img, k=3),
        tf_fn=lambda img: tf.image.rot90(img, k=3))

  # The functions below don't have a TF equivalent to compare to, we just check
  # that they run.
  @parameterized.named_parameters(("in_range", _RAND_FLOATS_IN_RANGE),
                                  ("out_of_range", _RAND_FLOATS_OUT_OF_RANGE))
  def test_flip(self, images_list):
    self._test_fn(
        images_list,
        jax_fn=augment.flip_left_right,
        tf_fn=tf.image.flip_left_right)
    self._test_fn(
        images_list, jax_fn=augment.flip_up_down, tf_fn=tf.image.flip_up_down)
    key = jax.random.PRNGKey(0)
    self._test_fn(
        images_list,
        jax_fn=functools.partial(augment.random_flip_left_right, key),
        tf_fn=None)
    self._test_fn(
        images_list,
        jax_fn=functools.partial(augment.random_flip_up_down, key),
        tf_fn=None)

  @parameterized.named_parameters(("in_range", _RAND_FLOATS_IN_RANGE),
                                  ("out_of_range", _RAND_FLOATS_OUT_OF_RANGE))
  def test_solarize(self, images_list):
    self._test_fn_with_random_arg(
        images_list, jax_fn=augment.solarize, tf_fn=None, threshold=(0., 1.))

  @parameterized.named_parameters(("in_range", _RAND_FLOATS_IN_RANGE),
                                  ("out_of_range", _RAND_FLOATS_OUT_OF_RANGE))
  def test_gaussian_blur(self, images_list):
    blur_fn = functools.partial(augment.gaussian_blur, kernel_size=_KERNEL_SIZE)
    self._test_fn_with_random_arg(
        images_list, jax_fn=blur_fn, tf_fn=None, sigma=(0.1, 2.0))

  @parameterized.named_parameters(("in_range", _RAND_FLOATS_IN_RANGE),
                                  ("out_of_range", _RAND_FLOATS_OUT_OF_RANGE))
  def test_random_crop(self, images_list):
    key = jax.random.PRNGKey(43)
    crop_fn = lambda img: augment.random_crop(key, img, (100, 100, 3))
    self._test_fn(images_list, jax_fn=crop_fn, tf_fn=None)


class TestMatchTensorflow(_ImageAugmentationTest):

  def _test_fn_with_random_arg(self, images_list, jax_fn, tf_fn, **kw_range):
    if tf_fn is None:
      return
    assert len(kw_range) == 1
    kw_name, (random_min, random_max) = list(kw_range.items())[0]
    for image_rgb in images_list:
      argument = np.random.uniform(random_min, random_max, size=())
      adjusted_jax = jax_fn(image_rgb, **{kw_name: argument})
      adjusted_tf = tf_fn(image_rgb, argument).numpy()
      self.assertAllCloseTolerant(adjusted_jax, adjusted_tf)

  def _test_fn(self, images_list, jax_fn, tf_fn):
    if tf_fn is None:
      return
    for image_rgb in images_list:
      adjusted_jax = jax_fn(image_rgb)
      adjusted_tf = tf_fn(image_rgb).numpy()
      self.assertAllCloseTolerant(adjusted_jax, adjusted_tf)


class TestVmap(_ImageAugmentationTest):

  def _test_fn_with_random_arg(self, images_list, jax_fn, tf_fn, **kw_range):
    del tf_fn  # unused.
    assert len(kw_range) == 1
    kw_name, (random_min, random_max) = list(kw_range.items())[0]
    arguments = [
        np.random.uniform(random_min, random_max, size=()) for _ in images_list
    ]
    fn_vmap = jax.vmap(jax_fn)
    outputs_vmaped = list(
        fn_vmap(np.stack(images_list, axis=0), np.stack(arguments, axis=0)))
    assert len(images_list) == len(outputs_vmaped)
    assert len(images_list) == len(arguments)
    for image_rgb, argument, adjusted_vmap in zip(images_list, arguments,
                                                  outputs_vmaped):
      adjusted_jax = jax_fn(image_rgb, **{kw_name: argument})
      self.assertAllCloseTolerant(adjusted_jax, adjusted_vmap)

  def _test_fn(self, images_list, jax_fn, tf_fn):
    del tf_fn  # unused.
    fn_vmap = jax.vmap(jax_fn)
    outputs_vmaped = list(fn_vmap(np.stack(images_list, axis=0)))
    assert len(images_list) == len(outputs_vmaped)
    for image_rgb, adjusted_vmap in zip(images_list, outputs_vmaped):
      adjusted_jax = jax_fn(image_rgb)
      self.assertAllCloseTolerant(adjusted_jax, adjusted_vmap)


class TestJit(_ImageAugmentationTest):

  def _test_fn_with_random_arg(self, images_list, jax_fn, tf_fn, **kw_range):
    del tf_fn  # unused.
    assert len(kw_range) == 1
    kw_name, (random_min, random_max) = list(kw_range.items())[0]
    jax_fn_jitted = jax.jit(jax_fn)
    for image_rgb in images_list:
      argument = np.random.uniform(random_min, random_max, size=())
      adjusted_jax = jax_fn(image_rgb, argument)
      adjusted_jit = jax_fn_jitted(image_rgb, **{kw_name: argument})
      self.assertAllCloseTolerant(adjusted_jax, adjusted_jit)

  def _test_fn(self, images_list, jax_fn, tf_fn):
    del tf_fn  # unused.
    jax_fn_jitted = jax.jit(jax_fn)
    for image_rgb in images_list:
      adjusted_jax = jax_fn(image_rgb)
      adjusted_jit = jax_fn_jitted(image_rgb)
      self.assertAllCloseTolerant(adjusted_jax, adjusted_jit)


if __name__ == "__main__":
  absltest.main()
