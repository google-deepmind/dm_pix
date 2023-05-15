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
import jax.numpy as jnp
import numpy as np
import scipy
import tensorflow as tf

_IMG_SHAPE = (131, 111, 3)
_RAND_FLOATS_IN_RANGE = list(
    np.random.uniform(0., 1., size=(10,) + _IMG_SHAPE).astype(np.float32))
_RAND_FLOATS_OUT_OF_RANGE = list(
    np.random.uniform(-0.5, 1.5, size=(10,) + _IMG_SHAPE).astype(np.float32))
_KERNEL_SIZE = _IMG_SHAPE[0] / 10.


class _ImageAugmentationTest(parameterized.TestCase):
  """Runs tests for the various augments with the correct arguments."""

  def _test_fn_with_random_arg(self, images_list, jax_fn, reference_fn,
                               **kw_range):
    pass

  def _test_fn(self, images_list, jax_fn, reference_fn):
    pass

  def assertAllCloseTolerant(self, x, y):
    # Increase tolerance on TPU due to lower precision.
    tol = 1e-2 if jax.local_devices()[0].platform == "tpu" else 1e-4
    np.testing.assert_allclose(x, y, rtol=tol, atol=tol)
    self.assertEqual(x.dtype, y.dtype)

  @parameterized.named_parameters(("in_range", _RAND_FLOATS_IN_RANGE),
                                  ("out_of_range", _RAND_FLOATS_OUT_OF_RANGE))
  def test_adjust_brightness(self, images_list):
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=augment.adjust_brightness,
        reference_fn=tf.image.adjust_brightness,
        delta=(-0.5, 0.5))

    key = jax.random.PRNGKey(0)
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=functools.partial(augment.random_brightness, key),
        reference_fn=None,
        max_delta=(0, 0.5))

  @parameterized.named_parameters(("in_range", _RAND_FLOATS_IN_RANGE),
                                  ("out_of_range", _RAND_FLOATS_OUT_OF_RANGE))
  def test_adjust_contrast(self, images_list):
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=augment.adjust_contrast,
        reference_fn=tf.image.adjust_contrast,
        factor=(0.5, 1.5))
    key = jax.random.PRNGKey(0)
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=functools.partial(augment.random_contrast, key, upper=1),
        reference_fn=None,
        lower=(0, 0.9))

  # Doesn't make sense outside of [0, 1].
  @parameterized.named_parameters(("in_range", _RAND_FLOATS_IN_RANGE))
  def test_adjust_gamma(self, images_list):
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=augment.adjust_gamma,
        reference_fn=tf.image.adjust_gamma,
        gamma=(0.5, 1.5))
    key = jax.random.PRNGKey(0)
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=functools.partial(augment.random_gamma, key, min_gamma=1),
        reference_fn=None,
        max_gamma=(1.5, 1.9))

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
        reference_fn=tf.image.adjust_saturation,
        factor=(0.5, 1.5))
    key = jax.random.PRNGKey(0)
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=functools.partial(augment.random_saturation, key, upper=1),
        reference_fn=None,
        lower=(0, 0.9))

  # CPU TF uses a different hue adjustment method outside of the [0, 1] range.
  # Disable out-of-range tests.
  @parameterized.named_parameters(
      ("in_range", _RAND_FLOATS_IN_RANGE),)
  def test_adjust_hue(self, images_list):
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=augment.adjust_hue,
        reference_fn=tf.image.adjust_hue,
        delta=(-0.5, 0.5))
    key = jax.random.PRNGKey(0)
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=functools.partial(augment.random_hue, key),
        reference_fn=None,
        max_delta=(0, 0.5))

  @parameterized.named_parameters(("in_range", _RAND_FLOATS_IN_RANGE),
                                  ("out_of_range", _RAND_FLOATS_OUT_OF_RANGE))
  def test_rot90(self, images_list):
    self._test_fn(
        images_list,
        jax_fn=lambda img: augment.rot90(img, k=1),
        reference_fn=lambda img: tf.image.rot90(img, k=1))
    self._test_fn(
        images_list,
        jax_fn=lambda img: augment.rot90(img, k=2),
        reference_fn=lambda img: tf.image.rot90(img, k=2))
    self._test_fn(
        images_list,
        jax_fn=lambda img: augment.rot90(img, k=3),
        reference_fn=lambda img: tf.image.rot90(img, k=3))

  # The functions below don't have a TF equivalent to compare to, we just check
  # that they run.
  @parameterized.named_parameters(("in_range", _RAND_FLOATS_IN_RANGE),
                                  ("out_of_range", _RAND_FLOATS_OUT_OF_RANGE))
  def test_flip(self, images_list):
    self._test_fn(
        images_list,
        jax_fn=augment.flip_left_right,
        reference_fn=tf.image.flip_left_right)
    self._test_fn(
        images_list,
        jax_fn=augment.flip_up_down,
        reference_fn=tf.image.flip_up_down)
    key = jax.random.PRNGKey(0)
    self._test_fn(
        images_list,
        jax_fn=functools.partial(augment.random_flip_left_right, key),
        reference_fn=None)
    self._test_fn(
        images_list,
        jax_fn=functools.partial(augment.random_flip_up_down, key),
        reference_fn=None)
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=functools.partial(augment.random_flip_left_right, key),
        reference_fn=None,
        probability=(0., 1.))
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=functools.partial(augment.random_flip_up_down, key),
        reference_fn=None,
        probability=(0., 1.))

  # Due to a bug in scipy we cannot test all available modes, refer to these
  # issues for more information: https://github.com/google/jax/issues/11097,
  # https://github.com/google/jax/issues/11097
  @parameterized.named_parameters(
      ("in_range_nearest_0", _RAND_FLOATS_IN_RANGE, "nearest", 0),
      ("in_range_nearest_1", _RAND_FLOATS_IN_RANGE, "nearest", 1),
      ("in_range_mirror_1", _RAND_FLOATS_IN_RANGE, "mirror", 1),
      ("out_of_range_nearest_0", _RAND_FLOATS_OUT_OF_RANGE, "nearest", 0),
      ("out_of_range_nearest_1", _RAND_FLOATS_OUT_OF_RANGE, "nearest", 1),
      ("out_of_range_mirror_1", _RAND_FLOATS_OUT_OF_RANGE, "mirror", 1),
  )
  def test_affine_transform(self, images_list, mode, order):
    # (ndim, ndim) no offset
    self._test_fn(
        images_list,
        jax_fn=functools.partial(
            augment.affine_transform, matrix=np.eye(3), mode=mode, order=order),
        reference_fn=functools.partial(
            scipy.ndimage.affine_transform,
            matrix=np.eye(3),
            order=order,
            mode=mode))

    # (ndim, ndim) with offset
    matrix = jnp.array([[-0.5, 0.2, 0.], [0.8, 0.5, 0.], [0., 0., 1.]])
    offset = jnp.array([40., 32., 0.])
    self._test_fn(
        images_list,
        jax_fn=functools.partial(
            augment.affine_transform,
            matrix=matrix,
            mode=mode,
            offset=offset,
            order=order),
        reference_fn=functools.partial(
            scipy.ndimage.affine_transform,
            matrix=matrix,
            offset=offset,
            order=order,
            mode=mode))

    # (ndim + 1, ndim + 1)
    matrix = jnp.array([[0.4, 0.2, 0, -10], [0.2, -0.5, 0, 5], [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    self._test_fn(
        images_list,
        jax_fn=functools.partial(
            augment.affine_transform, matrix=matrix, mode=mode, order=order),
        reference_fn=functools.partial(
            scipy.ndimage.affine_transform,
            matrix=matrix,
            order=order,
            mode=mode))

    # (ndim, ndim + 1)
    matrix = jnp.array([[0.4, 0.2, 0, -10], [0.2, -0.5, 0, 5], [0, 0, 1, 0]])
    self._test_fn(
        images_list,
        jax_fn=functools.partial(
            augment.affine_transform, matrix=matrix, mode=mode, order=order),
        reference_fn=functools.partial(
            scipy.ndimage.affine_transform,
            matrix=matrix,
            order=order,
            mode=mode))

    # (ndim,)
    matrix = jnp.array([0.4, 0.2, 1])
    self._test_fn(
        images_list,
        jax_fn=functools.partial(
            augment.affine_transform, matrix=matrix, mode=mode, order=order),
        reference_fn=functools.partial(
            scipy.ndimage.affine_transform,
            matrix=matrix,
            order=order,
            mode=mode))

  @parameterized.product(
      parameters_base=[
          ("in_range", "nearest", 0),
          ("in_range", "nearest", 1),
          ("in_range", "mirror", 1),
          ("in_range", "constant", 1),
          ("out_of_range", "nearest", 0),
          ("out_of_range", "nearest", 1),
          ("out_of_range", "mirror", 1),
          ("out_of_range", "constant", 1),
      ],
      cval=(0.0, 1.0, -2.0),
      angle=(0.0, np.pi / 4, -np.pi / 4),
  )
  def test_rotate(self, parameters_base, cval, angle):
    images_list_type, mode, order = parameters_base
    if images_list_type == "in_range":
      images_list = _RAND_FLOATS_IN_RANGE
    elif images_list_type == "out_of_range":
      images_list = _RAND_FLOATS_OUT_OF_RANGE
    else:
      raise ValueError(f"{images_list_type} not a valid image list for tests.")
    self._test_fn(
        images_list,
        jax_fn=functools.partial(
            augment.rotate, angle=angle, mode=mode, order=order, cval=cval),
        reference_fn=functools.partial(
            scipy.ndimage.rotate,
            angle=angle * 180 / np.pi,  # SciPy uses degrees.
            order=order,
            mode=mode,
            cval=cval,
            reshape=False))

  @parameterized.named_parameters(("in_range", _RAND_FLOATS_IN_RANGE),
                                  ("out_of_range", _RAND_FLOATS_OUT_OF_RANGE))
  def test_solarize(self, images_list):
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=augment.solarize,
        reference_fn=None,
        threshold=(0., 1.))

  @parameterized.named_parameters(("in_range", _RAND_FLOATS_IN_RANGE),
                                  ("out_of_range", _RAND_FLOATS_OUT_OF_RANGE))
  def test_gaussian_blur(self, images_list):
    blur_fn = functools.partial(augment.gaussian_blur, kernel_size=_KERNEL_SIZE)
    self._test_fn_with_random_arg(
        images_list, jax_fn=blur_fn, reference_fn=None, sigma=(0.1, 2.0))

  @parameterized.named_parameters(("in_range", _RAND_FLOATS_IN_RANGE),
                                  ("out_of_range", _RAND_FLOATS_OUT_OF_RANGE))
  def test_random_crop(self, images_list):
    key = jax.random.PRNGKey(43)
    crop_fn = lambda img: augment.random_crop(key, img, (100, 100, 3))
    self._test_fn(images_list, jax_fn=crop_fn, reference_fn=None)

  @parameterized.named_parameters(("in_range", _RAND_FLOATS_IN_RANGE),
                                  ("out_of_range", _RAND_FLOATS_OUT_OF_RANGE))
  def test_elastic_deformation(self, images_list):
    key = jax.random.PRNGKey(43)
    alpha = 10.
    sigma = 5.
    elastic_deformation = functools.partial(
        augment.elastic_deformation,
        key,
        alpha=alpha,
        sigma=sigma,
    )
    # Due to the difference between random number generation in numpy (in
    # reference function) and JAX's for the displacement fields we cannot test
    # this against some of the available functions. At the time of writing open
    # source options are either unmaintained or are not readily available.
    self._test_fn(
        images_list,
        jax_fn=elastic_deformation,
        reference_fn=None,
    )
    elastic_deformation = functools.partial(
        augment.elastic_deformation,
        key,
        sigma=sigma)
    # Sigma has to be constant for jit since kernel_size is derived from it.
    self._test_fn_with_random_arg(
        images_list,
        jax_fn=elastic_deformation,
        reference_fn=None,
        alpha=(40, 80))

  @parameterized.product(
      images_list=(_RAND_FLOATS_IN_RANGE, _RAND_FLOATS_OUT_OF_RANGE),
      height=(131, 111, 1, 88),
      width=(111, 105, 1, 40),
  )
  def test_center_crop(self, images_list, height, width):
    center_crop = functools.partial(
        augment.center_crop,
        height=height,
        width=width,
    )
    # Using layer as reference as other tf utility functions are not exactly
    # this same center crop:
    # - tf.image.crop_and_resize
    # - tf.image.central_crop
    reference = tf.keras.layers.CenterCrop(
        height=height,
        width=width,
    )
    self._test_fn(images_list, jax_fn=center_crop, reference_fn=reference)

  @parameterized.product(
      images_list=(_RAND_FLOATS_IN_RANGE, _RAND_FLOATS_OUT_OF_RANGE),
      target_height=(156, 131, 200, 251),
      target_width=(156, 111, 200, 251),
  )
  def test_pad_to_size(self, images_list, target_height, target_width):
    pad_fn = functools.partial(
        augment.pad_to_size,
        target_height=target_height,
        target_width=target_width,
        mode="constant",
        pad_kwargs={"constant_values": 0},
    )
    # We have to rely on `resize_with_crop_or_pad` as there are no pad to size
    # equivalents.
    reference_fn = functools.partial(
        tf.image.resize_with_crop_or_pad,
        target_height=target_height,
        target_width=target_width,
    )

    self._test_fn(images_list, jax_fn=pad_fn, reference_fn=reference_fn)

  @parameterized.product(
      images_list=(_RAND_FLOATS_IN_RANGE, _RAND_FLOATS_OUT_OF_RANGE),
      target_height=(156, 138, 200, 251),
      target_width=(156, 138, 200, 251),
  )
  def test_resize_with_crop_or_pad(
      self, images_list, target_height, target_width
  ):
    resize_crop_or_pad = functools.partial(
        augment.resize_with_crop_or_pad,
        target_height=target_height,
        target_width=target_width,
        pad_mode="constant",
        pad_kwargs={"constant_values": 0},
    )
    reference_fn = functools.partial(
        tf.image.resize_with_crop_or_pad,
        target_height=target_height,
        target_width=target_width,
    )

    self._test_fn(
        images_list, jax_fn=resize_crop_or_pad, reference_fn=reference_fn
    )


class TestMatchReference(_ImageAugmentationTest):

  def _test_fn_with_random_arg(
      self, images_list, jax_fn, reference_fn, **kw_range
  ):
    if reference_fn is None:
      return
    assert len(kw_range) == 1
    kw_name, (random_min, random_max) = list(kw_range.items())[0]
    for image_rgb in images_list:
      argument = np.random.uniform(random_min, random_max, size=())
      adjusted_jax = jax_fn(image_rgb, **{kw_name: argument})
      adjusted_reference = reference_fn(image_rgb, argument)
      if hasattr(adjusted_reference, "numpy"):
        adjusted_reference = adjusted_reference.numpy()
      self.assertAllCloseTolerant(adjusted_jax, adjusted_reference)

  def _test_fn(self, images_list, jax_fn, reference_fn):
    if reference_fn is None:
      return
    for image_rgb in images_list:
      adjusted_jax = jax_fn(image_rgb)
      adjusted_reference = reference_fn(image_rgb)
      if hasattr(adjusted_reference, "numpy"):
        adjusted_reference = adjusted_reference.numpy()
      self.assertAllCloseTolerant(adjusted_jax, adjusted_reference)


class TestVmap(_ImageAugmentationTest):

  def _test_fn_with_random_arg(self, images_list, jax_fn, reference_fn,
                               **kw_range):
    del reference_fn  # unused.
    assert len(kw_range) == 1
    kw_name, (random_min, random_max) = list(kw_range.items())[0]
    arguments = [
        np.random.uniform(random_min, random_max, size=()) for _ in images_list
    ]
    fn_vmap = jax.vmap(jax_fn)
    outputs_vmaped = list(
        fn_vmap(
            np.stack(images_list, axis=0),
            **{kw_name: np.stack(arguments, axis=0)}))
    assert len(images_list) == len(outputs_vmaped)
    assert len(images_list) == len(arguments)
    for image_rgb, argument, adjusted_vmap in zip(images_list, arguments,
                                                  outputs_vmaped):
      adjusted_jax = jax_fn(image_rgb, **{kw_name: argument})
      self.assertAllCloseTolerant(adjusted_jax, adjusted_vmap)

  def _test_fn(self, images_list, jax_fn, reference_fn):
    del reference_fn  # unused.
    fn_vmap = jax.vmap(jax_fn)
    outputs_vmaped = list(fn_vmap(np.stack(images_list, axis=0)))
    assert len(images_list) == len(outputs_vmaped)
    for image_rgb, adjusted_vmap in zip(images_list, outputs_vmaped):
      adjusted_jax = jax_fn(image_rgb)
      self.assertAllCloseTolerant(adjusted_jax, adjusted_vmap)


class TestJit(_ImageAugmentationTest):

  def _test_fn_with_random_arg(self, images_list, jax_fn, reference_fn,
                               **kw_range):
    del reference_fn  # unused.
    assert len(kw_range) == 1
    kw_name, (random_min, random_max) = list(kw_range.items())[0]
    jax_fn_jitted = jax.jit(jax_fn)
    for image_rgb in images_list:
      argument = np.random.uniform(random_min, random_max, size=())
      adjusted_jax = jax_fn(image_rgb, **{kw_name: argument})
      adjusted_jit = jax_fn_jitted(image_rgb, **{kw_name: argument})
      self.assertAllCloseTolerant(adjusted_jax, adjusted_jit)

  def _test_fn(self, images_list, jax_fn, reference_fn):
    del reference_fn  # unused.
    jax_fn_jitted = jax.jit(jax_fn)
    for image_rgb in images_list:
      adjusted_jax = jax_fn(image_rgb)
      adjusted_jit = jax_fn_jitted(image_rgb)
      self.assertAllCloseTolerant(adjusted_jax, adjusted_jit)


class TestCustom(parameterized.TestCase):
  """Tests custom logic that is not covered by reference functions."""

  @parameterized.product(
      images_list=(_RAND_FLOATS_IN_RANGE, _RAND_FLOATS_OUT_OF_RANGE),
      height=(250, 200),
      width=(250, 200),
      expected_height=(131, 131),
      expected_width=(111, 111),
  )
  def test_center_crop_size_bigger_than_original(
      self,
      images_list,
      height,
      width,
      expected_height,
      expected_width,
  ):
    output = augment.center_crop(
        image=jnp.array(images_list),
        height=height,
        width=width,
    )

    self.assertEqual(output.shape[1], expected_height)
    self.assertEqual(output.shape[2], expected_width)

  @parameterized.product(
      images_list=(_RAND_FLOATS_IN_RANGE, _RAND_FLOATS_OUT_OF_RANGE),
      target_height=(55, 84),
      target_width=(55, 84),
      expected_height=(131, 131),
      expected_width=(111, 111),
  )
  def test_pad_to_size_when_target_size_smaller_than_original(
      self,
      images_list,
      target_height,
      target_width,
      expected_height,
      expected_width,
  ):
    output = augment.pad_to_size(
        image=jnp.array(images_list),
        target_height=target_height,
        target_width=target_width,
    )

    self.assertEqual(output.shape[1], expected_height)
    self.assertEqual(output.shape[2], expected_width)


if __name__ == "__main__":
  jax.config.update("jax_default_matmul_precision", "float32")
  absltest.main()
