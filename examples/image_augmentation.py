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

"""Examples of image augmentations with Pix."""

from absl import app
import dm_pix as pix
import jax.numpy as jnp
import numpy as np
import PIL.Image as pil

_KERNEL_SIGMA = 5
_KERNEL_SIZE = 5
_MAGIC_VALUE = 0.42


def main(_) -> None:
  # Load an image.
  image = _get_image()

  # Flip up-down the image and visual it.
  flip_up_down_image = pix.flip_up_down(image=image)
  _imshow(flip_up_down_image)

  # Apply a Gaussian filter to the image and visual it.
  gaussian_blur_image = pix.gaussian_blur(
      image=image,
      sigma=_KERNEL_SIGMA,
      kernel_size=_KERNEL_SIZE,
  )
  _imshow(gaussian_blur_image)

  # Change image brightness and visual it.
  adjust_brightness_image = pix.adjust_brightness(
      image=image,
      delta=_MAGIC_VALUE,
  )
  _imshow(adjust_brightness_image)

  # Change image contrast and visual it.
  adjust_contrast_image = pix.adjust_contrast(
      image=image,
      factor=_MAGIC_VALUE,
  )
  _imshow(adjust_contrast_image)

  # Change image gamma and visual it.
  adjust_gamma_image = pix.adjust_gamma(
      image=image,
      gamma=_MAGIC_VALUE,
  )
  _imshow(adjust_gamma_image)

  # Change image hue and visual it.
  adjust_hue_image = pix.adjust_hue(
      image=image,
      delta=_MAGIC_VALUE,
  )
  _imshow(adjust_hue_image)


def _get_image():
  return jnp.array(pil.open("./assets/jax_logo.jpg"), dtype=jnp.float32) / 255.


def _imshow(image: jnp.ndarray) -> None:
  """Showes the input image using PIL/Pillow backend."""
  image = pil.fromarray(np.asarray(image * 255.).astype(np.uint8), "RGB")
  image.show()


if __name__ == "__main__":
  app.run(main)
