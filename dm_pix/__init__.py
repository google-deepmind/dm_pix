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
"""PIX public APIs."""

from dm_pix._src import augment
from dm_pix._src import color_conversion
from dm_pix._src import depth_and_space
from dm_pix._src import interpolation
from dm_pix._src import metrics
from dm_pix._src import patch

__version__ = "0.4.2"

# Augmentations.
adjust_brightness = augment.adjust_brightness
adjust_contrast = augment.adjust_contrast
adjust_gamma = augment.adjust_gamma
adjust_hue = augment.adjust_hue
adjust_saturation = augment.adjust_saturation
affine_transform = augment.affine_transform
center_crop = augment.center_crop
elastic_deformation = augment.elastic_deformation
flip_left_right = augment.flip_left_right
flip_up_down = augment.flip_up_down
gaussian_blur = augment.gaussian_blur
pad_to_size = augment.pad_to_size
random_brightness = augment.random_brightness
random_contrast = augment.random_contrast
random_crop = augment.random_crop
random_flip_left_right = augment.random_flip_left_right
random_flip_up_down = augment.random_flip_up_down
random_gamma = augment.random_gamma
random_hue = augment.random_hue
random_saturation = augment.random_saturation
resize_with_crop_or_pad = augment.resize_with_crop_or_pad
rotate = augment.rotate
rot90 = augment.rot90
solarize = augment.solarize

# Color conversions.
hsl_to_rgb = color_conversion.hsl_to_rgb
hsv_to_rgb = color_conversion.hsv_to_rgb
rgb_to_hsl = color_conversion.rgb_to_hsl
rgb_to_hsv = color_conversion.rgb_to_hsv
rgb_to_grayscale = color_conversion.rgb_to_grayscale

# Depth and space transformations.
depth_to_space = depth_and_space.depth_to_space
space_to_depth = depth_and_space.space_to_depth

# Interpolation functions.
flat_nd_linear_interpolate = interpolation.flat_nd_linear_interpolate
flat_nd_linear_interpolate_constant = (
    interpolation.flat_nd_linear_interpolate_constant)

# Metrics.
mae = metrics.mae
mse = metrics.mse
psnr = metrics.psnr
rmse = metrics.rmse
simse = metrics.simse
ssim = metrics.ssim

# Patch extraction functions.
extract_patches = patch.extract_patches

del augment, color_conversion, depth_and_space, interpolation, metrics, patch

__all__ = (
    "adjust_brightness",
    "adjust_contrast",
    "adjust_gamma",
    "adjust_hue",
    "adjust_saturation",
    "affine_transform",
    "center_crop",
    "depth_to_space",
    "elastic_deformation",
    "extract_patches",
    "flat_nd_linear_interpolate",
    "flat_nd_linear_interpolate_constant",
    "flip_left_right",
    "flip_up_down",
    "gaussian_blur",
    "hsl_to_rgb",
    "hsv_to_rgb",
    "mae",
    "mse",
    "pad_to_size",
    "psnr",
    "random_brightness",
    "random_contrast",
    "random_crop",
    "random_flip_left_right",
    "random_flip_up_down",
    "random_gamma",
    "random_hue",
    "random_saturation",
    "resize_with_crop_or_pad",
    "rotate",
    "rgb_to_hsl",
    "rgb_to_hsv",
    "rgb_to_grayscale",
    "rmse",
    "rot90",
    "simse",
    "ssim",
    "solarize",
    "space_to_depth",
)

#  _________________________________________
# / Please don't use symbols in `_src` they \
# \ are not part of the PIX public API.     /
#  -----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
try:
  del _src  # pylint: disable=undefined-variable
except NameError:
  pass
