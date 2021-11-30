# PIX

PIX is an image processing library in [JAX], for [JAX].

[![GitHub Workflow CI](https://img.shields.io/github/workflow/status/deepmind/dm_pix/ci?label=pytest&logo=python&logoColor=white&style=flat-square)](https://github.com/deepmind/dm_pix/actions/workflows/ci.yml)
[![Read the Docs](https://img.shields.io/readthedocs/dm_pix?label=ReadTheDocs&logo=readthedocs&logoColor=white&style=flat-square)](https://dm-pix.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/dm_pix?logo=pypi&logoColor=white&style=flat-square)](https://pypi.org/project/dm-pix/)

## Overview

[JAX] is a library resulting from the union of [Autograd] and [XLA] for
high-performance machine learning research. It provides [NumPy], [SciPy],
automatic differentiation and first-class GPU/TPU support.

PIX is a library built on top of JAX with the goal of providing image processing
functions and tools to JAX in a way that they can be optimised and parallelised
through [`jax.jit`][jit], [`jax.vmap`][vmap] and [`jax.pmap`][pmap].

## Installation

PIX is written in pure Python, but depends on C++ code via JAX.

Because JAX installation is different depending on your CUDA version, PIX does
not list JAX as a dependency in [`requirements.txt`], although it is technically
listed for reference, but commented.

First, follow [JAX installation instructions] to install JAX with the relevant
accelerator support.

Then, install PIX using `pip`:

```bash
$ pip install dm-pix
```

## Quickstart

To use `PIX`, you just need to `import dm_pix as pix` and use it right away!

For example, let's assume to have loaded the JAX logo (available in
`examples/assets/jax_logo.jpg`) in a variable called `image` and we want to flip
it left to right.

![JAX logo]

All it's needed is the following code!

```python
import dm_pix as pix

# Load an image into a NumPy array with your preferred library.
image = load_image()

flip_left_right_image = pix.flip_left_right(image)
```

And here is the result!

![JAX logo left-right]

All the functions in PIX can be [`jax.jit`][jit]ed, [`jax.vmap`][vmap]ed and
[`jax.pmap`][pmap]ed, so all the following functions can take advantage of
optimization and parallelization.

```python
import dm_pix as pix
import jax

# Load an image into a NumPy array with your preferred library.
image = load_image()

# Vanilla Python function.
flip_left_right_image = pix.flip_left_right(image)

# `jax.jit`ed function.
flip_left_right_image = jax.jit(pix.flip_left_right)(image)

# Assuming to have a single device, like a CPU or a single GPU, we add a
# single leading dimension for using `image` with the parallelized or
# the multi-device parallelization version of `pix.flip_left_right`.
# To know more, please refer to JAX documentation of `jax.vmap` and `jax.pmap`.
image = image[np.newaxis, ...]

# `jax.vmap`ed function.
flip_left_right_image = jax.vmap(pix.flip_left_right)(image)

# `jax.pmap`ed function.
flip_left_right_image = jax.pmap(pix.flip_left_right)(image)
```

You can check it yourself that the result from the four versions of
`pix.flip_left_right` is the same (up to the accelerator floating point
accuracy)!

## Examples

We have a few examples in the [`examples/`] folder. They are not much
more involved then the previous example, but they may be a good starting point
for you!

## Testing

We provide a suite of tests to help you both testing your development
environment and to know more about the library itself! All test files have
`_test` suffix, and can be executed using `pytest`.

If you already have PIX installed, you just need to install some extra
dependencies and run `pytest` as follows:

```bash
$ pip install -r requirements_tests.txt
$ python -m pytest [-n <NUMCPUS>] dm_pix
```

If you want an isolated virtual environment, you just need to run our utility
`bash` script as follows:

```bash
$ ./test.sh
```

## Citing PIX

This repository is part of the [DeepMind JAX Ecosystem], to cite PIX please use
the [DeepMind JAX Ecosystem citation].

## Contribute!

We are very happy to accept contributions!

Please read our [contributing guidelines](./CONTRIBUTING.md) and send us PRs!

[Autograd]: https://github.com/hips/autograd "Autograd on GitHub"
[DeepMind JAX Ecosystem]: https://deepmind.com/blog/article/using-jax-to-accelerate-our-research "DeepMind JAX Ecosystem"
[DeepMind JAX Ecosystem citation]: https://github.com/deepmind/jax/blob/main/deepmind2020jax.txt "Citation"
[JAX]: https://github.com/google/jax "JAX on GitHub"
[JAX installation instructions]: https://github.com/google/jax#installation "JAX installation"
[jit]: https://jax.readthedocs.io/en/latest/jax.html#jax.jit "jax.jit documentation"
[NumPy]: https://numpy.org/ "NumPy"
[pmap]: https://jax.readthedocs.io/en/latest/jax.html#jax.pmap "jax.pmap documentation"
[SciPy]: https://www.scipy.org/ "SciPy"
[XLA]: https://www.tensorflow.org/xla "XLA"
[vmap]: https://jax.readthedocs.io/en/latest/jax.html#jax.vmap "jax.vmap documentation"

[`examples/`]: ./examples/
[JAX logo]: ./examples/assets/jax_logo.jpg
[JAX logo left-right]: ./examples/assets/flip_left_right_jax_logo.jpg
[`requirements.txt`]: ./requirements.txt
