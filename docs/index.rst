:github_url: https://github.com/deepmind/dm_pix/tree/master/docs

===
PIX
===

PIX is an image processing library in JAX, for JAX.

Overview
========

JAX is a library resulting from the union of Autograd and XLA for
high-performance machine learning research. It provides NumPy, SciPy,
automatic differentiation and first-class GPU/TPU support.

PIX is a library built on top of JAX with the goal of providing image processing
functions and tools to JAX in a way that they can be optimised and parallelised
through `jax.jit`, `jax.vmap` and `jax.pmap`.

Installation
============

PIX is written in pure Python, but depends on C++ code via JAX.

Because JAX installation is different depending on your CUDA version, PIX does
not list JAX as a dependency in `requirements.txt`, although it is technically
listed for reference, but commented.

First, follow JAX installation instructions to install JAX with the relevant
accelerator support.

Then, install PIX using ``pip``:

.. code-block:: bash

   pip install dm-pix

.. toctree::
   :caption: API Documentation
   :maxdepth: 2

   api


Contribute
==========

- `Issue tracker <https://github.com/deepmind/dm_pix/issues>`_
- `Source code <https://github.com/deepmind/dm_pix/tree/master>`_

Support
=======

If you are having issues, please let us know by filing an issue on our
`issue tracker <https://github.com/deepmind/dm_pix/issues>`_.

License
=======

PIX is licensed under the Apache 2.0 License.


Indices and Tables
==================

* :ref:`genindex`
