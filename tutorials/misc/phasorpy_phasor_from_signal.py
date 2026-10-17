# Copyright (c) PhasorPy Contributors
# SPDX-License-Identifier: MIT
# See LICENSE.txt file in the project root for details.

"""
Benchmark phasor_from_signal
============================

Benchmark the ``phasor_from_signal`` function.

The :py:func:`~phasorpy.phasor.phasor_from_signal` function used to calculate
phasor coordinates from time-resolved or spectral signals can operate in
two modes:

- using an internal Cython function optimized for calculating a small number
  of harmonics, optionally using multiple threads.

- using a real forward Fast Fourier Transform (FFT), ``numpy.fft.rfft`` or
  a drop-in replacement function like ``scipy.fft.rfft``
  or ``mkl_fft.interfaces.numpy_fft.rfft``.

This tutorial compares the performance of the two modes.

Import required modules and functions:

"""

from timeit import timeit

import numpy
from numpy.fft import rfft as numpy_fft  # noqa: F401

from phasorpy.phasor import phasor_from_signal  # noqa: F401
from phasorpy.utils import number_threads

try:
    from scipy.fft import rfft as scipy_fft
except ImportError:
    scipy_fft = None

try:
    from mkl_fft.interfaces.numpy_fft import rfft as mkl_fft
except ImportError:
    mkl_fft = None

rng = numpy.random.default_rng(42)  # initialize random number generator

# %%
# Run benchmark
# -------------
#
# Create a random signal with a size and dtype similar to real-world data:

signal = rng.random((384, 384, 384))
signal += 1.1
signal *= 3723  # ~12 bit
signal = signal.astype(numpy.uint16)  # 108 MB
signal[signal < 4282] = 0.0  # ~5% no signal

# %%
# Print execution times depending on FFT function, axis, number of harmonics,
# and number of threads:

statement = """
phasor_from_signal(signal, axis=axis, harmonic=harmonic, **kwargs)
"""
number = 1  # how many times to execute statement
ref = None  # reference duration


def print_(descr, t):
    print(f'    {descr:20s}{t / number:>6.3f}s {t / ref:>6.2f}')


for harmonic in ([1], [1, 2, 3, 4, 5, 6, 7, 8]):
    print(f'harmonics {len(harmonic)}')
    for axis in (-1, 0, 2):
        print(f'  axis {axis}')
        kwargs = {'use_fft': False, 'num_threads': 1}
        t = timeit(statement, number=number, globals=globals())
        if ref is None:
            ref = t
        print_('not_fft', t)

        num_threads = number_threads(0, 6)
        if num_threads > 1:
            kwargs = {'use_fft': False, 'num_threads': num_threads}
            t = timeit(statement, number=number, globals=globals())
            print_(f'not_fft ({num_threads} threads)', t)

        for fft_name in ('numpy_fft', 'scipy_fft', 'mkl_fft'):
            fft_func = globals()[fft_name]
            if fft_func is None:
                continue
            kwargs = {'use_fft': True, 'rfft': fft_func}
            t = timeit(statement, number=number, globals=globals())
            print_(f'{fft_name}', t)

# %%
# For reference, the results on a Core i7-14700K CPU, Windows 11,
# Python 3.15.0, numpy 2.5.4 scipy 1.18.1, mkl_fft 2.3.2::
#
#     harmonics 1
#       axis -1
#         not_fft              0.047s   1.00
#         not_fft (6 threads)  0.008s   0.18
#         numpy_fft            0.266s   5.62
#         scipy_fft            0.199s   4.21
#         mkl_fft              0.123s   2.61
#       axis 0
#         not_fft              0.232s   4.91
#         not_fft (6 threads)  0.042s   0.88
#         numpy_fft            0.684s  14.47
#         scipy_fft            0.470s   9.95
#         mkl_fft              0.174s   3.67
#       axis 2
#         not_fft              0.058s   1.23
#         not_fft (6 threads)  0.008s   0.17
#         numpy_fft            0.268s   5.67
#         scipy_fft            0.195s   4.14
#         mkl_fft              0.124s   2.63
#     harmonics 8
#       axis -1
#         not_fft              0.385s   8.15
#         not_fft (6 threads)  0.060s   1.27
#         numpy_fft            0.276s   5.84
#         scipy_fft            0.212s   4.48
#         mkl_fft              0.148s   3.14
#       axis 0
#         not_fft              1.540s  32.59
#         not_fft (6 threads)  0.363s   7.68
#         numpy_fft            0.699s  14.80
#         scipy_fft            0.474s  10.03
#         mkl_fft              0.181s   3.82
#       axis 2
#         not_fft              0.384s   8.12
#         not_fft (6 threads)  0.062s   1.30
#         numpy_fft            0.276s   5.84
#         scipy_fft            0.210s   4.44
#         mkl_fft              0.147s   3.12

# %%
# Results
# -------
#
# - Using the Cython implementation is significantly faster than using the
#   ``numpy.fft``-based implementation for single harmonics.
# - Using multiple threads can significantly speed up the Cython mode.
# - The FFT functions from ``scipy`` and ``mkl_fft`` outperform the
#   ``numpy.fft`` function. Specifically, ``mkl_fft`` is very performant.
# - Using FFT becomes more competitive when calculating a larger number of
#   harmonics.
# - Computing over the last axis is significantly faster compared to the first
#   axis. That is because the samples in the last dimension are contiguous in
#   memory.
#
# These timings are illustrative, not a controlled benchmark:
# each configuration was measured once on a single randomly generated dataset
# during a CI or documentation build.
# Results may vary with system load, hardware, and software environment.

# %%
# Conclusions
# -----------
#
# Using the Cython implementation is a reasonable default when calculating
# a few harmonics. Using FFT is a better choice when computing a large number
# of harmonics, especially with an optimized FFT function.

# sphinx_gallery_start_ignore
# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
# sphinx_gallery_end_ignore
