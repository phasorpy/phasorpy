"""
Benchmark phasor_from_signal
============================

Benchmark the ``phasor_from_signal`` function.

The :py:func:`phasorpy.phasor.phasor_from_signal` function to calculate phasor
coordinates from time-resolved or spectral signals can operate in two modes:

- using an internal Cython function optimized for calculating a small number
  of harmonics, optionally using multiple threads.

- using a real forward Fast Fourier Transform (FFT), ``numpy.fft.rfft`` or
  a drop-in replacement function like ``scipy.fft.rfft``
  or ``mkl_fft._numpy_fft.rfft``.

This tutorial compares the performance of the two modes.

Import required modules and functions:

"""

from timeit import timeit

import numpy
from numpy.fft import rfft as numpy_fft  # noqa

from phasorpy.phasor import phasor_from_signal  # noqa
from phasorpy.utils import number_threads

try:
    from scipy.fft import rfft as scipy_fft
except ImportError:
    scipy_fft = None

try:
    from mkl_fft._numpy_fft import rfft as mkl_fft
except ImportError:
    mkl_fft = None

# %%
# Run benchmark
# -------------
#
# Create a random signal with a size and dtype similar to real world data:

signal = numpy.random.default_rng(1).random((384, 384, 384))
signal += 1.1
signal *= 3723  # ~12 bit
signal = signal.astype(numpy.uint16)  # 108 MB
signal[signal < 0.05] = 0.0  # 5% no signal

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
# For reference, the results on a Core i7-14700K CPU, Windows 11::
#
#     harmonics 1
#       axis -1
#         not_fft              0.036s   1.00
#         not_fft (6 threads)  0.008s   0.21
#         numpy_fft            0.285s   7.89
#         scipy_fft            0.247s   6.84
#         mkl_fft              0.124s   3.43
#       axis 0
#         not_fft              0.156s   4.32
#         not_fft (6 threads)  0.041s   1.14
#         numpy_fft            0.743s  20.60
#         scipy_fft            0.583s  16.16
#         mkl_fft              0.182s   5.03
#       axis 2
#         not_fft              0.037s   1.02
#         not_fft (6 threads)  0.009s   0.25
#         numpy_fft            0.282s   7.81
#         scipy_fft            0.244s   6.78
#         mkl_fft              0.125s   3.47
#     harmonics 8
#       axis -1
#         not_fft              0.275s   7.62
#         not_fft (6 threads)  0.041s   1.13
#         numpy_fft            0.295s   8.18
#         scipy_fft            0.267s   7.39
#         mkl_fft              0.145s   4.02
#       axis 0
#         not_fft              1.250s  34.66
#         not_fft (6 threads)  0.325s   9.01
#         numpy_fft            0.732s  20.28
#         scipy_fft            0.546s  15.13
#         mkl_fft              0.168s   4.67
#       axis 2
#         not_fft              0.278s   7.72
#         not_fft (6 threads)  0.040s   1.11
#         numpy_fft            0.298s   8.25
#         scipy_fft            0.270s   7.48
#         mkl_fft              0.143s   3.98

# %%
# Results
# -------
#
# - Using the Cython implementation is significantly faster than using the
#   ``numpy.fft`` based implementation for single harmonics.
# - Using multiple threads can significantly speed up the Cython mode.
# - The FFT functions from ``scipy`` and ``mkl_fft`` outperform numpy.fft.
#   Specifically, ``mkl_fft`` is very performant.
# - Using FFT becomes more competitive when calculating larger number of
#   harmonics.
# - Computing over the last axis is significantly faster compared to the first
#   axis. That is because the samples in the last dimension are contiguous,
#   closer together in memory.
#
# Note that these results were obtained on a single dataset of random numbers.

# %%
# Conclusions
# -----------
#
# Using the Cython implementation is a reasonable default when calculating
# a few harmonics. Using FFT is a better choice when computing large number
# of harmonics, especially with an optimized FFT function.

# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
