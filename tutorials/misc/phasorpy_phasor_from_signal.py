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
  or ``mkl_fft.interfaces.numpy_fft.rfft``.

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
    from mkl_fft.interfaces.numpy_fft import rfft as mkl_fft
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
# For reference, the results on a Core i7-14700K CPU, Windows 11,
# Python 3.14.0, numpy 2.3.5, scipy 1.16.3, mkl-fft 2.1.1::
#
#     harmonics 1
#       axis -1
#         not_fft              0.034s   1.00
#         not_fft (6 threads)  0.006s   0.17
#         numpy_fft            0.274s   8.03
#         scipy_fft            0.240s   7.04
#         mkl_fft              0.141s   4.14
#       axis 0
#         not_fft              0.162s   4.75
#         not_fft (6 threads)  0.038s   1.13
#         numpy_fft            0.697s  20.44
#         scipy_fft            0.496s  14.54
#         mkl_fft              0.167s   4.90
#       axis 2
#         not_fft              0.038s   1.12
#         not_fft (6 threads)  0.006s   0.16
#         numpy_fft            0.272s   7.99
#         scipy_fft            0.240s   7.04
#         mkl_fft              0.130s   3.83
#     harmonics 8
#       axis -1
#         not_fft              0.287s   8.43
#         not_fft (6 threads)  0.040s   1.17
#         numpy_fft            0.288s   8.45
#         scipy_fft            0.253s   7.43
#         mkl_fft              0.161s   4.74
#       axis 0
#         not_fft              1.161s  34.04
#         not_fft (6 threads)  0.425s  12.47
#         numpy_fft            0.715s  20.97
#         scipy_fft            0.541s  15.88
#         mkl_fft              0.183s   5.38
#       axis 2
#         not_fft              0.282s   8.27
#         not_fft (6 threads)  0.038s   1.13
#         numpy_fft            0.288s   8.46
#         scipy_fft            0.255s   7.49
#         mkl_fft              0.155s   4.56

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
