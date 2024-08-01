"""
Benchmark phasor_from_signal
============================

Benchmark the ``phasor_from_signal`` function.

The :doc:`/api/phasor` module provides two functions to calculate phasor
coordinates from time-resolved or spectral signals:

- :py:func:`phasorpy.phasor.phasor_from_signal`,
  implemented mostly in Cython for efficiency.

- :py:func:`phasorpy.phasor.phasor_from_signal_fft`,
  a pure Python reference implementation based on ``numpy.fft.rfft``.

This tutorial compares the performance of the two implementations.

"""

from timeit import timeit

import numpy

from phasorpy.phasor import phasor_from_signal, phasor_from_signal_fft
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

# %%
# Print execution times depending on function, axis, number of harmonics,
# and number of threads:


statement = 'func(signal, axis=axis, harmonic=harmonic, **kwargs)'
number = 1  # how many times to execute statement
ref = None  # reference duration


def print_(descr, t):
    print(f'    {descr:24s}{t / number:>6.3f}s {t / ref:>6.2f}')


for harmonic in ([1], [1, 2, 3, 4, 5, 6, 7, 8]):
    print(f'harmonics {len(harmonic)}')
    for axis in (-1, 0, 2):
        print(f'  axis {axis}')
        func = phasor_from_signal  # type: ignore
        kwargs = {}
        t = timeit(statement, number=number, globals=globals())
        if ref is None:
            ref = t
        print_(func.__name__, t)

        num_threads = number_threads(0, 6)
        if num_threads > 1:
            kwargs = {'num_threads': num_threads}
            t = timeit(statement, number=number, globals=globals())
            print_(f'  threads {num_threads}', t)

        func = phasor_from_signal_fft  # type: ignore
        kwargs = {}
        t = timeit(statement, number=number, globals=globals())
        print_(func.__name__, t)

        for fft_name in ('scipy_fft', 'mkl_fft'):
            fft_func = globals()[fft_name]
            if fft_func is None:
                continue
            kwargs = {'rfft_func': fft_func}
            t = timeit(statement, number=number, globals=globals())
            print_(f'  {fft_name}', t)

# %%
# For reference, the results on a Core i7-14700K CPU::
#
#      harmonics 1
#        axis -1
#          phasor_from_signal       0.035s   1.00
#            threads 6              0.008s   0.24
#          phasor_from_signal_fft   0.270s   7.72
#            scipy_fft              0.236s   6.74
#            mkl_fft                0.128s   3.67
#        axis 0
#          phasor_from_signal       0.158s   4.50
#            threads 6              0.036s   1.02
#          phasor_from_signal_fft   0.724s  20.66
#            scipy_fft              0.516s  14.72
#            mkl_fft                0.177s   5.04
#        axis 2
#          phasor_from_signal       0.035s   1.01
#            threads 6              0.007s   0.19
#          phasor_from_signal_fft   0.273s   7.79
#            scipy_fft              0.243s   6.93
#            mkl_fft                0.131s   3.73
#      harmonics 8
#        axis -1
#          phasor_from_signal       0.268s   7.65
#            threads 6              0.040s   1.15
#          phasor_from_signal_fft   0.289s   8.26
#            scipy_fft              0.252s   7.19
#            mkl_fft                0.148s   4.21
#        axis 0
#          phasor_from_signal       1.260s  35.96
#            threads 6              0.308s   8.78
#          phasor_from_signal_fft   0.718s  20.49
#            scipy_fft              0.528s  15.06
#            mkl_fft                0.172s   4.91
#        axis 2
#          phasor_from_signal       0.274s   7.82
#            threads 6              0.041s   1.18
#          phasor_from_signal_fft   0.283s   8.06
#            scipy_fft              0.253s   7.21
#            mkl_fft                0.143s   4.08

# %%
# Results
# -------
#
# - The ``phasor_from_signal`` implementation is significantly faster than
#   the ``numpy.fft`` based reference implementation for single harmonics.
# - The FFT functions from ``scipy`` and ``mkl_fft`` outperform numpy.fft.
#   ``mkl_fft`` is very performant.
# - Using FFT becomes more competitive when calculating larger number of
#   harmonics.
# - Computing over the last axis is significantly faster compared to the first
#   axis. That is because the samples in the last dimension are contiguous,
#   closer together in memory.
# - Using multiple threads can significantly speed up the calculation.
#
# Note that these results were obtained on a single dataset of random numbers.

# %%
# Conclusions
# -----------
#
# The ``phasor_from_signal`` implementation is a reasonable default.
# The ``phasor_from_signal_fft`` function may be a better choice, for example,
# when computing large number of harmonics with an optimized FFT function.
