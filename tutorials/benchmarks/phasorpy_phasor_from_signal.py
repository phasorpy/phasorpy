"""
Benchmark phasor_from_signal
============================

Benchmark the ``phasor_from_signal`` function.

The :doc:`/api/phasor` module provides two functions to calculate phasor
coordinates from time-resolved or spectral signals:

- :py:func:`phasorpy.phasor.phasor_from_signal`,
  implemented mostly in Cython for efficiency.

- :py:func:`phasorpy.phasor.phasor_from_signal_fft`,
  a pure Python reference implementation based on ``numpy.fft.fft``.

This tutorial compares the performance of the two implementations.

"""

from timeit import timeit

import numpy

from phasorpy.phasor import phasor_from_signal, phasor_from_signal_fft

try:
    from scipy.fft import fft as scipy_fft
except ImportError:
    scipy_fft = None

try:
    from mkl_fft._numpy_fft import fft as mkl_fft
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

        kwargs = dict(num_threads=8)
        t = timeit(statement, number=number, globals=globals())
        print_('  threads 8', t)

        func = phasor_from_signal_fft  # type: ignore
        kwargs = {}
        t = timeit(statement, number=number, globals=globals())
        print_(func.__name__, t)

        for fft_name in ('scipy_fft', 'mkl_fft'):
            fft_func = globals()[fft_name]
            if fft_func is None:
                continue
            kwargs = dict(fft_func=fft_func)
            t = timeit(statement, number=number, globals=globals())
            print_(f'  {fft_name}', t)

# %%
# Results
# -------
#
# - The ``phasor_from_signal`` implementation is an order of magnitude faster
#   than the ``numpy.fft`` based reference implementation for single harmonics.
# - The FFT functions from ``scipy`` and ``mkl_fft`` usually outperform
#   numpy.fft.
# - Using FFT becomes more competitive when calculating larger number of
#   harmonics.
# - ``mkl_fft`` is very performant for first and last axes.
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
