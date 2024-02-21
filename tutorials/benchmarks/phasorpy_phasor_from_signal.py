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
# Print execution times depending on function, axis, and number of threads:

for num_threads in (1, 8):
    print(f'nthreads={num_threads}')
    for axis in (0, 1, 2):
        print(f'  axis={axis}')
        for func in (phasor_from_signal, phasor_from_signal_fft):
            print(f'    {func.__name__}')
            print(
                '      ',
                timeit(
                    'func(signal, axis=axis, num_threads=num_threads)',
                    number=1,
                    globals=globals(),
                ),
            )

# %%
# Results
# -------
#
# - The ``phasor_from_signal`` implementation is an order of magnitude faster
#   than the reference implementation using FFT.
# - Computing over the last axis is significantly faster compared to the first
#   axis. That is because the samples in the last dimension are contiguous,
#   closer together in memory.
# - Using multiple threads can significantly speed up the calculation.
# - The ``phasor_from_signal_fft`` function does not support multi-threading.
