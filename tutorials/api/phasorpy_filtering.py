"""
Filter phasor coordinates
=========================

Methods for filtering phasor coordinates.

Filtering phasor coordinates improves signal quality by reducing noise while
preserving relevant features. Two of the most common methods for filtering
include the median and wavelet filtering.

"""

# %%
# Import required modules and functions:

import matplotlib.pyplot as plt

from phasorpy.datasets import fetch
from phasorpy.io import signal_from_imspector_tiff
from phasorpy.phasor import (
    phasor_calibrate,
    phasor_filter_median,
    phasor_filter_pawflim,
    phasor_from_signal,
    phasor_threshold,
)
from phasorpy.plot import PhasorPlot, plot_phasor_image, plot_signal_image

# %%
# Read signal and reference signal from files
# -------------------------------------------
#
# Read a time-correlated single photon counting (TCSPC) histogram, acquired
# at 80.11 MHz, from a file. A homogeneous solution of Fluorescein was imaged
# as a reference for calibration:

signal = signal_from_imspector_tiff(fetch('Embryo.tif'))
frequency = signal.attrs['frequency']

reference_signal = signal_from_imspector_tiff(fetch('Fluorescein_Embryo.tif'))
assert reference_signal.attrs['frequency'] == frequency

# %%
# Plot the signal image:

plot_signal_image(signal, title='Signal image')

# %%
# Calculate and calibrate phasor coordinates
# ------------------------------------------
#
# Phasor coordinates for the signal and reference signal are calculated and
# calibrated with the reference signal of known lifetime (4.2 ns):

mean, real, imag = phasor_from_signal(signal, axis=0)
reference_mean, reference_real, reference_imag = phasor_from_signal(
    reference_signal, axis=0
)

real, imag = phasor_calibrate(
    real,
    imag,
    reference_mean,
    reference_real,
    reference_imag,
    frequency=frequency,
    lifetime=4.2,
)

# %%
# Plot unfiltered phasor coordinates
# ----------------------------------
#
# Plot the calibrated and unfiltered phasor coordinates after applying a
# threshold based on the mean intensity to remove background values:

plot = PhasorPlot(frequency=frequency, title='Unfiltered phasor coordinates')
plot.hist2d(*phasor_threshold(mean, real, imag, 1)[1:], bins=300, cmap='turbo')

# %%
# Plot the unfiltered phasor images:

plot_phasor_image(
    *phasor_threshold(mean, real, imag, 1), title='Unfiltered phasor image'
)

# %%
# Median filtering
# ----------------
#
# Median filtering replaces each pixel value with the median of its
# neighboring values, reducing noise while preserving edges. The function
# :py:func:`phasorpy.phasor.phasor_filter_median` applies a median filter to
# phasor coordinates. Typically, using a 3×3 kernel applied 1 to 3 times is
# sufficient to remove noise while maintaining important features.

mean_filtered, real_filtered, imag_filtered = phasor_filter_median(
    mean, real, imag, repeat=3, size=3
)

# %%
# When filtering phasor coordinates, all thresholds should be applied after
# filtering:
mean_filtered, real_filtered, imag_filtered = phasor_threshold(
    mean_filtered, real_filtered, imag_filtered, 1
)

# %%
# Plot the median filtered and thresholded phasor coordinates:
plot = PhasorPlot(
    frequency=frequency, title='Median filtered phasor coordinates'
)
plot.hist2d(real_filtered, imag_filtered, bins=300, cmap='turbo')

# %%
# The smoothing of phasor coordinates can also be visualized by plotting the
# filtered phasor image and comparing it with the original:

plot_phasor_image(
    mean_filtered,
    real_filtered,
    imag_filtered,
    title='Median filtered phasor image',
)

# %%
# Increasing the number of repetitions or the kernel size can further reduce
# noise, but may also remove relevant features:
mean_filtered, real_filtered, imag_filtered = phasor_filter_median(
    mean, real, imag, repeat=6, size=5
)

mean_filtered, real_filtered, imag_filtered = phasor_threshold(
    mean_filtered, real_filtered, imag_filtered, 1
)

plot = PhasorPlot(
    frequency=frequency, title='Median filtered phasor coordinates'
)
plot.hist2d(real_filtered, imag_filtered, bins=300, cmap='turbo')

# %%
# Wavelet filtering
# -----------------
#
# Wavelet filtering is another method that can be applied to reduce noise based
# on the wavelet decomposition. The function
# :py:func:`phasorpy.phasor.phasor_filter_pawflim` can be used to apply wavelet
# filtering to phasor coordinates. This method is based on the
# `pawFLIM <https://github.com/maurosilber/pawflim>`_ library. This
# implementation requires the information of at least one harmonic and it's
# corresponding double:

harmonic = [1, 2]

mean, real, imag = phasor_from_signal(signal, axis=0, harmonic=harmonic)
reference_mean, reference_real, reference_imag = phasor_from_signal(
    reference_signal, axis=0, harmonic=harmonic
)

real, imag = phasor_calibrate(
    real,
    imag,
    reference_mean,
    reference_real,
    reference_imag,
    frequency=frequency,
    lifetime=4.2,
    harmonic=harmonic,
)

# %%
# Apply wavelet filtering to phasor coordinates and plot the results:

mean_filtered, real_filtered, imag_filtered = phasor_filter_pawflim(
    mean, real, imag, harmonic=harmonic
)
mean_filtered, real_filtered, imag_filtered = phasor_threshold(
    mean_filtered, real_filtered, imag_filtered, 1
)

plot = PhasorPlot(
    frequency=frequency,
    title='Wavelet filtered phasor coordinates for first harmonic',
)
plot.hist2d(real_filtered[0], imag_filtered[0], bins=300, cmap='turbo')

# %%
# Increasing the significance level of the comparison between phasor
# coordinates and/or the maximum averaging area can further reduce noise:

mean_filtered, real_filtered, imag_filtered = phasor_filter_pawflim(
    mean, real, imag, harmonic=harmonic, sigma=5, levels=3
)
mean_filtered, real_filtered, imag_filtered = phasor_threshold(
    mean_filtered, real_filtered, imag_filtered, 1
)

plot = PhasorPlot(
    frequency=frequency,
    title='Wavelet filtered phasor coordinates for first harmonic',
)
plot.hist2d(real_filtered[0], imag_filtered[0], bins=300, cmap='turbo')

# %%
# sphinx_gallery_thumbnail_number = -1
# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
