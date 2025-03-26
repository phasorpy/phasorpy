"""
Filter phasor coordinates
=========================

Functions for filtering phasor coordinates.

Filtering phasor coordinates improves signal quality by reducing noise while
preserving relevant features. Two of the most common methods for filtering
include the median and wavelet filtering.

"""

# %%
# Import required modules and functions:

from phasorpy.datasets import fetch
from phasorpy.io import signal_from_imspector_tiff
from phasorpy.phasor import (
    phasor_calibrate,
    phasor_filter_median,
    phasor_filter_pawflim,
    phasor_from_signal,
    phasor_threshold,
)
from phasorpy.plot import plot_image, plot_phasor

# %%
# Get calibrated phasor coordinates
# ---------------------------------
#
# Read a time-correlated single photon counting (TCSPC) histogram from a file.
# A homogeneous solution of Fluorescein (4.2 ns) was imaged as a reference:

signal = signal_from_imspector_tiff(fetch('Embryo.tif'))
frequency = signal.attrs['frequency']

reference_signal = signal_from_imspector_tiff(fetch('Fluorescein_Embryo.tif'))
assert reference_signal.attrs['frequency'] == frequency

# %%
# Calculate and calibrate phasor coordinates:

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
# Unfiltered
# ----------
#
# Plot the unfiltered, calibrated phasor coordinates after applying a
# threshold based on the mean intensity to remove background values:

plot_phasor(
    *phasor_threshold(mean, real, imag, mean_min=1)[1:],
    frequency=frequency,
    title='Unfiltered phasor coordinates',
)

# %%
# Median filter
# -------------
#
# Median filtering replaces each pixel value with the median of its
# neighboring values, reducing noise while preserving edges.
# The function :py:func:`phasorpy.phasor.phasor_filter_median` applies a
# median filter to phasor coordinates. Typically, applying a 3×3 kernel
# one to three times is sufficient to remove noise while maintaining
# important features:

mean_filtered, real_filtered, imag_filtered = phasor_filter_median(
    mean, real, imag, repeat=3, size=3
)

# %%
# Thresholds should be applied after filtering:

mean_filtered, real_filtered, imag_filtered = phasor_threshold(
    mean_filtered, real_filtered, imag_filtered, mean_min=1
)

# %%
# Plot the median-filtered and thresholded phasor coordinates:

plot_phasor(
    real_filtered,
    imag_filtered,
    frequency=frequency,
    title='Median-filtered phasor coordinates (3x3 kernel, 3 repetitions)',
)

# %%
# Increasing the number of repetitions or the filter kernel size can further
# reduce noise, but may also remove relevant features:

mean_filtered, real_filtered, imag_filtered = phasor_threshold(
    *phasor_filter_median(mean, real, imag, repeat=6, size=5), mean_min=1
)

plot_phasor(
    real_filtered,
    imag_filtered,
    frequency=frequency,
    title='Median-filtered phasor coordinates (5x5 kernel, 6 repetitions)',
)

# %%
# The smoothing effect of median-filtering is demonstrated by plotting the
# real components of the filtered and unfiltered phasor coordinates as images:

plot_image(
    phasor_threshold(mean, real, imag, mean_min=1)[1],
    real_filtered,
    vmin=0.4,
    vmax=0.9,
    title='Real component of phasor coordinates',
    labels=['Unfiltered', 'Median-filtered'],
)

# %%
# pawFLIM wavelet filter
# ----------------------
#
# Filtering based on wavelet decomposition is another method to reduce noise.
# The function :py:func:`phasorpy.phasor.phasor_filter_pawflim` is based
# on the `pawFLIM <https://github.com/maurosilber/pawflim>`_ library.
# While the median filter is applicable to any type of phasor coordinates,
# the pawFLIM filter requires calibrated phasor coordinates from FLIM
# measurements and at least one harmonic and its corresponding double:

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
# Apply the pawFLIM wavelet filter to the calibrated phasor coordinates:

mean_filtered, real_filtered, imag_filtered = phasor_threshold(
    *phasor_filter_pawflim(mean, real, imag, harmonic=harmonic), mean_min=1
)

# %%
# Plot the pawFLIM-filtered and thresholded phasor coordinates:

plot_phasor(
    real_filtered[0],
    imag_filtered[0],
    frequency=frequency,
    title='pawFLIM-filtered phasor coordinates (sigma=2, levels=1)',
)

# %%
# Increasing the significance level of the comparison between phasor
# coordinates or the maximum averaging area can further reduce noise:

mean_filtered, real_filtered, imag_filtered = phasor_filter_pawflim(
    mean, real, imag, harmonic=harmonic, sigma=5, levels=3
)

mean_filtered, real_filtered, imag_filtered = phasor_threshold(
    mean_filtered, real_filtered, imag_filtered, 1
)

plot_phasor(
    real_filtered[0],
    imag_filtered[0],
    frequency=frequency,
    title='pawFLIM-filtered phasor coordinates (sigma=5, levels=3)',
)

# %%
# Plot the real components of the filtered and unfiltered phasor coordinates
# as images:

plot_image(
    phasor_threshold(mean, real, imag, mean_min=1)[1],
    real_filtered,
    vmin=0.4,
    vmax=0.9,
    title='Real component of phasor coordinates',
    labels=['Unfiltered', 'pawFLIM-filtered'],
)

# %%
# sphinx_gallery_thumbnail_number = -2
# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
