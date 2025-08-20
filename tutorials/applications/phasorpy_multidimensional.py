"""
Multidimensional phasor approach
================================

Simultaneous phasor analysis of lifetime and spectral data.

Multidimensional phasor analysis enables the correlation and classification
of pixels based on multiple fluorescence characteristics simultaneously.
This approach combines phasor coordinates from different measurement domains
to provide enhanced discrimination and identification of molecular species
or cellular components that may appear similar in single-domain analysis.

Phasor coordinates in one dimension can be mapped to other dimensions by masks
(or cursors) that define regions of interest in the phasor space. The dimension
where cursors are defined serves as the "main" dimension, controlling the
classification that is then applied to correlate with other dimensions.

This analysis method is demonstrated using a dataset combining fluorescence
lifetime imaging microscopy (FLIM) and hyperspectral imaging (HSI) of LAURDAN
fluorescence as presented in:

  Malacrida L, Jameson D, and Gratton E.
  A multidimensional phasor approach reveals LAURDAN photophysics in NIH-3T3
  cell membranes.
  *Sci Rep*, 7: 9215 (2017).

The dataset is available at https://zenodo.org/records/16894639.

The multidimensional analysis correlates FLIM phasor coordinates from two
detection channels with spectral phasor coordinates, enabling enhanced
classification of cellular regions based on both lifetime and spectral
properties of LAURDAN fluorescence.

"""

# %%
# Import required modules, functions, and classes:

import numpy

from phasorpy.color import CATEGORICAL
from phasorpy.cursor import mask_from_circular_cursor, pseudo_color
from phasorpy.datasets import fetch
from phasorpy.io import signal_from_fbd, signal_from_lsm
from phasorpy.lifetime import phasor_calibrate
from phasorpy.phasor import (
    phasor_filter_median,
    phasor_from_signal,
    phasor_threshold,
)
from phasorpy.plot import PhasorPlot, plot_image, plot_phasor

# %%
# Load hyperspectral and FLIM data
# --------------------------------
#
# Load hyperspectral and FLIM signals from Zenodo dataset. This data
# includes LAURDAN fluorescence measurements. Spectral signal data is
# stored in LSM format, while FLIM data is in FBD format.

frequency = 80.0  # MHz
reference_lifetime = 2.5  # Coumarin 6 in ethanol lifetime in ns
laser_factor = 0.99168

# Read spectral signal
spectral_signal = signal_from_lsm(fetch('04 NIH3T3LAURDAN8meanspectra.lsm'))

# Read FLIM signals from both channels and stack them
flim_signal = numpy.stack(
    [
        signal_from_fbd(
            fetch('04NIH3T3_LAURDAN_000$CC0Z.fbd'),
            frame=-1,
            channel=0,
            laser_factor=laser_factor,
        ),
        signal_from_fbd(
            fetch('04NIH3T3_LAURDAN_000$CC0Z.fbd'),
            frame=-1,
            channel=1,
            laser_factor=laser_factor,
        ),
    ],
    axis=0,
)  # Shape: (channels, height, width, time)

calibration_signal = numpy.stack(
    [
        signal_from_fbd(
            fetch('cumarinech1_780LAURDAN_000$CC0Z.fbd'),
            frame=-1,
            channel=0,
            laser_factor=laser_factor,
        ),
        signal_from_fbd(
            fetch('cumarinech2_780LAURDAN_000$CC0Z.fbd'),
            frame=-1,
            channel=1,
            laser_factor=laser_factor,
        ),
    ],
    axis=0,
)  # Shape: (channels, height, width, time)

# %%
# Compute phasor coordinates from spectral signal
spectral_mean, spectral_real, spectral_imag = phasor_from_signal(
    spectral_signal, axis=0
)

# Compute phasor coordinates for both FLIM
flim_mean, flim_real, flim_imag = phasor_from_signal(flim_signal)
calibration_mean, calibration_real, calibration_imag = phasor_from_signal(
    calibration_signal
)

# Calibrate FLIM signals from both channels
flim_real, flim_imag = phasor_calibrate(
    flim_real,
    flim_imag,
    calibration_mean,
    calibration_real,
    calibration_imag,
    frequency=frequency,
    lifetime=reference_lifetime,
    skip_axis=0,
)

# %%
# Preprocess phasor data
# ----------------------
#
# Apply median filtering to reduce noise and threshold based on minimum
# intensity to remove low signal pixels. Different thresholds are used
# for FLIM and spectral data due to their different intensity ranges.

# Filter and threshold spectral data
spectral_mean, spectral_real, spectral_imag = phasor_filter_median(
    spectral_mean,
    spectral_real,
    spectral_imag,
    repeat=3,
)
spectral_mean, spectral_real, spectral_imag = phasor_threshold(
    spectral_mean, spectral_real, spectral_imag, mean_min=18
)

# Filter and threshold both FLIM channels
flim_mean, flim_real, flim_imag = phasor_threshold(
    *phasor_filter_median(
        flim_mean, flim_real, flim_imag, repeat=3, skip_axis=0
    ),
    mean_min=0.5,
)

# %%
# Visualize individual phasor plots
# ---------------------------------
#
# Display the spectral phasor coordinates:

plot_phasor(
    spectral_real,
    spectral_imag,
    xlim=(-0.5, 1.05),
    ylim=(-0.1, 1.05),
    allquadrants=True,
    title='Spectral phasor coordinates',
)

# %%
# Display the FLIM phasor coordinates:

plot = PhasorPlot(frequency=frequency, title='FLIM phasor plot')
plot.hist2d(flim_real[0], flim_imag[0], cmap='Blues')
plot.hist2d(flim_real[1], flim_imag[1], cmap='Oranges', alpha=0.5)
plot.show()

# %%
# Define spectral regions using circular cursors
# ----------------------------------------------
#
# Create circular masks on the spectral phasor plot to define regions of
# interest corresponding to different spectral characteristics. In this
# analysis, the spectral dimension serves as the "main" dimension,
# meaning that spectral cursors control the classification that will be
# applied to correlate with FLIM data.

# Define center coordinates for three spectral regions
spectral_center_real = 0.04, 0.24, 0.14
spectral_center_imag = 0.74, 0.70, 0.72

# Define radius for the circular masks
radius = 0.07

# Create circular masks for each spectral region
spectral_masks = mask_from_circular_cursor(
    spectral_real,
    spectral_imag,
    spectral_center_real,
    spectral_center_imag,
    radius=radius,
)

# %%
# Spectral phasor plot with main cursors
# ----------------------------------------
#
# Display the spectral phasor coordinates as a 2D histogram and overlay
# the circular selection regions. These main cursors define the
# classification scheme for the multidimensional analysis.

plot = PhasorPlot(
    xlim=(-0.5, 1.05),
    ylim=(-0.1, 1.05),
    allquadrants=True,
    title='Spectral phasor plot with main cursors',
)
plot.hist2d(
    spectral_real,
    spectral_imag,
    cmap='Grays',
    bins=200,
)
plot.cursor(
    spectral_center_real,
    spectral_center_imag,
    radius=radius,
    color=CATEGORICAL[:3],
    linestyle='-',
    linewidth=2,
)
plot.show()

# %%
# Spectral pseudo-color image
# ---------------------------
#
# Create a pseudo-colored image where each pixel is colored according to
# its spectral classification based on the main dimension cursors.

spectral_pseudo_color = pseudo_color(*spectral_masks)
plot_image(
    spectral_pseudo_color,
    title='Spectral pseudo-color image',
)

# %%
# First FLIM channel phasor coordinates (spectrally classified)
# -------------------------------------------------------------
#
# Apply the spectral classification masks (from the main dimension) to
# the FLIM phasor coordinates to examine how pixels with different spectral
# properties distribute in the FLIM phasor space. This reveals the correlation
# between spectral and lifetime characteristics.

plot = PhasorPlot(
    title='First FLIM channel phasor coordinates (spectrally classified)',
)

# Plot FLIM coordinates for each spectral class
for i in range(spectral_masks.shape[0]):
    plot.plot(
        flim_real[0][spectral_masks[i]],
        flim_imag[0][spectral_masks[i]],
        color=CATEGORICAL[i],
        alpha=0.05,
        markersize=0.5,
    )

plot.show()

# %%
# Define FLIM first channel as main dimension
# ---------------------------------------------
#
# The inverse analysis can also be performed: select lifetime regions in the
# FLIM phasor space as the new "main" dimension and visualize how they
# distribute in the spectral phasor space. Now FLIM channel 1 serves as the
# main dimension controlling the classification.

# Define center coordinates for FLIM regions of interest
flim_center_real = 0.35, 0.19, 0.27
flim_center_imag = 0.44, 0.38, 0.41

flim_radius = 0.07

flim_ch1_masks = mask_from_circular_cursor(
    flim_real[0],
    flim_imag[0],
    flim_center_real,
    flim_center_imag,
    radius=flim_radius,
)

# %%
# First FLIM channel phasor plot with main cursors
# --------------------------------------------------
#
# Display the FLIM phasor coordinates as a 2D histogram and overlay
# the circular selection regions. FLIM channel 1 now serves as the main
# dimension, with cursors defining the classification scheme.

plot = PhasorPlot(
    title='First FLIM channel phasor plot with main cursors',
)
plot.hist2d(flim_real[0], flim_imag[0], cmap='Grays', bins=200)
plot.cursor(
    flim_center_real,
    flim_center_imag,
    radius=flim_radius,
    color=CATEGORICAL[:3],
    linestyle='-',
    linewidth=2,
)
plot.show()

# %%
# First FLIM channel pseudo-color image
# -------------------------------------
#
# Create a pseudo-colored image where each pixel is colored according to
# its FLIM classification based on the new main dimension cursors.

flim_pseudo_color = pseudo_color(*flim_ch1_masks)
plot_image(
    flim_pseudo_color,
    title='First FLIM channel pseudo-color image',
)

# %%
# Spectral phasor coordinates (FLIM classified)
# ---------------------------------------------
#
# Apply the FLIM classification masks (from the new main dimension) to
# the spectral phasor coordinates to examine how pixels with different
# lifetime properties distribute in the spectral phasor space. This shows
# the correlation from the lifetime perspective.

# Plot spectral coordinates defined by FLIM main cursors
plot = PhasorPlot(
    xlim=(-0.1, 0.4),
    ylim=(0.6, 0.8),
    xticks=[x for x in numpy.arange(-0.1, 0.5, 0.1)],
    yticks=(0.6, 0.7, 0.8),
    allquadrants=True,
    title='Spectral phasor coordinates (FLIM classified)',
)

for i in range(flim_ch1_masks.shape[0]):
    plot.plot(
        spectral_real[flim_ch1_masks[i]],
        spectral_imag[flim_ch1_masks[i]],
        color=CATEGORICAL[i],
        alpha=0.05,
        markersize=1,
    )

plot.show()

# %%
# Second FLIM channel phasor coordinates classified by first FLIM channel
# -----------------------------------------------------------------------
#
# Analyze how the same spectral regions (defined by the original spectral
# main cursors) appear in both FLIM detection channels to understand
# channel-specific lifetime characteristics. This uses the spectral
# classification as the reference.

plot = PhasorPlot(
    title=(
        'Second FLIM channel phasor coordinates classified by first FLIM channel'
    ),
)

# Plot FLIM channel 2 coordinates using spectral main classification
for i in range(spectral_masks.shape[0]):
    plot.plot(
        flim_real[1][spectral_masks[i]],
        flim_imag[1][spectral_masks[i]],
        color=CATEGORICAL[i],
        alpha=0.05,
        markersize=0.5,
    )

plot.show()

# %%
# sphinx_gallery_thumbnail_number = -3
# mypy: allow-untyped-defs, allow-untyped-calls
