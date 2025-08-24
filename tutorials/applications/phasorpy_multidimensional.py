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
  `A multidimensional phasor approach reveals LAURDAN photophysics in NIH-3T3
  cell membranes <https://doi.org/10.1038/s41598-017-08564-z>`_.
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
from matplotlib import pyplot

from phasorpy.color import CATEGORICAL
from phasorpy.cursor import mask_from_circular_cursor, pseudo_color
from phasorpy.datasets import fetch
from phasorpy.filter import phasor_filter_median, phasor_threshold
from phasorpy.io import signal_from_fbd, signal_from_lsm
from phasorpy.lifetime import phasor_calibrate
from phasorpy.phasor import phasor_from_signal
from phasorpy.plot import PhasorPlot, plot_image, plot_phasor

# %%
# Read spectral and FLIM signals
# ------------------------------
#
# Read hyperspectral and FLIM signals and metadata from the Zenodo dataset
# of NIH-3T3 fibroblast cells stained with the membrane probe LAURDAN.
# The spectral data is stored in Zeiss LSM format, while the two-channel
# FLIM data is in FLIMbox FBD format. The calibration data for both FLIM
# channels are stored in separate files:

spectral_signal = signal_from_lsm(fetch('04 NIH3T3LAURDAN8meanspectra.lsm'))

flim_signal = signal_from_fbd(
    fetch('04NIH3T3_LAURDAN_000$CC0Z.fbd'),
    frame=-1,  # integrate all frames
    channel=None,  # load all channels
    laser_factor=0.99168,  # override incorrect metadata in FBD file
)

reference_signal = numpy.stack(
    [
        signal_from_fbd(
            fetch(f'cumarinech{ch + 1}_780LAURDAN_000$CC0Z.fbd'),
            frame=-1,
            channel=ch,
            laser_factor=0.99168,
        )
        for ch in (0, 1)
    ],
)

frequency = flim_signal.attrs['frequency'] * flim_signal.attrs['harmonic']

# %%
# Phasor transform
# ----------------
#
# Compute phasor coordinates from the spectral and FLIM signals. Calibrate
# the FLIM phasor coordinates with the reference coordinates of known lifetime
# (2.5 ns, Coumarin 6 in ethanol):

spectral_mean, spectral_real, spectral_imag = phasor_from_signal(
    spectral_signal
)

flim_mean, flim_real, flim_imag = phasor_from_signal(flim_signal)

flim_real, flim_imag = phasor_calibrate(
    flim_real,
    flim_imag,
    *phasor_from_signal(reference_signal),
    frequency=frequency,
    lifetime=2.5,
    skip_axis=0,
)

# %%
# Preprocess phasor coordinates
# -----------------------------
#
# Apply median filtering to reduce noise and threshold based on minimum
# intensity to remove low signal pixels. Different thresholds are used
# for FLIM and spectral data due to their different intensity ranges:

spectral_mean, spectral_real, spectral_imag = phasor_threshold(
    *phasor_filter_median(
        spectral_mean, spectral_real, spectral_imag, repeat=3
    ),
    mean_min=18,
)

flim_mean, flim_real, flim_imag = phasor_threshold(
    *phasor_filter_median(
        flim_mean, flim_real, flim_imag, repeat=3, skip_axis=0
    ),
    mean_min=0.3,
)

# %%
# Plot phasor coordinates
# -----------------------
#
# Plot the spectral phasor coordinates:

plot_phasor(
    spectral_real,
    spectral_imag,
    xlim=(-0.5, 1.05),
    ylim=(-0.1, 1.05),
    allquadrants=True,
    title='Spectral phasor',
)

# %%
# Plot the FLIM phasor coordinates of channel 1 (blue) and 2 (orange):

fig, axs = pyplot.subplots(2, 1, figsize=(6.4, 9))
fig.suptitle('FLIM phasor')
for ch in (0, 1):
    plot = PhasorPlot(
        ax=axs[ch], frequency=frequency, title=f'Channel {ch + 1}'
    )
    plot.hist2d(flim_real[ch], flim_imag[ch])
fig.tight_layout()
plot.show()

# %%
# Mask spectral phasor using cursors
# ----------------------------------
#
# Create circular masks on the spectral phasor plot to define regions of
# interest corresponding to different spectral characteristics.
# The spectral dimension serves as the "main" dimension, meaning that
# spectral cursors control the classification that will be applied to
# correlate with the FLIM dimension.

spectral_cursor_real = [0.04, 0.24, 0.14]
spectral_cursor_imag = [0.74, 0.70, 0.72]
spectral_cursor_radius = 0.05

spectral_mask = mask_from_circular_cursor(
    spectral_real,
    spectral_imag,
    spectral_cursor_real,
    spectral_cursor_imag,
    radius=spectral_cursor_radius,
)

# %%
# Plot the spectral phasor and main cursors:

plot = PhasorPlot(
    xlim=(-0.5, 1.05),
    ylim=(-0.1, 1.05),
    xticks=None,
    yticks=None,
    allquadrants=True,
    title='Spectral phasor and cursors',
)
plot.hist2d(
    spectral_real,
    spectral_imag,
    cmap='Grays',
    bins=200,
)
plot.cursor(
    spectral_cursor_real,
    spectral_cursor_imag,
    radius=spectral_cursor_radius,
    color=CATEGORICAL[:3],
    linewidth=2,
)
plot.show()

# %%
# Spectral pseudo-color image
# ---------------------------
#
# Create a pseudo-colored image where each pixel is colored according to
# its spectral classification based on the main cursors:

spectral_pseudo_color = pseudo_color(*spectral_mask)
plot_image(spectral_pseudo_color, title='Spectral pseudo-color image')

# %%
# Spectral classified FLIM phasor
# -------------------------------
#
# Apply the spectral classification masks (from the main dimension) to
# the phasor coordinates of both FLIM channels.
# This shows how pixels with different spectral properties distribute in
# the FLIM phasor space and reveals the correlation between spectral and
# lifetime characteristics:

fig, axs = pyplot.subplots(2, 1, figsize=(6.4, 9))
fig.suptitle('Spectral classified FLIM phasor')

for ch in (0, 1):
    plot = PhasorPlot(
        ax=axs[ch], frequency=frequency, title=f'Channel {ch + 1}'
    )
    for i in range(spectral_mask.shape[0]):
        plot.plot(
            flim_real[ch][spectral_mask[i]],
            flim_imag[ch][spectral_mask[i]],
            color=CATEGORICAL[i],
            alpha=0.05,
            markersize=0.5,
        )
fig.tight_layout()
plot.show()

# %%
# FLIM classified spectral phasor
# -------------------------------
#
# The inverse analysis can also be performed: select regions in the
# FLIM phasor space as the "main" dimension and correlate with the spectral
# phasor space.
#
# Mask the FLIM phasor coordinates of the first channel using cursors:

flim_cursor_real = [0.35, 0.19, 0.27]
flim_cursor_imag = [0.44, 0.38, 0.41]
flim_cursor_radius = 0.05

flim_mask = mask_from_circular_cursor(
    flim_real[0],
    flim_imag[0],
    flim_cursor_real,
    flim_cursor_imag,
    radius=flim_cursor_radius,
)

# %%
# Plot the FLIM phasor and cursors:

plot = PhasorPlot(
    frequency=frequency,
    title='FLIM phasor and cursors (first channel)',
)
plot.hist2d(flim_real[0], flim_imag[0], cmap='Grays', bins=200)
plot.cursor(
    flim_cursor_real,
    flim_cursor_imag,
    radius=flim_cursor_radius,
    color=CATEGORICAL[:3],
    linewidth=2,
)
plot.show()

# %%
# Display the FLIM pseudo-color image:

plot_image(
    pseudo_color(*flim_mask), title='FLIM pseudo-color image (first channel)'
)

# %%
# Plot the FLIM classified spectral phasor:

plot = PhasorPlot(
    xlim=(-0.5, 1.05),
    ylim=(-0.1, 1.05),
    allquadrants=True,
    title='FLIM classified spectral phasor',
)

for i in range(flim_mask.shape[0]):
    plot.plot(
        spectral_real[flim_mask[i]],
        spectral_imag[flim_mask[i]],
        color=CATEGORICAL[i],
        alpha=0.05,
        markersize=1,
    )

plot.show()

# %%
# sphinx_gallery_thumbnail_number = 4
# mypy: allow-untyped-defs, allow-untyped-calls
