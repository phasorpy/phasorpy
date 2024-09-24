"""
Multi-harmonic analysis with PhasorPy
=====================================

An introduction to manipulating multiple harmonics from fluorescence lifetime
and hyperspectral images with the PhasorPy library.

"""

# %%
# Import phasorpy
# ---------------
#
# Start the Python interpreter, import the ``phasorpy`` package,
# and print its version:

import phasorpy

print(phasorpy.__version__)

# %%
# Besides the PhasorPy library, the `numpy <https://numpy.org/>`_ and
# `matplotlib <https://matplotlib.org/>`_ libraries are used for
# array computing and plotting throughout this tutorial:

import numpy
import tifffile  # TODO: from phasorpy.io import read_ometiff
from matplotlib import pyplot

from phasorpy.datasets import fetch

# %%
# Read signal from file
# ---------------------
#
# The :py:mod:`phasorpy.io` module provides functions to read time-resolved
# and hyperspectral image stacks and metadata from many file formats used
# in microscopy, for example PicoQuant PTU, OME-TIFF, Zeiss LSM, and files
# written by SimFCS software.
# However, any other means that yields image stacks in numpy-array compatible
# form can be used instead.
# Image stacks, which may have any number of dimensions, are referred to as
# ``signal`` in the PhasorPy library.
#
# The :py:mod:`phasorpy.datasets` module provides access to various sample
# files. For example, an Imspector TIFF file from the
# `FLUTE <https://zenodo.org/records/8046636>`_  project containing a
# time-correlated single photon counting (TCSPC) histogram
# of a zebrafish embryo at day 3, acquired at 80 MHz:


signal = tifffile.imread(fetch('Embryo.tif'))
frequency = 80.11  # MHz; from the XML metadata in the file

print(signal.shape, signal.dtype)

# %%
# Plot the spatial and histogram averages:

from phasorpy.plot import plot_signal_image

plot_signal_image(signal, axis=0)

# %%
# Calculate phasor coordinates
# ----------------------------
#
# The :py:mod:`phasorpy.phasor` module provides functions to calculate,
# calibrate, filter, and convert phasor coordinates.
#
# Phasor coordinates are the real and imaginary components of the complex
# numbers returned by a real forward Digital Fourier Transform (DFT)
# of a signal at certain harmonics (multiples of the fundamental frequency),
# normalized by the mean intensity (the zeroth harmonic).
# Phasor coordinates are named ``real`` and ``imag`` in the PhasorPy library.
# In literature and other software, they are also known as
# :math:`G` and :math:`S` or :math:`a` and :math:`b` (as in :math:`a + bi`).
#
# Phasor coordinates at multiple harmonics can be calculated from the signal,
# a TCSPC histogram in this case. First and second harmonics are calculated
# in this example. For all harmonics, use ``harmonic='all'``.
# The histogram samples are in the first dimension (`axis=0`):

from phasorpy.phasor import phasor_from_signal

mean, real, imag = phasor_from_signal(signal, harmonic=[1, 2], axis=0)

# %%
# Plot the calculated phasor coordinates at the first and second harmonics:

from phasorpy.plot import plot_phasor_image

plot_phasor_image(mean, real, imag, title='Sample')

# %%
# Calibrate phasor coordinates
# ----------------------------
#
# The signals from time-resolved measurements are convoluted with an
# instrument response function, causing the phasor-coordinates to be
# phase-shifted and modulated (scaled) by unknown amounts.
# The phasor coordinates must therefore be calibrated with coordinates
# obtained from a reference standard of known lifetime, acquired with
# the same instrument settings.
#
# In this case, a homogeneous solution of Fluorescein with a lifetime of
# 4.2 ns was imaged.
#
# Read the signal of the reference measurement from a file:

reference_signal = tifffile.imread(fetch('Fluorescein_Embryo.tif'))

# %%
# Calculate phasor coordinates from the measured reference signal at
# the first and second harmonics:

reference_mean, reference_real, reference_imag = phasor_from_signal(
    reference_signal, harmonic=[1, 2], axis=0
)

# %%
# Show the calculated reference phasor coordinates:

plot_phasor_image(
    reference_mean, reference_real, reference_imag, title='Reference'
)

# %%
# Calibrate the raw phasor coordinates with the reference coordinates of known
# lifetime (Fluorescein, 4.2 ns), at multiple harmonics simultaneously.
# The `skip_axis` parameter must be specified when calibrating multiple
# harmonics, indicating the axes along which the harmonics are calculated:

from phasorpy.phasor import phasor_calibrate

real, imag = phasor_calibrate(
    real,
    imag,
    reference_real,
    reference_imag,
    frequency=frequency,
    harmonic=[1, 2],
    lifetime=4.2,
    skip_axis=0,
)

# %%
# Show the calibrated phasor coordinates at the first and second harmonics:

plot_phasor_image(mean, real, imag, title='Calibrated')

# %%
# The phasor coordinates are now located in the first quadrant, except for
# some with low signal to noise level.
#
# If necessary, the calibration can be undone/reversed using the
# same reference:

uncalibrated_real, uncalibrated_imag = phasor_calibrate(
    real,
    imag,
    reference_real,
    reference_imag,
    frequency=frequency,
    harmonic=[1, 2],
    lifetime=4.2,
    reverse=True,
    skip_axis=0,
)

numpy.testing.assert_allclose(
    (uncalibrated_real, uncalibrated_imag),
    phasor_from_signal(signal, harmonic=[1, 2], axis=0)[1:],
    atol=1e-3,
)

# %%
# Filter phasor coordinates
# -------------------------
#
# Applying median filter to the calibrated phasor coordinates,
# often multiple times, improves contrast and reduces noise. This can
# also be done at multiple harmonics simultaneously by excluding the
# harmonic axis from the filtering:

from phasorpy.phasor import phasor_filter

real, imag = phasor_filter(
    real, imag, method='median', size=3, repeat=2, axes=(1, 2)
)

# %%
# Pixels with low intensities are commonly excluded from analysis and
# visualization of phasor coordinates. For now, harmonics should be treated
# separately when thresholding:

from phasorpy.phasor import phasor_threshold

real1, real2 = real
imag1, imag2 = imag
mean, real1, imag1 = phasor_threshold(mean, real1, imag1, mean_min=1)
mean, real2, imag2 = phasor_threshold(mean, real2, imag2, mean_min=1)

# %%
# Show the calibrated, filtered phasor coordinates:

plot_phasor_image(
    mean,
    real1,
    imag1,
    title='Calibrated, filtered phasor coordinates at first harmonic ',
)
# %%
plot_phasor_image(
    mean,
    real2,
    imag2,
    title='Calibrated, filtered phasor coordinates at second harmonic',
)
# %%
# Store phasor coordinates
# ------------------------
#
# Phasor coordinates and select metadata can be exported to
# `OME-TIFF <https://ome-model.readthedocs.io/en/stable/ome-tiff/>`_
# formatted files, which are compatible with Bio-Formats and Fiji.
#
# Write the calibrated and filtered phasor coordinates at multiple harmonics,
# and frequency to an OME-TIFF file:

from phasorpy.io import phasor_from_ometiff, phasor_to_ometiff

phasor_to_ometiff(
    'phasors.ome.tif',
    mean,
    real,
    imag,
    frequency=frequency,
    harmonic=[1, 2],
    description=(
        'Phasor coordinates at first and second harmonics of a zebrafish '
        'embryo at day 3, calibrated, median-filtered, and thresholded.'
    ),
)

# %%
# Read the phasor coordinates and metadata back from the OME-TIFF file:

mean_, real_, imag_, attrs = phasor_from_ometiff(
    'phasors.ome.tif', harmonic='all'
)

numpy.allclose(real_, real)
assert real_.dtype == numpy.float32
assert attrs['frequency'] == frequency
assert attrs['harmonic'] == [1, 2]
assert attrs['description'].startswith(
    'Phasor coordinates at first and second'
)

# %%
# Plot phasor coordinates
# -----------------------
#
# The :py:mod:`phasorpy.plot` module provides functions and classes for
# plotting phasor and polar coordinates.
#
# Large number of phasor coordinates, such as obtained from imaging,
# are commonly visualized as 2D histograms:

from phasorpy.plot import PhasorPlot

phasorplot1 = PhasorPlot(
    frequency=frequency,
    title='Calibrated, filtered phasor coordinates at first harmonic.',
)
phasorplot1.hist2d(real1, imag1)
phasorplot1.show()
# %%
phasorplot2 = PhasorPlot(
    frequency=frequency,
    title='Calibrated, filtered phasor coordinates at second harmonic.',
)
phasorplot2.hist2d(real2, imag2)
phasorplot2.show()

# %%
# The calibrated phasor coordinates of all pixels lie inside the universal
# semicircle (on which theoretically the phasor coordinates of all single
# exponential lifetimes are located).
# That means, all pixels contain mixtures of signals from multiple lifetime
# components.

# %%
# For comparison, the uncalibrated, unfiltered phasor coordinates:

phasorplot1 = PhasorPlot(
    allquadrants=True, title='Raw phasor coordinates at first harmonic'
)
phasorplot1.hist2d(uncalibrated_real[0], uncalibrated_imag[0])
phasorplot1.show()
# %%
phasorplot2 = PhasorPlot(
    allquadrants=True, title='Raw phasor coordinates at second harmonic'
)
phasorplot2.hist2d(uncalibrated_real[1], uncalibrated_imag[1])
phasorplot2.show()

# %%
# Select phasor coordinates
# -------------------------

# The :py:mod:`phasorpy.cursors` module provides functions for selecting phasor
# coordinates to define and mask regions of interest within the phasor space.

# Mask regions of interest in the phasor space using circular cursors:

from phasorpy.color import CATEGORICAL
from phasorpy.cursors import mask_from_circular_cursor

radius = 0.05
cursors_real1 = [0.69, 0.59]
cursors_imag1 = [0.32, 0.33]
cursors_masks1 = mask_from_circular_cursor(
    real1, imag1, cursors_real1, cursors_imag1, radius=radius
)

phasorplot1 = PhasorPlot(
    frequency=frequency, title='Cursors at first harmonic'
)
phasorplot1.hist2d(real1, imag1)
for i in range(len(cursors_real1)):
    phasorplot1.circle(
        cursors_real1[i],
        cursors_imag1[i],
        radius=radius,
        color=CATEGORICAL[i],
        linestyle='-',
    )
phasorplot1.show()

# %%
cursors_real2 = [0.53, 0.43]
cursors_imag2 = [0.38, 0.35]
cursors_masks2 = mask_from_circular_cursor(
    real2, imag2, cursors_real2, cursors_imag2, radius=radius
)

phasorplot2 = PhasorPlot(
    frequency=frequency, title='Cursors at second harmonic'
)
phasorplot2.hist2d(real2, imag2)
for i in range(len(cursors_real2)):
    phasorplot2.circle(
        cursors_real2[i],
        cursors_imag2[i],
        radius=radius,
        color=CATEGORICAL[i + 2],
        linestyle='-',
    )
phasorplot2.show()

# %%
# The cursor masks can be blended with the mean intensity image to produce
# a pseudo-colored image:

from phasorpy.cursors import pseudo_color

pseudo_color_image1 = pseudo_color(*cursors_masks1, intensity=mean)

fig, ax = pyplot.subplots()
ax.set_title('Pseudo-color image from circular cursors at first harmonic')
ax.imshow(pseudo_color_image1)
pyplot.show()

# %%
pseudo_color_image2 = pseudo_color(
    *cursors_masks2, intensity=mean, colors=CATEGORICAL[2:]
)

fig, ax = pyplot.subplots()
ax.set_title('Pseudo-color image from circular cursors at second harmonic')
ax.imshow(pseudo_color_image2)
pyplot.show()

# %%
# Component analysis
# ------------------

# TODO

# %%
# Spectral phasors
# ----------------
#
# Phasor coordinates can be calculated from hyperspectral images (acquired
# at many equidistant emission wavelengths) and processed in much the same
# way as time-resolved signals. Calibration is not necessary.
#
# Open a hyperspectral dataset acquired with a laser scanning microscope
# at 30 emission wavelengths:

from phasorpy.io import read_lsm

hyperspectral_signal = read_lsm(fetch('paramecium.lsm'))

plot_signal_image(hyperspectral_signal, axis=0, title='Hyperspectral image')

# %%
# Calculate phasor coordinates at the first and second harmonic and filter out
# pixels with low intensities:

mean, real, imag = phasor_from_signal(
    hyperspectral_signal, harmonic=[1, 2], axis=0
)
real1, real2 = real
imag1, imag2 = imag
_, real1, imag1 = phasor_threshold(mean, real1, imag1, mean_min=1)
_, real2, imag2 = phasor_threshold(mean, real2, imag2, mean_min=1)

# %%
# Plot the phasor coordinates as a two-dimensional histogram and select two
# clusters in the phasor plot by means of elliptical cursors:

radius = [0.1, 0.06]
radius_minor = [0.3, 0.25]
cursors_real1 = [-0.33, 0.54]
cursors_imag1 = [-0.72, -0.74]

phasorplot1 = PhasorPlot(
    allquadrants=True, title='Spectral phasor plot at first harmonic'
)
phasorplot1.hist2d(real1, imag1, cmap='Greys')
for i in range(len(cursors_real1)):
    phasorplot1.cursor(
        cursors_real1[i],
        cursors_imag1[i],
        radius=radius[i],
        radius_minor=radius_minor[i],
        color=CATEGORICAL[i],
        linestyle='-',
    )
phasorplot1.show()

# %%
cursors_real2 = [-0.23, -0.2]
cursors_imag2 = [0.27, -0.7]

phasorplot2 = PhasorPlot(
    allquadrants=True, title='Spectral phasor plot at second harmonic'
)
phasorplot2.hist2d(real2, imag2, cmap='Greys')
for i in range(len(cursors_real2)):
    phasorplot2.cursor(
        cursors_real2[i],
        cursors_imag2[i],
        radius=radius[i],
        radius_minor=radius_minor[i],
        color=CATEGORICAL[i + 2],
        linestyle='-',
    )
phasorplot2.show()

# %%
# Use the elliptic cursors to mask regions of interest in the phasor space:

from phasorpy.cursors import mask_from_elliptic_cursor

elliptic_masks1 = mask_from_elliptic_cursor(
    real1,
    imag1,
    cursors_real1,
    cursors_imag1,
    radius=radius,
    radius_minor=radius_minor,
)
elliptic_masks2 = mask_from_elliptic_cursor(
    real2,
    imag2,
    cursors_real2,
    cursors_imag2,
    radius=radius,
    radius_minor=radius_minor,
)

# %%
# Plot a pseudo-color image, composited from the elliptic cursor masks and
# the mean intensity image:

pseudo_color_image1 = pseudo_color(*elliptic_masks1, intensity=mean)

fig, ax = pyplot.subplots()
ax.set_title('Pseudo-color image from elliptic cursors at first harmonic')
ax.imshow(pseudo_color_image1)
pyplot.show()

# %%
pseudo_color_image2 = pseudo_color(
    *elliptic_masks2, intensity=mean, colors=CATEGORICAL[2:]
)

fig, ax = pyplot.subplots()
ax.set_title('Pseudo-color image from elliptic cursors at second harmonic')
ax.imshow(pseudo_color_image2)
pyplot.show()

# %%
# Appendix
# --------
#
# Print information about Python interpreter and installed packages:

print(phasorpy.versions())

# %%
# sphinx_gallery_thumbnail_number = 12
# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
