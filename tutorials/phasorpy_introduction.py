"""
Introduction to PhasorPy
========================

An introduction to using the PhasorPy library.

PhasorPy is an open-source Python library for the analysis of fluorescence
lifetime and hyperspectral images using the :doc:`/phasor_approach`.

Using the PhasorPy library requires familiarity with the phasor approach,
image processing, array programming, and Python.

"""

# %%
# Install Python
# --------------
#
# An installation of Python version 3.10 or higher is required to use the
# PhasorPy library.
# Python is an easy to learn, powerful programming language.
# Python installers can be obtained from, for example,
# `Python.org <https://www.python.org/downloads/>`_ or
# `Anaconda.com <https://www.anaconda.com/>`_.
# Refer to the `Python Tutorial <https://docs.python.org/3/tutorial/>`_
# for an introduction to Python.
#
# Install PhasorPy
# ----------------
#
# To download and install the PhasorPy library and all its dependencies from
# the `Python Package Index <https://pypi.org/project/phasorpy/>`_ (PyPI),
# run the following command on a command prompt, shell, or terminal::
#
#     python -m pip install -U "phasorpy[all]"
#
# .. note::
#    The PhasorPy library is in its early stages of development
#    and has not yet been released to PyPI.
#
# The development version of PhasorPy can be installed instead from the
# latest source code on GitHub. This requires a C compiler, such as
# XCode, Visual Studio, or gcc, to be installed::
#
#     python -m pip install git+https://github.com/phasorpy/phasorpy.git

# %%
# Import phasorpy
# ---------------
#
# Start the Python interpreter, import the ``phasorpy`` package,
# and print its version:

import phasorpy

print(phasorpy.__version__)

# %%
# Besides the phasorpy library, the `numpy <https://numpy.org/>`_ and
# `matplotlib <https://matplotlib.org/>`_ libraries are used for
# array computing and plotting throughout this tutorial:

import numpy
import tifffile
from matplotlib import pyplot

from phasorpy.datasets import fetch

# %%
# Read signal from file
# ---------------------
#
# The :py:mod:`phasorpy.datasets` module provides access to various sample
# files, for example, a TIFF file containing a time-correlated
# single photon counting (TCSPC) histogram obtained at 80 MHz.
#
# The :py:mod:`phasorpy.io` module provides many functions to read
# time-resolved and hyperspectral image and metadata from file formats used
# in microscopy. However, here the
# `tifffile <https://pypi.org/project/tifffile/>`_ library is used directly
# to read image stacks from files.
#
# The image data is referred to as the ``signal`` in the phasorpy library.

# TODO: use phasorpy.io function to read histogram and metadata from PTU file


signal = tifffile.imread(fetch('Embryo.tif'))
frequency = 80.0

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
# of a signal at certain harmonics, normalized by the mean intensity
# (the zeroth harmonic).
# Phasor coordinates are named ``real`` and ``imag`` in the phasorpy library.
# In the literature and other software, they are also known as
# (:math:`G`) and (:math:`S`).
#
# Phasor coordinates of the first harmonic are calculated from the signal,
# a TCSPC histogram in this case.
# The histogram samples are in the first dimension (`axis=0`):

from phasorpy.phasor import phasor_from_signal

mean, real, imag = phasor_from_signal(signal, axis=0)

# %%
# The phasor coordinates are undefined if the mean intensity is zero.
# In that case, the arrays contain special NaN (Not a Number) values,
# which are ignored in further analysis:

print(real[:4, :4])

# %%
# Plot the calculated phasor coordinates:

from phasorpy.plot import plot_phasor_image

plot_phasor_image(mean, real, imag, title='Sample')

# %%
# By default, only the phasor coordinates at the first harmonic is calculated.
# However, only when the phasor coordinates at all harmonics are considered
# (including the mean intensity) is the signal completely described:

from phasorpy.phasor import phasor_to_signal

phasor_all_harmonics = phasor_from_signal(signal, axis=0, harmonic='all')
reconstructed_signal = phasor_to_signal(
    *phasor_all_harmonics, axis=0, samples=signal.shape[0]
)

numpy.testing.assert_allclose(
    numpy.nan_to_num(reconstructed_signal), signal, atol=1e-3
)

# %%
# Calibrate phasor coordinates
# ----------------------------
#
# The signals from time-resolved measurements are convoluted with an
# instrument response function, causing the phasor-coordinates to be
# phase shifted and modulated (scaled) by unknown amounts.
# The phasor coordinates must therefore be calibrated with coordinates
# obtained from a reference standard of known lifetime, acquired with
# exactly the same instrument settings.
#
# In this case, a homogeneous solution of Fluorescein with a lifetime of
# 4.2 ns was imaged.
#
# Read the signal of the reference measurement from a file:

reference_signal = tifffile.imread(fetch('Fluorescein_Embryo.tif'))

# %%
# Calculate phasor coordinates from the measured reference signal:

reference_mean, reference_real, reference_imag = phasor_from_signal(
    reference_signal, axis=0
)

# %%
# Show the calculated reference phasor coordinates:

plot_phasor_image(
    reference_mean, reference_real, reference_imag, title='Reference'
)

# %%
# Calibrate the raw phasor coordinates with the reference coordinates of known
# lifetime (Fluorescein, 4.2 ns):

from phasorpy.phasor import phasor_calibrate

real, imag = phasor_calibrate(
    real,
    imag,
    reference_real,
    reference_imag,
    frequency=frequency,
    lifetime=4.2,
)

# %%
# Show the calibrated phasor coordinates:

plot_phasor_image(mean, real, imag, title='Calibrated')

# %%
# The phasor coordinates are now located in the first quadrant, except for
# those with low signal to noise level.
#
# If necessary, the calibration can be undone/reversed using the
# same reference:

uncalibrated_real, uncalibrated_imag = phasor_calibrate(
    real,
    imag,
    reference_real,
    reference_imag,
    frequency=frequency,
    lifetime=4.2,
    reverse=True,
)

numpy.testing.assert_allclose(
    (mean, uncalibrated_real, uncalibrated_imag),
    phasor_from_signal(signal, axis=0),
    atol=1e-3,
)

# %%
# Filter phasor coordinates
# -------------------------
#
# Applying median filter to the calibrated phasor coordinates,
# often multiple times, improves contrast and reduces noise:

from phasorpy.phasor import phasor_filter

real, imag = phasor_filter(real, imag, method='median', size=3, repeat=2)

# %%
# Pixels with low intensities are commonly excluded from analysis and
# visualization of phasor coordinates:

from phasorpy.phasor import phasor_threshold

mean, real, imag = phasor_threshold(mean, real, imag, mean_min=1)

# %%

plot_phasor_image(
    mean, real, imag, title='Calibrated, filtered phasor coordinates'
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

phasorplot = PhasorPlot(
    frequency=frequency, title='Calibrated, filtered phasor coordinates'
)
phasorplot.hist2d(real, imag)
phasorplot.show()

# %%
# The calibrated phasor coordinates of all pixels lie inside the universal
# semicircle (on which theoretically the phasor coordinates of all single
# exponential lifetimes are located).
# That means, all pixels contain mixtures of signals from multiple lifetime
# components.

# %%
# For comparison, the uncalibrated, unfiltered phasor coordinates:

phasorplot = PhasorPlot(allquadrants=True, title='Raw phasor coordinates')
phasorplot.hist2d(uncalibrated_real, uncalibrated_imag)
phasorplot.show()

# %%
# Convert to apparent single lifetimes
# ------------------------------------
#
# The cartesian phasor coordinates can be converted to polar coordinates,
# phase and modulation. Theoretical single exponential lifetimes, the
# "apparent single lifetimes", can be calculated from either phase or
# modulation at each pixel:

from phasorpy.phasor import phasor_to_apparent_lifetime

phase_lifetime, modulation_lifetime = phasor_to_apparent_lifetime(
    real, imag, frequency=frequency
)

# %%
# Plot the apparent single lifetimes as histograms or images using the
# matplotlib library:

fig = pyplot.figure()
fig.suptitle('Apparent single lifetimes')
ax0 = fig.add_subplot(1, 2, 1)
ax0.set(title='from Phase')
ax0.imshow(phase_lifetime, vmin=0.0, vmax=3.5)
ax1 = fig.add_subplot(1, 2, 2)
ax1.set(title='from Modulation')
pos = ax1.imshow(modulation_lifetime, vmin=0.0, vmax=3.5)
fig.colorbar(pos, ax=[ax0, ax1], shrink=0.4, location='bottom')
pyplot.show()

# %%

fig, ax = pyplot.subplots()
ax.set(
    title='Apparent single lifetimes',
    xlim=[0, 3.5],
    xlabel='lifetime [ns]',
    ylabel='number pixels',
)
ax.hist(phase_lifetime.flat, bins=64, range=(0, 3.5), label='from Phase')
ax.hist(
    modulation_lifetime.flat, bins=64, range=(0, 3.5), label='from Modulation'
)
ax.legend()
pyplot.show()

# %%
# Geometrically, the apparent single lifetimes are defined by the intersections
# of the universal semicircle with a line (phase) and circle (modulation)
# from/around the origin through the phasor coordinate,
# here demonstrated for the average phasor coordinates:

from phasorpy.phasor import phasor_center, phasor_from_apparent_lifetime

real_center, imag_center = phasor_center(real, imag)
apparent_lifetimes = phasor_to_apparent_lifetime(
    real_center, imag_center, frequency=frequency
)
apparent_lifetime_phasors = phasor_from_apparent_lifetime(
    apparent_lifetimes, None, frequency=frequency
)

phasorplot = PhasorPlot(
    frequency=frequency, title='Average apparent single lifetimes'
)
phasorplot.hist2d(real, imag)
phasorplot.cursor(real_center, imag_center, color='tab:orange')
phasorplot.plot(real_center, imag_center, color='tab:orange')
phasorplot.plot(*apparent_lifetime_phasors, color='tab:orange')
phasorplot.ax.annotate(
    f'{apparent_lifetimes[0]:.2} ns',
    (
        apparent_lifetime_phasors[0][0] + 0.02,
        apparent_lifetime_phasors[1][0] + 0.02,
    ),
)
phasorplot.ax.annotate(
    f'{apparent_lifetimes[1]:.2} ns',
    (
        apparent_lifetime_phasors[0][1],
        apparent_lifetime_phasors[1][1] + 0.02,
    ),
)
phasorplot.show()

# %%
# To be continued
# ---------------

# %%
# Appendix
# --------
#
# Print information about Python interpreter and installed packages:

print(phasorpy.versions())

# %%
# sphinx_gallery_thumbnail_number = -1
# mypy: disable-error-code="arg-type"
