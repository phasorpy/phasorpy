"""
Introduction to PhasorPy
========================

An introduction to using the PhasorPy library.

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
# run the following command on a command prompt, shell or terminal::
#
#     python -m pip install -U "phasorpy[all]"
#
# .. note::
#    The PhasorPy library is in its early stages of development
#    and has not yet been released to PyPI.
#    The development version of PhasorPy can be `installed manually
#    <https://packaging.python.org/en/latest/tutorials/installing-packages/>`_,
#    for example, using the binary wheels from `GitHub Actions
#    <https://github.com/phasorpy/phasorpy/actions/workflows/build_wheels.yml>`_,
#    or the source code on GitHub (requires a C compiler)::
#
#        python -m pip install git+https://github.com/phasorpy/phasorpy.git

# %%
# Import phasorpy
# ---------------
#
# Start the Python interpreter, import the ``phasorpy`` package,
# and print its version:

import phasorpy

print(phasorpy.__version__)

# %%
# Read signal from file
# ---------------------
#
# The :py:mod:`phasorpy.datasets` module provides access to various sample
# files, for example, a TIFF file containing a time-correlated
# single photon counting (TCSPC) histogram obtained at 80 MHz.
#
# The :py:mod:`phasorpy.io` module provides many functions to read image
# and metadata from file formats used in microscopy. However, here the
# `tifffile <https://pypi.org/project/tifffile/>`_ library is used directly:

# TODO: use phasorpy.io function to read histogram and metadata from PTU file

import tifffile

from phasorpy.datasets import fetch

signal = tifffile.imread(fetch('Embryo.tif'))
frequency = 80.0

print(signal.shape, signal.dtype)

# %%
# Plot the spatial and histogram averages:

from matplotlib import pyplot

fig, (ax0, ax1) = pyplot.subplots(1, 2)
fig.set_size_inches(8, 3)
ax0.set_title('average image')
ax0.imshow(signal.mean(axis=0))
ax1.set_title('average histogram')
ax1.plot(signal.mean(axis=(1, 2)))
pyplot.show()

# %%
# Calculate phasor coordinates
# ----------------------------
#
# The :py:mod:`phasorpy.phasor` module provides functions to calculate,
# convert, and correct phasor coordinates.
#
# Phasor coordinate are calculated from the signal, a TCSPC histogram in
# this case. The histogram samples are in the first dimension (`axis=0`):

from phasorpy.phasor import phasor_from_signal_f1

mean, real, imag = phasor_from_signal_f1(signal, axis=0)

# %%
# Plot the calculated phasor coordinates:

import numpy

numpy.testing.assert_array_equal(mean, signal.mean(axis=0))

fig, (ax0, ax1) = pyplot.subplots(1, 2)
fig.set_size_inches(8, 3)
ax0.set_title('G, real')
ax0.imshow(real)
ax1.set_title('S, imag')
ax1.imshow(imag)
pyplot.show()

# %%
# Calibrate phasor coordinates
# ----------------------------
#
# Phasor coordinates from time-resolved measurements must be calibrated
# with the polar coordinates (phase and modulation) obtained from a
# reference standard of known lifetime, acquired with the same instrument
# settings.
#
# Read the signal of the reference measurement from a file:

reference_signal = tifffile.imread(fetch('Fluorescein_Embryo.tif'))

# %%
# Calculate phasor coordinates from the measured reference signal:

_, measured_real, measured_imag = phasor_from_signal_f1(
    reference_signal, axis=0
)

# %%
# Average the phasor coordinates, assuming there are no spatial aberations:

from phasorpy.phasor import phasor_center

measured_real, measured_imag = phasor_center(measured_real, measured_imag)

# %%
# Calculate absolute phasor coordinates from the known lifetime of the
# reference (Fluorescein, 4.2 ns):

from phasorpy.phasor import phasor_from_lifetime

known_real, known_imag = phasor_from_lifetime(frequency, 4.2)

# %%
# Calculate polar coordinates (phase shift and relative modulation) to
# correct phasor coordinates:

from phasorpy.phasor import polar_from_reference_phasor

phase0, modulation0 = polar_from_reference_phasor(
    measured_real, measured_imag, known_real, known_imag
)

# %%
# Finally, calibrate the raw phasor coordinates obtained from the signal:

from phasorpy.phasor import phasor_calibrate

real, imag = phasor_calibrate(real, imag, phase0, modulation0)

# %%
# Filter phasor coordinates
# -------------------------
#
# Applying median filter to the calibrated phasor coordinates,
# often multiple times, improves contrast:

# TODO: replace this with a ``phasor_filter`` function?
from skimage.filters import median

for _ in range(2):
    real = median(real)
    imag = median(imag)

# %%
# Pixels with low intensities are commonly excluded from analysis and
# visualization of phasor coordinates:

# TODO: replace this with a ``phasor_mask`` function?
mask = mean > 1
real = real[mask]
imag = imag[mask]

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

plot = phasorpy.plot.PhasorPlot(
    frequency=frequency, title='Calibrated, filtered phasor coordinates'
)
plot.hist2d(real, imag)
plot.show()

# %%
# For comparison, the uncalibrated, unfiltered phasor coordinates:

plot = PhasorPlot(allquadrants=True, title='Raw phasor coordinates')
plot.semicircle()
plot.hist2d(*phasor_from_signal_f1(signal, axis=0)[1:])
plot.show()

# %%
# To be continued
# ---------------
#

# %%
# sphinx_gallery_thumbnail_number = 3
