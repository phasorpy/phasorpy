"""
File input/output
=================

Read and write phasor related data from and to various file formats.

The :py:mod:`phasorpy.io` module provides functions to read phasor
coordinates, FLIM/TCSPC histograms, hyperspectral image stacks, lifetime
images, and relevant metadata from various file formats used in bio-imaging.
The module also includes functions to write phasor coordinates to OME-TIFF
and SimFCS referenced files.

"""

# %%
# .. note::
#   This tutorial is work in progress.
#   Not all supported file formats are included yet.

# %%
# Import required modules and functions.
# Define a helper function to compare image histograms:

import math

import numpy
from matplotlib import pyplot

from phasorpy.phasor import (
    phasor_from_signal,
    phasor_threshold,
    phasor_to_apparent_lifetime,
    phasor_transform,
)
from phasorpy.plot import plot_phasor, plot_phasor_image, plot_signal_image


def plot_histograms(
    *images, title=None, xlabel=None, ylabel=None, labels=None, **kwargs
):
    # TODO: replace by future phasorpy.plot.plot_histograms
    if labels is None:
        labels = [None] * len(images)
    fig, ax = pyplot.subplots()
    for image, label in zip(images, labels):
        ax.hist(image.flatten(), label=label, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    pyplot.tight_layout()
    pyplot.show()


# %%
# Sample files
# ------------
#
# PhasorPy provides access to sample files in various formats shared publicly
# on Zenodo, Figshare, or GitHub.
# The files in these repositories are accessed using the
# :py:func:`phasorpy.datasets.fetch` function, which transparently downloads
# files if they were not already downloaded before. The function returns
# the path to the downloaded file:

from phasorpy.datasets import fetch

filename = fetch('FLIM_testdata.lif')
print(filename)

# %%
# Consider sharing datasets with the `PhasorPy community on Zenodo
# <https://zenodo.org/communities/phasorpy/>`_.

# %%
# Leica LIF and XLEF
# ------------------
#
# Leica image files (LIF and XLEF) are written by Leica LAS X software.
# They contain collections of multi-dimensional images and metadata from
# a variety of microscopy acquisition and analysis modes.
# The PhasorPy library currently supports reading hyperspectral images,
# phasor coordinates, and lifetime images from Leica image files.
# The implementation is based on the
# `liffile <https://github.com/cgohlke/liffile/>`_ library.
#
# LIF-FLIM files that were analyzed with the LAS X software contain
# calculated phasor coordinates, lifetime images, and relevant metadata.
# The :py:func:`phasorpy.io.phasor_from_lif` and
# :py:func:`phasorpy.io.lifetime_from_lif` functions are used to read those
# data from the `FLIM_testdata
# <https://dx.doi.org/10.6084/m9.figshare.22336594.v1>`_ dataset:

from phasorpy.io import lifetime_from_lif, phasor_from_lif

filename = 'FLIM_testdata.lif'
mean, real, imag, attrs = phasor_from_lif(fetch(filename))

plot_phasor_image(mean, real, imag, title=filename)

# %%
# The returned mean intensity and uncalibrated phasor coordinates are
# numpy arrays. ``attrs`` is a dictionary containing metadata, including the
# auto-reference phase (in degrees) and modulation for all image channels,
# as well as the fundamental laser frequency (in MHz):

frequency = attrs['frequency']
channel_0 = attrs['flim_phasor_channels'][0]
reference_phase = channel_0['AutomaticReferencePhase']
reference_modulation = channel_0['AutomaticReferenceAmplitude']
intensity_min = channel_0['IntensityThreshold'] / attrs['samples']

# %%
# These metadata are used to calibrate and threshold the phasor coordinates:

real, imag = phasor_transform(
    real, imag, -math.radians(reference_phase), 1 / reference_modulation
)

mean, real, imag = phasor_threshold(mean, real, imag, mean_min=intensity_min)

plot_phasor(
    real,
    imag,
    frequency=frequency,
    title=f'{filename} ({frequency} MHz)',
    cmin=10,
)

# %%
# Apparent single lifetimes are calculated from the calibrated phasor
# coordinates and compared to the lifetimes calculated by LAS X software:

phase_lifetime, modulation_lifetime = phasor_to_apparent_lifetime(
    real, imag, frequency
)

fitted_lifetime = lifetime_from_lif(fetch(filename))[0]
fitted_lifetime[numpy.isnan(mean)] = numpy.nan

plot_histograms(
    phase_lifetime,
    modulation_lifetime,
    fitted_lifetime,
    range=(0, 10),
    bins=100,
    alpha=0.66,
    title='Lifetime histograms',
    xlabel='Lifetime (ns)',
    ylabel='Counts',
    labels=[
        'Phase lifetime',
        'Modulation lifetime',
        'Fitted lifetimes from LIF',
    ],
)

# %%
# The apparent single lifetimes from phase and modulation do not exactly match.
# Most likely there is more than one lifetime component in the sample.
# This could also explain the difference from the lifetimes fitted by the
# LAS X software.

# %%
# .. note::
#   FLIM/TCSPC histograms cannot currently be read directly from
#   LIF-FLIM files since the storage scheme for those data is undocumented
#   or patent-pending. However, TTTR records can be exported from LIF-FLIM
#   files to PicoQuant PTU format by the LAS X software.

# %%
# .. todo::
#   No public, hyperspectral dataset in LIF format is currently available
#   for demonstrating the :py:func:`phasorpy.io.signal_from_lif` function.

# %%
# PicoQuant PTU
# -------------
#
# PicoQuant PTU files are written by PicoQuant SymPhoTime, Leica LAS X, and
# other software. The files contain time-correlated single-photon
# counting (TCSPC) measurement data and instrumentation parameters.
# The PhasorPy library supports reading TCSPC histograms from PicoQuant PTU
# files acquired in T3 imaging mode. The implementation is based on the
# `ptufile <https://github.com/cgohlke/ptufile/>`_ library.
#
# The :py:func:`phasorpy.io.signal_from_ptu` function is used to read
# the TCSPC histogram from a PTU file exported from the `FLIM_testdata
# <https://dx.doi.org/10.6084/m9.figshare.22336594.v1>`_ dataset.
# The function by default returns a 5-dimensional image with dimension order
# TYXCH. Channel and frames are specified to reduce the dimensionality:

from phasorpy.io import signal_from_ptu

filename = 'FLIM_testdata.lif.ptu'
signal = signal_from_ptu(fetch(filename), channel=0, frame=0, keepdims=False)

plot_signal_image(signal, title=filename)

# %%
# The TCSPC histogram contains more photons than the phasor intensity image
# stored in the LIF-FLIM file. The LAS X software likely applies a filter to
# the TCSPC histogram before phasor analysis.
#
# The returned ``signal`` is an `xarray.DataArray
# <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
# containing the TCSPC histogram as a numpy array, and metadata as a
# dictionary in the ``attrs`` property.
# The metadata includes all PTU tags and the fundamental laser frequency,
# which is needed to interpret the phasor coordinates.
# The reference phase and modulation previously loaded from the LIF-FLIM file
# is again used to calibrate the phasor coordinates. The same intensity
# threshold is applied:

frequency = signal.attrs['frequency']
assert frequency == attrs['frequency']  # frequency matches LIF metadata

mean, real, imag = phasor_from_signal(signal)

real, imag = phasor_transform(
    real, imag, -math.radians(reference_phase), 1 / reference_modulation
)

mean, real, imag = phasor_threshold(mean, real, imag, mean_min=intensity_min)

plot_phasor(
    real,
    imag,
    frequency=frequency,
    title=f'{filename} ({frequency} MHz)',
    cmin=10,
)

# %%
# Compare the apparent single lifetimes calculated from the PTU with the
# lifetimes previously read from the LIF-FLIM file:

plot_histograms(
    phasor_to_apparent_lifetime(real, imag, frequency)[0],
    phase_lifetime,
    range=(0, 10),
    bins=100,
    alpha=0.66,
    title='Lifetime histograms',
    xlabel='Lifetime (ns)',
    ylabel='Counts',
    labels=['Phase lifetime from PTU', 'Phase lifetime from LIF'],
)

# %%
# Zeiss CZI
# ---------
#
# Carl Zeiss image files (CZI) are written by Zeiss ZEN software.
# They contain images and metadata from a variety of microscopy acquisition
# and analysis modes, including hyperspectral imaging.
# PhasorPy does not currently support reading CZI files.
# However, hyperspectral images can be read from CZI files using, for example,
# the `pylibCZIrw  <https://github.com/ZEISS/pylibczirw/>`_ or
# `BioIO <https://github.com/bioio-devs/bioio>`_ libraries.

# %%
# Zeiss LSM
# ---------
#
# .. todo::
#   Read hyperspectral image stack from Zeiss LSM file.

# %%
# Becker & Hickl SDT
# ------------------
#
# .. todo::
#   Read TCSPC histogram from Becker & Hickl SDT file.

# %%
# FLIMbox FBD
# -----------
#
# .. todo::
#   Read TCSPC histogram from FLIMbox FBD file.

# %%
# FLIM LABS JSON
# --------------
#
# .. todo::
#   Read TCSPC histogram from FLIM LABS JSON file.

# %%
# ISS VistaVision IFLI
# --------------------
#
# .. todo::
#   Read phasor coordinates from ISS VistaVision IFLI file.

# %%
# SimFCS REF and R64
# ------------------
#
# .. todo::
#   Read and write phasor coordinates from and to SimFCS referenced files.

# %%
# PhasorPy OME-TIFF
# -----------------
#
# .. todo::
#  Read and write phasor coordinates from and to PhasorPy OME-TIFF files.

# %%
# sphinx_gallery_thumbnail_number = 3
# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
