"""
File input/output
=================

Read and write phasor-related data from and to various file formats.

The :py:mod:`phasorpy.io` module provides functions to read phasor
coordinates, FLIM/TCSPC histograms, hyperspectral image stacks, lifetime
images, and relevant metadata from various file formats used in bio-imaging.
The module also includes functions to write phasor coordinates to OME-TIFF
and SimFCS Referenced files.

"""

# %%
# Import required modules and functions:

import math
import os
from tempfile import TemporaryDirectory

import numpy
from numpy.testing import assert_allclose

from phasorpy.phasor import (
    phasor_calibrate,
    phasor_filter_median,
    phasor_from_signal,
    phasor_threshold,
    phasor_to_apparent_lifetime,
    phasor_transform,
)
from phasorpy.plot import (
    plot_histograms,
    plot_phasor,
    plot_phasor_image,
    plot_signal_image,
)

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

from phasorpy.datasets import fetch  # isort: skip

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
#
# PhasorPy currently supports reading hyperspectral images,
# phasor coordinates, and lifetime images from Leica image files via the
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
# NumPy arrays. ``attrs`` is a dictionary holding metadata, including the
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
#
# PhasorPy supports reading TCSPC histograms from PTU files acquired in T3
# imaging mode via the `ptufile <https://github.com/cgohlke/ptufile/>`_
# library.
#
# The :py:func:`phasorpy.io.signal_from_ptu` function is used to read
# the TCSPC histogram from a PTU file exported from the `FLIM_testdata
# <https://dx.doi.org/10.6084/m9.figshare.22336594.v1>`_ dataset with the
# Leica LAS X software.
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
# holding the TCSPC histogram as a NumPy array, and metadata as a
# dictionary in the ``attrs`` property.
# The metadata includes all PTU tags and the fundamental laser frequency,
# which is needed to interpret the phasor coordinates.
# The reference phase and modulation previously loaded from the LIF-FLIM file
# are again used to calibrate the phasor coordinates. The same intensity
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
# Compare the apparent single lifetimes calculated from the PTU file with the
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
#
# PhasorPy does not currently support reading CZI files.
# However, hyperspectral images can be read from CZI files using, for example,
# the `pylibCZIrw  <https://github.com/ZEISS/pylibczirw/>`_ or
# `BioIO <https://github.com/bioio-devs/bioio>`_ libraries. Another option is
# to export CZI files as LSM using the ZEN software.

# %%
# Zeiss LSM
# ---------
#
# Carl Zeiss LSM files, a predecessor of the CZI format, are written by
# Zeiss ZEN software. They contain images and metadata from laser-scanning
# microscopy.
#
# PhasorPy supports reading hyperspectral image data from Zeiss LSM files
# via the `tifffile <https://github.com/cgohlke/tifffile/>`_ library.
#
# The :py:func:`phasorpy.io.signal_from_lsm` function is used to read
# a hyperspectral dataset with 30 emission wavelengths:

from phasorpy.io import signal_from_lsm

filename = 'paramecium.lsm'
signal = signal_from_lsm(fetch('paramecium.lsm'))

plot_signal_image(signal, title=filename)

# %%
# Note that the signal is not well sampled and shows discontinuity.
# It may not be accurately represented by the phasor coordinates calculated
# via DFT.
#
# Emission wavelengths (in nm) are available in the coordinates of the
# channel axis:

print(signal.coords['C'].values.astype(int))

# %%
# Plot the first harmonic phasor coordinates after applying a median filter.

plot_phasor(
    *phasor_threshold(
        *phasor_filter_median(*phasor_from_signal(signal)), mean_min=1
    )[1:],
    allquadrants=True,
    title=filename,
)

# %%
# Becker & Hickl SDT
# ------------------
#
# SDT files are written by Becker & Hickl software.
# They may contain TCSPC histograms and metadata from laser-scanning
# microscopy.
#
# PhasorPy supports reading TCSPC histograms from FBD files via the
# `lfdfiles <https://github.com/cgohlke/lfdfiles/>`_ library.
#
# The :py:func:`phasorpy.io.signal_from_sdt` function is used to read a
# TCSPC histogram from a SDT file:

from phasorpy.io import signal_from_sdt

filename = 'tcspc.sdt'
signal = signal_from_sdt(fetch(filename))

plot_signal_image(signal, title=filename)

# %%
# Plot the uncalibrated phasor coordinates:

frequency = signal.attrs['frequency']

plot_phasor(
    *phasor_from_signal(signal)[1:],
    title=f'{filename} ({frequency:.1f} MHz)',
    allquadrants=True,
    style='hist2d',
)

# %%
# .. todo::
#   No accompanying IRF dataset is available to calibrate the phasor
#   coordinates.

# %%
# FLIMbox FBD
# -----------
#
# FLIMbox data files, FBD, are written by SimFCS and ISS software.
# They contain encoded TCSPC lifetime histograms from digital frequency-domain
# measurements acquired with the FLIMbox device. Newer file versions
# contain metadata. The file format is undocumented, not standardized,
# and files are frequently found corrupted. It is recommended to export
# FLIMbox data to another format from the software used to acquire the data.
#
# PhasorPy supports reading some FLIMbox FBD files via the
# `lfdfiles <https://github.com/cgohlke/lfdfiles/>`_ library.
#
# The :py:func:`phasorpy.io.signal_from_fbd` function is used to read
# a TCSPC lifetime histograms from the
# `Convallaria <https://zenodo.org/records/14026720>`_ dataset, which was
# acquired at the second harmonic. The dataset is a time series of two
# channels. Since the photon count is low and the second channel empty,
# only the first channel is read and the time-axis integrated:

from phasorpy.io import signal_from_fbd

filename = 'Convallaria_$EI0S.fbd'
signal = signal_from_fbd(fetch(filename), frame=-1, channel=0)

frequency = signal.attrs['frequency'] * signal.attrs['harmonic']
print(signal.sizes)

plot_signal_image(signal, title=filename)

# %%
# The measurement of a solution of Rhodamine 110 with known lifetime of 4 ns
# is used as a calibration reference:

reference_filename = 'Calibration_Rhodamine110_$EI0S.fbd'
reference_signal = signal_from_fbd(
    fetch(reference_filename), frame=-1, channel=0
)
reference_lifetime = 4.0

plot_signal_image(reference_signal, title=reference_filename)

# %%
# Phasor coordinates are calculated from the signal and calibrated with
# the reference signal:

mean, real, imag = phasor_from_signal(signal)

real, imag = phasor_calibrate(
    real,
    imag,
    *phasor_from_signal(reference_signal),
    frequency,
    reference_lifetime,
)

plot_phasor(
    *phasor_threshold(
        *phasor_filter_median(mean, real, imag, repeat=3), mean_min=1
    )[1:],
    title=f'{filename} ({frequency} MHz)',
    frequency=frequency,
)

# %%
# FLIM LABS JSON
# --------------
#
# FLIM LABS JSON files are written by FLIM Studio software.
# They contain multi-channel TCSPC histogram images, optional multi-harmonic
# calibrated phasor coordinates, and metadata from digital frequency-domain
# measurements.
#
# .. todo::
#   The FLIM LABS JSON datasets currently available are not suitable to
#   demonstrate the :py:func:`phasorpy.io.signal_from_flimlabs_json` and
#   :py:func:`phasorpy.io.phasor_from_flimlabs_json` functions.

# %%
# ISS IFLI
# --------
#
# IFLI files are written by ISS VistaVision software. They contain calibrated
# phasor coordinates from analog or digital frequency-domain fluorescence
# lifetime measurements.
# IFLI datasets can be up to 9-dimensional in the order of mean/real/imag,
# frequency, position, emission wavelength, time, channel, Z, Y, and X.
#
# PhasorPy supports reading ISS IFLI files via the
# `lfdfiles <https://github.com/cgohlke/lfdfiles/>`_ library.
#
# The :py:func:`phasorpy.io.phasor_from_ifli` function is used to read
# calibrated phasor coordinates from a measurement of mouse liver fed with
# Western diet. The second channel and the first three harmonics are
# selected:

from phasorpy.io import phasor_from_ifli

filename = 'NADHandSHG.ifli'
mean, real, imag, attrs = phasor_from_ifli(
    fetch(filename), channel=1, harmonic='all'
)

plot_phasor_image(mean, real, imag, title=filename)

# %%
# The first channel in the file contains an SHG image with phasor coordinates
# about zero:

assert (
    phasor_from_ifli(fetch(filename), channel=0, harmonic='all')[1].mean()
    < 1e-2
)

# %%
# The ``attrs`` dictionary holds metadata including the fundamental laser
# frequency, the harmonics in the phasor coordinates, and the reference phasor
# coordinates used for calibration:

frequency = attrs['frequency']
harmonic = attrs['harmonic']
reference_phasor = attrs['ifli_header']['RefDCPhasor']
reference_lifetime = attrs['ifli_header']['RefLifetime']

# %%
# Plot the first harmonic phasor coordinates after applying a median filter:

plot_phasor(
    *phasor_filter_median(mean, real[0], imag[0], repeat=2)[1:],
    title=f'{filename} ({frequency:.2f} MHz)',
    frequency=frequency,
    cmin=10,
)

# %%
# Three main lifetime components are expected in the sample:
# free NADH (~0.4 ns), bound NADH (3.4 ns) and a long lifetime species (~8 ns).
# It appears the calibration is off in this sample.

# %%
# SimFCS REF and R64
# ------------------
#
# Referenced files, REF and R64, are written by the SimFCS software and
# supported in several other software.
# The files most commonly contain a square-sized average intensity image and
# the calibrated phasor coordinates of the first two harmonics.
#
# PhasorPy supports reading and writing SimFCS Referenced files via the
# `lfdfiles <https://github.com/cgohlke/lfdfiles/>`_ library.
#
# The :py:func:`phasorpy.io.phasor_from_simfcs_referenced` function is used
# to read calibrated phasor coordinates from a REF file
# from the `LFD workshop <https://zenodo.org/records/8411056>`_ dataset:

from phasorpy.io import phasor_from_simfcs_referenced

filename = 'capillaries1001.ref'
mean, real, imag, attrs = phasor_from_simfcs_referenced(
    fetch(filename), harmonic='all'
)

plot_phasor_image(mean, real, imag, title=filename)

# %%
# Plot the first harmonic phasor coordinates after applying a median filter.
# SimFCS Referenced files do not contain metadata. The frequency and harmonics
# must be known by the user:

frequency = 80.0  # MHz
harmonic = [1, 2]

plot_phasor(
    *phasor_threshold(
        *phasor_filter_median(mean, real[0], imag[0]),
        mean_min=25,
        real_min=1e-3,
    )[1:],
    title=f'{filename} ({frequency:.2f} MHz)',
    frequency=frequency,
    cmin=2,
)

# %%
# The :py:func:`phasorpy.io.phasor_to_simfcs_referenced` function is used
# to write calibrated phasor coordinates to R64 files in a temporary directory.
# Images with more than two dimensions or larger than square size are
# chunked to square-sized images and saved to separate files.
# Images or chunks with less than two dimensions or smaller than square size
# are padded with NaN values:

from phasorpy.io import phasor_to_simfcs_referenced

with TemporaryDirectory() as tmpdir:

    fname = os.path.join(tmpdir, 'capillaries1001.r64')
    phasor_to_simfcs_referenced(fname, mean, real, imag, size=160)

    # print file names
    filenames = sorted(os.listdir(tmpdir))
    for fname in filenames:
        print(fname)

    # verify the first harmonic phasor coordinates in the last file
    assert_allclose(
        phasor_from_simfcs_referenced(os.path.join(tmpdir, filenames[-1]))[1],
        numpy.pad(real[0, 160:, 160:], (0, 64), constant_values=numpy.nan),
        atol=1e-3,
        equal_nan=True,
    )

# %%
# PhasorPy OME-TIFF
# -----------------
#
# PhasorPy can store phasor coordinates and select metadata in
# `OME-TIFF <https://ome-model.readthedocs.io/en/stable/ome-tiff/>`_
# formatted files, which are compatible with Bio-Formats, Fiji, and other
# software. The implementation is based on the
# `tifffile <https://github.com/cgohlke/tifffile/>`_ library.
#
# In comparison with the SimFCS R64 format, OME-TIFF can store higher
# dimensional, higher precision images of any size, any number of harmonics,
# and select metadata.
#
# PhasorPy OME-TIFF files are intended for temporarily exchanging phasor
# coordinates with other software, not as a long-term storage solution.
# It is best practice to archive the original data files in the native format.
#
# The :py:func:`phasorpy.io.phasor_to_ometiff` and
# :py:func:`phasorpy.io.phasor_from_ometiff` functions are used to write and
# read back calibrated phasor coordinates to/from PhasorPy OME-TIFF files:

from phasorpy.io import phasor_from_ometiff, phasor_to_ometiff

filename = f'{filename}.ome.tiff'

with TemporaryDirectory() as tmpdir:

    phasor_to_ometiff(
        filename,
        mean,
        real,
        imag,
        frequency=frequency,
        dims='YX',
        description='Written by PhasorPy',
    )

    mean1, real1, imag1, attrs = phasor_from_ometiff(filename, harmonic='all')
    assert_allclose(mean, mean1)
    assert attrs['frequency'] == frequency
    assert attrs['harmonic'] == [1, 2]
    assert attrs['description'] == 'Written by PhasorPy'

# %%
# Other means
# -----------
#
# While PhasorPy provides many functions to read phasor related data and
# metadata from file formats commonly used in the field, it is by no means
# required to use those functions.
# Instead, any other means that yields image stacks in numpy-array compatible
# form can be used (for example) for advanced use cases, or when a file
# format is not supported by PhasorPy.
#
# For example, most imaging software can export image data to generic
# TIFF files.
# The `tifffile <https://github.com/cgohlke/tifffile/>`_ library is used to
# read a TCSPC histogram exported to a TIFF file by ImSpector software:

from tifffile import imread

filename = 'Embryo.tif'

image_stack = imread(fetch(filename))

# %%
# Since the image stack array contains no domain-specific metadata, the
# fundamental frequency and the axis over which to calculate phasor
# coordinates must be known. In this case, the TCSPC histogram bins are in
# the first array dimension:

plot_signal_image(image_stack, axis=0, title=filename)

mean, real, imag = phasor_from_signal(image_stack, axis=0)

# %%
# Plot the uncalibrated phasor coordinates:

plot_phasor(real, imag, frequency=80.0, allquadrants=True, title=filename)

# %%
# sphinx_gallery_thumbnail_number = 3
# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
