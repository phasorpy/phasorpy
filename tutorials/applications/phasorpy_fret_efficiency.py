"""
FRET efficiency image
=====================

Estimate FRET efficiencies in an image using a phasor-based model.

The :py:func:`phasorpy.phasor.phasor_from_fret_donor` function is used to
calculate a theoretical FRET efficiency trajectory and the
:py:func:`phasorpy.phasor.phasor_nearest_neighbor` function is then used
to estimate the FRET efficiencies of measured phasor coordinates in an image
from the trajectory.

"""

# %%
# Import required modules, functions, and classes:

import numpy

from phasorpy.datasets import fetch
from phasorpy.io import phasor_from_simfcs_referenced
from phasorpy.phasor import (
    phasor_filter_median,
    phasor_from_fret_donor,
    phasor_nearest_neighbor,
    phasor_threshold,
    phasor_to_normal_lifetime,
)
from phasorpy.plot import (
    PhasorPlot,
    plot_histograms,
    plot_image,
    plot_phasor_image,
)

# %%
# Read dataset
# ------------
#
# Read phasor coordinates from the
# `LFD workshop dataset <https://zenodo.org/records/8411056>`_,
# containing fixed samples of CFP and CFP-YFP expressing cells.
# The data were acquired using a Lambert frequency-domain FLIM instrument
# at 80 MHz. The phasor coordinates are already referenced:

filename = 'CFP and CFP-YFp.ref'
frequency = 80.0  # MHz
mean, real, imag, attrs = phasor_from_simfcs_referenced(fetch(filename))

plot_phasor_image(mean, real, imag, title=filename)

# Filter the phasor coordinates and set the intensity threshold
# to 9000 counts to remove background:

mean, real, imag = phasor_filter_median(mean, real, imag, repeat=2)
mean, real, imag = phasor_threshold(
    mean, real, imag, mean_min=9000, real_min=0, imag_min=0, open_interval=True
)

# %%
# FRET efficiency trajectory
# --------------------------
#
# Calculate a theoretical FRET efficiency trajectory of phasor coordinates
# for the CFP FRET donor:

donor_real, donor_imag = 0.72, 0.45  # estimated pure CFP phasor coordinates
background_real, background_imag = 0.6, 0.41  # estimated background phasor

fret_efficiency_range = numpy.linspace(0.0, 1.0, 100)
donor_lifetime = phasor_to_normal_lifetime(donor_real, donor_imag, frequency)

fret_trajectory = phasor_from_fret_donor(
    frequency,
    donor_lifetime,
    fret_efficiency=fret_efficiency_range,
    donor_fretting=1.0,
    donor_background=0.1,  # 10% background signal
    background_real=background_real,
    background_imag=background_imag,
)

phasor_plot = PhasorPlot(frequency=frequency, xlim=(0.5, 1), ylim=(0.2, 0.6))
phasor_plot.hist2d(real, imag)
phasor_plot.line([donor_real, background_real], [donor_imag, background_imag])
phasor_plot.plot(
    donor_real,
    donor_imag,
    'o',
    label='Donor only',
    color='tab:green',
    markeredgecolor='black',
    markersize=10,
    zorder=10,
)
phasor_plot.plot(
    background_real,
    background_imag,
    'o',
    label='Background',
    color='black',
    markersize=10,
    zorder=10,
)
phasor_plot.plot(
    *fret_trajectory,
    '-',
    label='FRET trajectory',
    color='tab:orange',
    lw=4,
    alpha=0.8,
)
phasor_plot.show()

# %%
# Estimate FRET efficiency
# ------------------------
#
# Estimate FRET efficiencies for each pixel in the image by finding
# the closest phasor coordinates in the FRET efficiency trajectory:

fret_efficiencies = phasor_nearest_neighbor(
    real,
    imag,
    *fret_trajectory,
    values=fret_efficiency_range,
    dtype=real.dtype,
    num_threads=4,
)

# %%
# Visualize the spatial distribution of FRET efficiencies:

plot_image(fret_efficiencies, title='Estimated FRET efficiency')

# %%
# Visualize the distribution of FRET efficiencies as a histogram:

plot_histograms(
    fret_efficiencies * 100,  # convert to percentage
    title='FRET efficiency histogram',
    xlabel='FRET efficiency (%)',
    ylabel='Counts',
    range=(0, 35),
    bins=35,
)

# %%
# sphinx_gallery_thumbnail_number = -2
# mypy: allow-untyped-defs, allow-untyped-calls
