"""
FRET efficiency image
=====================

Estimate FRET efficiencies in an image using a phasor-based model.

Förster resonance energy transfer (FRET) is a distance-dependent interaction
between two luminescing molecules where energy is transferred from a donor
to an acceptor molecule. FRET efficiency is sensitive to donor-acceptor
distances, making it useful for studying molecular interactions.

The :py:func:`phasorpy.lifetime.phasor_from_fret_donor` function is used to
calculate a theoretical FRET efficiency trajectory and the
:py:func:`phasorpy.phasor.phasor_nearest_neighbor` function is then used
to estimate the FRET efficiencies of measured phasor coordinates in an image
from the trajectory.

"""

# %%
# Import required modules, functions, and classes:

import numpy

from phasorpy.datasets import fetch
from phasorpy.filter import phasor_filter_median, phasor_threshold
from phasorpy.io import phasor_from_simfcs_referenced
from phasorpy.lifetime import phasor_from_fret_donor, phasor_to_normal_lifetime
from phasorpy.phasor import phasor_nearest_neighbor
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
# to 9000 counts to remove low-intensity background pixels:

mean, real, imag = phasor_filter_median(mean, real, imag, repeat=2)
mean, real, imag = phasor_threshold(
    mean, real, imag, mean_min=9000, real_min=0, imag_min=0, open_interval=True
)

# %%
# FRET efficiency trajectory
# --------------------------
#
# Calculate the theoretical FRET efficiency trajectory in phasor space
# for the CFP FRET donor.
# The trajectory represents the path phasor coordinates follow as FRET
# efficiency increases from 0% (donor only) to 100% (complete energy transfer):

donor_real, donor_imag = 0.72, 0.45  # estimated pure CFP phasor coordinates
background_real, background_imag = 0.6, 0.41  # estimated background phasor

fret_efficiency_range = numpy.linspace(0.0, 1.0, 100)
donor_lifetime = phasor_to_normal_lifetime(donor_real, donor_imag, frequency)

fret_trajectory = phasor_from_fret_donor(
    frequency,
    donor_lifetime,
    fret_efficiency=fret_efficiency_range,
    donor_fretting=1.0,  # all donor molecules can undergo FRET
    donor_background=0.1,  # 10% background signal contribution
    background_real=background_real,
    background_imag=background_imag,
)

phasor_plot = PhasorPlot(
    frequency=frequency,
    xlim=(0.5, 1),
    ylim=(0.2, 0.6),
    title='FRET efficiency trajectory',
)
phasor_plot.hist2d(real, imag)
phasor_plot.line([donor_real, background_real], [donor_imag, background_imag])
phasor_plot.plot(
    donor_real,
    donor_imag,
    'o',
    color='tab:green',
    markeredgecolor='black',
    markersize=10,
    zorder=10,
    label='Donor only',
)
phasor_plot.plot(
    background_real,
    background_imag,
    'o',
    color='black',
    markersize=10,
    zorder=10,
    label='Background',
)
phasor_plot.plot(
    *fret_trajectory,
    '-',
    color='tab:orange',
    linewidth=4,
    alpha=0.8,
    label='FRET trajectory',
)
phasor_plot.show()

# %%
# Estimate FRET efficiency
# ------------------------
#
# Estimate FRET efficiencies for each pixel in the image by finding
# the nearest point on the FRET efficiency trajectory to each measured
# phasor coordinate:

fret_efficiencies = phasor_nearest_neighbor(
    real,
    imag,
    *fret_trajectory,
    values=fret_efficiency_range,
    dtype=real.dtype,
    num_threads=4,  # use multiple threads for faster computation
)

# %%
# Visualize the spatial distribution of FRET efficiencies:

plot_image(fret_efficiencies, title='Estimated FRET efficiency')

# %%
# Visualize the distribution of FRET efficiencies as a histogram:

plot_histograms(
    fret_efficiencies * 100,  # convert to percentage
    range=(0, 35),
    bins=35,
    xlabel='FRET efficiency (%)',
    ylabel='Counts',
    title='FRET efficiency histogram',
)

# %%
# Conclusions
# -----------
#
# The FRET efficiency image shows spatial heterogeneity in donor-acceptor
# interactions across the sample. Higher FRET efficiencies (warmer colors)
# indicate closer proximity between CFP and YFP molecules, suggesting
# successful FRET pair formation. Lower efficiencies (cooler colors) may
# represent cells expressing primarily CFP donor without acceptor, or
# regions where CFP and YFP are too far apart for efficient energy transfer.
# The histogram reveals the distribution of FRET states in the population.

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = -2
# mypy: allow-untyped-defs, allow-untyped-calls
# sphinx_gallery_end_ignore
