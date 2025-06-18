"""
FRET efficiency
===============

The FRET efficiency can be estimated from the phasor coordinates
of the donor and acceptor channels by projecting the phasor
coordinates onto the FRET efficiency trajectory.

The :py:func:`phasorpy.phasor.phasor_from_fret_donor` function can be
used to calculate the FRET efficiency trajectory and the
:py:func::py:func:`phasorpy.phasor.phasor_nearest_neighbor` function
can be used to estimate the FRET efficiency from the phasor coordinates.

"""

# %%
# Import required modules, functions, and classes:

import matplotlib.pyplot as plt
import numpy

from phasorpy.datasets import fetch
from phasorpy.io import phasor_from_simfcs_referenced
from phasorpy.phasor import (
    phasor_filter_median,
    phasor_from_fret_donor,
    phasor_nearest_neighbor,
    phasor_threshold,
    phasor_to_apparent_lifetime,
)
from phasorpy.plot import PhasorPlotFret, plot_histograms

# %%
# Load the reference dataset and process the phasor data:

frequency = 80
mean, real, imag, attrs = phasor_from_simfcs_referenced(
    fetch('CFP and CFP-YFp.ref')
)
mean, real, imag = phasor_filter_median(mean, real, imag, repeat=2)
mean, real, imag = phasor_threshold(
    mean, real, imag, mean_min=9000, real_min=0, imag_min=0, open_interval=True
)

# Define parameters for FRET calculation
background_phasor = 0.6, 0.41
donor_phasor = 0.72, 0.45
donor_lifetime = numpy.mean(
    phasor_to_apparent_lifetime(*donor_phasor, frequency)
)

# Configure settings for FRET plot and efficiency calculation
settings = {
    'frequency': frequency,
    'donor_lifetime': donor_lifetime,
    'donor_fretting': 1.0,
    'donor_background': 0.1,
    'background_real': background_phasor[0],
    'background_imag': background_phasor[1],
}

fret_plot = PhasorPlotFret(**settings, xlim=(0.5, 1), ylim=(0.2, 0.6))
fret_plot.hist2d(real, imag)
fret_plot.show()

# %%
# Calculate the FRET efficiency trajectory for the donor phasor:

efficiency_values = numpy.linspace(0.0, 1.0, 1000)

donor_trajectory_real, donor_trajectory_imag = phasor_from_fret_donor(
    **settings, fret_efficiency=efficiency_values
)

# %%
# Estimate and visualize the FRET efficiency values
# from donor phasor coordinates:

trajectory_indices, fret_efficiency_values = phasor_nearest_neighbor(
    mean,
    real,
    imag,
    donor_trajectory_real,
    donor_trajectory_imag,
    values=efficiency_values,
)

fig, ax = plt.subplots()
im = ax.imshow(fret_efficiency_values, cmap='plasma')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('FRET Efficiency')
ax.set_title('FRET Efficiency')
plt.tight_layout()
plt.show()

# %%
# Visualize the FRET efficiency values distribution:

plot_histograms(
    fret_efficiency_values,
    xlabel='FRET Efficiency',
    ylabel='Counts',
    title='FRET Efficiency Histogram',
    bins=100,
)

# %%
# sphinx_gallery_thumbnail_number = -2
# mypy: allow-untyped-defs, allow-untyped-calls
