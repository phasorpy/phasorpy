"""
Multi-component fit
===================

Spectral unmixing using multi-component analysis in phasor space.

The fluorescent components comprising a fluorescence spectrum can be unmixed
if the spectra of the individual components are known.
This can be achieved by solving a system of linear equations, fitting
the fractional contributions of the phasor coordinates of the component spectra
to the phasor coordinates of the mixture spectrum. Phasor coordinates
at multiple harmonics may be used to ensure the linear system is not
underdetermined.

This analysis method is demonstrated using a hyperspectral imaging dataset
containing five fluorescent markers as presented in:

  Vallmitjana A, Lepanto P, Irigoin F, and Malacrida L.
  Phasor-based multi-harmonic unmixing for in-vivo hyperspectral imaging.
  *Methods Appl Fluoresc*, 11(1): 014001 (2022).

The dataset is available at https://zenodo.org/records/13625087.

The spectral unmixing of the five components is performed using phasor
coordinates at two harmonics.

"""

# %%
# Import required modules, functions, and classes:

import numpy
from matplotlib import pyplot

from phasorpy.components import phasor_component_fit
from phasorpy.datasets import fetch
from phasorpy.io import signal_from_lsm
from phasorpy.phasor import (
    phasor_center,
    phasor_filter_median,
    phasor_from_signal,
    phasor_threshold,
)
from phasorpy.plot import PhasorPlot, plot_image, plot_signal_image

# %%
# Define dataset and processing parameters:

# hyperspectral image containing of all five components
samplefile = '38_Hoechst_Golgi_Mito_Lyso_CellMAsk_404_488_561_633_SP.lsm'

# hyperspectral images of individual components
components = {
    'Hoechst': 'spectral hoehst.lsm',
    'LysoTracker': 'spectral lyso tracker green.lsm',
    'Golgi': 'spectral golgi.lsm',
    'MitoTracker': 'spectral mito tracker.lsm',
    'CellMask': 'spectral cell mask.lsm',
}

# analysis parameters
harmonic = [1, 2]  # which harmonics to use for analysis
median_size = 5  # size of median filter window
median_repeat = 3  # number of times to apply median filter
threshold = 3  # minimum signal threshold

# %%
# Individual components
# ---------------------
#
# Calculate and plot phasor coordinates for each component and harmonic.
# For each component:
#
# 1. Load spectral data and calculate phasor coordinates
# 2. Apply median filtering to reduce noise
# 3. Apply threshold to remove low-intensity pixels
# 4. Calculate center coordinates using mean method
# 5. Plot in phasor space

num_harmonics = len(harmonic)
num_components = len(components)
component_real = numpy.zeros((num_harmonics, num_components))
component_imag = numpy.zeros((num_harmonics, num_components))
component_mean = []

fig, axs = pyplot.subplots(
    num_harmonics, 1, figsize=(4.8, num_harmonics * 4.8)
)
fig.suptitle('Components')

for i, (name, filename) in enumerate(components.items()):
    mean, real, imag = phasor_from_signal(
        signal_from_lsm(fetch(filename)), axis=0, harmonic=harmonic
    )
    mean, real, imag = phasor_filter_median(
        mean, real, imag, size=median_size, repeat=median_repeat
    )
    mean, real, imag = phasor_threshold(mean, real, imag, mean_min=threshold)
    mean, center_real, center_imag = phasor_center(
        mean, real, imag, method='mean'
    )
    component_mean.append(mean)
    component_real[:, i] = center_real
    component_imag[:, i] = center_imag

    for j in range(num_harmonics):
        plot = PhasorPlot(
            ax=axs[j], allquadrants=True, title=f'Harmonic {harmonic[j]}'
        )
        plot.hist2d(real[j], imag[j], cmap='Greys')
        plot.plot(center_real[j], center_imag[j], label=name, markersize=10)
        plot.ax.legend(loc='right').set_visible(j == 0)
fig.tight_layout()
fig.show()

# %%
# Component mixture
# -----------------
#
# Read the mixture sample image containing all markers and visualize the
# raw spectral data:

signal = signal_from_lsm(fetch(samplefile))

plot_signal_image(signal, title='Component mixture')

# %%
# Calculate and plot phasor coordinates for the component mixture:

mean, real, imag = phasor_from_signal(signal, axis=0, harmonic=harmonic)
mean, real, imag = phasor_filter_median(
    mean, real, imag, size=median_size, repeat=median_repeat
)
# optional: apply threshold to remove low-intensity pixels
# mean, real, imag = phasor_threshold(mean, real, imag, mean_min=threshold)

fig, axs = pyplot.subplots(
    num_harmonics, 1, figsize=(4.8, num_harmonics * 4.8)
)
fig.suptitle('Component mixture')
for i in range(num_harmonics):
    plot = PhasorPlot(
        ax=axs[i], allquadrants=True, title=f'Harmonic {harmonic[i]}'
    )
    plot.hist2d(real[i], imag[i], cmap='Greys')
    for j, name in enumerate(components):
        plot.plot(
            component_real[i, j],
            component_imag[i, j],
            label=name,
            markersize=10,
        )
    plot.ax.legend(loc='right').set_visible(i == 0)
fig.tight_layout()
fig.show()

# %%
# Fractions of components in mixture
# ----------------------------------
#
# Fit fractions of each component to the phasor coordinates at each pixel
# of the mixture and plot the component fraction images:

fractions = phasor_component_fit(
    mean, real, imag, component_real, component_imag
)

plot_image(
    mean / mean.max(),
    *fractions,
    title='Fractions of components in mixture',
    labels=['Mixture'] + list(components.keys()),
    vmin=0,
    vmax=1,
)

# %%
# Plot the intensity of each component in the mixture:

plot_image(
    mean,
    *(f * mean for f in fractions),
    title='Intensity of components in mixture',
    labels=['Mixture'] + list(components.keys()),
    vmin=0,
    vmax=mean.max(),
)

# %%
# sphinx_gallery_thumbnail_number = -2
# mypy: allow-untyped-defs, allow-untyped-calls
