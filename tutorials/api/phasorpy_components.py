"""
Component analysis
==================

An introduction to component analysis in phasor space.

"""

# %%
# Import required modules, functions, and classes:

import math

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy

from phasorpy.components import (
    graphical_component_analysis,
    phasor_based_unmixing,
    two_fractions_from_phasor,
)
from phasorpy.datasets import fetch
from phasorpy.io import signal_from_lsm
from phasorpy.phasor import (
    phasor_center,
    phasor_filter_median,
    phasor_from_lifetime,
    phasor_from_signal,
    phasor_threshold,
)
from phasorpy.plot import PhasorPlot

numpy.random.seed(42)
component_style = {
    'linestyle': '-',
    'marker': 'o',
    'color': 'tab:blue',
    'fontsize': 14,
}

# %%
# Fractions of two components
# ---------------------------
#
# The phasor coordinates of combinations of two lifetime components lie on
# the line between the two components. For example, a combination with
# 60% contribution (fraction 0.6) of a component A with lifetime 1.0 ns and
# 40% contribution (fraction 0.4) of a component B with lifetime 8.0 ns
# at 80 MHz:

frequency = 80.0
component_lifetimes = [1.0, 8.0]
component_fractions = [0.6, 0.4]

components_real, components_imag = phasor_from_lifetime(
    frequency, component_lifetimes
)

plot = PhasorPlot(frequency=frequency, title='Combination of two components')
plot.components(
    components_real,
    components_imag,
    component_fractions,
    labels=['A', 'B'],
    **component_style,
)
plot.show()

# %%
# If the location of both components is known, their contributions (fractions)
# to the phasor point that lies on the line between the components
# can be calculated:

real, imag = phasor_from_lifetime(
    frequency, component_lifetimes, component_fractions
)

fraction_of_first_component = two_fractions_from_phasor(
    real, imag, components_real, components_imag
)

assert math.isclose(fraction_of_first_component, component_fractions[0])

# %%
# Distribution of fractions of two components
# -------------------------------------------
#
# Phasor coordinates can have different contributions of two components with
# known phasor coordinates:

real, imag = numpy.random.multivariate_normal(
    (real, imag), [[5e-3, 1e-3], [1e-3, 1e-3]], (100, 100)
).T

plot = PhasorPlot(
    frequency=frequency, title='Distribution of fractions of two components'
)
plot.hist2d(real, imag, cmap='Greys')
plot.components(
    components_real, components_imag, labels=['A', 'B'], **component_style
)
plot.show()

# %%
# If the phasor coordinates of two components contributing to multiple
# phasor coordinates are known, their fractional contributions to each phasor
# coordinate can be calculated by projecting the phasor coordinate onto
# the line between the components. The fractions are plotted as histograms:

fraction_of_first_component = two_fractions_from_phasor(
    real, imag, components_real, components_imag
)

fig, ax = plt.subplots()
ax.hist(
    fraction_of_first_component.flatten(),
    range=(0, 1),
    bins=100,
    alpha=0.75,
    label='A',
)
ax.hist(
    1.0 - fraction_of_first_component.flatten(),
    range=(0, 1),
    bins=100,
    alpha=0.75,
    label='B',
)
ax.set_title('Histograms of fractions of two components')
ax.set_xlabel('Fraction')
ax.set_ylabel('Counts')
ax.legend()
plt.tight_layout()
plt.show()

# %%
# Graphical solution for contributions of two components
# ------------------------------------------------------
#
# The :py:func:`phasorpy.components.graphical_component_analysis`
# function for two components counts the number of phasor coordinates
# that fall within a radius at given fractions along the line between
# the components.
# Compare the plot of counts vs fraction to the previous histogram:

radius = 0.025
fractions = numpy.linspace(0.0, 1.0, 20)

counts = graphical_component_analysis(
    real,
    imag,
    components_real,
    components_imag,
    fractions=fractions,
    radius=radius,
)

fig, ax = plt.subplots()
ax.plot(fractions, counts[0], '-', label='A vs B')
ax.set_title('Graphical solution for contributions of two components')
ax.set_xlabel('Fraction')
ax.set_ylabel('Counts')
ax.legend()
plt.show()

# %%
# Graphical solution for contributions of three components
# --------------------------------------------------------
#
# The graphical solution can similarly be applied to the contributions of
# three components.

component_lifetimes = [1.0, 4.0, 15.0]
components_real, components_imag = phasor_from_lifetime(
    frequency, component_lifetimes
)

plot = PhasorPlot(
    frequency=frequency,
    title='Distribution of three known components',
)
plot.hist2d(real, imag, cmap='Greys')
plot.components(
    components_real, components_imag, labels=['A', 'B', 'C'], **component_style
)
plot.show()

# %%
# The results of the graphical component analysis are plotted as
# histograms for each component pair:

counts = graphical_component_analysis(
    real,
    imag,
    components_real,
    components_imag,
    fractions=fractions,
    radius=radius,
)

fig, ax = plt.subplots()
ax.plot(fractions, counts[0], '-', label='A vs B')
ax.plot(fractions, counts[1], '-', label='A vs C')
ax.plot(fractions, counts[2], '-', label='B vs C')
ax.set_title('Graphical solution for contributions of three components')
ax.set_xlabel('Fraction')
ax.set_ylabel('Counts')
ax.legend()
plt.show()

# %%
# The graphical method for resolving the contribution of three components
# (pairwise) to a phasor coordinate is based on the quantification of moving
# circular cursors along the line between the components, demonstrated in the
# following animation for component A vs B.
# For the full analysis, the process is repeated for the other combinations
# of components, A vs C and B vs C:

fig, (ax, hist) = plt.subplots(nrows=2, ncols=1, figsize=(5.5, 8))

plot = PhasorPlot(
    frequency=frequency,
    ax=ax,
    title='Graphical solution for contribution of A vs B',
)
plot.hist2d(real, imag, cmap='Greys')
plot.components(
    components_real[:2],
    components_imag[:2],
    labels=['A', 'B'],
    **component_style,
)
plot.components(
    components_real[2], components_imag[2], labels=['C'], **component_style
)

hist.set_xlim(0, 1)
hist.set_xlabel('Fraction')
hist.set_ylabel('Counts')

direction_real = components_real[0] - components_real[1]
direction_imag = components_imag[0] - components_imag[1]

plots = []
for i in range(fractions.size):
    cursor_real = components_real[1] + fractions[i] * direction_real
    cursor_imag = components_imag[1] + fractions[i] * direction_imag
    plot_lines = plot.plot(
        [cursor_real, components_real[2]],
        [cursor_imag, components_imag[2]],
        '-',
        linewidth=plot.dataunit_to_point * radius * 2 + 5,
        solid_capstyle='round',
        color='red',
        alpha=0.5,
    )
    hist_artists = plt.plot(
        fractions[: i + 1], counts[0][: i + 1], linestyle='-', color='tab:blue'
    )
    plots.append(plot_lines + hist_artists)

_ = animation.ArtistAnimation(fig, plots, interval=100, blit=True)
plt.tight_layout()
plt.show()


# %%
# Theoretical solution for contributions of n components
# ------------------------------------------------------
#
# The theoretical solution can be applied to the contributions
# of n components

# %%

# List of components and corresponding image files
components = ['Hoechst', 'Lyso Tracker', 'Golgi', 'Mito Tracker', 'CellMask']
image_files = [
    'spectral hoehst.lsm',
    'spectral lyso tracker green.lsm',
    'spectral golgi.lsm',
    'spectral mito tracker.lsm',
    'spectral cell mask.lsm',
]

# Load images efficiently
images = [signal_from_lsm(fetch(img)) for img in image_files]

# Initialize results array
results = numpy.zeros((5, 4))

# Create PhasorPlots for both harmonics
plot1 = PhasorPlot(
    allquadrants=True, title='Components: First Harmonic Phasor Plot'
)
plot2 = PhasorPlot(
    allquadrants=True, title='Components: Second Harmonic Phasor Plot'
)


# Function to process an image and extract phasor centers
def process_image(img):
    avg, real, imag = phasor_from_signal(img, axis=0, harmonic=[1, 2])
    avg, real, imag = phasor_filter_median(avg, real, imag, size=5, repeat=3)
    avg, real, imag = phasor_threshold(avg, real, imag, mean_min=3)

    return avg, real, imag


# Process each image and plot the phasor centers
for i, img in enumerate(images):
    avg, real, imag = process_image(img)

    # Compute phasor centers for both harmonics
    _, real_center_h1, imag_center_h1 = phasor_center(avg, real[0], imag[0])
    _, real_center_h2, imag_center_h2 = phasor_center(avg, real[1], imag[1])

    # Plot first harmonic
    plot1.hist2d(real[0], imag[0], cmap='RdYlBu_r', bins=300)
    plot1.plot(
        real_center_h1,
        imag_center_h1,
        marker='o',
        markersize=8,
        label=components[i],
    )

    # Plot second harmonic
    plot2.hist2d(real[1], imag[1], cmap='RdYlBu_r', bins=300)
    plot2.plot(
        real_center_h2,
        imag_center_h2,
        marker='o',
        markersize=8,
        label=components[i],
    )

    # Store results
    results[i] = [
        real_center_h1,
        imag_center_h1,
        real_center_h2,
        imag_center_h2,
    ]

# Load test image and process it
test_image = signal_from_lsm(
    fetch('38_Hoechst_Golgi_Mito_Lyso_CellMAsk_404_488_561_633_SP.lsm')
)
avg, real, imag = phasor_from_signal(test_image, axis=0, harmonic=[1, 2])
avg, real, imag = phasor_filter_median(avg, real, imag, size=5, repeat=3)

# Reshape results into matrices for plotting
first_h, second_h = results[:, :2], results[:, 2:]

# Plot the average image
plt.figure()
plt.imshow(avg, cmap='gray')
plt.title('Average Image')
plt.axis('off')

# Plot first and second harmonic phasors
for title, real_data, imag_data, centers in zip(
    ["First Harmonic Phasor", "Second Harmonic Phasor"],
    [real[0], real[1]],
    [imag[0], imag[1]],
    [first_h, second_h],
):
    plot = PhasorPlot(allquadrants=True, title=title)
    plot.hist2d(real_data, imag_data, cmap='RdYlBu_r', bins=300)
    plot.plot(centers[:, 0], centers[:, 1])

# Create the matrix A with pure component positions
matrixA = numpy.vstack(
    [
        results[:, 0],  # First harmonic real
        results[:, 2],  # Second harmonic real
        results[:, 1],  # First harmonic imaginary
        results[:, 3],  # Second harmonic imaginary
        numpy.ones(5),  # Row of ones
    ]
)

# Perform unmixing
fractions = numpy.asarray(
    phasor_based_unmixing(real, imag, matrixA)
)
fractions = fractions.reshape(5, real.shape[1], real.shape[2])

# Normalize fractions (clip negatives and scale to 0-1)
fractions = numpy.clip(fractions, 0, None)
fractions /= numpy.max(fractions)

# Plot the fractions of each component
cmap = plt.cm.inferno
for i, fraction in enumerate(fractions):
    plt.figure()
    plt.imshow(fraction, cmap=cmap, vmin=0, vmax=1)
    plt.title(f'Fraction of component {components[i]}')
    plt.colorbar()
    plt.tight_layout()

plt.show()

# %%
# sphinx_gallery_thumbnail_number = 5
# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
