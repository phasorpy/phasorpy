"""
Component analysis
==================

An introduction to component analysis in the phasor space.

"""

# %%
# Import required modules, functions, and classes:

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy

from phasorpy._utils import line_from_components, mask_segment
from phasorpy.components import (
    graphical_component_analysis,
    two_fractions_from_phasor,
)
from phasorpy.phasor import phasor_from_lifetime
from phasorpy.plot import PhasorPlot

numpy.random.seed(42)

# %%
# Fractions of combination of two components
# ------------------------------------------
#
# The phasor coordinate of a combination of two lifetime components lie on
# the line between the two components. For example, a combination with 25%
# contribution of a component with lifetime 8.0 ns and 75% contribution of
# a second component with lifetime 1.0 ns at 80 MHz:

frequency = 80.0
components_lifetimes = [8.0, 1.0]
component_fractions = [0.25, 0.75]
real, imag = phasor_from_lifetime(
    frequency, components_lifetimes, component_fractions
)
components_real, components_imag = phasor_from_lifetime(
    frequency, components_lifetimes
)
plot = PhasorPlot(frequency=frequency, title='Combination of two components')
plot.plot(components_real, components_imag, fmt='o-')
plot.plot(real, imag)
plot.show()

# %%
# If the location of both components is known, their contributions
# to the phasor point that lies on the line between the components
# can be calculated:

(
    fraction_of_first_component,
    fraction_of_second_component,
) = two_fractions_from_phasor(real, imag, components_real, components_imag)
print(f'Fraction of first component:  {fraction_of_first_component:.3f}')
print(f'Fraction of second component: {fraction_of_second_component:.3f}')

# %%
# Contribution of two known components in multiple phasors
# --------------------------------------------------------
#
# Phasors can have different contributions of two components with known
# phasor coordinates:

real, imag = numpy.random.multivariate_normal(
    (0.6, 0.35), [[8e-3, 1e-3], [1e-3, 1e-3]], (100, 100)
).T
plot = PhasorPlot(
    frequency=frequency,
    title='Phasor with contribution of two known components',
)
plot.hist2d(real, imag, cmap='plasma')
plot.plot(*phasor_from_lifetime(frequency, components_lifetimes), fmt='o-')
plot.show()

# %%
# If the phasor coordinates of two components contributing to multiple
# phasors are known, their fractional contributions to each phasor coordinate
# can be calculated and plotted as histograms:

(
    fraction_from_first_component,
    fraction_from_second_component,
) = two_fractions_from_phasor(real, imag, components_real, components_imag)
fig, ax = plt.subplots()
ax.hist(
    fraction_from_first_component.flatten(),
    range=(0, 1),
    bins=100,
    alpha=0.75,
    label='First',
)
ax.hist(
    fraction_from_second_component.flatten(),
    range=(0, 1),
    bins=100,
    alpha=0.75,
    label='Second',
)
ax.set_title('Histograms of fractions of first and second component')
ax.set_xlabel('Fraction')
ax.set_ylabel('Counts')
ax.legend()
plt.tight_layout()
plt.show()

# %%
# Graphical solution for the contribution of multiple components
# --------------------------------------------------------------
#
# The graphical method for resolving the contribution of two or
# three components (pairwise) to a phasor coordinate is based on
# the quantification of moving circular cursors along the line
# between the components, demonstrated in the following animation:

grid_x = numpy.linspace(-0.2, 1, int((1 + 0.2) / 0.001) + 1)
grid_y = numpy.linspace(-0.2, 0.6, int((0.6 + 0.2) / 0.001) + 1)
grid_x, grid_y = numpy.meshgrid(grid_x, grid_y)
cursor_diameter = 0.05
number_of_steps = 30
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5, 8))
real_a, imag_a = 0.8, 0.4
real_b, imag_b = 0.2, 0.4
real_c, imag_c = 0.042, 0.2
component_counts = []
fractions = numpy.asarray(numpy.linspace(0, 1, number_of_steps + 1))
unit_vector, distance = line_from_components(
    [real_b, real_a], [imag_b, imag_a]
)
cursor_real, cursor_imag = real_b, imag_b
step_size = distance / number_of_steps
plot = PhasorPlot(ax=ax1)
plot.hist2d(real, imag, cmap='plasma')
plot.plot([real_a, real_b], [imag_a, imag_b], fmt='o', linestyle='-')
plot.plot(real_c, imag_c, fmt='o', color='green')
plots = []
for step in range(number_of_steps + 1):
    mask_shape = mask_segment(
        grid_x,
        grid_y,
        cursor_real,
        cursor_imag,
        real_c,
        imag_c,
        cursor_diameter / 2,
    )
    plot.plot(grid_x[mask_shape], grid_y[mask_shape], color='red', alpha=0.01)
    mask_phasors = mask_segment(
        real,
        imag,
        cursor_real,
        cursor_imag,
        real_c,
        imag_c,
        cursor_diameter / 2,
    )
    fraction_counts = numpy.sum(mask_phasors)
    component_counts.append(fraction_counts)
    hist_artists = plt.plot(
        fractions[: step + 1], component_counts, linestyle='-', color='blue'
    )
    plots.append(plot._lines + hist_artists)
    cursor_real += step_size * unit_vector[0]
    cursor_imag += step_size * unit_vector[1]
moving_cursor_animation = animation.ArtistAnimation(
    fig, plots, interval=100, blit=True
)
ax2.set_xlim(0, 1)
ax2.set_title('Fraction of component A (respect to B)')
ax2.set_xlabel('Fraction of A')
ax2.set_ylabel('Counts')
ax1.annotate(
    'A', xy=(0.84, 0.43), fontsize=16, fontweight='bold', color='blue'
)
ax1.annotate(
    'B', xy=(0.17, 0.43), fontsize=16, fontweight='bold', color='blue'
)
ax1.annotate(
    'C', xy=(0.0, 0.23), fontsize=16, fontweight='bold', color='green'
)
plt.tight_layout()
plt.show()

# %%
# The function `graphical_component_analysis` performs this graphical
# analysis for two or three components and return the number of phasors
# for each fraction of the components (with respect to other components):

counts, fractions = graphical_component_analysis(
    real, imag, [real_a, real_b, real_c], [imag_a, imag_b, imag_c]
)

# %%
# The results can be plotted as histograms for each component pair:

fig, ax = plt.subplots()
ax.plot(fractions, counts[0], linestyle='-', label='Component A vs B')
ax.plot(fractions, counts[1], linestyle='-', label='Component A vs C')
ax.plot(fractions, counts[2], linestyle='-', label='Component B vs C')
ax.set_xlabel('Fraction of component')
ax.set_ylabel('Counts')
ax.legend()
plt.show()

# %%
# sphinx_gallery_thumbnail_number = 2
