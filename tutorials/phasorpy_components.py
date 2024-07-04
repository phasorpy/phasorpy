"""
Component analysis
==================

An introduction to component analysis in the phasor space.

"""

# %%
# Import required modules, functions, and classes:

import matplotlib.pyplot as plt
import numpy

from phasorpy.components import two_fractions_from_phasor
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
# sphinx_gallery_thumbnail_number = 2
