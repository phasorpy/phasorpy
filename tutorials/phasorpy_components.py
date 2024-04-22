"""
Component analysis
===========

An introduction to component analysis in the phasor space. The
:py:func:`phasorpy.phasor.phasor_from_lifetime` function is used to calculate
phasor coordinates as a function of frequency, single or multiple lifetime
components, and the pre-exponential amplitudes or fractional intensities of the
components.

"""

# %%
# Import required modules, functions, and classes:

import matplotlib.pyplot as plt
import numpy

from phasorpy.components import two_fractions_from_phasor
from phasorpy.phasor import phasor_from_lifetime
from phasorpy.plot import PhasorPlot

# %%
# Fractions of combination of two components
# ------------------
#
# A phasor that lies in the line between two components with 0.25 contribution
# of the first components and 0.75 contribution of the second component:

frequency = 80.0
components_lifetimes = [8.0, 1.0]
real, imag = phasor_from_lifetime(
    frequency, components_lifetimes, [0.25, 0.75]
)
components_real, components_imag = phasor_from_lifetime(
    frequency, components_lifetimes
)
plot = PhasorPlot(
    frequency=frequency, title='Phasor lying on the line between components'
)
plot.plot(components_real, components_imag, fmt='o-')
plot.plot(real, imag)
plot.show()

# %%

# If we know the location of both components, we can compute the contribution
# of both components to the phasor point that lies in the line between the two
# components:

(
    fraction_of_first_component,
    fraction_of_second_component,
) = two_fractions_from_phasor(real, imag, components_real, components_imag)
print('Fraction of first component: ', fraction_of_first_component)
print('Fraction of second component: ', fraction_of_second_component)

# %%
# Contribution of two known components in multiple phasors
# ------------------
#
# Phasors can have different contributions of two components with known phasor
# coordinates:

real, imag = numpy.random.multivariate_normal(
    (0.6, 0.35), [[8e-3, 1e-3], [1e-3, 1e-3]], (100, 100)
).T
plot = PhasorPlot(
    frequency=frequency,
    title='Phasor with contibution of two known components',
)
plot.hist2d(real, imag, cmap='plasma')
plot.plot(*phasor_from_lifetime(frequency, components_lifetimes), fmt='o-')
plot.show()

# %%
# If we know the phasor coordinates of two components that contribute to
# multiple phasors, we can compute the contribution of both components for each
# phasor and plot the distributions:

(
    fraction_from_first_component,
    fraction_from_second_component,
) = two_fractions_from_phasor(real, imag, components_real, components_imag)

plt.figure()
plt.hist(fraction_from_first_component.flatten(), range=(0, 1), bins=100)
plt.title('Histogram of fractions of first component')
plt.xlabel('Fraction of first component')
plt.ylabel('Counts')
plt.show()

plt.figure()
plt.hist(fraction_from_second_component.flatten(), range=(0, 1), bins=100)
plt.title('Histogram of fractions of second component')
plt.xlabel('Fraction of second component')
plt.ylabel('Counts')
plt.show()

# %%
# sphinx_gallery_thumbnail_number = 2
