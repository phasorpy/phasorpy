"""
Phasor coordinates from lifetimes
=================================

An introduction to the `phasor_from_lifetime` function.

The :py:func:`phasorpy.phasor.phasor_from_lifetime` function is used
to calculate phasor coordinates as a function of frequency,
single or multiple lifetime components, and the pre-exponential amplitudes
or fractional intensities of the components.

"""

# %%
# Import required modules and functions:

import numpy

from phasorpy.phasor import phasor_from_lifetime, phasor_to_polar
from phasorpy.plot import PhasorPlot, plot_phasor, plot_polar_frequency

# %%
# Single-component lifetimes
# --------------------------
#
# The phasor coordinates of single-component lifetimes are located
# on the universal circle.
# For example, 3.9788735 ns and 0.9947183 ns at a frequency of 80 MHz:

lifetime = numpy.array([3.9788735, 0.9947183])

plot_phasor(*phasor_from_lifetime(80.0, lifetime), frequency=80.0)

# %%
# Multi-component lifetimes
# -------------------------
#
# The phasor coordinates of two lifetime components with varying
# fractional intensities are linear combinations of the coordinates
# of the pure components:

fraction = numpy.array(
    [[1, 0], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0, 1]]
)

plot_phasor(
    *phasor_from_lifetime(80.0, lifetime, fraction), fmt='o-', frequency=80.0
)

# %%
# Pre-exponential amplitudes
# --------------------------
#
# The phasor coordinates of two lifetime components with varying
# pre-exponential amplitudes are also located on a line:

plot_phasor(
    *phasor_from_lifetime(80.0, lifetime, fraction, preexponential=True),
    fmt='o-',
    frequency=80.0,
)

# %%
# Lifetime distributions at multiple frequencies
# ----------------------------------------------
#
# Phasor coordinates can be calculated at once for many frequencies,
# lifetime components, and their fractions. As an example, random distrinutions
# of lifetimes and their fractions are plotted at three frequencies.
# Lifetimes are passed in units of s and frequencies in Hz, requiring to
# specify a `unit_conversion` factor:

rng = numpy.random.default_rng()

samples = 100
lifetime_distribution = (
    numpy.column_stack(
        (
            rng.normal(3.9788735, 0.05, samples),
            rng.normal(1.9894368, 0.05, samples),
            rng.normal(0.9947183, 0.05, samples),
        )
    )
    * 1e-9
)
fraction_distribution = numpy.column_stack(
    (rng.random(samples), rng.random(samples), rng.random(samples))
)

plot_phasor(
    *phasor_from_lifetime(
        frequency=[40e6, 80e6, 160e6],
        lifetime=lifetime_distribution,
        fraction=fraction_distribution,
        unit_conversion=1.0,
    ),
    fmt='.',
    label=('40 MHz', '80 MHz', '160 MHz'),
)

# %%
# FRET efficiency
# ---------------
#
# The phasor coordinates of a fluorescence energy transfer donor
# with a single lifetime component of 4.2 ns as a function of FRET efficiency
# at a frequency of 80 MHz, with some background signal and about 90 %
# of the donors participating in energy transfer, are on a curved trajectory.
# For comparison, when 100% donors participate in FRET and there is no
# background signal, the phasor coordinates lie on the universal semicircle:

samples = 25
efficiency = numpy.linspace(0.0, 1.0, samples)

plot = PhasorPlot(frequency=80.0)
plot.plot(
    *phasor_from_lifetime(80.0, 4.2 * (1.0 - efficiency)),
    label='100% Donor in FRET',
    fmt='k.',
)
plot.plot(
    *phasor_from_lifetime(
        80.0,
        lifetime=numpy.column_stack(
            (
                numpy.full(samples, 4.2),  # donor-only lifetime
                4.2 * (1.0 - efficiency),  # donor lifetime with FRET
                numpy.full(samples, 1e9),  # background with long lifetime
            )
        ),
        fraction=[0.1, 0.9, 0.1 / 1e9],
        preexponential=True,
    ),
    label='90% Donor in FRET',
    fmt='o-',
)
plot.show()

# %%
# Multi-frequency plot
# --------------------
#
# Phase shift and demodulation of multi-component lifetimes can be calculated
# as a function of the excitation light frequency and fractional intensities:

frequency = numpy.logspace(-1, 4, 32)
fraction = numpy.array([[1, 0], [0.5, 0.5], [0, 1]])

plot_polar_frequency(
    frequency,
    *phasor_to_polar(
        *phasor_from_lifetime(frequency, [3.9788735, 0.9947183], fraction)
    ),
)

# %%
# sphinx_gallery_thumbnail_number = -2
