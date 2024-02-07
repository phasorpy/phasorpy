"""
Phasor coordinates from lifetimes
=================================

The :py:func:`phasorpy.phasor.phasor_from_lifetime` function is used
to calculate phasor coordinates as a function of frequency,
single or multiple lifetime components, and the pre-exponential amplitudes
or fractional intensities of the components.

"""

# %%
# Import required modules and functions, and define helper functions for
# plotting phasor or polar coordinates:

import numpy
from matplotlib import pyplot

from phasorpy.phasor import (
    phasor_from_lifetime,
    phasor_semicircle,
    phasor_to_polar,
)


def phasor_plot(real, imag, fmt='o', title='', ax=None, show=True):
    """Plot phasor coordinates."""
    # TODO: replace this function with phasorpy.plot once available
    if ax is None:
        ax = pyplot.subplots()[1]
    ax.set(
        title='Phasor plot' if title is None else title,
        xlabel='G, real',
        ylabel='S, imag',
        aspect='equal',
        ylim=[-0.05, 0.65],
    )
    ax.plot(*phasor_semicircle(), color='k', lw=0.25)
    for re, im in zip(numpy.array(real, ndmin=2), numpy.array(imag, ndmin=2)):
        ax.plot(re, im, fmt)
    if not show:
        return ax
    pyplot.show()


def polar_plot(frequency, phase, modulation, title=''):
    """Plot phase and modulation vs frequency."""
    # TODO: replace this function with phasorpy.plot once available
    ax = pyplot.subplots()[1]
    ax.set_title('Multi-frequency plot' if title is None else title)
    ax.set_xscale('log', base=10)
    ax.set_xlabel('frequency (MHz)')
    ax.set_ylabel('phase (°)', color='tab:blue')
    for phi in numpy.array(phase, ndmin=2).swapaxes(0, 1):
        ax.plot(frequency, numpy.rad2deg(phi), color='tab:blue')
    ax = ax.twinx()
    ax.set_ylabel('modulation (%)', color='tab:red')
    for mod in numpy.array(modulation, ndmin=2).swapaxes(0, 1):
        ax.plot(frequency, mod * 100, color='tab:red')
    pyplot.show()


# %%
# Single-component lifetimes
# --------------------------
#
# The phasor coordinates of single-component lifetimes are located
# on the universal circle.
# For example, 3.9788735 ns and 0.9947183 ns at a frequency of 80 MHz:

lifetime = numpy.array([3.9788735, 0.9947183])

phasor_plot(*phasor_from_lifetime(80.0, lifetime))

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

phasor_plot(*phasor_from_lifetime(80.0, lifetime, fraction), fmt='o-')

# %%
# Pre-exponential amplitudes
# --------------------------
#
# The phasor coordinates of two lifetime components with varying
# pre-exponential amplitudes are also located on a line:

phasor_plot(
    *phasor_from_lifetime(80.0, lifetime, fraction, preexponential=True),
    fmt='o-',
)

# %%
# Lifetime distributions at multiple frequencies
# ----------------------------------------------
#
# Phasor coordinates can be calculated at once for many frequencies,
# lifetime components, and their fractions.
# As an example, lifetimes are passed in units of s and frequencies in Hz,
# requiring to specify a unit_conversion factor:

samples = 100
rng = numpy.random.default_rng()
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

phasor_plot(
    *phasor_from_lifetime(
        [40e6, 80e6, 160e6],
        lifetime_distribution,
        fraction_distribution,
        unit_conversion=1.0,
    ),
    fmt='.',
)

# %%
# FRET efficiency
# ---------------
#
# The phasor coordinates of a fluorescence energy transfer donor
# with a lifetime of 4.2 ns as a function of FRET efficiency
# at a frequency of 80 MHz, with some background signal and about 90 %
# of the donors participating in energy transfer, are on a curved trajectory:

samples = 25
efficiency = numpy.linspace(0.0, 1.0, samples)

# for reference, just donor with FRET
ax = phasor_plot(
    *phasor_from_lifetime(80.0, 4.2 * (1.0 - efficiency)), fmt='k.', show=False
)

phasor_plot(
    *phasor_from_lifetime(
        80.0,
        numpy.column_stack(
            (
                numpy.full(samples, 4.2),  # donor-only lifetime
                4.2 * (1.0 - efficiency),  # donor lifetime with FRET
                numpy.full(samples, 1e9),  # background with long lifetime
            )
        ),
        fraction=[0.1, 0.9, 0.1 / 1e9],
        preexponential=True,
    ),
    fmt='o-',
    ax=ax,
)

# %%
# Multi-frequency plot
# --------------------
#
# Phase shift and demodulation of multi-component lifetimes can be calculated
# as a function of the excitation light frequency and fractional intensities:

frequency = numpy.logspace(-1, 4, 32)
fraction = numpy.array([[1, 0], [0.5, 0.5], [0, 1]])

polar_plot(
    frequency,
    *phasor_to_polar(
        *phasor_from_lifetime(frequency, [3.9788735, 0.9947183], fraction)
    ),
)
