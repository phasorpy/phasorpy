"""
Phasor coordinates from lifetimes
=================================

An introduction to the :py:func:`phasorpy.phasor.phasor_from_lifetime`
function to calculate phasor coordinates as a function of frequency,
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


def phasor_plot(real, imag):
    """Plot phasor coordinates."""
    fig, ax = pyplot.subplots()
    ax.set(xlabel='real', ylabel='imag')
    ax.axis('equal')
    ax.plot(*phasor_semicircle(), color='k', lw=0.25)
    for re, im in zip(numpy.array(real, ndmin=2), numpy.array(imag, ndmin=2)):
        ax.scatter(re, im)
    pyplot.show()


def polar_plot(frequency, phase, modulation):
    """Plot phase and modulation vs frequency."""
    fig, ax = pyplot.subplots()
    ax.set_xscale('log', base=10)
    ax.set_xlabel('frequency (MHz)')
    ax.set_ylabel('phase (°)', color='tab:blue')
    ax.plot(frequency, numpy.rad2deg(phase), color='tab:blue')
    ax = ax.twinx()
    ax.set_ylabel('modulation (%)', color='tab:red')
    ax.plot(frequency, modulation * 100, color='tab:red')
    pyplot.show()


# %%
# Single-component lifetimes
# --------------------------
#
# The phasor coordinates of single-component lifetimes are located
# on the universal circle:

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
phasor_plot(*phasor_from_lifetime(80.0, lifetime, fraction))


# %%
# Pre-exponential amplitudes
# --------------------------
#
# The phasor coordinates of two lifetime components with varying
# pre-exponential amplitudes are also located on a line:

phasor_plot(*phasor_from_lifetime(80.0, lifetime, fraction, is_preexp=True))


# %%
# Lifetime distributions at multiple frequencies
# ----------------------------------------------
#
# Phasor coordinates can be calculated at once for many frequencies,
# lifetime components, and their fractions:

size = 100
rng = numpy.random.default_rng()
lifetime_distribution = numpy.column_stack(
    (
        rng.normal(3.9788735, 0.05, size),
        rng.normal(1.9894368, 0.05, size),
        rng.normal(0.9947183, 0.05, size),
    )
)
fraction_distribution = numpy.column_stack(
    (rng.random(size), rng.random(size), rng.random(size))
)
phasor_plot(
    *phasor_from_lifetime(
        [40.0, 80.0, 160.0], lifetime_distribution, fraction_distribution
    )
)

# %%
# Classic frequency-domain plots
# ------------------------------
#
# Phase-shift and demodulation of multi-component lifetimes can easily
# be calculated as a function of the excitation light frequency:

frequency = numpy.logspace(-1, 4, 32)
polar_plot(
    frequency,
    *phasor_to_polar(
        *phasor_from_lifetime(frequency, [3.9788735, 0.9947183], [0.8, 0.2])
    ),
)
