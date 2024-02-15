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

import math

import numpy

from phasorpy.phasor import (
    phasor_from_lifetime,
    phasor_semicircle,
    phasor_to_polar,
)


def phasor_plot(
    real,
    imag,
    fmt: str = 'o',
    *,
    ax=None,
    mode: str = 'lifetime',
    style: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    bins: int | None = None,
    cmap=None,
    show: bool = True,
    return_ax: bool = False,
):
    """Plot phasor coordinates using matplotlib."""
    # TODO: move this function to phasorpy.plot
    from matplotlib import pyplot
    from matplotlib.lines import Line2D

    if mode == 'lifetime':
        xlim = [-0.05, 1.05]
        ylim = [-0.05, 0.65]
        ranges = [[0, 1], [0, 0.625]]
        bins = 256 if bins is None else bins
        bins_list = [bins, int(bins * 0.625)]
    elif mode == 'spectral':
        xlim = [-1.05, 1.05]
        ylim = [-1.05, 1.05]
        ranges = [[0, 1], [0, 1]]
        bins = 256 if bins is None else bins
        bins_list = [bins, bins]
    else:
        raise ValueError(f'unknown {mode=!r}')
    if ax is None:
        ax = pyplot.subplots()[1]
    if style is None:
        style = 'scatter' if real.size < 1024 else 'histogram'
    if real is None or imag is None:
        pass
    elif style == 'histogram':
        ax.hist2d(
            real,
            imag,
            range=ranges,
            bins=bins_list,
            cmap='Blues' if cmap is None else cmap,
            norm='log',
        )
    elif style == 'scatter':
        for re, im in zip(
            numpy.array(real, ndmin=2), numpy.array(imag, ndmin=2)
        ):
            ax.plot(re, im, fmt)
    if mode == 'lifetime':
        ax.plot(*phasor_semicircle(100), color='k', lw=0.5)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    elif mode == 'spectral':
        ax.add_patch(
            pyplot.Circle((0, 0), 1, color='k', lw=0.5, ls='--', fill=False)
        )
        ax.add_patch(
            pyplot.Circle(
                (0, 0), 2 / 3, color='0.5', lw=0.25, ls='--', fill=False
            )
        )
        ax.add_patch(
            pyplot.Circle(
                (0, 0), 1 / 3, color='0.5', lw=0.25, ls='--', fill=False
            )
        )
        ax.add_line(Line2D([-1, 1], [0, 0], color='k', lw=0.5, ls='--'))
        ax.add_line(Line2D([0, 0], [-1, 1], color='k', lw=0.5, ls='--'))
        for a in (3, 6):
            x = math.cos(math.pi / a)
            y = math.sin(math.pi / a)
            ax.add_line(
                Line2D([-x, x], [-y, y], color='0.5', lw=0.25, ls='--')
            )
            ax.add_line(
                Line2D([-x, x], [y, -y], color='0.5', lw=0.25, ls='--')
            )
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    ax.set(
        title='Phasor plot' if title is None else title,
        xlabel='G, real' if xlabel is None else xlabel,
        ylabel='S, imag' if ylabel is None else ylabel,
        aspect='equal',
        xlim=xlim,
        ylim=ylim,
    )
    if show:
        pyplot.show()
    if return_ax:
        return ax


def multi_frequency_plot(frequency, phase, modulation, title=None):
    """Plot phase and modulation vs frequency."""
    # TODO: move this function to phasorpy.plot
    from matplotlib import pyplot

    ax = pyplot.subplots()[1]
    ax.set_title('Multi-frequency plot' if title is None else title)
    ax.set_xscale('log', base=10)
    ax.set_xlabel('frequency (MHz)')
    ax.set_ylabel('phase (°)', color='tab:blue')
    ax.set_yticks([0.0, 30.0, 60.0, 90.0])
    for phi in numpy.array(phase, ndmin=2).swapaxes(0, 1):
        ax.plot(frequency, numpy.rad2deg(phi), color='tab:blue')
    ax = ax.twinx()
    ax.set_ylabel('modulation (%)', color='tab:red')
    ax.set_yticks([0.0, 25.0, 50.0, 75.0, 100.0])
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
        frequency=[40e6, 80e6, 160e6],
        lifetime=lifetime_distribution,
        fraction=fraction_distribution,
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
    *phasor_from_lifetime(80.0, 4.2 * (1.0 - efficiency)),
    fmt='k.',
    show=False,
    return_ax=True,
)

phasor_plot(
    *phasor_from_lifetime(
        frequency=80.0,
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

multi_frequency_plot(
    frequency,
    *phasor_to_polar(
        *phasor_from_lifetime(frequency, [3.9788735, 0.9947183], fraction)
    ),
)
