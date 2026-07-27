# Copyright (c) PhasorPy Contributors
# SPDX-License-Identifier: MIT
# See LICENSE.txt file in the project root for details.

"""PhasorPy package command-line interface.

Invoke the command-line application with::

    $ python -m phasorpy --help

"""

from __future__ import annotations

__all__: list[str] = []

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Iterable

import click

from . import __version__


@click.group(help='PhasorPy package command-line interface.')
@click.version_option(version=__version__)
def main() -> int:
    """PhasorPy command-line interface."""
    return 0


@main.command(help='Show runtime versions.')
@click.option(
    '--verbose',
    default=False,
    is_flag=True,
    type=click.BOOL,
    help='Show full module paths.',
)
def versions(*, verbose: bool) -> None:
    """Show runtime versions."""
    from .utils import versions

    click.echo(versions(verbose=verbose))


@main.command(help='Fetch sample files from remote repositories.')
@click.argument('files', nargs=-1)
@click.option(
    '--show-progress/--hide-progress',
    default=True,
    help='Show progress bar.',
)
def fetch(*, files: Iterable[str], show_progress: bool) -> None:
    """Fetch sample files from remote repositories."""
    from . import datasets

    files = datasets.fetch(
        *files, return_scalar=False, progressbar=show_progress
    )
    if files:
        click.echo(f'Cached at {os.path.commonpath(files)}')
    else:
        click.echo('No files fetched')


@main.command(help='Start interactive FRET phasor plot.')
@click.option(
    '--hide',
    default=False,
    is_flag=True,
    type=click.BOOL,
    help='Do not show interactive plot.',
)
def fret(*, hide: bool) -> None:
    """Start interactive FRET phasor plot."""
    from .plot import PhasorPlotFret

    plot = PhasorPlotFret(
        frequency=60.0,
        donor_lifetime=4.2,
        acceptor_lifetime=3.0,
        fret_efficiency=0.5,
        interactive=True,
    )
    if not hide:
        plot.show()


@main.command(help='Start interactive lifetime plots.')
@click.argument(
    'number_lifetimes',
    default=2,
    type=click.IntRange(1, 5),
    required=False,
    # help='Number of preconfigured lifetimes.',
)
@click.option(
    '-f',
    '--frequency',
    type=float,
    required=False,
    help='Laser/modulation frequency in MHz.',
)
@click.option(
    '-l',
    '--lifetime',
    # default=(4.0, 1.0),
    type=float,
    multiple=True,
    required=False,
    help='Lifetime in ns.',
)
@click.option(
    '-a',
    '--fraction',
    type=float,
    multiple=True,
    required=False,
    help='Fractional intensity of lifetime.',
)
@click.option(
    '--hide',
    default=False,
    is_flag=True,
    type=click.BOOL,
    help='Do not show interactive plot.',
)
def lifetime(
    *,
    number_lifetimes: int,
    frequency: float | None,
    lifetime: tuple[float, ...],
    fraction: tuple[float, ...],
    hide: bool,
) -> None:
    """Start interactive lifetime plots."""
    from .lifetime import phasor_semicircle, phasor_to_normal_lifetime
    from .plot import LifetimePlots

    if not lifetime:
        if number_lifetimes == 2:
            lifetime = (4.0, 1.0)
        else:
            real, imag = phasor_semicircle(number_lifetimes + 2)
            lifetime = phasor_to_normal_lifetime(
                real[1:-1], imag[1:-1], frequency or 80.0
            )  # type: ignore[assignment]

    plot = LifetimePlots(
        frequency,
        lifetime,
        fraction if len(fraction) > 0 else None,
        interactive=True,
    )
    if not hide:
        plot.show()


@main.command(help='Start interactive spectral plots.')
@click.argument(
    'number_spectra',
    default=1,
    type=click.IntRange(1, 4),
    required=False,
)
@click.option(
    '-o',
    '--origin',
    type=float,
    multiple=True,
    required=False,
    help='Origin wavelength.',
)
@click.option(
    '-s',
    '--sigma',
    type=float,
    multiple=True,
    required=False,
    help='Spectral broadening factor.',
)
@click.option(
    '-a',
    '--fraction',
    type=float,
    multiple=True,
    required=False,
    help='Fractional intensity of component.',
)
@click.option(
    '--hr-factor',
    type=float,
    default=0.4,
    show_default=True,
    help='Huang-Rhys factor.',
)
@click.option(
    '--vib-frequency',
    type=float,
    default=1200.0,
    show_default=True,
    help='Vibrational frequency.',
)
@click.option(
    '--wavelength-min',
    type=float,
    default=450.0,
    show_default=True,
    help='Minimum wavelength to display.',
)
@click.option(
    '--wavelength-max',
    type=float,
    default=650.0,
    show_default=True,
    help='Maximum wavelength to display.',
)
@click.option(
    '--samples',
    type=int,
    default=128,
    show_default=True,
    help='Number of wavelength samples.',
)
@click.option(
    '--dho',
    default=False,
    is_flag=True,
    type=click.BOOL,
    help='Use Displaced Harmonic Oscillator model.',
)
@click.option(
    '--hide',
    default=False,
    is_flag=True,
    type=click.BOOL,
    help='Do not show interactive plot.',
)
def spectral(
    *,
    number_spectra: int,
    origin: tuple[float, ...],
    sigma: tuple[float, ...],
    fraction: tuple[float, ...],
    hr_factor: float,
    vib_frequency: float,
    wavelength_min: float,
    wavelength_max: float,
    samples: int,
    dho: bool,
    hide: bool,
) -> None:
    """Start interactive spectral plots."""
    from .plot import SpectralPlots

    _default_origins = {
        1: (518.0,),
        2: (490.0, 570.0),
        3: (480.0, 530.0, 590.0),
        4: (470.0, 510.0, 560.0, 620.0),
    }

    if not origin:
        n = min(number_spectra, 2 if dho else 4)
        origin = _default_origins[n]

    plot = SpectralPlots(
        origin,
        sigma or None,
        fraction if len(fraction) > 0 else None,
        hr_factor=hr_factor,
        vib_frequency=vib_frequency,
        wavelength_range=(wavelength_min, wavelength_max),
        samples=samples,
        dho=dho,
        interactive=True,
    )
    if not hide:
        plot.show()


if __name__ == '__main__':
    import sys

    sys.exit(main())  # pylint: disable=no-value-for-parameter
