"""PhasorPy package command line interface.

Invoke the command line application with::

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


@click.group(help='PhasorPy package command line interface.')
@click.version_option(version=__version__)
def main() -> int:
    """PhasorPy command line interface."""
    return 0


@main.command(help='Show runtime versions.')
@click.option(
    '--verbose',
    default=False,
    is_flag=True,
    type=click.BOOL,
    help='Show module paths.',
)
def versions(verbose: bool) -> None:
    """Versions command group."""
    from .utils import versions

    click.echo(versions(verbose=verbose))


@main.command(help='Fetch sample files from remote repositories.')
@click.argument('files', nargs=-1)
@click.option(
    '--hideprogress',
    default=False,
    is_flag=True,
    type=click.BOOL,
    help='Hide progressbar.',
)
def fetch(files: Iterable[str], hideprogress: bool) -> None:
    """Fetch command group."""
    from . import datasets

    files = datasets.fetch(
        *files, return_scalar=False, progressbar=not hideprogress
    )
    click.echo(f'Cached at {os.path.commonpath(files)}')


@main.command(help='Start interactive FRET phasor plot.')
@click.option(
    '--hide',
    default=False,
    is_flag=True,
    type=click.BOOL,
    help='Do not show interactive plot.',
)
def fret(hide: bool) -> None:
    """FRET command group."""
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
    number_lifetimes: int,
    frequency: float | None,
    lifetime: tuple[float, ...],
    fraction: tuple[float, ...],
    hide: bool,
) -> None:
    """Lifetime command group."""
    from .lifetime import phasor_semicircle, phasor_to_normal_lifetime
    from .plot import LifetimePlots

    if not lifetime:
        if number_lifetimes == 2:
            lifetime = (4.0, 1.0)
        else:
            real, imag = phasor_semicircle(number_lifetimes + 2)
            lifetime = phasor_to_normal_lifetime(
                real[1:-1], imag[1:-1], frequency if frequency else 80.0
            )  # type: ignore[assignment]

    plot = LifetimePlots(
        frequency,
        lifetime,
        fraction if len(fraction) > 0 else None,
        interactive=True,
    )
    if not hide:
        plot.show()


if __name__ == '__main__':
    import sys

    sys.exit(main())  # pylint: disable=no-value-for-parameter
