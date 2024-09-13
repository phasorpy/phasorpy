"""PhasorPy package command line interface.

Invoke the command line application with::

    $ python -m phasorpy --help

"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Iterable

import click

from . import version


@click.group(help='PhasorPy package command line interface.')
@click.version_option(version=version.__version__)
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
    click.echo(version.versions(verbose=verbose))


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


if __name__ == '__main__':
    import sys

    sys.exit(main())  # pylint: disable=no-value-for-parameter
