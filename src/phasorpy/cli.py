"""PhasorPy package command line interface.

Invoke the command line application with::

    $ python -m phasorpy --help

"""

from __future__ import annotations

import os

import click

from . import version


@click.group(help='PhasorPy package command line interface.')
@click.version_option(version=version.__version__)
def main() -> int:
    """PhasorPy command line interface."""
    return 0


@main.command(help='Show runtime versions.')
def versions():
    """Versions command group."""
    click.echo(version.versions())


@main.command(help='Fetch sample files from remote repositories.')
@click.argument('files', nargs=-1)
@click.option(
    '--hideprogress',
    default=False,
    is_flag=True,
    type=click.BOOL,
    help='Hide progressbar.',
)
def fetch(files, hideprogress):
    """Fetch command group."""
    from . import datasets

    files = datasets.fetch(
        *files, return_scalar=False, progressbar=not hideprogress
    )
    click.echo(f'Cached at {os.path.commonpath(files)}')


if __name__ == '__main__':
    import sys

    sys.exit(main())  # pylint: disable=no-value-for-parameter
