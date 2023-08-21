"""PhasorPy package command line interface.

Invoke the command line application with::

    $ python -m phasorpy --help

"""

from __future__ import annotations

import click

from . import version


@click.command(help='PhasorPy package command line interface.')
@click.version_option(version=version.__version__)
@click.option(
    '--versions',
    default=False,
    is_flag=True,
    show_default=True,
    help='Show runtime versions and exit.',
    type=click.BOOL,
)
def main(versions: bool) -> int:
    """PhasorPy command line interface."""
    if versions:
        click.echo(version.versions())
        return 0
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main())
