"""PhasorPy package command line interface."""

import sys

from . import cli

sys.exit(cli.main())  # pylint: disable=no-value-for-parameter
