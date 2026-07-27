# Copyright (c) PhasorPy Contributors
# SPDX-License-Identifier: MIT
# See LICENSE.txt file in the project root for details.

"""PhasorPy package command-line interface."""

import sys

from . import cli

sys.exit(cli.main())  # pylint: disable=no-value-for-parameter
