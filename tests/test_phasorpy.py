# Copyright (c) PhasorPy Contributors
# SPDX-License-Identifier: MIT
# See LICENSE.txt file in the project root for details.

"""Test the phasorpy module."""


def test_version() -> None:
    """Test phasorpy.__version__ attribute."""
    from phasorpy import __version__

    assert '.' in __version__
