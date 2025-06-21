"""Tests for the phasorpy module."""


def test_version():
    """Test phasorpy.__version__ attribute."""
    from phasorpy import __version__

    assert '.' in __version__


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
