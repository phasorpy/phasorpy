"""Test the phasorpy._typing module."""


def test_import_typing():
    """Test import phasorpy._typing module."""
    from phasorpy._typing import Any  # noqa: F401


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
