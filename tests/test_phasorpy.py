"""Tests for the phasorpy module."""

from phasorpy import __version__, versions


def test_versions():
    """Test phasorpy.versions function."""
    ver = versions()
    assert 'Python-' in ver
    assert f'phasorpy-{__version__}\nnumpy-' in ver
    assert '(' not in ver

    ver = versions(sep=', ', dash=' ', verbose=True)
    assert f', phasorpy {__version__}  (' in ver


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
