"""Tests for the phasorpy module."""

from phasorpy import __version__, versions


def test_versions():
    """Test phasorpy.versions function."""
    ver = versions()
    assert f'phasorpy {__version__}' in ver
    assert 'numpy' in ver
