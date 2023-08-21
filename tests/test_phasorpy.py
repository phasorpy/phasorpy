"""Tests for the phasorpy module."""

from phasorpy import __version__, versions


def test_versions():
    v = versions()
    assert f'phasorpy {__version__}' in v
    assert 'numpy' in v
