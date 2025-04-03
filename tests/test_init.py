"""Test the phasorpy.__init__ module."""

from phasorpy import __version__
from phasorpy.utils import versions


def test_versions():
    """Test phasorpy.versions function."""
    ver = versions()
    assert 'Python-' in ver
    assert f'phasorpy-{__version__}\nnumpy-' in ver
    assert '(' not in ver

    ver = versions(sep=', ', dash=' ', verbose=True)
    assert f', phasorpy {__version__}  (' in ver


def test_module():
    """Test phasorpy module namespace."""
    from phasorpy import __all__, phasor_from_signal

    assert len(__all__) >= 38
    assert 'utils' not in __all__
    assert 'annotations' not in __all__
    assert 'phasor_from_signal' in __all__
    assert phasor_from_signal.__module__ == 'phasorpy'


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
