"""Test the phasorpy.__init__ module."""


def test_module():
    """Test phasorpy module namespace."""
    from phasorpy import __all__, phasor_from_signal

    assert len(__all__) >= 38
    assert 'utils' not in __all__
    assert 'annotations' not in __all__
    assert 'phasor_from_signal' in __all__
    assert '__version__' in __all__
    assert phasor_from_signal.__module__ == 'phasorpy'


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
