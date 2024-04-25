"""Tests for the phasorpy.components module."""

import pytest
from numpy.testing import assert_allclose

from phasorpy.components import two_fractions_from_phasor


def test_two_fractions_from_phasor():
    """Test two_fractions_from_phasor function."""
    assert_allclose(
        two_fractions_from_phasor(
            [0.2, 0.5, 0.7],
            [0.2, 0.4, 0.3],
            [0.0582399, 0.79830002],
            [0.23419652, 0.40126936],
        ),
        (
            [0.82766281, 0.38389704, 0.15577992],
            [0.17233719, 0.61610296, 0.84422008],
        ),
    )
    assert_allclose(
        two_fractions_from_phasor(
            [0.0, 0.5, 0.9],
            [0.4, 0.4, 0.6],
            [0.0582399, 0.79830002],
            [0.23419652, 0.40126936],
        ),
        (
            [1.0, 0.38389704, 0.0],
            [0.0, 0.61610296, 1.0],
        ),
    )
    with pytest.raises(ValueError):
        two_fractions_from_phasor([0], [0], [0.1, 0.1], [0.2, 0.2])
    with pytest.raises(ValueError):
        two_fractions_from_phasor([0], [0], [0.3], [0.1, 0.2])
    with pytest.raises(ValueError):
        two_fractions_from_phasor([0], [0], [0.1, 0.2], [0.3])
    with pytest.raises(ValueError):
        two_fractions_from_phasor([0], [0], [0.1], [0.3])
    with pytest.raises(ValueError):
        two_fractions_from_phasor([0], [0], [0.1, 0.1, 0, 1], [0.1, 0, 2])


@pytest.mark.xfail
def test_two_fractions_from_phasor_channels():
    """Test two_fractions_from_phasor function for multiple channels."""
    assert_allclose(
        two_fractions_from_phasor(
            [[[0.1, 0.2, 0.3]]],
            [[[0.1, 0.2, 0.3]]],
            [[0.2, 0.2, 0.2], [0.9, 0.9, 0.9]],
            [[0.4, 0.4, 0.4], [0.3, 0.3, 0.3]],
        ),
        (
            [[[1.0, 0.96, 0.84]]],
            [[[0.0, 0.04, 0.16]]],
        ),
    )
