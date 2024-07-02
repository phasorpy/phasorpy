"""Tests for the phasorpy.components module."""

import pytest
from numpy.testing import assert_allclose

from phasorpy.components import (
    graphical_component_analysis,
    two_fractions_from_phasor,
)


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


@pytest.mark.parametrize(
    """real, imag,
    real_components, imag_components,
    cursor_diameter, number_of_steps,
    expected_counts, expected_fractions""",
    [
        (
            0.6,
            0.35,
            [0.2, 0.9],
            [0.4, 0.3],
            0.05,
            5,
            ([0, 0, 1, 0, 0, 0],),
            [0, 0.2, 0.4, 0.6, 0.8, 1],
        ),  # Two components, phasor as scalar
        (
            [0.6, 0.4],
            [0.35, 0.38],
            [0.2, 0.9],
            [0.4, 0.3],
            0.15,
            5,
            ([0, 0, 1, 0, 1, 0],),
            [0, 0.2, 0.4, 0.6, 0.8, 1],
        ),  # Two components, phasors as list. Increase cursor diameter.
        (
            [0.6, 0.4],
            [0.35, 0.38],
            [0.2, 0.9],
            [0.4, 0.3],
            0.05,
            10,
            ([0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],),
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        ),  # Two components, phasors as list. Increase number of steps.
        (
            0.3,
            0.2,
            [0.0, 0.2, 0.9],
            [0.0, 0.4, 0.3],
            0.05,
            5,
            ([0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0]),
            [0, 0.2, 0.4, 0.6, 0.8, 1],
        ),  # Three components, phasor as scalar
        (
            [0.3, 0.5],
            [0.2, 0.3],
            [0.0, 0.2, 0.9],
            [0.0, 0.4, 0.3],
            0.05,
            5,
            ([0, 1, 0, 1, 0, 0], [0, 1, 0, 1, 0, 0], [0, 0, 1, 1, 0, 0]),
            [0, 0.2, 0.4, 0.6, 0.8, 1],
        ),  # Three components, phasors as list
        (
            [0.4, 0.82],
            [0.38, 0.4],
            [0.8, 0.2, 0.042],
            [0.4, 0.4, 0.2],
            0.05,
            4,
            ([0, 0, 1, 0, 1], [0, 0, 0, 1, 2], [1, 1, 1, 2, 2]),
            [0, 0.25, 0.5, 0.75, 1],
        ),  # Phasor outside semicircle but inside cursor of component 1
        (
            [0.4, 0.84],
            [0.38, 0.4],
            [0.8, 0.2, 0.042],
            [0.4, 0.4, 0.2],
            0.05,
            4,
            ([0, 0, 1, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1]),
            [0, 0.25, 0.5, 0.75, 1],
        ),  # Phasor outside semicircle and outside cursor of component 1
    ],
)
def test_graphical_component_analysis(
    real,
    imag,
    real_components,
    imag_components,
    cursor_diameter,
    number_of_steps,
    expected_counts,
    expected_fractions,
):
    """Test graphical_component_analysis function."""
    actual_counts, actual_fractions = graphical_component_analysis(
        real,
        imag,
        real_components,
        imag_components,
        cursor_diameter=cursor_diameter,
        number_of_steps=number_of_steps,
    )
    for actual_count, expected_count in zip(actual_counts, expected_counts):
        assert_allclose(actual_count, expected_count)
    assert_allclose(actual_fractions, expected_fractions)


@pytest.mark.parametrize(
    """real, imag,
    real_components, imag_components""",
    [
        ([0], [0, 0], [0, 1], [0, 1]),  # imag.shape != real.shape
        ([0, 0], [0], [0, 1], [0, 1]),  # real.shape != imag.shape
        (
            [0],
            [0],
            [0, 1, 2],
            [0, 1],
        ),  # imag_components.shape != real_components.shape
        (
            [0],
            [0],
            [0, 1],
            [0, 1, 2],
        ),  # real_components.shape != imag_components.shape
        ([0], [0], [0], [0]),  # Number of components is 1
        (
            [0],
            [0],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ),  # Number of components is more than 3
        ([0], [0], [0, 0], [0, 0]),  # Components have same coordinates
    ],
)
def test_errors_graphical_component_analysis(
    real,
    imag,
    real_components,
    imag_components,
):
    """Test errors in graphical_component_analysis function."""
    with pytest.raises(ValueError):
        graphical_component_analysis(
            real, imag, real_components, imag_components
        )
