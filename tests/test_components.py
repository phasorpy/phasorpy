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
            [0.0, 0.5, 0.6, 0.75, 1.0, 1.5],
            [0.0, 0.5, 0.6, 0.75, 1.0, 1.5],
            [0.5, 1.0],
            [0.5, 1.0],
        ),
        [1.0, 1.0, 0.8, 0.5, 0.0, 0.0],
        1e-6,
    )
    assert_allclose(
        two_fractions_from_phasor(
            [0.2, 0.5, 0.7],
            [0.2, 0.4, 0.3],
            [0.0582399, 0.79830002],
            [0.23419652, 0.40126936],
        ),
        [0.82766281, 0.38389704, 0.15577992],
        1e-6,
    )
    assert_allclose(
        two_fractions_from_phasor(
            [0.0, 0.5, 0.9],
            [0.4, 0.4, 0.6],
            [0.0582399, 0.79830002],
            [0.23419652, 0.40126936],
        ),
        [1.0, 0.38389704, 0.0],
        1e-6,
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
    components_real, components_imag,
    radius, fractions,
    expected_counts""",
    [
        # Two components, phasor as scalar
        (
            0.6,
            0.35,
            [0.2, 0.9],
            [0.4, 0.3],
            0.05,
            6,
            ([0, 0, 1, 0, 0, 0],),
        ),
        # Two components, phasors as list. Increase cursor radius.
        (
            [0.6, 0.4],
            [0.35, 0.38],
            [0.2, 0.9],
            [0.4, 0.3],
            0.15,
            [0, 0.2, 0.4, 0.6, 0.8, 1],
            ([0, 0, 1, 2, 1, 0],),
        ),
        # Two components, phasors as list. Increase number of steps.
        (
            [0.6, 0.4],
            [0.35, 0.38],
            [0.2, 0.9],
            [0.4, 0.3],
            0.05,
            11,
            ([0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],),
        ),
        # Three components, phasor as scalar
        (
            0.3,
            0.2,
            [0.0, 0.2, 0.9],
            [0.0, 0.4, 0.3],
            0.05,
            [0, 0.2, 0.4, 0.6, 0.8, 1],
            ([0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 1, 1, 0, 0]),
        ),
        # Three components, phasors as list
        (
            [0.3, 0.5],
            [0.2, 0.3],
            [0.0, 0.2, 0.9],
            [0.0, 0.4, 0.3],
            0.05,
            6,
            ([0, 1, 1, 1, 0, 0], [0, 1, 0, 1, 0, 0], [0, 0, 2, 1, 0, 0]),
        ),
        # Phasor outside semicircle but inside cursor of component 1
        (
            [0.4, 0.82],
            [0.38, 0.4],
            [0.8, 0.2, 0.042],
            [0.4, 0.4, 0.2],
            0.05,
            5,
            ([0, 0, 1, 0, 1], [0, 0, 0, 1, 2], [1, 1, 2, 2, 2]),
        ),
        # Phasor outside semicircle and outside cursor of component 1
        (
            [0.4, 0.86],
            [0.38, 0.4],
            [0.8, 0.2, 0.042],
            [0.4, 0.4, 0.2],
            0.05,
            [0, 0.25, 0.5, 0.75, 1],
            ([0, 0, 1, 0, 0], [0, 0, 0, 1, 1], [0, 0, 1, 1, 1]),
        ),
        # Two components no fractions provided
        (
            0,
            0,
            [0, 0],
            [0, 0.2],
            0.05,
            None,
            ([0, 0, 0, 0, 0, 0, 1, 1, 1],),
        ),
        # Three components no fractions provided
        (
            0,
            0,
            [0, 0.1, 0],
            [0, 0.1, 0.2],
            0.05,
            None,
            (
                [0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
        ),
    ],
)
def test_graphical_component_analysis(
    real,
    imag,
    components_real,
    components_imag,
    radius,
    fractions,
    expected_counts,
):
    """Test graphical_component_analysis function."""
    actual_counts = graphical_component_analysis(
        real,
        imag,
        components_real,
        components_imag,
        radius=radius,
        fractions=fractions,
    )
    for actual_count, expected_count in zip(actual_counts, expected_counts):
        assert_allclose(actual_count, expected_count)


@pytest.mark.parametrize(
    """real, imag,
    components_real, components_imag,
    fractions
    """,
    [
        # imag.shape != real.shape
        ([0], [0, 0], [0, 1], [0, 1], 10),
        # real.shape != imag.shape
        ([0, 0], [0], [0, 1], [0, 1], 10),
        # components_imag.shape != components_real.shape
        (
            0,
            0,
            [0, 1, 2],
            [0, 1],
            10,
        ),
        # components_real.shape != components_imag.shape
        (
            0,
            0,
            [0, 1],
            [0, 1, 2],
            10,
        ),
        # Number of components is 1
        ([0], [0], [0], [0], 10),
        # number of components is more than 3
        (
            0,
            0,
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            10,
        ),
        # components are not one dimensional
        (
            0,
            0,
            [[0], [1]],
            [[0], [1]],
            10,
        ),
        # negative fraction
        ([0, 0], [0, 0], [0, 1], [0, 1], -10),
        # negative fraction
        ([0, 0], [0, 0], [0, 1], [0, 1], [0.5, -0.5]),
        # fraction not 1D
        ([0, 0], [0, 0], [0, 1], [0, 1], [[0.5], [-0.5]]),
    ],
)
def test_errors_graphical_component_analysis(
    real, imag, components_real, components_imag, fractions
):
    """Test errors in graphical_component_analysis function."""
    with pytest.raises(ValueError):
        graphical_component_analysis(
            real, imag, components_real, components_imag, fractions=fractions
        )


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
