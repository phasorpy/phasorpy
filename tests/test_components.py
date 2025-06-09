"""Tests for the phasorpy.components module."""

import numpy
import pytest
from numpy.testing import assert_allclose

from phasorpy.components import (
    phasor_component_fit,
    phasor_component_fraction,
    phasor_component_graphical,
)

numpy.random.seed(42)


def test_phasor_component_fraction():
    """Test phasor_component_fraction function."""
    assert_allclose(
        phasor_component_fraction(
            [0.0, 0.5, 0.6, 0.75, 1.0, 1.5],
            [0.0, 0.5, 0.6, 0.75, 1.0, 1.5],
            [0.5, 1.0],
            [0.5, 1.0],
        ),
        [1.0, 1.0, 0.8, 0.5, 0.0, 0.0],
        1e-6,
    )
    assert_allclose(
        phasor_component_fraction(
            [0.2, 0.5, 0.7],
            [0.2, 0.4, 0.3],
            [0.0582399, 0.79830002],
            [0.23419652, 0.40126936],
        ),
        [0.82766281, 0.38389704, 0.15577992],
        1e-6,
    )
    assert_allclose(
        phasor_component_fraction(
            [0.0, 0.5, 0.9],
            [0.4, 0.4, 0.6],
            [0.0582399, 0.79830002],
            [0.23419652, 0.40126936],
        ),
        [1.0, 0.38389704, 0.0],
        1e-6,
    )

    with pytest.raises(ValueError):
        phasor_component_fraction([0], [0], [0.1, 0.1], [0.2, 0.2])
    with pytest.raises(ValueError):
        phasor_component_fraction([0], [0], [0.3], [0.1, 0.2])
    with pytest.raises(ValueError):
        phasor_component_fraction([0], [0], [0.1, 0.2], [0.3])
    with pytest.raises(ValueError):
        phasor_component_fraction([0], [0], [0.1], [0.3])
    with pytest.raises(ValueError):
        phasor_component_fraction([0], [0], [0.1, 0.1, 0, 1], [0.1, 0, 2])


@pytest.mark.xfail
def test_phasor_component_fraction_channels():
    """Test phasor_component_fraction function for multiple channels."""
    assert_allclose(
        phasor_component_fraction(
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
    component_real, component_imag,
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
def test_phasor_component_graphical(
    real,
    imag,
    component_real,
    component_imag,
    radius,
    fractions,
    expected_counts,
):
    """Test phasor_component_graphical function."""
    actual_counts = phasor_component_graphical(
        real,
        imag,
        component_real,
        component_imag,
        radius=radius,
        fractions=fractions,
    )
    for actual_count, expected_count in zip(actual_counts, expected_counts):
        assert_allclose(actual_count, expected_count)


@pytest.mark.parametrize(
    """real, imag,
    component_real, component_imag,
    fractions
    """,
    [
        # imag.shape != real.shape
        ([0], [0, 0], [0, 1], [0, 1], 10),
        # real.shape != imag.shape
        ([0, 0], [0], [0, 1], [0, 1], 10),
        # component_imag.shape != component_real.shape
        (
            0,
            0,
            [0, 1, 2],
            [0, 1],
            10,
        ),
        # component_real.shape != component_imag.shape
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
def test_errors_phasor_component_graphical(
    real, imag, component_real, component_imag, fractions
):
    """Test errors in phasor_component_graphical function."""
    with pytest.raises(ValueError):
        phasor_component_graphical(
            real, imag, component_real, component_imag, fractions=fractions
        )


def test_phasor_component_fit():
    """Test phasor_component_fit function."""
    size = 5
    component_real = [0.1, 0.9]
    component_imag = [0.2, 0.3]
    mean = numpy.random.rand(size)
    real = numpy.linspace(*component_real, size)
    imag = numpy.linspace(*component_imag, size)
    fractions = (numpy.linspace(1, 0, size), numpy.linspace(0, 1, size))

    # precise solution
    assert_allclose(
        phasor_component_fit(mean, real, imag, component_real, component_imag),
        fractions,
        atol=1e-5,
    )

    # scalars are returned as arrays
    assert_allclose(
        phasor_component_fit(
            mean[1], real[1], imag[1], component_real, component_imag
        ),
        (fractions[0][1:2], fractions[1][1:2]),
        atol=1e-5,
    )

    # add noise
    real += numpy.random.rand(size) * 1e-3
    imag += numpy.random.rand(size) * 1e-3
    assert_allclose(
        phasor_component_fit(mean, real, imag, component_real, component_imag),
        fractions,
        atol=1e-2,
    )

    # lapack_driver
    assert_allclose(
        phasor_component_fit(
            mean,
            real,
            imag,
            component_real,
            component_imag,
            lapack_driver='gelsd',
        ),
        fractions,
        atol=1e-2,
    )

    # NaN handling
    mean[0] = numpy.nan
    real[1] = numpy.nan
    fractions[0][:2] = numpy.nan
    fractions[1][:2] = numpy.nan
    assert_allclose(
        phasor_component_fit(mean, real, imag, component_real, component_imag),
        fractions,
        atol=1e-2,
    )

    # 3 harmonics, 4 components
    phasor_component_fit(
        numpy.zeros((3, 4)),
        numpy.zeros((3, 3, 4)),
        numpy.zeros((3, 3, 4)),
        numpy.zeros((3, 4)),
        numpy.zeros((3, 4)),
    )

    with pytest.raises(ValueError):
        phasor_component_fit(
            numpy.zeros((3, 4)),
            numpy.zeros((2, 3, 4)),  # 2 harmonics
            numpy.zeros((2, 3, 4)),
            numpy.zeros((3, 4)),  # 3 harmonics
            numpy.zeros((3, 4)),
        )

    # system is undetermined: 4 components, 1 harmonic
    with pytest.raises(ValueError):
        phasor_component_fit(
            numpy.zeros(10),
            numpy.zeros(10),
            numpy.zeros(10),
            numpy.zeros(4),
            numpy.zeros(4),
        )

    with pytest.raises(ValueError):
        phasor_component_fit(
            mean, real, imag, numpy.zeros((1, 2, 2)), numpy.zeros((1, 2, 2))
        ),

    with pytest.raises(ValueError):
        phasor_component_fit(
            mean[:-1], real, imag, component_real, component_imag
        )

    with pytest.raises(ValueError):
        phasor_component_fit(
            mean, real[:-1], imag, component_real, component_imag
        )

    with pytest.raises(ValueError):
        phasor_component_fit(
            mean, real, imag, component_real[:-1], component_imag
        )

    component_real[0] = numpy.nan
    with pytest.raises(ValueError):
        phasor_component_fit(mean, real, imag, component_real, component_real)

    component_imag[0] = numpy.inf
    with pytest.raises(ValueError):
        phasor_component_fit(mean, real, imag, component_imag, component_imag)


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type, call-overload"
