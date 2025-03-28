"""Tests for the phasorpy.components module."""

import numpy
import pytest
from numpy.testing import assert_allclose

from phasorpy.components import (
    graphical_component_analysis,
    n_fractions_from_phasor,
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
def test_graphical_component_analysis(
    real,
    imag,
    component_real,
    component_imag,
    radius,
    fractions,
    expected_counts,
):
    """Test graphical_component_analysis function."""
    actual_counts = graphical_component_analysis(
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
def test_errors_graphical_component_analysis(
    real, imag, component_real, component_imag, fractions
):
    """Test errors in graphical_component_analysis function."""
    with pytest.raises(ValueError):
        graphical_component_analysis(
            real, imag, component_real, component_imag, fractions=fractions
        )


# def test_n_fractions_from_phasorg():
#     """Test phasor_based_unmixing function."""
#     assert_allclose(
#         n_fractions_from_phasor(
#             [0.5, 0.3],
#             [0.2, 0.7],
#             [[0.1, 0.3], [0.2, 0.8], [1.0, 1.0]],
#         ),
#         (0.8161838161838166, 0.19580419580419536),
#     )

#     with pytest.raises(ValueError):
#         n_fractions_from_phasor(
#             [0.5, 0.3], [0.2], [[0.5, 0.3], [0.2, 0.7], [1.0, 1.0]]
#         )
#     with pytest.raises(ValueError):
#         n_fractions_from_phasor([0.5, 0.3], [0.2, 0.7], [])


# def test_n_fractions_from_phasor_nan_inf_handling():
#     result = n_fractions_from_phasor(
#         [[numpy.nan, numpy.inf, -numpy.inf], [0.5, 0.3, 0.1]],
#         [[0.2, 0.7, -0.5], [numpy.nan, numpy.inf, -numpy.inf]],
#         numpy.random.rand(3, 3),
#     )

#     assert isinstance(result, tuple)
#     assert all(isinstance(arr, numpy.ndarray) for arr in result)
#     assert not numpy.any([numpy.isnan(arr).any() for arr in result])
#     assert not numpy.any([numpy.isinf(arr).any() for arr in result])


# def test_n_fractions_from_phasor_mismatched_shapes():
#     with pytest.raises(ValueError, match='real.shape=.* != imag.shape=.*'):
#         n_fractions_from_phasor([[0.5, 0.3]], [[0.5]], numpy.random.rand(3, 2))


# def test_n_fractions_from_phasor_empty_coeff_matrix():
#     with pytest.raises(ValueError):
#         n_fractions_from_phasor([[0.5, 0.3]], [[0.5, 0.3]], [])


# def test_n_fractions_from_phasor_1d_input():
#     # ensure coeff_matrix has rows equal to the length of vecB
#     result = n_fractions_from_phasor(
#         [0.5, 0.3], [0.2, 0.7], numpy.random.rand(3, 2)
#     )
#     assert isinstance(result, (tuple))
#     assert all(isinstance(arr, numpy.ndarray) for arr in result)


# @pytest.mark.parametrize('lapack_driver', ['gelsd', 'gelss', 'gelsy'])
# def test_lapack_driver_options(lapack_driver):
#     # ensure M > N for residuals
#     result = n_fractions_from_phasor(
#         numpy.random.rand(3, 10, 10),
#         numpy.random.rand(3, 10, 10),
#         numpy.random.rand(7, 7),
#         lapack_driver=lapack_driver,
#     )
#     assert isinstance(result, tuple)
#     assert all(isinstance(arr, numpy.ndarray) for arr in result)


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
