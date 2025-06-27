"""Tests for the phasorpy.components module."""

import numpy
import pytest
from numpy.testing import assert_allclose

from phasorpy._phasorpy import _is_near_semicircle
from phasorpy.components import (
    phasor_component_fit,
    phasor_component_fraction,
    phasor_component_graphical,
    phasor_component_search,
)
from phasorpy.phasor import phasor_from_lifetime, phasor_to_normal_lifetime

numpy.random.seed(42)


def _plot(frequency, real, imag, component_real=None, component_imag=None):
    # helper function to visualize component distribution results
    from phasorpy.plot import PhasorPlot

    pp = PhasorPlot(frequency=frequency, allquadrants=False)
    if real.size > 100:
        for i in range(real.shape[0]):
            pp.hist2d(real[i], imag[i], cmap='Greys')
    else:
        pp.plot(real, imag)
    if component_real is not None and component_imag is not None:
        cmap = 'Reds', 'Greens', 'Blues'
        for i in range(component_real.shape[0]):
            pp.hist2d(component_real[i], component_imag[i], cmap=cmap[i])
            pp.plot(
                numpy.nanmean(component_real[i]),
                numpy.nanmean(component_imag[i]),
            )
    pp.show()


def test_phasor_component_search_two():
    """Test phasor_component_search function with two components."""
    # scalar
    frequency = 60.0
    lifetime = [0.5, 4.2]
    fraction = [0.3, 0.7]
    component_real, component_imag, fractions = phasor_component_search(
        *phasor_from_lifetime([frequency, frequency * 2], lifetime, fraction),
        frequency=frequency,
        num_components=2,
        lifetime_range=(0.4, 0.6, 0.1),
    )
    expected_real, expected_imag = phasor_from_lifetime(frequency, lifetime)
    assert_allclose(component_real, expected_real, atol=1e-6)
    assert_allclose(component_imag, expected_imag, atol=1e-6)
    assert_allclose(fractions, fraction, atol=1e-6)

    # boundary conditions
    nan = numpy.nan
    component_real, component_imag, fractions = phasor_component_search(
        [[0.0, 1.0, nan], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        num_components=2,
        frequency=frequency,
    )
    assert_allclose(
        component_real, [[1.0, 1.0, nan], [0.0, 1.0, nan]], atol=1e-3
    )
    assert_allclose(
        component_imag, [[0.0, 0.0, nan], [0.0, 0.0, nan]], atol=1e-3
    )
    assert_allclose(fractions, [[0.0, 0.0, nan], [1.0, 1.0, nan]], atol=1e-3)


@pytest.mark.parametrize('exact', [True, False])
def test_phasor_component_search_two_distribution(exact):
    """Test phasor_component_search function with two components."""
    # test that two lifetime components can be recovered from a distribution
    shape = (256, 256)
    frequency = 60.0
    lifetime = [0.5, 4.2]
    fraction = numpy.empty((*shape, 2))
    fraction[..., 0] = numpy.random.normal(0.3, 0.01, shape)
    fraction[..., 1] = 1.0 - fraction[..., 0]
    fraction = numpy.clip(fraction, 0.0, 1.0)

    real, imag = phasor_from_lifetime(
        [frequency, 2 * frequency], lifetime, fraction.reshape(-1, 2)
    )
    real = real.reshape(2, *shape)
    imag = imag.reshape(2, *shape)
    if not exact:
        # add noise to the imaginary parts
        imag += numpy.random.normal(0.0, 0.005, (2, *shape))
        dtype = 'float32'
    else:
        dtype = 'float64'

    component_real, component_imag, fractions = phasor_component_search(
        real,
        imag,
        num_components=2,
        frequency=frequency,
        lifetime_range=(0.4, 2.0, 0.01),
        dtype=dtype,
        num_threads=2,
    )

    lifetimes = phasor_to_normal_lifetime(
        component_real, component_imag, frequency
    )

    # _plot(frequency, real, imag, component_real, component_imag)

    assert not (component_real[1] > 0.5).all()
    assert _is_near_semicircle(component_real, component_imag, 1e-4).all()
    assert_allclose(lifetimes[0].mean(), lifetime[0], atol=1e-2)
    assert_allclose(lifetimes[1].mean(), lifetime[1], atol=1e-2)
    assert_allclose(fractions[0].mean(), 0.3, atol=1e-3)
    assert_allclose(fractions[1].mean(), 0.7, atol=1e-3)


def test_phasor_component_search_three():
    """Test phasor_component_search function with three components."""
    # scalar
    frequency = 80.0
    lifetime = [0.5, 4.2, 12.0]
    fraction = [0.3, 0.5, 0.2]
    real, imag = phasor_from_lifetime(
        [frequency, frequency * 2, frequency * 3], lifetime, fraction
    )
    component_real, component_imag, fractions = phasor_component_search(
        real,
        imag,
        frequency=frequency,
        num_components=3,
        lifetime_range=(0.4, 13, 0.1),
    )
    expected_real, expected_imag = phasor_from_lifetime(frequency, lifetime)
    assert_allclose(component_real, expected_real, atol=1e-6)
    assert_allclose(component_imag, expected_imag, atol=1e-6)
    assert_allclose(fractions, fraction, atol=1e-6)

    # boundary conditions
    nan = numpy.nan
    component_real, component_imag, fractions = phasor_component_search(
        [[0.0, 1.0, nan], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        num_components=3,
        frequency=frequency,
        lifetime_range=(0.0, 1.0, 0.01),
    )
    assert_allclose(
        component_real,
        [[nan, 1.0, nan], [nan, 1.0, nan], [nan, 1.0, nan]],
        atol=1e-3,
    )
    assert_allclose(
        component_imag,
        [[nan, 0.0, nan], [nan, 0.005, nan], [nan, 0.01, nan]],
        atol=1e-3,
    )
    assert_allclose(
        fractions,
        [[nan, 1.0, nan], [nan, 0.0, nan], [nan, 0.0, nan]],
        atol=1e-3,
    )


@pytest.mark.parametrize('exact', [True, False])
def test_phasor_component_search_three_distribution(exact):
    """Test phasor_component_search function with three components."""
    # test that three lifetime components can be recovered from a distribution
    shape = (32, 32)
    frequency = 40.0
    lifetime = [0.5, 4.2, 12.0]
    fraction = numpy.empty((*shape, 3))
    fraction[..., 0] = numpy.random.normal(0.3, 0.01, shape)
    fraction[..., 1] = numpy.random.normal(0.5, 0.01, shape)
    fraction[..., 2] = numpy.random.normal(0.2, 0.01, shape)
    fraction = numpy.clip(fraction, 0.0, 1.0)

    real, imag = phasor_from_lifetime(
        [frequency, 2 * frequency, 3 * frequency],
        lifetime,
        fraction.reshape(-1, 3),
    )
    real = real.reshape(3, *shape)
    imag = imag.reshape(3, *shape)
    if not exact:
        # add noise to the imaginary parts
        imag += numpy.random.normal(0.0, 0.001, (3, *shape))
        dtype = 'float32'
        atol = 0.5  # disable
    else:
        dtype = 'float64'
        atol = 1e-3

    component_real, component_imag, fractions = phasor_component_search(
        real,
        imag,
        num_components=3,
        frequency=frequency,
        lifetime_range=(0.4, 14.0, 0.1),
        dtype=dtype,
        num_threads=2,
    )

    # _plot(frequency, real, imag, component_real, component_imag)

    lifetimes = phasor_to_normal_lifetime(
        component_real, component_imag, frequency
    )

    assert not (component_real[1] > 0.5).all()
    assert _is_near_semicircle(component_real, component_imag, 1e-4).all()
    assert_allclose(lifetimes[0].mean(), lifetime[0], atol=atol)
    assert_allclose(lifetimes[1].mean(), lifetime[1], atol=atol)
    assert_allclose(lifetimes[2].mean(), lifetime[2], atol=atol)
    assert_allclose(fractions[0].mean(), 0.3, atol=atol)
    assert_allclose(fractions[1].mean(), 0.5, atol=atol)
    assert_allclose(fractions[2].mean(), 0.2, atol=atol)


def test_phasor_component_search_exceptions():
    real = [0.1, 0.2]
    imag = [0.4, 0.3]
    frequency = 60.0
    phasor_component_search(real, imag, 2, frequency)

    # shape mismatch
    with pytest.raises(ValueError):
        phasor_component_search(real, imag[0], 2, frequency)

    # no harmonics
    with pytest.raises(ValueError):
        phasor_component_search(real[0], imag[0], 2, frequency)

    # number of components < 2
    with pytest.raises(ValueError):
        phasor_component_search(real, imag, 1, frequency)

    # number of components does not match harmonics
    with pytest.raises(ValueError):
        phasor_component_search(real, imag, 3, frequency)

    # samples < 1
    with pytest.raises(ValueError):
        phasor_component_search(
            real, imag, 2, frequency, lifetime_range=(0, 1, 2)
        )

    # samples < 3 for 3 components
    with pytest.raises(ValueError):
        phasor_component_search(
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            3,
            frequency,
            lifetime_range=(0, 1, 0.5),
        )

    # dtype not float
    with pytest.raises(ValueError):
        phasor_component_search(real, imag, 2, frequency, dtype=numpy.int32)

    # too many components
    with pytest.raises(ValueError):
        phasor_component_search(
            [0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], 4, frequency
        )


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
