"""Tests for the phasorpy.components module."""

from math import nan as NAN

import numpy
import pytest
from numpy.testing import assert_allclose

from phasorpy.component import (
    phasor_component_fit,
    phasor_component_fraction,
    phasor_component_graphical,
    phasor_component_mvc,
    phasor_from_component,
)
from phasorpy.lifetime import phasor_from_lifetime

numpy.random.seed(42)


@pytest.mark.parametrize('swap', [False, True])
@pytest.mark.parametrize('func', [phasor_component_mvc, phasor_component_fit])
def test_three_components(swap, func):
    """Test functions can calculate barycentric coordinates within triangle."""
    frequency = 40.0
    lifetime = [0.5, 4.2, 12.0]
    fraction = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0.0, 0.6, 0.4],
        [0.6, 0.0, 0.4],
        [0.6, 0.4, 0.0],
        [0.3, 0.5, 0.2],
        [1 / 3, 1 / 3, 1 / 3],
        [NAN, NAN, NAN],
    ]
    if swap:
        lifetime = [lifetime[i] for i in (1, 0, 2)]
    real, imag = phasor_from_lifetime(frequency, lifetime, fraction)

    if func is phasor_component_fit:
        result = phasor_component_fit(
            numpy.ones_like(real), real, imag, real[:3], imag[:3]
        )
    elif func is phasor_component_mvc:
        result = phasor_component_mvc(real, imag, real[:3], imag[:3])
    assert_allclose(result, numpy.asarray(fraction).T, atol=1e-6)


def test_phasor_component_mvc():
    """Test phasor_component_mvc function."""
    component_real = [1.0, 0.0, 0.0, -0.5]
    component_imag = [0.0, -0.5, 1.0, 0.0]
    fraction = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0.6, 0.4, 0.0, 0.0],
        [0.6, 0.0, 0.4, 0.0],
        [0.6, 0.0, 0.0, 0.4],
        [0.0, 0.6, 0.4, 0.0],
        [0.0, 0.6, 0.0, 0.4],
        [0.0, 0.0, 0.6, 0.4],
        [0.6, 0.3, 0.1, 0.0],
        [0.6, 0.3, 0.0, 0.1],
        [0.6, 0.0, 0.3, 0.1],
        [0.0, 0.6, 0.3, 0.1],
        [0.3, 0.4, 0.2, 0.1],
        [-0.1, 0.4, -0.2, 0.9],
        [1 / 4, 1 / 4, 1 / 4, 1 / 4],
        [NAN, NAN, NAN, NAN],
    ]

    real, imag = phasor_from_component(
        component_real, component_imag, fraction, axis=-1
    )

    result = phasor_component_mvc(real, imag, component_real, component_imag)

    # a four component system is underdetermined and fractions cannot
    # be restored unambiguously
    # assert_allclose(result, numpy.asarray(fraction).T, atol=1e-6)

    # instead verify that phasor coordinates can be restored from result
    real_restored, imag_restored = phasor_from_component(
        component_real, component_imag, result, axis=0
    )
    assert_allclose(real_restored, real, atol=1e-6)
    assert_allclose(imag_restored, imag, atol=1e-6)

    with pytest.raises(ValueError):
        phasor_component_mvc(
            real, imag, component_real[:2], component_imag[:2]
        )

    with pytest.raises(ValueError):
        phasor_component_mvc(real, imag, component_real, component_imag[:3])

    with pytest.raises(ValueError):
        phasor_component_mvc(real[:4], imag, component_real, component_imag)

    with pytest.raises(ValueError):
        phasor_component_mvc(real, imag, [NAN, 1, 1, 1], component_imag)

    with pytest.raises(ValueError):
        phasor_component_mvc(real, imag, [numpy.inf, 1, 1, 1], component_imag)

    with pytest.raises(ValueError):
        phasor_component_mvc(
            real, imag, component_real, component_imag, dtype=numpy.int32
        )


@pytest.mark.parametrize('num_components', [3, 4, 5])
def test_phasor_component_mvc_plot(num_components):
    """Test phasor_component_mvc function visually."""
    from matplotlib import pyplot

    show = False  # enable to see figure

    coords = numpy.linspace(-0.05, 1.05, 501)
    real, imag = numpy.meshgrid(coords, coords)

    dtype = numpy.float32
    angles = numpy.linspace(0, 2 * numpy.pi, num_components, endpoint=False)
    component_real = 0.5 + 0.5 * numpy.cos(angles)
    component_imag = 0.5 + 0.5 * numpy.sin(angles)
    component_real[0] = 0.500001  # numerical instabilities with 0.5, float32

    fraction = phasor_component_mvc(
        real,
        imag,
        component_real,
        component_imag,
        num_threads=3,
        dtype=dtype,
    )

    assert fraction.dtype == dtype

    real_restored, imag_restored = phasor_from_component(
        component_real, component_imag, fraction, axis=0
    )
    assert_allclose(real_restored, real, atol=1e-6)
    assert_allclose(imag_restored, imag, atol=1e-6)

    _, axs = pyplot.subplots(
        1,
        num_components,
        figsize=(4 * num_components, 4),
        layout='constrained',
    )
    for i, ax in enumerate(axs):
        ax.imshow(fraction[i], vmin=0.0, vmax=1.0)
        ax.plot(
            component_real * 455 + 23,
            component_imag * 455 + 23,
            'r.',
            markersize=10,
        )
        ax.set(xticks=[], yticks=[])
    if show:
        pyplot.show()
    else:
        pyplot.close()


def test_phasor_from_component():
    """Test phasor_from_component function."""
    frequency = 40.0
    lifetime = [0.5, 4.2, 12.0]
    fraction = numpy.asarray(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.3, 0.5, 0.2],
            [0.0, 0.6, 0.4],
            [0.0, 0.0, 0.0],  # NaN
        ]
    )
    known_real, known_imag = phasor_from_lifetime(
        frequency, lifetime, fraction
    )

    real, imag = phasor_from_component(
        known_real[:3], known_imag[:3], fraction, axis=-1
    )
    assert_allclose(real, known_real, atol=1e-6)
    assert_allclose(imag, known_imag, atol=1e-6)

    real, imag = phasor_from_component(
        known_real[:3], known_imag[:3], fraction, axis=-1, dtype=numpy.float32
    )
    assert real.dtype == numpy.float32
    assert_allclose(real, known_real, atol=1e-6)
    assert_allclose(imag, known_imag, atol=1e-6)

    with pytest.raises(ValueError):
        real, imag = phasor_from_component(
            known_real[:3], known_imag[:3], fraction, axis=-1, dtype='int32'
        )
    with pytest.raises(ValueError):
        phasor_from_component(
            numpy.ones((2, 2)), numpy.ones((2, 2)), numpy.ones((2, 2))
        )
    with pytest.raises(ValueError):
        phasor_from_component(known_real[:3], known_imag[:3], fraction[0, 0])
    with pytest.raises(ValueError):
        phasor_from_component(
            known_real[:3], known_imag[:3], fraction[:, :1], axis=-1
        )
    with pytest.raises(ValueError):
        phasor_from_component(known_real[:3], known_imag[:3], fraction)
    with pytest.raises(ValueError):
        phasor_from_component(
            known_real[:3], known_imag[:2], fraction, axis=-1
        )
    with pytest.raises(ValueError):
        phasor_from_component(
            known_real[:2], known_imag[:2], fraction, axis=-1
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
            [0, 0, 1, 0, 0, 0],
        ),
        # Two components, phasors as list. Increase cursor radius.
        (
            [0.6, 0.4],
            [0.35, 0.38],
            [0.2, 0.9],
            [0.4, 0.3],
            0.15,
            [0, 0.2, 0.4, 0.6, 0.8, 1],
            [0, 0, 1, 2, 1, 0],
        ),
        # Two components, phasors as list. Increase number of steps.
        (
            [0.6, 0.4],
            [0.35, 0.38],
            [0.2, 0.9],
            [0.4, 0.3],
            0.05,
            11,
            [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
        ),
        # Three components, phasor as scalar
        (
            0.3,
            0.2,
            [0.0, 0.2, 0.9],
            [0.0, 0.4, 0.3],
            0.05,
            [0, 0.2, 0.4, 0.6, 0.8, 1],
            [[0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 1, 1, 0, 0]],
        ),
        # Three components, phasors as list
        (
            [0.3, 0.5],
            [0.2, 0.3],
            [0.0, 0.2, 0.9],
            [0.0, 0.4, 0.3],
            0.05,
            6,
            [[0, 1, 1, 1, 0, 0], [0, 1, 0, 1, 0, 0], [0, 0, 2, 1, 0, 0]],
        ),
        # Phasor outside semicircle but inside cursor of component 1
        (
            [0.4, 0.82],
            [0.38, 0.4],
            [0.8, 0.2, 0.042],
            [0.4, 0.4, 0.2],
            0.05,
            5,
            [[0, 0, 1, 0, 1], [0, 0, 0, 1, 2], [1, 1, 2, 2, 2]],
        ),
        # Phasor outside semicircle and outside cursor of component 1
        (
            [0.4, 0.86],
            [0.38, 0.4],
            [0.8, 0.2, 0.042],
            [0.4, 0.4, 0.2],
            0.05,
            [0, 0.25, 0.5, 0.75, 1],
            [[0, 0, 1, 0, 0], [0, 0, 0, 1, 1], [0, 0, 1, 1, 1]],
        ),
        # Two components no fractions provided
        (
            0,
            0,
            [0, 0],
            [0, 0.2],
            0.05,
            None,
            [0, 0, 0, 0, 0, 0, 1, 1, 1],
        ),
        # Three components no fractions provided
        (
            0,
            0,
            [0, 0.1, 0],
            [0, 0.1, 0.2],
            0.05,
            None,
            [
                [0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
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
    assert_allclose(actual_counts, expected_counts)


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
    mean[0] = NAN
    real[1] = NAN
    fractions[0][:2] = NAN
    fractions[1][:2] = NAN
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

    component_real[0] = NAN
    with pytest.raises(ValueError):
        phasor_component_fit(mean, real, imag, component_real, component_real)

    component_imag[0] = numpy.inf
    with pytest.raises(ValueError):
        phasor_component_fit(mean, real, imag, component_imag, component_imag)


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type, call-overload"
