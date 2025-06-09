"""Tests handling NaN coordinates."""

import warnings

import numpy
import pytest
from matplotlib import pyplot
from numpy import nan
from numpy.testing import assert_allclose, assert_array_equal

from phasorpy.components import (
    phasor_component_fraction,
    phasor_component_graphical,
)
from phasorpy.cursors import (
    mask_from_circular_cursor,
    mask_from_elliptic_cursor,
    mask_from_polar_cursor,
)
from phasorpy.phasor import (
    phasor_at_harmonic,
    phasor_calibrate,
    phasor_center,
    phasor_divide,
    phasor_filter_median,
    phasor_from_apparent_lifetime,
    phasor_from_polar,
    phasor_from_signal,
    phasor_multiply,
    phasor_threshold,
    phasor_to_apparent_lifetime,
    phasor_to_complex,
    phasor_to_normal_lifetime,
    phasor_to_polar,
    phasor_to_principal_plane,
    phasor_to_signal,
    phasor_transform,
    polar_from_apparent_lifetime,
    polar_from_reference,
    polar_from_reference_phasor,
    polar_to_apparent_lifetime,
)
from phasorpy.plot import PhasorPlot, plot_phasor_image

VALUES_WITH_NAN = [1.1, 1.1, 1.1], [0.5, nan, 0.1], [0.5, 0.5, 0.1]
VALUES_WITH_NAN_2D = (
    [[1.1, 1.1, 1.1], [1.1, 1.1, 1.1], [1.1, 1.1, 1.1]],
    [[0.5, nan, 0.1], [0.5, 0.5, 0.1], [0.2, 0.3, nan]],
    [[0.5, 0.5, 0.1], [0.2, nan, 0.3], [0.4, 0.4, 0.4]],
)


def test_phasorplot_nan():
    """Test PhasorPlot methods with NaN values."""
    plot = PhasorPlot()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        plot.plot(*VALUES_WITH_NAN[1:])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        plot.hist2d(*VALUES_WITH_NAN[1:])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        plot.contour(*VALUES_WITH_NAN[1:])

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        plot.components(*VALUES_WITH_NAN[1:])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        plot.line(*VALUES_WITH_NAN[1:])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        plot.circle(nan, 0.5, radius=0.1)
    with pytest.raises(ValueError):
        plot.cursor(nan, 0.5, radius=0.1)
    # pyplot.show()
    pyplot.close()


def test_plot_phasor_image_nan():
    """Test plot_phasor_image function with NaN values."""
    data = numpy.zeros((2, 3, 2))
    data[0, 0, 0] = numpy.nan
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        plot_phasor_image(data[0], data, data, show=False)
    pyplot.close()


def test_mask_from_circular_cursor_nan():
    """Test mask_from_circular_cursor function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        mask = mask_from_circular_cursor(
            *VALUES_WITH_NAN[1:], 0.55, 0.55, radius=0.1
        )
    assert_allclose(mask, [True, False, False])


def test_mask_from_elliptic_cursor_nan():
    """Test mask_from_elliptic_cursor function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        mask = mask_from_elliptic_cursor(
            *VALUES_WITH_NAN[1:], 0.55, 0.55, radius=0.1
        )
    assert_allclose(mask, [True, False, False])


def test_mask_from_polar_cursor_nan():
    """Test mask_from_polar_cursor function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        mask = mask_from_polar_cursor(*VALUES_WITH_NAN[1:], 0, 1, 0.6, 0.8)
    assert_allclose(mask, [True, False, False])


def test_phasor_at_harmonic_nan():
    """Test phasor_at_harmonic function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        phasor = phasor_at_harmonic(VALUES_WITH_NAN[1], 1, 2)
    assert_allclose(
        phasor, [[0.2, nan, 0.027027], [0.4, nan, 0.162162]], atol=1e-3
    )


@pytest.mark.parametrize('nan_safe', (True, False))
def test_phasor_calibrate_nan(nan_safe):
    """Test phasor_calibrate function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        phasor = phasor_calibrate(
            *VALUES_WITH_NAN[1:], 1.0, 0.0, 1.0, 80, 4.2, nan_safe=nan_safe
        )
    assert_allclose(
        phasor, [[0.28506, nan, 0.05701], [0.10181, nan, 0.02036]], atol=1e-3
    )


def test_phasor_center_nan():
    """Test phasor_center function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        center = phasor_center(*VALUES_WITH_NAN)
    assert_allclose(center, [1.1, 0.3, 0.3], atol=1e-3)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        center = phasor_center(*VALUES_WITH_NAN, nan_safe=False)
    assert_allclose(center, [1.1, 0.3, 0.366667], atol=1e-3)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        center = phasor_center(*VALUES_WITH_NAN, method='median')
    assert_allclose(center, [1.1, 0.3, 0.3], atol=1e-3)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        center = phasor_center(
            *VALUES_WITH_NAN, method='median', nan_safe=False
        )
    assert_allclose(center, [1.1, 0.3, 0.5], atol=1e-3)


def test_phasor_divide_nan():
    """Test phasor_divide function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        phasor = phasor_divide(*VALUES_WITH_NAN[1:], 1.0, 1.0)
    assert_allclose(phasor, [[0.5, nan, 0.1], [0.0, nan, 0.0]], atol=1e-3)


def test_phasor_filter_median_nan():
    """Test phasor_filter_median function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        phasor = phasor_filter_median(*VALUES_WITH_NAN_2D)
    assert_array_equal(phasor[0], VALUES_WITH_NAN_2D[0])
    assert_allclose(
        phasor[1],
        [[0.5, nan, 0.1], [0.5, 0.3, 0.1], [0.3, 0.3, nan]],
        atol=1e-3,
    )
    assert_allclose(
        phasor[2],
        [[0.5, 0.4, 0.2], [0.4, nan, 0.35], [0.4, 0.4, 0.4]],
        atol=1e-3,
    )


def test_phasor_from_apparent_lifetime_nan():
    """Test phasor_from_apparent_lifetime function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        phasor = phasor_from_apparent_lifetime(*VALUES_WITH_NAN[1:], 80)
    assert_allclose(
        phasor,
        [[0.940587, nan, 0.99748], [0.236395, nan, 0.050139]],
        atol=1e-3,
    )


def test_phasor_from_polar_nan():
    """Test phasor_from_polar function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        phasor = phasor_from_polar(*VALUES_WITH_NAN[1:])
    assert_allclose(
        phasor,
        [[0.438791, nan, 0.0995], [0.239713, nan, 0.009983]],
        atol=1e-3,
    )


def test_phasor_from_signal_nan():
    """Test phasor_from_signal function with NaN values."""
    sample_phase = numpy.linspace(0, 2 * numpy.pi, 4, endpoint=False)
    signal = 1.1 * (numpy.cos(sample_phase - 0.4) * 0.8 + 1)
    signal = numpy.stack((signal, signal))
    signal[0, 2] = numpy.nan
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        phasor = phasor_from_signal(signal)
    assert_allclose(
        phasor,
        [[nan, 1.1], [nan, 0.368424], [nan, 0.155767]],
        atol=1e-3,
    )


def test_phasor_multiply_nan():
    """Test phasor_multiply function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        phasor = phasor_multiply(*VALUES_WITH_NAN[1:], 1.0, 1.0)
    assert_allclose(phasor, [[0.0, nan, 0.0], [1.0, nan, 0.2]], atol=1e-3)


def test_phasor_threshold_nan():
    """Test phasor_threshold function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        phasor = phasor_threshold(*VALUES_WITH_NAN, real_min=0.2)
    assert_allclose(
        phasor, [[1.1, nan, nan], [0.5, nan, nan], [0.5, nan, nan]], atol=1e-3
    )


def test_phasor_to_apparent_lifetime_nan():
    """Test phasor_to_apparent_lifetime function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        lifetimes = phasor_to_apparent_lifetime(*VALUES_WITH_NAN[1:], 80)
    assert_allclose(
        lifetimes,
        [[1.989437, nan, 1.989437], [1.989437, nan, 13.926058]],
        atol=1e-3,
    )


def test_phasor_to_complex_nan():
    """Test phasor_to_complex function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        phasor = phasor_to_complex(*VALUES_WITH_NAN[1:])
    assert_allclose(
        phasor,
        [0.5 + 0.5j, nan + 0.5j, 0.1 + 0.1j],
        atol=1e-3,
    )


def test_phasor_to_polar_nan():
    """Test phasor_to_polar function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        polar = phasor_to_polar(*VALUES_WITH_NAN[1:])
    assert_allclose(
        polar,
        [[0.785398, nan, 0.785398], [0.707107, nan, 0.141421]],
        atol=1e-3,
    )


def test_phasor_to_principal_plane_nan():
    """Test phasor_to_principal_plane function with NaN values."""
    real = [[0.495, 0.35], [0.354, 0.304], [0.3, 0.37]]
    imag = [[0.333, 0.33], [0.301, 0.349], [0.3, 0.36]]

    x, y, transformation_matrix = phasor_to_principal_plane(real, imag)
    # assert_allclose(x, [0.458946, 0.202887], atol=1e-3)

    real[0][0] = nan
    with pytest.raises(Exception):
        # numpy.linalg.LinAlgError: Eigenvalues did not converge
        x, y, transformation_matrix = phasor_to_principal_plane(real, imag)


def test_phasor_to_signal_nan():
    """Test phasor_to_signal function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        signal = phasor_to_signal(*VALUES_WITH_NAN, samples=16)
    assert signal.shape == (3, 16)
    assert_allclose(
        signal[0][:9],
        [2.2, 2.5, 2.6, 2.5, 2.2, 1.69, 1.1, 0.5, 0.0],
        atol=1e-1,
    )
    # TODO: this might be dependent on FFT implementation?
    assert_allclose(signal[1][:4], [nan, nan, nan, nan], atol=1e-3)


def test_phasor_transform_nan():
    """Test phasor_transform function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        phasor = phasor_transform(*VALUES_WITH_NAN[1:], 1, 0.9)
    assert_allclose(
        phasor,
        [[-0.135526, nan, -0.027105], [0.621798, nan, 0.12436]],
        atol=1e-3,
    )


def test_polar_from_apparent_lifetime_nan():
    """Test polar_from_apparent_lifetime function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        polar = polar_from_apparent_lifetime(*VALUES_WITH_NAN[1:], 80)
    assert_allclose(
        polar,
        [[0.246228, nan, 0.050223], [0.969839, nan, 0.998739]],
        atol=1e-3,
    )


def test_polar_from_reference_nan():
    """Test polar_from_reference function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        polar = polar_from_reference(*VALUES_WITH_NAN[1:], 0.5, 0.5)
    assert_allclose(
        polar,
        [[0.0, nan, 0.4], [1.0, nan, 5.0]],
        atol=1e-3,
    )


def test_polar_from_reference_phasor_nan():
    """Test polar_from_reference_phasor function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        polar = polar_from_reference_phasor(*VALUES_WITH_NAN[1:], 0.5, 0.5)
    assert_allclose(
        polar,
        [[0.0, nan, 0.0], [1.0, nan, 5.0]],
        atol=1e-3,
    )


def test_polar_to_apparent_lifetime_nan():
    """Test polar_to_apparent_lifetime function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        lifetime = polar_to_apparent_lifetime(*VALUES_WITH_NAN[1:], 80)
    assert_allclose(
        lifetime,
        [[1.086834, nan, 0.199609], [3.445806, nan, 19.794646]],
        atol=1e-3,
    )


def test_phasor_to_normal_lifetime_nan():
    """Test phasor_to_normal_lifetime function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        taunorm = phasor_to_normal_lifetime(*VALUES_WITH_NAN[1:], 80)
    assert_allclose(taunorm, [1.989437, nan, 16.160405], atol=1e-3)


def test_phasor_component_fraction_nan():
    """Test phasor_component_fraction function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        fractions = phasor_component_fraction(
            *VALUES_WITH_NAN[1:], [0.5, 1.0], [0.5, 1.0]
        )
    assert_allclose(fractions, [1.0, nan, 1.0])


def test_phasor_component_graphical_nan():
    """Test phasor_component_graphical function with NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        counts = phasor_component_graphical(
            *VALUES_WITH_NAN[1:],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            fractions=10,
        )
    assert_allclose(
        counts,
        (
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        ),
    )


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
