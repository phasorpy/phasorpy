"""Tests for the phasorpy.flim module."""

import numpy
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from phasorpy.flim import (  # polar_center_of_mass,
    calibrate_phasor,
    calibration_parameters,
    cartesian_center_of_mass,
)

SYNTH_DATA_ARRAY = numpy.array([[50, 1], [1, 1]])
SYNTH_DATA_LIST = [1, 2, 4]


@pytest.mark.parametrize(
    "input_array, expected_phi, expected_modulation",
    [
        (SYNTH_DATA_ARRAY, 0.3238654275023325, 0.023765245028289427),
        (SYNTH_DATA_ARRAY * -1, -1.2469308992925638, 0.023765245028289427),
    ],
)
def test_calibration_parameters(
    input_array, expected_phi, expected_modulation
):
    phi_correction, modulation_correction = calibration_parameters(
        4, 80, reference_real=input_array, reference_imag=SYNTH_DATA_ARRAY
    )
    assert_almost_equal(phi_correction, expected_phi)
    assert_almost_equal(modulation_correction, expected_modulation)


@pytest.mark.parametrize(
    "center_function, expected_phi, expected_modulation",
    [
        ('spatial_median', 0.3238654275023325, 0.31488949662483484),
        ('geometric_median', 0.3238654275023325, 0.31488949662483484),
    ],
)
def test_calibration_center_function(
    center_function, expected_phi, expected_modulation
):
    phi_correction, modulation_correction = calibration_parameters(
        4,
        80,
        reference_real=SYNTH_DATA_ARRAY,
        reference_imag=SYNTH_DATA_ARRAY,
        center_function=center_function,
    )
    assert_almost_equal(phi_correction, expected_phi)
    assert_almost_equal(modulation_correction, expected_modulation)


@pytest.mark.parametrize(
    "method, expected_x, expected_y",
    [
        ('mean', 13.25, 13.25),
        ('spatial_median', 1.0, 1.0),
        ('geometric_median', 1.0, 1.0),
    ],
)
def test_center_methods_array(method, expected_x, expected_y):
    x_center, y_center = cartesian_center_of_mass(
        SYNTH_DATA_ARRAY, SYNTH_DATA_ARRAY, method=method
    )
    assert_almost_equal(x_center, expected_x)
    assert_almost_equal(y_center, expected_y)


@pytest.mark.parametrize(
    "method, expected_x, expected_y",
    [
        ('mean', 2.3333333333333335, 2.3333333333333335),
        ('spatial_median', 2.0, 2.0),
        ('geometric_median', 2.0, 2.0),
    ],
)
def test_center_methods_list(method, expected_x, expected_y):
    x_center, y_center = cartesian_center_of_mass(
        SYNTH_DATA_LIST, SYNTH_DATA_LIST, method=method
    )
    assert_almost_equal(x_center, expected_x)
    assert_almost_equal(y_center, expected_y)


@pytest.mark.parametrize(
    "method", ['mean', 'spatial_median', 'geometric_median']
)
def test_center_methods_scalar(method):
    """Test all possible methods of the `cartesian_center_of_mass` function
    with a scalar as  input"""
    x_center, y_center = cartesian_center_of_mass(2, 2, method=method)
    assert_almost_equal(x_center, 2.0)
    assert_almost_equal(y_center, 2.0)


def test_center_methods_error():
    """Test method not supported for center calculation"""
    with pytest.raises(ValueError):
        cartesian_center_of_mass(0, 0, method='method_not_supported')


@pytest.mark.parametrize(
    "input_data, expected_real, expected_imaginary",
    [
        (SYNTH_DATA_ARRAY, SYNTH_DATA_ARRAY, SYNTH_DATA_ARRAY),
        (
            SYNTH_DATA_LIST,
            numpy.array(SYNTH_DATA_LIST),
            numpy.array(SYNTH_DATA_LIST),
        ),
        (2, 2.0, 2.0),
    ],
)
def test_phasor_calibration_unchanged(
    input_data, expected_real, expected_imaginary
):
    """Test calibration of phasor data with phase correction = 0 and modulation
    correction =1, which must return the original phasor data unchanged"""
    real_phasor_calibrated, imaginary_phasor_calibrated = calibrate_phasor(
        0, 1, real=input_data, imag=input_data
    )
    assert_allclose(real_phasor_calibrated, expected_real)
    assert_allclose(imaginary_phasor_calibrated, expected_imaginary)


@pytest.mark.parametrize(
    "input_data, expected_real, expected_imaginary",
    [
        (
            SYNTH_DATA_ARRAY,
            numpy.array([[9.95392558, 0.19907851], [0.19907851, 0.19907851]]),
            numpy.array([[33.92520251, 0.67850405], [0.67850405, 0.67850405]]),
        ),
        (
            SYNTH_DATA_LIST,
            numpy.array([0.19907851, 0.39815702, 0.79631405]),
            numpy.array([0.67850405, 1.3570081, 2.7140162]),
        ),
        (2, 0.39815702328616975, 1.3570081004945758),
    ],
)
def test_phasor_calibration_with_parameters(
    input_data, expected_real, expected_imaginary
):
    (
        real_phasor_calibrated,
        imaginary_phasor_calibrated,
    ) = calibrate_phasor(0.5, 0.5, real=input_data, imag=input_data)

    assert_allclose(real_phasor_calibrated, expected_real)
    assert_allclose(imaginary_phasor_calibrated, expected_imaginary)
