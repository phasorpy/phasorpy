"""Tests for the phasorpy.flim module."""

import os

import numpy
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from phasorpy.flim import (
    calibrate_phasor,
    calibration_parameters,
    center_of_mass,
)

SKIP_FETCH = os.environ.get('SKIP_FETCH', False)

synth_data_array = numpy.array([[50, 1], [1, 1]])
synth_data_list = [1, 2, 4]


def test_calibration_parameters():
    """Test calculation of phase and modulation correction parameters
    with real center > 0"""
    phi_correction, modulation_correction = calibration_parameters(
        synth_data_array, synth_data_array, 4, 80
    )
    assert_almost_equal(phi_correction, 0.3238654275023325)
    assert_almost_equal(modulation_correction, 0.023765245028289427)
    phi_correction, modulation_correction = calibration_parameters(
        synth_data_array * -1, synth_data_array, 4, 80
    )
    assert_almost_equal(phi_correction, -1.2469308992925638)
    assert_almost_equal(modulation_correction, 0.023765245028289427)


def test_calibration_center_function():
    """Test the argument for selection of method to be used to calculate
    the center of mass"""
    phi_correction, modulation_correction = calibration_parameters(
        synth_data_array,
        synth_data_array,
        4,
        80,
        center_function='spatial_median',
    )
    assert_almost_equal(phi_correction, 0.3238654275023325)
    assert_almost_equal(modulation_correction, 0.31488949662483484)
    phi_correction, modulation_correction = calibration_parameters(
        synth_data_array,
        synth_data_array,
        4,
        80,
        center_function='geometric_median',
    )
    assert_almost_equal(phi_correction, 0.3238654275023325)
    assert_almost_equal(modulation_correction, 0.31488949662483484)


def test_center_methods_array():
    """Test all possible methods of the `center_of_mass` function with
    an array as  input"""
    x_center, y_center = center_of_mass(
        synth_data_array, synth_data_array, method='mean'
    )
    assert_almost_equal(x_center, 13.25)
    assert_almost_equal(y_center, 13.25)
    x_center, y_center = center_of_mass(
        synth_data_array, synth_data_array, method='spatial_median'
    )
    assert_almost_equal(x_center, 1.0)
    assert_almost_equal(y_center, 1.0)
    x_center, y_center = center_of_mass(
        synth_data_array, synth_data_array, method='geometric_median'
    )
    assert_almost_equal(x_center, 1.0)
    assert_almost_equal(y_center, 1.0)


def test_center_methods_list():
    """Test all possible methods of the `center_of_mass` function with a
    list as  input"""
    x_center, y_center = center_of_mass(
        synth_data_list, synth_data_list, method='mean'
    )
    assert_almost_equal(x_center, 2.3333333333333335)
    assert_almost_equal(y_center, 2.3333333333333335)
    x_center, y_center = center_of_mass(
        synth_data_list, synth_data_list, method='spatial_median'
    )
    assert_almost_equal(x_center, 2.0)
    assert_almost_equal(y_center, 2.0)
    x_center, y_center = center_of_mass(
        synth_data_list, synth_data_list, method='geometric_median'
    )
    assert_almost_equal(x_center, 2.0)
    assert_almost_equal(y_center, 2.0)


def test_center_methods_scalar():
    """Test all possible methods of the `center_of_mass` function with a scalar
    as  input"""
    x_center, y_center = center_of_mass(2, 2, method='mean')
    assert_almost_equal(x_center, 2.0)
    assert_almost_equal(y_center, 2.0)
    x_center, y_center = center_of_mass(2, 2, method='spatial_median')
    assert_almost_equal(x_center, 2.0)
    assert_almost_equal(y_center, 2.0)
    x_center, y_center = center_of_mass(2, 2, method='geometric_median')
    assert_almost_equal(x_center, 2.0)
    assert_almost_equal(y_center, 2.0)


def test_center_methods_error():
    """Test method not supported for center calculation"""
    with pytest.raises(ValueError):
        center_of_mass(0, 0, method='method_not_supported')


def test_phasor_calibration_unchanged():
    """Test calibration of phasor data with phase correction = 0 and modulation
    correction =1, which must return the original phasor data unchanged"""
    (
        array_real_phasor_calibrated,
        array_imaginary_phasor_calibrated,
    ) = calibrate_phasor(synth_data_array, synth_data_array, (0, 1))
    assert_allclose(array_real_phasor_calibrated, synth_data_array)
    assert_allclose(array_imaginary_phasor_calibrated, synth_data_array)
    (
        list_real_phasor_calibrated,
        list_imaginary_phasor_calibrated,
    ) = calibrate_phasor(synth_data_list, synth_data_list, (0, 1))
    assert_allclose(list_real_phasor_calibrated, numpy.array(synth_data_list))
    assert_allclose(
        list_imaginary_phasor_calibrated, numpy.array(synth_data_list)
    )
    (
        scalar_real_phasor_calibrated,
        scalar_imaginary_phasor_calibrated,
    ) = calibrate_phasor(2, 2, (0, 1))
    assert_almost_equal(scalar_real_phasor_calibrated, 2.0)
    assert_almost_equal(scalar_imaginary_phasor_calibrated, 2.0)


def test_phasor_calibration_with_parameters():
    (
        array_real_phasor_calibrated,
        array_imaginary_phasor_calibrated,
    ) = calibrate_phasor(synth_data_array, synth_data_array, (0.5, 0.5))
    expected_real_array = numpy.array(
        [[9.95392558, 0.19907851], [0.19907851, 0.19907851]]
    )
    expected_imaginary_array = numpy.array(
        [[33.92520251, 0.67850405], [0.67850405, 0.67850405]]
    )
    assert_allclose(array_real_phasor_calibrated, expected_real_array)
    assert_allclose(
        array_imaginary_phasor_calibrated, expected_imaginary_array
    )
    (
        list_real_phasor_calibrated,
        list_imaginary_phasor_calibrated,
    ) = calibrate_phasor(synth_data_list, synth_data_list, (0.5, 0.5))
    expected_real_from_list = numpy.array([0.19907851, 0.39815702, 0.79631405])
    expected_imaginary_from_list = numpy.array(
        [0.67850405, 1.3570081, 2.7140162]
    )
    assert_allclose(list_real_phasor_calibrated, expected_real_from_list)
    assert_allclose(
        list_imaginary_phasor_calibrated, expected_imaginary_from_list
    )
    (
        scalar_real_phasor_calibrated,
        scalar_imaginary_phasor_calibrated,
    ) = calibrate_phasor(2, 2, (0.5, 0.5))
    assert_almost_equal(scalar_real_phasor_calibrated, 0.39815702328616975)
    assert_almost_equal(scalar_imaginary_phasor_calibrated, 1.3570081004945758)
