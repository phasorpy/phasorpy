"""Tests for the phasorpy.cursors module."""

import numpy
import pytest
from numpy.testing import assert_array_equal

from phasorpy.cursors import *


def test_label_from_phasor_circular():
    # real, imag, center, radius=radius
    real = numpy.array([-0.5, -0.5, 0.5, 0.5])
    imag = numpy.array([-0.5, 0.5, -0.5, 0.5])
    center = numpy.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    radius = [0.1, 0.1, 0.1, 0.1]
    labels = label_from_phasor_circular(real, imag, center, radius=radius)
    assert labels.dtype == 'uint8'
    assert_array_equal(labels, [1.0, 2.0, 3.0, 4.0])


def test_label_from_phasor_circular_erros():
    # Test ValueErrors
    real = numpy.array([-0.5, -0.5, 0.5, 0.5])
    imag = numpy.array([-0.5, 0.5, -0.5, 0.5])
    center = numpy.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    radius = [0.1, 0.1, 0.1, 0.1]
    with pytest.raises(ValueError):
        label_from_phasor_circular(
            real, imag, center, radius=[-0.1, 0.1, 0.1, 0.1]
        )
    with pytest.raises(ValueError):
        label_from_phasor_circular(
            numpy.array([-0.5, -0.5, 0.5]), imag, center, radius=radius
        )


def test_mask_from_cursor():
    xarray = [[337, 306, 227], [21, 231, 235], [244, 328, 116]]
    yarray = [[0.22, 0.40, 0.81], [0.33, 0.43, 0.36], [0.015, 0.82, 0.58]]
    mask = mask_from_cursor(
        xarray=xarray, yarray=yarray, xrange=[0, 270], yrange=[0, 0.5]
    )
    mk = [[False, False, False], [True, True, True], [True, False, False]]
    assert_array_equal(mask, mk)


def test_mask_from_cursor_erros():
    # Test ValueErrors
    xarray = [[337, 306], [21, 231, 235], [244, 328, 116]]
    yarray = [[0.22, 0.40, 0.81], [0.33, 0.43, 0.36], [0.015, 0.82, 0.58]]
    with pytest.raises(ValueError):
        mask_from_cursor(
            xarray=xarray, yarray=yarray, xrange=[0, 270], yrange=[0, 0.5]
        )

    xarray = [[337, 306, 227], [21, 231, 235], [244, 328, 116]]
    with pytest.raises(ValueError):
        mask_from_cursor(
            xarray=xarray, yarray=yarray, xrange=[0], yrange=[0, 0.5]
        )
