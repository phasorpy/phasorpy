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


def test_create_lut():
    # min1, max1, min2, max2
    min_vals1 = numpy.array([0, 3, 6])
    max_vals1 = numpy.array([2, 5, 8])
    min_vals2 = numpy.array([1, 4, 7])
    max_vals2 = numpy.array([3, 6, 9])
    lut = create_lut(min_vals1, max_vals1, min_vals2, max_vals2)
    expected_lut = {
        ((0, 2), (1, 3)): 1,
        ((0, 2), (4, 6)): 2,
        ((0, 2), (7, 9)): 3,
        ((3, 5), (1, 3)): 4,
        ((3, 5), (4, 6)): 5,
        ((3, 5), (7, 9)): 6,
        ((6, 8), (1, 3)): 7,
        ((6, 8), (4, 6)): 8,
        ((6, 8), (7, 9)): 9
    }
    assert_array_equal(lut, expected_lut)


def test_create_lut_errors():
    min_vals1 = numpy.array([0, 3])
    max_vals1 = numpy.array([2, 5, 8])
    min_vals2 = numpy.array([1, 4, 7])
    max_vals2 = numpy.array([3, 6, 9])
    with pytest.raises(ValueError):
        create_lut(min_vals1, max_vals1, min_vals2, max_vals2)


def test_label_from_lut():
    arr1 = numpy.array([[1.2, 2.4, 3.5], [4.7, 5.1, 6.9], [7.3, 8.6, 9.0]])
    arr2 = numpy.array([[0.8, 2.1, 3.9], [4.2, 5.7, 6.3], [7.5, 8.2, 9.5]])
    lut = {
        ((0, 2), (1, 3)): 1,
        ((0, 2), (4, 6)): 2,
        ((0, 2), (7, 9)): 3,
        ((3, 5), (1, 3)): 4,
        ((3, 5), (4, 6)): 5,
        ((3, 5), (7, 9)): 6,
        ((6, 8), (1, 3)): 7,
        ((6, 8), (4, 6)): 8,
        ((6, 8), (7, 9)): 9
    }
    label = label_from_lut(arr1, arr2, lut)
    assert_array_equal(label, [[0, 0, 0], [5, 0, 0], [9, 0, 0]])


def test_label_from_lut_errors():
    arr1 = numpy.array([[1.2, 2.4, 3.5], [7.3, 8.6, 9.0]])
    arr2 = numpy.array([[0.8, 2.1, 3.9], [4.2, 5.7, 6.3], [7.5, 8.2, 9.5]])
    lut = {
        ((0, 2), (1, 3)): 1,
        ((0, 2), (4, 6)): 2,
        ((0, 2), (7, 9)): 3,
        ((3, 5), (1, 3)): 4,
        ((3, 5), (4, 6)): 5,
        ((3, 5), (7, 9)): 6,
        ((6, 8), (1, 3)): 7,
        ((6, 8), (4, 6)): 8,
        ((6, 8), (7, 9)): 9
    }
    with pytest.raises(ValueError):
        label_from_lut(arr1, arr2, lut)
