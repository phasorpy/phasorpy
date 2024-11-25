"""Tests for the phasorpy.utils module."""

import os

import numpy
import pytest
from numpy.testing import assert_allclose, assert_equal

from phasorpy.datasets import fetch
from phasorpy.io import read_lsm
from phasorpy.phasor import phasor_from_signal
from phasorpy.utils import (
    anscombe_transformation,
    anscombe_transformation_inverse,
    number_threads,
    spectral_vector_denoise,
)

numpy.random.seed(42)


def test_number_threads():
    """Test `number_threads` function."""
    assert number_threads() == 1
    assert number_threads(None, 0) == 1
    assert number_threads(1) == 1
    assert number_threads(-1) == 1
    assert number_threads(-1, 2) == 1
    assert number_threads(6) == 6
    assert number_threads(100) == 100
    assert number_threads(6, 5) == 5
    num_threads = number_threads(0)
    assert num_threads >= 1
    if num_threads > 4:
        assert number_threads(0, 4) == 4
        os.environ['PHASORPY_NUM_THREADS'] = '4'
        assert number_threads(0) == 4
        assert number_threads(6) == 6
        del os.environ['PHASORPY_NUM_THREADS']


@pytest.mark.parametrize('dtype', ['float32', 'uint16'])
def test_anscombe_transformation(dtype):
    """Test anscombe_transformation and inverse functions."""
    x = numpy.random.poisson(10, 100000).astype(dtype)
    if dtype == 'float32':
        x[0] = numpy.nan
    z = anscombe_transformation(x)
    assert_allclose(numpy.std(z[1:]), 1.0, atol=0.01)

    x2 = anscombe_transformation_inverse(z)
    assert_allclose(x2[1:], x[1:], atol=0.01)

    x3 = anscombe_transformation_inverse(z, approx=True)
    assert_allclose(x3[1:], x[1:], atol=1.0)

    if dtype == 'float32':
        assert numpy.isnan(z[0])
        assert numpy.isnan(x2[0])
        assert numpy.isnan(x3[0])


@pytest.mark.parametrize('dtype', [None, 'float32'])
@pytest.mark.parametrize('spectral_vector', [None, True])
def test_spectral_vector_denoise(dtype, spectral_vector):
    """Test spectral_vector_denoise function."""
    # TODO: test synthetic data

    signal = read_lsm(fetch('paramecium.lsm')).data[:, ::16, ::16]
    if dtype is not None:
        signal = signal.astype(dtype)
    mean, real, imag = phasor_from_signal(signal, axis=0)

    if spectral_vector is not None:
        spectral_vector = numpy.moveaxis(numpy.stack((real, imag)), 0, -1)

    denoised = spectral_vector_denoise(
        signal,
        spectral_vector,
        axis=0,
        sigma=0.1,
        vmin=None,
        harmonic=None,
        dtype=dtype,
        num_threads=1,
    )

    mean1, real1, imag1 = phasor_from_signal(denoised, axis=0)
    assert_allclose(mean, mean1, atol=1e-3)
    assert_allclose(signal, denoised, atol=22)
    assert denoised.dtype == dtype


def test_spectral_vector_nan():
    """Test spectral_vector_denoise function NaN handling."""
    signal = read_lsm(fetch('paramecium.lsm')).data[:, ::16, ::16]
    signal = signal.astype(numpy.float64)
    signal[0, 0, 0] = numpy.nan

    mean, real, imag = phasor_from_signal(signal, axis=0)
    spectral_vector = numpy.moveaxis(numpy.stack((real, imag)), 0, -1)
    spectral_vector[0, 1] = numpy.nan
    assert numpy.all(numpy.isnan(spectral_vector[0, 0]))

    denoised = spectral_vector_denoise(
        signal, spectral_vector, vmin=20, axis=0
    )

    assert_allclose(signal, denoised, atol=22)
    # spectral_vector is NaN
    assert_allclose(denoised[:, 0, 1], signal[:, 0, 1], atol=1e-3)
    # signal < vmin
    assert_allclose(denoised[:, -1, 0], signal[:, -1, 0], atol=1e-3)
    # no signal
    assert_allclose(denoised[:, -1, -1], signal[:, -1, -1], atol=1e-3)
    # signal is NaN
    assert numpy.isnan(denoised[0, 0, 0])

    mean1, real1, imag1 = phasor_from_signal(denoised, axis=0)
    assert_allclose(mean, mean1, atol=1e-3)


def test_spectral_vector_denoise_exceptions():
    """Test spectral_vector_denoise function exceptions."""
    signal = numpy.random.randint(0, 255, (16, 8, 16)).astype(numpy.float32)
    spectral_vector = numpy.random.random((16, 16, 2))

    spectral_vector_denoise(signal, spectral_vector, axis=1)

    with pytest.raises(ValueError):
        spectral_vector_denoise(
            signal, spectral_vector, axis=1, dtype=numpy.uint8
        )

    with pytest.raises(ValueError):
        spectral_vector_denoise(signal, spectral_vector[:15], axis=1)


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
