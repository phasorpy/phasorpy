"""Test Leica image file reader functions."""

import math

import numpy
import pytest
from _conftest import SKIP_FETCH, SKIP_PRIVATE, private_file
from numpy.testing import assert_allclose, assert_array_equal

from phasorpy.datasets import fetch
from phasorpy.io import lifetime_from_lif, phasor_from_lif, signal_from_lif


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
@pytest.mark.parametrize('format', ('lif', 'xlef'))
def test_phasor_from_lif(format):
    """Test read phasor coordinates from Leica LIF file."""
    filename = fetch(f'FLIM_testdata.{format}')
    mean, real, imag, attrs = phasor_from_lif(filename)
    for data in (mean, real, imag):
        assert data.shape == (1024, 1024)
        assert data.dtype == numpy.float32
    assert (mean * 529).sum() == 9602774.0
    assert attrs['frequency'] == 19.505
    assert attrs['samples'] == 529
    assert 'harmonic' not in attrs
    assert attrs['flim_rawdata']['ClockPeriod'] == 9.696969697e-11
    assert (
        attrs['flim_phasor_channels'][0]['AutomaticReferencePhase']
        == 7.017962169
    )

    # select image
    mean1, real1, imag1, attrs = phasor_from_lif(
        filename, image='FLIM Compressed'
    )
    assert_array_equal(mean1, mean)

    # TODO: file does not contain FLIM raw metadata
    # filename = private_file('....lif')
    # mean, real, imag, attrs = phasor_from_lif(filename)
    # assert 'frequency' not in attrs

    # file does not contain FLIM data
    if not SKIP_PRIVATE:
        filename = private_file('ScanModesExamples.lif')
        with pytest.raises(ValueError):
            phasor_from_lif(filename)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
@pytest.mark.parametrize('format', ('lif', 'xlef'))
def test_lifetime_from_lif(format):
    """Test read lifetime image from Leica LIF file."""
    filename = fetch(f'FLIM_testdata.{format}')
    lifetime, intensity, stddev, attrs = lifetime_from_lif(filename)
    for data in (intensity, lifetime, stddev):
        assert data.shape == (1024, 1024)
        assert data.dtype == numpy.float32
    assert intensity.sum(dtype=numpy.float64) == 19278552.0
    assert pytest.approx(lifetime.mean(dtype=numpy.float64)) == 4.352868
    assert (
        pytest.approx(
            attrs['flim_phasor_channels'][0]['AutomaticReferencePhase']
        )
        == 7.017962
    )
    assert attrs['frequency'] == 19.505
    assert attrs['samples'] == 529
    assert 'harmonic' not in attrs

    # select series
    lifetime1, intensity1, stddev1, attrs = lifetime_from_lif(
        filename, image='FLIM Compressed'
    )
    assert_array_equal(intensity1, intensity)
    assert_array_equal(lifetime1, lifetime)

    # calibrate
    frequency = attrs['frequency']
    reference = attrs['flim_phasor_channels'][0]['AutomaticReferencePhase']
    lifetime -= math.radians(reference) / (2 * math.pi) / frequency * 1000
    assert pytest.approx(lifetime.mean(dtype=numpy.float64)) == 3.353414

    # file does not contain FLIM data
    if not SKIP_PRIVATE:
        filename = private_file('ScanModesExamples.lif')
        with pytest.raises(ValueError):
            lifetime_from_lif(filename)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_signal_from_lif():
    """Test read hyperspectral signal from Leica LIF file."""
    filename = fetch('Convalaria_LambdaScan.lif')
    signal = signal_from_lif(filename)
    assert signal.dims == ('C', 'Y', 'X')
    assert signal.shape == (29, 512, 512)
    assert signal.dtype == numpy.uint16
    assert numpy.round(signal.mean(), 2) == 62.59
    assert_allclose(signal.coords['C'].data[[0, -1]], [420.0, 700.0])

    # file does not contain hyperspectral signal
    filename = fetch('FLIM_testdata.lif')
    with pytest.raises(ValueError):
        signal_from_lif(filename)


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_signal_from_lif_private():
    """Test read hyperspectral signal from Leica LIF file."""
    filename = private_file('ScanModesExamples.lif')
    signal = signal_from_lif(filename)
    assert signal.dims == ('C', 'Y', 'X')
    assert signal.shape == (9, 128, 128)
    assert signal.dtype == numpy.uint8
    assert_allclose(signal.coords['C'].data[[0, 1]], [560.0, 580.0])

    # select image
    signal = signal_from_lif(filename, image='XYZLambdaT')
    assert signal.dims == ('T', 'C', 'Z', 'Y', 'X')
    assert signal.shape == (7, 9, 5, 128, 128)
    assert_allclose(signal.coords['C'].data[[0, 1]], [560.0, 580.0])
    assert_allclose(signal.coords['T'].data[[0, 1]], [0.0, 23.897167])
    assert_allclose(
        signal.coords['Z'].data[[0, 1]], [4.999881e-6, 2.499821e-6]
    )

    # select excitation
    signal = signal_from_lif(filename, dim='Λ')
    assert signal.dims == ('C', 'Y', 'X')
    assert signal.shape == (10, 128, 128)
    assert_allclose(signal.coords['C'].data[[0, 1]], [470.0, 492.0])


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
