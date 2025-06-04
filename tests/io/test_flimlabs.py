"""Test FLIM LABS file reader functions."""

import numpy
import pytest
from _conftest import SKIP_FETCH, SKIP_PRIVATE, private_file
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
)

from phasorpy.datasets import fetch
from phasorpy.io import phasor_from_flimlabs_json, signal_from_flimlabs_json
from phasorpy.phasor import phasor_from_signal, phasor_transform


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_flimlabs_reproduce():
    """Test FLIM LABS multiharmonic results can be reproduced with PhasorPy."""
    import json

    channel = 0
    filename = fetch('Convallaria_m2_1740751781_phasor_ch1.json')
    signal = signal_from_flimlabs_json(filename, channel=channel)
    mean, real, imag, attrs = phasor_from_flimlabs_json(
        filename, channel=channel, harmonic='all'
    )
    harmonic = attrs['harmonic']

    filename = fetch(
        'Fluorescein_Calibration_m2_1740751189_imaging_calibration.json'
    )
    with open(filename, 'rb') as fh:
        attrs = json.load(fh)

    mean1, real1, imag1 = phasor_from_signal(signal, harmonic=harmonic)
    assert mean.shape == mean1.shape
    assert real.shape == real1.shape
    assert imag.shape == imag1.shape

    calibration = numpy.asarray(attrs['calibrations'][channel])
    phase = -calibration[:, 0, None, None]
    modulation = 1.0 / calibration[:, 1, None, None]

    real1, imag1 = phasor_transform(real1, imag1, phase, modulation)

    assert_allclose(mean, mean1, atol=1e-3)
    assert_allclose(real, real1, atol=1e-3, equal_nan=True)
    assert_allclose(imag, imag1, atol=1e-3, equal_nan=True)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_phasor_from_flimlabs_json():
    """Test phasor_from_flimlabs_json function."""
    filename = fetch('Convallaria_m2_1740751781_phasor_ch1.json')
    mean, real, imag, attrs = phasor_from_flimlabs_json(
        filename, harmonic='all', channel=0
    )
    assert mean.dtype == numpy.float32
    assert mean.shape == (256, 256)
    assert real.shape == (3, 256, 256)
    assert imag.shape == (3, 256, 256)
    assert pytest.approx(mean.mean(), abs=1e-2) == 14201097 / 256 / 256 / 256
    assert_allclose(
        numpy.nanmean(real, axis=(1, 2)),
        [0.649459, 0.465521, 0.368166],
        atol=1e-3,
    )
    assert_allclose(
        numpy.nanmean(imag, axis=(1, 2)),
        [0.318597, 0.317059, 0.287719],
        atol=1e-3,
    )
    assert attrs['dims'] == ('Y', 'X')
    assert attrs['harmonic'] == [1, 2, 3]
    assert attrs['samples'] == 256
    assert pytest.approx(attrs['frequency']) == 40.00017
    lpns = attrs['flimlabs_header']['laser_period_ns']
    assert pytest.approx(lpns) == 24.9998932

    # first harmonic by default
    mean, real, imag, attrs = phasor_from_flimlabs_json(filename)
    assert real.shape == (256, 256)
    assert imag.shape == (256, 256)
    assert pytest.approx(numpy.nanmean(real), abs=1e-3) == 0.64946
    assert pytest.approx(numpy.nanmean(imag), abs=1e-3) == 0.318597
    assert attrs['harmonic'] == 1

    # second harmonic, keep axis
    mean, real, imag, attrs = phasor_from_flimlabs_json(filename, harmonic=[2])
    assert real.shape == (1, 256, 256)
    assert imag.shape == (1, 256, 256)
    assert pytest.approx(numpy.nanmean(real), abs=1e-3) == 0.465521
    assert pytest.approx(numpy.nanmean(imag), abs=1e-3) == 0.317059
    assert attrs['harmonic'] == [2]

    # first and third harmonic
    mean, real, imag, attrs = phasor_from_flimlabs_json(
        filename, harmonic=[1, 3]
    )
    assert real.shape == (2, 256, 256)
    assert imag.shape == (2, 256, 256)
    assert_allclose(
        numpy.nanmean(real, axis=(1, 2)), [0.649459, 0.368166], atol=1e-3
    )
    assert_allclose(
        numpy.nanmean(imag, axis=(1, 2)), [0.318597, 0.287719], atol=1e-3
    )
    assert attrs['harmonic'] == [1, 3]

    # harmonic out of range
    with pytest.raises(IndexError):
        phasor_from_flimlabs_json(filename, harmonic=[1, 5])

    # channel out of range
    with pytest.raises(IndexError):
        phasor_from_flimlabs_json(filename, channel=1)

    # not a file containing phasor coordinates
    filename = fetch('Fluorescein_Calibration_m2_1740751189_imaging.json')
    with pytest.raises(ValueError):
        phasor_from_flimlabs_json(filename)

    # not a JSON file
    filename = fetch('simfcs.r64')
    with pytest.raises(ValueError):
        phasor_from_flimlabs_json(filename)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_signal_from_flimlabs_json():
    """Test signal_from_flimlabs_json function."""
    filename = fetch('Convallaria_m2_1740751781_phasor_ch1.json')
    signal = signal_from_flimlabs_json(filename)
    assert signal.values.sum(dtype=numpy.uint64) == 14201097
    assert signal.dtype == numpy.uint16
    assert signal.shape == (256, 256, 256)
    assert signal.dims == ('Y', 'X', 'H')
    assert 'C' not in signal.coords
    assert_almost_equal(
        signal.coords['H'][[0, -1]], [0.0, 24.902237], decimal=6
    )
    assert pytest.approx(signal.attrs['frequency']) == 40.000171
    lpns = signal.attrs['flimlabs_header']['laser_period_ns']
    assert pytest.approx(lpns) == 24.99989318828099

    signal = signal_from_flimlabs_json(filename, channel=0)
    assert signal.shape == (256, 256, 256)

    # channel does not exist
    with pytest.raises(IndexError):
        signal_from_flimlabs_json(filename, channel=1)

    # not an unsigned int dtype
    with pytest.raises(ValueError):
        signal_from_flimlabs_json(filename, dtype=numpy.int8)

    # not a file containing TCSPC histogram
    filename = fetch(
        'Fluorescein_Calibration_m2_1740751189_imaging_calibration.json'
    )
    with pytest.raises(ValueError):
        signal_from_flimlabs_json(filename)

    # not a JSON file
    filename = fetch('simfcs.r64')
    with pytest.raises(ValueError):
        signal_from_flimlabs_json(filename)


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_signal_from_flimlabs_json_old():
    """Test signal_from_flimlabs_json function with old format file."""
    filename = private_file('calibrator_2_5_1737112045_imaging.json')
    signal = signal_from_flimlabs_json(filename)
    assert signal.values.sum(dtype=numpy.uint64) == 6152493
    assert signal.dtype == numpy.uint16
    assert signal.shape == (256, 256, 256)
    assert signal.dims == ('Y', 'X', 'H')
    assert pytest.approx(signal.attrs['frequency']) == 79.510243


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_signal_from_flimlabs_json_channel():
    """Test read FLIM LABS JSON multi-channel image file."""
    filename = private_file('test03_1733492714_imaging.json')
    signal = signal_from_flimlabs_json(filename, channel=None)
    assert signal.values.sum(dtype=numpy.uint64) == 4680256
    assert signal.dtype == numpy.uint16
    assert signal.shape == (3, 256, 256, 256)
    assert signal.dims == ('C', 'Y', 'X', 'H')
    assert_almost_equal(
        signal.coords['H'][[0, -1]], [0.0, 12.451171875], decimal=12
    )
    assert_array_equal(signal.coords['C'], [0, 1, 2])
    assert signal.attrs['frequency'] == 80.0

    signal = signal_from_flimlabs_json(filename, channel=1, dtype=numpy.uint8)
    assert signal.values.sum(dtype=numpy.uint64) == 1388562
    assert signal.dtype == numpy.uint8
    assert signal.shape == (256, 256, 256)
    assert signal.dims == ('Y', 'X', 'H')

    with pytest.raises(ValueError):
        signal_from_flimlabs_json(filename, channel=1, dtype=numpy.int8)


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
@pytest.mark.parametrize('channel', (0, 1))
def test_phasor_from_flimlabs_json_channel(channel):
    """Test read FLIM LABS JSON phasor file from multi-channel dataset."""
    filename = private_file(
        f'FLIMLABS/convallaria-03_1742566249_phasor_ch{channel + 1}.json'
    )
    for c in (None, channel):
        mean, real, imag, attrs = phasor_from_flimlabs_json(
            filename, channel=c
        )
        assert attrs['dims'] == ('Y', 'X')
        assert mean.shape == (247, 245)
        assert real.shape == (247, 245)
        assert imag.shape == (247, 245)
        if channel == 0:
            assert pytest.approx(mean.mean()) == 0.215034812
            assert pytest.approx(numpy.nanmean(real)) == 0.5872460
        else:
            assert pytest.approx(mean.mean()) == 0.08891675
            assert pytest.approx(numpy.nanmean(real)) == 0.61652845

    with pytest.raises(IndexError):
        phasor_from_flimlabs_json(filename, channel=-1)
    with pytest.raises(IndexError):
        phasor_from_flimlabs_json(filename, channel=channel + 1)
    with pytest.raises(IndexError):
        phasor_from_flimlabs_json(filename, channel=channel - 1)


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
