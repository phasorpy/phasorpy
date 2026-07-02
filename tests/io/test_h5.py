"""Test BrightEyes MCS-H5 file reader functions."""

import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from phasorpy.io import phasor_from_h5, signal_from_h5
from phasorpy.phasor import phasor_from_signal

h5py = pytest.importorskip('h5py')


def _write_h5(filename):
    data = numpy.arange(2 * 3 * 4 * 5 * 8 * 2, dtype=numpy.uint16).reshape(
        2, 3, 4, 5, 8, 2
    )
    reference = numpy.arange(8 * 2, dtype=numpy.float32).reshape(8, 2)
    irf = reference + 100.0

    with h5py.File(filename, 'w') as h5:
        h5.attrs['data_format_version'] = '0.0.6'
        h5.attrs['default'] = '/raw/spad'

        raw = h5.create_group('raw')
        raw.attrs['metadata_path'] = '/raw/metadata'
        raw.attrs['axes_path'] = '/raw/axes'
        dataset = raw.create_dataset('spad', data=data)
        dataset.attrs['axis_order'] = (
            'repetition,z,y,x,time_bin,detector_channel'
        )
        dataset.attrs['metadata_path'] = '/raw/metadata'
        dataset.attrs['time_axis_path'] = '/raw/axes/digital_time_ns'
        dataset.attrs['calibration_result_path'] = (
            '/calibration/results/spad'
        )

        metadata = raw.create_group('metadata')
        metadata.attrs['laser_frequency_mhz'] = 80.0
        metadata.attrs['digital_time_bin_ns'] = 12.5 / 8
        timing = metadata.create_group('acquisition').create_group('timing')
        timing.attrs['laser_frequency_mhz'] = 80.0
        timing.attrs['digital_time_bin_ns'] = 12.5 / 8

        axes = raw.create_group('axes')
        axes.create_dataset(
            'digital_time_ns',
            data=numpy.linspace(0.0, 12.5, 8, endpoint=False),
        )

        calibration = h5.create_group('calibration')
        calibration.create_group('metadata')
        calibration.create_group('axes').create_dataset(
            'time_ns',
            data=numpy.linspace(0.0, 12.5, 8, endpoint=False),
        )
        group = calibration.create_group('results').create_group('spad')
        group.attrs['laser_frequency_mhz'] = 80.0
        group.attrs['time_bin_ns'] = 12.5 / 8
        group.attrs['time_axis_path'] = '/calibration/axes/time_ns'
        fit = group.create_group('fit')
        fit.create_dataset('tau_reference_ns', data=numpy.array([2.7, 2.7]))
        aligned = group.create_group('aligned')
        aligned.create_dataset('reference_trace', data=reference)
        aligned.create_dataset('irf_trace', data=irf)

        output = h5.create_group('output')
        output.create_dataset(
            'sum_reference_trace',
            data=reference.sum(axis=1),
        ).attrs['time_axis_path'] = '/calibration/axes/time_ns'
        output.create_dataset(
            'sum_irf_trace',
            data=irf.sum(axis=1),
        ).attrs['time_axis_path'] = '/calibration/axes/time_ns'

    return data, reference.sum(axis=1), irf.sum(axis=1)


def _write_generic_h5(filename):
    data = numpy.arange(2 * 3 * 4 * 5 * 8 * 2, dtype=numpy.uint16).reshape(
        2, 3, 4, 5, 8, 2
    )
    reference = numpy.arange(8 * 2, dtype=numpy.float32).reshape(8, 2)
    irf = reference + 100.0

    with h5py.File(filename, 'w') as h5:
        h5.attrs['frequency'] = 80.0
        dataset = h5.create_dataset('data', data=data)
        dataset.attrs['time'] = numpy.linspace(0.0, 12.5, 8, endpoint=False)
        group = h5.create_group('calibration/data')
        ref = group.create_dataset('ref_common_delay_realigned', data=reference)
        ref.attrs['time'] = numpy.linspace(0.0, 12.5, 8, endpoint=False)
        irf_dataset = group.create_dataset(
            'irf_common_delay_realigned', data=irf
        )
        irf_dataset.attrs['time'] = numpy.linspace(
            0.0, 12.5, 8, endpoint=False
        )

    return data, reference, irf


def test_signal_from_h5(tmp_path):
    """Test reading a selected MCS-H5 histogram image."""
    filename = tmp_path / 'histogram.h5'
    data, _, _ = _write_h5(filename)

    signal = signal_from_h5(filename, repetition=1, z=2, channel=1)

    assert signal.dims == ('Y', 'X', 'H')
    assert signal.shape == (4, 5, 8)
    assert_array_equal(signal.values, data[1, 2, :, :, :, 1])
    assert_array_equal(
        signal.coords['H'].values,
        numpy.linspace(0.0, 12.5, 8, endpoint=False),
    )
    assert signal.attrs['frequency'] == 80.0
    assert signal.attrs['reference_lifetime_ns'] == 2.7
    assert signal.attrs['samples'] == 8
    assert signal.attrs['h5_dataset'] == '/raw/spad'
    assert signal.attrs['h5_shape'] == data.shape
    assert signal.attrs['h5_axes'] == (
        'repetition',
        'z',
        'y',
        'x',
        'time_bin',
        'detector_channel',
    )
    assert signal.attrs['h5_selection'] == {
        'repetition': 1,
        'z': 2,
        'channel': 1,
    }


def test_signal_from_h5_negative_indices(tmp_path):
    """Test reading MCS-H5 histogram image with negative indices."""
    filename = tmp_path / 'histogram.h5'
    data, _, _ = _write_h5(filename)

    signal = signal_from_h5(filename, repetition=-1, z=-1, channel=-1)

    assert_array_equal(signal.values, data[-1, -1, :, :, :, -1])
    assert signal.attrs['h5_selection'] == {
        'repetition': 1,
        'z': 2,
        'channel': 1,
    }


def test_signal_from_h5_sum_channel(tmp_path):
    """Test reading an MCS-H5 histogram image summed over channels."""
    filename = tmp_path / 'histogram.h5'
    data, reference, _ = _write_h5(filename)

    signal = signal_from_h5(filename, repetition=1, z=2, channel='sum')

    assert signal.dims == ('Y', 'X', 'H')
    assert signal.shape == (4, 5, 8)
    assert_array_equal(signal.values, data[1, 2, :, :, :, :].sum(axis=-1))
    assert signal.attrs['h5_selection'] == {
        'repetition': 1,
        'z': 2,
        'channel': 'sum',
    }

    signal = signal_from_h5(filename, reference=True, channel='sum')

    assert signal.dims == ('Y', 'X', 'H')
    assert signal.shape == (1, 1, 8)
    assert_array_equal(signal.values, reference.reshape(1, 1, 8))
    assert signal.attrs['h5_selection'] == {
        'reference': True,
        'irf': False,
        'channel': None,
    }


def test_signal_from_h5_reference(tmp_path):
    """Test reading an MCS-H5 reference histogram as a 1x1 image."""
    filename = tmp_path / 'histogram.h5'
    _, reference, _ = _write_h5(filename)

    signal = signal_from_h5(filename, reference=True)

    assert signal.dims == ('Y', 'X', 'H')
    assert signal.shape == (1, 1, 8)
    assert_array_equal(signal.values, reference.reshape(1, 1, 8))
    assert_array_equal(
        signal.coords['H'].values,
        numpy.linspace(0.0, 12.5, 8, endpoint=False),
    )
    assert signal.attrs['reference'] is True
    assert signal.attrs['h5_dataset'] == (
        '/output/sum_reference_trace'
    )
    assert signal.attrs['h5_shape'] == reference.shape
    assert signal.attrs['h5_axes'] == ('H',)
    assert signal.attrs['h5_selection'] == {
        'reference': True,
        'irf': False,
        'channel': None,
    }


def test_signal_from_h5_irf(tmp_path):
    """Test reading an MCS-H5 IRF histogram as a 1x1 image."""
    filename = tmp_path / 'histogram.h5'
    _, _, irf = _write_h5(filename)

    signal = signal_from_h5(filename, irf=True)

    assert signal.dims == ('Y', 'X', 'H')
    assert signal.shape == (1, 1, 8)
    assert_array_equal(signal.values, irf.reshape(1, 1, 8))
    assert signal.attrs['reference'] is True
    assert signal.attrs['irf'] is True
    assert signal.attrs['h5_dataset'] == (
        '/output/sum_irf_trace'
    )
    assert signal.attrs['h5_selection'] == {
        'reference': True,
        'irf': True,
        'channel': None,
    }


def test_signal_from_h5_generic_fallback(tmp_path):
    """Test reading a non-BrightEyes generic HDF5 histogram file."""
    filename = tmp_path / 'generic.h5'
    data, reference, _ = _write_generic_h5(filename)

    signal = signal_from_h5(filename, repetition=1, z=2, channel=1)

    assert signal.dims == ('Y', 'X', 'H')
    assert signal.shape == (4, 5, 8)
    assert_array_equal(signal.values, data[1, 2, :, :, :, 1])
    assert_array_equal(
        signal.coords['H'].values,
        numpy.linspace(0.0, 12.5, 8, endpoint=False),
    )
    assert signal.attrs['frequency'] == 80.0
    assert signal.attrs['h5_dataset'] == '/data'
    assert signal.attrs['h5_axes'] == ('R', 'Z', 'Y', 'X', 'H', 'C')
    assert signal.attrs['h5_selection'] == {
        'repetition': 1,
        'z': 2,
        'channel': 1,
    }

    signal = signal_from_h5(filename, reference=True, channel=1)

    assert signal.shape == (1, 1, 8)
    assert_array_equal(signal.values, reference[:, 1].reshape(1, 1, 8))
    assert signal.attrs['h5_dataset'] == (
        '/calibration/data/ref_common_delay_realigned'
    )
    assert signal.attrs['h5_selection'] == {
        'reference': True,
        'irf': False,
        'channel': 1,
    }


def test_phasor_from_h5(tmp_path):
    """Test calculating phasor coordinates from MCS-H5 histogram image."""
    filename = tmp_path / 'histogram.h5'
    data, _, _ = _write_h5(filename)
    selected = data[1, 2, :, :, :, 1]

    mean, real, imag, attrs = phasor_from_h5(
        filename,
        repetition=1,
        z=2,
        channel=1,
        harmonic=[1, 2],
    )
    mean_expected, real_expected, imag_expected = phasor_from_signal(
        selected,
        axis=-1,
        harmonic=[1, 2],
    )

    assert_allclose(mean, mean_expected)
    assert_allclose(real, real_expected)
    assert_allclose(imag, imag_expected)
    assert attrs['dims'] == ('Y', 'X')
    assert attrs['signal_dims'] == ('Y', 'X', 'H')
    assert attrs['harmonic'] == [1, 2]
    assert attrs['h5_selection'] == {
        'repetition': 1,
        'z': 2,
        'channel': 1,
    }


def test_signal_from_h5_errors(tmp_path):
    """Test invalid MCS-H5 histogram image errors."""
    filename = tmp_path / 'invalid.h5'

    with h5py.File(filename, 'w') as h5:
        h5.create_group('raw').create_dataset(
            'spad', data=numpy.zeros((2, 3, 4))
        )

    with pytest.raises(ValueError, match='expected'):
        signal_from_h5(filename)

    with h5py.File(filename, 'w') as h5:
        h5.create_group('raw').create_dataset(
            'spad', data=numpy.zeros((1, 1, 2, 2, 8, 1))
        )

    with pytest.raises(IndexError, match='repetition'):
        signal_from_h5(filename, repetition=1)

    with pytest.raises(IndexError, match='z'):
        signal_from_h5(filename, z=1)

    with pytest.raises(IndexError, match='channel'):
        signal_from_h5(filename, channel=1)
