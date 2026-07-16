"""Test other file reader functions."""

import builtins
from pathlib import Path

import lfdfiles
import numpy
import ptufile
import pytest
import tifffile
from _conftest import SKIP_FETCH, SKIP_PRIVATE, private_file
from numpy.testing import assert_almost_equal, assert_array_equal

from phasorpy._typing import Any, NDArray
from phasorpy.datasets import fetch
from phasorpy.io import (
    lifetime_from_tdflim,
    phasor_from_ifli,
    signal_from_brighteyes_mcs,
    signal_from_czi,
    signal_from_flif,
    signal_from_imspector_tiff,
    signal_from_lsm,
    signal_from_pqbin,
    signal_from_ptu,
    signal_from_sdt,
    signal_from_tdflim,
)


def _write_brighteyes_mcs_h5(
    filename: str | Path,
) -> tuple[NDArray[Any], NDArray[Any]]:
    import h5py

    data = numpy.arange(2 * 3 * 4 * 5 * 8 * 2, dtype=numpy.uint16).reshape(
        2, 3, 4, 5, 8, 2
    )
    output = data.sum(axis=-1)
    reference = numpy.arange(8, dtype=numpy.float32)

    with h5py.File(filename, 'w') as h5:
        h5.attrs['schema_name'] = 'brighteyes_mcs_file'
        h5.attrs['data_format_version'] = '0.0.6'
        h5.attrs['default'] = '/raw/spad'

        raw = h5.create_group('raw')
        raw.attrs['metadata_path'] = '/raw/metadata'
        raw.attrs['axes_path'] = '/raw/axes'
        raw_data = raw.create_dataset('spad', data=data)
        raw_data.attrs['axis_order'] = (
            'repetition,z,y,x,time_bin,detector_channel'
        )
        raw_data.attrs['source_key'] = 'data'
        raw_data.attrs['time_axis_path'] = '/raw/axes/digital_time_ns'
        raw_data.attrs['metadata_path'] = '/raw/metadata'

        metadata = raw.create_group('metadata')
        metadata.attrs['laser_frequency_mhz'] = 80.0
        timing = metadata.create_group('acquisition').create_group('timing')
        timing.attrs['digital_time_bin_ns'] = 12.5 / 8
        axes = raw.create_group('axes')
        axes.create_dataset(
            'digital_time_ns',
            data=numpy.linspace(0.0, 12.5, 8, endpoint=False),
        )

        calibration = h5.create_group('calibration')
        result = calibration.create_group('results').create_group('spad')
        result.attrs['laser_frequency_mhz'] = 80.0
        fit = result.create_group('fit')
        fit.create_dataset('tau_reference_ns', data=numpy.array([2.7, 2.7]))

        output_group = h5.create_group('output')
        output_group.attrs['default'] = '/output/sum_001/products/spad'
        output_group.attrs['default_run'] = '/output/sum_001'
        output_group.attrs['default_ref_trace_id'] = (
            '/output/sum_ref_001/products/trace'
        )

        run = output_group.create_group('sum_001')
        run.create_group('metadata').attrs['laser_frequency_mhz'] = 80.0
        run.create_group('axes').create_dataset(
            'time_ns', data=numpy.linspace(0.0, 12.5, 8, endpoint=False)
        )
        product = run.create_group('products').create_dataset(
            'spad', data=output
        )
        product.attrs['axis_order'] = 'repetition,z,y,x,time_bin'
        product.attrs['time_axis_path'] = '/output/sum_001/axes/time_ns'
        product.attrs['metadata_path'] = '/output/sum_001/metadata'

        ref_run = output_group.create_group('sum_ref_001')
        ref_run.create_group('axes').create_dataset(
            'time_ns', data=numpy.linspace(0.0, 12.5, 8, endpoint=False)
        )
        ref = ref_run.create_group('products').create_dataset(
            'trace', data=reference
        )
        ref.attrs['output_type'] = 'trace'
        ref.attrs['trace_kind'] = 'sum_reference_trace'
        ref.attrs['time_axis_path'] = '/output/sum_ref_001/axes/time_ns'

    return output, reference


def test_signal_from_brighteyes_mcs(tmp_path: Path) -> None:
    """Test wrapping a BrightEyes-MCS signal as xarray."""
    filename = tmp_path / 'histogram.h5'
    data, _ = _write_brighteyes_mcs_h5(filename)

    signal = signal_from_brighteyes_mcs(filename, time=1, depth=2)

    assert signal.dims == ('Y', 'X', 'H')
    assert signal.shape == (4, 5, 8)
    assert_array_equal(signal.values, data[1, 2])
    assert_array_equal(
        signal.coords['H'].values,
        numpy.linspace(0.0, 12.5, 8, endpoint=False),
    )
    assert signal.attrs['frequency'] == 80.0
    assert signal.attrs['reference_lifetime_ns'] == 2.7
    assert signal.attrs['h5_dataset'] == '/output/sum_001/products/spad'


def test_signal_from_brighteyes_mcs_reference(tmp_path: Path) -> None:
    """Test reading a default BrightEyes-MCS reference trace."""
    filename = tmp_path / 'histogram.h5'
    _, reference = _write_brighteyes_mcs_h5(filename)

    signal = signal_from_brighteyes_mcs(filename, dataset='reference')

    assert signal.dims == ('Y', 'X', 'H')
    assert signal.shape == (1, 1, 8)
    assert_array_equal(signal.values, reference.reshape(1, 1, 8))
    assert signal.attrs['reference'] is True
    assert signal.attrs['irf'] is False


def test_signal_from_brighteyes_mcs_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test optional dependency error message."""
    real_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == 'brighteyes_mcs_reader':
            msg = 'blocked'
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', blocked_import)

    with pytest.raises(ImportError, match='brighteyes-mcs-reader'):
        signal_from_brighteyes_mcs('missing.h5')


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_signal_from_czi() -> None:
    """Test read CZI hyperspectral image."""
    import czifile

    filename = fetch('test_file.zstd.czi')
    signal = signal_from_czi(filename)
    assert signal.values.sum(dtype=numpy.uint64) == 2103903
    assert signal.dtype == numpy.uint8
    assert signal.shape == (28, 512, 512)
    assert signal.dims == ('C', 'Y', 'X')
    assert_almost_equal(
        signal.coords['C'][[0, -1]], [422.9785, 692.9785], decimal=4
    )
    assert_almost_equal(
        signal.coords['X'][[0, -1]], [0.0, 4.4896e-05], decimal=6
    )

    # test read RGB CZI image
    filename = fetch('rgb.czi')
    signal = signal_from_czi(filename)
    assert signal.sizes == {'S': 3, 'Y': 32, 'X': 31}

    # reject non-CZI file
    with pytest.raises(czifile.CziFileError):
        signal_from_czi(fetch('paramecium.lsm'))


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_signal_from_lsm_non_hyperspectral() -> None:
    """Test read non-hyperspectral LSM image fails."""
    filename = private_file('non_hyperspectral.lsm')
    with pytest.raises(ValueError):
        signal_from_lsm(filename)

    filename = fetch('simfcs.r64')
    with pytest.raises(tifffile.TiffFileError):
        signal_from_lsm(filename)


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_signal_from_lsm_tzcyx() -> None:
    """Test read TZC hyperspectral LSM image."""
    filename = private_file('tzcyx.lsm')
    signal = signal_from_lsm(filename)
    assert signal.values.sum(dtype=numpy.uint64) == 142328063165
    assert signal.dtype == numpy.uint16
    assert signal.shape == (10, 21, 32, 256, 256)
    assert signal.dims == ('T', 'Z', 'C', 'Y', 'X')
    assert_almost_equal(
        signal.coords['C'][[0, -1]], [414.936272, 690.47537], decimal=4
    )
    assert_almost_equal(
        signal.coords['T'][[0, -1]], [0.0, 1930.4651], decimal=4
    )
    assert_almost_equal(signal.coords['Z'][[0, -1]], [0.0, 7.4440772e-05])


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_signal_from_lsm_paramecium() -> None:
    """Test read paramecium.lsm."""
    filename = fetch('paramecium.lsm')
    signal = signal_from_lsm(filename)
    assert signal.values.sum(dtype=numpy.uint64) == 14050194
    assert signal.dtype == numpy.uint8
    assert signal.shape == (30, 512, 512)
    assert signal.dims == ('C', 'Y', 'X')
    assert_almost_equal(
        signal.coords['C'][[0, -1]], [423.0133, 713.0133], decimal=4
    )
    assert_almost_equal(
        signal.coords['X'][[0, -1]], [0.0, 0.000424265835], decimal=9
    )


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_signal_from_imspector_tiff() -> None:
    """Test read Imspector FLIM TIFF file."""
    signal = signal_from_imspector_tiff(fetch('Embryo.tif'))
    assert signal.values.sum(dtype=numpy.uint64) == 31348436
    assert signal.dtype == numpy.uint16
    assert signal.shape == (56, 512, 512)
    assert signal.dims == ('H', 'Y', 'X')
    assert_almost_equal(
        signal.coords['H'][[0, -1]], [0.0, 12.259995], decimal=12
    )
    assert pytest.approx(signal.attrs['frequency']) == 80.1095

    filename = fetch('paramecium.lsm')
    with pytest.raises(ValueError):
        signal_from_imspector_tiff(filename)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_signal_from_sdt() -> None:
    """Test read Becker & Hickl SDT file."""
    filename = fetch('tcspc.sdt')
    signal = signal_from_sdt(filename)
    assert signal.values.sum(dtype=numpy.uint64) == 224606420
    assert signal.dtype == numpy.uint16
    assert signal.shape == (128, 128, 256)
    assert signal.dims == ('Y', 'X', 'H')
    assert_almost_equal(
        signal.coords['H'][[0, -1]], [0.0, 12.451172], decimal=5
    )
    assert pytest.approx(signal.attrs['frequency']) == 79.999999

    filename = fetch('simfcs.r64')
    with pytest.raises(ValueError):
        signal_from_sdt(filename)


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_signal_from_sdt_fcs() -> None:
    """Test read Becker & Hickl SDT FCS file."""
    # file provided by lmalacrida via email on Nov 13, 2023
    filename = private_file('j3_405_z1.sdt')
    signal = signal_from_sdt(filename)
    assert signal.values.sum(dtype=numpy.uint64) == 16929780
    assert signal.dtype == numpy.uint16
    assert signal.shape == (512, 512, 1024)
    assert signal.dims == ('Y', 'X', 'H')
    assert_almost_equal(
        signal.coords['H'][[0, -1]], [0.0, 16.66157], decimal=5
    )
    assert pytest.approx(signal.attrs['frequency']) == 59.959740


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_signal_from_sdt_bruker() -> None:
    """Test read Becker & Hickl SDT file with routing channel."""
    # file provided by bruno-pannunzio via email on March 25, 2025
    filename = private_file('LifetimeData_Cycle00001_000001.sdt')
    signal = signal_from_sdt(filename)
    assert signal.dtype == numpy.uint16
    assert signal.shape == (2, 512, 512, 256)
    assert signal.dims == ('C', 'Y', 'X', 'H')
    assert_almost_equal(signal.coords['H'][[0, -1]], [0.0, 12.24], decimal=2)
    assert pytest.approx(signal.attrs['frequency']) == 81.3802
    assert signal[0].values.sum(dtype=numpy.uint64) == 15234486
    assert signal[1].values.sum(dtype=numpy.uint64) == 0


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_phasor_from_ifli() -> None:
    """Test read ISS VistaVision file."""
    # TODO: test spectral file
    filename = fetch('frequency_domain.ifli')
    mean, real, imag, attr = phasor_from_ifli(filename, harmonic='all')
    assert mean.sum(dtype=numpy.float64) == 15614850.0
    assert mean.dtype == numpy.float32
    assert mean.shape == (256, 256)
    assert real.shape == (4, 256, 256)
    assert imag.shape == (4, 256, 256)
    assert attr['dims'] == ('Y', 'X')
    assert attr['frequency'] == 80.332416
    assert attr['harmonic'] == [1, 2, 3, 5]
    assert attr['samples'] == 64

    mean, real1, imag1, attr = phasor_from_ifli(
        filename, harmonic='any', memmap=True
    )
    assert mean.sum(dtype=numpy.float64) == 15614850.0
    assert attr['harmonic'] == [1, 2, 3, 5]
    assert_array_equal(real1, real)
    assert_array_equal(imag1, imag)

    mean, real1, imag1, attr = phasor_from_ifli(filename)
    assert mean.shape == (256, 256)
    assert real1.shape == (256, 256)
    assert imag1.shape == (256, 256)
    assert attr['harmonic'] == [1]
    assert_array_equal(real1, real[0])

    mean, real1, imag1, attr = phasor_from_ifli(filename, harmonic=2)
    assert mean.shape == (256, 256)
    assert real1.shape == (256, 256)
    assert imag1.shape == (256, 256)
    assert attr['harmonic'] == [2]
    assert_array_equal(real1, real[1])

    mean, real1, imag1, attr = phasor_from_ifli(filename, harmonic=[3])
    assert real1.shape == (1, 256, 256)
    assert imag1.shape == (1, 256, 256)
    assert attr['harmonic'] == [3]
    assert_array_equal(real1, real[2:3])

    mean, real1, imag1, attr = phasor_from_ifli(filename, harmonic=[2, 5])
    assert real1.shape == (2, 256, 256)
    assert imag1.shape == (2, 256, 256)
    assert attr['harmonic'] == [2, 5]
    assert_array_equal(real1, real[[1, 3]])

    with pytest.raises(IndexError):
        phasor_from_ifli(filename, channel=1)

    with pytest.raises(IndexError):
        phasor_from_ifli(filename, harmonic=4)

    filename = fetch('simfcs.r64')
    with pytest.raises(lfdfiles.LfdFileError):
        phasor_from_ifli(filename)


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_signal_from_tdflim() -> None:
    """Test read ISS Vista TDFLIM file."""
    filename = private_file('version1.iss-tdflim')
    signal = signal_from_tdflim(filename, channel=None)
    assert signal.values.sum(dtype=numpy.uint64) == 1098769946
    assert signal.dtype == numpy.uint16
    assert signal.shape == (2, 128, 128, 1024)
    assert signal.dims == ('C', 'Y', 'X', 'H')
    assert_almost_equal(
        signal.coords['H'].data[[1, -1]], [0.012215228, 12.496178]
    )
    assert pytest.approx(signal.attrs['frequency']) == 79.946319

    signal = signal_from_tdflim(filename)
    assert signal.values.sum(dtype=numpy.uint64) == 694177148
    assert signal.shape == (128, 128, 1024)
    assert signal.dims == ('Y', 'X', 'H')

    with pytest.raises(lfdfiles.LfdFileError):
        signal_from_tdflim(fetch('flimfast.flif'))


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_lifetime_from_tdflim() -> None:
    """Test read ISS Vista TDFLIM file."""
    filename = private_file('version2.iss-tdflim')
    lifetime = lifetime_from_tdflim(filename, channel=None)
    assert lifetime.values.mean() == pytest.approx(3.0074844)
    assert lifetime.dtype == numpy.float32
    assert lifetime.shape == (128, 128)
    assert lifetime.dims == ('Y', 'X')
    assert lifetime.attrs['frequency'] == 20.0
    assert_almost_equal(
        lifetime.coords['X'].data[[1, -1]], [-3.8046875, 20.8046875]
    )

    lifetime = lifetime_from_tdflim(filename)
    assert lifetime.values.mean() == pytest.approx(3.0074844)
    assert lifetime.dtype == numpy.float32
    assert lifetime.shape == (128, 128)
    assert lifetime.dims == ('Y', 'X')

    with pytest.raises(ValueError):
        lifetime_from_tdflim(private_file('version1.iss-tdflim'))


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_signal_from_flif() -> None:
    """Test read FlimFast FLIF file."""
    # TODO: test time series
    filename = fetch('flimfast.flif')
    signal = signal_from_flif(filename)
    signal = signal_from_flif(fetch('flimfast.flif'))
    assert signal.values.sum(dtype=numpy.uint64) == 706233156
    assert signal.dtype == numpy.uint16
    assert signal.shape == (32, 220, 300)
    assert signal.dims == ('H', 'Y', 'X')
    assert_almost_equal(
        signal.coords['H'].data[[1, -1]], [0.1963495, 6.086836]
    )
    assert_almost_equal(signal.attrs['frequency'], 80.6520004272461)
    assert_almost_equal(signal.attrs['ref_phase'], 120.63999938964844)
    assert_almost_equal(signal.attrs['ref_mod'], 31.670000076293945)
    assert_almost_equal(signal.attrs['ref_tauphase'], 1.0160000324249268)
    assert_almost_equal(signal.attrs['ref_taumod'], 1.2580000162124634)

    filename = fetch('simfcs.r64')
    with pytest.raises(lfdfiles.LfdFileError):
        signal_from_flif(filename)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_signal_from_ptu() -> None:
    """Test read PicoQuant PTU file."""
    filename = fetch('hazelnut_FLIM_single_image.ptu')
    signal = signal_from_ptu(
        filename, frame=-1, channel=0, dtime=0, keepdims=False
    )
    assert signal.values.sum(dtype=numpy.uint64) == 6064854
    assert signal.dtype == numpy.uint16
    assert signal.shape == (256, 256, 132)
    assert signal.dims == ('Y', 'X', 'H')
    assert_almost_equal(
        signal.coords['H'].data[[1, -1]], [0.0969697, 12.7030303], decimal=4
    )
    assert signal.attrs['frequency'] == 78.02
    assert signal.attrs['ptu_tags']['HW_Type'] == 'PicoHarp'

    signal = signal_from_ptu(
        filename,
        frame=-1,
        channel=0,
        dtime=None,
        keepdims=True,
        trimdims='TC',
        # pixel_time=6.331709817995386e-06  # requires ptufile 2025.7.30
    )
    assert signal.values.sum(axis=(0, 3, 4))[128, 128] == 223
    assert signal.dtype == numpy.uint16
    assert signal.shape == (1, 256, 256, 1, 4096)
    assert signal.dims == ('T', 'Y', 'X', 'C', 'H')
    assert_almost_equal(
        signal.coords['H'].data[[1, -1]], [0.0969697, 397.09091], decimal=4
    )
    assert signal.attrs['frequency'] == 78.02

    filename = fetch('simfcs.r64')
    with pytest.raises(ptufile.PqFileError):
        signal_from_ptu(filename)


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_signal_from_ptu_irf() -> None:
    """Test read PicoQuant PTU file containing IRF."""
    # data file from PicoQuant's Samples.sptw
    filename = private_file('Cy5_diff_IRF+FLCS-pattern.ptu')
    signal = signal_from_ptu(filename, channel=None, keepdims=True)
    assert signal.values.sum(dtype=numpy.uint64) == 13268548
    assert signal.dtype == numpy.uint32
    assert signal.shape == (1, 1, 1, 2, 6250)
    assert signal.dims == ('T', 'Y', 'X', 'C', 'H')
    assert_almost_equal(
        signal.coords['H'].data[[1, -1]], [0.007999, 49.991999], decimal=4
    )
    assert pytest.approx(signal.attrs['frequency'], abs=1e-4) == 19.999732
    assert signal.attrs['ptu_tags']['HW_Type'] == 'PicoHarp 300'

    signal = signal_from_ptu(filename)
    assert signal.values[0, 0, 100] == 130
    assert signal.values.sum(dtype=numpy.uint64) == 6984849
    assert signal.shape == (1, 1, 6250)
    assert signal.dims == ('Y', 'X', 'H')

    with pytest.raises(ValueError):
        signal_from_ptu(filename, dtime=-1)

    signal = signal_from_ptu(filename, channel=0, dtime=None, keepdims=False)
    assert signal.values.sum(dtype=numpy.uint64) == 6984849
    assert signal.shape == (1, 1, 4096)
    assert signal.dims == ('Y', 'X', 'H')
    assert_almost_equal(
        signal.coords['H'].data[[1, -1]], [0.007999, 32.759999], decimal=4
    )

    signal = signal_from_ptu(filename, channel=0, dtime=None, keepdims=True)
    assert signal.values.sum(dtype=numpy.uint64) == 6984849
    assert signal.shape == (1, 1, 1, 1, 4096)
    assert signal.dims == ('T', 'Y', 'X', 'C', 'H')
    assert_almost_equal(
        signal.coords['H'].data[[1, -1]], [0.007999, 32.759999], decimal=4
    )


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_signal_from_pqbin() -> None:
    """Test read PicoQuant BIN file."""
    filename = private_file('picoquant.bin')
    signal = signal_from_pqbin(filename)
    assert signal.values.sum(dtype=numpy.uint64) == 43071870
    assert signal.dtype == numpy.uint32
    assert signal.shape == (256, 256, 2000)
    assert signal.dims == ('Y', 'X', 'H')
    assert signal.attrs['frequency'] == pytest.approx(19.999999, abs=1e-6)
    assert signal.attrs['pixel_resolution'] == 0.078125
    assert signal.attrs['tcspc_resolution'] == pytest.approx(0.025, abs=1e-6)
    assert_almost_equal(signal.coords['H'][[0, -1]], [0.0, 49.975], decimal=4)
    assert_almost_equal(
        signal.coords['X'][[0, -1]], [0.0, 1.9921875e-05], decimal=9
    )
    assert_almost_equal(
        signal.coords['Y'][[0, -1]], [0.0, 1.9921875e-05], decimal=9
    )

    filename = fetch('simfcs.r64')
    with pytest.raises(ValueError):
        signal_from_pqbin(filename)
