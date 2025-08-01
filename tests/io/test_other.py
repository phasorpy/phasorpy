"""Test other file reader functions."""

import lfdfiles
import numpy
import ptufile
import pytest
import tifffile
from _conftest import SKIP_FETCH, SKIP_PRIVATE, private_file
from numpy.testing import assert_almost_equal, assert_array_equal

from phasorpy.datasets import fetch
from phasorpy.io import (
    phasor_from_ifli,
    signal_from_flif,
    signal_from_imspector_tiff,
    signal_from_lsm,
    signal_from_pqbin,
    signal_from_ptu,
    signal_from_sdt,
)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_signal_from_lsm_non_hyperspectral():
    """Test read non-hyperspectral LSM image fails."""
    filename = private_file('non_hyperspectral.lsm')
    with pytest.raises(ValueError):
        signal_from_lsm(filename)

    filename = fetch('simfcs.r64')
    with pytest.raises(tifffile.TiffFileError):
        signal_from_lsm(filename)


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_signal_from_lsm_tzcyx():
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
def test_signal_from_lsm_paramecium():
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
def test_signal_from_imspector_tiff():
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
def test_signal_from_sdt():
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
def test_signal_from_sdt_fcs():
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
def test_signal_from_sdt_bruker():
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
def test_phasor_from_ifli():
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


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_signal_from_flif():
    """Test read FlimFast FLIF file."""
    # TODO: test time series
    filename = fetch('flimfast.flif')
    signal = signal_from_flif(filename)
    signal = signal_from_flif(fetch('flimfast.flif'))
    assert signal.values.sum(dtype=numpy.uint64) == 706233156
    assert signal.dtype == 'uint16'
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
def test_signal_from_ptu():
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
def test_signal_from_ptu_irf():
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
def test_signal_from_pqbin():
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


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
