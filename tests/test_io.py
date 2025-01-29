"""Tests for the phasorpy.io module."""

import os
import tempfile
from glob import glob

import lfdfiles
import numpy
import ptufile
import pytest
import tifffile
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
)

from phasorpy.datasets import fetch
from phasorpy.io import (
    phasor_from_flimlabs_json,
    phasor_from_ifli,
    phasor_from_lif,
    phasor_from_ometiff,
    phasor_from_simfcs_referenced,
    phasor_to_ometiff,
    phasor_to_simfcs_referenced,
    signal_from_b64,
    signal_from_bh,
    signal_from_bhz,
    signal_from_fbd,
    signal_from_flif,
    signal_from_flimlabs_json,
    signal_from_imspector_tiff,
    signal_from_lif,
    signal_from_lsm,
    signal_from_ptu,
    signal_from_sdt,
    signal_from_z64,
)
from phasorpy.phasor import phasor_from_signal, phasor_transform

HERE = os.path.dirname(__file__)
TEMP_DIR = os.path.normpath(
    os.environ.get('PHASORPY_TEMP', tempfile.gettempdir())
)
DATA_DIR = os.path.normpath(
    os.environ.get('PHASORPY_DATA', os.path.join(HERE, '..', 'data'))
)
PRIVATE_DIR = os.path.join(DATA_DIR, 'private')

SKIP_PRIVATE = not os.path.exists(PRIVATE_DIR)
SKIP_FETCH = os.environ.get('SKIP_FETCH', False)


class TempFileName:
    """Temporary file name context manager."""

    name: str
    remove: bool
    pattern: bool

    def __init__(self, name=None, remove=False, pattern=False):
        self.remove = remove or TEMP_DIR == tempfile.gettempdir()
        if not name:
            fh = tempfile.NamedTemporaryFile(prefix='test_')
            self.name = fh.named
            fh.close()
        else:
            self.name = os.path.join(TEMP_DIR, f'test_{name}')
        self.pattern = pattern

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        if self.remove:
            if self.pattern:
                name, ext = os.path.splitext(self.name)
                for fname in glob(name + '*' + ext):
                    try:
                        os.remove(fname)
                    except Exception:
                        pass
            try:
                os.remove(self.name)
            except Exception:
                pass


def private_file(filename: str, /) -> str:
    """Return path to private test file."""
    return os.path.normpath(os.path.join(PRIVATE_DIR, filename))


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


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_signal_from_imspector_tiff_t():
    """Test read Imspector FLIM TIFF file with TCSPC in T-axis."""
    signal = signal_from_imspector_tiff(private_file('ZF-1100_noEF.tif'))
    assert signal.values.sum(dtype=numpy.uint64) == 18636271
    assert signal.dtype == numpy.uint16
    assert signal.shape == (56, 512, 512)
    assert signal.dims == ('H', 'Y', 'X')
    assert_almost_equal(
        signal.coords['H'][[0, -1]], [0.0, 12.259995], decimal=12
    )
    assert pytest.approx(signal.attrs['frequency']) == 80.109564


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_flimlabs_reproduce():
    """Test FLIM LABS multiharmonic results can be reproduced with PhasorPy."""
    import json

    channel = 0
    filename = fetch('convallaria_2_1737113097_phasor_ch1.json')
    signal = signal_from_flimlabs_json(filename, channel=channel)
    mean, real, imag, attrs = phasor_from_flimlabs_json(
        filename, channel=channel, harmonic='all'
    )
    harmonic = attrs['harmonic']

    filename = fetch('calibrator_2_5_1737112045_imaging_calibration.json')
    with open(filename, 'rb') as fh:
        attrs = json.load(fh)

    mean1, real1, imag1 = phasor_from_signal(signal, harmonic=harmonic)
    assert mean.shape == mean1.shape
    assert real.shape == real1.shape
    assert imag.shape == imag1.shape

    calibration = numpy.asarray(attrs['calibrations'][channel])
    phase = -calibration[:, 0].reshape(-1, 1, 1)
    modulation = 1.0 / calibration[:, 1].reshape(-1, 1, 1)

    real1, imag1 = phasor_transform(real1, imag1, phase, modulation)

    assert_allclose(mean, mean1, atol=1e-2)
    assert_allclose(real, real1, atol=1e-2)
    assert_allclose(imag, imag1, atol=1e-2)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_phasor_from_flimlabs_json():
    """Test phasor_from_flimlabs_json function."""
    filename = fetch('convallaria_2_1737113097_phasor_ch1.json')
    mean, real, imag, attrs = phasor_from_flimlabs_json(
        filename, harmonic='all', channel=0
    )
    assert mean.dtype == numpy.float32
    assert mean.shape == (256, 256)
    assert real.shape == (4, 256, 256)
    assert imag.shape == (4, 256, 256)
    assert pytest.approx(mean.mean(), abs=1e-2) == 4896155 / 256 / 256 / 256
    assert_allclose(
        real.mean(axis=(1, 2)),
        [0.30063, 0.198757, 0.15185, 0.10898],
        atol=1e-3,
    )
    assert_allclose(
        imag.mean(axis=(1, 2)),
        [0.202482, 0.059337, -0.018447, -0.063713],
        atol=1e-3,
    )
    assert attrs['dims'] == ('Y', 'X')
    assert attrs['harmonic'] == [1, 2, 3, 4]
    assert attrs['samples'] == 256
    assert pytest.approx(attrs['frequency']) == 79.51024
    lpns = attrs['flimlabs_header']['laser_period_ns']
    assert pytest.approx(lpns) == 12.576995

    # first harmonic by default
    mean, real, imag, attrs = phasor_from_flimlabs_json(filename)
    assert real.shape == (256, 256)
    assert imag.shape == (256, 256)
    assert pytest.approx(real.mean(), abs=1e-3) == 0.30063
    assert pytest.approx(imag.mean(), abs=1e-3) == 0.202482
    assert attrs['harmonic'] == 1

    # second harmonic, keep axis
    mean, real, imag, attrs = phasor_from_flimlabs_json(filename, harmonic=[2])
    assert real.shape == (1, 256, 256)
    assert imag.shape == (1, 256, 256)
    assert pytest.approx(real.mean(), abs=1e-3) == 0.198757
    assert pytest.approx(imag.mean(), abs=1e-3) == 0.059337
    assert attrs['harmonic'] == [2]

    # first and third harmonic
    mean, real, imag, attrs = phasor_from_flimlabs_json(
        filename, harmonic=[1, 3]
    )
    assert real.shape == (2, 256, 256)
    assert imag.shape == (2, 256, 256)
    assert_allclose(real.mean(axis=(1, 2)), [0.30063, 0.15185], atol=1e-3)
    assert_allclose(imag.mean(axis=(1, 2)), [0.202482, -0.018447], atol=1e-3)
    assert attrs['harmonic'] == [1, 3]

    # harmonic out of range
    with pytest.raises(IndexError):
        phasor_from_flimlabs_json(filename, harmonic=[1, 5])

    # channel out of range
    with pytest.raises(IndexError):
        phasor_from_flimlabs_json(filename, channel=1)

    # not a file containing phasor coordinates
    filename = fetch('calibrator_2_5_1737112045_imaging_calibration.json')
    with pytest.raises(ValueError):
        phasor_from_flimlabs_json(filename)

    # not a JSON file
    filename = fetch('simfcs.r64')
    with pytest.raises(ValueError):
        phasor_from_flimlabs_json(filename)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_signal_from_flimlabs_json():
    """Test signal_from_flimlabs_json function."""
    filename = fetch('convallaria_2_1737113097_phasor_ch1.json')
    signal = signal_from_flimlabs_json(filename)
    assert signal.values.sum(dtype=numpy.uint64) == 4896155
    assert signal.dtype == numpy.uint16
    assert signal.shape == (256, 256, 256)
    assert signal.dims == ('Y', 'X', 'H')
    assert 'C' not in signal.coords
    assert_almost_equal(
        signal.coords['H'][[0, -1]], [0.0, 12.527867], decimal=6
    )
    assert pytest.approx(signal.attrs['frequency']) == 79.510243
    lpns = signal.attrs['flimlabs_header']['laser_period_ns']
    assert pytest.approx(lpns) == 12.57699584916508

    signal = signal_from_flimlabs_json(filename, channel=0)
    assert signal.shape == (256, 256, 256)

    # channel does not exist
    with pytest.raises(IndexError):
        signal_from_flimlabs_json(filename, channel=1)

    # not an unsigned int dtype
    with pytest.raises(ValueError):
        signal_from_flimlabs_json(filename, dtype=numpy.int8)

    # old format file
    filename = fetch('calibrator_2_5_1737112045_imaging.json')
    signal = signal_from_flimlabs_json(filename)
    assert signal.values.sum(dtype=numpy.uint64) == 6152493
    assert signal.dtype == numpy.uint16
    assert signal.shape == (256, 256, 256)
    assert signal.dims == ('Y', 'X', 'H')
    assert pytest.approx(signal.attrs['frequency']) == 79.510243

    # not a file containing a TCSPC signal
    filename = fetch('calibrator_2_5_1737112045_imaging_calibration.json')
    with pytest.raises(ValueError):
        signal_from_flimlabs_json(filename)

    # not a JSON file
    filename = fetch('simfcs.r64')
    with pytest.raises(ValueError):
        signal_from_flimlabs_json(filename)


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_signal_from_flimlabs_json_channel():
    """Test read FLIM LABS JSON image file."""
    filename = private_file('test03_1733492714_imaging.json')
    signal = signal_from_flimlabs_json(filename)
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
    )
    assert signal.values.sum(dtype=numpy.uint64) == 6065123
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
    signal = signal_from_ptu(filename)
    assert signal.values.sum(dtype=numpy.uint64) == 13268548
    assert signal.dtype == numpy.uint32
    assert signal.shape == (1, 1, 1, 2, 6250)
    assert signal.dims == ('T', 'Y', 'X', 'C', 'H')
    assert_almost_equal(
        signal.coords['H'].data[[1, -1]], [0.007999, 49.991999], decimal=4
    )
    assert pytest.approx(signal.attrs['frequency'], abs=1e-4) == 19.999732
    assert signal.attrs['ptu_tags']['HW_Type'] == 'PicoHarp 300'

    signal = signal_from_ptu(filename, channel=0, keepdims=True)
    assert signal.values.sum(dtype=numpy.uint64) == 6984849
    assert signal.shape == (1, 1, 1, 1, 6250)
    assert signal.dims == ('T', 'Y', 'X', 'C', 'H')

    with pytest.raises(ValueError):
        signal_from_ptu(filename, dtime=-1)

    signal = signal_from_ptu(filename, channel=0, dtime=None, keepdims=False)
    assert signal.values.sum(dtype=numpy.uint64) == 6984849
    assert signal.shape == (1, 1, 4096)
    assert signal.dims == ('Y', 'X', 'H')
    assert_almost_equal(
        signal.coords['H'].data[[1, -1]], [0.007999, 32.759999], decimal=4
    )


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_signal_from_fbd():
    """Test read FLIMbox FBD file."""
    # TODO: test files with different firmwares
    # TODO: gather public FBD files and upload to Zenodo
    filename = private_file('convallaria_000$EI0S.fbd')
    signal = signal_from_fbd(filename)
    assert signal.values.sum(dtype=numpy.uint64) == 9310275
    assert signal.dtype == numpy.uint16
    assert signal.shape == (9, 2, 256, 256, 64)
    assert signal.dims == ('T', 'C', 'Y', 'X', 'H')
    assert_almost_equal(
        signal.coords['H'].data[[1, -1]], [0.0981748, 6.1850105]
    )
    assert_almost_equal(signal.attrs['frequency'], 40.0)

    signal = signal_from_fbd(filename, frame=-1, channel=0)
    assert signal.values.sum(dtype=numpy.uint64) == 9310275
    assert signal.shape == (1, 1, 256, 256, 64)
    assert signal.dims == ('T', 'C', 'Y', 'X', 'H')

    signal = signal_from_fbd(filename, frame=-1, channel=1, keepdims=False)
    assert signal.values.sum(dtype=numpy.uint64) == 0  # channel 1 is empty
    assert signal.shape == (256, 256, 64)
    assert signal.dims == ('Y', 'X', 'H')

    signal = signal_from_fbd(filename, frame=1, channel=0, keepdims=False)
    assert signal.values.sum(dtype=numpy.uint64) == 1033137
    assert signal.shape == (256, 256, 64)
    assert signal.dims == ('Y', 'X', 'H')

    signal = signal_from_fbd(filename, frame=1, channel=0)
    assert signal.values.sum(dtype=numpy.uint64) == 1033137
    assert signal.shape == (1, 1, 256, 256, 64)
    assert signal.dims == ('T', 'C', 'Y', 'X', 'H')

    filename = fetch('simfcs.r64')
    with pytest.raises(lfdfiles.LfdFileError):
        signal_from_fbd(filename)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_signal_from_bh():
    """Test read SimFCS B&H file."""
    filename = fetch('simfcs.b&h')
    signal = signal_from_bh(filename)
    assert signal.values.sum() == 7973051.0
    assert signal.dtype == numpy.float32
    assert signal.shape == (256, 256, 256)
    assert signal.dims == ('H', 'Y', 'X')
    assert not signal.coords

    filename = fetch('simfcs.r64')
    with pytest.raises(lfdfiles.LfdFileError):
        signal_from_bh(filename)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_signal_from_bhz():
    """Test read SimFCS BHZ file."""
    filename = fetch('simfcs.bhz')
    signal = signal_from_bhz(filename)
    assert signal.values.sum() == 7973051.0
    assert signal.dtype == numpy.float32
    assert signal.shape == (256, 256, 256)
    assert signal.dims == ('H', 'Y', 'X')
    assert not signal.coords

    filename = fetch('simfcs.r64')
    with pytest.raises(lfdfiles.LfdFileError):
        signal_from_bhz(filename)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_signal_from_b64():
    """Test read SimFCS B64 file."""
    filename = fetch('simfcs.b64')
    signal = signal_from_b64(filename)
    assert signal.values.sum(dtype=numpy.int64) == 8386914853
    assert signal.dtype == numpy.int16
    assert signal.shape == (22, 1024, 1024)
    assert signal.dims == ('I', 'Y', 'X')
    assert not signal.coords

    filename = fetch('simfcs.r64')
    with pytest.raises(lfdfiles.LfdFileError):
        signal_from_b64(filename)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_signal_from_z64():
    """Test read SimFCS Z64 file."""
    filename = fetch('simfcs.z64')
    signal = signal_from_z64(filename)
    assert signal.values.sum() == 5536049.0
    assert signal.dtype == numpy.float32
    assert signal.shape == (256, 256, 256)
    assert signal.dims == ('Q', 'Y', 'X')
    assert not signal.coords

    filename = fetch('simfcs.r64')
    with pytest.raises(lfdfiles.LfdFileError):
        signal_from_z64(filename)


def test_phasor_ometiff_multiharmonic():
    """Test storing multi-harmonic phasor coordinates as OME-TIFF."""
    description = 'PhasorPy\n  <&test> test'
    data = numpy.random.random_sample((3, 31, 35, 31))
    data[:, 0, 0] = numpy.nan

    with TempFileName('multiharmonic.ome.tif') as filename:
        phasor_to_ometiff(
            filename,
            data[0],
            data,
            data[::-1],
            dims='TYX',
            description=description,
        )

        with tifffile.TiffFile(filename) as tif:
            assert tif.is_ome
            assert not tif.is_bigtiff
            assert not tif.pages.first.is_tiled
            assert tif.pages.first.dtype == numpy.float32
            assert len(tif.series) == 3

        # TODO: test that file can be opened by Bio-Formats/Fiji

        mean, real, imag, attrs = phasor_from_ometiff(filename)
        assert attrs['harmonic'] == 1
        assert_almost_equal(mean, data[0])
        assert_almost_equal(real, data[0])
        assert_almost_equal(imag, data[-1])

        for harmonic in ('all', [1, 2, 3]):
            mean, real, imag, attrs = phasor_from_ometiff(
                filename, harmonic=harmonic
            )
            assert attrs['harmonic'] == [1, 2, 3]
            assert attrs['dims'] == ('T', 'Y', 'X')
            assert attrs['description'] == description
            assert 'frequency' not in attrs
            assert_almost_equal(mean, data[0])
            assert_almost_equal(real, data)
            assert_almost_equal(imag, data[::-1])

        mean, real, imag, attrs = phasor_from_ometiff(
            filename, harmonic=[1, 3]
        )
        assert attrs['harmonic'] == [1, 3]
        assert_almost_equal(mean, data[0])
        assert_almost_equal(real, data[[0, 2]])
        assert_almost_equal(imag, data[[2, 0]])

        mean, real, imag, attrs = phasor_from_ometiff(filename, harmonic=2)
        assert attrs['harmonic'] == 2
        assert_almost_equal(mean, data[0])
        assert_almost_equal(real, data[1])
        assert_almost_equal(imag, data[1])

        mean, real, imag, attrs = phasor_from_ometiff(filename, harmonic=[2])
        assert attrs['harmonic'] == [2]
        assert_almost_equal(mean, data[0])
        assert_almost_equal(real, data[[1]])
        assert_almost_equal(imag, data[[1]])

        with pytest.raises(IndexError):
            mean, real, imag, attrs = phasor_from_ometiff(filename, harmonic=4)


def test_phasor_ometiff_tiled():
    """Test storing phasor coordinates as tiled OME-TIFF."""
    data = numpy.random.random_sample((1281, 1283))
    data[0, 0] = numpy.nan

    with TempFileName('tiled.ome.tif') as filename:
        phasor_to_ometiff(
            filename,
            data,
            data,
            data,
            compression='zlib',
            frequency=80.0,
            harmonic=2,
            dtype=numpy.float64,
            bigtiff=True,
        )

        with tifffile.TiffFile(filename) as tif:
            assert tif.is_ome
            assert tif.is_bigtiff
            assert tif.pages.first.is_tiled
            assert tif.pages.first.dtype == numpy.float64
            assert len(tif.series) == 5

        # TODO: test that file can be opened by Bio-Formats/Fiji

        for harmonic in (None, 'all', 2):
            mean, real, imag, attrs = phasor_from_ometiff(
                filename, harmonic=harmonic
            )
            assert attrs['harmonic'] == 2
            assert attrs['dims'] == ('Y', 'X')
            assert attrs['frequency'] == 80.0
            assert_almost_equal(mean, data)
            assert_almost_equal(real, data)
            assert_almost_equal(imag, data)

        mean, real, imag, attrs = phasor_from_ometiff(filename, harmonic=[2])
        assert attrs['harmonic'] == [2]
        assert_almost_equal(mean, data)
        assert_almost_equal(real, data.reshape(1, *mean.shape))
        assert_almost_equal(imag, data.reshape(1, *mean.shape))

        with pytest.raises(IndexError):
            mean, real, imag, attrs = phasor_from_ometiff(filename, harmonic=0)
        with pytest.raises(IndexError):
            mean, real, imag, attrs = phasor_from_ometiff(filename, harmonic=1)


def test_phasor_ometiff_scalar():
    """Test scalar storing phasor coordinates as OME-TIFF."""
    data = numpy.random.random_sample((1,))

    # no harmonic dimension
    with TempFileName('scalar.ome.tif') as filename:
        phasor_to_ometiff(filename, data, data, data, harmonic=[1])

        with tifffile.TiffFile(filename) as tif:
            assert tif.is_ome
        # TODO: test that file can be opened by Bio-Formats/Fiji

        for harmonic in (None, 'all', 1):
            mean, real, imag, attrs = phasor_from_ometiff(
                filename, harmonic=harmonic
            )
            assert attrs['harmonic'] == 1
            assert attrs['dims'] == ('Y', 'X')
            assert mean.shape == (1, 1)
            assert_almost_equal(mean, data.reshape(mean.shape))
            assert_almost_equal(real, data.reshape(mean.shape))
            assert_almost_equal(imag, data.reshape(mean.shape))

        mean, real, imag, attrs = phasor_from_ometiff(filename, harmonic=[1])
        assert attrs['harmonic'] == [1]
        assert mean.shape == (1, 1)
        assert_almost_equal(mean, data.reshape(mean.shape))
        assert_almost_equal(real, data.reshape(1, *mean.shape))
        assert_almost_equal(imag, data.reshape(1, *mean.shape))

        with pytest.raises(IndexError):
            mean, real, imag, attrs = phasor_from_ometiff(filename, harmonic=2)


def test_phasor_ometiff_scalar_multiharmonic():
    """Test scalar storing phasor coordinates as OME-TIFF."""
    data = numpy.random.random_sample((3, 1))

    with TempFileName('scalar.ome.tif') as filename:
        phasor_to_ometiff(
            filename,
            data[0],
            data,
            data,
            harmonic=[2, 3, 4],
            # dims='X',
        )

        with tifffile.TiffFile(filename) as tif:
            assert tif.is_ome

        # TODO: test that file can be opened by Bio-Formats/Fiji

        mean, real, imag, attrs = phasor_from_ometiff(filename)
        assert attrs['harmonic'] == 2
        assert_almost_equal(mean, data[0].reshape(mean.shape))
        assert_almost_equal(real, data[0].reshape(mean.shape))
        assert_almost_equal(imag, data[0].reshape(mean.shape))

        for harmonic in ('all', [2, 3, 4]):
            mean, real, imag, attrs = phasor_from_ometiff(
                filename, harmonic=harmonic
            )
            assert attrs['dims'] == ('Y', 'X')
            assert attrs['harmonic'] == [2, 3, 4]
            assert 'frequency' not in attrs
            assert_almost_equal(mean, data[0].reshape(mean.shape))
            assert_almost_equal(real, data.reshape(real.shape))
            assert_almost_equal(imag, data.reshape(imag.shape))

        mean, real, imag, attrs = phasor_from_ometiff(filename, harmonic=3)
        assert attrs['harmonic'] == 3
        assert_almost_equal(mean, data[0].reshape(mean.shape))
        assert_almost_equal(real, data[1].reshape(mean.shape))
        assert_almost_equal(imag, data[1].reshape(mean.shape))

        mean, real, imag, attrs = phasor_from_ometiff(filename, harmonic=[3])
        assert attrs['harmonic'] == [3]
        assert_almost_equal(mean, data[0].reshape(mean.shape))
        assert_almost_equal(real, data[1].reshape(1, *mean.shape))
        assert_almost_equal(imag, data[1].reshape(1, *mean.shape))


def test_phasor_to_ometiff_exceptions():
    """Test phasor_to_ometiff function exceptions."""
    data = numpy.random.random_sample((3, 35, 31))

    with TempFileName('exception.ome.tif') as filename:

        # not a floating point type
        with pytest.raises(ValueError):
            phasor_to_ometiff(filename, *data, dtype=numpy.int16)

        # real.shape != imag.shape
        with pytest.raises(ValueError):
            phasor_to_ometiff(filename, data, data, data[0])

        # mean.shape != real.shape[-mean.ndim :]
        with pytest.raises(ValueError):
            phasor_to_ometiff(filename, data[:1], data, data)

        # invalid harmonic, not an integer
        with pytest.raises(TypeError):
            phasor_to_ometiff(
                filename, *data, harmonic=[[1]]  # type: ignore[list-item]
            )

        # invalid harmonic
        with pytest.raises(ValueError):
            phasor_to_ometiff(filename, data[0], data, data, harmonic=[1, 2])

        # frequency must be scalar
        with pytest.raises(ValueError):
            phasor_to_ometiff(filename, *data, frequency=[80, 90])

        # len(dims) != mean.ndim
        with pytest.raises(ValueError):
            phasor_to_ometiff(filename, *data, dims='ZYX')


def test_phasor_from_ometiff_exceptions(caplog):
    """Test phasor_from_ometiff function exceptions and warnings."""
    data = numpy.random.random_sample((3, 35, 31)).astype(numpy.float32)
    kwargs = {'photometric': 'minisblack'}

    with TempFileName('invalid.ome.tif') as filename:

        # series[0].name != 'Phasor mean'
        with tifffile.TiffWriter(filename, ome=True) as tif:
            tif.write(data)

        with pytest.raises(ValueError):
            phasor_from_ometiff(filename)

        # harmonic does not match phasor shape
        with tifffile.TiffWriter(filename, ome=True) as tif:
            tif.write(data[0], metadata={'Name': 'Phasor mean'}, **kwargs)
            tif.write(data, metadata={'Name': 'Phasor real'}, **kwargs)
            tif.write(data, metadata={'Name': 'Phasor imag'}, **kwargs)
            tif.write([[1, 2]], metadata={'Name': 'Phasor harmonic'})

        mean, real, imag, attrs = phasor_from_ometiff(filename)
        assert 'does not match phasor' in caplog.text
        assert attrs['harmonic'] == 1

        # shapes don't match
        with tifffile.TiffWriter(filename, ome=True) as tif:
            tif.write(data[0], metadata={'Name': 'Phasor mean'}, **kwargs)
            tif.write(data[:, :-1], metadata={'Name': 'Phasor real'}, **kwargs)
            tif.write(data, metadata={'Name': 'Phasor imag'}, **kwargs)

        mean, real, imag, attrs = phasor_from_ometiff(filename)
        assert 'imag.shape' in caplog.text
        assert 'mean.shape' in caplog.text
        assert attrs['harmonic'] == 1
        assert mean.shape == (35, 31)
        assert real.shape == (34, 31)
        assert imag.shape == (35, 31)

        # harmonic not present
        with tifffile.TiffWriter(filename, ome=True) as tif:
            tif.write(data[0], metadata={'Name': 'Phasor mean'}, **kwargs)
            tif.write(data[1], metadata={'Name': 'Phasor real'}, **kwargs)
            tif.write(data[2], metadata={'Name': 'Phasor imag'}, **kwargs)

        mean, real, imag, attrs = phasor_from_ometiff(filename)
        assert attrs['harmonic'] == 1

    filename = fetch('paramecium.lsm')
    with pytest.raises(ValueError):
        phasor_from_ometiff(filename)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_phasor_from_simfcs_referenced_ref():
    """Test phasor_from_simfcs_referenced with SimFCS REF file."""
    filename = fetch('simfcs.ref')
    mean, real, imag, attrs = phasor_from_simfcs_referenced(filename)
    assert mean.dtype == numpy.float32
    assert mean.shape == (256, 256)
    assert real.shape == (256, 256)
    assert imag.shape == (256, 256)
    assert attrs['dims'] == ('Y', 'X')
    assert_allclose(numpy.nanmean(mean), 213.09485, atol=1e-3)
    assert_allclose(numpy.nanmean(real), 0.40588844, atol=1e-3)
    assert_allclose(numpy.nanmean(imag), 0.34678984, atol=1e-3)

    for harmonic in ('all', [1, 2]):
        mean, real, imag, attrs = phasor_from_simfcs_referenced(
            filename, harmonic=harmonic
        )
        assert mean.shape == (256, 256)
        assert real.shape == (2, 256, 256)
        assert imag.shape == (2, 256, 256)
        assert_allclose(numpy.nanmean(mean), 213.09485)
        assert_allclose(
            numpy.nanmean(real, axis=(1, 2)),
            [0.40588844, 0.21527097],
            atol=1e-3,
        )
        assert_allclose(
            numpy.nanmean(imag, axis=(1, 2)),
            [0.34678984, 0.26586965],
            atol=1e-3,
        )

    filename = fetch('simfcs.b64')
    with pytest.raises(ValueError):
        phasor_from_simfcs_referenced(filename)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_phasor_from_simfcs_referenced_r64():
    """Test phasor_from_simfcs_referenced with SimFCS R64 file."""
    filename = fetch('simfcs.r64')
    mean, real, imag, attrs = phasor_from_simfcs_referenced(filename)
    assert mean.dtype == numpy.float32
    assert mean.shape == (256, 256)
    assert real.shape == (256, 256)
    assert imag.shape == (256, 256)
    assert attrs['dims'] == ('Y', 'X')
    assert_allclose(numpy.nanmean(mean), 0.562504, atol=1e-3)
    assert_allclose(numpy.nanmean(real), 0.48188266, atol=1e-3)
    assert_allclose(numpy.nanmean(imag), 0.32413888, atol=1e-3)

    mean, real, imag, attrs = phasor_from_simfcs_referenced(
        filename, harmonic=[1, 2]
    )
    assert mean.shape == (256, 256)
    assert real.shape == (2, 256, 256)
    assert imag.shape == (2, 256, 256)
    assert_allclose(numpy.nanmean(mean), 0.562504, atol=1e-3)
    assert_allclose(
        numpy.nanmean(real, axis=(1, 2)), [0.48188266, 0.13714617], atol=1e-3
    )
    assert_allclose(
        numpy.nanmean(imag, axis=(1, 2)), [0.32413888, 0.38899058], atol=1e-3
    )


def test_phasor_to_simfcs_referenced():
    """Test phasor_to_simfcs_referenced with square image."""
    data = numpy.random.random_sample((3, 32, 32))
    data[..., 0, 0] = numpy.nan

    with TempFileName('simple.r64') as filename:
        phasor_to_simfcs_referenced(filename, *data)

        mean, real, imag, attrs = phasor_from_simfcs_referenced(filename)
        assert mean.shape == (32, 32)
        assert real.shape == (32, 32)
        assert imag.shape == (32, 32)
        assert attrs['dims'] == ('Y', 'X')
        assert_allclose(mean, data[0], atol=1e-3)
        assert_allclose(real, data[1], atol=1e-3)
        assert_allclose(imag, data[2], atol=1e-3)


def test_phasor_to_simfcs_referenced_scalar():
    """Test phasor_to_simfcs_referenced with scalar."""
    data = numpy.random.random_sample((3, 1))

    with TempFileName('simple.r64') as filename:
        phasor_to_simfcs_referenced(filename, *data)

        mean, real, imag, attrs = phasor_from_simfcs_referenced(filename)
        assert mean.shape == (4, 4)
        assert real.shape == (4, 4)
        assert imag.shape == (4, 4)
        assert attrs['dims'] == ('Y', 'X')
        assert_allclose(mean[0, 0], data[0], atol=1e-3)
        assert_allclose(real[0, 0], data[1], atol=1e-3)
        assert_allclose(imag[0, 0], data[2], atol=1e-3)
        with pytest.warns(RuntimeWarning):
            assert numpy.isnan(numpy.nanmean(real[1:, 1:]))
            assert numpy.isnan(numpy.nanmean(imag[1:, 1:]))


def test_phasor_to_simfcs_referenced_exceptions():
    """Test phasor_to_simfcs_referenced exceptions."""
    data = numpy.random.random_sample((32, 32))

    with TempFileName('simple.r64') as filename:
        phasor_to_simfcs_referenced(filename, data, data, data)
        with pytest.raises(ValueError):
            phasor_to_simfcs_referenced(filename + '.bin', data, data, data)
        with pytest.raises(ValueError):
            phasor_to_simfcs_referenced(filename, data, data, data[1:])
        with pytest.raises(ValueError):
            phasor_to_simfcs_referenced(filename, data[1:], data, data)
        with pytest.raises(ValueError):
            phasor_to_simfcs_referenced(filename, data, data, data, size=3)


def test_phasor_to_simfcs_referenced_multiharmonic():
    """Test phasor_to_simfcs_referenced with multi-harmonic phasor."""
    data = numpy.random.random_sample((3, 4, 35, 31))
    data[..., 0, 0] = numpy.nan

    with TempFileName('multiharmonic.r64', pattern=True) as filename:
        phasor_to_simfcs_referenced(
            filename,
            data[0],
            data,
            data[::-1],
            size=32,
            dims='tyx',
        )

        name, ext = os.path.splitext(filename)
        files = glob(name + '*' + ext)
        assert len(files) == 16

        mean, real, imag, attrs = phasor_from_simfcs_referenced(
            name + '_h2_t3_y32_x00' + ext, harmonic='all'
        )
        assert mean.shape == (32, 32)
        assert real.shape == (2, 32, 32)
        assert imag.shape == (2, 32, 32)
        assert attrs['dims'] == ('Y', 'X')
        with pytest.warns(RuntimeWarning):
            assert numpy.isnan(numpy.nanmean(real[1]))
            assert numpy.isnan(numpy.nanmean(imag[1]))
            assert numpy.isnan(numpy.nanmean(mean[..., -1]))
            assert numpy.isnan(numpy.nanmean(mean[..., 3, :]))
        assert_allclose(mean[:3, :31], data[0, 3, 32:35, :31], atol=1e-3)
        assert_allclose(real[0, :3, :31], data[2, 3, 32:35, :31], atol=1e-3)
        assert_allclose(imag[0, :3, :31], data[0, 3, 32:35, :31], atol=1e-3)

        for fname in files:
            mean, real, imag, attrs = phasor_from_simfcs_referenced(
                fname, harmonic='all'
            )
            assert mean.shape == (32, 32)
            assert real.shape == (2, 32, 32)
            assert imag.shape == (2, 32, 32)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_phasor_from_lif():
    """Test read phasor coordinates from Leica LIF file."""
    filename = fetch('FLIM_testdata.lif')
    mean, real, imag, attrs = phasor_from_lif(filename)
    for data in (mean, real, imag):
        assert data.shape == (1024, 1024)
        assert data.dtype == numpy.float32
    assert attrs['frequency'] == 19.505
    assert 'harmonic' not in attrs

    # select series
    mean1, real1, imag1, attrs = phasor_from_lif(
        filename, series='FLIM Compressed'
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


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_signal_from_lif():
    """Test read hyperspectral signal from Leica LIF file."""
    filename = private_file('ScanModesExamples.lif')
    signal = signal_from_lif(filename)
    assert signal.dims == ('C', 'Y', 'X')
    assert signal.shape == (9, 128, 128)
    assert signal.dtype == numpy.uint8
    assert_allclose(signal.coords['C'].data[[0, 1]], [560.0, 580.0])

    # select series
    signal = signal_from_lif(filename, series='XYZLambdaT')
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

    # series does not contain dim
    with pytest.raises(ValueError):
        signal_from_lif(filename, series='XYZLambdaT', dim='Λ')

    # file does not contain hyperspectral signal
    filename = fetch('FLIM_testdata.lif')
    with pytest.raises(ValueError):
        signal_from_lif(filename)


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
