"""Tests for the phasorpy.io module."""

import os
import tempfile

import numpy
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from phasorpy.datasets import fetch
from phasorpy.io import (
    read_b64,
    read_bh,
    read_bhz,
    read_ifli,
    read_lsm,
    read_ometiff_phasor,
    read_r64,
    read_ref,
    read_sdt,
    read_z64,
    write_ometiff_phasor,
)

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

    def __init__(self, name=None, remove=False):
        self.remove = remove or TEMP_DIR == tempfile.gettempdir()
        if not name:
            fh = tempfile.NamedTemporaryFile(prefix='test_')
            self.name = fh.named
            fh.close()
        else:
            self.name = os.path.join(TEMP_DIR, f'test_{name}')

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        if self.remove:
            try:
                os.remove(self.name)
            except Exception:
                pass


def private_file(filename: str, /) -> str:
    """Return path to private test file."""
    return os.path.normpath(os.path.join(PRIVATE_DIR, filename))


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_read_lsm_non_hyperspectral():
    """Test read non-hyperspectral LSM image fails."""
    filename = private_file('non_hyperspectral.lsm')
    with pytest.raises(ValueError):
        read_lsm(filename)


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_read_lsm_tzcyx():
    """Test read TZC hyperspectral LSM image."""
    filename = private_file('tzcyx.lsm')
    data = read_lsm(filename)
    assert data.shape == (10, 21, 32, 256, 256)
    assert data.dtype == numpy.uint16
    assert data.dims == ('T', 'Z', 'C', 'Y', 'X')
    assert_almost_equal(
        data.coords['C'][[0, -1]],
        [414.936272e-9, 690.47537e-9],
        decimal=12,
    )
    assert_almost_equal(
        data.coords['T'][[0, -1]], [40557.75229085, 42488.21737206]
    )
    assert_almost_equal(data.coords['Z'][[0, -1]], [0.0, 7.4440772e-05])


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_read_lsm_paramecium():
    """Test read paramecium.lsm."""
    filename = fetch('paramecium.lsm')
    data = read_lsm(filename)
    assert data.shape == (30, 512, 512)
    assert data.dtype == numpy.uint8
    assert data.dims == ('C', 'Y', 'X')
    assert_almost_equal(
        data.coords['C'][[0, -1]], [4.2301329e-7, 7.1301329e-7], decimal=12
    )
    assert_almost_equal(
        data.coords['X'][[0, -1]], [0.0, 0.000424265835], decimal=9
    )


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_read_sdt():
    """Test read Becker & Hickl SDT file."""
    filename = fetch('tcspc.sdt')
    data = read_sdt(filename)
    assert data.shape == (128, 128, 256)
    assert data.dtype == numpy.uint16
    assert data.dims == ('Y', 'X', 'H')
    assert_almost_equal(
        data.coords['H'][[0, -1]], [0.0, 1.2451172e-8], decimal=12
    )


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_read_ifli():
    """Test read ISS VistaVision file."""
    filename = fetch('frequency_domain.ifli')
    data = read_ifli(filename, memmap=True)
    assert data.shape == (256, 256, 4, 3)
    assert data.dtype == numpy.float32
    assert data.dims == ('Y', 'X', 'F', 'S')
    assert_array_equal(data.coords['S'].data, ['dc', 're', 'im'])
    assert_almost_equal(
        data.coords['F'],
        (80332416.0, 160664832.0, 240997248.0, 401662080.0),
    )
    assert_almost_equal(
        data.attrs['ref_phasor'][0], (1.1425294e7, 5.9600395e-1, -9.4883347e-1)
    )
    assert data.attrs['ref_tau'] == (2.5, 0.0)
    assert data.attrs['ref_tau_frac'] == (1.0, 0.0)
    assert data.attrs['ref_phasor'].shape == (4, 3)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_read_bh():
    """Test read SimFCS B&H file."""
    filename = fetch('simfcs.b&h')
    data = read_bh(filename)
    assert data.shape == (256, 256, 256)
    assert data.dtype == numpy.float32
    assert data.dims == ('H', 'Y', 'X')
    assert not data.coords


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_read_bhz():
    """Test read SimFCS BHZ file."""
    filename = fetch('simfcs.bhz')
    data = read_bhz(filename)
    assert data.shape == (256, 256, 256)
    assert data.dtype == numpy.float32
    assert data.dims == ('H', 'Y', 'X')
    assert not data.coords


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_read_ref():
    """Test read SimFCS REF file."""
    filename = fetch('simfcs.ref')
    data = read_ref(filename)
    assert data.shape == (5, 256, 256)
    assert data.dtype == numpy.float32
    assert data.dims == ('S', 'Y', 'X')
    assert_array_equal(
        data.coords['S'].data, ['dc', 'ph1', 'md1', 'ph2', 'md2']
    )


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_read_r64():
    """Test read SimFCS R64 file."""
    filename = fetch('simfcs.r64')
    data = read_r64(filename)
    assert data.shape == (5, 256, 256)
    assert data.dtype == numpy.float32
    assert data.dims == ('S', 'Y', 'X')
    assert_array_equal(
        data.coords['S'].data, ['dc', 'ph1', 'md1', 'ph2', 'md2']
    )


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_read_b64():
    """Test read SimFCS B64 file."""
    filename = fetch('simfcs.b64')
    data = read_b64(filename)
    assert data.shape == (22, 1024, 1024)
    assert data.dtype == numpy.int16
    assert data.dims == ('I', 'Y', 'X')
    assert not data.coords
    # filename = fetch('simfcs_image.b64')
    # with pytest.raises(ValueError):
    #     read_b64(filename)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_read_z64():
    """Test read SimFCS Z64 file."""
    filename = fetch('simfcs.z64')
    data = read_z64(filename)
    assert data.shape == (256, 256, 256)
    assert data.dtype == numpy.float32
    assert data.dims == ('Q', 'Y', 'X')
    assert not data.coords


def test_ometiff_phasor():
    """Test writing and reading phasor coordinates to OME-TIFF."""
    data = numpy.random.random_sample((3, 31, 35, 31)).astype(numpy.float32)
    dims = ('T', 'Y', 'X')
    with TempFileName('ometiff_phasor.ome.tif') as filename:
        write_ometiff_phasor(
            filename,
            data[0],
            data[1],
            data[2],
            bigtiff=False,
            axes=''.join(dims),
            compression='adobe_deflate',
        )
        dc, re, im = read_ometiff_phasor(filename)
        # TODO: test that file can be opened by Bio-Formats/Fiji

    assert_array_equal(dc, data[0])
    assert_array_equal(re, data[1])
    assert_array_equal(im, data[2])
    assert dc.dims == dims
    assert re.dims == dims
    assert im.dims == dims
    assert not dc.coords
