"""Tests for the phasorpy.io module."""

import os
import tempfile
from glob import glob

import numpy
import pytest
import tifffile
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
)

from phasorpy.datasets import fetch
from phasorpy.io import (
    phasor_from_ometiff,
    phasor_from_simfcs_referenced,
    phasor_to_ometiff,
    phasor_to_simfcs_referenced,
    read_b64,
    read_bh,
    read_bhz,
    read_fbd,
    read_flif,
    read_flimlabs_json,
    read_ifli,
    read_imspector_tiff,
    read_lsm,
    read_ptu,
    read_sdt,
    read_z64,
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
    assert data.values.sum(dtype=numpy.uint64) == 142328063165
    assert data.dtype == numpy.uint16
    assert data.shape == (10, 21, 32, 256, 256)
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
    assert data.values.sum(dtype=numpy.uint64) == 14050194
    assert data.dtype == numpy.uint8
    assert data.shape == (30, 512, 512)
    assert data.dims == ('C', 'Y', 'X')
    assert_almost_equal(
        data.coords['C'][[0, -1]], [4.2301329e-7, 7.1301329e-7], decimal=12
    )
    assert_almost_equal(
        data.coords['X'][[0, -1]], [0.0, 0.000424265835], decimal=9
    )


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_imspector_tiff():
    """Test read Imspector FLIM TIFF file."""
    data = read_imspector_tiff(fetch('Embryo.tif'))
    assert data.values.sum(dtype=numpy.uint64) == 31348436
    assert data.dtype == numpy.uint16
    assert data.shape == (56, 512, 512)
    assert data.dims == ('H', 'Y', 'X')
    assert_almost_equal(
        data.coords['H'][[0, -1]], [0.0, 0.218928482143], decimal=12
    )
    assert data.attrs['frequency'] == 80.10956424883184


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_imspector_tiff_t():
    """Test read Imspector FLIM TIFF file with TCSPC in T-axis."""
    data = read_imspector_tiff(private_file('ZF-1100_noEF.tif'))
    assert data.values.sum(dtype=numpy.uint64) == 18636271
    assert data.dtype == numpy.uint16
    assert data.shape == (56, 512, 512)
    assert data.dims == ('H', 'Y', 'X')
    assert_almost_equal(
        data.coords['H'][[0, -1]], [0.0, 0.218928482143], decimal=12
    )
    assert data.attrs['frequency'] == 80.10956424883184


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_flimlabs_json():
    """Test read FLIMLABS JSON image file."""
    filename = private_file('test03_1733492714_imaging.json')
    data = read_flimlabs_json(filename)
    assert data.values.sum(dtype=numpy.uint64) == 4680256
    assert data.dtype == numpy.uint16
    assert data.shape == (3, 256, 256, 256)
    assert data.dims == ('C', 'Y', 'X', 'H')
    assert_almost_equal(
        data.coords['H'][[0, -1]], [0.0, 12.451171875], decimal=12
    )
    assert_array_equal(data.coords['C'], [0, 1, 2])
    assert data.attrs['frequency'] == 80.0

    data = read_flimlabs_json(filename, channel=1, dtype=numpy.uint8)
    assert data.values.sum(dtype=numpy.uint64) == 1388562
    assert data.dtype == numpy.uint8
    assert data.shape == (256, 256, 256)
    assert data.dims == ('Y', 'X', 'H')

    with pytest.raises(ValueError):
        data = read_flimlabs_json(filename, channel=1, dtype=numpy.int8)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_read_sdt():
    """Test read Becker & Hickl SDT file."""
    filename = fetch('tcspc.sdt')
    data = read_sdt(filename)
    assert data.values.sum(dtype=numpy.uint64) == 224606420
    assert data.dtype == numpy.uint16
    assert data.shape == (128, 128, 256)
    assert data.dims == ('Y', 'X', 'H')
    assert_almost_equal(
        data.coords['H'][[0, -1]], [0.0, 1.2451172e-8], decimal=12
    )


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_read_sdt_fcs():
    """Test read Becker & Hickl SDT FCS file."""
    # file provided by lmalacrida via email on Nov 13, 2023
    filename = private_file('j3_405_z1.sdt')
    data = read_sdt(filename)
    assert data.values.sum(dtype=numpy.uint64) == 16929780
    assert data.dtype == numpy.uint16
    assert data.shape == (512, 512, 1024)
    assert data.dims == ('Y', 'X', 'H')
    assert_almost_equal(
        data.coords['H'][[0, -1]], [0.0, 1.666157034737e-08], decimal=12
    )


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_read_ifli():
    """Test read ISS VistaVision file."""
    # TODO: test spectral file
    filename = fetch('frequency_domain.ifli')
    data = read_ifli(filename, memmap=True)
    assert data.values.sum() == 62603316.0
    assert data.dtype == numpy.float32
    assert data.shape == (256, 256, 4, 3)
    assert data.dims == ('Y', 'X', 'F', 'S')
    assert_array_equal(data.coords['S'].data, ['mean', 'real', 'imag'])
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
def test_read_flif():
    """Test read FlimFast FLIF file."""
    # TODO: test time series
    filename = fetch('flimfast.flif')
    data = read_flif(filename)
    data = read_flif(fetch('flimfast.flif'))
    assert data.values.sum(dtype=numpy.uint64) == 706233156
    assert data.dtype == 'uint16'
    assert data.shape == (32, 220, 300)
    assert data.dims == ('H', 'Y', 'X')
    assert_almost_equal(data.coords['H'].data[[1, -1]], [0.1963495, 6.086836])
    assert_almost_equal(data.attrs['frequency'], 80.6520004272461)
    assert_almost_equal(data.attrs['ref_phase'], 120.63999938964844)
    assert_almost_equal(data.attrs['ref_mod'], 31.670000076293945)
    assert_almost_equal(data.attrs['ref_tauphase'], 1.0160000324249268)
    assert_almost_equal(data.attrs['ref_taumod'], 1.2580000162124634)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_read_ptu():
    """Test read PicoQuant PTU file."""
    filename = fetch('hazelnut_FLIM_single_image.ptu')
    data = read_ptu(filename, frame=-1, channel=0, dtime=0, keepdims=False)
    assert data.values.sum(dtype=numpy.uint64) == 6064854
    assert data.dtype == numpy.uint16
    assert data.shape == (256, 256, 132)
    assert data.dims == ('Y', 'X', 'H')
    assert_almost_equal(
        data.coords['H'].data[[1, -1]], [9.69696970e-11, 1.2703030e-08]
    )
    assert data.attrs['frequency'] == 78.02

    data = read_ptu(
        filename,
        frame=-1,
        channel=0,
        dtime=None,
        keepdims=True,
        trimdims='TC',
    )
    assert data.values.sum(dtype=numpy.uint64) == 6065123
    assert data.dtype == numpy.uint16
    assert data.shape == (1, 256, 256, 1, 4096)
    assert data.dims == ('T', 'Y', 'X', 'C', 'H')
    assert_almost_equal(
        data.coords['H'].data[[1, -1]], [9.69696970e-11, 3.97090909e-07]
    )
    assert data.attrs['frequency'] == 78.02


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_read_fbd():
    """Test read FLIMbox FBD file."""
    # TODO: test files with different firmwares
    # TODO: gather public FBD files and upload to Zenodo
    filename = private_file('convallaria_000$EI0S.fbd')
    data = read_fbd(filename)
    assert data.values.sum(dtype=numpy.uint64) == 9310275
    assert data.dtype == numpy.uint16
    assert data.shape == (9, 2, 256, 256, 64)
    assert data.dims == ('T', 'C', 'Y', 'X', 'H')
    assert_almost_equal(data.coords['H'].data[[1, -1]], [0.0981748, 6.1850105])
    assert_almost_equal(data.attrs['frequency'], 40.0)

    data = read_fbd(filename, frame=-1, channel=0)
    assert data.values.sum(dtype=numpy.uint64) == 9310275
    assert data.shape == (1, 1, 256, 256, 64)
    assert data.dims == ('T', 'C', 'Y', 'X', 'H')

    data = read_fbd(filename, frame=-1, channel=1, keepdims=False)
    assert data.values.sum(dtype=numpy.uint64) == 0  # channel 1 is empty
    assert data.shape == (256, 256, 64)
    assert data.dims == ('Y', 'X', 'H')

    data = read_fbd(filename, frame=1, channel=0, keepdims=False)
    assert data.values.sum(dtype=numpy.uint64) == 1033137
    assert data.shape == (256, 256, 64)
    assert data.dims == ('Y', 'X', 'H')

    data = read_fbd(filename, frame=1, channel=0)
    assert data.values.sum(dtype=numpy.uint64) == 1033137
    assert data.shape == (1, 1, 256, 256, 64)
    assert data.dims == ('T', 'C', 'Y', 'X', 'H')


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_read_bh():
    """Test read SimFCS B&H file."""
    filename = fetch('simfcs.b&h')
    data = read_bh(filename)
    assert data.values.sum() == 7973051.0
    assert data.dtype == numpy.float32
    assert data.shape == (256, 256, 256)
    assert data.dims == ('H', 'Y', 'X')
    assert not data.coords


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_read_bhz():
    """Test read SimFCS BHZ file."""
    filename = fetch('simfcs.bhz')
    data = read_bhz(filename)
    assert data.values.sum() == 7973051.0
    assert data.dtype == numpy.float32
    assert data.shape == (256, 256, 256)
    assert data.dims == ('H', 'Y', 'X')
    assert not data.coords


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_read_b64():
    """Test read SimFCS B64 file."""
    filename = fetch('simfcs.b64')
    data = read_b64(filename)
    assert data.values.sum(dtype=numpy.int64) == 8386914853
    assert data.dtype == numpy.int16
    assert data.shape == (22, 1024, 1024)
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
    assert data.values.sum() == 5536049.0
    assert data.dtype == numpy.float32
    assert data.shape == (256, 256, 256)
    assert data.dims == ('Q', 'Y', 'X')
    assert not data.coords


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
            axes='TYX',
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
            assert attrs['axes'] == 'TYX'
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
            assert attrs['axes'] == 'YX'
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
            assert attrs['axes'] == 'YX'
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
            # axes='X',
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
            assert attrs['axes'] == 'YX'
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

        # len(axes) != mean.ndim
        with pytest.raises(ValueError):
            phasor_to_ometiff(filename, *data, axes='ZYX')


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


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_phasor_from_simfcs_referenced_ref():
    """Test phasor_from_simfcs_referenced with SimFCS REF file."""
    filename = fetch('simfcs.ref')
    mean, real, imag = phasor_from_simfcs_referenced(filename)
    assert mean.dtype == numpy.float32
    assert mean.shape == (256, 256)
    assert real.shape == (256, 256)
    assert imag.shape == (256, 256)
    assert_allclose(numpy.nanmean(mean), 213.09485, atol=1e-3)
    assert_allclose(numpy.nanmean(real), 0.40588844, atol=1e-3)
    assert_allclose(numpy.nanmean(imag), 0.34678984, atol=1e-3)

    for harmonic in ('all', [1, 2]):
        mean, real, imag = phasor_from_simfcs_referenced(
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


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_phasor_from_simfcs_referenced_r64():
    """Test phasor_from_simfcs_referenced with SimFCS R64 file."""
    filename = fetch('simfcs.r64')
    mean, real, imag = phasor_from_simfcs_referenced(filename)
    assert mean.dtype == numpy.float32
    assert mean.shape == (256, 256)
    assert real.shape == (256, 256)
    assert imag.shape == (256, 256)
    assert_allclose(numpy.nanmean(mean), 0.562504, atol=1e-3)
    assert_allclose(numpy.nanmean(real), 0.48188266, atol=1e-3)
    assert_allclose(numpy.nanmean(imag), 0.32413888, atol=1e-3)

    mean, real, imag = phasor_from_simfcs_referenced(filename, harmonic=[1, 2])
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

        mean, real, imag = phasor_from_simfcs_referenced(filename)
        assert mean.shape == (32, 32)
        assert real.shape == (32, 32)
        assert imag.shape == (32, 32)
        assert_allclose(mean, data[0], atol=1e-3)
        assert_allclose(real, data[1], atol=1e-3)
        assert_allclose(imag, data[2], atol=1e-3)


def test_phasor_to_simfcs_referenced_scalar():
    """Test phasor_to_simfcs_referenced with scalar."""
    data = numpy.random.random_sample((3, 1))

    with TempFileName('simple.r64') as filename:
        phasor_to_simfcs_referenced(filename, *data)

        mean, real, imag = phasor_from_simfcs_referenced(filename)
        assert mean.shape == (4, 4)
        assert real.shape == (4, 4)
        assert imag.shape == (4, 4)
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
            axes='tyx',
        )

        name, ext = os.path.splitext(filename)
        files = glob(name + '*' + ext)
        assert len(files) == 16

        mean, real, imag = phasor_from_simfcs_referenced(
            name + '_h2_t3_y32_x00' + ext, harmonic='all'
        )
        assert mean.shape == (32, 32)
        assert real.shape == (2, 32, 32)
        assert imag.shape == (2, 32, 32)
        with pytest.warns(RuntimeWarning):
            assert numpy.isnan(numpy.nanmean(real[1]))
            assert numpy.isnan(numpy.nanmean(imag[1]))
            assert numpy.isnan(numpy.nanmean(mean[..., -1]))
            assert numpy.isnan(numpy.nanmean(mean[..., 3, :]))
        assert_allclose(mean[:3, :31], data[0, 3, 32:35, :31], atol=1e-3)
        assert_allclose(real[0, :3, :31], data[2, 3, 32:35, :31], atol=1e-3)
        assert_allclose(imag[0, :3, :31], data[0, 3, 32:35, :31], atol=1e-3)

        for fname in files:
            mean, real, imag = phasor_from_simfcs_referenced(
                fname, harmonic='all'
            )
            assert mean.shape == (32, 32)
            assert real.shape == (2, 32, 32)
            assert imag.shape == (2, 32, 32)


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
