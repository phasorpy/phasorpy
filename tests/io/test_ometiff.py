"""Test OME-TIFF file reader and writer functions."""

import numpy
import pytest
import tifffile
from _conftest import SKIP_FETCH, TempFileName
from numpy.testing import assert_almost_equal

from phasorpy.datasets import fetch
from phasorpy.io import phasor_from_ometiff, phasor_to_ometiff


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

    if not SKIP_FETCH:
        filename = fetch('paramecium.lsm')
        with pytest.raises(ValueError):
            phasor_from_ometiff(filename)


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
