"""Test SimFCS file reader functions."""

import os
from glob import glob
from tempfile import TemporaryDirectory

import lfdfiles
import numpy
import pytest
from _conftest import SKIP_FETCH, SKIP_PRIVATE, TempFileName, private_file
from numpy.testing import assert_allclose, assert_almost_equal

from phasorpy.datasets import fetch
from phasorpy.io import (
    phasor_from_simfcs_referenced,
    phasor_to_simfcs_referenced,
    signal_from_b64,
    signal_from_bh,
    signal_from_bhz,
    signal_from_fbd,
    signal_from_z64,
)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_signal_from_fbd():
    """Test read FLIMbox FBD file."""
    # TODO: test files with different firmwares
    # TODO: gather public FBD files and upload to Zenodo
    filename = fetch('Convallaria_$EI0S.fbd')
    signal = signal_from_fbd(filename, channel=None, keepdims=True)
    assert signal.values.sum(dtype=numpy.uint64) == 9310275
    assert signal.dtype == numpy.uint16
    assert signal.shape == (9, 2, 256, 256, 64)
    assert signal.dims == ('T', 'C', 'Y', 'X', 'H')
    assert_almost_equal(
        signal.coords['H'].data[[1, -1]], [0.0981748, 6.1850105]
    )
    assert_almost_equal(signal.attrs['frequency'], 40.0)

    attrs = signal.attrs
    assert attrs['frequency'] == 40.0
    assert attrs['harmonic'] == 2
    assert attrs['flimbox_firmware']['secondharmonic'] == 1
    assert attrs['flimbox_header'] is not None
    assert 'flimbox_settings' not in attrs

    signal = signal_from_fbd(filename, frame=-1, channel=0, keepdims=True)
    assert signal.values.sum(dtype=numpy.uint64) == 9310275
    assert signal.shape == (1, 1, 256, 256, 64)
    assert signal.dims == ('T', 'C', 'Y', 'X', 'H')

    signal = signal_from_fbd(filename, frame=-1, channel=1)
    assert signal.values.sum(dtype=numpy.uint64) == 0  # channel 1 is empty
    assert signal.shape == (256, 256, 64)
    assert signal.dims == ('Y', 'X', 'H')

    signal = signal_from_fbd(filename, frame=1, channel=0, keepdims=False)
    assert signal.values.sum(dtype=numpy.uint64) == 1033137
    assert signal.shape == (256, 256, 64)
    assert signal.dims == ('Y', 'X', 'H')

    signal = signal_from_fbd(filename, frame=1, channel=0, keepdims=True)
    assert signal.values.sum(dtype=numpy.uint64) == 1033137
    assert signal.shape == (1, 1, 256, 256, 64)
    assert signal.dims == ('T', 'C', 'Y', 'X', 'H')

    with pytest.raises(IndexError):
        signal_from_fbd(filename, frame=9)

    with pytest.raises(IndexError):
        signal_from_fbd(filename, channel=2)

    filename = fetch('simfcs.r64')
    with pytest.raises(ValueError):
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
    assert_allclose(
        numpy.nanmean(mean, dtype=numpy.float64), 213.09485, atol=1e-3
    )
    assert_allclose(
        numpy.nanmean(real, dtype=numpy.float64), 0.40588844, atol=1e-3
    )
    assert_allclose(
        numpy.nanmean(imag, dtype=numpy.float64), 0.34678984, atol=1e-3
    )

    for harmonic in ('all', [1, 2]):
        mean, real, imag, attrs = phasor_from_simfcs_referenced(
            filename, harmonic=harmonic
        )
        assert mean.shape == (256, 256)
        assert real.shape == (2, 256, 256)
        assert imag.shape == (2, 256, 256)
        assert_allclose(
            numpy.nanmean(mean, dtype=numpy.float64), 213.09485, atol=1e-3
        )
        assert_allclose(
            numpy.nanmean(real, axis=(1, 2), dtype=numpy.float64),
            [0.40588844, 0.21527097],
            atol=1e-3,
        )
        assert_allclose(
            numpy.nanmean(imag, axis=(1, 2), dtype=numpy.float64),
            [0.34678984, 0.26586965],
            atol=1e-3,
        )

    filename = fetch('simfcs.b64')
    with pytest.raises(ValueError):
        phasor_from_simfcs_referenced(filename)

    # wrong extension
    with TempFileName('empty.ret') as filename:
        with open(filename, 'wb') as fh:
            fh.write(b'')
        with pytest.raises(ValueError):
            phasor_from_simfcs_referenced(filename)

    # empty file
    with TempFileName('empty.ref') as filename:
        with open(filename, 'wb') as fh:
            fh.write(b'0')
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
    assert_allclose(
        numpy.nanmean(mean, dtype=numpy.float64), 0.562504, atol=1e-3
    )
    assert_allclose(
        numpy.nanmean(real, dtype=numpy.float64), 0.48188266, atol=1e-3
    )
    assert_allclose(
        numpy.nanmean(imag, dtype=numpy.float64), 0.32413888, atol=1e-3
    )

    mean, real, imag, attrs = phasor_from_simfcs_referenced(
        filename, harmonic=[1, 2]
    )
    assert mean.shape == (256, 256)
    assert real.shape == (2, 256, 256)
    assert imag.shape == (2, 256, 256)
    assert_allclose(
        numpy.nanmean(mean, dtype=numpy.float64), 0.562504, atol=1e-3
    )
    assert_allclose(
        numpy.nanmean(real, axis=(1, 2), dtype=numpy.float64),
        [0.48188266, 0.13714617],
        atol=1e-3,
    )
    assert_allclose(
        numpy.nanmean(imag, axis=(1, 2), dtype=numpy.float64),
        [0.32413888, 0.38899058],
        atol=1e-3,
    )


@pytest.mark.skipif(SKIP_PRIVATE, reason='file is private')
def test_phasor_from_simfcs_referenced_re2():
    """Test phasor_from_simfcs_referenced with RE2 file."""
    filename = private_file(
        'Mosaic04_10x3_FOV600_z95_32A1_t06_uncalibrated.re2'
    )
    mean, real, imag, attrs = phasor_from_simfcs_referenced(filename)
    assert mean.dtype == numpy.float32
    assert mean.shape == (1200, 1200)
    assert real.shape == (1200, 1200)
    assert imag.shape == (1200, 1200)
    assert attrs['dims'] == ('Y', 'X')
    assert_allclose(
        numpy.nanmean(mean, dtype=numpy.float64), 93.368903, atol=1e-3
    )
    assert_allclose(
        numpy.nanmean(real, dtype=numpy.float64), 0.289324, atol=1e-3
    )
    assert_allclose(
        numpy.nanmean(imag, dtype=numpy.float64), 0.363937, atol=1e-3
    )

    for harmonic in ('all', [1, 2]):
        mean, real, imag, attrs = phasor_from_simfcs_referenced(
            filename, harmonic=harmonic
        )
        assert mean.shape == (1200, 1200)
        assert real.shape == (2, 1200, 1200)
        assert imag.shape == (2, 1200, 1200)
        assert_allclose(
            numpy.nanmean(mean, dtype=numpy.float64), 93.368903, atol=1e-3
        )
        assert_allclose(
            numpy.nanmean(real, axis=(1, 2), dtype=numpy.float64),
            [0.289324, 0.119052],
            atol=1e-3,
        )
        assert_allclose(
            numpy.nanmean(imag, axis=(1, 2), dtype=numpy.float64),
            [0.363937, 0.293647],
            atol=1e-3,
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


def test_phasor_to_simfcs_referenced_nanpad():
    """Test phasor_to_simfcs_referenced with NaN padding."""
    data = numpy.random.random_sample((2, 95, 97))
    with TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, 'nanpad.r64')
        phasor_to_simfcs_referenced(
            filename, data[0], data, data[::-1], size=80
        )
        filename = os.path.join(tempdir, 'nanpad_0_80_80.r64')
        mean, real, imag, attrs = phasor_from_simfcs_referenced(
            filename, harmonic='all'
        )
        assert_allclose(
            mean,
            numpy.pad(
                data[0, 80:, 80:],
                [(0, 65), (0, 63)],
                constant_values=numpy.nan,
            ),
            atol=1e-3,
            equal_nan=True,
        )
        assert_allclose(
            real[1],
            numpy.pad(
                data[1, 80:, 80:],
                [(0, 65), (0, 63)],
                constant_values=numpy.nan,
            ),
            atol=1e-3,
            equal_nan=True,
        )


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
