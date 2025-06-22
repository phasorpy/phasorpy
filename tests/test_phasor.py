"""Tests for the phasorpy.phasor module."""

import copy
import math

import numpy
import numpy.fft as numpy_fft
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
)

try:
    import scipy.fft as scipy_fft
except ImportError:
    scipy_fft = None

try:
    import mkl_fft._numpy_fft as mkl_fft
except ImportError:
    mkl_fft = None

from phasorpy.phasor import (
    lifetime_fraction_from_amplitude,
    lifetime_fraction_to_amplitude,
    lifetime_from_frequency,
    lifetime_to_frequency,
    lifetime_to_signal,
    phasor_at_harmonic,
    phasor_calibrate,
    phasor_center,
    phasor_divide,
    phasor_filter_median,
    phasor_filter_pawflim,
    phasor_from_apparent_lifetime,
    phasor_from_fret_acceptor,
    phasor_from_fret_donor,
    phasor_from_lifetime,
    phasor_from_polar,
    phasor_from_signal,
    phasor_multiply,
    phasor_nearest_neighbor,
    phasor_normalize,
    phasor_semicircle,
    phasor_semicircle_intersect,
    phasor_threshold,
    phasor_to_apparent_lifetime,
    phasor_to_complex,
    phasor_to_normal_lifetime,
    phasor_to_polar,
    phasor_to_principal_plane,
    phasor_to_signal,
    phasor_transform,
    polar_from_apparent_lifetime,
    polar_from_reference,
    polar_from_reference_phasor,
    polar_to_apparent_lifetime,
)

SYNTH_DATA_ARRAY = numpy.array([[50, 1], [1, 1]])
SYNTH_DATA_ARRAY_3D = numpy.stack(
    (
        SYNTH_DATA_ARRAY,
        SYNTH_DATA_ARRAY / 2,
        SYNTH_DATA_ARRAY / 3,
    ),
    axis=0,
)
SYNTH_DATA_NAN = numpy.array([[50, numpy.nan], [1, 1]])
SYNTH_DATA_LIST = [1, 2, 4]
SYNTH_PHI = numpy.array([[0.5, 0.5], [0.5, 0.5]])
SYNTH_MOD = numpy.array([[2, 2], [2, 2]])

numpy.random.seed(42)


@pytest.mark.parametrize('use_fft', (True, False))
def test_phasor_from_signal(use_fft):
    """Test phasor_from_signal function."""
    samples = 7
    sample_phase = numpy.linspace(0, 2 * math.pi, samples, endpoint=False)
    signal = 1.1 * (numpy.cos(sample_phase - 0.46364761) * 2 * 0.44721359 + 1)
    signal_copy = signal.copy()

    # scalar type
    mean, real, imag = phasor_from_signal(signal, use_fft=use_fft)
    assert mean.ndim == 0
    assert real.ndim == 0
    assert imag.ndim == 0

    # default is first harmonic
    assert_allclose(
        phasor_from_signal(signal, use_fft=use_fft), (1.1, 0.4, 0.2), atol=1e-6
    )
    assert_array_equal(signal, signal_copy)

    # specify first harmonic
    assert_allclose(
        phasor_from_signal(signal, harmonic=1, use_fft=use_fft),
        (1.1, 0.4, 0.2),
        atol=1e-6,
    )
    assert_array_equal(signal, signal_copy)

    # keep harmonic axis
    dc, re, im = phasor_from_signal(signal, harmonic=[1], use_fft=use_fft)
    assert_array_equal(signal, signal_copy)
    assert_allclose(dc, 1.1, atol=1e-6)
    assert_allclose(re, [0.4], atol=1e-6)
    assert_allclose(im, [0.2], atol=1e-6)

    # specific harmonic
    assert_allclose(
        phasor_from_signal(signal, harmonic=2, use_fft=use_fft),
        (1.1, 0.0, 0.0),
        atol=1e-6,
    )

    # list harmonics
    dc, re, im = phasor_from_signal(signal, harmonic=[1, 2], use_fft=use_fft)
    assert_array_equal(signal, signal_copy)
    assert_allclose(dc, 1.1, atol=1e-6)
    assert_allclose(re, [0.4, 0.0], atol=1e-6)
    assert_allclose(im, [0.2, 0.0], atol=1e-6)

    # all harmonics
    dc, re, im = phasor_from_signal(signal, harmonic='all', use_fft=use_fft)
    assert_array_equal(signal, signal_copy)
    assert_allclose(dc, 1.1, atol=1e-6)
    assert_allclose(re, (0.4, 0.0, 0.0), atol=1e-6)
    assert_allclose(im, (0.2, 0.0, 0.0), atol=1e-6)

    # zero signal
    assert_allclose(
        phasor_from_signal(numpy.zeros(256), use_fft=use_fft),
        (0.0, numpy.nan, numpy.nan),
        atol=1e-6,
    )

    # no modulation
    assert_allclose(
        phasor_from_signal(numpy.ones(256), use_fft=use_fft),
        (1.0, 0.0, 0.0),
        atol=1e-6,
    )

    # numerically unstable?
    # assert_allclose(
    #     phasor_from_signal(numpy.cos(sample_phase), use_fft=use_fft),
    #     [0.0, 0.0, 0.0],
    #     atol=1e-6,
    # )

    if not use_fft:
        assert_allclose(
            phasor_from_signal(
                signal[::-1],
                sample_phase=sample_phase[::-1],
                num_threads=0,
                use_fft=use_fft,
            ),
            (1.1, 0.4, 0.2),
            atol=1e-6,
        )
        assert_allclose(
            phasor_from_signal(
                signal[::-1],
                sample_phase=sample_phase[::-1],
                harmonic=1,
                use_fft=use_fft,
            ),
            (1.1, 0.4, 0.2),
            atol=1e-6,
        )

    with pytest.raises(ValueError):
        phasor_from_signal(signal[:2], use_fft=use_fft)
    with pytest.raises(TypeError):
        phasor_from_signal(signal, harmonic=1.0, use_fft=use_fft)
    with pytest.raises(ValueError):
        phasor_from_signal(signal, harmonic=[], use_fft=use_fft)
    with pytest.raises(TypeError):
        phasor_from_signal(
            signal, harmonic=[1.0], use_fft=use_fft  # type: ignore[list-item]
        )
    with pytest.raises(IndexError):
        phasor_from_signal(signal, harmonic=0, use_fft=use_fft)
    with pytest.raises(ValueError):
        phasor_from_signal(signal, harmonic='none', use_fft=use_fft)
    with pytest.raises(IndexError):
        phasor_from_signal(signal, harmonic=[0], use_fft=use_fft)
    with pytest.raises(IndexError):
        phasor_from_signal(signal, harmonic=[4], use_fft=use_fft)
    with pytest.raises(IndexError):
        phasor_from_signal(signal, harmonic=4, use_fft=use_fft)
    with pytest.raises(ValueError):
        phasor_from_signal(signal, harmonic=[1, 1], use_fft=use_fft)
    with pytest.raises(TypeError):
        phasor_from_signal(signal.astype('complex64'), use_fft=use_fft)
    if not use_fft:
        with pytest.raises(ValueError):
            phasor_from_signal(
                signal, sample_phase=sample_phase, harmonic=2, use_fft=use_fft
            )
        with pytest.raises(ValueError):
            phasor_from_signal(
                signal, sample_phase=sample_phase[::-2], use_fft=use_fft
            )
        with pytest.raises(TypeError):
            phasor_from_signal(signal, dtype='int8', use_fft=use_fft)
    else:
        with pytest.raises(ValueError):
            phasor_from_signal(
                signal, sample_phase=sample_phase, use_fft=use_fft
            )


@pytest.mark.parametrize('use_fft', (True, False))
@pytest.mark.parametrize('samples', [2, 3])
def test_phasor_from_signal_min_samples(samples, use_fft):
    """Test phasor_from_signal function with two and three samples."""
    sample_phase = numpy.linspace(0, 2 * math.pi, samples, endpoint=False)
    signal = 1.1 * (numpy.cos(sample_phase - 0.46364761) * 2 * 0.44721359 + 1)

    # a minimum of three samples is required to calculate correct 1st harmonic
    if samples < 3:
        with pytest.raises(ValueError):
            phasor = phasor_from_signal(signal, use_fft=use_fft)
    else:
        phasor = phasor_from_signal(signal, use_fft=use_fft)
        assert_allclose(phasor, (1.1, 0.4, 0.2), atol=1e-3)


@pytest.mark.parametrize('use_fft', (True, False))
@pytest.mark.parametrize(
    'shape, axis, dtype, dtype_out',
    [
        ((3,), 0, 'float64', 'float64'),
        ((1, 3), 1, 'float64', 'float64'),
        ((1, 3, 1), 1, 'float64', 'float64'),
        ((5, 2), 0, 'float64', 'float64'),
        ((2, 5), 1, 'float64', 'float64'),
        ((5, 2, 2), 0, 'float64', 'float64'),
        ((2, 5, 2), 1, 'float64', 'float64'),
        ((2, 2, 5), 2, 'float64', 'float64'),
        ((2, 2, 5, 2), 2, 'float64', 'float64'),
        ((2, 5, 2), 1, 'float32', 'float32'),
        ((2, 5, 2), 1, 'int16', 'float32'),
        ((2, 5, 2), 1, 'int32', 'float32'),
        ((64, 128, 128, 2, 32), 4, 'float32', 'float32'),  # 256 MB
        ((32, 32, 256, 256), 1, 'float32', 'float32'),  # 256 MB
        # TODO: can't test uint with this
    ],
)
def test_phasor_from_signal_param(use_fft, shape, axis, dtype, dtype_out):
    """Test phasor_from_signal function parameters."""
    samples = shape[axis]
    dtype = numpy.dtype(dtype)
    signal = numpy.empty(shape, dtype)
    sample_phase = numpy.linspace(0, 2 * math.pi, samples, endpoint=False)
    if not use_fft:
        sample_phase[0] = sample_phase[-1]  # out of order
        sample_phase[-1] = 0.0
    sig = 2.1 * (numpy.cos(sample_phase - 0.46364761) * 2 * 0.44721359 + 1)
    if dtype.kind != 'f':
        sig *= 1000
    sig = sig.astype(dtype)
    reshape = [1] * len(shape)
    reshape[axis] = samples
    signal[:] = sig.reshape(reshape)
    if use_fft:
        mean, real, imag = phasor_from_signal(
            signal, axis=axis, use_fft=True, dtype=dtype_out
        )
    else:
        num_threads = 4 if signal.size > 4096 else 1
        mean, real, imag = phasor_from_signal(
            signal,
            axis=axis,
            sample_phase=sample_phase,
            dtype=dtype_out,
            num_threads=num_threads,
            use_fft=False,
        )
    if isinstance(mean, numpy.ndarray):
        assert mean.dtype == dtype_out
        assert mean.shape == shape[:axis] + shape[axis + 1 :]
    if dtype.kind == 'f':
        assert_allclose(numpy.mean(mean), 2.1, 1e-3)
    else:
        assert_allclose(numpy.mean(mean), 2100, 1)
    assert_allclose(numpy.mean(real), 0.4, 1e-3)
    assert_allclose(numpy.mean(imag), 0.2, 1e-3)


@pytest.mark.parametrize('use_fft', (True, False))
@pytest.mark.parametrize('dtype', ('float32', 'float64'))
def test_phasor_from_signal_noncontig(use_fft, dtype):
    """Test phasor_from_signal functions with non-contiguous input."""
    dtype = numpy.dtype(dtype)
    samples = 31
    signal = numpy.empty((7, 19, samples, 11), dtype)
    sample_phase = numpy.linspace(0, 2 * math.pi, samples, endpoint=False)
    sig = 2.1 * (numpy.cos(sample_phase - 0.46364761) * 2 * 0.44721359 + 1)
    sig = sig.astype(dtype)
    reshape = [1] * 4
    reshape[2] = samples
    signal[:] = sig.reshape(reshape)
    signal = numpy.moveaxis(signal, 1, 2)
    assert signal.shape == (7, samples, 19, 11)
    assert not signal.flags['C_CONTIGUOUS']
    signal_copy = signal.copy()
    mean, real, imag = phasor_from_signal(
        signal, axis=-3, dtype=dtype, use_fft=use_fft
    )
    assert_array_equal(signal, signal_copy)
    assert real.dtype == dtype
    assert mean.dtype == dtype
    assert mean.shape == signal.shape[:1] + signal.shape[1 + 1 :]
    assert_allclose(numpy.mean(mean), 2.1, 1e-3)
    assert_allclose(numpy.mean(real), 0.4, 1e-3)
    assert_allclose(numpy.mean(imag), 0.2, 1e-3)


@pytest.mark.parametrize('scalar', (True, False))
@pytest.mark.parametrize('harmonic', ('all', 1, 2, 8, [1], [1, 2, 8]))
def test_phasor_from_signal_harmonic(scalar, harmonic):
    """Test phasor_from_signal function harmonic parameter."""
    rng = numpy.random.default_rng(1)
    signal = rng.random((33,) if scalar else (3, 33, 61, 63))
    signal += 1.1
    kwargs = dict(axis=0 if scalar else 1, harmonic=harmonic)
    mean0, real0, imag0 = phasor_from_signal(signal, use_fft=False, **kwargs)
    mean1, real1, imag1 = phasor_from_signal(signal, use_fft=True, **kwargs)
    assert_allclose(mean0, mean1, 1e-8)
    assert_allclose(real0, real1, 1e-8)
    assert_allclose(imag0, imag1, 1e-8)


@pytest.mark.parametrize('fft', (numpy_fft, scipy_fft, mkl_fft))
@pytest.mark.parametrize('scalar', (True, False))
@pytest.mark.parametrize('harmonic', (1, [4], [1, 4]))
def test_phasor_from_signal_fft_func(fft, scalar, harmonic):
    """Test phasor_from_signal_fft function rfft parameter."""
    if fft is None:
        pytest.skip('rfft function could not be imported')
    rng = numpy.random.default_rng(1)
    signal = rng.random((33,) if scalar else (3, 33, 61, 63))
    signal += 1.1
    kwargs = dict(axis=0 if scalar else 1, harmonic=harmonic)
    mean0, real0, imag0 = phasor_from_signal(signal, use_fft=True, **kwargs)
    mean1, real1, imag1 = phasor_from_signal(
        signal, rfft=fft.rfft, use_fft=True, **kwargs
    )
    assert_allclose(mean0, mean1, 1e-8)
    assert_allclose(real0, real1, 1e-8)
    assert_allclose(imag0, imag1, 1e-8)


@pytest.mark.parametrize('harmonics', (True, False))
@pytest.mark.parametrize('normalize', (True, False))
def test_phasor_normalization(harmonics, normalize):
    """Test phasor_from_signal and phasor_normalize functions."""
    rng = numpy.random.default_rng(1)
    samples = 33
    signal = rng.random((3, samples, 61, 63))
    signal += 1.1
    if normalize:
        # TODO: NaN handling seems to differ with FFT in unnormalized case
        signal[0, 0, 0, 0] = numpy.nan
        signal[0, :, 1, 1] = 0.0

    harmonic = [1, 2] if harmonics else None
    mean0, real0, imag0 = phasor_from_signal(
        signal, axis=1, harmonic=harmonic, normalize=normalize
    )
    mean1, real1, imag1 = phasor_from_signal(
        signal, axis=1, harmonic=harmonic, normalize=normalize, use_fft=True
    )
    assert_allclose(mean0, mean1, 1e-8)
    assert_allclose(real0, real1, 1e-8)
    assert_allclose(imag0, imag1, 1e-8)

    if not normalize:
        mean1, real1, imag1 = phasor_from_signal(
            signal, axis=1, harmonic=harmonic
        )
        assert_allclose(mean0, mean1 * samples, 1e-8)
        assert_allclose(real0, real1 * mean0, 1e-8)
        assert_allclose(imag0, imag1 * mean0, 1e-8)

        mean_ = mean0.copy()
        real_ = real0.copy()
        imag_ = imag0.copy()

        mean2, real2, imag2 = phasor_normalize(
            mean0, real0, imag0, samples=samples, dtype=numpy.float64
        )
        assert_array_equal(mean0, mean_)
        assert_array_equal(real0, real_)
        assert_array_equal(imag0, imag_)
        assert_allclose(mean2, mean1, 1e-8)
        assert_allclose(real2, real1, 1e-8)
        assert_allclose(imag2, imag1, 1e-8)

        mean2, real2, imag2 = phasor_normalize(
            mean0, real0.astype(numpy.float32), imag0, samples=samples
        )
        assert mean2.dtype == numpy.float32
        assert real2.dtype == numpy.float32
        assert imag2.dtype == numpy.float32
        assert_array_equal(mean0, mean_)
        assert_array_equal(real0, real_)
        assert_array_equal(imag0, imag_)
        assert_allclose(mean2, mean1, 1e-4)
        assert_allclose(real2, real1, 1e-4)
        assert_allclose(imag2, imag1, 1e-4)

        mean2, real2, imag2 = phasor_normalize(
            mean0, real0, imag0, samples=1, dtype=numpy.float64
        )
        assert_array_equal(mean0, mean_)
        assert_array_equal(real0, real_)
        assert_array_equal(imag0, imag_)
        assert_array_equal(mean2, mean0)
        assert_allclose(real2, real1, 1e-8)
        assert_allclose(imag2, imag1, 1e-8)

        with pytest.raises(ValueError):
            phasor_normalize(mean0, real0, imag0, samples=0)


@pytest.mark.parametrize(
    'shape, axis',
    [
        ((15,), 0),
        ((15,), -1),
        ((15, 1), 0),
        ((1, 15), -1),
        ((15, 15), 0),
        ((15, 15), -1),
        ((16, 3, 4), 0),
        ((3, 4, 15), -1),
    ],
)
def test_phasor_to_signal_roundtrip(shape, axis):
    """Test phasor_to_signal and phasor_from_signal functions in roundtrip."""
    samples = shape[axis]
    harmonic = list(range(1, samples // 2 + 1))
    signal0 = numpy.random.normal(1.1, 0.1, shape)

    # get all harmonics
    mean, real, imag = phasor_from_signal(signal0, harmonic='all', axis=axis)
    assert_allclose(numpy.mean(mean), 1.1, atol=0.1)

    # synthesize all harmonics
    signal1 = phasor_to_signal(
        mean, real, imag, samples=samples, harmonic='all', axis=axis
    )
    assert_allclose(signal1, signal0, atol=1e-4)

    if signal0.size > 15:
        # synthesize all harmonics found in first axes
        signal1 = phasor_to_signal(
            mean, real, imag, samples=samples, axis=axis
        )
        assert_allclose(signal1, signal0, atol=1e-4)

    # synthesize specified all harmonics
    signal1 = phasor_to_signal(
        mean, real, imag, samples=samples, harmonic=harmonic, axis=axis
    )
    assert_allclose(signal1, signal0, atol=1e-4)

    # synthesize first harmonic only
    signal1 = phasor_to_signal(
        mean, real[0], imag[0], samples=samples, axis=axis
    )
    mean1, real1, imag1 = phasor_from_signal(signal0, axis=axis)
    assert_allclose(mean1, mean, atol=1e-3)
    assert_allclose(real1, real[0], atol=1e-3)
    assert_allclose(imag1, imag[0], atol=1e-3)

    # synthesize first harmonic, keep harmonic axis
    signal1 = phasor_to_signal(
        mean, real[:1], imag[:1], samples=samples, harmonic=[1], axis=axis
    )
    mean1, real1, imag1 = phasor_from_signal(signal0, harmonic=[1], axis=axis)
    assert_allclose(mean1, mean, atol=1e-3)
    assert_allclose(real1, real[:1], atol=1e-3)
    assert_allclose(imag1, imag[:1], atol=1e-3)

    # synthesize second harmonic
    signal1 = phasor_to_signal(
        mean, real[1], imag[1], samples=samples, harmonic=2, axis=axis
    )
    mean1, real1, imag1 = phasor_from_signal(signal0, harmonic=2, axis=axis)
    assert_allclose(mean1, mean, atol=1e-3)
    assert_allclose(real1, real[1], atol=1e-3)
    assert_allclose(imag1, imag[1], atol=1e-3)

    # synthesize second harmonic, keep harmonic axis
    signal1 = phasor_to_signal(
        mean, real[1:2], imag[1:2], samples=samples, harmonic=[2], axis=axis
    )
    mean1, real1, imag1 = phasor_from_signal(signal0, harmonic=[2], axis=axis)
    assert_allclose(mean1, mean, atol=1e-3)
    assert_allclose(real1, real[1:2], atol=1e-3)
    assert_allclose(imag1, imag[1:2], atol=1e-3)

    # synthesize first two harmonics
    signal1 = phasor_to_signal(
        mean, real[:2], imag[:2], samples=samples, harmonic=[1, 2], axis=axis
    )
    mean1, real1, imag1 = phasor_from_signal(
        signal0, harmonic=[1, 2], axis=axis
    )
    assert_allclose(mean1, mean, atol=1e-3)
    assert_allclose(real1, real[:2], atol=1e-3)
    assert_allclose(imag1, imag[:2], atol=1e-3)


def test_phasor_to_signal():
    """Test phasor_to_signal function."""
    sample_phase = numpy.linspace(0, 2 * math.pi, 5, endpoint=False)
    signal = 1.1 * (numpy.cos(sample_phase - 0.78539816) * 2 * 0.70710678 + 1)
    assert_allclose(phasor_from_signal(signal, use_fft=True), (1.1, 0.5, 0.5))

    assert_allclose(
        phasor_to_signal(1.1, 0.5, 0.5, samples=5), signal, atol=1e-4
    )
    assert_allclose(
        phasor_to_signal(1.1, 0.5, 0.5, samples=5, harmonic=1),
        signal,
        atol=1e-4,
    )
    assert_allclose(
        phasor_to_signal(1.1, [0.5], [0.5], samples=5), [signal], atol=1e-4
    )
    assert_allclose(
        phasor_to_signal(1.1, [0.5], [0.5], samples=5, harmonic=[1]),
        signal,
        atol=1e-4,
    )
    assert_allclose(
        phasor_to_signal(1.1, [[0.5]], [[0.5]], samples=5, harmonic=[1]),
        [signal],
        atol=1e-4,
    )
    assert_allclose(
        phasor_to_signal(1.1, [[0.5]], [[0.5]], samples=5),
        [[signal]],
        atol=1e-4,
    )
    assert_allclose(
        phasor_to_signal([1.1], [[0.5]], [[0.5]], samples=5),
        [[signal]],
        atol=1e-4,
    )
    assert_allclose(
        phasor_to_signal(
            1.1, [[0.5, 0.5]], [[0.5, 0.5]], harmonic=1, samples=5
        ),
        [[signal, signal]],
        atol=1e-4,
    )
    assert_allclose(
        phasor_to_signal(
            1.1, [[0.5, 0.5]], [[0.5, 0.5]], harmonic=[1], samples=5
        ),
        [signal, signal],
        atol=1e-4,
    )

    frequency, lifetime, fraction = [20, 40, 60], [0.9, 4.2], [0.2, 0.8]

    phasor = phasor_from_lifetime(frequency, lifetime)
    assert_allclose(
        phasor_to_signal([1.1, 1.2], *phasor, samples=6, harmonic='all'),
        [
            [6.351575, 0.775947, -0.243349, 0.034261, 0.1511, -0.469533],
            [4.554442, 3.124505, -0.143694, 0.115879, 0.21576, -0.666891],
        ],
        atol=1e-4,
    )

    phasor = phasor_from_lifetime(frequency, lifetime, fraction, keepdims=True)
    assert_allclose(
        phasor_to_signal(1.1, *phasor, samples=6, harmonic='all'),
        [[4.610239, 2.446493, -0.154045, 0.09183, 0.188444, -0.58296]],
        atol=1e-4,
    )

    phasor = phasor_from_lifetime(frequency, lifetime, fraction)
    assert_allclose(
        phasor_to_signal(1.1, *phasor, samples=6, harmonic='all'),
        [4.610239, 2.446493, -0.154045, 0.09183, 0.188444, -0.58296],
        atol=1e-4,
    )


def test_phasor_to_signal_error():
    """Test phasor_to_signal functions exceptions."""
    with pytest.raises(ValueError):
        # not floating point
        phasor_to_signal(1.0, 1, 1.0, samples=5)
    with pytest.raises(ValueError):
        # phasor shape mismatch
        phasor_to_signal(1.1, 0.5, [[0.5]], samples=5)
    with pytest.raises(ValueError):
        # mean/phasor shape mismatch
        phasor_to_signal([1.1, 1.1], 0.5, 0.5, samples=5)
    with pytest.raises(ValueError):
        # harmonic not unique
        phasor_to_signal(1.1, 0.5, 0.5, harmonic=[1, 1], samples=5)
    with pytest.raises(ValueError):
        # len(harmonic) != real.shape[0]
        phasor_to_signal(1.1, [0.5, 0.5], [0.5, 0.5], harmonic=[1], samples=5)
    with pytest.raises(ValueError):
        # samples < 3
        phasor_to_signal(1.1, [0.5, 0.5], [0.5, 0.5], samples=2)
    with pytest.raises(IndexError):
        # harmonic < 1
        phasor_to_signal(1.1, [0.5, 0.5], [0.5, 0.5], harmonic=0)
    with pytest.raises(ValueError):
        # harmonic str != 'all'
        phasor_to_signal(1.1, [0.5, 0.5], [0.5, 0.5], harmonic='none')


@pytest.mark.parametrize(
    'harmonic, expected, zero_expected',
    [
        (
            # time domain
            'all',
            [0.355548, 0.245101, 0.748013, 0.515772],
            [0.0, 0.479172, 0.246015, 0.0],
        ),
        (
            # frequency-domain
            1,
            [0.204701, -0.056023, 1.031253, 0.586501],
            [-0.095829, 0.35908, 0.307823, -0.07783],
        ),
    ],
)
def test_lifetime_to_signal(harmonic, expected, zero_expected):
    """Test lifetime_to_signal function."""
    index = [0, 1, -2, -1]
    # single lifetime
    signal, zero, time = lifetime_to_signal(
        40.0, 4.2, samples=16, harmonic=harmonic
    )
    assert signal.shape == (16,)
    assert zero.shape == (16,)
    assert time.shape == (16,)
    assert_allclose(signal[index], expected, atol=1e-3)
    assert_allclose(zero[[0, 9, 10, -1]], zero_expected, atol=1e-3)
    assert_allclose(time[index], [0.0, 1.666667, 23.333333, 25.0], atol=1e-3)

    # two lifetimes
    signal, zero, time = lifetime_to_signal(
        40.0, [4.2, 4.2], samples=16, harmonic=harmonic
    )
    assert signal.shape == (2, 16)
    assert zero.shape == (16,)
    assert time.shape == (16,)
    assert_allclose(signal[1, index], expected, atol=1e-3)

    # one multi-components
    signal, zero, time = lifetime_to_signal(
        40.0, [4.2, 4.2], [0.5, 0.5], samples=16, harmonic=harmonic
    )
    assert signal.shape == (16,)
    assert zero.shape == (16,)
    assert time.shape == (16,)
    assert_allclose(signal[index], expected, atol=1e-3)

    # two multi-components
    signal, zero, time = lifetime_to_signal(
        40.0,
        [[4.2, 4.2], [4.2, 4.2]],
        [[0.5, 0.5], [0.5, 0.5]],
        samples=16,
        harmonic=harmonic,
    )
    assert signal.shape == (2, 16)
    assert zero.shape == (16,)
    assert time.shape == (16,)
    assert_allclose(signal[1, index], expected, atol=1e-3)


def test_lifetime_to_signal_parameters():
    """Test lifetime_to_signal function parameters."""
    # TODO: test mean, background, zero_phase, zero_stdev parameters


def test_lifetime_to_signal_error():
    """Test lifetime_to_signal function exceptions."""
    lifetime_to_signal(40.0, 4.2)
    with pytest.raises(ValueError):
        lifetime_to_signal(0.0, 4.2)
    with pytest.raises(ValueError):
        lifetime_to_signal([40.0, 80.0], 4.2)
    with pytest.raises(ValueError):
        lifetime_to_signal(40.0, 4.2, samples=15)
    with pytest.raises(ValueError):
        lifetime_to_signal(40.0, 4.2, mean=-0.1)
    with pytest.raises(ValueError):
        lifetime_to_signal(40.0, 4.2, mean=1.0, background=1.1)
    with pytest.raises(ValueError):
        lifetime_to_signal(40.0, 4.2, zero_phase=-1.0)
    with pytest.raises(ValueError):
        lifetime_to_signal(40.0, 4.2, zero_phase=7.0)
    with pytest.raises(ValueError):
        lifetime_to_signal(40.0, 4.2, samples=100, zero_stdev=math.pi / 100)
    with pytest.raises(ValueError):
        lifetime_to_signal(40.0, 4.2, samples=100, zero_stdev=math.pi / 5)
    with pytest.raises(IndexError):
        lifetime_to_signal(40.0, 4.2, harmonic=0)


def test_phasor_semicircle():
    """Test phasor_semicircle function."""
    real, imag = phasor_semicircle(1)
    assert_allclose(real, 0.0, atol=1e-6)
    assert_allclose(imag, 0.0, atol=1e-6)
    real, imag = phasor_semicircle(2)
    assert_allclose(real, [0, 1], atol=1e-6)
    assert_allclose(imag, [0.0, 0], atol=1e-6)
    real, imag = phasor_semicircle(3)
    assert_allclose(real, [0, 0.5, 1], atol=1e-6)
    assert_allclose(imag, [0.0, 0.5, 0], atol=1e-6)
    real, imag = phasor_semicircle()
    assert len(real) == 101
    with pytest.raises(ValueError):
        phasor_semicircle(0)


def test_phasor_semicircle_intersect():
    """Test phasor_semicircle_intersect function."""
    assert_allclose(
        phasor_semicircle_intersect(
            [0.2, 0.2, math.nan], [0.25, 0.0, 0.25], 0.6, 0.25
        ),
        (
            [0.066, numpy.nan, numpy.nan],
            [0.25, numpy.nan, numpy.nan],
            [0.933, 0.817, numpy.nan],
            [0.25, 0.386, numpy.nan],
        ),
        atol=1e-3,
    )
    # reverse order
    assert_allclose(
        phasor_semicircle_intersect(
            0.6, 0.25, [0.2, 0.2, math.nan], [0.25, 0.0, 0.25]
        ),
        (
            [0.933, numpy.nan, numpy.nan],
            [0.25, numpy.nan, numpy.nan],
            [0.066, 0.817, numpy.nan],
            [0.25, 0.386, numpy.nan],
        ),
        atol=1e-3,
    )
    # no intersection
    assert_allclose(
        phasor_semicircle_intersect(0.1, -0.1, 0.9, -0.1),
        (numpy.nan, numpy.nan, numpy.nan, numpy.nan),
        atol=1e-3,
    )
    # no line
    assert_allclose(
        phasor_semicircle_intersect(0.25, 0.25, 0.25, 0.25),
        (numpy.nan, numpy.nan, numpy.nan, numpy.nan),
        atol=1e-3,
    )
    # tangent
    assert_allclose(
        phasor_semicircle_intersect(0.4, 0.5, 0.6, 0.5),
        (0.5, 0.5, 0.5, 0.5),
        atol=1e-3,
    )
    # lifetime
    assert_allclose(
        phasor_semicircle_intersect(
            *phasor_from_lifetime(80, 4.2),  # single component
            *phasor_from_lifetime(80, [4.2, 1.1], [1, 1]),  # mixture
        ),
        numpy.array(
            phasor_from_lifetime(80, [4.2, 1.1])  # two single components
        ).T.flat,
        atol=1e-4,
    )


def test_phasor_from_polar():
    """Test phasor_from_polar function."""
    real, imag = phasor_from_polar(
        [0.0, math.pi / 4, math.pi / 2], [1.0, math.sqrt(0.5), 1.0]
    )
    assert_allclose(real, [1, 0.5, 0.0], atol=1e-6)
    assert_allclose(imag, [0, 0.5, 1], atol=1e-6)

    # roundtrip
    rng = numpy.random.default_rng()
    phase = rng.random((63, 65)).astype(numpy.float32) * (2.0 * math.pi)
    modulation = rng.random((63, 65)).astype(numpy.float32)
    phase_, modulation_ = phasor_from_polar(
        *phasor_to_polar(phase, modulation)
    )
    assert_allclose(phase, phase_, atol=1e-6)
    assert_allclose(modulation, modulation_, atol=1e-6)

    # scalars
    real, imag = phasor_from_polar(math.pi / 4, math.sqrt(0.5))
    assert isinstance(real, float)

    # TODO: keep float32 dtype
    # phase = numpy.array([0, 0.785398], dtype=numpy.float32)
    # modulation = numpy.array([0, 0.707107], dtype=numpy.float32)
    # real, imag = phasor_from_polar(phase, modulation)
    # assert_allclose(real, [0, 0.5], atol=1e-3)
    # assert_allclose(imag, [0, 0.5], atol=1e-3)
    # assert real.dtype == 'float32'
    # assert imag.dtype == 'float32'

    # broadcast
    assert_allclose(
        phasor_from_polar(0.785398, [0.707107, 1.0]),
        [[0.5, 0.707107], [0.5, 0.707107]],
        atol=1e-4,
    )

    # exceptions
    with pytest.raises(ValueError):
        phasor_from_polar(
            [0.0, math.pi / 4, math.pi / 2], [1.0, math.sqrt(0.5)]
        )


@pytest.mark.parametrize(
    'real, imag, expected_phase, expected_modulation',
    [
        (1, 0, 0.0, 1.0),
        (-0.5, -0.7, -2.191045812777718, 0.8602325267042626),
        (
            SYNTH_DATA_LIST,
            SYNTH_DATA_LIST,
            [0.78539816, 0.78539816, 0.78539816],
            [1.41421356, 2.82842712, 5.65685425],
        ),
        (
            SYNTH_DATA_ARRAY,
            SYNTH_DATA_ARRAY,
            numpy.asarray(
                [[0.78539816, 0.78539816], [0.78539816, 0.78539816]]
            ),
            numpy.asarray(
                [[70.71067812, 1.41421356], [1.41421356, 1.41421356]]
            ),
        ),
    ],
)
def test_phasor_to_polar(real, imag, expected_phase, expected_modulation):
    """Test phasor_to_polar function with various inputs."""
    real_copy = copy.deepcopy(real)
    imag_copy = copy.deepcopy(imag)
    phase, modulation = phasor_to_polar(real_copy, imag_copy)
    assert_array_equal(real, real_copy)
    assert_array_equal(imag, imag_copy)
    assert_almost_equal(phase, expected_phase)
    assert_almost_equal(modulation, expected_modulation)

    # roundtrip
    real2, imag2 = phasor_from_polar(phase, modulation)
    assert_allclose(real2, real, atol=1e-3)
    assert_allclose(imag2, imag, atol=1e-3)


def test_phasor_to_polar_more():
    """Test phasor_to_polar function."""
    # scalars
    phase, modulation = phasor_to_polar(0.5, 0.5)
    assert isinstance(phase, float)
    assert isinstance(modulation, float)

    # TODO: keep float32 dtype
    # real = numpy.array([0, 0.5], dtype=numpy.float32)
    # imag = numpy.array([0, 0.5], dtype=numpy.float32)
    # phase, modulation = phasor_to_polar(real, imag)
    # assert_allclose(phase, [0, 0.785398], atol=1e-3)
    # assert_allclose(modulation, [0, 0.707107], atol=1e-3)
    # assert phase.dtype == 'float32'
    # assert modulation.dtype == 'float32'

    # broadcast
    assert_allclose(
        phasor_to_polar([0.5], [0.1, 0.5]),
        [[0.197396, 0.785398], [0.509902, 0.707107]],
        atol=1e-4,
    )


@pytest.mark.parametrize(
    """measured_phase, measured_modulation,
    known_phase, known_modulation,
    expected_phase, expected_modulation""",
    [
        (2, 2, 0.2, 0.5, -1.8, 0.25),
        (-2, -2, 0.2, 0.5, 2.2, -0.25),
        (
            SYNTH_DATA_LIST,
            SYNTH_DATA_LIST,
            numpy.full(len(SYNTH_DATA_LIST), 0.2),
            numpy.full(len(SYNTH_DATA_LIST), 0.5),
            [-0.8, -1.8, -3.8],
            [0.5, 0.25, 0.125],
        ),
        (
            SYNTH_DATA_ARRAY,
            SYNTH_DATA_ARRAY,
            numpy.full(SYNTH_DATA_ARRAY.shape, 0.2),
            numpy.full(SYNTH_DATA_ARRAY.shape, 0.5),
            numpy.asarray([[-49.8, -0.8], [-0.8, -0.8]]),
            numpy.asarray([[1e-2, 0.5], [0.5, 0.5]]),
        ),
    ],
)
def test_polar_from_reference(
    measured_phase,
    measured_modulation,
    known_phase,
    known_modulation,
    expected_phase,
    expected_modulation,
):
    """Test polar_from_reference function with various inputs."""
    measured_phase_copy = copy.deepcopy(measured_phase)
    measured_modulation_copy = copy.deepcopy(measured_modulation)
    known_phase_copy = copy.deepcopy(known_phase)
    known_modulation_copy = copy.deepcopy(known_modulation)
    phase0, modulation0 = polar_from_reference(
        measured_phase_copy,
        measured_modulation_copy,
        known_phase_copy,
        known_modulation_copy,
    )
    assert_array_equal(measured_phase, measured_phase_copy)
    assert_array_equal(measured_modulation, measured_modulation_copy)
    assert_array_equal(known_phase, known_phase_copy)
    assert_array_equal(known_modulation, known_modulation_copy)
    assert_almost_equal(phase0, expected_phase)
    assert_almost_equal(modulation0, expected_modulation)


@pytest.mark.parametrize(
    """measured_real, measured_imag,
    known_real, known_imag,
    expected_phase, expected_modulation""",
    [
        (2, 2, 0.2, 0.5, 0.4048917862850834, 0.19039432764659772),
        (-2, -2, 0.2, 0.5, 3.5464844398748765, 0.19039432764659772),
        (
            SYNTH_DATA_LIST,
            SYNTH_DATA_LIST,
            numpy.full(len(SYNTH_DATA_LIST), 0.2),
            numpy.full(len(SYNTH_DATA_LIST), 0.5),
            [0.40489179, 0.40489179, 0.40489179],
            [0.38078866, 0.19039433, 0.09519716],
        ),
        (
            SYNTH_DATA_ARRAY,
            SYNTH_DATA_ARRAY,
            numpy.full(SYNTH_DATA_ARRAY.shape, 0.2),
            numpy.full(SYNTH_DATA_ARRAY.shape, 0.5),
            numpy.full(SYNTH_DATA_ARRAY.shape, 0.40489179),
            numpy.asarray(
                [[0.00761577, 0.38078866], [0.38078866, 0.38078866]]
            ),
        ),
    ],
)
def test_polar_from_reference_phasor(
    measured_real,
    measured_imag,
    known_real,
    known_imag,
    expected_phase,
    expected_modulation,
):
    """Test polar_from_reference_phasor function with various inputs."""
    measured_real_copy = copy.deepcopy(measured_real)
    measured_imag_copy = copy.deepcopy(measured_imag)
    known_real_copy = copy.deepcopy(known_real)
    known_imag_copy = copy.deepcopy(known_imag)
    phase0, modulation0 = polar_from_reference_phasor(
        measured_real_copy,
        measured_imag_copy,
        known_real_copy,
        known_imag_copy,
    )
    assert_array_equal(measured_real, measured_real_copy)
    assert_array_equal(measured_imag, measured_imag_copy)
    assert_array_equal(known_real, known_real_copy)
    assert_array_equal(known_imag, known_imag_copy)
    assert_almost_equal(phase0, expected_phase)
    assert_almost_equal(modulation0, expected_modulation)


def test_polar_from_reference_functions():
    """Test polar_from_reference and polar_from_reference_phasor match."""
    # https://github.com/phasorpy/phasorpy/issues/43
    measured_real = numpy.random.rand(5, 7)
    measured_imag = numpy.random.rand(5, 7)
    known_real = 0.5
    known_imag = 0.5
    phi0, mod0 = polar_from_reference_phasor(
        measured_real, measured_imag, known_real, known_imag
    )
    phi1, mod1 = polar_from_reference(
        *phasor_to_polar(measured_real, measured_imag),
        *phasor_to_polar(known_real, known_imag),
    )
    assert_allclose(phi0, phi1, atol=1e-3)
    assert_allclose(mod0, mod1, atol=1e-3)


def test_phasor_to_complex():
    """Test phasor_to_complex function."""
    real = [numpy.nan, 0.1, 0.2]
    imag = [numpy.nan, 0.3, 0.4]
    assert_allclose(phasor_to_complex(real, imag).real, real)
    assert_allclose(phasor_to_complex(real, imag).imag, imag)
    assert_allclose(phasor_to_complex(0, imag).real, 0)
    assert_allclose(phasor_to_complex(real, 0).imag, 0)

    assert (
        phasor_to_complex(real, imag, dtype=numpy.complex64).dtype
        == numpy.complex64
    )

    assert (
        phasor_to_complex(
            numpy.array(real, dtype=numpy.float32),
            numpy.array(imag, dtype=numpy.float32),
        ).dtype
        == numpy.complex64
    )

    assert (
        phasor_to_complex(
            numpy.array(real, dtype=numpy.float64),
            numpy.array(imag, dtype=numpy.float32),
        ).dtype
        == numpy.complex128
    )

    with pytest.raises(ValueError):
        phasor_to_complex(0.0, 0.0, dtype=numpy.float64)


def test_phasor_multiply():
    """Test phasor_multiply function."""
    real1 = [0.0, 0.1, 0.2]
    imag1 = [0.0, 0.3, 0.4]
    real2 = [0.0, 0.5, 0.6]
    imag2 = [0.0, 0.7, 0.8]
    real = [0.0, -0.16, -0.2]
    imag = [0.0, 0.22, 0.4]

    assert_allclose(
        phasor_to_complex(*phasor_multiply(real1, imag1, real2, imag2)),
        phasor_to_complex(real, imag),
    )
    assert_allclose(
        phasor_to_complex(real1, imag1) * phasor_to_complex(real2, imag2),
        phasor_to_complex(real, imag),
    )


def test_phasor_divide():
    """Test phasor_divide function."""
    real1 = [0.0, -0.16, -0.2]
    imag1 = [0.0, 0.22, 0.4]
    real2 = [0.0, 0.5, 0.6]
    imag2 = [0.0, 0.7, 0.8]
    real = [numpy.nan, 0.1, 0.2]
    imag = [numpy.nan, 0.3, 0.4]

    assert_allclose(
        phasor_to_complex(*phasor_divide(real1, imag1, real2, imag2)),
        phasor_to_complex(real, imag),
    )
    with pytest.warns(RuntimeWarning):
        assert_allclose(
            phasor_to_complex(real1, imag1) / phasor_to_complex(real2, imag2),
            phasor_to_complex(real, imag),
        )


@pytest.mark.parametrize(
    'real, imag, phase, modulation, expected_real, expected_imag',
    [
        (2, 2, None, None, 2, 2),
        (2, 2, 0, 1, 2, 2),
        (
            2,
            2,
            0.5,
            2.0,
            1.592628093144679,
            5.428032401978303,
        ),
        (
            2,
            2,
            -0.5,
            -2.0,
            -5.428032401978303,
            -1.592628093144679,
        ),
        (
            SYNTH_DATA_LIST,
            SYNTH_DATA_LIST,
            None,
            None,
            SYNTH_DATA_LIST,
            SYNTH_DATA_LIST,
        ),
        (
            SYNTH_DATA_LIST,
            SYNTH_DATA_LIST,
            0,
            1,
            SYNTH_DATA_LIST,
            SYNTH_DATA_LIST,
        ),
        (
            SYNTH_DATA_LIST,
            SYNTH_DATA_LIST,
            0.5,
            2.0,
            [0.79631405, 1.59262809, 3.18525619],
            [2.7140162, 5.4280324, 10.8560648],
        ),
        (
            SYNTH_DATA_ARRAY,
            SYNTH_DATA_ARRAY,
            None,
            None,
            SYNTH_DATA_ARRAY,
            SYNTH_DATA_ARRAY,
        ),
        (
            SYNTH_DATA_ARRAY,
            SYNTH_DATA_ARRAY,
            0,
            1,
            SYNTH_DATA_ARRAY,
            SYNTH_DATA_ARRAY,
        ),
        (
            SYNTH_DATA_ARRAY,
            SYNTH_DATA_ARRAY,
            0.5,
            2.0,
            numpy.array([[39.81570233, 0.79631405], [0.79631405, 0.79631405]]),
            numpy.array([[135.70081005, 2.7140162], [2.7140162, 2.7140162]]),
        ),
        (
            SYNTH_DATA_ARRAY,
            SYNTH_DATA_ARRAY,
            SYNTH_PHI,
            SYNTH_MOD,
            numpy.array([[39.81570233, 0.79631405], [0.79631405, 0.79631405]]),
            numpy.array([[135.70081005, 2.7140162], [2.7140162, 2.7140162]]),
        ),  # test with phase and modulation as arrays
    ],
)
def test_phasor_transform(
    real,
    imag,
    phase,
    modulation,
    expected_real,
    expected_imag,
):
    """Test phasor_transform function with various inputs."""
    real_copy = copy.deepcopy(real)
    imag_copy = copy.deepcopy(imag)
    if phase is not None and modulation is not None:
        calibrated_real, calibrated_imag = phasor_transform(
            real_copy, imag_copy, phase, modulation
        )
    else:
        calibrated_real, calibrated_imag = phasor_transform(
            real_copy, imag_copy
        )
    assert_array_equal(real, real_copy)
    assert_array_equal(imag, imag_copy)
    assert_almost_equal(calibrated_real, expected_real)
    assert_almost_equal(calibrated_imag, expected_imag)


def test_phasor_transform_more():
    """Test phasor_transform function."""
    # scalars
    real, imag = phasor_transform(1, 0, math.pi / 4, 0.5)
    assert isinstance(real, float)

    # TODO: keep float32 dtype
    # real = numpy.array([0, 1], dtype=numpy.float32)
    # imag = numpy.array([0, 0], dtype=numpy.float32)
    # real, imag = phasor_transform(real, imag, math.pi / 4, 0.5)
    # assert_allclose(real, [0, 0.353553], atol=1e-3)
    # assert_allclose(imag, [0, 0.353553], atol=1e-3)
    # assert real.dtype == 'float32'
    # assert imag.dtype == 'float32'

    # broadcast
    real, imag = phasor_transform(1, 0, [math.pi / 4, math.pi / 8], [0.5, 0.9])
    assert_allclose(real, [0.353553, 0.831492], atol=1e-3)
    assert_allclose(imag, [0.353553, 0.344415], atol=1e-3)


@pytest.mark.parametrize(
    'real, imag, kwargs, expected_real_center, expected_imag_center',
    [
        (1.0, 4.0, {'skip_axis': None, 'method': 'mean'}, 1.0, 4.0),
        (1.0, -4.0, {'skip_axis': None, 'method': 'median'}, 1.0, -4.0),
        (
            SYNTH_DATA_LIST,
            SYNTH_DATA_LIST,
            {'skip_axis': None, 'method': 'mean'},
            2.3333333333333335,
            2.3333333333333335,
        ),
        (
            SYNTH_DATA_LIST,
            SYNTH_DATA_LIST,
            {'skip_axis': None, 'method': 'median'},
            2.0,
            2.0,
        ),
        (
            SYNTH_DATA_ARRAY,
            SYNTH_DATA_ARRAY,
            {'skip_axis': None, 'method': 'mean'},
            13.25,
            13.25,
        ),
        (
            SYNTH_DATA_ARRAY,
            SYNTH_DATA_ARRAY,
            {'skip_axis': None, 'method': 'median'},
            1.0,
            1.0,
        ),
        # with skip_axis
        (
            SYNTH_DATA_ARRAY,
            SYNTH_DATA_ARRAY,
            {'skip_axis': 0, 'method': 'mean'},
            numpy.asarray([25.5, 1.0]),
            numpy.asarray([25.5, 1.0]),
        ),
        (
            SYNTH_DATA_ARRAY,
            SYNTH_DATA_ARRAY,
            {'skip_axis': (-2,), 'method': 'median'},
            numpy.asarray([25.5, 1.0]),
            numpy.asarray([25.5, 1.0]),
        ),
        (
            SYNTH_DATA_ARRAY,
            SYNTH_DATA_ARRAY,
            {'keepdims': True},
            [[13.25]],
            [[13.25]],
        ),  # with kwargs for numpy functions
        (
            SYNTH_DATA_NAN,
            SYNTH_DATA_NAN,
            {'skip_axis': None, 'method': 'median'},
            1.0,
            1.0,
        ),
        (
            SYNTH_DATA_NAN,
            SYNTH_DATA_NAN,
            {'skip_axis': None, 'method': 'mean'},
            52 / 3,
            52 / 3,
        ),
    ],
)
def test_phasor_center(
    real,
    imag,
    kwargs,
    expected_real_center,
    expected_imag_center,
):
    """Test phasor_center function with various inputs and methods."""
    real_copy = copy.deepcopy(real)
    imag_copy = copy.deepcopy(imag)
    mean = numpy.full_like(real_copy, 0.666, dtype=numpy.float64)
    mean_center, real_center, imag_center = phasor_center(
        mean, real_copy, imag_copy, **kwargs
    )
    assert_array_equal(mean, 0.666)
    assert_array_equal(real, real_copy)
    assert_array_equal(imag, imag_copy)
    assert_almost_equal(real_center, expected_real_center)
    assert_almost_equal(imag_center, expected_imag_center)


def test_phasor_center_mean():
    """Test phasor_center function mean normalization."""
    assert_allclose(
        phasor_center(
            [3.3, 2.2, 1.1], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], method='mean'
        ),
        [2.2, 0.166667, 0.466667],  # not [2.2, 0.2, 0.5]
        atol=1e-5,
    )


def test_phasor_center_exceptions():
    """Test exceptions in phasor_center function."""
    with pytest.raises(ValueError):
        phasor_center(0, 0, 0, method='method_not_supported')
    with pytest.raises(ValueError):
        phasor_center([0], [0, 0], [0, 0])
    with pytest.raises(ValueError):
        phasor_center([0, 0], [0, 0], [0])
    with pytest.raises(IndexError):
        phasor_center([0, 0], [0, 0], [0, 0], skip_axis=1)


@pytest.mark.parametrize(
    'args, kwargs, expected',
    [
        # single lifetime
        ((80.0, 0.0), {}, (1.0, 0.0)),  # shortest lifetime
        ((80.0, 1e9), {}, (0.0, 0.0)),  # very long lifetime
        ((80.0, 3.9788735), {}, (0.2, 0.4)),
        ((80.0, 0.9947183), {}, (0.8, 0.4)),
        ((80.0, 1.9894368), {}, (0.5, 0.5)),
        ((80.0, 1.9894368, 1.0), {}, (0.5, 0.5)),
        ((80.0, 1.9894368, 0.1), {}, (0.5, 0.5)),
        ((80.0, [1.9894368]), {}, (0.5, 0.5)),
        ((80.0, [1.9894368], [1.0]), {}, (0.5, 0.5)),
        ((80.0, [1.9894368], [0.6]), {}, (0.5, 0.5)),
        # two lifetime components
        ((80.0, [0.0, 1e9], [0.5, 0.5]), {}, (0.5, 0.0)),
        ((80.0, [0.0, 1e9], [0.6, 0.4]), {}, (0.6, 0.0)),
        ((80.0, [3.9788735, 0.9947183], [0.0, 1.0]), {}, (0.8, 0.4)),
        ((80.0, [3.9788735, 0.9947183], [1.0, 0.0]), {}, (0.2, 0.4)),
        ((80.0, [3.9788735, 0.9947183], [0.5, 0.5]), {}, (0.5, 0.4)),
        ((80.0, [3.9788735, 0.9947183], [0.25, 0.75]), {}, (0.65, 0.4)),
        # three single lifetimes, fraction is None
        (
            (80.0, [0.0, 1.9894368, 1e9]),
            {},
            ([1.0, 0.5, 0.0], [0.0, 0.5, 0.0]),
        ),
        (
            ([80.0, 80.0], [0.0, 1.9894368, 1e9]),
            {},
            (
                [[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]],
                [[0.0, 0.5, 0.0], [0.0, 0.5, 0.0]],
            ),
        ),
        # three lifetime components
        ((80.0, [0.0, 1.9894368, 1e9], [1, 1, 1]), {}, (0.5, 0.5 / 3)),
        ((80.0, [0.0, 1.9894368, 1e9], [0, 1, 0]), {}, (0.5, 0.5)),
        ((80.0, [0.0, 1.9894368, 1e9], [1, 1, 0]), {}, (0.75, 0.5 / 2)),
        ((80.0, [0.0, 1.9894368, 1e9], [0, 1, 1]), {}, (0.25, 0.5 / 2)),
        ((80.0, [0.0, 1.9894368, 1e9], [0, 0, 0]), {}, (numpy.nan, numpy.nan)),
        (
            (80.0, [3.9788735, 1.9894368, 0.9947183], [1.0, 1.0, 1.0]),
            {},
            (0.5, (0.4 + 0.5 + 0.4) / 3),
        ),
        # multiple frequencies
        (
            ([40.0, 80.0, 160.0], 1.9894368),  # single lifetime
            {},
            ([0.8, 0.5, 0.2], [0.4, 0.5, 0.4]),
        ),
        (
            ([40.0, 80.0], [3.9788735, 0.9947183]),  # two single lifetimes
            {},
            ([[0.5, 0.94117648], [0.2, 0.8]], [[0.5, 0.2352941], [0.4, 0.4]]),
        ),
        (
            ([40.0, 80.0], [3.9788735, 0.9947183], [0.5, 0.5]),  # 2 components
            {},
            ([0.72058825, 0.5], [0.36764705, 0.4]),
        ),
        # preexponential amplitudes
        ((80.0, 0.0), {'preexponential': True}, (1.0, 0.0)),
        ((80.0, 1e9), {'preexponential': True}, (0.0, 0.0)),
        ((80.0, 3.9788735), {'preexponential': True}, (0.2, 0.4)),
        ((80.0, [0.0, 1e9], [0.5, 0.5]), {'preexponential': True}, (0.0, 0.0)),
        (
            (80.0, [3.9788735, 0.9947183], [0.0, 1.0]),
            {'preexponential': True},
            (0.8, 0.4),
        ),
        (
            (80.0, [3.9788735, 0.9947183], [1.0, 0.0]),
            {'preexponential': True},
            (0.2, 0.4),
        ),
        (
            (80.0, [3.9788735, 0.9947183], [0.5, 0.5]),
            {'preexponential': True},
            (0.32, 0.4),
        ),
        (
            (80.0, [3.9788735, 0.9947183], [0.25, 0.75]),
            {'preexponential': True},
            (0.457143, 0.4),
        ),
        # variable lifetime, constant fraction
        (
            (
                80.0,
                [[3.9788735, 1.9894368], [1.9894368, 0.9947183]],
                [0.25, 0.75],
            ),
            {},
            ([0.425, 0.725], [0.475, 0.425]),
        ),
        # constant lifetime, variable fraction
        (
            (
                80.0,
                [3.9788735, 0.9947183],
                [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]],
            ),
            {},
            ([0.8, 0.5, 0.2], [0.4, 0.4, 0.4]),
        ),
    ],
)
def test_phasor_from_lifetime(args, kwargs, expected):
    """Test phasor_from_lifetime function."""
    result = phasor_from_lifetime(*args, **kwargs, keepdims=True)
    for actual, desired in zip(result, expected):
        assert actual.ndim == 2
        assert_allclose(actual.squeeze(), desired, atol=1e-6)


def test_phasor_from_lifetime_exceptions():
    """Test exceptions in phasor_from_lifetime function."""
    with pytest.raises(ValueError):
        phasor_from_lifetime(80.0, 0.0, unit_conversion=0.0)
    with pytest.raises(ValueError):
        phasor_from_lifetime(80.0, [[0.0]])
    with pytest.raises(ValueError):
        phasor_from_lifetime(80.0, 0.0, [])
    with pytest.raises(ValueError):
        phasor_from_lifetime(80.0, [0.0, 1e9], [1.0])
    with pytest.raises(ValueError):
        phasor_from_lifetime([[80.0]], 1.9894368)  # frequency is not 1D
    with pytest.raises(ValueError):
        phasor_from_lifetime(80.0, [[[0.0]]])  # lifetime is > 2D
    with pytest.raises(ValueError):
        phasor_from_lifetime(80.0, 0.0, [[[1.0]]])  # fraction is > 2D
    with pytest.raises(ValueError):
        phasor_from_lifetime(80.0, [[0.0, 1e9]], [[1.0]])  # shape mismatch
    with pytest.raises(ValueError):
        phasor_from_lifetime(80.0, [[[0.0]]], [[[1.0]]])  # matching but > 2D


def test_phasor_from_lifetime_modify():
    """Test phasor_from_lifetime function does not modify input."""
    frequency = 80.0
    lifetime = numpy.array([0.0, 1.9894368, 1e9], dtype=numpy.float64)
    fraction = numpy.array([1.0, 1.0, 1.0], dtype=numpy.float64)
    real, imag = phasor_from_lifetime(
        lifetime=lifetime, fraction=fraction, frequency=frequency
    )
    assert_allclose(real, 0.5)
    assert_allclose(imag, 0.5 / 3)
    assert_array_equal(frequency, 80.0)  # for future revisions
    assert_array_equal(lifetime, [0.0, 1.9894368, 1e9])
    assert_array_equal(fraction, [1.0, 1.0, 1.0])


@pytest.mark.parametrize(
    'args, kwargs, expected',
    [
        # scalar data
        # single lifetime
        (
            (2, 2, 2, 2),
            {'frequency': 80, 'lifetime': 1.9894368},
            (numpy.array(0.5), numpy.array(0.5)),
        ),
        (
            (0.5, 0.7, 0.4, 0.3),
            {'frequency': 80, 'lifetime': 4},
            (numpy.array(0.11789139), numpy.array(0.75703471)),
        ),
        (
            (0.5, 0.7, 0.4, 0.3),
            {'frequency': 40, 'lifetime': 4, 'harmonic': 2},
            (numpy.array(0.11789139), numpy.array(0.75703471)),
        ),
        (
            (-0.5, -0.7, -0.4, -0.3),
            {'frequency': 80, 'lifetime': 4},
            (numpy.array(0.11789139), numpy.array(0.75703471)),
        ),
        # two lifetimes with fractions
        (
            (2, 2, 2, 2),
            {
                'frequency': 80,
                'lifetime': [3.9788735, 0.9947183],
                'fraction': [0.25, 0.75],
            },
            (numpy.array(0.65), numpy.array(0.4)),
        ),
        (
            (0.5, 0.7, 0.4, 0.3),
            {
                'frequency': 80,
                'lifetime': [3.9788735, 0.9947183],
                'fraction': [0.25, 0.75],
            },
            (numpy.array(0.85800005), numpy.array(0.99399999)),
        ),
        (
            (-0.5, -0.7, -0.4, -0.3),
            {
                'frequency': 80,
                'lifetime': [3.9788735, 0.9947183],
                'fraction': [0.25, 0.75],
            },
            (numpy.array(0.85800005), numpy.array(0.99399999)),
        ),
        # list data
        # single lifetime
        (
            (
                SYNTH_DATA_LIST,
                SYNTH_DATA_LIST,
                SYNTH_DATA_LIST,
                SYNTH_DATA_LIST,
            ),
            {'frequency': 80, 'lifetime': 4},
            (
                numpy.array([0.08499034, 0.16998068, 0.33996135]),
                numpy.array([0.17088322, 0.34176643, 0.68353286]),
            ),
        ),
        # multiple lifetime
        (
            (
                SYNTH_DATA_LIST,
                SYNTH_DATA_LIST,
                SYNTH_DATA_LIST,
                SYNTH_DATA_LIST,
            ),
            {
                'frequency': 80,
                'lifetime': [3.9788735, 0.9947183],
                'fraction': [0.25, 0.75],
            },
            (
                numpy.array([0.27857144, 0.55714288, 1.11428576]),
                numpy.array([0.17142856, 0.34285713, 0.68571426]),
            ),
        ),
        # multiple lifetime with median method
        (
            (
                SYNTH_DATA_LIST,
                SYNTH_DATA_LIST,
                SYNTH_DATA_LIST,
                SYNTH_DATA_LIST,
            ),
            {
                'frequency': 80,
                'lifetime': [3.9788735, 0.9947183],
                'fraction': [0.25, 0.75],
                'method': 'median',
            },
            (
                numpy.array([0.32500001, 0.65000002, 1.30000005]),
                numpy.array([0.19999999, 0.39999998, 0.79999997]),
            ),
        ),
        # array data
        # single lifetime
        (
            (
                SYNTH_DATA_ARRAY,
                SYNTH_DATA_ARRAY,
                SYNTH_DATA_ARRAY,
                SYNTH_DATA_ARRAY,
            ),
            {'frequency': 80, 'lifetime': 4},
            (
                numpy.array(
                    [[0.7483426, 0.01496685], [0.01496685, 0.01496685]]
                ),
                numpy.array(
                    [[1.50463208, 0.03009264], [0.03009264, 0.03009264]]
                ),
            ),
        ),
        # multiple lifetime
        (
            (
                SYNTH_DATA_ARRAY,
                SYNTH_DATA_ARRAY,
                SYNTH_DATA_ARRAY,
                SYNTH_DATA_ARRAY,
            ),
            {
                'frequency': 80,
                'lifetime': [3.9788735, 0.9947183],
                'fraction': [0.25, 0.75],
            },
            (
                numpy.array(
                    [[2.45283028, 0.04905661], [0.04905661, 0.04905661]]
                ),
                numpy.array(
                    [[1.5094339, 0.03018868], [0.03018868, 0.03018868]]
                ),
            ),
        ),
        # multiple lifetime with median method
        (
            (
                SYNTH_DATA_ARRAY,
                SYNTH_DATA_ARRAY,
                SYNTH_DATA_ARRAY,
                SYNTH_DATA_ARRAY,
            ),
            {
                'frequency': 80,
                'lifetime': [3.9788735, 0.9947183],
                'fraction': [0.25, 0.75],
                'method': 'median',
            },
            (
                numpy.array(
                    [[32.50000122, 0.65000002], [0.65000002, 0.65000002]]
                ),
                numpy.array(
                    [[19.9999992, 0.39999998], [0.39999998, 0.39999998]]
                ),
            ),
        ),
        # multiple frequencies with skip_axis
        (
            (
                numpy.repeat(
                    numpy.expand_dims(SYNTH_DATA_ARRAY_3D, axis=1),
                    repeats=4,
                    axis=1,
                ),
                numpy.repeat(
                    numpy.expand_dims(SYNTH_DATA_ARRAY_3D, axis=1),
                    repeats=4,
                    axis=1,
                ),
                [
                    [0.5, 0.5, 0.5, 0.5],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.125, 0.125, 0.125, 0.125],
                ],
                [
                    [0.5, 0.5, 0.5, 0.5],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.125, 0.125, 0.125, 0.125],
                ],
            ),
            {
                'frequency': [80, 160, 240],
                'lifetime': 4,
                'skip_axis': 0,
            },
            (
                numpy.array(
                    [
                        [
                            [
                                [19.83107902, 0.39662158],
                                [0.39662158, 0.39662158],
                            ],
                            [
                                [19.83107902, 0.39662158],
                                [0.39662158, 0.39662158],
                            ],
                            [
                                [19.83107902, 0.39662158],
                                [0.39662158, 0.39662158],
                            ],
                            [
                                [19.83107902, 0.39662158],
                                [0.39662158, 0.39662158],
                            ],
                        ],
                        [
                            [[5.82398976, 0.1164798], [0.1164798, 0.1164798]],
                            [[5.82398976, 0.1164798], [0.1164798, 0.1164798]],
                            [[5.82398976, 0.1164798], [0.1164798, 0.1164798]],
                            [[5.82398976, 0.1164798], [0.1164798, 0.1164798]],
                        ],
                        [
                            [
                                [3.56665406, 0.07133308],
                                [0.07133308, 0.07133308],
                            ],
                            [
                                [3.56665406, 0.07133308],
                                [0.07133308, 0.07133308],
                            ],
                            [
                                [3.56665406, 0.07133308],
                                [0.07133308, 0.07133308],
                            ],
                            [
                                [3.56665406, 0.07133308],
                                [0.07133308, 0.07133308],
                            ],
                        ],
                    ]
                ),
                numpy.array(
                    [
                        [
                            [[39.87275018, 0.797455], [0.797455, 0.797455]],
                            [[39.87275018, 0.797455], [0.797455, 0.797455]],
                            [[39.87275018, 0.797455], [0.797455, 0.797455]],
                            [[39.87275018, 0.797455], [0.797455, 0.797455]],
                        ],
                        [
                            [
                                [23.41965242, 0.46839305],
                                [0.46839305, 0.46839305],
                            ],
                            [
                                [23.41965242, 0.46839305],
                                [0.46839305, 0.46839305],
                            ],
                            [
                                [23.41965242, 0.46839305],
                                [0.46839305, 0.46839305],
                            ],
                            [
                                [23.41965242, 0.46839305],
                                [0.46839305, 0.46839305],
                            ],
                        ],
                        [
                            [
                                [21.51355047, 0.43027101],
                                [0.43027101, 0.43027101],
                            ],
                            [
                                [21.51355047, 0.43027101],
                                [0.43027101, 0.43027101],
                            ],
                            [
                                [21.51355047, 0.43027101],
                                [0.43027101, 0.43027101],
                            ],
                            [
                                [21.51355047, 0.43027101],
                                [0.43027101, 0.43027101],
                            ],
                        ],
                    ]
                ),
            ),
        ),
        # multiple harmonics with skip_axis
        (
            (
                numpy.repeat(
                    numpy.expand_dims(SYNTH_DATA_ARRAY_3D, axis=1),
                    repeats=4,
                    axis=1,
                ),
                numpy.repeat(
                    numpy.expand_dims(SYNTH_DATA_ARRAY_3D, axis=1),
                    repeats=4,
                    axis=1,
                ),
                [
                    [0.5, 0.5, 0.5, 0.5],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.125, 0.125, 0.125, 0.125],
                ],
                [
                    [0.5, 0.5, 0.5, 0.5],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.125, 0.125, 0.125, 0.125],
                ],
            ),
            {
                'frequency': 80,
                'harmonic': [1, 2, 3],
                'lifetime': 4,
                'skip_axis': 0,
            },
            (
                numpy.array(
                    [
                        [
                            [
                                [19.83107902, 0.39662158],
                                [0.39662158, 0.39662158],
                            ],
                            [
                                [19.83107902, 0.39662158],
                                [0.39662158, 0.39662158],
                            ],
                            [
                                [19.83107902, 0.39662158],
                                [0.39662158, 0.39662158],
                            ],
                            [
                                [19.83107902, 0.39662158],
                                [0.39662158, 0.39662158],
                            ],
                        ],
                        [
                            [[5.82398976, 0.1164798], [0.1164798, 0.1164798]],
                            [[5.82398976, 0.1164798], [0.1164798, 0.1164798]],
                            [[5.82398976, 0.1164798], [0.1164798, 0.1164798]],
                            [[5.82398976, 0.1164798], [0.1164798, 0.1164798]],
                        ],
                        [
                            [
                                [3.56665406, 0.07133308],
                                [0.07133308, 0.07133308],
                            ],
                            [
                                [3.56665406, 0.07133308],
                                [0.07133308, 0.07133308],
                            ],
                            [
                                [3.56665406, 0.07133308],
                                [0.07133308, 0.07133308],
                            ],
                            [
                                [3.56665406, 0.07133308],
                                [0.07133308, 0.07133308],
                            ],
                        ],
                    ]
                ),
                numpy.array(
                    [
                        [
                            [[39.87275018, 0.797455], [0.797455, 0.797455]],
                            [[39.87275018, 0.797455], [0.797455, 0.797455]],
                            [[39.87275018, 0.797455], [0.797455, 0.797455]],
                            [[39.87275018, 0.797455], [0.797455, 0.797455]],
                        ],
                        [
                            [
                                [23.41965242, 0.46839305],
                                [0.46839305, 0.46839305],
                            ],
                            [
                                [23.41965242, 0.46839305],
                                [0.46839305, 0.46839305],
                            ],
                            [
                                [23.41965242, 0.46839305],
                                [0.46839305, 0.46839305],
                            ],
                            [
                                [23.41965242, 0.46839305],
                                [0.46839305, 0.46839305],
                            ],
                        ],
                        [
                            [
                                [21.51355047, 0.43027101],
                                [0.43027101, 0.43027101],
                            ],
                            [
                                [21.51355047, 0.43027101],
                                [0.43027101, 0.43027101],
                            ],
                            [
                                [21.51355047, 0.43027101],
                                [0.43027101, 0.43027101],
                            ],
                            [
                                [21.51355047, 0.43027101],
                                [0.43027101, 0.43027101],
                            ],
                        ],
                    ]
                ),
            ),
        ),
        # multiple harmonics without skip_axis
        (
            (
                SYNTH_DATA_ARRAY_3D,
                SYNTH_DATA_ARRAY_3D,
                [0.5, 0.25, 0.1],
                [0.3, 0.2, 0.1],
            ),
            {
                'frequency': 80,
                'harmonic': [1, 2, 3],
                'lifetime': 4,
            },
            (
                numpy.array(
                    [
                        [[11.60340173, 0.23206803], [0.23206803, 0.23206803]],
                        [[3.53612871, 0.07072257], [0.07072257, 0.07072257]],
                        [[4.45831758, 0.08916635], [0.08916635, 0.08916635]],
                    ]
                ),
                numpy.array(
                    [
                        [[52.74178815, 1.05483576], [1.05483576, 1.05483576]],
                        [[26.41473921, 0.52829478], [0.52829478, 0.52829478]],
                        [[26.89193809, 0.53783876], [0.53783876, 0.53783876]],
                    ]
                ),
            ),
        ),
        # harmonics higher than size of first dimension in `reference_real`
        (
            (
                SYNTH_DATA_LIST,
                SYNTH_DATA_LIST,
                SYNTH_DATA_LIST,
                SYNTH_DATA_LIST,
            ),
            {
                'frequency': 80,
                'harmonic': [1, 2, 6],
                'lifetime': 4,
            },
            (
                [0.19831079, 0.0582399, 0.00682439],
                [0.3987275, 0.23419652, 0.0823275],
            ),
        ),
        # harmonics = 'all' parameter
        (
            (
                SYNTH_DATA_ARRAY_3D,
                SYNTH_DATA_ARRAY_3D,
                [0.5, 0.25, 0.1],
                [0.3, 0.2, 0.1],
            ),
            {
                'frequency': 80,
                'harmonic': 'all',
                'lifetime': 4,
            },
            (
                numpy.array(
                    [
                        [[11.60340173, 0.23206803], [0.23206803, 0.23206803]],
                        [[3.53612871, 0.07072257], [0.07072257, 0.07072257]],
                        [[4.45831758, 0.08916635], [0.08916635, 0.08916635]],
                    ]
                ),
                numpy.array(
                    [
                        [[52.74178815, 1.05483576], [1.05483576, 1.05483576]],
                        [[26.41473921, 0.52829478], [0.52829478, 0.52829478]],
                        [[26.89193809, 0.53783876], [0.53783876, 0.53783876]],
                    ]
                ),
            ),
        ),
    ],
)
def test_phasor_calibrate(args, kwargs, expected):
    """Test phasor_calibrate function with various inputs."""
    real, imag, real_ref, imag_ref = args
    mean_ref = numpy.full_like(real_ref, 2)
    if ('harmonic' in kwargs and not isinstance(kwargs['harmonic'], int)) or (
        isinstance(kwargs.get('frequency', None), (list, numpy.ndarray))
        and len(kwargs['frequency']) > 1
    ):
        mean_ref = mean_ref[0]
    result = phasor_calibrate(
        real, imag, mean_ref, real_ref, imag_ref, **kwargs
    )
    assert_almost_equal(result, expected)
    result = phasor_calibrate(
        *result, mean_ref, real_ref, imag_ref, reverse=True, **kwargs
    )
    assert_almost_equal(result, args[:2])


def test_phasor_calibrate_exceptions():
    """Test exceptions in phasor_calibrate function."""
    kwargs = {'frequency': 1, 'lifetime': 1}
    phasor_calibrate(0, 0, 0, 0, 0, **kwargs)
    with pytest.raises(ValueError):
        phasor_calibrate(0, 0, 0, [0], [0, 0], **kwargs)
    with pytest.raises(ValueError):
        phasor_calibrate([0], [0, 0], 0, 0, 0, **kwargs)
    with pytest.raises(ValueError):
        phasor_calibrate(0, 0, 0, [0], [0, 0], **kwargs)
    with pytest.raises(ValueError):
        phasor_calibrate(0, [0], 0, 0, 0, **kwargs)
    with pytest.raises(ValueError):
        phasor_calibrate(0, 0, 0, 0, 0, **kwargs, harmonic=[1, 2])
    with pytest.raises((ValueError, IndexError)):
        phasor_calibrate(0, 0, 0, [0], [0], **kwargs, harmonic=[1, 2])
    with pytest.raises(ValueError):
        phasor_calibrate(
            [0], [0], 0, [0, 0], [0, 0], **kwargs, harmonic=[1, 2]
        )
    with pytest.raises(ValueError):
        phasor_calibrate(0, 0, 0, [0, 0], [0, 0], **kwargs)
    with pytest.raises(ValueError):
        phasor_calibrate(0, 0, 0, [[0]], [[0]], **kwargs)
    with pytest.raises(ValueError):
        phasor_calibrate([0], [0], [0], [[0], [0]], [[0], [0]], **kwargs)
    with pytest.raises(ValueError):
        phasor_calibrate(
            [[0], [0]],
            [[0], [0]],
            [0, 0],
            [[0], [0]],
            [[0], [0]],
            **kwargs,
            harmonic=[1, 2],
        )
    stack_array = numpy.stack([SYNTH_DATA_ARRAY] * 3, axis=0)
    with pytest.raises(ValueError):
        phasor_calibrate(
            stack_array,
            stack_array,
            stack_array,
            stack_array,
            stack_array,
            frequency=[80, 160, 240, 320],
            lifetime=4,
            skip_axis=0,
        )
    with pytest.raises(ValueError):
        phasor_calibrate(
            stack_array,
            stack_array,
            stack_array,
            stack_array,
            stack_array,
            frequency=80,
            harmonic=[1, 2, 3, 4],
            lifetime=4,
        )


def test_phasor_to_normal_lifetime():
    """Test phasor_to_normal_lifetime function."""
    taunorm = phasor_to_normal_lifetime(
        [0.5, 0.5, 0, 1, -1.1], [0.5, 0.45, 0, 0, 1.1], frequency=80
    )
    assert_allclose(
        taunorm, [1.989437, 1.989437, math.inf, 0.0, 6.405351], atol=1e-3
    )

    # verify against phasor_to_apparent_lifetime
    real, imag = phasor_semicircle(11)
    expected = phasor_to_apparent_lifetime(real, imag, frequency=80)[0]
    assert numpy.isinf(expected[0])
    assert expected[-1] == 0.0
    assert_allclose(
        phasor_to_normal_lifetime(real, imag, frequency=80),
        expected,
        atol=1e-3,
    )

    phase, modulation = phasor_to_polar(real - 0.5, imag)
    real, imag = phasor_from_polar(phase, modulation * 0.6)
    assert_allclose(
        phasor_to_normal_lifetime(real + 0.5, imag, frequency=80),
        expected,
        atol=1e-3,
    )

    # broadcast, mix dtypes
    taunorm = phasor_to_normal_lifetime(
        0.5,
        numpy.array([0.5, 0.45], dtype=numpy.float32),
        frequency=numpy.array([[20], [40], [80]], dtype=numpy.int32),
        #  dtype=numpy.float32
    )
    assert taunorm.shape == (3, 2)
    assert taunorm.dtype == 'float64'
    assert_allclose(
        taunorm,
        [[7.957747, 7.957747], [3.978874, 3.978874], [1.989437, 1.989437]],
        atol=1e-3,
    )


def test_phasor_to_apparent_lifetime():
    """Test phasor_to_apparent_lifetime function."""
    tauphi, taumod = phasor_to_apparent_lifetime(
        [0.5, 0.5, 0, 1, -1.1], [0.5, 0.45, 0, 0, 1.1], frequency=80
    )
    assert_allclose(
        tauphi, [1.989437, 1.790493, math.inf, 0.0, -1.989437], atol=1e-3
    )
    assert_allclose(
        taumod, [1.989437, 2.188331, math.inf, 0.0, 0.0], atol=1e-3
    )

    # broadcast, mix dtypes
    tauphi, taumod = phasor_to_apparent_lifetime(
        0.5,
        numpy.array([0.5, 0.45], dtype=numpy.float32),
        frequency=numpy.array([[20], [40], [80]], dtype=numpy.int32),
        #  dtype=numpy.float32
    )
    assert tauphi.shape == (3, 2)
    assert tauphi.dtype == 'float64'
    assert_allclose(
        tauphi,
        [[7.957747, 7.161972], [3.978874, 3.580986], [1.989437, 1.790493]],
        atol=1e-3,
    )
    assert_allclose(
        taumod,
        [[7.957747, 8.753322], [3.978874, 4.376661], [1.989437, 2.188331]],
        atol=1e-3,
    )


def test_phasor_from_apparent_lifetime():
    """Test phasor_from_apparent_lifetime function."""
    real, imag = phasor_from_apparent_lifetime(
        [1.989437, 1.790493, 1e9, 0.0],
        [1.989437, 2.188331, 1e9, 0.0],
        frequency=80,
    )
    assert_allclose(real, [0.5, 0.5, 0.0, 1.0], atol=1e-3)
    assert_allclose(imag, [0.5, 0.45, 0.0, 0.0], atol=1e-3)

    # roundtrip
    tauphi, taumod = phasor_to_apparent_lifetime(real, imag, frequency=80)
    assert_allclose(tauphi, [1.989437, 1.790493, 1e9, 0.0], atol=1e-3)
    assert_allclose(taumod, [1.989437, 2.188331, 1e9, 0.0], atol=1e-3)

    # modulation_lifetime = None
    real, imag = phasor_from_apparent_lifetime(
        [1.989437, 1e9, 0.0],
        None,
        frequency=80,
    )
    assert_allclose(real, [0.5, 0.0, 1.0], atol=1e-3)
    assert_allclose(imag, [0.5, 0.0, 0.0], atol=1e-3)

    # verify against phasor_from_lifetime
    real, imag = phasor_from_apparent_lifetime(
        [1.989437, 1e9, 0.0],
        None,
        frequency=[[40], [80]],
    )
    real2, imag2 = phasor_from_lifetime([40, 80], [1.989437, 1e9, 0.0])
    assert_allclose(real, real2, atol=1e-3)
    assert_allclose(imag, imag2, atol=1e-3)


def test_polar_to_apparent_lifetime():
    """Test test_polar_to_apparent_lifetime function."""
    tauphi, taumod = polar_to_apparent_lifetime(
        *phasor_to_polar([0.5, 0.5, 0, 1, -1.1], [0.5, 0.45, 0, 0, 1.1]),
        frequency=80,
    )
    assert_allclose(
        tauphi, [1.989437, 1.790493, math.inf, 0.0, -1.989437], atol=1e-3
    )
    assert_allclose(
        taumod, [1.989437, 2.188331, math.inf, 0.0, 0.0], atol=1e-3
    )

    # broadcast, mix dtypes
    tauphi, taumod = polar_to_apparent_lifetime(
        [0.78539816, 0.7328151],
        numpy.array([0.70710678, 0.6726812], dtype=numpy.float32),
        frequency=numpy.array([[20], [40], [80]], dtype=numpy.int32),
        #  dtype=numpy.float32
    )
    assert tauphi.shape == (3, 2)
    assert tauphi.dtype == 'float64'
    assert_allclose(
        tauphi,
        [[7.957747, 7.161972], [3.978874, 3.580986], [1.989437, 1.790493]],
        atol=1e-3,
    )
    assert_allclose(
        taumod,
        [[7.957747, 8.753322], [3.978874, 4.376661], [1.989437, 2.188331]],
        atol=1e-3,
    )


def test_polar_from_apparent_lifetime():
    """Test polar_from_apparent_lifetime function."""
    phase, modulation = polar_from_apparent_lifetime(
        [1.989437, 1.790493, 1e9, 0.0],
        [1.989437, 2.188331, 1e9, 0.0],
        frequency=80,
    )
    real, imag = phasor_from_polar(phase, modulation)
    assert_allclose(real, [0.5, 0.5, 0.0, 1.0], atol=1e-3)
    assert_allclose(imag, [0.5, 0.45, 0.0, 0.0], atol=1e-3)

    # roundtrip
    tauphi, taumod = polar_to_apparent_lifetime(
        phase, modulation, frequency=80
    )
    assert_allclose(tauphi, [1.989437, 1.790493, 1e9, 0.0], atol=1e-3)
    assert_allclose(taumod, [1.989437, 2.188331, 1e9, 0.0], atol=1e-3)

    # verify against phasor_from_apparent_lifetime
    real, imag = phasor_from_polar(
        *polar_from_apparent_lifetime(
            [1.989437, 1.790493, 1e9, 0.0],
            [1.989437, 2.188331, 1e9, 0.0],
            frequency=80,
        )
    )
    real2, imag2 = phasor_from_apparent_lifetime(
        [1.989437, 1.790493, 1e9, 0.0],
        [1.989437, 2.188331, 1e9, 0.0],
        frequency=80,
    )
    assert_allclose(real, real2, atol=1e-3)
    assert_allclose(imag, imag2, atol=1e-3)

    # modulation_lifetime is None
    phase, modulation = polar_from_apparent_lifetime(
        [1.989437, 1.790493, 1e9, 0.0], None, frequency=80
    )
    real, imag = phasor_from_polar(phase, modulation)
    assert_allclose(real, [0.5, 0.55248, 0.0, 1.0], atol=1e-3)
    assert_allclose(imag, [0.5, 0.49723, 0.0, 0.0], atol=1e-3)


def test_phasor_from_fret_donor():
    """Test phasor_from_fret_donor function."""
    re, im = phasor_from_lifetime(80, 4.2)
    # no FRET
    assert_allclose(
        phasor_from_fret_donor(80, 4.2, fret_efficiency=0),
        [re, im],
        atol=1e-3,
    )
    # fret_efficiency
    assert_allclose(
        phasor_from_fret_donor(80, 4.2, fret_efficiency=[0.0, 0.3, 1.0]),
        phasor_from_lifetime(80, [4.2, 4.2 * 0.7, 0.0]),
        atol=1e-3,
    )
    # frequency
    assert_allclose(
        phasor_from_fret_donor([40, 80], 4.2, fret_efficiency=0.3),
        phasor_from_lifetime([40, 80], 4.2 * 0.7),
        atol=1e-3,
    )
    # donor_fretting
    assert_allclose(
        phasor_from_fret_donor(
            80, 4.2, fret_efficiency=[0.0, 0.3, 1.0], donor_fretting=0.9
        ),
        [[re, 0.296158, re], [im, 0.453563, im]],
        atol=1e-3,
    )
    # background
    assert_allclose(
        phasor_from_fret_donor(
            80,
            4.2,
            fret_efficiency=[0.0, 0.3, 1.0],
            donor_background=0.1,
            background_real=0.11,
            background_imag=0.12,
        ),
        [[0.176593, 0.288569, 0.11], [0.362612, 0.42113, 0.12]],
        atol=1e-3,
    )
    # complex
    assert_allclose(
        phasor_from_fret_donor(
            80,
            4.2,
            fret_efficiency=[0.0, 0.3, 1.0],
            donor_fretting=0.9,
            donor_background=0.1,
            background_real=0.11,
            background_imag=0.12,
        ),
        [[0.176593, 0.273729, 0.146626], [0.362612, 0.413374, 0.253437]],
        atol=1e-3,
    )


def test_phasor_from_fret_acceptor():
    """Test phasor_from_fret_acceptor function."""
    re, im = phasor_from_lifetime(80, 3.0)
    # no FRET
    assert_allclose(
        phasor_from_fret_acceptor(80, 4.2, 3.0, fret_efficiency=1),
        [re, im],
        atol=1e-3,
    )
    # fret_efficiency
    assert_allclose(
        phasor_from_fret_acceptor(
            80, 4.2, 3.0, fret_efficiency=[0.0, 0.3, 1.0]
        ),
        [[-0.122219, -0.117851, re], [0.202572, 0.286433, im]],
        atol=1e-3,
    )
    # frequency
    assert_allclose(
        phasor_from_fret_acceptor([40, 80], 4.2, 3.0, fret_efficiency=0.3),
        [[0.182643, -0.117851], [0.615661, 0.286433]],
        atol=1e-3,
    )
    # acceptor_bleedthrough
    assert_allclose(
        phasor_from_fret_acceptor(
            80,
            4.2,
            3.0,
            fret_efficiency=[0.0, 0.3, 1.0],
            acceptor_bleedthrough=0.1,
        ),
        [[re, -0.012028, re], [im, 0.329973, im]],
        atol=1e-3,
    )
    # donor_bleedthrough
    dre, dim = phasor_from_lifetime(80, 4.2)
    assert_allclose(
        phasor_from_fret_acceptor(
            80,
            4.2,
            3.0,
            fret_efficiency=[0.0, 0.3, 1.0],
            donor_bleedthrough=0.1,
        ),
        [[dre, -0.036135, re], [dim, 0.320055, im]],
        atol=1e-3,
    )
    # donor_fretting
    assert_allclose(
        phasor_from_fret_acceptor(
            80,
            4.2,
            3.0,
            fret_efficiency=[0.0, 0.3, 1.0],
            donor_bleedthrough=0.1,
            donor_fretting=0.9,
        ),
        [[dre, -0.02974, 0.3041], [dim, 0.322, 0.4598]],
        atol=1e-3,
    )
    # background
    assert_allclose(
        phasor_from_fret_acceptor(
            80,
            4.2,
            3.0,
            fret_efficiency=[0.0, 0.3, 1.0],
            acceptor_background=0.1,
            background_real=0.11,
            background_imag=0.12,
        ),
        [[0.11, -0.060888, 0.287673], [0.12, 0.244825, 0.429631]],
        atol=1e-3,
    )
    # complex
    assert_allclose(
        phasor_from_fret_acceptor(
            80,
            4.2,
            3.0,
            fret_efficiency=[0.0, 0.3, 1.0],
            donor_fretting=0.9,
            donor_bleedthrough=0.1,
            acceptor_bleedthrough=0.1,
            acceptor_background=0.1,
            background_real=0.11,
            background_imag=0.12,
        ),
        [[0.199564, 0.057723, 0.286733], [0.322489, 0.310325, 0.429246]],
        atol=1e-3,
    )


def test_lifetime_to_frequency():
    """Test lifetime_to_frequency function."""
    assert isinstance(lifetime_to_frequency(1.0), float)
    assert lifetime_to_frequency(1.0) == pytest.approx(186.015665)
    assert_allclose(
        lifetime_to_frequency([4.0, 1.0]), [46.503916, 186.015665], atol=1e-3
    )


def test_lifetime_from_frequency():
    """Test lifetime_from_frequency function."""
    assert isinstance(lifetime_from_frequency(186.015665), float)
    assert lifetime_from_frequency(186.015665) == pytest.approx(1.0)
    assert_allclose(
        lifetime_from_frequency([46.503916, 186.015665]), [4.0, 1.0], atol=1e-3
    )


def test_lifetime_fraction_to_amplitude():
    """Test lifetime_fraction_to_amplitude function."""
    # assert isinstance(lifetime_fraction_to_amplitude(1.0, 1.0), float)
    assert_allclose(lifetime_fraction_to_amplitude(1.0, 1.0), 1.0, atol=1e-3)
    assert_allclose(
        lifetime_fraction_to_amplitude([4.0, 1.0], [1.6, 0.4]),
        [0.2, 0.2],
        atol=1e-3,
    )
    assert_allclose(
        lifetime_fraction_to_amplitude([[4.0], [1.0]], [[1.6], [0.4]], axis=0),
        [[0.2], [0.2]],
        atol=1e-3,
    )
    assert_allclose(
        lifetime_fraction_to_amplitude([4.0, 1.0], [1.6, 0.0]),
        [0.25, 0.0],
        atol=1e-3,
    )
    with pytest.warns(RuntimeWarning):
        assert_allclose(
            lifetime_fraction_to_amplitude([4.0, 0.0], [1.6, 0.4]),
            [0.2, numpy.inf],
            atol=1e-3,
        )
    with pytest.warns(RuntimeWarning):
        assert_allclose(
            lifetime_fraction_to_amplitude([4.0, 1.0], [0.0, 0.0]),
            [numpy.nan, numpy.nan],
            atol=1e-3,
        )


def test_lifetime_fraction_from_amplitude():
    """Test lifetime_fraction_from_amplitude function."""
    # assert isinstance(lifetime_fraction_from_amplitude(1.0, 1.0), float)
    assert_allclose(lifetime_fraction_from_amplitude(1.0, 1.0), 1.0, atol=1e-3)
    assert_allclose(
        lifetime_fraction_from_amplitude([4.0, 1.0], [0.4, 0.4]),
        [0.8, 0.2],
        atol=1e-3,
    )
    assert_allclose(
        lifetime_fraction_from_amplitude(
            [[4.0], [1.0]], [[0.4], [0.4]], axis=0
        ),
        [[0.8], [0.2]],
        atol=1e-3,
    )
    assert_allclose(
        lifetime_fraction_from_amplitude([4.0, 1.0], [0.5, 0.0]),
        [1.0, 0.0],
        atol=1e-3,
    )
    assert_allclose(
        lifetime_fraction_from_amplitude([4.0, 0.0], [0.4, 10.0]),
        [1.0, 0.0],
        atol=1e-3,
    )
    with pytest.warns(RuntimeWarning):
        assert_allclose(
            lifetime_fraction_from_amplitude([0.0, 0.0], [0.4, 0.4]),
            [numpy.nan, numpy.nan],
            atol=1e-3,
        )
    with pytest.warns(RuntimeWarning):
        assert_allclose(
            lifetime_fraction_from_amplitude([4.0, 1.0], [0.0, 0.0]),
            [numpy.nan, numpy.nan],
            atol=1e-3,
        )


def test_phasor_at_harmonic():
    """Test phasor_at_harmonic function."""
    # identity
    assert_allclose(phasor_at_harmonic(0.5, 1, 1), [0.5, 0.5], atol=1e-6)
    assert_allclose(phasor_at_harmonic(0.5, 2, 2), [0.5, 0.5], atol=1e-6)
    # up
    assert_allclose(phasor_at_harmonic(0.5, 1, 2), [0.2, 0.4], atol=1e-6)
    # down
    assert_allclose(phasor_at_harmonic(0.5, 2, 1), [0.8, 0.4], atol=1e-6)
    # phasor array
    assert_allclose(
        phasor_at_harmonic([0.4, 0.6], 1, 2),
        [[0.14285714, 0.27272727], [0.34992711, 0.44536177]],
        atol=1e-6,
    )
    # harmonic array
    assert_allclose(
        phasor_at_harmonic(0.5, 1, [1, 2, 4, 8]),
        [[0.5, 0.2, 0.058824, 0.015385], [0.5, 0.4, 0.235294, 0.123077]],
        atol=1e-6,
    )
    # out of bounds
    assert_array_equal(
        phasor_at_harmonic([-0.1, 1.0], 1, 1),
        [[0.0, 1.0], [0.0, 0.0]],
    )
    # test against phasor_from_lifetime
    real = 0.8
    harmonic = 1
    other_harmonic = [2, 3]
    assert_allclose(
        phasor_at_harmonic(real, harmonic, other_harmonic),
        phasor_from_lifetime(
            frequency=other_harmonic,
            lifetime=phasor_to_apparent_lifetime(
                real, math.sqrt(real - real * real), frequency=harmonic
            )[0],
        ),
        atol=1e-6,
    )
    # errors
    with pytest.raises(ValueError):
        phasor_at_harmonic(0.5, 0, 1)
    with pytest.raises(ValueError):
        phasor_at_harmonic(0.5, 1, 0)


def test_phasor_to_principal_plane():
    """Test phasor_to_principal_plane function."""
    # see phasorpy_principal_components.py for comments and visualization

    def distribution(values, stddev=0.05, samples=1000):
        return numpy.ascontiguousarray(
            numpy.vstack(
                [
                    numpy.random.normal(value, stddev, samples)
                    for value in values
                ]
            ).T
        )

    frequency = [80, 160, 240, 320, 400]
    real0, imag0 = phasor_from_lifetime(
        frequency,
        lifetime=distribution([0.5, 4.0]),
        fraction=distribution([0.4, 0.6]),
    )
    real1, imag1 = phasor_from_lifetime(
        frequency,
        lifetime=distribution([1.0, 8.0]),
        fraction=distribution([0.6, 0.4]),
    )
    real = numpy.hstack([real0, real1])
    imag = numpy.hstack([imag0, imag1])

    x, y, transformation_matrix = phasor_to_principal_plane(real, imag)
    assert x.shape == (2000,)
    assert y.shape == (2000,)
    assert transformation_matrix.shape == (2, 10)
    # TODO: these values are not always reproducible, why?
    assert_allclose(x.mean(), 0.307, atol=1e-2, rtol=1e-2)
    assert_allclose(y.mean(), 0.277, atol=1e-2, rtol=1e-2)

    # for single harmonics, reoriented projection matches phasor coordinates
    real = real[:1]
    imag = imag[:1]
    x, y, transformation_matrix = phasor_to_principal_plane(real, imag)
    assert_allclose(x, real[0], atol=1e-3)
    assert_allclose(y, imag[0], atol=1e-3)

    x, y, transformation_matrix = phasor_to_principal_plane(
        real, imag, reorient=False
    )
    with pytest.raises(AssertionError):
        assert_allclose(x, real[0], atol=1e-3)

    # exception
    with pytest.raises(ValueError):
        phasor_to_principal_plane([0.0, 1.0], [0.0])


@pytest.mark.parametrize(
    'real, imag, use_scipy, repeat, size, skip_axis, kwargs, expected',
    [
        # single element
        ([0], [0], False, 1, 3, None, {}, ([0], [0])),
        # all equal
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            False,
            1,
            3,
            None,
            {},
            (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            ),
        ),
        # random float values
        (
            [[0.5, 0.5, 2.0], [1.0, 2.0, 3.0], [10.0, 5.0, 1.0]],
            [[5.0, 6.0, 2.0], [10.0, 4.0, 8.0], [0.0, 7.0, 8.0]],
            False,
            1,
            3,
            None,
            {},
            (
                [[0.5, 1.0, 2.0], [1.0, 2.0, 2.0], [5.0, 3.0, 2.0]],
                [[5.0, 5.0, 4.0], [5.0, 6.0, 7.0], [4.0, 7.0, 8.0]],
            ),
        ),
        # 5x5 array with 3x3 filter
        (
            numpy.arange(25).reshape(5, 5),
            numpy.arange(25, 50).reshape(5, 5),
            False,
            1,
            3,
            None,
            {},
            (
                [
                    [1, 2, 3, 4, 4],
                    [5, 6, 7, 8, 9],
                    [10, 11, 12, 13, 14],
                    [15, 16, 17, 18, 19],
                    [20, 20, 21, 22, 23],
                ],
                [
                    [26, 27, 28, 29, 29],
                    [30, 31, 32, 33, 34],
                    [35, 36, 37, 38, 39],
                    [40, 41, 42, 43, 44],
                    [45, 45, 46, 47, 48],
                ],
            ),
        ),
        # 5x5 array with 3x3 filter repeated 5 times
        (
            numpy.arange(25).reshape(5, 5),
            numpy.arange(25, 50).reshape(5, 5),
            False,
            5,
            3,
            None,
            {},
            (
                [
                    [4, 4, 4, 4, 4],
                    [5, 6, 7, 8, 9],
                    [10, 11, 12, 13, 14],
                    [15, 16, 17, 18, 19],
                    [20, 20, 20, 20, 20],
                ],
                [
                    [29, 29, 29, 29, 29],
                    [30, 31, 32, 33, 34],
                    [35, 36, 37, 38, 39],
                    [40, 41, 42, 43, 44],
                    [45, 45, 45, 45, 45],
                ],
            ),
        ),
        # 5x5 array with 5x5 filter repeated 1 time
        (
            numpy.arange(25).reshape(5, 5),
            numpy.arange(25, 50).reshape(5, 5),
            False,
            1,
            5,
            None,
            {},
            (
                [
                    [2, 3, 4, 4, 4],
                    [5, 6, 7, 8, 9],
                    [10, 11, 12, 13, 14],
                    [15, 16, 17, 18, 19],
                    [20, 20, 20, 21, 22],
                ],
                [
                    [27, 28, 29, 29, 29],
                    [30, 31, 32, 33, 34],
                    [35, 36, 37, 38, 39],
                    [40, 41, 42, 43, 44],
                    [45, 45, 45, 46, 47],
                ],
            ),
        ),
        # 5x5 array with float32 dtype values
        (
            numpy.arange(25, dtype=numpy.float32).reshape(5, 5),
            numpy.arange(25, 50, dtype=numpy.float32).reshape(5, 5),
            False,
            5,
            3,
            None,
            {},
            (
                [
                    [4, 4, 4, 4, 4],
                    [5, 6, 7, 8, 9],
                    [10, 11, 12, 13, 14],
                    [15, 16, 17, 18, 19],
                    [20, 20, 20, 20, 20],
                ],
                [
                    [29, 29, 29, 29, 29],
                    [30, 31, 32, 33, 34],
                    [35, 36, 37, 38, 39],
                    [40, 41, 42, 43, 44],
                    [45, 45, 45, 45, 45],
                ],
            ),
        ),
        # 3x3x3 array with 3x3 filter repeated 3 with first axis skipped
        (
            numpy.arange(27).reshape(3, 3, 3),
            numpy.arange(10, 37).reshape(3, 3, 3),
            False,
            3,
            3,
            0,
            {},
            (
                [
                    [[2, 2, 2], [3, 4, 5], [6, 6, 6]],
                    [[11, 11, 11], [12, 13, 14], [15, 15, 15]],
                    [[20, 20, 20], [21, 22, 23], [24, 24, 24]],
                ],
                [
                    [[12, 12, 12], [13, 14, 15], [16, 16, 16]],
                    [[21, 21, 21], [22, 23, 24], [25, 25, 25]],
                    [[30, 30, 30], [31, 32, 33], [34, 34, 34]],
                ],
            ),
        ),
        # 'median_scipy' method with axes as kwarg
        (
            numpy.arange(27).reshape(3, 3, 3),
            numpy.arange(10, 37).reshape(3, 3, 3),
            True,
            3,
            3,
            0,  # None,
            {},  # {'axes': (1, 2)},
            (
                [
                    [[2, 2, 2], [3, 4, 5], [6, 6, 6]],
                    [[11, 11, 11], [12, 13, 14], [15, 15, 15]],
                    [[20, 20, 20], [21, 22, 23], [24, 24, 24]],
                ],
                [
                    [[12, 12, 12], [13, 14, 15], [16, 16, 16]],
                    [[21, 21, 21], [22, 23, 24], [25, 25, 25]],
                    [[30, 30, 30], [31, 32, 33], [34, 34, 34]],
                ],
            ),
        ),
        # same output for methods from 2D array without NaN
        (
            numpy.arange(25).reshape(5, 5),
            numpy.arange(25, 50).reshape(5, 5),
            False,
            1,
            3,
            None,
            {},
            phasor_filter_median(
                numpy.ones((5, 5)),
                numpy.arange(25).reshape(5, 5),
                numpy.arange(25, 50).reshape(5, 5),
                use_scipy=True,
            )[1:],
        ),
        # same output for methods from 3D array without NaN
        (
            numpy.arange(27).reshape(3, 3, 3),
            numpy.arange(10, 37).reshape(3, 3, 3),
            False,
            1,
            3,
            None,
            {},
            phasor_filter_median(
                numpy.ones((3, 3, 3)),
                numpy.arange(27).reshape(3, 3, 3),
                numpy.arange(10, 37).reshape(3, 3, 3),
                use_scipy=True,
            )[1:],
        ),
        # same output for methods from 3D array without NaN and skip axes
        (
            numpy.arange(27).reshape(3, 3, 3),
            numpy.arange(10, 37).reshape(3, 3, 3),
            False,
            1,
            3,
            0,
            {},
            phasor_filter_median(
                numpy.ones((3, 3, 3)),
                numpy.arange(27).reshape(3, 3, 3),
                numpy.arange(10, 37).reshape(3, 3, 3),
                use_scipy=True,
                skip_axis=0,
            )[1:],
        ),
        # non-contiguos axes for 2D filtering
        (
            numpy.arange(81).reshape(3, 3, 3, 3),
            numpy.arange(10, 91).reshape(3, 3, 3, 3),
            False,
            1,
            3,
            [0, 2],
            {},
            phasor_filter_median(
                numpy.ones((3, 3, 3, 3)),
                numpy.arange(81).reshape(3, 3, 3, 3),
                numpy.arange(10, 91).reshape(3, 3, 3, 3),
                use_scipy=True,
                skip_axis=[0, 2],
            )[1:],
        ),
        # repeat = 0
        (
            numpy.arange(9).reshape(3, 3),
            numpy.arange(10, 19).reshape(3, 3),
            False,
            0,
            3,
            None,
            {},
            (
                numpy.arange(9).reshape(3, 3),
                numpy.arange(10, 19).reshape(3, 3),
            ),
        ),
        # size = 1
        (
            numpy.arange(9).reshape(3, 3),
            numpy.arange(10, 19).reshape(3, 3),
            False,
            1,
            1,
            None,
            {},
            (
                numpy.arange(9).reshape(3, 3),
                numpy.arange(10, 19).reshape(3, 3),
            ),
        ),
    ],
)
def test_phasor_filter_median(
    real, imag, use_scipy, repeat, size, skip_axis, kwargs, expected
):
    """Test phasor_filter_median function."""
    assert_allclose(
        phasor_filter_median(
            numpy.full_like(real, 1),
            real,
            imag,
            use_scipy=use_scipy,
            repeat=repeat,
            size=size,
            skip_axis=skip_axis,
            **kwargs,
        )[1:],
        expected,
    )


def test_phasor_filter_median_errors():
    """Test phasor_filter_median function errors."""
    # shape mismatch
    with pytest.raises(ValueError):
        phasor_filter_median([[0]], [0], [0], repeat=1)
    # shape mismatch
    with pytest.raises(ValueError):
        phasor_filter_median([0], [0], [[0]], repeat=1)
    # repeat < 0
    with pytest.raises(ValueError):
        phasor_filter_median([[0]], [[0]], [[0]], repeat=-3)
    # size < 1
    with pytest.raises(ValueError):
        phasor_filter_median([[0]], [[0]], [[0]], size=0)


@pytest.mark.parametrize(
    'mean, real, imag, sigma, levels, harmonic, skip_axis, expected',
    [
        # single element
        (
            [1],
            [[1], [1]],
            [[1], [1]],
            2,
            1,
            None,
            None,
            ([1], [[1], [1]], [[1], [1]]),
        ),
        # 2D arrays with 2 harmonics not specified
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [
                [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
            ],
            [
                [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
            ],
            2,
            1,
            None,
            None,
            (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [
                    [
                        [0.425, 0.5, 0.575],
                        [0.425, 0.5, 0.575],
                        [0.425, 0.5, 0.575],
                    ],
                    [
                        [0.2125, 0.25, 0.2875],
                        [0.2125, 0.25, 0.2875],
                        [0.2125, 0.25, 0.2875],
                    ],
                ],
                [
                    [
                        [0.275, 0.275, 0.275],
                        [0.3, 0.3, 0.3],
                        [0.325, 0.325, 0.325],
                    ],
                    [
                        [0.1375, 0.1375, 0.1375],
                        [0.15, 0.15, 0.15],
                        [0.1625, 0.1625, 0.1625],
                    ],
                ],
            ),
        ),
        # 2D arrays with 1 and 2 harmonics specified
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [
                [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
            ],
            [
                [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
            ],
            2,
            1,
            [1, 2],
            None,
            (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [
                    [
                        [0.425, 0.5, 0.575],
                        [0.425, 0.5, 0.575],
                        [0.425, 0.5, 0.575],
                    ],
                    [
                        [0.2125, 0.25, 0.2875],
                        [0.2125, 0.25, 0.2875],
                        [0.2125, 0.25, 0.2875],
                    ],
                ],
                [
                    [
                        [0.275, 0.275, 0.275],
                        [0.3, 0.3, 0.3],
                        [0.325, 0.325, 0.325],
                    ],
                    [
                        [0.1375, 0.1375, 0.1375],
                        [0.15, 0.15, 0.15],
                        [0.1625, 0.1625, 0.1625],
                    ],
                ],
            ),
        ),
        # 2D arrays with 2 and 4 harmonics specified
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [
                [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
            ],
            [
                [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
            ],
            2,
            1,
            [1, 2],
            None,
            (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [
                    [
                        [0.425, 0.5, 0.575],
                        [0.425, 0.5, 0.575],
                        [0.425, 0.5, 0.575],
                    ],
                    [
                        [0.2125, 0.25, 0.2875],
                        [0.2125, 0.25, 0.2875],
                        [0.2125, 0.25, 0.2875],
                    ],
                ],
                [
                    [
                        [0.275, 0.275, 0.275],
                        [0.3, 0.3, 0.3],
                        [0.325, 0.325, 0.325],
                    ],
                    [
                        [0.1375, 0.1375, 0.1375],
                        [0.15, 0.15, 0.15],
                        [0.1625, 0.1625, 0.1625],
                    ],
                ],
            ),
        ),
        # 2D arrays with 1, 2 and 4 harmonics specified
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [
                [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
                [[0.05, 0.125, 0.2], [0.05, 0.125, 0.2], [0.05, 0.125, 0.2]],
            ],
            [
                [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
                [[0.05, 0.05, 0.05], [0.075, 0.075, 0.075], [0.1, 0.1, 0.1]],
            ],
            2,
            1,
            [1, 2, 4],
            None,
            (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [
                    [
                        [0.425, 0.5, 0.575],
                        [0.425, 0.5, 0.575],
                        [0.425, 0.5, 0.575],
                    ],
                    [
                        [0.2125, 0.25, 0.2875],
                        [0.2125, 0.25, 0.2875],
                        [0.2125, 0.25, 0.2875],
                    ],
                    [
                        [0.10625, 0.125, 0.14375],
                        [0.10625, 0.125, 0.14375],
                        [0.10625, 0.125, 0.14375],
                    ],
                ],
                [
                    [
                        [0.275, 0.275, 0.275],
                        [0.3, 0.3, 0.3],
                        [0.325, 0.325, 0.325],
                    ],
                    [
                        [0.1375, 0.1375, 0.1375],
                        [0.15, 0.15, 0.15],
                        [0.1625, 0.1625, 0.1625],
                    ],
                    [
                        [0.06875, 0.06875, 0.06875],
                        [0.075, 0.075, 0.075],
                        [0.08125, 0.08125, 0.08125],
                    ],
                ],
            ),
        ),
        # 2D arrays with 1, 2, 4 and 8 harmonics specified
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [
                [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
                [[0.05, 0.125, 0.2], [0.05, 0.125, 0.2], [0.05, 0.125, 0.2]],
                [
                    [0.025, 0.0625, 0.1],
                    [0.025, 0.0625, 0.1],
                    [0.025, 0.0625, 0.1],
                ],
            ],
            [
                [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
                [[0.05, 0.05, 0.05], [0.075, 0.075, 0.075], [0.1, 0.1, 0.1]],
                [
                    [0.025, 0.025, 0.025],
                    [0.0375, 0.0375, 0.0375],
                    [0.05, 0.05, 0.05],
                ],
            ],
            2,
            1,
            [1, 2, 4, 8],
            None,
            (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [
                    [
                        [0.425, 0.5, 0.575],
                        [0.425, 0.5, 0.575],
                        [0.425, 0.5, 0.575],
                    ],
                    [
                        [0.2125, 0.25, 0.2875],
                        [0.2125, 0.25, 0.2875],
                        [0.2125, 0.25, 0.2875],
                    ],
                    [
                        [0.10625, 0.125, 0.14375],
                        [0.10625, 0.125, 0.14375],
                        [0.10625, 0.125, 0.14375],
                    ],
                    [
                        [0.053125, 0.0625, 0.071875],
                        [0.053125, 0.0625, 0.071875],
                        [0.053125, 0.0625, 0.071875],
                    ],
                ],
                [
                    [
                        [0.275, 0.275, 0.275],
                        [0.3, 0.3, 0.3],
                        [0.325, 0.325, 0.325],
                    ],
                    [
                        [0.1375, 0.1375, 0.1375],
                        [0.15, 0.15, 0.15],
                        [0.1625, 0.1625, 0.1625],
                    ],
                    [
                        [0.06875, 0.06875, 0.06875],
                        [0.075, 0.075, 0.075],
                        [0.08125, 0.08125, 0.08125],
                    ],
                    [
                        [0.034375, 0.034375, 0.034375],
                        [0.0375, 0.0375, 0.0375],
                        [0.040625, 0.040625, 0.040625],
                    ],
                ],
            ),
        ),
        # levels == 0 (no filtering)
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [
                [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
            ],
            [
                [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
            ],
            2,
            0,
            None,
            None,
            (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [
                    [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                    [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
                ],
                [
                    [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                    [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
                ],
            ),
        ),
        # sigma == 0 (no filtering)
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [
                [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
            ],
            [
                [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
            ],
            0,
            1,
            None,
            None,
            (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [
                    [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                    [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
                ],
                [
                    [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                    [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
                ],
            ),
        ),
        # sigma < 0 (no filtering)
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [
                [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
            ],
            [
                [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
            ],
            -2,
            1,
            None,
            None,
            (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [
                    [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                    [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
                ],
                [
                    [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                    [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
                ],
            ),
        ),
        # skip_axis
        (
            [
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            ],
            [
                [
                    [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                    [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                    [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                    [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                ],
                [
                    [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
                    [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
                    [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
                    [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
                ],
            ],
            [
                [
                    [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                    [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                    [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                    [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                ],
                [
                    [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
                    [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
                    [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
                    [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
                ],
            ],
            2,
            1,
            None,
            0,
            (
                [
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                ],
                [
                    [
                        [
                            [0.425, 0.5, 0.575],
                            [0.425, 0.5, 0.575],
                            [0.425, 0.5, 0.575],
                        ],
                        [
                            [0.425, 0.5, 0.575],
                            [0.425, 0.5, 0.575],
                            [0.425, 0.5, 0.575],
                        ],
                        [
                            [0.425, 0.5, 0.575],
                            [0.425, 0.5, 0.575],
                            [0.425, 0.5, 0.575],
                        ],
                        [
                            [0.425, 0.5, 0.575],
                            [0.425, 0.5, 0.575],
                            [0.425, 0.5, 0.575],
                        ],
                    ],
                    [
                        [
                            [0.2125, 0.25, 0.2875],
                            [0.2125, 0.25, 0.2875],
                            [0.2125, 0.25, 0.2875],
                        ],
                        [
                            [0.2125, 0.25, 0.2875],
                            [0.2125, 0.25, 0.2875],
                            [0.2125, 0.25, 0.2875],
                        ],
                        [
                            [0.2125, 0.25, 0.2875],
                            [0.2125, 0.25, 0.2875],
                            [0.2125, 0.25, 0.2875],
                        ],
                        [
                            [0.2125, 0.25, 0.2875],
                            [0.2125, 0.25, 0.2875],
                            [0.2125, 0.25, 0.2875],
                        ],
                    ],
                ],
                [
                    [
                        [
                            [0.275, 0.275, 0.275],
                            [0.3, 0.3, 0.3],
                            [0.325, 0.325, 0.325],
                        ],
                        [
                            [0.275, 0.275, 0.275],
                            [0.3, 0.3, 0.3],
                            [0.325, 0.325, 0.325],
                        ],
                        [
                            [0.275, 0.275, 0.275],
                            [0.3, 0.3, 0.3],
                            [0.325, 0.325, 0.325],
                        ],
                        [
                            [0.275, 0.275, 0.275],
                            [0.3, 0.3, 0.3],
                            [0.325, 0.325, 0.325],
                        ],
                    ],
                    [
                        [
                            [0.1375, 0.1375, 0.1375],
                            [0.15, 0.15, 0.15],
                            [0.1625, 0.1625, 0.1625],
                        ],
                        [
                            [0.1375, 0.1375, 0.1375],
                            [0.15, 0.15, 0.15],
                            [0.1625, 0.1625, 0.1625],
                        ],
                        [
                            [0.1375, 0.1375, 0.1375],
                            [0.15, 0.15, 0.15],
                            [0.1625, 0.1625, 0.1625],
                        ],
                        [
                            [0.1375, 0.1375, 0.1375],
                            [0.15, 0.15, 0.15],
                            [0.1625, 0.1625, 0.1625],
                        ],
                    ],
                ],
            ),
        ),
        # NANs
        # zeroes in mean
    ],
)
def test_phasor_filter_pawflim(
    mean, real, imag, sigma, levels, harmonic, skip_axis, expected
):
    """Test phasor_filter_pawflim function."""
    mean, real, imag = phasor_filter_pawflim(
        mean,
        real,
        imag,
        sigma=sigma,
        levels=levels,
        harmonic=harmonic,
        skip_axis=skip_axis,
    )
    assert_allclose(mean, expected[0])
    assert_allclose(real, expected[1])
    assert_allclose(imag, expected[2])


def test_phasor_filter_pawflim_errors():
    """Test phasor_filter_pawflim function errors."""
    # shape mismatch between real and imag
    with pytest.raises(ValueError):
        phasor_filter_pawflim([1], [[1]], [[1], [1]])
    # shape mismatch between real and imag
    with pytest.raises(ValueError):
        phasor_filter_pawflim([1], [[1], [1]], [[1]])
    # shape mismatch between mean and real
    with pytest.raises(ValueError):
        phasor_filter_pawflim(
            [[1, 1, 1], [1, 1, 1]],
            [[[1, 1], [1, 1]], [[1, 1], [1, 1]]],
            [[[1, 1], [1, 1]], [[1, 1], [1, 1]]],
        )
    # levels < 1
    with pytest.raises(ValueError):
        phasor_filter_pawflim([1], [[1], [1]], [[1], [1]], levels=-1)
    # not all harmonics with corresponding double/half
    with pytest.raises(ValueError):
        phasor_filter_pawflim([1], [[1], [1]], [[1], [1]], harmonic=[1, 3])
    # not all harmonics with corresponding double/half
    with pytest.raises(ValueError):
        phasor_filter_pawflim([1], [[1], [1]], [[1], [1]], harmonic=[2, 8])
    # no harmonic axis
    with pytest.raises(ValueError):
        phasor_filter_pawflim([1], [1], [1])
    # no harmonic axis
    with pytest.raises(ValueError):
        phasor_filter_pawflim([[1], [1]], [[1], [1]], [[1], [1]])
    # no harmonic axis 2D
    with pytest.raises(ValueError):
        phasor_filter_pawflim([[1], [1]], [[1], [1]], [[1], [1]])
    # less than two harmonics
    with pytest.raises(ValueError):
        phasor_filter_pawflim([1], [[1]], [[1]])
    # number of harmonics does not match first axis of `real` and `imag`
    with pytest.raises(ValueError):
        phasor_filter_pawflim([1], [[1], [1]], [[1], [1]], harmonic=[1, 2, 3])


def test_phasor_threshold():
    """Test phasor_threshold function."""
    nan = numpy.nan
    # no threshold
    assert_allclose(
        phasor_threshold([0.5, 0.4], [0.2, 0.5], [0.3, 0.5]),
        ([0.5, 0.4], [0.2, 0.5], [0.3, 0.5]),
    )
    # lower mean threshold
    assert_allclose(
        phasor_threshold([0.5, 0.4], [0.2, 0.5], [0.3, 0.5], 0.5),
        ([0.5, nan], [0.2, nan], [0.3, nan]),
    )
    # nan in mean propagated to real and imag
    assert_allclose(
        phasor_threshold([0.5, nan], [0.2, 0.5], [0.3, 0.5], 0.5),
        ([0.5, nan], [0.2, nan], [0.3, nan]),
    )
    # nan in real propagated to mean and imag
    assert_allclose(
        phasor_threshold([0.5, 0.4], [0.2, nan], [0.3, 0.5], 0.5),
        ([0.5, nan], [0.2, nan], [0.3, nan]),
    )
    # nan in imag propagated to real and mean
    assert_allclose(
        phasor_threshold([0.5, 0.4], [0.2, 0.5], [0.3, nan], 0.5),
        ([0.5, nan], [0.2, nan], [0.3, nan]),
    )
    # 2D array with lower mean threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            0.5,
        ),
        (
            [[0.5, nan], [0.8, 0.6]],
            [[0.1, nan], [0.3, 0.4]],
            [[0.5, nan], [0.7, 0.8]],
        ),
    )
    # 2D array with lower and upper mean threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            0.5,
            0.7,
        ),
        (
            [[0.5, nan], [nan, 0.6]],
            [[0.1, nan], [nan, 0.4]],
            [[0.5, nan], [nan, 0.8]],
        ),
    )
    # 2D array with lower mean and real threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            0.5,
            real_min=0.3,
            real_max=0.35,
        ),
        (
            [[nan, nan], [0.8, nan]],
            [[nan, nan], [0.3, nan]],
            [[nan, nan], [0.7, nan]],
        ),
    )
    # 2D array with lower mean and imag threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            0.5,
            imag_min=0.7,
            imag_max=0.75,
        ),
        (
            [[nan, nan], [0.8, nan]],
            [[nan, nan], [0.3, nan]],
            [[nan, nan], [0.7, nan]],
        ),
    )
    # 3D array with different real threshold for first dimension
    assert_allclose(
        phasor_threshold(
            [[[0.4, 0.5]], [[0.8, 0.9]]],
            [[[0.1, 0.2]], [[0.5, 0.6]]],
            [[[0.5, 0.3]], [[0.6, 0.7]]],
            real_min=[[[0.2]], [[0.6]]],
        ),
        (
            [[[nan, 0.5]], [[nan, 0.9]]],
            [[[nan, 0.2]], [[nan, 0.6]]],
            [[[nan, 0.3]], [[nan, 0.7]]],
        ),
    )
    # 2D array with lower and upper phase threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            phase_min=1.2,
            phase_max=1.3,
        ),
        (
            [[nan, 0.4], [nan, nan]],
            [[nan, 0.2], [nan, nan]],
            [[nan, 0.6], [nan, nan]],
        ),
    )
    # 2D array with lower and upper modulation threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            modulation_min=0.7,
            modulation_max=0.8,
        ),
        (
            [[nan, nan], [0.8, nan]],
            [[nan, nan], [0.3, nan]],
            [[nan, nan], [0.7, nan]],
        ),
    )
    # 2D array with lower and upper phase and modulation threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            phase_min=1.2,
            phase_max=1.3,
            modulation_min=0.7,
            modulation_max=0.8,
        ),
        (
            [[nan, nan], [nan, nan]],
            [[nan, nan], [nan, nan]],
            [[nan, nan], [nan, nan]],
        ),
    )
    # 2D array with open interval, lower and upper mean threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            0.5,
            0.8,
            open_interval=True,
        ),
        (
            [[nan, nan], [nan, 0.6]],
            [[nan, nan], [nan, 0.4]],
            [[nan, nan], [nan, 0.8]],
        ),
    )
    # 2D array with open interval, lower and upper real threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            real_min=0.2,
            real_max=0.4,
            open_interval=True,
        ),
        (
            [[nan, nan], [0.8, nan]],
            [[nan, nan], [0.3, nan]],
            [[nan, nan], [0.7, nan]],
        ),
    )
    # 2D array with open interval, lower and upper imag threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            imag_min=0.6,
            imag_max=0.8,
            open_interval=True,
        ),
        (
            [[nan, nan], [0.8, nan]],
            [[nan, nan], [0.3, nan]],
            [[nan, nan], [0.7, nan]],
        ),
    )
    # 2D array with open interval, lower and upper phase threshold
    real, imag = phasor_from_polar(
        [[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]
    )
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            real,
            imag,
            phase_min=0.2,
            phase_max=0.4,
            open_interval=True,
        ),
        (
            [[nan, nan], [0.8, nan]],
            [[nan, nan], [real[1][0], nan]],
            [[nan, nan], [imag[1][0], nan]],
        ),
    )
    # 2D array with open interval, lower and upper modulation threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            real,
            imag,
            modulation_min=0.6,
            modulation_max=0.8,
            open_interval=True,
        ),
        (
            [[nan, nan], [0.8, nan]],
            [[nan, nan], [real[1][0], nan]],
            [[nan, nan], [imag[1][0], nan]],
        ),
    )


@pytest.mark.parametrize('dtype', [numpy.float32, numpy.float64])
@pytest.mark.parametrize(
    'kwargs',
    [
        {},
        {'mean_min': 0.5},
        {'real_min': 0.5, 'phase_max': 0.7},
    ],
)
def test_phasor_threshold_dtype(dtype, kwargs):
    """Test phasor_threshold function preserves dtype."""
    mean, real, imag = phasor_threshold(
        numpy.asarray([0.5, 0.4], dtype=dtype),
        numpy.asarray([0.2, 0.5], dtype=dtype),
        numpy.asarray([0.3, 0.5], dtype=dtype),
        **kwargs,
    )
    assert mean.dtype == dtype
    assert real.dtype == dtype
    assert imag.dtype == dtype


def test_phasor_threshold_harmonic():
    """Test phasor_threshold function with multiple harmonics."""
    nan = numpy.nan
    data = numpy.random.random((3, 2, 8))
    data[0, 0, 0] = nan
    data[1, 0, 1] = nan
    data[1, 1, 2] = nan
    data[2, 0, 3] = nan
    data[2, 1, 4] = nan
    mean, real, imag = data
    mean_copy, real_copy, imag_copy = data.copy()

    # NaNs should propagate to all dimensions
    result = data.copy()
    result[:, :, :5] = nan
    mean_, real_, imag_ = result

    # detect harmonic axis
    mean1, real1, imag1 = phasor_threshold(mean[0], real, imag)
    assert_array_equal(mean, mean_copy)
    assert_array_equal(real, real_copy)
    assert_array_equal(imag, imag_copy)
    assert_allclose(mean1, mean_[0])
    assert_allclose(real1, real_)
    assert_allclose(imag1, imag_)

    # scalar
    mean, real, imag = data[..., 0]
    mean_, real_, imag_ = result[..., 0]

    mean1, real1, imag1 = phasor_threshold(mean[0], real, imag)
    assert_allclose(mean1, mean_[0])
    assert_allclose(real1, real_)
    assert_allclose(imag1, imag_)

    # use ufunc: disable detect harmonic axis
    mean, real, imag = data
    result = data.copy()
    result[0, 1, :] = result[0, 0, :]
    result[0, 0, 1] = nan
    result[0, 1, 2] = nan
    result[0, 0, 3] = nan
    result[0, 1, 4] = nan
    result[1, 0, 0] = nan
    result[1, 0, 3] = nan
    result[1, 1, 0] = nan
    result[1, 1, 4] = nan
    result[2, 0, 0] = nan
    result[2, 0, 1] = nan
    result[2, 1, 0] = nan
    result[2, 1, 2] = nan
    mean_, real_, imag_ = result

    mean1, real1, imag1 = phasor_threshold(
        mean[0], real, imag, detect_harmonics=False
    )
    assert_allclose(mean1, mean_)
    assert_allclose(real1, real_)
    assert_allclose(imag1, imag_)


def test_phasor_nearest_neighbor():
    """Test phasor_nearest_neighbor function."""
    nan = numpy.nan

    # test scalar inputs
    assert_array_equal(phasor_nearest_neighbor(1, 1, 1, 1), 0)
    assert_array_equal(phasor_nearest_neighbor(1, 1, 1, 1, values=1), 1.0)
    assert_array_equal(phasor_nearest_neighbor(nan, 1, 1, 1), -1)
    assert_array_equal(phasor_nearest_neighbor(nan, 1, 1, 1, values=1), nan)

    # Test input arrays are not modified, no values
    arr = numpy.array([[nan, 2], [3, 4]])
    original_arr = arr.copy()
    result = phasor_nearest_neighbor(arr, arr, arr, arr)
    assert result.dtype == numpy.int8
    assert_array_equal(arr, original_arr)
    assert_array_equal(result, [[-1, 1], [2, 3]])

    # Test input arrays are not modified, with values
    values = numpy.array([[5, 6], [7, 8]])
    original_values = values.copy()
    result = phasor_nearest_neighbor(arr, arr, arr / 2, arr / 2, values=values)
    assert_array_equal(arr, original_arr)
    assert_array_equal(values, original_values)
    assert_array_equal(result, [[nan, 8], [8, 8]])

    # test dtype parameter
    result = phasor_nearest_neighbor(
        arr, arr, arr / 2, arr / 2, values=values, dtype=numpy.float32
    )
    assert result.dtype == numpy.float32
    assert_array_equal(result, [[nan, 8], [8, 8]])

    # test num_threads parameter
    result = phasor_nearest_neighbor(
        arr, arr, arr / 2, arr / 2, values=values, num_threads=2
    )
    assert_array_equal(result, [[nan, 8], [8, 8]])

    # Test distance_max parameter
    result = phasor_nearest_neighbor(arr, arr, [2], [2], distance_max=2)
    assert_array_equal(result, [[-1, 0], [0, -1]])

    # Test distance_max with values
    result = phasor_nearest_neighbor(
        arr, arr, [2, 2], [2, 2], values=[10, 20], distance_max=2
    )
    assert_array_equal(result, [[nan, 10], [10, nan]])

    # Test multi-dimensional inputs
    # TODO: modify this when multiple dimensions are supported
    arr = numpy.array(
        [[[1, 1], [1, nan]], [[2, 2], [2, 2]], [[3, 3], [nan, 3]]]
    )
    neighbor_arr = numpy.array(
        [[[1, 2], [3, 4]], [[2, 3], [4, 5]], [[3, 4], [5, 6]]]
    )
    values = numpy.array(
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]]
    )
    result = phasor_nearest_neighbor(
        arr, arr, neighbor_arr, neighbor_arr, values=values
    )
    assert_array_equal(
        result, [[[0, 0], [0, nan]], [[1, 1], [1, 1]], [[2, 2], [nan, 2]]]
    )


@pytest.mark.xfail(reason='Multiple harmonics not yet supported')
def test_phasor_nearest_neighbor_harmonics():
    """Test phasor_nearest_neighbor function with multiple harmonics."""
    nan = numpy.nan
    arr = numpy.array(
        [[[1, 1], [1, nan]], [[2, 2], [2, 2]], [[3, 3], [nan, 3]]]
    )
    neighbor_arr = numpy.array(
        [[[1, 2], [3, 4]], [[2, 3], [4, 5]], [[3, 4], [5, 6]]]
    )
    values = numpy.array([[0, 1], [2, 3]])
    result = phasor_nearest_neighbor(
        arr, arr, neighbor_arr, neighbor_arr, values=values
    )
    assert_array_equal(
        result,
        [[[0, 0], [nan, nan]], [[0, 0], [nan, nan]], [[0, 0], [nan, nan]]],
    )


def test_phasor_nearest_neighbor_errors():
    """Test phasor_nearest_neighbor function errors."""
    arr = numpy.ones((3, 2))

    # Shape mismatch between real and imag
    with pytest.raises(ValueError):
        phasor_nearest_neighbor(arr, numpy.ones((3, 3)), arr, arr)

    # Shape mismatch between neighbor_real and neighbor_imag
    with pytest.raises(ValueError):
        phasor_nearest_neighbor(arr, arr, arr, numpy.ones((3, 3)))

    # Shape mismatch between values and neighbor_real
    with pytest.raises(ValueError):
        phasor_nearest_neighbor(arr, arr, arr, arr, values=numpy.ones((3, 3)))

    # distance_max = 0
    with pytest.raises(ValueError):
        phasor_nearest_neighbor(arr, arr, arr, arr, distance_max=0)

    # distance_max < 0
    with pytest.raises(ValueError):
        phasor_nearest_neighbor(arr, arr, arr, arr, distance_max=-1)

    # not a floating point types
    with pytest.raises(ValueError):
        phasor_nearest_neighbor(arr, arr, arr, arr, dtype=numpy.int8)


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type, unreachable"
