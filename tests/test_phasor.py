"""Test the phasorpy._phasor module."""

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

from phasorpy import (
    phasor_center,
    phasor_divide,
    phasor_from_lifetime,
    phasor_from_polar,
    phasor_from_signal,
    phasor_multiply,
    phasor_normalize,
    phasor_to_complex,
    phasor_to_polar,
    phasor_to_principal_plane,
    phasor_to_signal,
    phasor_transform,
)

SYNTH_DATA_ARRAY = numpy.array([[50, 1], [1, 1]])
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


def test_phasor_to_principal_plane():
    """Test phasor_to_principal_plane function."""
    # see phasorpy_principal_components.py for comments and visualization

    def distribution(values, stddev=0.05, samples=100):
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
    assert x.shape == (200,)
    assert y.shape == (200,)
    assert transformation_matrix.shape == (2, 10)
    assert_allclose(x.mean(), 0.306839, atol=1e-2, rtol=1e-2)
    assert_allclose(y.mean(), 0.281617, atol=1e-2, rtol=1e-2)

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


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type, unreachable"
