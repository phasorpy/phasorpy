"""Tests for the phasorpy.phasor module."""

import copy
import math

import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
)

try:
    from scipy.fft import fft as scipy_fft
except ImportError:
    scipy_fft = None

try:
    from mkl_fft._numpy_fft import fft as mkl_fft
except ImportError:
    mkl_fft = None

from phasorpy.phasor import (
    _parse_skip_axis,
    fraction_from_amplitude,
    fraction_to_amplitude,
    frequency_from_lifetime,
    frequency_to_lifetime,
    phasor_at_harmonic,
    phasor_calibrate,
    phasor_center,
    phasor_from_apparent_lifetime,
    phasor_from_fret_acceptor,
    phasor_from_fret_donor,
    phasor_from_lifetime,
    phasor_from_polar,
    phasor_from_signal,
    phasor_from_signal_fft,
    phasor_semicircle,
    phasor_to_apparent_lifetime,
    phasor_to_polar,
    phasor_to_principal_plane,
    phasor_transform,
    polar_from_apparent_lifetime,
    polar_from_reference,
    polar_from_reference_phasor,
    polar_to_apparent_lifetime,
)

SYNTH_DATA_ARRAY = numpy.array([[50, 1], [1, 1]])
SYNTH_DATA_LIST = [1, 2, 4]
SYNTH_PHI = numpy.array([[0.5, 0.5], [0.5, 0.5]])
SYNTH_MOD = numpy.array([[2, 2], [2, 2]])


@pytest.mark.parametrize('fft', (True, False))
def test_phasor_from_signal(fft):
    """Test `phasor_from_signal` functions."""
    func = phasor_from_signal_fft if fft else phasor_from_signal
    sample_phase = numpy.linspace(0, 2 * math.pi, 7, endpoint=False)
    signal = 1.1 * (numpy.cos(sample_phase - 0.46364761) * 2 * 0.44721359 + 1)
    signal_copy = signal.copy()
    mean, real, imag = func(signal)
    assert_array_equal(signal, signal_copy)
    assert isinstance(real, float)
    assert_allclose((mean, real, imag), (1.1, 0.4, 0.2), atol=1e-6)
    assert_allclose(
        func(signal),
        (1.1, 0.4, 0.2),
        atol=1e-6,
    )
    if not fft:
        assert_allclose(
            func(signal[::-1], sample_phase=sample_phase[::-1], num_threads=0),
            (1.1, 0.4, 0.2),
            atol=1e-6,
        )
        assert_allclose(
            func(numpy.cos(sample_phase)), (0.0, 0.0, 0.0), atol=1e-6
        )
    assert_allclose(
        func(numpy.zeros(256)),
        (0.0, numpy.nan, numpy.nan) if fft else (0.0, 0.0, 0.0),
        atol=1e-6,
    )
    assert_allclose(func(signal, harmonic=2), (1.1, 0.0, 0.0), atol=1e-6)
    dc, re, im = func(signal, harmonic=[1, 2])
    assert_allclose(dc, 1.1, atol=1e-6)
    assert_allclose(re, [0.4, 0.0], atol=1e-6)
    assert_allclose(im, [0.2, 0.0], atol=1e-6)
    with pytest.raises(IndexError):
        func(signal, harmonic=0)
    with pytest.raises(ValueError):
        func(signal[:2])
    with pytest.raises(TypeError):
        func(signal, harmonic=1.0)
    with pytest.raises(TypeError):
        func(signal, harmonic=[])
    with pytest.raises(TypeError):
        func(signal, harmonic=[1.0])
    with pytest.raises(IndexError):
        func(signal, harmonic=[0])
    with pytest.raises(IndexError):
        func(signal, harmonic=[4])
    if not fft:
        with pytest.raises(IndexError):
            func(signal, harmonic=4)
        with pytest.raises(ValueError):
            func(signal, sample_phase=sample_phase, harmonic=1)
        with pytest.raises(ValueError):
            func(signal, sample_phase=sample_phase[::-2])
        with pytest.raises(TypeError):
            func(signal.astype('complex64'))
        with pytest.raises(TypeError):
            func(signal, dtype='int8')


@pytest.mark.parametrize('fft', (True, False))
@pytest.mark.parametrize(
    "shape, axis, dtype, dtype_out",
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
def test_phasor_from_signal_param(fft, shape, axis, dtype, dtype_out):
    """Test `phasor_from_signal` functions parameters."""
    samples = shape[axis]
    dtype = numpy.dtype(dtype)
    signal = numpy.empty(shape, dtype)
    sample_phase = numpy.linspace(0, 2 * math.pi, samples, endpoint=False)
    if not fft:
        sample_phase[0] = sample_phase[-1]  # out of order
        sample_phase[-1] = 0.0
    sig = 2.1 * (numpy.cos(sample_phase - 0.46364761) * 2 * 0.44721359 + 1)
    if dtype.kind != 'f':
        sig *= 1000
    sig = sig.astype(dtype)
    reshape = [1] * len(shape)
    reshape[axis] = samples
    signal[:] = sig.reshape(reshape)
    if fft:
        mean, real, imag = phasor_from_signal_fft(signal, axis=axis)
    else:
        num_threads = 4 if signal.size > 4096 else 1
        mean, real, imag = phasor_from_signal(
            signal,
            axis=axis,
            sample_phase=sample_phase,
            dtype=dtype_out,
            num_threads=num_threads,
        )
    if isinstance(mean, numpy.ndarray):
        if not fft:
            assert mean.dtype == dtype_out
        assert mean.shape == shape[:axis] + shape[axis + 1 :]
    if dtype.kind == 'f':
        assert_allclose(numpy.mean(mean), 2.1, 1e-3)
    else:
        assert_allclose(numpy.mean(mean), 2100, 1)
    assert_allclose(numpy.mean(real), 0.4, 1e-3)
    assert_allclose(numpy.mean(imag), 0.2, 1e-3)


@pytest.mark.parametrize('fft', (True, False))
def test_phasor_from_signal_noncontig(fft):
    """Test `phasor_from_signal` functions with non-contiguous input."""
    dtype = numpy.float64
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
    if fft:
        mean, real, imag = phasor_from_signal_fft(signal, axis=-3)
    else:
        mean, real, imag = phasor_from_signal(signal, axis=-3, dtype=dtype)
    assert_array_equal(signal, signal_copy)
    assert mean.shape == signal.shape[:1] + signal.shape[1 + 1 :]
    assert_allclose(numpy.mean(mean), 2.1, 1e-3)
    assert_allclose(numpy.mean(real), 0.4, 1e-3)
    assert_allclose(numpy.mean(imag), 0.2, 1e-3)


@pytest.mark.parametrize('scalar', (True, False))
@pytest.mark.parametrize('harmonic', (1, 2, 8, [1], [1, 2, 8]))
def test_phasor_from_signal_harmonic(scalar, harmonic):
    """Test `phasor_from_signal` functions harmonic parameter."""
    rng = numpy.random.default_rng(1)
    signal = rng.random((33,) if scalar else (3, 33, 61, 63))
    signal += 1.1
    kwargs = dict(axis=0 if scalar else 1, harmonic=harmonic)
    mean0, real0, imag0 = phasor_from_signal(signal, **kwargs)
    mean1, real1, imag1 = phasor_from_signal_fft(signal, **kwargs)
    assert_allclose(mean0, mean1, 1e-8)
    assert_allclose(real0, real1, 1e-8)
    assert_allclose(imag0, imag1, 1e-8)


@pytest.mark.parametrize('fft_func', (scipy_fft, mkl_fft))
@pytest.mark.parametrize('scalar', (True, False))
@pytest.mark.parametrize('harmonic', (1, [4], [1, 4]))
def test_phasor_from_signal_fft_func(fft_func, scalar, harmonic):
    """Test `phasor_from_signal_fft` functions `fft_func` parameter."""
    if fft_func is None:
        pytest.skip('fft_func could not be imported')
    rng = numpy.random.default_rng(1)
    signal = rng.random((33,) if scalar else (3, 33, 61, 63))
    signal += 1.1
    kwargs = dict(axis=0 if scalar else 1, harmonic=harmonic)
    mean0, real0, imag0 = phasor_from_signal_fft(signal, **kwargs)
    mean1, real1, imag1 = phasor_from_signal_fft(
        signal, fft_func=fft_func, **kwargs
    )
    assert_allclose(mean0, mean1, 1e-8)
    assert_allclose(real0, real1, 1e-8)
    assert_allclose(imag0, imag1, 1e-8)


def test_phasor_semicircle():
    """Test `phasor_semicircle` function."""
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
    "real, imag, expected_phase, expected_modulation",
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
    """Test `polar_from_reference` function with various inputs."""
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
    """Test `polar_from_reference_phasor` function with various inputs."""
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


@pytest.mark.parametrize(
    """real, imag,
    phase_zero, modulation_zero,
    expected_real, expected_imag""",
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
        ),  # test with phase_zero and modulation_zero as arrays
    ],
)
def test_phasor_transform(
    real,
    imag,
    phase_zero,
    modulation_zero,
    expected_real,
    expected_imag,
):
    """Test `phasor_transform` function with various inputs."""
    real_copy = copy.deepcopy(real)
    imag_copy = copy.deepcopy(imag)
    if phase_zero is not None and modulation_zero is not None:
        calibrated_real, calibrated_imag = phasor_transform(
            real_copy, imag_copy, phase_zero, modulation_zero
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
    """real, imag, kwargs,
    expected_real_center, expected_imag_center""",
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
    ],
)
def test_phasor_center(
    real,
    imag,
    kwargs,
    expected_real_center,
    expected_imag_center,
):
    """Test `phasor_center` function with various inputs and methods."""
    real_copy = copy.deepcopy(real)
    imag_copy = copy.deepcopy(imag)
    real_center, imag_center = phasor_center(real_copy, imag_copy, **kwargs)
    assert_array_equal(real, real_copy)
    assert_array_equal(imag, imag_copy)
    assert_almost_equal(real_center, expected_real_center)
    assert_almost_equal(imag_center, expected_imag_center)


def test_phasor_center_exceptions():
    """Test exceptions in `phasor_center` function."""
    with pytest.raises(ValueError):
        phasor_center(0, 0, method='method_not_supported')
    with pytest.raises(ValueError):
        phasor_center([0], [0, 0])
    with pytest.raises(IndexError):
        phasor_center([0, 0], [0, 0], skip_axis=1)


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
    """Test `phasor_from_lifetime` function."""
    result = phasor_from_lifetime(*args, **kwargs, keepdims=True)
    for actual, desired in zip(result, expected):
        assert actual.ndim == 2
        assert_allclose(actual.squeeze(), desired, atol=1e-6)


def test_phasor_from_lifetime_exceptions():
    """Test exceptions in `phasor_from_lifetime` function."""
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
    """Test `phasor_from_lifetime` function does not modify input."""
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
        (
            (2, 2, 2, 2),
            {'frequency': 80, 'lifetime': 1.9894368},
            (numpy.array(0.5), numpy.array(0.5)),
        ),  # single lifetime
        (
            (0.5, 0.7, 0.4, 0.3),
            {'frequency': 80, 'lifetime': 4},
            (numpy.array(0.11789139), numpy.array(0.75703471)),
        ),
        (
            (-0.5, -0.7, -0.4, -0.3),
            {'frequency': 80, 'lifetime': 4},
            (numpy.array(0.11789139), numpy.array(0.75703471)),
        ),
        (
            (2, 2, 2, 2),
            {
                'frequency': 80,
                'lifetime': [3.9788735, 0.9947183],
                'fraction': [0.25, 0.75],
            },
            (numpy.array(0.65), numpy.array(0.4)),
        ),  # two lifetimes with fractions
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
        ),  # single lifetime
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
        ),  # multiple lifetime
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
        ),  # multiple lifetime with median method
        # array data
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
        ),  # single lifetime
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
        ),  # multiple lifetime
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
        ),  # multiple lifetime with median method
        (
            (
                numpy.stack(
                    (
                        SYNTH_DATA_ARRAY,
                        SYNTH_DATA_ARRAY / 2,
                        SYNTH_DATA_ARRAY / 3,
                    ),
                    axis=0,
                ),
                numpy.stack(
                    (
                        SYNTH_DATA_ARRAY,
                        SYNTH_DATA_ARRAY / 2,
                        SYNTH_DATA_ARRAY / 3,
                    ),
                    axis=0,
                ),
                [0.5, 0.25, 0.1],
                [0.3, 0.2, 0.1],
            ),
            {
                'frequency': [80, 160, 240],
                'lifetime': 4,
                'skip_axis': 0,
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
        ),  # multiple harmonics with skip_axis
    ],
)
def test_phasor_calibrate(args, kwargs, expected):
    """Test `phasor_calibrate` function with various inputs."""
    result = phasor_calibrate(*args, **kwargs)
    assert_almost_equal(result, expected)


def test_phasor_calibrate_exceptions():
    """Test exceptions in `phasor_calibrate` function."""
    with pytest.raises(ValueError):
        phasor_calibrate(0, 0, [0], [0, 0], frequency=1, lifetime=1)
    with pytest.raises(ValueError):
        phasor_calibrate([0], [0, 0], 0, 0, frequency=1, lifetime=1)


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
    # donor_freting
    assert_allclose(
        phasor_from_fret_donor(
            80, 4.2, fret_efficiency=[0.0, 0.3, 1.0], donor_freting=0.9
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
            donor_freting=0.9,
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
            donor_freting=0.9,
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
            donor_freting=0.9,
            donor_bleedthrough=0.1,
            acceptor_bleedthrough=0.1,
            acceptor_background=0.1,
            background_real=0.11,
            background_imag=0.12,
        ),
        [[0.199564, 0.057723, 0.286733], [0.322489, 0.310325, 0.429246]],
        atol=1e-3,
    )


def test_frequency_from_lifetime():
    """Test frequency_from_lifetime function."""
    assert isinstance(frequency_from_lifetime(1.0), float)
    assert frequency_from_lifetime(1.0) == pytest.approx(186.015665)
    assert_allclose(
        frequency_from_lifetime([4.0, 1.0]), [46.503916, 186.015665], atol=1e-3
    )


def test_frequency_to_lifetime():
    """Test frequency_to_lifetime function."""
    assert isinstance(frequency_to_lifetime(186.015665), float)
    assert frequency_to_lifetime(186.015665) == pytest.approx(1.0)
    assert_allclose(
        frequency_to_lifetime([46.503916, 186.015665]), [4.0, 1.0], atol=1e-3
    )


def test_fraction_to_amplitude():
    """Test fraction_to_amplitude function."""
    # assert isinstance(fraction_to_amplitude(1.0, 1.0), float)
    assert_allclose(fraction_to_amplitude(1.0, 1.0), 1.0, atol=1e-3)
    assert_allclose(
        fraction_to_amplitude([4.0, 1.0], [1.6, 0.4]),
        [0.2, 0.2],
        atol=1e-3,
    )
    assert_allclose(
        fraction_to_amplitude([[4.0], [1.0]], [[1.6], [0.4]], axis=0),
        [[0.2], [0.2]],
        atol=1e-3,
    )
    assert_allclose(
        fraction_to_amplitude([4.0, 1.0], [1.6, 0.0]),
        [0.25, 0.0],
        atol=1e-3,
    )
    with pytest.warns(RuntimeWarning):
        assert_allclose(
            fraction_to_amplitude([4.0, 0.0], [1.6, 0.4]),
            [0.2, numpy.inf],
            atol=1e-3,
        )
    with pytest.warns(RuntimeWarning):
        assert_allclose(
            fraction_to_amplitude([4.0, 1.0], [0.0, 0.0]),
            [numpy.nan, numpy.nan],
            atol=1e-3,
        )


def test_fraction_from_amplitude():
    """Test fraction_from_amplitude function."""
    # assert isinstance(fraction_from_amplitude(1.0, 1.0), float)
    assert_allclose(fraction_from_amplitude(1.0, 1.0), 1.0, atol=1e-3)
    assert_allclose(
        fraction_from_amplitude([4.0, 1.0], [0.4, 0.4]),
        [0.8, 0.2],
        atol=1e-3,
    )
    assert_allclose(
        fraction_from_amplitude([[4.0], [1.0]], [[0.4], [0.4]], axis=0),
        [[0.8], [0.2]],
        atol=1e-3,
    )
    assert_allclose(
        fraction_from_amplitude([4.0, 1.0], [0.5, 0.0]),
        [1.0, 0.0],
        atol=1e-3,
    )
    assert_allclose(
        fraction_from_amplitude([4.0, 0.0], [0.4, 10.0]),
        [1.0, 0.0],
        atol=1e-3,
    )
    with pytest.warns(RuntimeWarning):
        assert_allclose(
            fraction_from_amplitude([0.0, 0.0], [0.4, 0.4]),
            [numpy.nan, numpy.nan],
            atol=1e-3,
        )
    with pytest.warns(RuntimeWarning):
        assert_allclose(
            fraction_from_amplitude([4.0, 1.0], [0.0, 0.0]),
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

    real, imag = phasor_from_lifetime(
        frequency=[40, 80, 160],
        lifetime=[[0.5, 4.0], [1.0, 8.0]],
        fraction=[[0.4, 0.6], [0.6, 0.4]],
    )

    assert_allclose(
        real,
        [[0.69219, 0.64368], [0.49522, 0.50228], [0.35426, 0.30450]],
        atol=1e-3,
    )
    assert_allclose(
        imag,
        [[0.34948, 0.301328], [0.333795, 0.33444], [0.301026, 0.348974]],
        atol=1e-3,
    )

    x, y, transformation_matrix = phasor_to_principal_plane(real, imag)
    assert_allclose(x, [0.53084, 0.430805], atol=1e-4)
    assert_allclose(y, [0.079848, 0.059642], atol=1e-4)
    assert_allclose(
        transformation_matrix,
        [
            [0.443737, 0.083286, 0.472978, 0.561342, -0.006637, -0.594889],
            [0.436153, -0.795182, 0.359029, -0.165872, -0.002176, 0.342964],
        ],
        atol=1e-4,
    )

    with pytest.raises(ValueError):
        phasor_to_principal_plane([0.0, 1.0], [0.0])


def test_parse_skip_axis():
    """Test _parse_skip_axis function."""
    assert _parse_skip_axis(None, 0) == ((), ())
    assert _parse_skip_axis(None, 1) == ((), (0,))
    assert _parse_skip_axis((), 1) == ((), (0,))
    assert _parse_skip_axis(0, 1) == ((0,), ())
    assert _parse_skip_axis(0, 2) == ((0,), (1,))
    assert _parse_skip_axis(-1, 2) == ((1,), (0,))
    assert _parse_skip_axis((1, -2), 5) == ((1, 3), (0, 2, 4))
    with pytest.raises(ValueError):
        _parse_skip_axis(0, -1)
    with pytest.raises(IndexError):
        _parse_skip_axis(0, 0)
    with pytest.raises(IndexError):
        _parse_skip_axis(1, 1)
    with pytest.raises(IndexError):
        _parse_skip_axis(-2, 1)
