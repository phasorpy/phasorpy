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

from phasorpy.phasor import (
    phasor_calibrate,
    phasor_center,
    phasor_from_lifetime,
    phasor_from_polar,
    phasor_from_signal,
    phasor_from_signal_fft,
    phasor_semicircle,
    phasor_to_polar,
    polar_from_reference,
    polar_from_reference_phasor,
)

SYNTH_DATA_ARRAY = numpy.array([[50, 1], [1, 1]])
SYNTH_DATA_LIST = [1, 2, 4]
SYNTH_PHI = numpy.array([[0.5, 0.5], [0.5, 0.5]])
SYNTH_MOD = numpy.array([[2, 2], [2, 2]])


@pytest.mark.parametrize('fft', (True, False))
def test_phasor_from_signal(fft):
    """Test `phasor_from_signal` functions."""
    func = phasor_from_signal_fft if fft else phasor_from_signal
    sample_phase = numpy.linspace(0, 2 * math.pi, 5, endpoint=False)
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
            func(signal[::-1], sample_phase=sample_phase[::-1]),
            (1.1, 0.4, 0.2),
            atol=1e-6,
        )
    assert_allclose(
        func(numpy.zeros(256)),
        (0.0, numpy.nan, numpy.nan) if fft else (0.0, 0.0, 0.0),
        atol=1e-6,
    )
    assert_allclose(
        func(numpy.cos(sample_phase)),
        (0.0, -1.125900e16, -1.244949) if fft else (0.0, 0.0, 0.0),
        atol=1e-6,
    )
    assert_allclose(func(signal, harmonic=2), (1.1, 0.0, 0.0), atol=1e-6)
    with pytest.raises(ValueError):
        func(signal, harmonic=0)
    with pytest.raises(ValueError):
        func(signal[:2])
    if not fft:
        with pytest.raises(ValueError):
            func(signal, num_threads=-1)
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
        mean, real, imag = phasor_from_signal(signal, axis=axis)
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


@pytest.mark.parametrize('harmonic', (1, 2, 3, 4, 8))
def test_phasor_from_signal_harmonic(harmonic):
    """Test `phasor_from_signal` functions harmonic parameter."""
    rng = numpy.random.default_rng(1)
    signal = rng.random((3, 33, 61, 63))
    signal += 1.1
    kwargs = dict(axis=1, harmonic=harmonic)
    mean0, real0, imag0 = phasor_from_signal(signal, **kwargs)
    mean1, real1, imag1 = phasor_from_signal_fft(signal, **kwargs)
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
    with pytest.raises(ValueError):
        phasor_semicircle(0)


def test_phasor_from_polar():
    """Test `phasor_from_polar` function."""
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
    """Test `phasor_to_polar` function with various inputs."""
    real_copy = copy.deepcopy(real)
    imag_copy = copy.deepcopy(imag)
    polar_phase, polar_modulation = phasor_to_polar(real_copy, imag_copy)
    assert_array_equal(real, real_copy)
    assert_array_equal(imag, imag_copy)
    assert_almost_equal(polar_phase, expected_phase)
    assert_almost_equal(polar_modulation, expected_modulation)


def test_phasor_to_polar_exceptions():
    """Test exceptions in `phasor_to_polar` function."""
    with pytest.raises(ValueError):
        phasor_to_polar([0], [0, 0])


@pytest.mark.parametrize(
    """measured_phase, measured_modulation,
    known_phase, known_modulation,
    expected_phase, expected_modulation""",
    [
        (2, 2, 0.2, 0.5, 1.8, 4.0),
        (-2, -2, 0.2, 0.5, -2.2, -4.0),
        (
            SYNTH_DATA_LIST,
            SYNTH_DATA_LIST,
            numpy.full(len(SYNTH_DATA_LIST), 0.2),
            numpy.full(len(SYNTH_DATA_LIST), 0.5),
            [0.8, 1.8, 3.8],
            [2.0, 4.0, 8.0],
        ),
        (
            SYNTH_DATA_ARRAY,
            SYNTH_DATA_ARRAY,
            numpy.full(SYNTH_DATA_ARRAY.shape, 0.2),
            numpy.full(SYNTH_DATA_ARRAY.shape, 0.5),
            numpy.asarray([[49.8, 0.8], [0.8, 0.8]]),
            numpy.asarray([[100.0, 2.0], [2.0, 2.0]]),
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


def test_polar_from_reference_exceptions():
    """Test exceptions in `polar_from_reference` function."""
    with pytest.raises(ValueError):
        polar_from_reference(0, 0, [0], [0, 0])
    with pytest.raises(ValueError):
        polar_from_reference([0], [0, 0], 0, 0)


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


def test_polar_from_reference_phasor_exceptions():
    """Test exceptions in `polar_from_reference_phasor` function."""
    with pytest.raises(ValueError):
        polar_from_reference_phasor(0, 0, [0], [0, 0])
    with pytest.raises(ValueError):
        polar_from_reference_phasor([0], [0, 0], 0, 0)


@pytest.mark.parametrize(
    """real, imag,
    phase0, modulation0,
    expected_real, expected_imag""",
    [
        (2, 2, None, None, 2, 2),
        (2, 2, 0, 1, 2, 2),
        (2, 2, 0.5, 2.0, 1.592628093144679, 5.428032401978303),
        (2, 2, -0.5, -2.0, -5.428032401978303, -1.592628093144679),
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
        ),  # test with phase0 and modulation0 as arrays
    ],
)
def test_phasor_calibrate(
    real,
    imag,
    phase0,
    modulation0,
    expected_real,
    expected_imag,
):
    """Test `phasor_calibrate` function with various inputs."""
    real_copy = copy.deepcopy(real)
    imag_copy = copy.deepcopy(imag)
    if phase0 is not None and modulation0 is not None:
        calibrated_real, calibrated_imag = phasor_calibrate(
            real_copy,
            imag_copy,
            phase0,
            modulation0,
        )
    else:
        calibrated_real, calibrated_imag = phasor_calibrate(
            real_copy, imag_copy
        )
    assert_array_equal(real, real_copy)
    assert_array_equal(imag, imag_copy)
    assert_almost_equal(calibrated_real, expected_real)
    assert_almost_equal(calibrated_imag, expected_imag)


def test_phasor_calibrate_exceptions():
    """Test exceptions in `phasor_calibrate` function."""
    with pytest.raises(ValueError):
        phasor_calibrate(0, 0, [0], [0, 0])
    with pytest.raises(ValueError):
        phasor_calibrate([0], [0, 0], 0, 0)


@pytest.mark.parametrize(
    """real, imag,
    skip_axes, method,
    expected_real_center, expected_imag_center""",
    [
        (1.0, 4.0, None, 'mean', 1.0, 4.0),
        (1.0, -4.0, None, 'median', 1.0, -4.0),
        (
            SYNTH_DATA_LIST,
            SYNTH_DATA_LIST,
            None,
            'mean',
            2.3333333333333335,
            2.3333333333333335,
        ),
        (SYNTH_DATA_LIST, SYNTH_DATA_LIST, None, 'median', 2.0, 2.0),
        (SYNTH_DATA_ARRAY, SYNTH_DATA_ARRAY, None, 'mean', 13.25, 13.25),
        (SYNTH_DATA_ARRAY, SYNTH_DATA_ARRAY, None, 'median', 1.0, 1.0),
        # with skip_axes
        (
            SYNTH_DATA_ARRAY,
            SYNTH_DATA_ARRAY,
            (0,),
            'mean',
            numpy.asarray([25.5, 1.0]),
            numpy.asarray([25.5, 1.0]),
        ),
        (
            SYNTH_DATA_ARRAY,
            SYNTH_DATA_ARRAY,
            (0,),
            'median',
            numpy.asarray([25.5, 1.0]),
            numpy.asarray([25.5, 1.0]),
        ),
    ],
)
def test_phasor_center(
    real,
    imag,
    skip_axes,
    method,
    expected_real_center,
    expected_imag_center,
):
    """Test `phasor_center` function with various inputs and methods."""
    real_copy = copy.deepcopy(real)
    imag_copy = copy.deepcopy(imag)
    real_center, imag_center = phasor_center(
        real_copy, imag_copy, skip_axes=skip_axes, method=method
    )
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
    result = phasor_from_lifetime(*args, **kwargs, squeeze=False)
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
