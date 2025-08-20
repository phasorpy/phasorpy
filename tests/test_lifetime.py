"""Tests for the phasorpy.lifetime module."""

import copy
import math

import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
)

from phasorpy.lifetime import (
    lifetime_fraction_from_amplitude,
    lifetime_fraction_to_amplitude,
    lifetime_from_frequency,
    lifetime_to_frequency,
    lifetime_to_signal,
    phasor_at_harmonic,
    phasor_calibrate,
    phasor_from_apparent_lifetime,
    phasor_from_fret_acceptor,
    phasor_from_fret_donor,
    phasor_from_lifetime,
    phasor_semicircle,
    phasor_semicircle_intersect,
    phasor_to_apparent_lifetime,
    phasor_to_lifetime_search,
    phasor_to_normal_lifetime,
    polar_from_apparent_lifetime,
    polar_from_reference,
    polar_from_reference_phasor,
    polar_to_apparent_lifetime,
)
from phasorpy.phasor import phasor_from_polar, phasor_to_polar

NAN = math.nan
SYNTH_DATA_ARRAY = numpy.array([[50, 1], [1, 1]])
SYNTH_DATA_ARRAY_3D = numpy.stack(
    [
        SYNTH_DATA_ARRAY,
        SYNTH_DATA_ARRAY / 2,
        SYNTH_DATA_ARRAY / 3,
    ],
    axis=0,
)
SYNTH_DATA_NAN = numpy.array([[50, NAN], [1, 1]])
SYNTH_DATA_LIST = [1, 2, 4]
SYNTH_PHI = numpy.array([[0.5, 0.5], [0.5, 0.5]])
SYNTH_MOD = numpy.array([[2, 2], [2, 2]])

numpy.random.seed(42)


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
            [0.2, 0.2, NAN], [0.25, 0.0, 0.25], 0.6, 0.25
        ),
        (
            [0.066, NAN, NAN],
            [0.25, NAN, NAN],
            [0.933, 0.817, NAN],
            [0.25, 0.386, NAN],
        ),
        atol=1e-3,
    )
    # reverse order
    assert_allclose(
        phasor_semicircle_intersect(
            0.6, 0.25, [0.2, 0.2, NAN], [0.25, 0.0, 0.25]
        ),
        (
            [0.933, NAN, NAN],
            [0.25, NAN, NAN],
            [0.066, 0.817, NAN],
            [0.25, 0.386, NAN],
        ),
        atol=1e-3,
    )
    # no intersection
    assert_allclose(
        phasor_semicircle_intersect(0.1, -0.1, 0.9, -0.1),
        (NAN, NAN, NAN, NAN),
        atol=1e-3,
    )
    # no line
    assert_allclose(
        phasor_semicircle_intersect(0.25, 0.25, 0.25, 0.25),
        (NAN, NAN, NAN, NAN),
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
        ((80.0, [0.0, 1.9894368, 1e9], [0, 0, 0]), {}, (NAN, NAN)),
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
            [NAN, NAN],
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
            [NAN, NAN],
            atol=1e-3,
        )
    with pytest.warns(RuntimeWarning):
        assert_allclose(
            lifetime_fraction_from_amplitude([4.0, 1.0], [0.0, 0.0]),
            [NAN, NAN],
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


def test_phasor_component_search_exceptions():
    """Test exceptions in phasor_to_lifetime_search function."""
    real = [0.1, 0.2]
    imag = [0.4, 0.3]
    frequency = 60.0
    phasor_to_lifetime_search(real, imag, frequency)

    # shape mismatch
    with pytest.raises(ValueError):
        phasor_to_lifetime_search(real, imag[0], frequency)

    # no harmonics
    with pytest.raises(ValueError):
        phasor_to_lifetime_search(real[0], imag[0], frequency)

    # number of components < 2
    # with pytest.raises(ValueError):
    #     phasor_to_lifetime_search(real, imag, 1, frequency)

    # number of components does not match harmonics
    # with pytest.raises(ValueError):
    #     phasor_to_lifetime_search(real, imag, 3, frequency)

    # samples < 1
    with pytest.raises(ValueError):
        phasor_to_lifetime_search(
            real, imag, frequency, lifetime_range=(0, 1, 2)
        )

    # samples < 3 for 3 components
    # with pytest.raises(ValueError):
    #     phasor_to_lifetime_search(
    #         [0.1, 0.2, 0.3],
    #         [0.1, 0.2, 0.3],
    #         3,
    #         frequency,
    #         lifetime_range=(0, 1, 0.5),
    #     )

    # dtype not float
    with pytest.raises(ValueError):
        phasor_to_lifetime_search(real, imag, frequency, dtype=numpy.int32)

    # too many components
    # with pytest.raises(ValueError):
    #     phasor_to_lifetime_search(
    #         [0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], 4, frequency
    #     )


@pytest.mark.parametrize(
    'real, imag, expected_real, expected_imag, expected_fraction',
    [
        # inside semicircle
        (
            *phasor_from_lifetime([80, 160], [0.5, 4.2], [0.3, 0.7]),
            *phasor_from_lifetime(80, [0.5, 4.2]),
            [0.3, 0.7],
        ),
        # infinite lifetime
        ([0, 0], [0, 0], [1, 0], [0, 0], [0, 1]),
        # zero lifetime
        ([1, 1], [0, 0], [1, 1], [0, 0], [0, 1]),
        # on semicircle
        ([0.5, 0.2], [0.5, 0.4], [1, 0.5], [0, 0.5], [0, 1]),
        # outside semicircle
        ([0.5, 0.2], [0.6, 10], [NAN, NAN], [NAN, NAN], [NAN, NAN]),
        ([0.5, 0.2], [0.5, -1], [NAN, NAN], [NAN, NAN], [NAN, NAN]),
        # NAN
        ([NAN, 0], [0, 0], [NAN, NAN], [NAN, NAN], [NAN, NAN]),
        ([0, 0], [0, NAN], [NAN, NAN], [NAN, NAN], [NAN, NAN]),
    ],
)
def test_phasor_to_lifetime_search_two(
    real, imag, expected_real, expected_imag, expected_fraction
):
    """Test phasor_to_lifetime_search function with two components."""
    expected_lifetime = phasor_to_normal_lifetime(
        expected_real, expected_imag, frequency=80.0
    )
    lifetime, fraction = phasor_to_lifetime_search(real, imag, 80.0)
    assert_allclose(lifetime, expected_lifetime, atol=1e-6)
    assert_allclose(fraction, expected_fraction, atol=1e-6)


def test_phasor_to_lifetime_search_two_range():
    """Test phasor_to_lifetime_search function, two components with range."""
    lifetime = [0.5, 4.2]
    fraction = [0.3, 0.7]
    frequency = 80.0
    real, imag = phasor_from_lifetime(
        [frequency, frequency * 2], lifetime, fraction
    )
    phase_lifetime = phasor_to_apparent_lifetime(real[0], imag[0], frequency)[
        0
    ]
    phase_lifetime = numpy.round(phase_lifetime + 0.01, 2)
    normal_lifetime = phasor_to_normal_lifetime(real[0], imag[0], frequency)
    normal_lifetime = numpy.round(normal_lifetime - 0.01, 2)
    print(phase_lifetime, normal_lifetime)

    # lower lifetime is out of range
    lifetimes, fractions = phasor_to_lifetime_search(
        real,
        imag,
        frequency,
        lifetime_range=(phase_lifetime, lifetime[1] + 1.0, 0.01),
    )
    assert_allclose(lifetimes, lifetime, atol=1e-6)
    assert_allclose(fractions, fraction, atol=1e-6)

    # upper lifetime is out of range
    lifetimes, fractions = phasor_to_lifetime_search(
        real,
        imag,
        frequency,
        lifetime_range=(lifetime[0] - 0.1, normal_lifetime, 0.01),
    )
    assert_allclose(lifetimes, lifetime, atol=1e-6)
    assert_allclose(fractions, fraction, atol=1e-6)

    # exact upper range
    lifetimes, fractions = phasor_to_lifetime_search(
        real,
        imag,
        frequency,
        lifetime_range=(phase_lifetime, lifetime[1] + 0.01, 0.01),
    )
    assert_allclose(lifetimes, lifetime, atol=1e-6)
    assert_allclose(fractions, fraction, atol=1e-6)

    # both lifetimes are out of range
    lifetimes, fractions = phasor_to_lifetime_search(
        real,
        imag,
        frequency,
        lifetime_range=(lifetime[0] + 0.1, lifetime[1] - 0.1, 0.01),
    )
    with pytest.raises(AssertionError):
        assert_allclose(lifetimes, lifetime, atol=1e-6)

    # both lifetimes are out of range, no solution
    lifetimes, fractions = phasor_to_lifetime_search(
        real,
        imag,
        frequency,
        lifetime_range=(phase_lifetime, normal_lifetime, 0.01),
    )
    assert_allclose(lifetimes, [NAN, NAN], atol=1e-6)
    assert_allclose(fractions, [NAN, NAN], atol=1e-6)


@pytest.mark.parametrize('exact', [True, False])
def test_phasor_to_lifetime_search_two_distribution(exact):
    """Test phasor_to_lifetime_search function with two components."""
    # test that two lifetime components can be recovered from a distribution
    shape = (256, 256)
    frequency = 60.0
    lifetime = [0.5, 4.2]
    fraction = numpy.empty((*shape, 2))
    fraction[..., 0] = numpy.random.normal(0.3, 0.01, shape)
    fraction[..., 1] = 1.0 - fraction[..., 0]
    fraction = numpy.clip(fraction, 0.0, 1.0)

    real, imag = phasor_from_lifetime(
        [frequency, 2 * frequency], lifetime, fraction.reshape(-1, 2)
    )
    real = real.reshape(2, *shape)
    imag = imag.reshape(2, *shape)
    if not exact:
        # add noise to the imaginary parts
        imag += numpy.random.normal(0.0, 0.005, (2, *shape))
        dtype = 'float32'
        atol = 5e-2
    else:
        dtype = 'float64'
        atol = 1e-3

    lifetimes, fractions = phasor_to_lifetime_search(
        real,
        imag,
        frequency=frequency,
        lifetime_range=(0.4, 2.0, 0.01),
        dtype=dtype,
        num_threads=2,
    )

    component_real, component_imag = phasor_from_lifetime(
        frequency, lifetimes.reshape(-1)
    )
    component_real = component_real.reshape(2, *shape)
    component_imag = component_imag.reshape(2, *shape)
    # _plot(frequency, real, imag, component_real, component_imag)

    assert_allclose(lifetimes[0].mean(), lifetime[0], atol=atol)
    assert_allclose(lifetimes[1].mean(), lifetime[1], atol=atol)
    assert_allclose(fractions[0].mean(), 0.3, atol=1e-3)
    assert_allclose(fractions[1].mean(), 0.7, atol=1e-3)


def _plot(frequency, real, imag, component_real=None, component_imag=None):
    # helper function to visualize lifetime component distribution results
    from phasorpy.plot import PhasorPlot

    pp = PhasorPlot(frequency=frequency, allquadrants=False)
    if real.size > 100:
        for i in range(real.shape[0]):
            pp.hist2d(real[i], imag[i], cmap='Greys')
    else:
        pp.plot(real, imag)
    if component_real is not None and component_imag is not None:
        cmap = 'Reds', 'Greens', 'Blues'
        for i in range(component_real.shape[0]):
            pp.hist2d(component_real[i], component_imag[i], cmap=cmap[i])
            pp.plot(
                numpy.nanmean(component_real[i]),
                numpy.nanmean(component_imag[i]),
            )
    pp.show()


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type, unreachable"
