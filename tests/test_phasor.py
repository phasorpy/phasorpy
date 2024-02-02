"""Tests for the phasorpy.phasor module."""

import copy

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
    phasor_to_polar,
    polar_from_reference,
    polar_from_reference_phasor,
)

SYNTH_DATA_ARRAY = numpy.array([[50, 1], [1, 1]])
SYNTH_DATA_LIST = [1, 2, 4]
SYNTH_PHI = numpy.array([[0.5, 0.5], [0.5, 0.5]])
SYNTH_MOD = numpy.array([[2, 2], [2, 2]])


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
    """Test `phasor_to_polar` function with scalar, list and array inputs"""
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
    """Test `polar_from_reference` function with scalar, list and
    array inputs"""
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
    """Test `polar_from_reference_phasor` function with scalar, list and
    array inputs"""
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
    """Test `phasor_calibrate` function with scalar, list and array inputs"""
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
    """Test `phasor_center` function with scalar, list and array inputs with
    all methods available"""
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
        # two lifetimes
        ((80.0, [0.0, 1e9], [0.5, 0.5]), {}, (0.5, 0.0)),
        ((80.0, [0.0, 1e9], [0.6, 0.4]), {}, (0.6, 0.0)),
        ((80.0, [3.9788735, 0.9947183], [0.0, 1.0]), {}, (0.8, 0.4)),
        ((80.0, [3.9788735, 0.9947183], [1.0, 0.0]), {}, (0.2, 0.4)),
        ((80.0, [3.9788735, 0.9947183], [0.5, 0.5]), {}, (0.5, 0.4)),
        ((80.0, [3.9788735, 0.9947183], [0.25, 0.75]), {}, (0.65, 0.4)),
        # three lifetimes
        ((80.0, [0.0, 1.9894368, 1e9]), {}, (0.5, 0.5 / 3)),
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
        # preexponential amplitudes
        ((80.0, 0.0), {'is_preexp': True}, (numpy.nan, numpy.nan)),  # ?
        ((80.0, 1e9), {'is_preexp': True}, (0.0, 0.0)),
        ((80.0, 3.9788735), {'is_preexp': True}, (0.2, 0.4)),
        ((80.0, [0.0, 1e9], [0.5, 0.5]), {'is_preexp': True}, (0.0, 0.0)),
        (
            (80.0, [3.9788735, 0.9947183], [0.0, 1.0]),
            {'is_preexp': True},
            (0.8, 0.4),
        ),
        (
            (80.0, [3.9788735, 0.9947183], [1.0, 0.0]),
            {'is_preexp': True},
            (0.2, 0.4),
        ),
        (
            (80.0, [3.9788735, 0.9947183], [0.5, 0.5]),
            {'is_preexp': True},
            (0.32, 0.4),
        ),
        (
            (80.0, [3.9788735, 0.9947183], [0.25, 0.75]),
            {'is_preexp': True},
            (0.457143, 0.4),
        ),
    ],
)
def test_phasor_from_lifetime(args, kwargs, expected):
    """Test `phasor_from_lifetime` function."""
    for actual, desired in zip(
        phasor_from_lifetime(*args, **kwargs), expected
    ):
        assert_allclose(actual, desired, atol=1e-6)


def test_phasor_from_lifetime_exceptions():
    """Test exceptions in `phasor_from_lifetime` function."""
    with pytest.raises(ValueError):
        phasor_from_lifetime(80.0, [[0.0]])
    with pytest.raises(ValueError):
        phasor_from_lifetime(80.0, 0.0, [])
    with pytest.raises(ValueError):
        phasor_from_lifetime(80.0, [0.0, 1e9], [1.0])


def test_phasor_from_lifetime_modify():
    """Test `phasor_from_lifetime` function does not modify input."""
    frequency = 80.0
    lifetime = numpy.array([0.0, 1.9894368, 1e9], dtype=numpy.float64)
    fraction = numpy.array([1.0, 1.0, 1.0], dtype=numpy.float64)
    real, imag = phasor_from_lifetime(frequency, lifetime, fraction=fraction)
    assert_allclose(real, 0.5)
    assert_allclose(imag, 0.5 / 3)
    assert_array_equal(frequency, 80.0)  # for future revisions
    assert_array_equal(lifetime, [0.0, 1.9894368, 1e9])
    assert_array_equal(fraction, [1.0, 1.0, 1.0])
