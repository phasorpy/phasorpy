"""Test the phasorpy.experimental module."""

import numpy
import pytest
from numpy.testing import assert_allclose

from phasorpy.experimental import (
    anscombe_transform,
    anscombe_transform_inverse,
    signal_from_dho,
)

rng = numpy.random.default_rng(42)


@pytest.mark.parametrize('dtype', ['float32', 'uint16'])
def test_anscombe_transform(dtype):
    """Test anscombe_transform and inverse functions."""
    x = rng.poisson(10, 100000).astype(dtype)
    if dtype == 'float32':
        x[0] = numpy.nan
    z = anscombe_transform(x)
    assert_allclose(numpy.std(z[1:]), 1.0, atol=0.01)

    x2 = anscombe_transform_inverse(z)
    assert_allclose(x2[1:], x[1:], atol=0.01)

    x3 = anscombe_transform_inverse(z, approx=True)
    assert_allclose(x3[1:], x[1:], atol=1.0)

    if dtype == 'float32':
        assert numpy.isnan(z[0])
        assert numpy.isnan(x2[0])
        assert numpy.isnan(x3[0])


def test_signal_from_dho():
    """Test signal_from_dho function."""
    wavelength = numpy.linspace(450, 650, 100)
    base = signal_from_dho(
        wavelength,
        origin=518,
        sigma=500,
        hr_factor=0.38,
        vib_frequency=1200,
    )
    emission = signal_from_dho(
        wavelength,
        origin=518,
        sigma=500,
        hr_factor=0.38,
        vib_frequency=1200,
        offset=5.0,
        scale=100.0,
    )
    absorption = signal_from_dho(
        wavelength,
        origin=518,
        sigma=500,
        hr_factor=0.38,
        vib_frequency=1200,
        absorption=True,
    )

    assert emission.shape == wavelength.shape
    assert emission.dtype == numpy.float64
    assert numpy.all(numpy.isfinite(emission))
    assert numpy.all(emission >= 5.0)

    assert_allclose(emission, base * 100.0 + 5.0)

    assert absorption.shape == wavelength.shape
    assert numpy.all(numpy.isfinite(absorption))
    assert not numpy.allclose(base, absorption)

    invalid_sigma = signal_from_dho(
        wavelength,
        origin=518,
        sigma=0.0,
        hr_factor=0.38,
        vib_frequency=1200,
    )
    assert numpy.all(numpy.isnan(invalid_sigma))

    zero_vib_emission = signal_from_dho(
        wavelength,
        origin=518,
        sigma=500,
        hr_factor=0.38,
        vib_frequency=0.0,
    )
    zero_vib_absorption = signal_from_dho(
        wavelength,
        origin=518,
        sigma=500,
        hr_factor=0.38,
        vib_frequency=0.0,
        absorption=True,
    )
    assert numpy.all(numpy.isfinite(zero_vib_emission))
    assert_allclose(zero_vib_emission, zero_vib_absorption)

    # ufunc-style behavior: broadcasting over array-like parameters
    wavelength_2d = wavelength.reshape(1, -1)
    sigma_2d = numpy.array([[450.0], [550.0]])
    broadcasted = signal_from_dho(
        wavelength_2d,
        origin=518,
        sigma=sigma_2d,
        hr_factor=0.38,
        vib_frequency=1200,
    )
    assert broadcasted.shape == (2, wavelength.size)
    assert numpy.all(numpy.isfinite(broadcasted))


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
