# Copyright (c) PhasorPy Contributors
# SPDX-License-Identifier: MIT
# See LICENSE.txt file in the project root for details.

"""Test LifetimePlots class."""

import numpy
import pytest
from matplotlib import pyplot
from numpy.testing import assert_allclose

from phasorpy.plot import LifetimePlots

INTERACTIVE = False  # enable for interactive plotting


def test_lifetime_plots() -> None:
    """Test LifetimePlots."""
    plot = LifetimePlots(frequency=60.0, lifetime=4.2)
    if INTERACTIVE:
        plot.show()
    pyplot.close()


def test_lifetime_plots_one() -> None:
    """Test LifetimePlots interactive with one component."""
    plot = LifetimePlots(
        frequency=100.0,
        lifetime=0.1,
        interactive=True,
    )
    assert len(plot._lifetime_sliders) == 1
    assert len(plot._fraction_sliders) == 0
    plot._frequency_slider.set_val(80.0)
    plot._lifetime_sliders[0].set_val(4.2)
    if INTERACTIVE:
        plot.show()
    pyplot.close()


def test_lifetime_plots_two() -> None:
    """Test LifetimePlots interactive with two components."""
    plot = LifetimePlots(
        frequency=100.0,
        lifetime=[0.1, 0.1],
        interactive=True,
    )
    assert len(plot._lifetime_sliders) == 2
    assert len(plot._fraction_sliders) == 1
    plot._frequency_slider.set_val(80.0)
    plot._lifetime_sliders[0].set_val(4.2)
    plot._lifetime_sliders[1].set_val(1.0)
    plot._fraction_sliders[0].set_val(0.7)
    if INTERACTIVE:
        plot.show()
    pyplot.close()


def test_lifetime_plots_three() -> None:
    """Test LifetimePlots interactive with three components."""
    plot = LifetimePlots(
        frequency=60.0,
        lifetime=[0.1, 0.1, 0.1],
        fraction=[1 / 3, 1 / 3, 1 / 3],
        frequency_range=(10.0, 1000.0, 10.0),
        lifetime_range=(0.1, 10.0, 0.1),
        interactive=True,
        dpi=150.0,
    )
    assert len(plot._lifetime_sliders) == 3
    assert len(plot._fraction_sliders) == 3
    plot._frequency_slider.set_val(80.0)
    plot._lifetime_sliders[0].set_val(8.0)
    plot._lifetime_sliders[1].set_val(4.0)
    plot._lifetime_sliders[2].set_val(0.0)
    plot._fraction_sliders[0].set_val(0.5)
    plot._fraction_sliders[1].set_val(0.3)
    plot._fraction_sliders[2].set_val(0.2)
    if INTERACTIVE:
        plot.show()
    pyplot.close()


def test_lifetime_plots_four() -> None:
    """Test LifetimePlots interactive with four components."""
    plot = LifetimePlots(
        frequency=None,
        lifetime=[8, 4, 2, 1],
        fraction=[0, 0, 0, 0],
        interactive=True,
    )
    assert len(plot._lifetime_sliders) == 4
    assert len(plot._fraction_sliders) == 4
    assert plot._fraction_sliders[0].val == 0.25
    assert plot._frequency_slider.val == 50.0
    if INTERACTIVE:
        plot.show()
    pyplot.close()


def test_lifetime_plots_apparent_lifetime() -> None:
    """Test LifetimePlots apparent and normal lifetimes."""
    plot = LifetimePlots(frequency=60.0, lifetime=4.2, interactive=True)
    assert plot._apparent_points is None
    assert plot._phase_lifetime_line is not None
    assert plot._modulation_lifetime_arc is not None
    plot._lifetime_sliders[0].set_val(2.0)
    pyplot.close()

    plot = LifetimePlots(frequency=60.0, lifetime=[0.4, 4.0], apparent=False)
    assert plot._apparent_points is None
    assert plot._phase_lifetime_line is None
    assert plot._modulation_lifetime_arc is None
    pyplot.close()

    # disabled by default for three or more components
    plot = LifetimePlots(frequency=60.0, lifetime=[0.4, 1.0, 4.0])
    assert plot._apparent_points is None
    assert plot._phase_lifetime_line is None
    pyplot.close()

    plot = LifetimePlots(
        frequency=60.0, lifetime=[0.4, 1.0, 4.0], apparent=True
    )
    assert plot._apparent_points is not None
    pyplot.close()

    plot = LifetimePlots(
        frequency=60.0,
        lifetime=[4.0, 4.0],
        fraction=[0.5, 0.5],
        interactive=True,
    )
    plot._frequency_slider.set_val(80.0)
    plot._lifetime_sliders[0].set_val(0.4)
    plot._fraction_sliders[0].set_val(0.6)
    assert plot._apparent_points is not None
    assert plot._phase_lifetime_line is not None
    assert plot._modulation_lifetime_arc is not None
    real, imag = numpy.asarray(
        plot._phasor_points[0].get_data(), dtype=numpy.float64
    )[:, 0]
    points_real, points_imag = numpy.asarray(
        plot._apparent_points.get_data(), dtype=numpy.float64
    )
    assert len(points_real) == 3
    # markers on universal semicircle
    assert_allclose((points_real - 0.5) ** 2 + points_imag**2, 0.25)
    # phase line from origin through mixture to phase lifetime
    x, y = numpy.asarray(
        plot._phase_lifetime_line.get_data(), dtype=numpy.float64
    )
    assert_allclose([x[0], y[0]], [0.0, 0.0])
    assert_allclose([x[1], y[1]], [points_real[0], points_imag[0]])
    assert_allclose(y[1] * real, x[1] * imag, atol=1e-12)
    # modulation arc from real axis to modulation lifetime
    x, y = numpy.asarray(
        plot._modulation_lifetime_arc.get_data(), dtype=numpy.float64
    )
    assert_allclose([x[0], y[0]], [numpy.hypot(real, imag), 0.0])
    assert_allclose([x[-1], y[-1]], [points_real[1], points_imag[1]])
    assert_allclose(numpy.hypot(x, y), numpy.hypot(real, imag))
    # normal lifetime on line through semicircle center and mixture
    assert_allclose(
        points_imag[2] * (real - 0.5),
        (points_real[2] - 0.5) * imag,
        atol=1e-12,
    )
    if INTERACTIVE:
        plot.show()
    pyplot.close()


def test_lifetime_plots_exceptions() -> None:
    """Test LifetimePlots exceptions."""
    with pytest.raises(ValueError):
        LifetimePlots(frequency=60.0, lifetime=[1.0] * 7)

    with pytest.raises(ValueError):
        LifetimePlots(frequency=60.0, lifetime=4.2, fraction=[0.5, 0.5])


def test_lifetime_plots_format_coord_frequency() -> None:
    """Test phasor plot frequency follows frequency slider."""
    plot = LifetimePlots(frequency=60.0, lifetime=4.2, interactive=True)
    plot._frequency_slider.set_val(80.0)
    assert plot._phasorplot._frequency == 80.0
    pyplot.close()


def test_lifetime_plots_negative_fraction() -> None:
    """Test negative fractions are clipped before normalization."""
    plot = LifetimePlots(frequency=60.0, lifetime=4.2)
    fractions = plot._calculate(60.0, [1.0, 4.0], [-1.0, 2.0])[2]
    assert_allclose(fractions, [0.0, 1.0])
    pyplot.close()
