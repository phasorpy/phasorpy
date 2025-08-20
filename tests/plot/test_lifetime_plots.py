"""Test LifetimePlots class."""

import pytest
from matplotlib import pyplot

from phasorpy.plot import LifetimePlots

INTERACTIVE = False  # enable for interactive plotting


def test_lifetime_plots():
    """Test LifetimePlots."""
    plot = LifetimePlots(frequency=60.0, lifetime=4.2)
    if INTERACTIVE:
        plot.show()
    pyplot.close()


def test_lifetime_plots_one():
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


def test_lifetime_plots_two():
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


def test_lifetime_plots_three():
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


def test_lifetime_plots_four():
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


def test_lifetime_plots_exceptions():
    """Test LifetimePlots exceptions."""
    with pytest.raises(ValueError):
        LifetimePlots(frequency=60.0, lifetime=[1.0] * 7)

    with pytest.raises(ValueError):
        LifetimePlots(frequency=60.0, lifetime=4.2, fraction=[0.5, 0.5])


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
