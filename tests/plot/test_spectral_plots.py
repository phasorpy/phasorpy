"""Test SpectralPlots class."""

import numpy
import pytest
from matplotlib import pyplot
from numpy.testing import assert_allclose

from phasorpy.plot import SpectralPlots

INTERACTIVE = False  # enable for interactive plotting


def test_spectral_plots() -> None:
    """Test SpectralPlots single Gaussian component, non-interactive."""
    plot = SpectralPlots(origin=518.0)
    if INTERACTIVE:
        plot.show()
    pyplot.close()


def test_spectral_plots_one() -> None:
    """Test SpectralPlots interactive, one Gaussian component."""
    plot = SpectralPlots(origin=[518.0], sigma=[30.0], interactive=True)
    assert len(plot._origin_sliders) == 1
    assert len(plot._sigma_sliders) == 1
    assert len(plot._fraction_sliders) == 0
    assert len(plot._hr_factor_sliders) == 0
    assert len(plot._vib_frequency_sliders) == 0
    plot._origin_sliders[0].set_val(540.0)
    plot._sigma_sliders[0].set_val(40.0)
    plot._samples_slider.set_val(64.0)
    plot._wavelength_slider.set_val((460.0, 620.0))
    if INTERACTIVE:
        plot.show()
    pyplot.close()


def test_spectral_plots_two() -> None:
    """Test SpectralPlots interactive, two Gaussian components."""
    plot = SpectralPlots(
        origin=[490.0, 570.0],
        sigma=[25.0, 35.0],
        fraction=[0.6, 0.4],
        interactive=True,
    )
    assert len(plot._origin_sliders) == 2
    assert len(plot._sigma_sliders) == 2
    assert len(plot._fraction_sliders) == 1
    assert len(plot._hr_factor_sliders) == 0
    assert len(plot._vib_frequency_sliders) == 0
    plot._origin_sliders[0].set_val(510.0)
    plot._origin_sliders[1].set_val(590.0)
    plot._sigma_sliders[0].set_val(30.0)
    plot._fraction_sliders[0].set_val(0.4)
    plot._wavelength_slider.set_val((440.0, 660.0))
    if INTERACTIVE:
        plot.show()
    pyplot.close()


def test_spectral_plots_dho_one() -> None:
    """Test SpectralPlots interactive, one DHO component."""
    plot = SpectralPlots(
        origin=[518.0],
        sigma=[500.0],
        dho=True,
        interactive=True,
    )
    assert len(plot._origin_sliders) == 1
    assert len(plot._sigma_sliders) == 1
    assert len(plot._fraction_sliders) == 0
    assert len(plot._hr_factor_sliders) == 1
    assert len(plot._vib_frequency_sliders) == 1
    plot._origin_sliders[0].set_val(540.0)
    plot._sigma_sliders[0].set_val(400.0)
    plot._hr_factor_sliders[0].set_val(0.6)
    plot._vib_frequency_sliders[0].set_val(1400.0)
    plot._wavelength_slider.set_val((460.0, 620.0))
    if INTERACTIVE:
        plot.show()
    pyplot.close()


def test_spectral_plots_dho_two() -> None:
    """Test SpectralPlots interactive, two DHO components."""
    plot = SpectralPlots(
        origin=[490.0, 570.0],
        sigma=[500.0, 450.0],
        fraction=[0.6, 0.4],
        dho=True,
        interactive=True,
    )
    assert len(plot._origin_sliders) == 2
    assert len(plot._sigma_sliders) == 2
    assert len(plot._fraction_sliders) == 1
    assert len(plot._hr_factor_sliders) == 2
    assert len(plot._vib_frequency_sliders) == 2
    plot._origin_sliders[0].set_val(510.0)
    plot._origin_sliders[1].set_val(590.0)
    plot._hr_factor_sliders[0].set_val(0.5)
    plot._hr_factor_sliders[1].set_val(0.3)
    plot._vib_frequency_sliders[0].set_val(1300.0)
    plot._vib_frequency_sliders[1].set_val(1100.0)
    plot._fraction_sliders[0].set_val(0.4)
    if INTERACTIVE:
        plot.show()
    pyplot.close()


def test_spectral_plots_wavelength_slider() -> None:
    """Test SpectralPlots wavelength RangeSlider."""
    plot = SpectralPlots(
        origin=[490.0, 570.0],
        interactive=True,
        wavelength_range=(450.0, 650.0),
    )
    plot._wavelength_slider.set_val((400.0, 700.0))
    plot._wavelength_slider.set_val((500.0, 600.0))
    pyplot.close()


def test_spectral_plots_polygon() -> None:
    """Test samples polygon update for <=16 and >16 samples."""
    plot = SpectralPlots(origin=[518.0], interactive=True)
    assert plot._samples_polygon is not None

    plot._samples_slider.set_val(16.0)
    assert plot._samples_polygon.get_visible()
    polygon_x, polygon_y = plot._samples_polygon.get_data()
    polygon_x = numpy.asarray(polygon_x)
    polygon_y = numpy.asarray(polygon_y)
    assert len(polygon_x) == 17
    assert len(polygon_y) == 17
    assert_allclose(polygon_x[0], polygon_x[-1])
    assert_allclose(polygon_y[0], polygon_y[-1])

    plot._samples_slider.set_val(17.0)
    assert not plot._samples_polygon.get_visible()
    pyplot.close(plot._spectrum_plot.figure)  # type: ignore[arg-type]


def test_spectral_plots_zero_sum_fractions() -> None:
    """Test zero-sum fractions fallback to uniform in __init__."""
    plot_default = SpectralPlots(
        origin=[490.0, 570.0, 630.0],
        sigma=[25.0, 35.0, 30.0],
    )
    plot_zero = SpectralPlots(
        origin=[490.0, 570.0, 630.0],
        sigma=[25.0, 35.0, 30.0],
        fraction=[0.0, 0.0, 0.0],
    )

    assert_allclose(
        numpy.asarray(plot_zero._spectrum_line.get_ydata()),
        numpy.asarray(plot_default._spectrum_line.get_ydata()),
    )
    assert_allclose(
        numpy.asarray(plot_zero._phasor_point.get_xdata()),
        numpy.asarray(plot_default._phasor_point.get_xdata()),
    )
    assert_allclose(
        numpy.asarray(plot_zero._phasor_point.get_ydata()),
        numpy.asarray(plot_default._phasor_point.get_ydata()),
    )
    pyplot.close(plot_zero._spectrum_plot.figure)  # type: ignore[arg-type]
    pyplot.close(plot_default._spectrum_plot.figure)  # type: ignore[arg-type]


def test_spectral_plots_raw_fractions() -> None:
    """Test 3-component slider branch uses raw fractions."""
    plot = SpectralPlots(
        origin=[470.0, 540.0, 620.0],
        sigma=[20.0, 28.0, 35.0],
        interactive=True,
    )
    assert len(plot._fraction_sliders) == 3

    raw_fractions = numpy.asarray([0.2, 0.7, 0.5])
    for slider, value in zip(
        plot._fraction_sliders, raw_fractions, strict=True
    ):
        slider.set_val(float(value))

    expected = plot._calculate(
        plot._wl_min,
        plot._wl_max,
        numpy.asarray([s.val for s in plot._origin_sliders]),
        numpy.asarray([s.val for s in plot._sigma_sliders]),
        raw_fractions,
        numpy.empty(3),
        numpy.empty(3),
    )
    expected_signal, expected_real, expected_imag = (
        expected[1],
        expected[5],
        expected[6],
    )

    assert_allclose(
        numpy.asarray(plot._spectrum_line.get_ydata()), expected_signal
    )
    assert_allclose(
        numpy.asarray(plot._phasor_point.get_xdata()), [expected_real]
    )
    assert_allclose(
        numpy.asarray(plot._phasor_point.get_ydata()), [expected_imag]
    )
    pyplot.close()


def test_spectral_plots_exceptions() -> None:
    """Test SpectralPlots raises on invalid parameters."""
    with pytest.raises(ValueError, match='num_components'):
        SpectralPlots(origin=[400.0, 450.0, 500.0, 550.0, 600.0])
    pyplot.close()

    with pytest.raises(ValueError, match='num_components'):
        SpectralPlots(origin=[400.0, 450.0, 500.0], dho=True)
    pyplot.close()

    with pytest.raises(ValueError, match=r'fractions.size'):
        SpectralPlots(origin=[490.0, 570.0], fraction=[0.5, 0.3, 0.2])
    pyplot.close()

    with pytest.raises(ValueError, match=r'sigmas.size'):
        SpectralPlots(origin=[490.0, 570.0], sigma=[30.0, 35.0, 40.0])
    pyplot.close()

    with pytest.raises(ValueError, match=r'hr_factors.size'):
        SpectralPlots(
            origin=[490.0, 570.0], hr_factor=[0.3, 0.4, 0.5], dho=True
        )
    pyplot.close()

    with pytest.raises(ValueError, match=r'vib_frequencies.size'):
        SpectralPlots(
            origin=[490.0, 570.0],
            vib_frequency=[1200.0, 1300.0, 1400.0],
            dho=True,
        )
    pyplot.close()
