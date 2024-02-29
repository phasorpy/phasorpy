"""Test the phasorpy.plot module."""

import math

import numpy
import pytest
from matplotlib import pyplot
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
)

from phasorpy.plot import (
    PhasorPlot,
    plot_phasor,
    plot_phasor_image,
    plot_polar_frequency,
    plot_signal_image,
)


def test_phasorplot():
    """Test PhasorPlot init and attributes."""
    plot = PhasorPlot()
    ...
    pyplot.close()


def test_phasorplot_save():
    """Test PhasorPlot.save method."""
    plot = PhasorPlot()
    # plot.save(...)
    pyplot.close()


def test_phasorplot_plot():
    """Test PhasorPlot.plot method."""
    plot = PhasorPlot()
    # plot.plot(...)
    pyplot.close()


def test_phasorplot_hist2d():
    """Test PhasorPlot.hist2d method."""
    plot = PhasorPlot()
    # plot.hist2d(...)
    pyplot.close()


def test_phasorplot_contour():
    """Test PhasorPlot.contour method."""
    pytest.skip('PhasorPlot.contour not implemented')
    # plot = PhasorPlot()
    # plot.contour(...)
    # pyplot.close()


def test_phasorplot_imshow():
    """Test PhasorPlot.imshow method."""
    pytest.skip('PhasorPlot.imshow not implemented')
    # plot = PhasorPlot()
    # plot.imshow(...)
    # pyplot.close()


def test_phasorplot_components():
    """Test PhasorPlot.components method."""
    plot = PhasorPlot()
    # plot.components(...)
    pyplot.close()


def test_phasorplot_circle():
    """Test PhasorPlot.circle method."""
    plot = PhasorPlot()
    # plot.circle(...)
    pyplot.close()


def test_phasorplot_polar_cursor():
    """Test PhasorPlot.polar_cursor method."""
    plot = PhasorPlot()
    # plot.polar_cursor(...)
    pyplot.close()


def test_phasorplot_polar_grid():
    """Test PhasorPlot.polar_grid method."""
    plot = PhasorPlot()
    # plot.polar_grid(...)
    pyplot.close()


def test_phasorplot_semicircle():
    """Test PhasorPlot.semicircle method."""
    plot = PhasorPlot()
    # plot.semicircle(...)
    pyplot.close()


def test_plot_phasor():
    """Test plot_phasor function."""
    # plot_phasor(...)
    # pyplot.close()


def test_plot_polar_frequency():
    """Test plot_polar_frequency function."""
    # plot_polar_frequency(...)
    # pyplot.close()


def test_plot_signal_image():
    """Test plot_signal_image function."""
    show = False  # enable interactive plotting
    shape = (7, 31, 33, 11)
    data = numpy.arange(math.prod(shape)).reshape(shape)
    data %= math.prod(shape[-2:])
    data = data / math.prod(shape[-2:])

    plot_signal_image(data, title='default', show=show)
    pyplot.close()
    plot_signal_image(data, axis=0, title='axis 0', show=show)
    pyplot.close()
    plot_signal_image(data, axis=2, title='axis 2', show=show)
    pyplot.close()
    plot_signal_image(
        data, cmap='hot', percentile=[5, 95], title='percentile', show=show
    )
    pyplot.close()

    with pytest.raises(ValueError):
        # not an image
        plot_signal_image(data[0, 0], show=False)
    pyplot.close()
    with pytest.raises(ValueError):
        # percentile out of range
        plot_signal_image(data, percentile=(-1, 101), show=False)
    pyplot.close()


def test_plot_phasor_image():
    """Test plot_phasor_image function."""
    show = False  # enable interactive plotting
    shape = (7, 11, 31, 33)
    data = numpy.arange(math.prod(shape)).reshape(shape)
    data %= math.prod(shape[-2:])
    data = data / math.prod(shape[-2:])

    # 2D data
    d = data[0, 0]
    plot_phasor_image(d, d, d, title='mean, real, imag', show=show)
    pyplot.close()
    plot_phasor_image(None, d, d, title='real, imag', show=show)
    pyplot.close()
    # 4D data
    d = data
    plot_phasor_image(d, d, d, title='4D images', show=show)
    pyplot.close()
    # 7 harmonics
    plot_phasor_image(d[0], d, d, title='harmonics up to 4', show=show)
    pyplot.close()
    plot_phasor_image(
        None,
        d,
        d,
        harmonics=2,
        title='real and imag harmonics up to 2',
        show=show,
    )
    pyplot.close()

    d = data[0, 0]
    plot_phasor_image(
        d,
        d,
        d,
        percentile=5.0,
        cmap='hot',
        title='5th percentile with colormap',
        show=show,
    )
    pyplot.close()

    d = data[0, 0, 0]
    with pytest.raises(ValueError):
        # not an image
        plot_phasor_image(d, d, d, show=False)
    pyplot.close()
    with pytest.raises(ValueError):
        # not an image
        plot_phasor_image(None, d, d, show=False)
    pyplot.close()

    d = data[0, 0]
    with pytest.raises(ValueError):
        # not an image
        plot_phasor_image(None, d, d, harmonics=2, show=False)
    pyplot.close()

    d = data[0]
    with pytest.raises(ValueError):
        # shape mismatch
        plot_phasor_image(d, d[0], d, show=False)
    pyplot.close()
    with pytest.raises(ValueError):
        # shape mismatch
        plot_phasor_image(d, d, d[0], show=False)
    pyplot.close()
    with pytest.raises(ValueError):
        # shape mismatch
        plot_phasor_image(d, d[0, :-1], d[0, :-1], show=False)
    pyplot.close()
    with pytest.raises(ValueError):
        # percentile out of range
        plot_phasor_image(d, d, d, percentile=-1, show=False)
    pyplot.close()
    with pytest.raises(ValueError):
        # percentile out of range
        plot_phasor_image(d, d, d, percentile=50, show=False)
    pyplot.close()
