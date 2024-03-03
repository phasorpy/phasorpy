"""Test the phasorpy.plot module."""

import io
import math

import numpy
import pytest
from matplotlib import pyplot

from phasorpy.plot import (
    PhasorPlot,
    plot_phasor,
    plot_phasor_image,
    plot_polar_frequency,
    plot_signal_image,
)


class TestPhasoPlot:
    """Test the PhasorPlot class."""

    def show(self, plot):
        """Show plot."""
        if False:  # enable for interactive plotting
            plot.show()
        pyplot.close()

    def test_init(self):
        """Test __init__ and attributes."""
        plot = PhasorPlot(title='default')
        self.show(plot)

        plot = PhasorPlot(frequency=80.0, title='frequency')
        self.show(plot)

        plot = PhasorPlot(grid=False, title='no grid')
        self.show(plot)

        plot = PhasorPlot(allquadrants=True, title='allquadrants')
        self.show(plot)

        plot = PhasorPlot(title='kwargs', xlim=(-0.1, 1.1), ylim=(-0.1, 0.9))
        self.show(plot)

        fig, ax = pyplot.subplots()
        plot = PhasorPlot(ax=ax, title='axes')
        assert plot.ax == ax
        assert plot.fig == fig
        self.show(plot)

    def test_save(self):
        """Test save method."""
        fh = io.BytesIO()
        plot = PhasorPlot(title='save')
        plot.save(fh, format='png')
        assert fh.getvalue()[:6] == b'\x89PNG\r\n'
        pyplot.close()

    def test_plot(self):
        """Test plot method."""
        plot = PhasorPlot(title='plot')
        # plot.plot(...)
        self.show(plot)

    def test_hist2d(self):
        """Test hist2d method."""
        plot = PhasorPlot(title='hist2d')
        # plot.hist2d(...)
        self.show(plot)

    def test_contour(self):
        """Test contour method."""
        plot = PhasorPlot(title='contour')
        with pytest.raises(NotImplementedError):
            plot.contour([[0]], [[0]])
        self.show(plot)

    def test_imshow(self):
        """Test imshow method."""
        plot = PhasorPlot(title='imshow')
        with pytest.raises(NotImplementedError):
            plot.imshow([[0]])
        self.show(plot)

    def test_components(self):
        """Test components method."""
        plot = PhasorPlot(title='components')
        # plot.components(...)
        self.show(plot)

    def test_circle(self):
        """Test circle method."""
        plot = PhasorPlot(title='circle')
        plot.circle(0.5, 0.2, 0.1, color='tab:red', linestyle='-')
        self.show(plot)

    def test_polar_cursor(self):
        """Test polar_cursor method."""
        plot = PhasorPlot(title='polar_cursor')
        # plot.polar_cursor(...)
        self.show(plot)

    def test_polar_grid(self):
        """Test polar_grid method."""
        plot = PhasorPlot(grid=False, allquadrants=True, title='polar_grid')
        plot.polar_grid(color='tab:red', linestyle='-')
        self.show(plot)

    def test_semicircle(self):
        """Test semicircle method."""
        plot = PhasorPlot(grid=False, title='empty')
        plot.semicircle()
        self.show(plot)

        plot = PhasorPlot(grid=False, title='frequency')
        plot.semicircle(frequency=80)
        self.show(plot)

        plot = PhasorPlot(grid=False, title='red')
        plot.semicircle(frequency=80, color='tab:red', linestyle=':')
        self.show(plot)

        plot = PhasorPlot(grid=False, title='lifetime')
        plot.semicircle(frequency=80, lifetime=[1, 2])
        self.show(plot)

        plot = PhasorPlot(grid=False, title='labels')
        plot.semicircle(
            frequency=80, lifetime=[1, 2], labels=['label 1', 'label 2']
        )
        self.show(plot)

        plot = PhasorPlot(title='polar_reference', xlim=(-0.2, 1.05))
        plot.semicircle(polar_reference=(0.9852, 0.5526))
        self.show(plot)

        plot = PhasorPlot(
            frequency=80.0, title='phasor_reference', xlim=(-0.2, 1.05)
        )
        plot.semicircle(frequency=80.0, phasor_reference=(0.2, 0.4))
        self.show(plot)


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
