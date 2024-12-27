"""Test the phasorpy.plot module."""

import io
import math

import numpy
import pytest
from matplotlib import pyplot
from xarray import DataArray

from phasorpy.plot import (
    PhasorPlot,
    PhasorPlotFret,
    plot_phasor,
    plot_phasor_image,
    plot_polar_frequency,
    plot_signal_image,
)

INTERACTIVE = False  # enable for interactive plotting


class TestPhasorPlot:
    """Test PhasorPlot class."""

    def show(self, plot):
        """Show plot."""
        if INTERACTIVE:
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

    def test_dataunit_to_point(self):
        """Test dataunit_to_point method."""
        plot = PhasorPlot(title='dataunit_to_point')
        assert 100 < plot.dataunit_to_point < 500
        self.show(plot)

    def test_on_format_coord(self):
        """Test on_format_coord callback."""
        plot = PhasorPlot(frequency=80.0, title='on_format_coord')
        coords = plot._on_format_coord(0.5, 0.5)
        assert '0.5' in coords
        assert 'ns' in coords
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
        plot.plot(0.6, 0.4, label='1')
        plot.plot([0.2, 0.9], [0.4, 0.3], '.-', label='2')
        plot.plot(
            [[0.29, 0.3, 0.31], [0.41, 0.29, 0.3]],
            [[0.31, 0.29, 0.2], [0.49, 0.5, 0.51]],
            'x',
            label='3',
        )
        plot.plot(-0.5, -0.5, label='4')
        plot.plot(0.5, 0.25, marker='v', label='v')
        self.show(plot)

    def test_hist2d(self):
        """Test hist2d method."""
        real, imag = numpy.random.multivariate_normal(
            (0.6, 0.4), [[3e-3, -1e-3], [-1e-3, 1e-3]], (256, 256)
        ).T
        plot = PhasorPlot(title='hist2d')
        plot.hist2d(real, imag)
        self.show(plot)

        plot = PhasorPlot(title='hist2d parameters', allquadrants=True)
        plot.hist2d(
            real, imag, bins=100, cmax=500, cmap='viridis', norm='linear'
        )
        self.show(plot)

    def test_contour(self):
        """Test contour method."""
        real, imag = numpy.random.multivariate_normal(
            (0.6, 0.4), [[3e-3, -1e-3], [-1e-3, 1e-3]], (256, 256)
        ).T
        plot = PhasorPlot(title='contour')
        plot.contour(real, imag)
        self.show(plot)

        plot = PhasorPlot(title='contour parameters', allquadrants=True)
        plot.contour(real, imag, bins=200, cmap='viridis', norm='linear')
        self.show(plot)

    def test_histogram_contour(self):
        """Test histogram and contour match."""
        real, imag = numpy.random.multivariate_normal(
            (0.6, 0.4), [[3e-3, -1e-3], [-1e-3, 1e-3]], (256, 256)
        ).T
        plot = PhasorPlot(
            title='histogram and contour', xlim=(0.4, 0.8), ylim=(0.25, 0.55)
        )
        plot.hist2d(real, imag, bins=32, cmap='Blues')
        plot.contour(real, imag, bins=32, levels=4, cmap='Reds')
        self.show(plot)

    def test_imshow(self):
        """Test imshow method."""
        plot = PhasorPlot(title='imshow')
        with pytest.raises(NotImplementedError):
            plot.imshow([[0]])
        self.show(plot)

    @pytest.mark.parametrize('allquadrants', (True, False))
    def test_components(self, allquadrants):
        """Test components method."""
        real = [0.1, 0.2, 0.5, 0.9]
        imag = [0.3, 0.4, 0.5, 0.3]
        weights = [2, 1, 2, 1]
        plot = PhasorPlot(title='components', allquadrants=allquadrants)
        with pytest.raises(ValueError):
            plot.components([0.0, 1.0], [0.0])
        with pytest.raises(ValueError):
            plot.components([0.0, 1.0], [0.0, 1.0], labels=['A'])
        plot.components(
            real, imag, fill=True, linestyle=':', facecolor='lightyellow'
        )
        plot.components(real, imag, weights, linestyle='-', color='tab:blue')
        plot.components(
            real,
            imag,
            marker='D',
            linestyle='',
            color='tab:red',
            label='components',
        )
        plot.components(
            real,
            imag,
            weights,
            linestyle='-',
            marker='.',
            color='tab:blue',
            label='mixture',
        )
        plot.components(
            real,
            imag,
            labels=['A', 'B', 'C', ''],
            fontsize=12,
            linestyle='',
            color='tab:green',
        )
        plot.components(real[-1], imag[-1], labels=['D'])
        self.show(plot)

    def test_line(self):
        """Test line method."""
        plot = PhasorPlot(title='line')
        plot.line([0.8, 0.4], [0.2, 0.3], color='tab:red', linestyle='--')
        self.show(plot)

    def test_circle(self):
        """Test circle method."""
        plot = PhasorPlot(title='circle')
        plot.circle(0.5, 0.2, 0.1, color='tab:red', linestyle='-')
        self.show(plot)

    def test_cursor(self):
        """Test cursor method."""
        plot = PhasorPlot(title='cursor')
        plot.cursor(0.4, 0.3, color='tab:blue', linestyle='-')
        plot.cursor(0.52, 0.3, 0.78, 0.16, color='tab:orange')
        plot.cursor(0.9, 0.3, radius=0.05, color='tab:green')
        plot.cursor(
            0.4, 0.3, radius=0.05, radius_minor=0.1, fill=True, alpha=0.5
        )
        plot.cursor(
            0.11, 0.3, radius=0.05, radius_minor=0.1, align_semicircle=True
        )
        self.show(plot)

    def test_cursor_allquadrants(self):
        """Test cursor method with allquadrants."""
        plot = PhasorPlot(title='cursor allquadrants', allquadrants=True)
        plot.cursor(-0.4, -0.3, color='tab:blue', linestyle='-')
        plot.cursor(-0.52, -0.3, -0.78, -0.16, color='tab:orange')
        plot.cursor(-0.9, -0.3, radius=0.1, color='tab:green')
        plot.cursor(
            -0.3, -0.6, radius=0.1, radius_minor=0.2, fill=True, alpha=0.5
        )
        plot.cursor(-0.6, 0.6, radius=0.1, radius_minor=0.2, angle=2.36)
        self.show(plot)

    def test_polar_cursor(self):
        """Test polar_cursor method."""
        plot = PhasorPlot(title='polar_cursor')
        plot.polar_cursor()
        plot.polar_cursor(0.6435, 0.5, color='tab:blue', linestyle='-')
        plot.polar_cursor(0.5236, 0.6, 0.1963, 0.8, color='tab:orange')
        plot.polar_cursor(0.3233, 0.9482, radius=0.05, color='tab:green')
        plot.polar_cursor(0.3, color='tab:red', linestyle='--')
        self.show(plot)

    def test_polar_cursor_allquadrants(self):
        """Test polar_cursor method with allquadrants."""
        plot = PhasorPlot(title='polar_cursor allquadrants', allquadrants=True)
        plot.polar_cursor()
        plot.polar_cursor(
            0.6435 + math.pi, 0.5, color='tab:blue', linestyle='-'
        )
        plot.polar_cursor(
            0.5236 + math.pi, 0.6, 0.1963 + math.pi, 0.8, color='tab:orange'
        )
        plot.polar_cursor(
            0.3233 + math.pi, 0.9482, radius=0.1, color='tab:green'
        )
        plot.polar_cursor(0.3 + math.pi, color='tab:red', linestyle='--')
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

        plot = PhasorPlot(grid=False, title='no labels')
        plot.semicircle(frequency=80, labels=())
        self.show(plot)

        plot = PhasorPlot(grid=False, title='no circle')
        plot.semicircle(frequency=80, show_circle=False)
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

        plot = PhasorPlot(title='limits', xlim=(0.4, 0.6), ylim=(0.4, 0.6))
        plot.semicircle(frequency=80.0)
        self.show(plot)

        plot = PhasorPlot(grid=False, title='use_lines')
        plot.semicircle(frequency=80, use_lines=True)
        self.show(plot)


def test_fret_phasorplot():
    """Test PhasorPlotFret."""
    plot = PhasorPlotFret(
        frequency=60.0,
        donor_lifetime=4.2,
        acceptor_lifetime=3.0,
        fret_efficiency=0.5,
        donor_freting=0.9,
        donor_bleedthrough=0.1,
        title='PhasorPlotFret',
    )
    if INTERACTIVE:
        plot.show()
    pyplot.close()


def test_fret_phasorplot_interactive():
    """Test PhasorPlotFret interactive."""
    plot = PhasorPlotFret(
        frequency=60.0,
        donor_lifetime=4.2,
        acceptor_lifetime=3.0,
        fret_efficiency=0.5,
        donor_background=0.1,
        acceptor_background=0.1,
        interactive=True,
        title='PhasorPlotFret interactive',
    )
    plot._frequency_slider.set_val(80.0)
    plot._donor_freting_slider.set_val(0.9)
    plot._donor_bleedthrough_slider.set_val(0.1)
    plot._donor_bleedthrough_slider.set_val(0.0)
    plot._donor_background_slider.set_val(0.1)
    plot._acceptor_background_slider.set_val(0.1)
    plot._donor_background_slider.set_val(0.0)
    plot._acceptor_background_slider.set_val(0.0)
    plot._donor_freting_slider.set_val(0.0)
    if INTERACTIVE:
        plot.show()
    pyplot.close()


def test_plot_phasor():
    """Test plot_phasor function."""
    real, imag = numpy.random.multivariate_normal(
        (0.6, 0.4), [[3e-3, -1e-3], [-1e-3, 1e-3]], 32
    ).T
    plot_phasor(
        real,
        imag,
        style='plot',
        title='plot',
        color='tab:red',
        frequency=80.0,
        show=INTERACTIVE,
    )
    pyplot.close()

    _, ax = pyplot.subplots()
    real, imag = numpy.random.multivariate_normal(
        (0.6, 0.4), [[3e-3, -1e-3], [-1e-3, 1e-3]], (256, 256)
    ).T
    plot_phasor(
        real,
        imag,
        ax=ax,
        title='hist2d',
        cmap='Blues',
        allquadrants=True,
        grid=False,
        show=INTERACTIVE,
    )
    pyplot.close()

    plot_phasor(
        real,
        imag,
        style='contour',
        title='contour',
        cmap='viridis',
        levels=4,
        show=INTERACTIVE,
    )
    pyplot.close()

    with pytest.raises(ValueError):
        plot_phasor(0, 0, style='invalid')


def test_plot_polar_frequency():
    """Test plot_polar_frequency function."""
    plot_polar_frequency(
        [1, 10, 100],
        [0, 0.5, 1],
        [1, 0.5, 0],
        title='plot_polar_frequency',
        show=INTERACTIVE,
    )
    pyplot.close()

    _, ax = pyplot.subplots()
    plot_polar_frequency(
        [1, 10, 100],
        [[0, 0.1], [0.5, 0.55], [1, 1]],
        [[[1, 0.9], [0.5, 0.45], [0, 0]]],
        ax=ax,
        show=INTERACTIVE,
    )
    pyplot.close()


def test_plot_signal_image():
    """Test plot_signal_image function."""
    shape = (7, 31, 33, 11)
    data_ = numpy.arange(math.prod(shape)).reshape(shape)
    data_ %= math.prod(shape[-2:])
    data = data_ / math.prod(shape[-2:])

    plot_signal_image(data, title='default', show=INTERACTIVE)
    pyplot.close()
    plot_signal_image(data, axis=0, title='axis 0', show=INTERACTIVE)
    pyplot.close()
    plot_signal_image(data, axis=2, title='axis 2', show=INTERACTIVE)
    pyplot.close()
    plot_signal_image(
        data,
        cmap='hot',
        percentile=[5, 95],
        title='percentile',
        show=INTERACTIVE,
    )
    pyplot.close()

    dataarray = DataArray(
        data, {'H': numpy.linspace(1, 2, 11)}, ('T', 'Y', 'X', 'H')
    )
    plot_signal_image(dataarray, title='DataArray', show=INTERACTIVE)
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
    shape = (7, 11, 31, 33)
    data_ = numpy.arange(math.prod(shape)).reshape(shape)
    data_ %= math.prod(shape[-2:])
    data = data_ / math.prod(shape[-2:])

    # 2D data
    d = data[0, 0]
    plot_phasor_image(d, d, d, title='mean, real, imag', show=INTERACTIVE)
    pyplot.close()
    plot_phasor_image(None, d, d, title='real, imag', show=INTERACTIVE)
    pyplot.close()
    # 4D data
    d = data
    plot_phasor_image(d, d, d, title='4D images', show=INTERACTIVE)
    pyplot.close()
    # 7 harmonics
    plot_phasor_image(d[0], d, d, title='harmonics up to 4', show=INTERACTIVE)
    pyplot.close()
    plot_phasor_image(
        None,
        d,
        d,
        harmonics=2,
        title='real and imag harmonics up to 2',
        show=INTERACTIVE,
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
        show=INTERACTIVE,
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


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
