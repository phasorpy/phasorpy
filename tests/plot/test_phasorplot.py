"""Test PhasorPlot class."""

import io
import math

import numpy
import pytest
from matplotlib import pyplot

from phasorpy.plot import PhasorPlot

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

        plot = PhasorPlot(title='pad', pad=0.1)
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

        plot = PhasorPlot(title='colors=red', allquadrants=True)
        plot.contour(real, imag, colors='red')
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

    def test_arrow(self):
        """Test arrow method."""
        plot = PhasorPlot(title='arrow')
        plot.arrow([0.0, 0.0], [0.8, 0.4], color='tab:blue', linewidth=2)
        plot.arrow(
            [math.hypot(0.8, 0.4), 0.0],
            [0.8, 0.4],
            angle=math.atan2(0.4, 0.8),
            arrowstyle='<->',
            linestyle='--',
            color='tab:red',
        )
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
        plot = PhasorPlot(grid=False, allquadrants=True, title='default')
        plot.polar_grid()
        self.show(plot)

        plot = PhasorPlot(
            grid=dict(radii=0, angles=0), allquadrants=True, title='empty'
        )
        self.show(plot)

        plot = PhasorPlot(grid=False, allquadrants=True, title='major axes')
        plot.polar_grid(radii=1, angles=4)
        self.show(plot)

        plot = PhasorPlot(grid=False, allquadrants=True, title='custom angles')
        plot.polar_grid(
            radii=2, angles=[-1, 0, math.pi / 2, 1, 2, 3, 4, 5, 6, 7]
        )
        self.show(plot)

        plot = PhasorPlot(grid=False, allquadrants=True, title='custom radii')
        plot.polar_grid(radii=[-0.1, 0.25, 0.75, 1.0, 1.1], angles=4)
        self.show(plot)

        plot = PhasorPlot(
            grid=False, pad=0.2, allquadrants=True, title='polygon'
        )
        plot.polar_grid(labels=['B', 'G', 'R'], samples=3, angles=3, radii=1)
        self.show(plot)

        plot = PhasorPlot(
            grid=False, allquadrants=True, pad=0.3, title='ticks'
        )
        plot.polar_grid(
            labels=[str(i) for i in range(0, 360, 45)], angles=0, radii=1
        )
        self.show(plot)

        plot = PhasorPlot(
            grid=False, allquadrants=True, pad=0.3, title='labels'
        )
        plot.polar_grid(
            ticks=[430, 450, 500, 550, 600, 650, 700, 750],
            tick_limits=(430, 780),
            # tick_format='{:.0f}',
            angles=0,
            radii=1,
        )
        self.show(plot)

        plot = PhasorPlot(
            grid=False, allquadrants=True, pad=0.25, title='ticks and labels'
        )
        plot.polar_grid(
            labels=['', '450', '500 nm', '550', '600', '650', '700', '750'],
            ticks=[430, 450, 500, 550, 600, 650, 700, 750],
            tick_limits=(430, 780),
        )
        self.show(plot)

        plot = PhasorPlot(grid=False, allquadrants=True, title='styled')
        plot.polar_grid(color='tab:red', linestyle='-', samples=0)
        self.show(plot)

        with pytest.raises(ValueError):
            plot.polar_grid(labels=['1', '2'], ticks=[1, 2, 3])

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

        plot = PhasorPlot(title='polar_reference', xlim=(-0.2, 1.1))
        plot.semicircle(polar_reference=(0.9852, 0.5526))
        self.show(plot)

        plot = PhasorPlot(
            frequency=80.0, title='phasor_reference', xlim=(-0.2, 1.1)
        )
        plot.semicircle(frequency=80.0, phasor_reference=(0.2, 0.4))
        self.show(plot)

        plot = PhasorPlot(title='limits', xlim=(0.4, 0.6), ylim=(0.4, 0.6))
        plot.semicircle(frequency=80.0)
        self.show(plot)

        plot = PhasorPlot(grid=False, title='use_lines')
        plot.semicircle(frequency=80, use_lines=True)
        self.show(plot)


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
