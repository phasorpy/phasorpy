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

        plot = PhasorPlot(
            xlim=(-0.1, 1.1),
            ylim=(-0.1, 0.6),
            xticks=[0, 0.5, 1],
            yticks=[0, 0.5, 1],
            xlabel='G',
            ylabel='S',
            title='kwargs',
        )
        self.show(plot)

        plot = PhasorPlot(xlim=None, xticks=None, title='del kwargs')
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

    def test_legend(self):
        plot = PhasorPlot(title='legend')
        plot.ax.plot(0.6, 0.4, 'o', label='label')
        plot.legend(loc='upper right')
        self.show(plot)

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
            [0.6, 0.4], [[3e-3, -1e-3], [-1e-3, 1e-3]], (256, 256)
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
            [0.6, 0.4], [[3e-3, -1e-3], [-1e-3, 1e-3]], (256, 256)
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
            xlim=(0.4, 0.8), ylim=(0.25, 0.55), title='histogram and contour'
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
        plot = PhasorPlot(allquadrants=allquadrants, title='components')
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
            fontsize=12,
            linestyle='',
            color='tab:green',
            labels=['A', 'B', 'C', ''],
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

    @pytest.mark.parametrize('allquadrants', (False, True))
    @pytest.mark.parametrize('polar', (False, True))
    def test_cursor(self, allquadrants, polar):
        """Test cursor method."""
        plot = PhasorPlot(
            allquadrants=allquadrants,
            title=f'cursor {allquadrants=} {polar=}',
        )

        def p(x, y):
            if polar:
                return plot, math.atan2(y, x), math.hypot(x, y)
            return plot, x, y

        cursor = PhasorPlot.polar_cursor if polar else PhasorPlot.cursor
        if polar:
            # these should not be drawn
            plot.polar_cursor(color='tab:red', label='c0')
            plot.polar_cursor(color='tab:red', polar=False, label='c0')
            plot.polar_cursor(
                None, None, 0.1, 0.1, color='tab:red', polar=False, label='c0'
            )

        cursor(*p(0.5, 0.25), color='tab:orange', label='c1')
        cursor(
            *p(0.5, 0.25),
            color='tab:orange',
            linestyle='--',
            polar=not polar,
            label='c2',
        )
        cursor(
            *p(0.2, 0.2),
            *p(0.4, 0.1)[1:],
            crosshair=True,
            fill=True,
            alpha=0.5,
            color='tab:cyan',
            label='c3',
        )
        cursor(
            *p(0.7, 0.35),
            radius=0.05,
            color='tab:green',
            fill=True,
            label='c4',
        )
        cursor(
            *p(0.8, 0.4),
            radius=0.05,
            crosshair=True,
            color='tab:green',
            label='c5',
        )
        cursor(
            *p(0.75, 0.15),
            radius=0.1,
            radius_minor=0.05,
            angle=0.0,
            color='tab:blue',
            label='c6',
        )
        cursor(
            *p(0.4, 0.38),
            radius=0.05,
            radius_minor=0.1,
            angle='phase',
            color='tab:blue',
            fill=True,
            alpha=0.2,
            label='c7',
        )
        cursor(
            *p(0.2, 0.4),
            radius=0.05,
            radius_minor=0.1,
            angle='semicircle',
            color='tab:blue',
            label='c8',
        )
        cursor(
            *p(0.0, 0.0),  # polar crosshair not defined
            radius=0.05,
            polar=True,
            crosshair=True,
            color='tab:red',
            label='c0',
        )

        self.show(plot)

    def test_cursor_special(self):
        """Test cursor method special cases."""
        plot = PhasorPlot(title='cursor special cases')
        plot._cursor(color='tab:olive', polar=False, label='none')
        plot.polar_cursor(0.5, color='tab:blue', label='phase only')
        plot.polar_cursor(
            None, 0.5, color='tab:orange', label='modulation only'
        )
        plot.cursor(
            [0.1, 0.9],
            [0.3, 0.3],
            color=[[1.0, 0, 0], [0, 1.0, 0]],
            label=['RGB red', 'RGB green'],
            radius=0.1,
        )

        # ndim > 1
        with pytest.raises(ValueError):
            plot.polar_cursor([[0.1, 0.2]])
        with pytest.raises(ValueError):
            plot.polar_cursor(None, [[0.1, 0.2]])
        # crosshair not supported with ellipse
        with pytest.raises(ValueError):
            plot.cursor(0.1, 0.2, radius=0.2, radius_minor=0.1, crosshair=True)
        # invalid angle
        with pytest.raises(ValueError):
            plot.cursor(0.1, 0.2, radius=0.05, radius_minor=0.1, angle='none')

        self.show(plot)

    def test_polar_grid(self):
        """Test polar_grid method."""
        phase_angles = numpy.linspace(0, 2 * math.pi, 8, endpoint=False)

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
        plot.polar_grid(samples=3, angles=3, radii=1, labels=['B', 'G', 'R'])
        self.show(plot)

        plot = PhasorPlot(
            grid=False, allquadrants=True, pad=0.3, title='labels, no ticks'
        )
        plot.polar_grid(
            labels=[f'{i:.2f}' for i in range(0, 360, 45)], angles=8
        )
        self.show(plot)

        plot = PhasorPlot(
            grid=False, allquadrants=True, pad=0.3, title='ticks, no labels'
        )
        plot.polar_grid(
            ticks=phase_angles,
            tick_format='{:.2f}',
            angles=8,
        )
        self.show(plot)

        plot = PhasorPlot(
            grid=False, allquadrants=True, pad=0.3, title='ticks and labels'
        )
        plot.polar_grid(
            ticks=phase_angles,
            labels=[f'{i:.2f}' for i in phase_angles],
            angles=8,
        )
        self.show(plot)

        plot = PhasorPlot(
            grid=False, allquadrants=True, pad=0.3, title='ticks and space'
        )
        plot.polar_grid(
            ticks=[430, 450, 500, 550, 600, 650, 700, 730],
            tick_space=numpy.linspace(430, 730, 16),
            angles=8,
        )
        self.show(plot)

        plot = PhasorPlot(
            grid=False,
            allquadrants=True,
            pad=0.25,
            title='ticks, labels, and space',
        )
        plot.polar_grid(
            labels=['', '450', '500 nm', '550', '600', '650', '700', '730'],
            ticks=[430, 450, 500, 550, 600, 650, 700, 730],
            tick_space=[430, 430 + (730 - 430) / 16, 730],
            angles=8,
        )
        self.show(plot)

        plot = PhasorPlot(grid=False, allquadrants=True, title='styled')
        plot.polar_grid(color='tab:red', linestyle='-', samples=0)
        self.show(plot)

        with pytest.raises(ValueError):
            plot.polar_grid(labels=['1', '2'], ticks=[1, 2, 3])

        with pytest.raises(ValueError):
            plot.polar_grid(
                labels=['1', '2'], ticks=[1, 2], tick_space=[[1, 2, 3]]
            )

    def test_semicircle(self):
        """Test semicircle method."""
        plot = PhasorPlot(grid=False, title='empty')
        plot.semicircle(label='semicircle')
        self.show(plot)

        plot = PhasorPlot(grid=False, title='frequency')
        plot.semicircle(frequency=80)
        self.show(plot)

        plot = PhasorPlot(grid=False, title='no labels')
        plot.semicircle(frequency=80, labels=[])
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

        plot = PhasorPlot(xlim=(-0.2, 1.1), title='polar_reference')
        plot.semicircle(polar_reference=(0.9852, 0.5526))
        self.show(plot)

        plot = PhasorPlot(
            frequency=80.0, xlim=(-0.2, 1.1), title='phasor_reference'
        )
        plot.semicircle(frequency=80.0, phasor_reference=(0.2, 0.4))
        self.show(plot)

        plot = PhasorPlot(xlim=(0.4, 0.6), ylim=(0.4, 0.6), title='limits')
        plot.semicircle(frequency=80.0)
        self.show(plot)

        plot = PhasorPlot(grid=False, title='use_lines')
        plot.semicircle(frequency=80, use_lines=True)
        self.show(plot)


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
