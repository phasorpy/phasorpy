"""Plot phasor coordinates and related data.

The ``phasorpy.plot`` module provides functions and classes to create
publication quality visualizations of phasor coordinates and related data
using the matplotlib library.

"""

from __future__ import annotations

__all__ = [
    'PhasorPlot',
    'phasor_plot',
    'multi_frequency_plot',
]

import math
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, Literal, BinaryIO

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

import numpy
from matplotlib import pyplot
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Polygon

from ._utils import (
    circle_circle_intersection,
    circle_line_intersection,
    parse_kwargs,
    scale_matrix,
    sort_coordinates,
    update_kwargs,
)
from .phasor import (
    phasor_calibrate,
    phasor_from_lifetime,
    phasor_semicircle,
    phasor_to_polar,
)

GRID_COLOR = '0.5'
GRID_LINESTYLE = ':'
GRID_LINESTYLE_MAJOR = '-'
GRID_LINEWIDH = 1.0
GRID_LINEWIDH_MINOR = 0.5
GRID_FILL = False


def phasor_plot(
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    style: Literal['plot', 'hist2d'] | None = None,
    allquadrants: bool | None = None,
    frequency: float | None = None,
    show: bool = True,
    **kwargs: Any,
) -> None:
    """Plot phasor coordinates.

    A simplified interface to the :py:class:`PhasorPlot` class.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
        Must be of same shape as `real`.
    style : {'plot', 'hist2d'}, optional
        Method used to plot phasor coordinates,
        :py:meth:`PhasorPlot.plot` or :py:meth:`PhasorPlot.hist2d`.
        By default, if the number of coordinates are less than 65536
        and the arrays are less than three-dimensional, `'plot'` style is used,
        else `'hist2d'`.
    allquadrants : bool, optional
        Show all quadrants of phasor space.
        By default, only the first quadrant is shown.
    frequency: float, optional
        Frequency of phasor plot.
        If provided, the universal circle is labeled with reference lifetimes.
    show : bool, optional
        Display figure. The default is True.
    **kwargs
        Additional parguments passed to :py:class:`PhasorPlot`,
        :py:meth:`PhasorPlot.plot`, or :py:meth:`PhasorPlot.hist2d`
        depending on `style`.

    """
    init_kwargs = parse_kwargs(
        kwargs,
        'ax',
        'title',
        'xlabel',
        'ylabel',
        'xlim',
        'ylim',
        'xticks',
        'yticks',
        'grid',
    )

    real = numpy.asanyarray(real)
    imag = numpy.asanyarray(imag)
    plot = PhasorPlot(
        frequency=frequency, allquadrants=allquadrants, **init_kwargs
    )
    if style is None:
        style = 'plot' if real.size < 65536 and real.ndim < 3 else 'hist2d'
    if style == 'plot':
        plot.plot(real, imag, **kwargs)
    elif style == 'hist2d':
        plot.hist2d(real, imag, **kwargs)
    else:
        raise ValueError(f'invalid {style=}')
    if show:
        plot.show()


class PhasorPlot:
    """Phasor plot.

    Parameters
    ----------
    allquadrants : bool, optional
        Show all quandrants of phasor space.
        By default, only the first quadrant with universal semicricle is shown.
    ax : matplotlib axis, optional
        ...
    frequency : float, optional
        ...
    grid : bool, optional
        ...
    **kwargs
        ...

    """

    _ax: Axes
    """Matplotlib axis."""

    _limits: tuple[tuple[float, float], tuple[float, float]]
    """Axes limits (xmin, xmax), (ymin, ymax)."""

    _full: bool
    """Show all quadrants of phasor space."""

    def __init__(
        self,
        /,
        allquadrants: bool | None = None,
        ax: Any = None,
        *,
        frequency: float | None = None,
        grid: bool = True,
        **kwargs,
    ):
        # initialize empty phasor plot
        self._ax = pyplot.subplots()[1] if ax is None else ax

        self._full = bool(allquadrants)
        if self._full:
            xlim = (-1.05, 1.05)
            ylim = (-1.05, 1.05)
            xticks: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
            yticks: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
            if grid:
                self.polar_grid()
        else:
            xlim = (-0.05, 1.05)
            ylim = (-0.05, 0.7)
            xticks = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
            yticks = (0.0, 0.2, 0.4, 0.6)
            if grid:
                self.semicircle(frequency=frequency)

        title = 'Phasor plot'
        if frequency is not None:
            title += f' ({frequency:g} MHz)'

        update_kwargs(
            kwargs,
            title=title,
            xlabel='G, real',
            ylabel='S, imag',
            aspect='equal',
            xlim=xlim,
            ylim=ylim,
            xticks=xticks,
            yticks=yticks,
        )
        self._limits = (kwargs['xlim'], kwargs['ylim'])
        self._ax.set(**kwargs)

    @property
    def ax(self) -> Axes:
        """Matplotlib :class:`matplotlib.axes.Axes`."""
        return self._ax

    @property
    def fig(self) -> Figure | None:
        """Matplotlib :class:`matplotlib.figure.Figure`."""
        return self._ax.get_figure()

    def show(self) -> None:
        """Display all open figures. Call :meth:`matplotlib.pyplot.show`."""
        # self.fig.show()
        pyplot.show()

    def save(
        self,
        file: str | os.PathLike[Any] | BinaryIO | None,
        /,
        **kwargs: Any,
    ) -> None:
        """Save current figure to file.

        Parameters
        ----------
        file : str, path-like, or binary file-like
            Path or Python file-like object to write the current figure to.
        **kwargs
            Additional keyword arguments passed to
            :meth:`matplotlib:pyplot.savefig`.

        """
        pyplot.savefig(file, **kwargs)

    def plot(
        self,
        real: ArrayLike,
        imag: ArrayLike,
        /,
        fmt='o',
        *,
        label: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Plot imag versus real coordinates as markers and/or lines.

        Parameters
        ----------
        real : array_like
            Real component of phasor coordinates.
            Must be one or two dimensional.
        imag : array_like
            Imaginary component of phasor coordinates.
            Must be of same shape as `real`.
        fmt : str, optional
            Matplotlib style format string. The default is 'o'.
        label : str or sequence of str, optional
            Plot label.
            May be a sequence if phasor coordinates are two dimensional arrays.
        **kwargs
            Additional parameters passed to :meth:`matplotlib.pyplot.plot`.

        """
        ax = self._ax
        if label is not None and (
            isinstance(label, str) or not isinstance(label, Sequence)
        ):
            label = (label,)
        for (
            i,
            (re, im),
        ) in enumerate(
            zip(numpy.array(real, ndmin=2), numpy.array(imag, ndmin=2))
        ):
            lbl = None
            if label is not None:
                try:
                    lbl = label[i]
                except KeyError:
                    pass
            ax.plot(re, im, fmt, label=lbl, **kwargs)
        if label is not None:
            ax.legend()

    def hist2d(
        self,
        real: ArrayLike,
        imag: ArrayLike,
        /,
        **kwargs: Any,
    ) -> None:
        """Plot 2D histogram of imag versus real coordinates.

        Parameters
        ----------
        real : array_like
            Real component of phasor coordinates.
        imag : array_like
            Imaginary component of phasor coordinates.
            Must be of same shape as `real`.
        **kwargs
            Additional parameters passed to :meth:`matplotlib.pyplot.hist2d`.

        """
        update_kwargs(
            kwargs,
            range=self._limits,
            cmap='Blues',
            norm='log',
            cmin=1,
        )

        (xmin, xmax), (ymin, ymax) = kwargs['range']
        assert xmax > xmin and ymax > ymin

        bins = kwargs.get('bins', 128)
        if isinstance(bins, int):
            assert bins > 0
            aspect = (xmax - xmin) / (ymax - ymin)
            if aspect >= 1:
                bins = (bins, max(int(bins / aspect), 1))
            else:
                bins = (max(int(bins * aspect), 1), bins)
        kwargs['bins'] = bins

        real = numpy.asanyarray(real).flat
        imag = numpy.asanyarray(imag).flat
        self._ax.hist2d(real, imag, **kwargs)

        # matplotlib's hist2d sets it's own axes limits, so reset it
        self._ax.set(xlim=self._limits[0], ylim=self._limits[1])

    def contour(
        self,
        real: ArrayLike,
        imag: ArrayLike,
        /,
    ) -> None:
        """Plot contours of imag versus real coordinates.

        Parameters
        ----------
        real : array_like
            Real component of phasor coordinates.
        imag : array_like
            Imaginary component of phasor coordinates.
            Must be of same shape as `real`.

        """
        raise NotImplementedError

    def imshow(
        self,
        image: ArrayLike,
        /,
    ) -> None:
        """Plot an image, for example, a 2D histogram.

        Parameters
        ----------
        image : array_like
            Image to display.

        """
        raise NotImplementedError

    def components(
        self,
        real: Sequence[float],
        imag: Sequence[float],
        /,
        fraction: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Plot linear combinations of phasor coordinates or ranges thereof.

        Parameters
        ----------
        real : sequence of float
            Real component of phasor coordinates.
        imag : sequence of float
            Imaginary component of phasor coordinates.
        fraction: sequence of float, optional
            ...

        """
        if fraction is None:
            update_kwargs(
                kwargs,
                edgecolor=GRID_COLOR,
                linestyle=GRID_LINESTYLE,
                linewidth=GRID_LINEWIDH,
                fill=GRID_FILL,
            )
            self._ax.add_patch(
                Polygon(numpy.vstack(sort_coordinates(real, imag)).T, **kwargs)
            )
            return
        update_kwargs(
            kwargs,
            color=GRID_COLOR,
            linestyle=GRID_LINESTYLE,
            linewidth=GRID_LINEWIDH,
        )
        center_re, center_im = numpy.average(
            numpy.vstack((real, imag)), axis=-1, weights=fraction
        )
        for re, im in zip(real, imag):
            self._ax.add_line(
                Line2D([center_re, re], [center_im, im], **kwargs)
            )
            # TODO: add fraction labels?

    def circle(
        self,
        real: float,
        imag: float,
        /,
        radius: float,
        **kwargs: Any,
    ) -> None:
        """Draw circle of radius around center point.

        Parameters
        ----------
        real, imag : float
            ...
        radius : float
            ...
        **kwargs
            ...

        """
        update_kwargs(
            kwargs,
            color=GRID_COLOR,
            linestyle=GRID_LINESTYLE,
            linewidth=GRID_LINEWIDH,
            fill=GRID_FILL,
        )
        ax = self._ax
        ax.add_patch(pyplot.Circle((real, imag), radius, **kwargs))

    def polar_cursor(
        self,
        phase: float | None = None,
        modulation: float | None = None,
        phase_limit: float | None = None,
        modulation_limit: float | None = None,
        radius: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Plot phase and modulation grid lines.

        Parameters
        ----------
        phase : float, optional
            ...
        modulation : float, optional
            ...
        phase_limit : float, optional
            ...
        modulation_limit : float, optional
            ...
        radius : float, optional
            ...
        **kwargs
            ...

        """
        update_kwargs(
            kwargs,
            color=GRID_COLOR,
            linestyle=GRID_LINESTYLE,
            linewidth=GRID_LINEWIDH,
            fill=GRID_FILL,
        )
        ax = self._ax
        if radius is not None and phase is not None and modulation is not None:
            x = modulation * math.cos(phase)
            y = modulation * math.sin(phase)
            ax.add_patch(pyplot.Circle((x, y), radius, **kwargs))
            del kwargs['fill']
            p0, p1 = circle_line_intersection(x, y, radius, 0, 0, x, y)
            ax.add_line(Line2D((p0[0], p1[0]), (p0[1], p1[1]), **kwargs))
            p0, p1 = circle_circle_intersection(0, 0, modulation, x, y, radius)
            ax.add_patch(
                Arc(
                    (0, 0),
                    modulation * 2,
                    modulation * 2,
                    theta1=math.degrees(math.atan(p0[1] / p0[0])),
                    theta2=math.degrees(math.atan(p1[1] / p1[0])),
                    fill=False,
                    **kwargs,
                )
            )

            return
        del kwargs['fill']
        for phi in (phase, phase_limit):
            if phi is not None:
                if modulation is not None and modulation_limit is not None:
                    x0 = modulation * math.cos(phi)
                    y0 = modulation * math.sin(phi)
                    x1 = modulation_limit * math.cos(phi)
                    y1 = modulation_limit * math.sin(phi)
                else:
                    x0 = 0
                    y0 = 0
                    x1 = math.cos(phi)
                    y1 = math.sin(phi)
                ax.add_line(Line2D((x0, x1), (y0, y1), **kwargs))
        for mod in (modulation, modulation_limit):
            if mod is not None:
                if phase is not None and phase_limit is not None:
                    theta1 = math.degrees(min(phase, phase_limit))
                    theta2 = math.degrees(max(phase, phase_limit))
                else:
                    theta1 = 0.0
                    theta2 = 360.0 if self._full else 90.0
                ax.add_patch(
                    Arc(
                        (0, 0),
                        mod * 2,
                        mod * 2,
                        theta1=theta1,
                        theta2=theta2,
                        fill=False,  # filling arc objects is not supported
                        **kwargs,
                    )
                )

    def polar_grid(self, **kwargs) -> None:
        """Draw polar coordinate system.

        Parameters
        ----------
        **kwargs
            ...

        """
        ax = self._ax
        # major gridlines
        kwargs_copy = kwargs.copy()
        update_kwargs(
            kwargs,
            color=GRID_COLOR,
            linestyle=GRID_LINESTYLE_MAJOR,
            linewidth=GRID_LINEWIDH,
            # fill=GRID_FILL,
        )
        ax.add_line(Line2D([-1, 1], [0, 0], **kwargs))
        ax.add_line(Line2D([0, 0], [-1, 1], **kwargs))
        ax.add_patch(pyplot.Circle((0, 0), 1, fill=False, **kwargs))
        # minor gridlines
        kwargs = kwargs_copy
        update_kwargs(
            kwargs,
            color=GRID_COLOR,
            linestyle=GRID_LINESTYLE,
            linewidth=GRID_LINEWIDH_MINOR,
        )
        for r in (1 / 3, 2 / 3):
            ax.add_patch(pyplot.Circle((0, 0), r, fill=False, **kwargs))
        for a in (3, 6):
            x = math.cos(math.pi / a)
            y = math.sin(math.pi / a)
            ax.add_line(Line2D([-x, x], [-y, y], **kwargs))
            ax.add_line(Line2D([-x, x], [y, -y], **kwargs))

    def semicircle(
        self,
        frequency: float | None = None,
        *,
        polar_reference: tuple[float, float] | None = None,
        phasor_reference: tuple[float, float] | None = None,
        lifetime: Sequence[float] | None = None,
        samples: int = 255,
        **kwargs,
    ) -> None:
        """Draw universal semicircle.

        Parameters
        ----------
        frequency : float, optional
            ...
        polar_reference : (float, float), optional
            ...
        phasor_reference : (float, float), optional
            ...
        lifetime : sequence of float, optional
            ...
        samples : int, optional
            ...
        **kwargs
            ...

        """
        update_kwargs(
            kwargs,
            color=GRID_COLOR,
            linestyle=GRID_LINESTYLE_MAJOR,
            linewidth=GRID_LINEWIDH,
        )
        if phasor_reference is not None:
            polar_reference = phasor_to_polar(
                *phasor_reference
            )  # type: ignore
        if polar_reference is None:
            polar_reference = (0.0, 1.0)
        ax = self._ax
        ax.plot(
            *phasor_calibrate(*phasor_semicircle(samples), *polar_reference),
            **kwargs,
        )
        if frequency is not None:
            if lifetime is None:
                # TODO: choose lifetimes based on frequency
                lifetime = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
            m = scale_matrix(1.05, (0.5, 0))
            for x, y, t in zip(
                *phasor_calibrate(
                    *phasor_from_lifetime(frequency, lifetime),
                    *polar_reference,
                ),
                lifetime,
            ):
                # TODO: use real ticks instead of annotation
                ax.annotate(
                    f'{t:g}',
                    xy=(x, y),
                    xytext=numpy.dot(m, (x, y, 1))[:2],
                    arrowprops=dict(arrowstyle='-', color=kwargs['color']),
                    color=kwargs['color'],
                    size=10,
                    ha='center',
                    va='center',
                )


def multi_frequency_plot(
    frequency: Sequence[float],
    phase: ArrayLike,
    modulation: ArrayLike,
    *,
    title: str | None = None,
) -> None:
    """Plot phase and modulation vs frequency.

    Parameters
    ----------
    frequency: sequence of float
        ...
    phase: array_like
        ...
    modulation: array_like
        ...
    title: str, optional
        ...

    """
    ax = pyplot.subplots()[1]
    ax.set_title('Multi-frequency plot' if title is None else title)
    ax.set_xscale('log', base=10)
    ax.set_xlabel('frequency (MHz)')
    ax.set_ylabel('phase (°)', color='tab:blue')
    ax.set_yticks([0.0, 30.0, 60.0, 90.0])
    for phi in numpy.array(phase, ndmin=2).swapaxes(0, 1):
        ax.plot(frequency, numpy.rad2deg(phi), color='tab:blue')
    ax = ax.twinx()
    ax.set_ylabel('modulation (%)', color='tab:red')
    ax.set_yticks([0.0, 25.0, 50.0, 75.0, 100.0])
    for mod in numpy.array(modulation, ndmin=2).swapaxes(0, 1):
        ax.plot(frequency, mod * 100, color='tab:red')
    pyplot.show()
