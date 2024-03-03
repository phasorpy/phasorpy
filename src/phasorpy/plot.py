"""Plot phasor coordinates and related data.

The ``phasorpy.plot`` module provides functions and classes to visualize
phasor coordinates and related data using the matplotlib library.

"""

from __future__ import annotations

__all__ = [
    'PhasorPlot',
    'plot_phasor',
    'plot_phasor_image',
    'plot_signal_image',
    'plot_polar_frequency',
]

import math
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, NDArray, Literal, BinaryIO

    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage
    from matplotlib.figure import Figure

import numpy
from matplotlib import patheffects, pyplot
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Polygon
from matplotlib.path import Path

from ._utils import (
    circle_circle_intersection,
    circle_line_intersection,
    parse_kwargs,
    phasor_from_polar_scalar,
    phasor_to_polar_scalar,
    sort_coordinates,
    update_kwargs,
)
from .phasor import phasor_calibrate, phasor_from_lifetime

GRID_COLOR = '0.5'
GRID_LINESTYLE = ':'
GRID_LINESTYLE_MAJOR = '-'
GRID_LINEWIDH = 1.0
GRID_LINEWIDH_MINOR = 0.5
GRID_FILL = False


class PhasorPlot:
    """Phasor plot.

    Create publication quality visualizations of phasor coordinates.

    Parameters
    ----------
    allquadrants : bool, optional
        Show all quandrants of phasor space.
        By default, only the first quadrant with universal semicricle is shown.
    ax : matplotlib axes, optional
        Matplotlib axes used for plotting.
        By default a new subplot axes is created.
    frequency : float, optional
        Laser pulse or modulation frequency in MHz.
    grid : bool, optional, default: False
        Display polar grid or semicircle.
    **kwargs
        Additional properties to set on `ax`.

    """

    _ax: Axes
    """Matplotlib axes."""

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
    ) -> None:
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
        """Matplotlib :py:class:`matplotlib.axes.Axes`."""
        return self._ax

    @property
    def fig(self) -> Figure | None:
        """Matplotlib :py:class:`matplotlib.figure.Figure`."""
        return self._ax.get_figure()

    def show(self) -> None:
        """Display all open figures. Call :py:func:`matplotlib.pyplot.show`."""
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
            :py:func:`matplotlib:pyplot.savefig`.

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
            Additional parameters passed to
            :py:meth:`matplotlib.axes.Axes.plot`.

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
                except IndexError:
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
            Additional parameters passed to
            py:meth:`matplotlib.axes.Axes.hist2d`.

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
            if aspect > 1:
                bins = (bins, max(int(bins / aspect), 1))
            else:
                bins = (max(int(bins * aspect), 1), bins)
        kwargs['bins'] = bins

        real = numpy.asanyarray(real).reshape(-1)
        imag = numpy.asanyarray(imag).reshape(-1)
        self._ax.hist2d(real, imag, **kwargs)

        # matplotlib's hist2d sets it's own axes limits, so reset it
        self._ax.set(xlim=self._limits[0], ylim=self._limits[1])

    def contour(
        self,
        real: ArrayLike,
        imag: ArrayLike,
        /,
        **kwargs: Any,
    ) -> None:
        """Plot contours of imag versus real coordinates (not implemented).

        Parameters
        ----------
        real : array_like
            Real component of phasor coordinates.
        imag : array_like
            Imaginary component of phasor coordinates.
            Must be of same shape as `real`.
        **kwargs
            Additional parameters passed to
            py:meth:`matplotlib.axes.Axes.contour`.

        """
        raise NotImplementedError

    def imshow(
        self,
        image: ArrayLike,
        /,
        **kwargs: Any,
    ) -> None:
        """Plot an image, for example, a 2D histogram (not implemented).

        Parameters
        ----------
        image : array_like
            Image to display.
        **kwargs
            Additional parameters passed to
            py:meth:`matplotlib.axes.Axes.imshow`.

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
            Weight associated with each component.
            If None (default), outline the polygon area of possible linear
            combinations of components.
            Else, draw lines from the component coordinates to the weighted
            average.
        **kwargs
            Additional parameters passed to
            :py:class:`matplotlib.patches.Polygon` or
            :py:class:`matplotlib.lines.Line2D`.

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

    def line(
        self,
        real: ArrayLike,
        imag: ArrayLike,
        /,
        **kwargs: Any,
    ) -> None:
        """Draw grid line.

        Parameters
        ----------
        real : array_like
            Real components of line start and end coordinates.
        imag : array_like
            Imaginary components of line start and end coordinates.
        **kwargs
            Additional parameters passed to
            :py:class:`matplotlib.lines.Line2D`.

        """
        update_kwargs(
            kwargs,
            color=GRID_COLOR,
            linestyle=GRID_LINESTYLE,
            linewidth=GRID_LINEWIDH,
        )
        self._ax.add_line(Line2D(real, imag, **kwargs))

    def circle(
        self,
        real: float,
        imag: float,
        /,
        radius: float,
        **kwargs: Any,
    ) -> None:
        """Draw grid circle of radius around center.

        Parameters
        ----------
        real : float
            Real component of circle center coordinate.
        imag : float
            Imaginary component of circle center coordinate.
        radius : float
            Circle radius.
        **kwargs
            Additional parameters passed to
            :py:class:`matplotlib.patches.Circle`.

        """
        update_kwargs(
            kwargs,
            color=GRID_COLOR,
            linestyle=GRID_LINESTYLE,
            linewidth=GRID_LINEWIDH,
            fill=GRID_FILL,
        )
        self._ax.add_patch(Circle((real, imag), radius, **kwargs))

    def polar_cursor(
        self,
        phase: float | None = None,
        modulation: float | None = None,
        phase_limit: float | None = None,
        modulation_limit: float | None = None,
        radius: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Plot phase and modulation grid lines and arcs.

        Parameters
        ----------
        phase : float, optional
            Angular component of polar coordinate in radians.
        modulation : float, optional
            Radial component of polar coordinate.
        phase_limit : float, optional
            Angular component of limiting polar coordinate (in radians).
            Modulation grid arcs are drawn between `phase` and `phase_limit`.
        modulation_limit : float, optional
            Radial component of limiting polar coordinate.
            Phase grid lines are drawn from `modulation` to `modulation_limit`.
        radius : float, optional
            Radius of circle limiting phase and modulation grid lines and arcs.
        **kwargs
            Additional parameters passed to
            :py:class:`matplotlib.lines.Line2D`,
            :py:class:`matplotlib.patches.Circle`, and
            :py:class:`matplotlib.patches.Arc`.

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
            ax.add_patch(Circle((x, y), radius, **kwargs))
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
            Parameters passed to
            :py:class:`matplotlib.patches.Circle` and
            :py:class:`matplotlib.lines.Line2D`.

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
        ax.add_patch(Circle((0, 0), 1, fill=False, **kwargs))
        # minor gridlines
        kwargs = kwargs_copy
        update_kwargs(
            kwargs,
            color=GRID_COLOR,
            linestyle=GRID_LINESTYLE,
            linewidth=GRID_LINEWIDH_MINOR,
        )
        for r in (1 / 3, 2 / 3):
            ax.add_patch(Circle((0, 0), r, fill=False, **kwargs))
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
        labels: Sequence[str] | None = None,
        show_circle: bool = True,
        **kwargs,
    ) -> None:
        """Draw universal semicircle.

        Parameters
        ----------
        frequency : float, optional
            Laser pulse or modulation frequency in MHz.
        polar_reference : (float, float), optional
            Polar coordinates of zero lifetime. The default is (0, 1).
        phasor_reference : (float, float), optional
            Phasor coordinates of zero lifetime.
            Alternative to `polar_reference`. The default is (1, 0).
        lifetime : sequence of float, optional
            Apparent single lifetimes at which to draw ticks and labels.
            Only applies when `frequency` is specified.
        labels : sequence of str, optional
            Tick labels. By default, the values of `lifetime`.
            Only applies when `frequency` and `lifetime` are specified.
        show_circle : bool, optional
            Draw universal semicircle.
        **kwargs
            Additional parameters passed to
            :py:class:`matplotlib.patches.Arc` and
            :py:meth:`matplotlib.axes.Axes.plot`.

        """
        update_kwargs(
            kwargs,
            color=GRID_COLOR,
            linestyle=GRID_LINESTYLE_MAJOR,
            linewidth=GRID_LINEWIDH,
        )
        if phasor_reference is not None:
            polar_reference = phasor_to_polar_scalar(*phasor_reference)
        if polar_reference is None:
            polar_reference = (0.0, 1.0)
        if phasor_reference is None:
            phasor_reference = phasor_from_polar_scalar(*polar_reference)
        ax = self._ax

        if show_circle:
            ax.add_patch(
                Arc(
                    (phasor_reference[0] / 2, phasor_reference[1] / 2),
                    polar_reference[1],
                    polar_reference[1],
                    theta1=math.degrees(polar_reference[0]),
                    theta2=math.degrees(polar_reference[0]) + 180.0,
                    fill=False,
                    **kwargs,
                )
            )

        if frequency is not None and polar_reference == (0.0, 1.0):
            # draw ticks and labels
            if lifetime is None:
                lifetime = [0] + [
                    2**t
                    for t in range(-8, 32)
                    if phasor_from_lifetime(frequency, 2**t)[1] >= 0.18
                ]
                unit = 'ns'
            else:
                unit = ''
            if labels is None:
                labels = [f'{tau:g}' for tau in lifetime]
                try:
                    labels[2] = f'{labels[2]} {unit}'
                except IndexError:
                    pass
            ax.plot(
                *phasor_calibrate(
                    *phasor_from_lifetime(frequency, lifetime),
                    *polar_reference,
                ),
                path_effects=[SemicircleTicks(labels=labels)],
                **kwargs,
            )


class SemicircleTicks(patheffects.AbstractPathEffect):
    """Draw ticks on universal semicircle.

    Parameters
    ----------
    size : float, optional
        Length of tick in dots.
        The default is ``rcParams['xtick.major.size']``.
    labels : sequence of str, optional
        Tick labels for each vertex in path.
    **kwargs
        Extra keywords passed to matplotlib's
        ``AbstractPathEffect._update_gc``.

    """

    _size: float  # tick length
    _labels: tuple[str, ...]  # tick labels
    _gc: dict[str, Any]  # keywords passed to _update_gc

    def __init__(
        self,
        size: float | None = None,
        labels: Sequence[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__((0.0, 0.0))

        if size is None:
            self._size = pyplot.rcParams['xtick.major.size']
        else:
            self._size = size
        if labels is None or not labels:
            self._labels = ()
        else:
            self._labels = tuple(labels)
        self._gc = kwargs

    def draw_path(self, renderer, gc, tpath, affine, rgbFace=None) -> None:
        """Draw path with updated gc."""
        gc0 = renderer.new_gc()
        gc0.copy_properties(gc)

        # TODO: this uses private methods of the base class
        gc0 = self._update_gc(gc0, self._gc)  # type: ignore
        trans = affine + self._offset_transform(renderer)  # type: ignore

        font = FontProperties()
        # approximate half size of 'x'
        fontsize = renderer.points_to_pixels(font.get_size_in_points()) / 4
        size = renderer.points_to_pixels(self._size)
        origin = affine.transform([[0.5, 0.0]])

        transpath = affine.transform_path(tpath)
        polys = transpath.to_polygons(closed_only=False)

        for p in polys:
            # coordinates of tick ends
            t = p - origin
            t /= numpy.hypot(t[:, 0], t[:, 1])[:, numpy.newaxis]
            d = t.copy()
            t *= size
            t += p

            xyt = numpy.empty((2 * p.shape[0], 2))
            xyt[0::2] = p
            xyt[1::2] = t

            renderer.draw_path(
                gc0,
                Path(xyt, numpy.tile([Path.MOVETO, Path.LINETO], p.shape[0])),
                affine.inverted() + trans,
                rgbFace,
            )

            if not self._labels:
                continue
            # coordinates of labels
            t = d * size * 2.5
            t += p

            if renderer.flipy():
                h = renderer.get_canvas_width_height()[1]
            else:
                h = 0.0

            for s, (x, y), (dx, _) in zip(self._labels, t, d):
                # TODO: get rendered text size from matplotlib.text.Text?
                # this did not work:
                # Text(d[i,0], h - d[i,1], label, ha='center', va='center')
                x = x + fontsize * len(s.split()[0]) * (dx - 1.0)
                y = h - y + fontsize
                renderer.draw_text(gc0, x, y, s, font, 0.0)

        gc0.restore()


def plot_phasor(
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
        Method used to plot phasor coordinates:
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
    show : bool, optional, default: True
        Display figure.
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


def plot_phasor_image(
    mean: ArrayLike | None,
    real: ArrayLike,
    imag: ArrayLike,
    *,
    harmonics: int | None = None,
    percentile: float | None = None,
    title: str | None = None,
    show: bool = True,
    **kwargs: Any,
) -> None:
    """Plot phasor coordinates as images.

    Preview phasor coordinates from time-resolved or hyperspectral
    image stacks as returned by :py:func:`phasorpy.phasor.phasor_from_signal`.

    The last two axes are assumed to be the image axes.
    Harmonics, if any, are in the first axes of `real` and `imag`.
    Other axes are averaged for display.

    Parameters
    ----------
    mean : array_like
        Image average. Must be two or more dimensional, or None.
    real : array_like
        Image of real component of phasor coordinates.
        The last dimensions must match shape of `mean`.
    imag : array_like
        Image of imaginary component of phasor coordinates.
        Must be same shape as `real`.
    harmonics : int, optional
        Number of harmonics to display.
        If `mean` is None, a nonzero value indicates the presence of harmonics
        in the first axes of `mean` and `real`. Else, the presence of harmonics
        is determined from the shapes of `mean` and `real`.
        By default, up to 4 harmonics are displayed.
    percentile : float, optional
        The (q, 100-q) percentiles of image data are covered by colormaps.
        By default, the complete value range of `mean` is covered,
        for `real` and `imag` the range [-1..1].
    title : str, optional
        Figure title.
    show : bool, optional, default: True
        Display figure.
    **kwargs
        Additional arguments passed to :func:`matplotlib.pyplot.imshow`.

    Raises
    ------
    ValueError
        The shapes of `mean`, `real`, and `image` do not match.
        Percentile is out of range.

    """
    update_kwargs(kwargs, interpolation='nearest')
    cmap = kwargs.pop('cmap', None)
    shape = None

    if mean is not None:
        mean = numpy.asarray(mean)
        if mean.ndim < 2:
            raise ValueError(f'not an image {mean.ndim=} < 2')
        shape = mean.shape
        mean = numpy.mean(mean.reshape(-1, *mean.shape[-2:]), axis=0)

    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    if real.ndim < 2:
        raise ValueError(f'not an image {real.ndim=} < 2')

    if (shape is not None and real.shape[1:] == shape) or (
        shape is None and harmonics
    ):
        # first image dimension contains harmonics
        if real.ndim < 3:
            raise ValueError(f'not a multi-harmonic image {real.shape=}')
        nh = real.shape[0]  # number harmonics
    elif shape is None or shape == real.shape:
        # single harmonic
        nh = 1
    else:
        raise ValueError(f'shape mismatch {real.shape[1:]=} != {shape}')

    # average extra image dimensions, but not harmonics
    real = numpy.mean(real.reshape(nh, -1, *real.shape[-2:]), axis=1)
    imag = numpy.mean(imag.reshape(nh, -1, *imag.shape[-2:]), axis=1)

    # for MyPy
    assert isinstance(mean, numpy.ndarray) or mean is None
    assert isinstance(real, numpy.ndarray)
    assert isinstance(imag, numpy.ndarray)

    # limit number of displayed harmonics
    nh = min(4 if harmonics is None else harmonics, nh)

    # create figure with size depending on image aspect and number of harmonics
    fig = pyplot.figure(layout='constrained')
    w, h = fig.get_size_inches()
    aspect = min(1.0, max(0.5, real.shape[-2] / real.shape[-1]))
    fig.set_size_inches(w, h * 0.4 * aspect * nh + h * 0.25 * aspect)
    gs = GridSpec(nh, 2 if mean is None else 3, figure=fig)
    if title:
        fig.suptitle(title)

    if mean is not None:
        _imshow(
            fig.add_subplot(gs[0, 0]),
            mean,
            percentile=percentile,
            vmin=None,
            vmax=None,
            cmap=cmap,
            axis=True,
            title='mean',
            **kwargs,
        )

    if percentile is None:
        vmin = -1.0
        vmax = 1.0
        if cmap is None:
            cmap = 'coolwarm_r'
    else:
        vmin = None
        vmax = None

    for h in range(nh):
        axs = []
        ax = fig.add_subplot(gs[h, -2])
        axs.append(ax)
        _imshow(
            ax,
            real[h],
            percentile=percentile,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            axis=mean is None and h == 0,
            colorbar=percentile is not None,
            title=None if h else 'G, real',
            **kwargs,
        )

        ax = fig.add_subplot(gs[h, -1])
        axs.append(ax)
        pos = _imshow(
            ax,
            imag[h],
            percentile=percentile,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            axis=False,
            colorbar=percentile is not None,
            title=None if h else 'S, imag',
            **kwargs,
        )
        if percentile is None and h == 0:
            fig.colorbar(pos, ax=axs, shrink=0.4, location='bottom')

    if show:
        pyplot.show()


def plot_signal_image(
    signal: ArrayLike,
    /,
    *,
    axis: int | None = None,
    percentile: float | Sequence[float] | None = None,
    title: str | None = None,
    show: bool = True,
    **kwargs: Any,
) -> None:
    """Plot average image and signal along axis.

    Preview time-resolved or hyperspectral image stacks to be anayzed with
    :py:func:`phasorpy.phasor.phasor_from_signal`.

    The last two axes, excluding `axis`, are assumed to be the image axes.
    Other axes are averaged for image display.

    Parameters
    ----------
    signal : array_like
        Image stack. Must be three or more dimensional.
    axis : int, optional, default: -1
        Axis over which phasor coordinates would be computed.
        The default is the last axis (-1).
    percentile : float or [float, float], optional
        The [q, 100-q] percentiles of image data are covered by colormaps.
        By default, the complete value range of `mean` is covered,
        for `real` and `imag` the range [-1..1].
    title : str, optional
        Figure title.
    show : bool, optional, default: True
        Display figure.
    **kwargs
        Additional arguments passed to :func:`matplotlib.pyplot.imshow`.

    Raises
    ------
    ValueError
        Signal is not an image stack.
        Percentile is out of range.

    """
    # TODO: add option to separate channels?
    # TODO: add option to plot non-images?
    update_kwargs(kwargs, interpolation='nearest')
    signal = numpy.asarray(signal)
    if signal.ndim < 3:
        raise ValueError(f'not an image stack {signal.ndim=} < 3')

    if axis is None:
        axis = -1
    axis %= signal.ndim

    # for MyPy
    assert isinstance(signal, numpy.ndarray)

    fig = pyplot.figure(layout='constrained')
    if title:
        fig.suptitle(title)
    w, h = fig.get_size_inches()
    fig.set_size_inches(w, h * 0.7)
    gs = GridSpec(1, 2, figure=fig, width_ratios=(1, 1))

    # histogram
    axes = list(range(signal.ndim))
    del axes[axis]
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title(f'mean, axis {axis}')
    ax.plot(signal.mean(axis=tuple(axes)))

    # image
    axes = list(sorted(axes[:-2] + [axis]))
    ax = fig.add_subplot(gs[0, 0])
    _imshow(
        ax,
        signal.mean(axis=tuple(axes)),
        percentile=percentile,
        shrink=0.5,
        title='mean',
    )

    if show:
        pyplot.show()


def plot_polar_frequency(
    frequency: ArrayLike,
    phase: ArrayLike,
    modulation: ArrayLike,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    show: bool = True,
    **kwargs,
) -> None:
    """Plot phase and modulation verus frequency.

    Parameters
    ----------
    frequency : array_like, shape (n, )
        Laser pulse or modulation frequency in MHz.
    phase : array_like
        Angular component of polar coordinates in radians.
    modulation : array_like
        Radial component of polar coordinates.
    ax : matplotlib axes, optional
        Matplotlib axes used for plotting.
        By default a new subplot axes is created.
    title : str, optional
        Figure title.
    show : bool, optional, default: True
        Display figure.
    **kwargs
        Additional arguments passed to :py:func:`matplotlib.pyplot.plot`.

    """
    # TODO: make this customizable: labels, colors, ...
    if ax is None:
        ax = pyplot.subplots()[1]
    if title is None:
        title = 'Multi-frequency plot'
    if title:
        ax.set_title(title)
    ax.set_xscale('log', base=10)
    ax.set_xlabel('frequency (MHz)')

    phase = numpy.asarray(phase)
    if phase.ndim < 2:
        phase = phase.reshape(-1, 1)
    modulation = numpy.asarray(modulation)
    if modulation.ndim < 2:
        modulation = modulation.reshape(-1, 1)

    ax.set_ylabel('phase (°)', color='tab:blue')
    ax.set_yticks([0.0, 30.0, 60.0, 90.0])
    for phi in phase.T:
        ax.plot(frequency, numpy.rad2deg(phi), color='tab:blue', **kwargs)
    ax = ax.twinx()  # type: ignore

    ax.set_ylabel('modulation (%)', color='tab:red')
    ax.set_yticks([0.0, 25.0, 50.0, 75.0, 100.0])
    for mod in modulation.T:
        ax.plot(frequency, mod * 100, color='tab:red', **kwargs)
    if show:
        pyplot.show()


def _imshow(
    ax: Axes,
    image: NDArray[Any],
    /,
    *,
    percentile: float | Sequence[float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    shrink: float | None = None,
    axis: bool = True,
    title: str | None = None,
    **kwargs,
) -> AxesImage:
    """Plot image array.

    Convenience wrapper around :py:func:`matplotlib.pyplot.imshow`.

    """
    update_kwargs(kwargs, interpolation='none')
    if percentile is not None:
        if isinstance(percentile, Sequence):
            percentile = percentile[0], percentile[1]
        else:
            # percentile = max(0.0, min(50, percentile))
            percentile = percentile, 100.0 - percentile
        if (
            percentile[0] >= percentile[1]
            or percentile[0] < 0
            or percentile[1] > 100
        ):
            raise ValueError(f'{percentile=} out of range')
        vmin, vmax = numpy.percentile(image, percentile)
    pos = ax.imshow(image, vmin=vmin, vmax=vmax, **kwargs)
    if colorbar:
        if percentile is not None and vmin is not None and vmax is not None:
            ticks = vmin, vmax
        else:
            ticks = None
        fig = ax.get_figure()
        if fig is not None:
            if shrink is None:
                shrink = 0.8
            fig.colorbar(pos, shrink=shrink, location='bottom', ticks=ticks)
    if title:
        ax.set_title(title)
    if not axis:
        ax.set_axis_off()
    # ax.set_anchor('C')
    return pos
