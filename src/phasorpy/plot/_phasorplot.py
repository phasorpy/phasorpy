"""PhasorPlot class."""

from __future__ import annotations

__all__ = ['PhasorPlot']

import math
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._typing import Any, ArrayLike, NDArray, IO

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

import numpy
from matplotlib import pyplot
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, Polygon
from matplotlib.path import Path
from matplotlib.patheffects import AbstractPathEffect

from .._phasorpy import _intersect_circle_circle, _intersect_circle_line
from .._utils import (
    dilate_coordinates,
    parse_kwargs,
    phasor_from_polar_scalar,
    phasor_to_polar_scalar,
    sort_coordinates,
    update_kwargs,
)
from ..phasor import (
    phasor_from_lifetime,
    phasor_semicircle,
    phasor_to_apparent_lifetime,
    phasor_transform,
)

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
        Show all quadrants of phasor space.
        By default, only the first quadrant with universal semicircle is shown.
    ax : matplotlib axes, optional
        Matplotlib axes used for plotting.
        By default, a new subplot axes is created.
    frequency : float, optional
        Laser pulse or modulation frequency in MHz.
    grid : bool, optional, default: True
        Display polar grid or universal semicircle.
    **kwargs
        Additional properties to set on `ax`.

    See Also
    --------
    phasorpy.plot.plot_phasor
    :ref:`sphx_glr_tutorials_api_phasorpy_phasorplot.py`

    """

    _ax: Axes
    """Matplotlib axes."""

    _limits: tuple[tuple[float, float], tuple[float, float]]
    """Axes limits (xmin, xmax), (ymin, ymax)."""

    _full: bool
    """Show all quadrants of phasor space."""

    _semicircle_ticks: SemicircleTicks | None
    """Last SemicircleTicks instance created."""

    _frequency: float
    """Laser pulse or modulation frequency in MHz."""

    def __init__(
        self,
        /,
        allquadrants: bool | None = None,
        ax: Axes | None = None,
        *,
        frequency: float | None = None,
        grid: bool = True,
        **kwargs: Any,
    ) -> None:
        # initialize empty phasor plot
        self._ax = pyplot.subplots()[1] if ax is None else ax
        self._ax.format_coord = (  # type: ignore[method-assign]
            self._on_format_coord
        )

        self._semicircle_ticks = None

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
            self._frequency = float(frequency)
            title += f' ({frequency:g} MHz)'
        else:
            self._frequency = 0.0

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
        try:
            # matplotlib >= 3.10.0
            return self._ax.get_figure(root=True)
        except TypeError:
            return self._ax.get_figure()  # type: ignore[return-value]

    @property
    def dataunit_to_point(self) -> float:
        """Factor to convert data to point unit."""
        fig = self.fig
        assert fig is not None
        length = fig.bbox_inches.height * self._ax.get_position().height * 72.0
        vrange: float = numpy.diff(self._ax.get_ylim()).item()
        return length / vrange

    def show(self) -> None:
        """Display all open figures. Call :py:func:`matplotlib.pyplot.show`."""
        # self.fig.show()
        pyplot.show()

    def save(
        self,
        file: str | os.PathLike[Any] | IO[bytes] | None,
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
        fmt: str = 'o',
        *,
        label: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> list[Line2D]:
        """Plot imaginary versus real coordinates as markers or lines.

        Parameters
        ----------
        real : array_like
            Real component of phasor coordinates.
            Must be one or two dimensional.
        imag : array_like
            Imaginary component of phasor coordinates.
            Must be of same shape as `real`.
        fmt : str, optional, default: 'o'
            Matplotlib style format string.
        label : sequence of str, optional
            Plot label.
            May be a sequence if phasor coordinates are two dimensional arrays.
        **kwargs
            Additional parameters passed to
            :py:meth:`matplotlib.axes.Axes.plot`.

        Returns
        -------
        list[matplotlib.lines.Line2D]
            Lines representing data plotted last.

        """
        lines = []
        if fmt == 'o':
            if 'marker' in kwargs:
                fmt = ''
                if 'linestyle' not in kwargs and 'ls' not in kwargs:
                    kwargs['linestyle'] = ''
        args = (fmt,) if fmt else ()
        ax = self._ax
        if label is not None and (
            isinstance(label, str) or not isinstance(label, Sequence)
        ):
            label = (label,)
        for (
            i,
            (re, im),
        ) in enumerate(
            zip(
                numpy.atleast_2d(numpy.asarray(real)),
                numpy.atleast_2d(numpy.asarray(imag)),
            )
        ):
            lbl = None
            if label is not None:
                try:
                    lbl = label[i]
                except IndexError:
                    pass
            lines = ax.plot(re, im, *args, label=lbl, **kwargs)
        if label is not None:
            ax.legend()
        self._reset_limits()
        return lines

    def _histogram2d(
        self,
        real: ArrayLike,
        imag: ArrayLike,
        /,
        **kwargs: Any,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Return two-dimensional histogram of imag versus real coordinates."""
        update_kwargs(kwargs, range=self._limits)
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
        return numpy.histogram2d(
            numpy.asanyarray(real).reshape(-1),
            numpy.asanyarray(imag).reshape(-1),
            **kwargs,
        )

    def _reset_limits(self) -> None:
        """Reset axes limits."""
        try:
            self._ax.set(xlim=self._limits[0], ylim=self._limits[1])
        except AttributeError:
            pass

    def hist2d(
        self,
        real: ArrayLike,
        imag: ArrayLike,
        /,
        **kwargs: Any,
    ) -> None:
        """Plot two-dimensional histogram of imag versus real coordinates.

        Parameters
        ----------
        real : array_like
            Real component of phasor coordinates.
        imag : array_like
            Imaginary component of phasor coordinates.
            Must be of same shape as `real`.
        **kwargs
            Additional parameters passed to :py:meth:`numpy.histogram2d`
            and :py:meth:`matplotlib.axes.Axes.pcolormesh`.

        """
        kwargs_hist2d = parse_kwargs(
            kwargs, 'bins', 'range', 'density', 'weights'
        )
        h, xedges, yedges = self._histogram2d(real, imag, **kwargs_hist2d)

        update_kwargs(kwargs, cmap='Blues', norm='log')
        cmin = kwargs.pop('cmin', 1)
        cmax = kwargs.pop('cmax', None)
        if cmin is not None:
            h[h < cmin] = None
        if cmax is not None:
            h[h > cmax] = None
        self._ax.pcolormesh(xedges, yedges, h.T, **kwargs)
        self._reset_limits()

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
            Additional parameters passed to :py:func:`numpy.histogram2d`
            and :py:meth:`matplotlib.axes.Axes.contour`.

        """
        if 'cmap' not in kwargs and 'colors' not in kwargs:
            kwargs['cmap'] = 'Blues'
        update_kwargs(kwargs, norm='log')
        kwargs_hist2d = parse_kwargs(
            kwargs, 'bins', 'range', 'density', 'weights'
        )
        h, xedges, yedges = self._histogram2d(real, imag, **kwargs_hist2d)
        xedges = xedges[:-1] + ((xedges[1] - xedges[0]) / 2.0)
        yedges = yedges[:-1] + ((yedges[1] - yedges[0]) / 2.0)
        self._ax.contour(xedges, yedges, h.T, **kwargs)
        self._reset_limits()

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
            :py:meth:`matplotlib.axes.Axes.imshow`.

        """
        raise NotImplementedError

    def components(
        self,
        real: ArrayLike,
        imag: ArrayLike,
        /,
        fraction: ArrayLike | None = None,
        labels: Sequence[str] | None = None,
        label_offset: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Plot linear combinations of phasor coordinates or ranges thereof.

        Parameters
        ----------
        real : (N,) array_like
            Real component of phasor coordinates.
        imag : (N,) array_like
            Imaginary component of phasor coordinates.
        fraction : (N,) array_like, optional
            Weight associated with each component.
            If None (default), outline the polygon area of possible linear
            combinations of components.
            Else, draw lines from the component coordinates to the weighted
            average.
        labels : Sequence of str, optional
            Text label for each component.
        label_offset : float, optional
            Distance of text label to component coordinate.
        **kwargs
            Additional parameters passed to
            :py:class:`matplotlib.patches.Polygon`,
            :py:class:`matplotlib.lines.Line2D`, or
            :py:class:`matplotlib.axes.Axes.annotate`

        """
        # TODO: use convex hull for outline
        # TODO: improve automatic placement of labels
        # TODO: catch more annotate properties?
        real, imag, indices = sort_coordinates(real, imag)

        label_ = kwargs.pop('label', None)
        marker = kwargs.pop('marker', None)
        color = kwargs.pop('color', None)
        fontsize = kwargs.pop('fontsize', 12)
        fontweight = kwargs.pop('fontweight', 'bold')
        horizontalalignment = kwargs.pop('horizontalalignment', 'center')
        verticalalignment = kwargs.pop('verticalalignment', 'center')
        if label_offset is None:
            label_offset = numpy.diff(self._ax.get_xlim()).item() * 0.04

        if labels is not None:
            if len(labels) != real.size:
                raise ValueError(
                    f'number labels={len(labels)} != components={real.size}'
                )
            labels = [labels[i] for i in indices]
            textposition = dilate_coordinates(real, imag, label_offset)
            for label, re, im, x, y in zip(labels, real, imag, *textposition):
                if not label:
                    continue
                self._ax.annotate(
                    label,
                    (re, im),
                    xytext=(x, y),
                    color=color,
                    fontsize=fontsize,
                    fontweight=fontweight,
                    horizontalalignment=horizontalalignment,
                    verticalalignment=verticalalignment,
                )

        if fraction is None:
            update_kwargs(
                kwargs,
                edgecolor=GRID_COLOR if color is None else color,
                linestyle=GRID_LINESTYLE,
                linewidth=GRID_LINEWIDH,
                fill=GRID_FILL,
            )
            self._ax.add_patch(Polygon(numpy.vstack((real, imag)).T, **kwargs))
            if marker is not None:
                self._ax.plot(
                    real,
                    imag,
                    marker=marker,
                    linestyle='',
                    color=color,
                    label=label_,
                )
                if label_ is not None:
                    self._ax.legend()
            return

        fraction = numpy.asarray(fraction)[indices]
        update_kwargs(
            kwargs,
            color=GRID_COLOR if color is None else color,
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
        if marker is not None:
            self._ax.plot(real, imag, marker=marker, linestyle='', color=color)
            self._ax.plot(
                center_re,
                center_im,
                marker=marker,
                linestyle='',
                color=color,
                label=label_,
            )
            if label_ is not None:
                self._ax.legend()

    def line(
        self,
        real: ArrayLike,
        imag: ArrayLike,
        /,
        **kwargs: Any,
    ) -> list[Line2D]:
        """Draw grid line.

        Parameters
        ----------
        real : array_like, shape (n, )
            Real components of line start and end coordinates.
        imag : array_like, shape (n, )
            Imaginary components of line start and end coordinates.
        **kwargs
            Additional parameters passed to
            :py:class:`matplotlib.lines.Line2D`.

        Returns
        -------
        list[matplotlib.lines.Line2D]
            List containing plotted line.

        """
        update_kwargs(
            kwargs,
            color=GRID_COLOR,
            linestyle=GRID_LINESTYLE,
            linewidth=GRID_LINEWIDH,
        )
        return [self._ax.add_line(Line2D(real, imag, **kwargs))]

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

    def arrow(
        self,
        point0: ArrayLike,
        point1: ArrayLike,
        /,
        *,
        angle: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Draw arrow between points.

        By default, draw a straight arrow with a `'-|>'` style, a mutation
        scale of 20, and a miter join style.

        Parameters
        ----------
        point0 : array_like
            X and y coordinates of start point of arrow.
        point1 : array_like
            X and y coordinates of end point of arrow.
        angle : float, optional
            Angle in radians, controlling curvature of line between points.
            If None (default), draw a straight line.
        **kwargs
            Additional parameters passed to
            :py:class:`matplotlib.patches.FancyArrowPatch`.

        """
        arrowstyle = kwargs.pop('arrowstyle', '-|>')
        mutation_scale = kwargs.pop('mutation_scale', 20)
        joinstyle = kwargs.pop('joinstyle', 'miter')
        if angle is not None:
            kwargs['connectionstyle'] = f'arc3,rad={math.tan(angle / 4.0)}'

        patch = FancyArrowPatch(
            point0,  # type: ignore[arg-type]
            point1,  # type: ignore[arg-type]
            arrowstyle=arrowstyle,
            mutation_scale=mutation_scale,
            # capstyle='projecting',
            joinstyle=joinstyle,
            **kwargs,
        )
        self._ax.add_patch(patch)

    def cursor(
        self,
        real: float,
        imag: float,
        /,
        real_limit: float | None = None,
        imag_limit: float | None = None,
        radius: float | None = None,
        radius_minor: float | None = None,
        angle: float | None = None,
        align_semicircle: bool = False,
        **kwargs: Any,
    ) -> None:
        """Plot phase and modulation grid lines and arcs at phasor coordinates.

        Parameters
        ----------
        real : float
            Real component of phasor coordinate.
        imag : float
            Imaginary component of phasor coordinate.
        real_limit : float, optional
            Real component of limiting phasor coordinate.
        imag_limit : float, optional
            Imaginary component of limiting phasor coordinate.
        radius : float, optional
            Radius of circle limiting phase and modulation grid lines and arcs.
        radius_minor : float, optional
            Radius of elliptic cursor along semi-minor axis.
            By default, `radius_minor` is equal to `radius`, that is,
            the ellipse is circular.
        angle : float, optional
            Rotation angle of semi-major axis of elliptic cursor in radians.
            If None (default), orient ellipse cursor according to
            `align_semicircle`.
        align_semicircle : bool, optional
            Determines elliptic cursor orientation if `angle` is not provided.
            If true, align the minor axis of the ellipse with the closest
            tangent on the universal semicircle, else align to the unit circle.
        **kwargs
            Additional parameters passed to
            :py:class:`matplotlib.lines.Line2D`,
            :py:class:`matplotlib.patches.Circle`,
            :py:class:`matplotlib.patches.Ellipse`, or
            :py:class:`matplotlib.patches.Arc`.

        See Also
        --------
        phasorpy.plot.PhasorPlot.polar_cursor

        """
        if real_limit is not None and imag_limit is not None:
            return self.polar_cursor(
                *phasor_to_polar_scalar(real, imag),
                *phasor_to_polar_scalar(real_limit, imag_limit),
                radius=radius,
                radius_minor=radius_minor,
                angle=angle,
                align_semicircle=align_semicircle,
                **kwargs,
            )
        return self.polar_cursor(
            *phasor_to_polar_scalar(real, imag),
            radius=radius,
            radius_minor=radius_minor,
            angle=angle,
            align_semicircle=align_semicircle,
            # _circle_only=True,
            **kwargs,
        )

    def polar_cursor(
        self,
        phase: float | None = None,
        modulation: float | None = None,
        phase_limit: float | None = None,
        modulation_limit: float | None = None,
        radius: float | None = None,
        radius_minor: float | None = None,
        angle: float | None = None,
        align_semicircle: bool = False,
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
        radius_minor : float, optional
            Radius of elliptic cursor along semi-minor axis.
            By default, `radius_minor` is equal to `radius`, that is,
            the ellipse is circular.
        angle : float, optional
            Rotation angle of semi-major axis of elliptic cursor in radians.
            If None (default), orient ellipse cursor according to
            `align_semicircle`.
        align_semicircle : bool, optional
            Determines elliptic cursor orientation if `angle` is not provided.
            If true, align the minor axis of the ellipse with the closest
            tangent on the universal semicircle, else align to the unit circle.
        **kwargs
            Additional parameters passed to
            :py:class:`matplotlib.lines.Line2D`,
            :py:class:`matplotlib.patches.Circle`,
            :py:class:`matplotlib.patches.Ellipse`, or
            :py:class:`matplotlib.patches.Arc`.

        See Also
        --------
        phasorpy.plot.PhasorPlot.cursor

        """
        update_kwargs(
            kwargs,
            color=GRID_COLOR,
            linestyle=GRID_LINESTYLE,
            linewidth=GRID_LINEWIDH,
            fill=GRID_FILL,
        )
        _circle_only = kwargs.pop('_circle_only', False)
        ax = self._ax
        if radius is not None and phase is not None and modulation is not None:
            x = modulation * math.cos(phase)
            y = modulation * math.sin(phase)
            if radius_minor is not None and radius_minor != radius:
                if angle is None:
                    if align_semicircle:
                        angle = math.atan2(y, x - 0.5)
                    else:
                        angle = phase
                angle = math.degrees(angle)
                ax.add_patch(
                    Ellipse(
                        (x, y),
                        radius * 2,
                        radius_minor * 2,
                        angle=angle,
                        **kwargs,
                    )
                )
                # TODO: implement gridlines intersecting with ellipse
                return None
            ax.add_patch(Circle((x, y), radius, **kwargs))
            if _circle_only:
                return None
            del kwargs['fill']
            x0, y0, x1, y1 = _intersect_circle_line(x, y, radius, 0, 0, x, y)
            ax.add_line(Line2D((x0, x1), (y0, y1), **kwargs))
            x0, y0, x1, y1 = _intersect_circle_circle(
                0, 0, modulation, x, y, radius
            )
            ax.add_patch(
                Arc(
                    (0, 0),
                    modulation * 2,
                    modulation * 2,
                    theta1=math.degrees(math.atan2(y0, x0)),
                    theta2=math.degrees(math.atan2(y1, x1)),
                    fill=False,
                    **kwargs,
                )
            )
            return None

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
        return None

    def polar_grid(self, **kwargs: Any) -> None:
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
        use_lines: bool = False,
        **kwargs: Any,
    ) -> list[Line2D]:
        """Draw universal semicircle.

        Parameters
        ----------
        frequency : float, optional
            Laser pulse or modulation frequency in MHz.
        polar_reference : (float, float), optional, default: (0, 1)
            Polar coordinates of zero lifetime.
        phasor_reference : (float, float), optional, default: (1, 0)
            Phasor coordinates of zero lifetime.
            Alternative to `polar_reference`.
        lifetime : sequence of float, optional
            Single component lifetimes at which to draw ticks and labels.
            Only applies when `frequency` is specified.
        labels : sequence of str, optional
            Tick labels. By default, the values of `lifetime`.
            Only applies when `frequency` and `lifetime` are specified.
        show_circle : bool, optional, default: True
            Draw universal semicircle.
        use_lines : bool, optional, default: False
            Draw universal semicircle using lines instead of arc.
        **kwargs
            Additional parameters passed to
            :py:class:`matplotlib.lines.Line2D` or
            :py:class:`matplotlib.patches.Arc` and
            :py:meth:`matplotlib.axes.Axes.plot`.

        Returns
        -------
        list[matplotlib.lines.Line2D]
            Lines representing plotted semicircle and ticks.

        """
        if frequency is not None:
            self._frequency = float(frequency)

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

        lines = []

        if show_circle:
            if use_lines:
                lines = [
                    ax.add_line(
                        Line2D(
                            *phasor_transform(
                                *phasor_semicircle(), *polar_reference
                            ),
                            **kwargs,
                        )
                    )
                ]
            else:
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
            lifetime, labels = _semicircle_ticks(frequency, lifetime, labels)
            self._semicircle_ticks = SemicircleTicks(labels=labels)
            lines.extend(
                ax.plot(
                    *phasor_transform(
                        *phasor_from_lifetime(frequency, lifetime),
                        *polar_reference,
                    ),
                    path_effects=[self._semicircle_ticks],
                    **kwargs,
                )
            )
        self._reset_limits()
        return lines

    def _on_format_coord(self, x: float, y: float) -> str:
        """Callback function to update coordinates displayed in toolbar."""
        phi, mod = phasor_to_polar_scalar(x, y)
        ret = [
            f'[{x:4.2f}, {y:4.2f}]',
            f'[{math.degrees(phi):.0f}°, {mod * 100:.0f}%]',
        ]
        if x > 0.0 and y > 0.0 and self._frequency > 0.0:
            tp, tm = phasor_to_apparent_lifetime(x, y, self._frequency)
            ret.append(f'[{tp:.2f}, {tm:.2f} ns]')
        return '  '.join(reversed(ret))


class SemicircleTicks(AbstractPathEffect):
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
        :py:meth:`matplotlib.patheffects.AbstractPathEffect._update_gc`.

    """

    _size: float  # tick length
    _labels: tuple[str, ...]  # tick labels
    _gc: dict[str, Any]  # keywords passed to _update_gc

    def __init__(
        self,
        size: float | None = None,
        labels: Sequence[str] | None = None,
        **kwargs: Any,
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

    @property
    def labels(self) -> tuple[str, ...]:
        """Tick labels."""
        return self._labels

    @labels.setter
    def labels(self, value: Sequence[str] | None, /) -> None:
        if value is None or not value:
            self._labels = ()
        else:
            self._labels = tuple(value)

    def draw_path(
        self,
        renderer: Any,
        gc: Any,
        tpath: Any,
        affine: Any,
        rgbFace: Any = None,
    ) -> None:
        """Draw path with updated gc."""
        gc0 = renderer.new_gc()
        gc0.copy_properties(gc)

        # TODO: this uses private methods of the base class
        gc0 = self._update_gc(gc0, self._gc)  # type: ignore[attr-defined]
        trans = affine
        trans += self._offset_transform(renderer)  # type: ignore[attr-defined]

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


def _semicircle_ticks(
    frequency: float,
    lifetime: Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    """Return semicircle tick lifetimes and labels at frequency."""
    if lifetime is None:
        lifetime = [0.0] + [
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
    return tuple(lifetime), tuple(labels)
