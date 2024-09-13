"""Plot phasor coordinates and related data.

The ``phasorpy.plot`` module provides functions and classes to visualize
phasor coordinates and related data using the
`matplotlib <https://matplotlib.org/>`_ library.

"""

from __future__ import annotations

__all__ = [
    'PhasorPlot',
    'PhasorPlotFret',
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
    from ._typing import Any, ArrayLike, NDArray, Literal, IO

    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage
    from matplotlib.figure import Figure

import numpy
from matplotlib import pyplot
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Ellipse, Polygon
from matplotlib.path import Path
from matplotlib.patheffects import AbstractPathEffect
from matplotlib.widgets import Slider

from ._phasorpy import _intersection_circle_circle, _intersection_circle_line
from ._utils import (
    dilate_coordinates,
    parse_kwargs,
    phasor_from_polar_scalar,
    phasor_to_polar_scalar,
    sort_coordinates,
    update_kwargs,
)
from .phasor import (
    phasor_from_fret_acceptor,
    phasor_from_fret_donor,
    phasor_from_lifetime,
    phasor_semicircle,
    phasor_to_apparent_lifetime,
    phasor_to_polar,
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
        Show all quandrants of phasor space.
        By default, only the first quadrant with universal semicircle is shown.
    ax : matplotlib axes, optional
        Matplotlib axes used for plotting.
        By default, a new subplot axes is created.
    frequency : float, optional
        Laser pulse or modulation frequency in MHz.
    grid : bool, optional, default: True
        Display polar grid or semicircle.
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
        return self._ax.get_figure()

    @property
    def dataunit_to_point(self) -> float:
        """Factor to convert data to point unit."""
        fig = self._ax.get_figure()
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
        label: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> list[Line2D]:
        """Plot imag versus real coordinates as markers and/or lines.

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
        label : str or sequence of str, optional
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
        """Return 2D histogram of imag versus real coordinates."""
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
        """Plot 2D histogram of imag versus real coordinates.

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
        update_kwargs(kwargs, cmap='Blues', norm='log')
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
            x0, y0, x1, y1 = _intersection_circle_line(
                x, y, radius, 0, 0, x, y
            )
            ax.add_line(Line2D((x0, x1), (y0, y1), **kwargs))
            x0, y0, x1, y1 = _intersection_circle_circle(
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


class PhasorPlotFret(PhasorPlot):
    """FRET phasor plot.

    Plot Förster Resonance Energy Transfer efficiency trajectories
    of donor and acceptor channels in phasor space.

    Parameters
    ----------
    frequency : array_like
        Laser pulse or modulation frequency in MHz.
    donor_lifetime : array_like
        Lifetime of donor without FRET in ns.
    acceptor_lifetime : array_like
        Lifetime of acceptor in ns.
    fret_efficiency : array_like, optional, default 0
        FRET efficiency in range [0..1].
    donor_freting : array_like, optional, default 1
        Fraction of donors participating in FRET. Range [0..1].
    donor_bleedthrough : array_like, optional, default 0
        Weight of donor fluorescence in acceptor channel
        relative to fluorescence of fully sensitized acceptor.
        A weight of 1 means the fluorescence from donor and fully sensitized
        acceptor are equal.
        The background in the donor channel does not bleed through.
    acceptor_bleedthrough : array_like, optional, default 0
        Weight of fluorescence from directly excited acceptor
        relative to fluorescence of fully sensitized acceptor.
        A weight of 1 means the fluorescence from directly excited acceptor
        and fully sensitized acceptor are equal.
    acceptor_background : array_like, optional, default 0
        Weight of background fluorescence in acceptor channel
        relative to fluorescence of fully sensitized acceptor.
        A weight of 1 means the fluorescence of background and fully
        sensitized acceptor are equal.
    donor_background : array_like, optional, default 0
        Weight of background fluorescence in donor channel
        relative to fluorescence of donor without FRET.
        A weight of 1 means the fluorescence of background and donor
        without FRET are equal.
    background_real : array_like, optional, default 0
        Real component of background fluorescence phasor coordinate
        at `frequency`.
    background_imag : array_like, optional, default 0
        Imaginary component of background fluorescence phasor coordinate
        at `frequency`.
    ax : matplotlib axes, optional
        Matplotlib axes used for plotting.
        By default, a new subplot axes is created.
        Cannot be used with `interactive` mode.
    interactive : bool, optional, default: False
        Use matplotlib slider widgets to interactively control parameters.
    **kwargs
        Additional parameters passed to :py:class:`phasorpy.plot.PhasorPlot`.

    See Also
    --------
    phasorpy.phasor.phasor_from_fret_donor
    phasorpy.phasor.phasor_from_fret_acceptor
    :ref:`sphx_glr_tutorials_api_phasorpy_fret.py`

    """

    _fret_efficiencies: NDArray[Any]

    _frequency_slider: Slider
    _donor_lifetime_slider: Slider
    _acceptor_lifetime_slider: Slider
    _fret_efficiency_slider: Slider
    _donor_freting_slider: Slider
    _donor_bleedthrough_slider: Slider
    _acceptor_bleedthrough_slider: Slider
    _acceptor_background_slider: Slider
    _donor_background_slider: Slider
    _background_real_slider: Slider
    _background_imag_slider: Slider

    _donor_line: Line2D
    _donor_only_line: Line2D
    _donor_fret_line: Line2D
    _donor_trajectory_line: Line2D
    _donor_semicircle_line: Line2D
    _donor_donor_line: Line2D
    _donor_background_line: Line2D
    _acceptor_line: Line2D
    _acceptor_only_line: Line2D
    _acceptor_trajectory_line: Line2D
    _acceptor_semicircle_line: Line2D
    _acceptor_background_line: Line2D
    _background_line: Line2D

    _donor_semicircle_ticks: SemicircleTicks | None

    def __init__(
        self,
        *,
        frequency: float = 60.0,
        donor_lifetime: float = 4.2,
        acceptor_lifetime: float = 3.0,
        fret_efficiency: float = 0.5,
        donor_freting: float = 1.0,
        donor_bleedthrough: float = 0.0,
        acceptor_bleedthrough: float = 0.0,
        acceptor_background: float = 0.0,
        donor_background: float = 0.0,
        background_real: float = 0.0,
        background_imag: float = 0.0,
        ax: Axes | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> None:
        update_kwargs(
            kwargs,
            title='FRET phasor plot',
            xlim=[-0.2, 1.1],
            ylim=[-0.1, 0.8],
        )
        kwargs['allquadrants'] = False
        kwargs['grid'] = False

        if ax is not None:
            interactive = False
        else:
            fig = pyplot.figure()
            ax = fig.add_subplot()
            if interactive:
                w, h = fig.get_size_inches()
                fig.set_size_inches(w, h * 1.66)
                fig.subplots_adjust(bottom=0.45)
                fcm = fig.canvas.manager
                if fcm is not None:
                    fcm.set_window_title(kwargs['title'])

        super().__init__(ax=ax, **kwargs)

        self._fret_efficiencies = numpy.linspace(0.0, 1.0, 101)

        donor_real, donor_imag = phasor_from_lifetime(
            frequency, donor_lifetime
        )
        donor_fret_real, donor_fret_imag = phasor_from_lifetime(
            frequency, donor_lifetime * (1.0 - fret_efficiency)
        )
        acceptor_real, acceptor_imag = phasor_from_lifetime(
            frequency, acceptor_lifetime
        )
        donor_trajectory_real, donor_trajectory_imag = phasor_from_fret_donor(
            frequency,
            donor_lifetime,
            fret_efficiency=self._fret_efficiencies,
            donor_freting=donor_freting,
            donor_background=donor_background,
            background_real=background_real,
            background_imag=background_imag,
        )
        (
            acceptor_trajectory_real,
            acceptor_trajectory_imag,
        ) = phasor_from_fret_acceptor(
            frequency,
            donor_lifetime,
            acceptor_lifetime,
            fret_efficiency=self._fret_efficiencies,
            donor_freting=donor_freting,
            donor_bleedthrough=donor_bleedthrough,
            acceptor_bleedthrough=acceptor_bleedthrough,
            acceptor_background=acceptor_background,
            background_real=background_real,
            background_imag=background_imag,
        )

        # add plots
        lines = self.semicircle(frequency=frequency)
        self._donor_semicircle_line = lines[0]
        self._donor_semicircle_ticks = self._semicircle_ticks

        lines = self.semicircle(
            phasor_reference=(float(acceptor_real), float(acceptor_imag)),
            use_lines=True,
        )
        self._acceptor_semicircle_line = lines[0]

        if donor_freting < 1.0 and donor_background == 0.0:
            lines = self.line(
                [donor_real, donor_fret_real],
                [donor_imag, donor_fret_imag],
            )
        else:
            lines = self.line([0.0, 0.0], [0.0, 0.0])
        self._donor_donor_line = lines[0]

        if acceptor_background > 0.0:
            lines = self.line(
                [float(acceptor_real), float(background_real)],
                [float(acceptor_imag), float(background_imag)],
            )
        else:
            lines = self.line([0.0, 0.0], [0.0, 0.0])
        self._acceptor_background_line = lines[0]

        if donor_background > 0.0:
            lines = self.line(
                [float(donor_real), float(background_real)],
                [float(donor_imag), float(background_imag)],
            )
        else:
            lines = self.line([0.0, 0.0], [0.0, 0.0])
        self._donor_background_line = lines[0]

        lines = self.plot(
            donor_trajectory_real,
            donor_trajectory_imag,
            '-',
            color='tab:green',
        )
        self._donor_trajectory_line = lines[0]

        lines = self.plot(
            acceptor_trajectory_real,
            acceptor_trajectory_imag,
            '-',
            color='tab:red',
        )
        self._acceptor_trajectory_line = lines[0]

        lines = self.plot(
            donor_real,
            donor_imag,
            '.',
            color='tab:green',
        )
        self._donor_only_line = lines[0]

        lines = self.plot(
            donor_real,
            donor_imag,
            '.',
            color='tab:green',
        )
        self._donor_fret_line = lines[0]

        lines = self.plot(
            acceptor_real,
            acceptor_imag,
            '.',
            color='tab:red',
        )
        self._acceptor_only_line = lines[0]

        lines = self.plot(
            donor_trajectory_real[int(fret_efficiency * 100.0)],
            donor_trajectory_imag[int(fret_efficiency * 100.0)],
            'o',
            color='tab:green',
            label='Donor',
        )
        self._donor_line = lines[0]

        lines = self.plot(
            acceptor_trajectory_real[int(fret_efficiency * 100.0)],
            acceptor_trajectory_imag[int(fret_efficiency * 100.0)],
            'o',
            color='tab:red',
            label='Acceptor',
        )
        self._acceptor_line = lines[0]

        lines = self.plot(
            background_real,
            background_imag,
            'o',
            color='black',
            label='Background',
        )
        self._background_line = lines[0]

        if not interactive:
            return

        # add sliders
        axes = []
        for i in range(11):
            axes.append(fig.add_axes((0.33, 0.05 + i * 0.03, 0.45, 0.01)))

        self._frequency_slider = Slider(
            ax=axes[10],
            label='Frequency ',
            valfmt=' %.0f MHz',
            valmin=10,
            valmax=200,
            valstep=1,
            valinit=frequency,
        )
        self._frequency_slider.on_changed(self._on_semicircle_changed)

        self._donor_lifetime_slider = Slider(
            ax=axes[9],
            label='Donor lifetime ',
            valfmt=' %.1f ns',
            valmin=0.1,
            valmax=16.0,
            valstep=0.1,
            valinit=donor_lifetime,
            # facecolor='tab:green',
            handle_style={'edgecolor': 'tab:green'},
        )
        self._donor_lifetime_slider.on_changed(self._on_changed)

        self._acceptor_lifetime_slider = Slider(
            ax=axes[8],
            label='Acceptor lifetime ',
            valfmt=' %.1f ns',
            valmin=0.1,
            valmax=16.0,
            valstep=0.1,
            valinit=acceptor_lifetime,
            # facecolor='tab:red',
            handle_style={'edgecolor': 'tab:red'},
        )
        self._acceptor_lifetime_slider.on_changed(self._on_semicircle_changed)

        self._fret_efficiency_slider = Slider(
            ax=axes[7],
            label='FRET efficiency ',
            valfmt=' %.2f',
            valmin=0.0,
            valmax=1.0,
            valstep=0.01,
            valinit=fret_efficiency,
        )
        self._fret_efficiency_slider.on_changed(self._on_changed)

        self._donor_freting_slider = Slider(
            ax=axes[6],
            label='Donors FRETing ',
            valfmt=' %.2f',
            valmin=0.0,
            valmax=1.0,
            valstep=0.01,
            valinit=donor_freting,
            # facecolor='tab:green',
            handle_style={'edgecolor': 'tab:green'},
        )
        self._donor_freting_slider.on_changed(self._on_changed)

        self._donor_bleedthrough_slider = Slider(
            ax=axes[5],
            label='Donor bleedthrough ',
            valfmt=' %.2f',
            valmin=0.0,
            valmax=5.0,
            valstep=0.01,
            valinit=donor_bleedthrough,
            # facecolor='tab:red',
            handle_style={'edgecolor': 'tab:red'},
        )
        self._donor_bleedthrough_slider.on_changed(self._on_changed)

        self._acceptor_bleedthrough_slider = Slider(
            ax=axes[4],
            label='Acceptor bleedthrough ',
            valfmt=' %.2f',
            valmin=0.0,
            valmax=5.0,
            valstep=0.01,
            valinit=acceptor_bleedthrough,
            # facecolor='tab:red',
            handle_style={'edgecolor': 'tab:red'},
        )
        self._acceptor_bleedthrough_slider.on_changed(self._on_changed)

        self._acceptor_background_slider = Slider(
            ax=axes[3],
            label='Acceptor background ',
            valfmt=' %.2f',
            valmin=0.0,
            valmax=5.0,
            valstep=0.01,
            valinit=acceptor_background,
            # facecolor='tab:red',
            handle_style={'edgecolor': 'tab:red'},
        )
        self._acceptor_background_slider.on_changed(self._on_changed)

        self._donor_background_slider = Slider(
            ax=axes[2],
            label='Donor background ',
            valfmt=' %.2f',
            valmin=0.0,
            valmax=5.0,
            valstep=0.01,
            valinit=donor_background,
            # facecolor='tab:green',
            handle_style={'edgecolor': 'tab:green'},
        )
        self._donor_background_slider.on_changed(self._on_changed)

        self._background_real_slider = Slider(
            ax=axes[1],
            label='Background real ',
            valfmt=' %.2f',
            valmin=0.0,
            valmax=1.0,
            valstep=0.01,
            valinit=background_real,
        )
        self._background_real_slider.on_changed(self._on_changed)

        self._background_imag_slider = Slider(
            ax=axes[0],
            label='Background imag ',
            valfmt=' %.2f',
            valmin=0.0,
            valmax=0.6,
            valstep=0.01,
            valinit=background_imag,
        )
        self._background_imag_slider.on_changed(self._on_changed)

    def _on_semicircle_changed(self, value: Any) -> None:
        """Callback function to update semicircles."""
        self._frequency = frequency = self._frequency_slider.val
        acceptor_lifetime = self._acceptor_lifetime_slider.val
        if self._donor_semicircle_ticks is not None:
            lifetime, labels = _semicircle_ticks(frequency)
            self._donor_semicircle_ticks.labels = labels
            self._donor_semicircle_line.set_data(
                *phasor_transform(*phasor_from_lifetime(frequency, lifetime))
            )
        self._acceptor_semicircle_line.set_data(
            *phasor_transform(
                *phasor_semicircle(),
                *phasor_to_polar(
                    *phasor_from_lifetime(frequency, acceptor_lifetime)
                ),
            )
        )
        self._on_changed(value)

    def _on_changed(self, value: Any) -> None:
        """Callback function to update plot with current slider values."""
        frequency = self._frequency_slider.val
        donor_lifetime = self._donor_lifetime_slider.val
        acceptor_lifetime = self._acceptor_lifetime_slider.val
        fret_efficiency = self._fret_efficiency_slider.val
        donor_freting = self._donor_freting_slider.val
        donor_bleedthrough = self._donor_bleedthrough_slider.val
        acceptor_bleedthrough = self._acceptor_bleedthrough_slider.val
        acceptor_background = self._acceptor_background_slider.val
        donor_background = self._donor_background_slider.val
        background_real = self._background_real_slider.val
        background_imag = self._background_imag_slider.val
        e = int(self._fret_efficiency_slider.val * 100)

        donor_real, donor_imag = phasor_from_lifetime(
            frequency, donor_lifetime
        )
        donor_fret_real, donor_fret_imag = phasor_from_lifetime(
            frequency, donor_lifetime * (1.0 - fret_efficiency)
        )
        acceptor_real, acceptor_imag = phasor_from_lifetime(
            frequency, acceptor_lifetime
        )
        donor_trajectory_real, donor_trajectory_imag = phasor_from_fret_donor(
            frequency,
            donor_lifetime,
            fret_efficiency=self._fret_efficiencies,
            donor_freting=donor_freting,
            donor_background=donor_background,
            background_real=background_real,
            background_imag=background_imag,
        )
        (
            acceptor_trajectory_real,
            acceptor_trajectory_imag,
        ) = phasor_from_fret_acceptor(
            frequency,
            donor_lifetime,
            acceptor_lifetime,
            fret_efficiency=self._fret_efficiencies,
            donor_freting=donor_freting,
            donor_bleedthrough=donor_bleedthrough,
            acceptor_bleedthrough=acceptor_bleedthrough,
            acceptor_background=acceptor_background,
            background_real=background_real,
            background_imag=background_imag,
        )

        if donor_background > 0.0:
            self._donor_background_line.set_data(
                [float(donor_real), float(background_real)],
                [float(donor_imag), float(background_imag)],
            )
        else:
            self._donor_background_line.set_data([0.0, 0.0], [0.0, 0.0])

        if donor_freting < 1.0 and donor_background == 0.0:
            self._donor_donor_line.set_data(
                [donor_real, donor_fret_real],
                [donor_imag, donor_fret_imag],
            )
        else:
            self._donor_donor_line.set_data([0.0, 0.0], [0.0, 0.0])

        if acceptor_background > 0.0:
            self._acceptor_background_line.set_data(
                [float(acceptor_real), float(background_real)],
                [float(acceptor_imag), float(background_imag)],
            )
        else:
            self._acceptor_background_line.set_data([0.0, 0.0], [0.0, 0.0])

        self._background_line.set_data([background_real], [background_imag])

        self._donor_only_line.set_data([donor_real], [donor_imag])
        self._donor_fret_line.set_data([donor_fret_real], [donor_fret_imag])
        self._donor_trajectory_line.set_data(
            donor_trajectory_real, donor_trajectory_imag
        )
        self._donor_line.set_data(
            [donor_trajectory_real[e]], [donor_trajectory_imag[e]]
        )

        self._acceptor_only_line.set_data([acceptor_real], [acceptor_imag])
        self._acceptor_trajectory_line.set_data(
            acceptor_trajectory_real, acceptor_trajectory_imag
        )
        self._acceptor_line.set_data(
            [acceptor_trajectory_real[e]], [acceptor_trajectory_imag[e]]
        )


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


def plot_phasor(
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    style: Literal['plot', 'hist2d', 'contour'] | None = None,
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
    style : {'plot', 'hist2d', 'contour'}, optional
        Method used to plot phasor coordinates.
        By default, if the number of coordinates are less than 65536
        and the arrays are less than three-dimensional, `'plot'` style is used,
        else `'hist2d'`.
    allquadrants : bool, optional
        Show all quadrants of phasor space.
        By default, only the first quadrant is shown.
    frequency : float, optional
        Frequency of phasor plot.
        If provided, the universal semicircle is labeled with reference
        lifetimes.
    show : bool, optional, default: True
        Display figure.
    **kwargs
        Additional parguments passed to :py:class:`PhasorPlot`,
        :py:meth:`PhasorPlot.plot`, :py:meth:`PhasorPlot.hist2d`, or
        :py:meth:`PhasorPlot.contour` depending on `style`.

    See Also
    --------
    phasorpy.plot.PhasorPlot
    :ref:`sphx_glr_tutorials_api_phasorpy_phasorplot.py`

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
    elif style == 'contour':
        plot.contour(real, imag, **kwargs)
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
        mean = mean.reshape(-1, *mean.shape[-2:])
        if mean.shape[0] == 1:
            mean = mean[0]
        else:
            mean = numpy.nanmean(mean, axis=0)

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

    real = real.reshape(nh, -1, *real.shape[-2:])
    imag = imag.reshape(nh, -1, *imag.shape[-2:])
    if real.shape[1] == 1:
        real = real[:, 0]
        imag = imag[:, 0]
    else:
        real = numpy.nanmean(real, axis=1)
        imag = numpy.nanmean(imag, axis=1)

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
    ax.plot(numpy.nanmean(signal, axis=tuple(axes)))

    # image
    axes = list(sorted(axes[:-2] + [axis]))
    ax = fig.add_subplot(gs[0, 0])
    _imshow(
        ax,
        numpy.nanmean(signal, axis=tuple(axes)),
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
    **kwargs: Any,
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
        By default, a new subplot axes is created.
    title : str, optional
        Figure title. The default is "Multi-frequency plot".
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
    ax.set_xlabel('Frequency (MHz)')

    phase = numpy.asarray(phase)
    if phase.ndim < 2:
        phase = phase.reshape(-1, 1)
    modulation = numpy.asarray(modulation)
    if modulation.ndim < 2:
        modulation = modulation.reshape(-1, 1)

    ax.set_ylabel('Phase (°)', color='tab:blue')
    ax.set_yticks([0.0, 30.0, 60.0, 90.0])
    for phi in phase.T:
        ax.plot(frequency, numpy.rad2deg(phi), color='tab:blue', **kwargs)
    ax = ax.twinx()

    ax.set_ylabel('Modulation (%)', color='tab:red')
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
    **kwargs: Any,
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
