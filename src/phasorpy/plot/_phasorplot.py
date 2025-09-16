"""PhasorPlot class."""

from __future__ import annotations

__all__ = ['PhasorPlot']

import math
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._typing import Any, ArrayLike, Literal, NDArray, IO

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

import numpy
from matplotlib import pyplot
from matplotlib.font_manager import FontProperties
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Arc,
    Circle,
    Ellipse,
    FancyArrowPatch,
    Polygon,
    Rectangle,
)
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
from ..lifetime import (
    phasor_from_lifetime,
    phasor_semicircle,
    phasor_to_apparent_lifetime,
)
from ..phasor import phasor_to_polar, phasor_transform

GRID_COLOR = '0.5'
GRID_LINESTYLE = ':'
GRID_LINESTYLE_MAJOR = '-'
GRID_LINEWIDTH = 1.0
GRID_LINEWIDTH_MINOR = 0.6
GRID_FILL = False
GRID_ZORDER = 2


class PhasorPlot:
    """Phasor plot.

    Create publication quality visualizations of phasor coordinates.

    Parameters
    ----------
    allquadrants : bool, optional
        Show all quadrants of phasor space.
        By default, only the first quadrant with universal semicircle is shown.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes used for plotting.
        By default, a new subplot axes is created.
    frequency : float, optional
        Laser pulse or modulation frequency in MHz.
    pad : float, optional
        Padding around the plot. The default is 0.05.
    grid : dict or bool, optional
        Display universal semicircle (default) or polar grid (allquadrants).
        If False, no grid is displayed.
        If a dictionary, it is passed to :py:meth:`PhasorPlot.polar_grid`
        or :py:meth:`PhasorPlot.semicircle`.
    **kwargs
        Additional properties to set on `ax`.

    See Also
    --------
    phasorpy.plot.plot_phasor
    :ref:`sphx_glr_tutorials_api_phasorpy_phasorplot.py`

    """

    _ax: Axes
    """Matplotlib axes."""

    _full: bool
    """Show all quadrants of phasor space."""

    _labels: bool
    """Plot has labels attached."""

    _semicircle_ticks: CircleTicks | None
    """Last CircleTicks instance created for semicircle."""

    _unitcircle_ticks: CircleTicks | None
    """Last CircleTicks instance created for unit circle."""

    _frequency: float
    """Laser pulse or modulation frequency in MHz."""

    def __init__(
        self,
        /,
        allquadrants: bool | None = None,
        ax: Axes | None = None,
        *,
        frequency: float | None = None,
        grid: dict[str, Any] | bool | None = None,
        pad: float | None = None,
        **kwargs: Any,
    ) -> None:
        # initialize empty phasor plot
        self._ax = pyplot.subplots()[1] if ax is None else ax
        self._ax.format_coord = (  # type: ignore[method-assign]
            self._on_format_coord
        )
        self._labels = False

        if grid is None:
            grid_kwargs = {}
            grid = True
        if isinstance(grid, dict):
            grid_kwargs = grid
            grid = True
        else:
            grid_kwargs = {}
            grid = bool(grid)

        self._semicircle_ticks = None
        self._unitcircle_ticks = None

        self._full = bool(allquadrants)
        if self._full:
            pad = 0.1 if pad is None else float(abs(pad))
            xlim = (-1.0 - pad, 1.0 + pad)
            ylim = (-1.0 - pad, 1.0 + pad)
            xticks: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
            yticks: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
            if grid:
                self.polar_grid(**grid_kwargs)
        else:
            pad = 0.05 if pad is None else float(abs(pad))
            xlim = (-pad, 1.0 + pad)
            ylim = (-pad, 0.65 + pad)
            xticks = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
            yticks = (0.0, 0.2, 0.4, 0.6)
            if grid:
                self.semicircle(frequency=frequency, **grid_kwargs)

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
            xlim=xlim,
            ylim=ylim,
            xticks=xticks,
            yticks=yticks,
            aspect='equal',
        )
        for key in ('xlim', 'ylim', 'xticks', 'yticks', 'title'):
            if kwargs[key] is None:
                del kwargs[key]
        self._ax.set(**kwargs)
        # set axis limits after ticks
        if 'xlim' in kwargs:
            self._ax.set_xlim(kwargs['xlim'])
        if 'ylim' in kwargs:
            self._ax.set_ylim(kwargs['ylim'])

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
        if self._labels:
            self._ax.legend()
        # self.fig.show()
        pyplot.show()

    def legend(self, **kwargs: Any) -> Legend:
        """Add legend to plot.

        Parameters
        ----------
        **kwargs
            Optional arguments passed to
            :py:func:`matplotlib.axes.Axes.legend`.

        """
        return self._ax.legend(**kwargs)

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
            Optional arguments passed to :py:func:`matplotlib:pyplot.savefig`.

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
            Optional arguments passed to :py:meth:`matplotlib.axes.Axes.plot`.

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
                    if lbl is not None:
                        self._labels = True
                except IndexError:
                    pass
            lines = ax.plot(re, im, *args, label=lbl, **kwargs)
        return lines

    def _histogram2d(
        self,
        real: ArrayLike,
        imag: ArrayLike,
        /,
        **kwargs: Any,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Return two-dimensional histogram of imag versus real coordinates."""
        update_kwargs(kwargs, range=(self._ax.get_xlim(), self._ax.get_ylim()))
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
            Optional arguments passed to :py:meth:`numpy.histogram2d`
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

        # TODO: create custom labels for pcolormesh?
        # if 'label' in kwargs:
        #     self._labels = True

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
            Optional arguments passed to :py:func:`numpy.histogram2d`
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

        # TODO: create custom labels for contour?
        # if 'label' in kwargs:
        #     self._labels = True

    def imshow(
        self,
        image: ArrayLike,
        /,
        **kwargs: Any,
    ) -> None:
        """Plot an image, for example, a 2D histogram (not implemented).

        This method is not yet implemented and raises NotImplementedError.

        Parameters
        ----------
        image : array_like
            Image to display.
        **kwargs
            Optional arguments passed to
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
            Optional arguments passed to
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
            linestyle = kwargs.pop('ls', GRID_LINESTYLE)
            linewidth = kwargs.pop('lw', GRID_LINEWIDTH)
            update_kwargs(
                kwargs,
                edgecolor=GRID_COLOR if color is None else color,
                linestyle=linestyle,
                linewidth=linewidth,
                fill=GRID_FILL,
            )
            self._ax.add_patch(Polygon(numpy.vstack([real, imag]).T, **kwargs))
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
                    self._labels = True
            return

        fraction = numpy.asarray(fraction)[indices]
        linestyle = kwargs.pop('ls', GRID_LINESTYLE)
        linewidth = kwargs.pop('lw', GRID_LINEWIDTH)
        update_kwargs(
            kwargs,
            color=GRID_COLOR if color is None else color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
        center_re, center_im = numpy.average(
            numpy.vstack([real, imag]), axis=-1, weights=fraction
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
                self._labels = True

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
            Optional arguments passed to :py:class:`matplotlib.lines.Line2D`.

        Returns
        -------
        list[matplotlib.lines.Line2D]
            List containing plotted line.

        """
        linestyle = kwargs.pop('ls', GRID_LINESTYLE)
        linewidth = kwargs.pop('lw', GRID_LINEWIDTH)
        update_kwargs(
            kwargs, color=GRID_COLOR, linestyle=linestyle, linewidth=linewidth
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
            Optional arguments passed to :py:class:`matplotlib.patches.Circle`.

        """
        linestyle = kwargs.pop('ls', GRID_LINESTYLE)
        linewidth = kwargs.pop('lw', GRID_LINEWIDTH)
        update_kwargs(
            kwargs,
            color=GRID_COLOR,
            linestyle=linestyle,
            linewidth=linewidth,
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

        By default, draw a straight arrow with a ``'-|>'`` style, a mutation
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
            Optional arguments passed to
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
        real: ArrayLike,
        imag: ArrayLike,
        real_limit: ArrayLike | None = None,
        imag_limit: ArrayLike | None = None,
        /,
        *,
        radius: ArrayLike | None = None,
        radius_minor: ArrayLike | None = None,
        angle: ArrayLike | Literal['phase', 'semicircle'] | str | None = None,
        color: ArrayLike | None = None,
        label: ArrayLike | None = None,
        crosshair: bool = False,
        polar: bool = False,
        **kwargs: Any,
    ) -> None:
        """Draw cursor(s) at phasor coordinates.

        Parameters
        ----------
        real : array_like
            Real component of phasor coordinate.
        imag : array_like
            Imaginary component of phasor coordinate.
        real_limit : array_like, optional
            Real component of limiting phasor coordinate.
        imag_limit : array_like, optional
            Imaginary component of limiting phasor coordinate.
        radius : array_like, optional
            Radius of circular cursor.
        radius_minor : array_like, optional
            Radius of elliptic cursor along semi-minor axis.
            By default, `radius_minor` is equal to `radius`, that is,
            the ellipse is circular.
        angle : array_like or {'phase', 'semicircle'}, optional
            Rotation angle of semi-major axis of elliptic cursor in radians.
            If None or 'phase', align the minor axis of the ellipse with
            the closest tangent on the unit circle.
            If 'semicircle', align the ellipse with the universal semicircle.
        color : array_like, optional
            Color of cursor.
        label : array_like, optional
            String label for cursor.
        crosshair : bool, optional
            If true, draw polar or Cartesian lines or arcs limited by radius.
            Else, draw circle or ellipse (default).
            Only applies if `radius` is provided.
        polar : bool, optional
            If true, draw phase line and modulation arc.
            Else, draw Cartesian lines.
        **kwargs
            Optional arguments passed to
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
                *phasor_to_polar(real, imag),
                *phasor_to_polar(real_limit, imag_limit),
                radius=radius,
                radius_minor=radius_minor,
                angle=angle,
                color=color,
                label=label,
                crosshair=crosshair,
                polar=polar,
                **kwargs,
            )
        return self.polar_cursor(
            *phasor_to_polar(real, imag),
            radius=radius,
            radius_minor=radius_minor,
            angle=angle,
            color=color,
            label=label,
            crosshair=crosshair,
            polar=polar,
            **kwargs,
        )

    def polar_cursor(
        self,
        phase: ArrayLike | None = None,
        modulation: ArrayLike | None = None,
        phase_limit: ArrayLike | None = None,
        modulation_limit: ArrayLike | None = None,
        *,
        radius: ArrayLike | None = None,
        radius_minor: ArrayLike | None = None,
        angle: ArrayLike | Literal['phase', 'semicircle'] | str | None = None,
        color: ArrayLike | None = None,
        label: ArrayLike | None = None,
        crosshair: bool = False,
        polar: bool = True,
        **kwargs: Any,
    ) -> None:
        """Draw cursor(s) at polar coordinates.

        Parameters
        ----------
        phase : array_like, optional
            Angular component of polar coordinate in radians.
        modulation : array_like, optional
            Radial component of polar coordinate.
        phase_limit : array_like, optional
            Angular component of limiting polar coordinate (in radians).
            Modulation arcs are drawn between `phase` and `phase_limit`
            if `polar` is true.
        modulation_limit : array_like, optional
            Radial component of limiting polar coordinate.
            Phase lines are drawn from `modulation` to `modulation_limit`
            if `polar` is true.
        radius : array_like, optional
            Radius of circular cursor.
        radius_minor : array_like, optional
            Radius of elliptic cursor along semi-minor axis.
            By default, `radius_minor` is equal to `radius`, that is,
            the ellipse is circular.
        angle : array_like or {'phase', 'semicircle'}, optional
            Rotation angle of semi-major axis of elliptic cursor in radians.
            If None or 'phase', align the minor axis of the ellipse with
            the closest tangent on the unit circle.
            If 'semicircle', align the ellipse with the universal semicircle.
        color : array_like, optional
            Color of cursor.
        label : array_like, optional
            String label for cursor.
        crosshair : bool, optional
            If true, draw polar or Cartesian lines or arcs limited by radius.
            Else, draw circle or ellipse (default).
            Only applies if `radius` is provided.
        polar : bool, optional
            If true, draw phase line and modulation arc.
            Else, draw Cartesian lines.
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
        shape = None
        if phase is not None:
            phase = numpy.atleast_1d(phase)
            if phase.ndim != 1:
                raise ValueError(f'invalid {phase.ndim=} != 1')
            shape = phase.shape
        if modulation is not None:
            if shape is not None:
                modulation = numpy.broadcast_to(modulation, shape)
            else:
                modulation = numpy.atleast_1d(modulation)
                if modulation.ndim != 1:
                    raise ValueError(f'invalid {modulation.ndim=} != 1')
                shape = modulation.shape
        if shape is None:
            return

        if phase_limit is not None:
            phase_limit = numpy.broadcast_to(phase_limit, shape)
        if modulation_limit is not None:
            modulation_limit = numpy.broadcast_to(modulation_limit, shape)
        if radius is not None:
            radius = numpy.broadcast_to(radius, shape)
        if radius_minor is not None:
            radius_minor = numpy.broadcast_to(radius_minor, shape)
        if angle is not None and not isinstance(angle, str):
            angle = numpy.broadcast_to(angle, shape)
        if label is not None:
            label = numpy.broadcast_to(label, shape)
            label = [str(c) for c in label]
        if color is not None:
            color = numpy.atleast_1d(color)
            if color.dtype.kind == 'U':
                color = numpy.broadcast_to(color, shape)
                color = [str(c) for c in color]
            else:
                color = numpy.broadcast_to(color, (shape[0], color.shape[-1]))

        for i in range(shape[0]):

            if color is not None:
                kwargs['color'] = color[i]
            if label is not None:
                kwargs['label'] = label[i]

            self._cursor(
                phase if phase is None else float(phase[i]),
                modulation if modulation is None else float(modulation[i]),
                phase_limit if phase_limit is None else float(phase_limit[i]),
                (
                    modulation_limit
                    if modulation_limit is None
                    else float(modulation_limit[i])
                ),
                radius=radius if radius is None else float(radius[i]),
                radius_minor=(
                    radius_minor
                    if radius_minor is None
                    else float(radius_minor[i])
                ),
                angle=(
                    angle
                    if (angle is None or isinstance(angle, str))
                    else float(angle[i])
                ),
                crosshair=crosshair,
                polar=polar,
                **kwargs,
            )

    def _cursor(
        self,
        phase: float | None = None,
        modulation: float | None = None,
        phase_limit: float | None = None,
        modulation_limit: float | None = None,
        *,
        radius: float | None = None,
        radius_minor: float | None = None,
        angle: float | Literal['phase', 'semicircle'] | str | None = None,
        crosshair: bool = False,
        polar: bool = True,
        **kwargs: Any,
    ) -> None:
        """Draw single cursor at polar coordinate."""
        linestyle = kwargs.pop('ls', GRID_LINESTYLE_MAJOR)
        linewidth = kwargs.pop('lw', GRID_LINEWIDTH)
        update_kwargs(
            kwargs,
            color=GRID_COLOR,
            linestyle=linestyle,
            linewidth=linewidth,
            fill=GRID_FILL,
            zorder=GRID_ZORDER,
        )

        ax = self._ax
        if radius is not None and phase is not None and modulation is not None:
            x = modulation * math.cos(phase)
            y = modulation * math.sin(phase)
            if radius_minor is not None and radius_minor != radius:
                if angle is None:
                    angle = phase
                elif isinstance(angle, str):
                    if angle == 'phase':
                        angle = phase
                    elif angle == 'semicircle':
                        angle = math.atan2(y, x - 0.5)
                    else:
                        raise ValueError(f'invalid {angle=}')
                angle = math.degrees(angle)

                if not crosshair:
                    # draw elliptical cursor
                    ax.add_patch(
                        Ellipse(
                            (x, y),
                            radius * 2,
                            radius_minor * 2,
                            angle=angle,
                            **kwargs,
                        )
                    )
                    if 'label' in kwargs:
                        self._labels = True
                    return None

                # TODO: implement crosshair intersecting with ellipse?
                raise ValueError('crosshair not implemented with ellipse')

            if not crosshair:
                # draw circlar cursor
                ax.add_patch(Circle((x, y), radius, **kwargs))
                if 'label' in kwargs:
                    self._labels = True
                return None

            del kwargs['fill']
            if not polar:
                # draw Cartesian crosshair lines limited by radius
                x0, y0, x1, y1 = _intersect_circle_line(
                    x, y, radius, x, y, x + 1, y
                )
                ax.add_line(Line2D([x0, x1], [y0, y1], **kwargs))
                if 'label' in kwargs:
                    self._labels = True
                    del kwargs['label']
                x0, y0, x1, y1 = _intersect_circle_line(
                    x, y, radius, x, y, x, y + 1
                )
                ax.add_line(Line2D([x0, x1], [y0, y1], **kwargs))
                return None

            if abs(x) < 1e-6 and abs(y) < 1e-6:
                # phase and modulation not defined at origin
                return None

            # draw crosshair phase line and modulation arc limited by circle
            x0, y0, x1, y1 = _intersect_circle_line(x, y, radius, 0, 0, x, y)
            ax.add_line(Line2D([x0, x1], [y0, y1], **kwargs))
            if 'label' in kwargs:
                self._labels = True
                del kwargs['label']
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

        if not polar:
            if phase is None or modulation is None:
                return None

            x0 = modulation * math.cos(phase)
            y0 = modulation * math.sin(phase)
            if phase_limit is None or modulation_limit is None:
                # draw Cartesian crosshair lines
                del kwargs['fill']
                ax.add_line(Line2D([x0, x0], [-2, 2], **kwargs))
                if 'label' in kwargs:
                    self._labels = True
                    del kwargs['label']
                ax.add_line(Line2D([-2, 2], [y0, y0], **kwargs))
            else:
                # draw rectangle
                x1 = modulation_limit * math.cos(phase_limit)
                y1 = modulation_limit * math.sin(phase_limit)
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, **kwargs))
                if 'label' in kwargs:
                    self._labels = True
            return None

        # TODO: implement filled polar region/rectangle
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
                    x1 = math.cos(phi) * 2
                    y1 = math.sin(phi) * 2
                ax.add_line(Line2D([x0, x1], [y0, y1], **kwargs))
                if 'label' in kwargs:
                    self._labels = True
                    del kwargs['label']
        for mod in (modulation, modulation_limit):
            if mod is not None:
                if phase is not None and phase_limit is not None:
                    theta1 = math.degrees(min(phase, phase_limit))
                    theta2 = math.degrees(max(phase, phase_limit))
                else:
                    theta1 = 0.0
                    theta2 = 360.0  # if self._full else 90.0
                # TODO: filling arc objects is not supported
                ax.add_patch(
                    Arc(
                        (0, 0),
                        mod * 2,
                        mod * 2,
                        theta1=theta1,
                        theta2=theta2,
                        fill=False,
                        **kwargs,
                    )
                )
                if 'label' in kwargs:
                    self._labels = True
                    del kwargs['label']
        return None

    def polar_grid(
        self,
        radii: int | Sequence[float] | None = None,
        angles: int | Sequence[float] | None = None,
        samples: int | None = None,
        labels: Sequence[str] | None = None,
        ticks: ArrayLike | None = None,
        tick_space: ArrayLike | None = None,
        tick_format: str | None = None,
        **kwargs: Any,
    ) -> None:
        r"""Draw polar coordinate system.

        Parameters
        ----------
        radii : int or sequence of float, optional
            Position of radial gridlines in range (0, 1].
            If an integer, the number of equidistant radial gridlines.
            By default, three equidistant radial gridlines are drawn.
            The unit circle (radius 1), if included, is drawn in major style.
        angles : int or sequence of float, optional
            Position of angular gridlines in range [0, 2 pi].
            If an integer, the number of equidistant angular gridlines.
            By default, 12 equidistant angular gridlines are drawn.
        samples : int, optional
            Number of vertices of polygon inscribed in unit circle.
            By default, no inscribed polygon is drawn.
        labels : sequence of str, optional
            Tick labels on unit circle.
            Labels are placed at equidistant angles if `ticks` are not
            provided.
        ticks : array_like, optional
            Values at which to place tick labels on unit circle.
            If `labels` are not provided, `ticks` values formatted with
            `tick_format` are used as labels.
            If `tick_space` is not provided, tick values are angles in radians.
        tick_space : array_like, optional
            Values used to convert `ticks` to angles.
            For example, the wavelengths used to calculate spectral phasors
            or the minimum and maximum wavelengths of a sine-cosine filter.
        tick_format : str, optional
            Format string for tick values if `labels` is None.
            By default, the tick format is "{}".
        **kwargs
            Optional arguments passed to
            :py:class:`matplotlib.patches.Circle` and
            :py:class:`matplotlib.lines.Line2D`.

        Raises
        ------
        ValueError
            If number of ticks doesn't match number of labels.
            If `tick_space` has less than two values.

        Notes
        -----
        Use ``radii=1, angles=4`` to draw major gridlines only.

        The values of ticks (:math:`v`) are converted to angles
        (:math:`\theta`) using `tick_space` (:math:`s`) according to:

        .. math::
            \theta = \frac{v - s_0}{s_{-1} + s_1 - 2 s_0} \cdot 2 \pi

        """
        ax = self._ax
        minor_kwargs = kwargs.copy()
        linestyle = minor_kwargs.pop('ls', GRID_LINESTYLE)
        linewidth = minor_kwargs.pop('lw', GRID_LINEWIDTH_MINOR)
        update_kwargs(
            minor_kwargs,
            color=GRID_COLOR,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=GRID_ZORDER,
        )
        linestyle = kwargs.pop('ls', GRID_LINESTYLE_MAJOR)
        linewidth = kwargs.pop('lw', GRID_LINEWIDTH)
        update_kwargs(
            kwargs,
            color=GRID_COLOR,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=GRID_ZORDER,
            # fill=GRID_FILL,
        )

        if samples is not None and samples > 1:
            angle = numpy.linspace(0, 2 * math.pi, samples, endpoint=False)
            xy = numpy.vstack([numpy.cos(angle), numpy.sin(angle)]).T
            ax.add_patch(Polygon(xy, fill=False, **kwargs))

        if radii is None:
            radii = [1 / 3, 2 / 3, 1.0]
        elif isinstance(radii, int):
            radii = numpy.linspace(0, 1, radii + 1, endpoint=True)[1:].tolist()
        for r in radii:  # type: ignore[union-attr]
            if r < 1e-3:
                # skip zero radius
                continue
            if abs(r - 1.0) < 1e-3:
                # unit circle
                circle = Circle((0, 0), 1, fill=False, **kwargs)
            elif r > 1.0:
                continue
            else:
                # minor circle
                circle = Circle((0, 0), r, fill=False, **minor_kwargs)
            ax.add_patch(circle)

        if angles is None:
            angles = 12
        if isinstance(angles, int):
            angles = numpy.linspace(
                0, 2 * math.pi, angles, endpoint=False
            ).tolist()
        for a in angles:  # type: ignore[union-attr]
            if a < 0 or a > 2 * math.pi:
                # skip angles out of range
                continue
            x = math.cos(a)
            y = math.sin(a)
            ax.add_line(Line2D([0.0, x], [0.0, y], **minor_kwargs))

        if labels is None and ticks is None:
            # no labels
            return
        if ticks is None:
            # equidistant labels
            assert labels is not None
            ticks = numpy.linspace(0, 2 * math.pi, len(labels), endpoint=False)
            tick_space = None
        elif labels is None:
            # use tick values as labels
            assert ticks is not None
            ticks = numpy.array(ticks, ndmin=1, copy=True)
            if tick_format is None:
                tick_format = '{}'
            labels = [tick_format.format(t) for t in ticks]
            ticks = ticks.astype(numpy.float64)
        else:
            # ticks and labels
            ticks = numpy.array(ticks, dtype=numpy.float64, ndmin=1, copy=True)
            if ticks.size != len(labels):
                raise ValueError(f'{ticks.size=} != {len(labels)=}')

        if tick_space is not None:
            tick_space = numpy.asarray(tick_space, dtype=numpy.float64)
            if tick_space.ndim != 1 or tick_space.size < 2:
                raise ValueError(
                    f'invalid {tick_space.ndim=} or {tick_space.size=} < 2'
                )
            assert isinstance(ticks, numpy.ndarray)  # for mypy
            ticks -= tick_space[0]
            ticks /= tick_space[-1] + tick_space[1] - 2 * tick_space[0]
            ticks *= 2 * math.pi

        real = numpy.cos(ticks)
        imag = numpy.sin(ticks)
        self._unitcircle_ticks = CircleTicks(labels=labels)
        ax.plot(real, imag, path_effects=[self._unitcircle_ticks], **kwargs)

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
            Optional arguments passed to
            :py:class:`matplotlib.lines.Line2D` or
            :py:class:`matplotlib.patches.Arc` and
            :py:meth:`matplotlib.axes.Axes.plot`.

        Returns
        -------
        list of matplotlib.lines.Line2D
            Lines representing plotted semicircle and ticks.

        """
        if frequency is not None:
            self._frequency = float(frequency)

        linestyle = kwargs.pop('ls', GRID_LINESTYLE_MAJOR)
        linewidth = kwargs.pop('lw', GRID_LINEWIDTH)
        update_kwargs(
            kwargs,
            linestyle=linestyle,
            linewidth=linewidth,
            color=GRID_COLOR,
            zorder=GRID_ZORDER,
        )
        if 'label' in kwargs:
            self._labels = True

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

        kwargs.pop('label', None)  # don't pass label to ticks
        kwargs.pop('capstyle', None)

        if frequency is not None and polar_reference == (0.0, 1.0):
            # draw ticks and labels
            lifetime, labels = _semicircle_ticks(frequency, lifetime, labels)
            self._semicircle_ticks = CircleTicks((0.5, 0.0), labels=labels)
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


class CircleTicks(AbstractPathEffect):
    """Draw ticks on unit circle or universal semicircle.

    Parameters
    ----------
    origin : (float, float), optional
        Origin of circle.
    size : float, optional
        Length of tick in dots.
        The default is ``rcParams['xtick.major.size']``.
    labels : sequence of str, optional
        Tick labels for each vertex in path.
    **kwargs
        Optional arguments passed to
        :py:meth:`matplotlib.patheffects.AbstractPathEffect._update_gc`.

    """

    _origin: tuple[float, float]  # origin of circle
    _size: float  # tick length
    _labels: tuple[str, ...]  # tick labels
    _gc: dict[str, Any]  # keywords passed to _update_gc

    def __init__(
        self,
        origin: tuple[float, float] | None = None,
        /,
        size: float | None = None,
        labels: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__((0.0, 0.0))

        if origin is None:
            self._origin = 0.0, 0.0
        else:
            self._origin = float(origin[0]), float(origin[1])

        if size is None:
            self._size = pyplot.rcParams['xtick.major.size']
        else:
            self._size = size
        if labels is None or len(labels) == 0:
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
        if value is None:
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
        origin = affine.transform((self._origin,))

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
                if not s:
                    continue
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
