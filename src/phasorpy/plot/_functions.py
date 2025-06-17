"""Higher level plot functions."""

from __future__ import annotations

__all__ = [
    'plot_histograms',
    'plot_image',
    'plot_phasor',
    'plot_phasor_image',
    'plot_polar_frequency',
    'plot_signal_image',
]

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._typing import Any, ArrayLike, NDArray, Literal

    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage

import numpy
from matplotlib import pyplot
from matplotlib.gridspec import GridSpec

from .._utils import parse_kwargs, parse_signal_axis, update_kwargs
from ._phasorplot import PhasorPlot


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
        for `real` and `imag` the range [-1, 1].
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
    axis: int | str | None = None,
    percentile: float | Sequence[float] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
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
    axis : int or str, optional
        Axis over which phasor coordinates would be computed.
        By default, the 'H' or 'C' axes if signal contains such dimension
        names, else the last axis (-1).
    percentile : float or [float, float], optional
        The [q, 100-q] percentiles of image data are covered by colormaps.
        By default, the complete value range of `mean` is covered,
        for `real` and `imag` the range [-1, 1].
    title : str, optional
        Figure title.
    xlabel : str, optional
        Label of axis over which phasor coordinates would be computed.
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

    axis, axis_label = parse_signal_axis(signal, axis)
    if (
        axis_label
        and hasattr(signal, 'coords')
        and axis_label in signal.coords
    ):
        axis_coords = signal.coords[axis_label]
    else:
        axis_coords = None

    update_kwargs(kwargs, interpolation='nearest')
    signal = numpy.asarray(signal)
    if signal.ndim < 3:
        raise ValueError(f'not an image stack {signal.ndim=} < 3')

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

    if axis_coords is not None:
        ax.set_title(f'{axis=} {axis_label!r}')
        ax.plot(axis_coords, numpy.nanmean(signal, axis=tuple(axes)))
    else:
        ax.set_title(f'{axis=}')
        ax.plot(numpy.nanmean(signal, axis=tuple(axes)))

    ax.set_ylim(kwargs.get('vmin', None), kwargs.get('vmax', None))

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    # image
    axes = list(sorted(axes[:-2] + [axis]))
    ax = fig.add_subplot(gs[0, 0])
    _imshow(
        ax,
        numpy.nanmean(signal, axis=tuple(axes)),
        percentile=percentile,
        shrink=0.5,
        title='mean',
        **kwargs,
    )

    if show:
        pyplot.show()


def plot_image(
    *images: ArrayLike,
    percentile: float | None = None,
    columns: int | None = None,
    title: str | None = None,
    labels: Sequence[str | None] | None = None,
    show: bool = True,
    **kwargs: Any,
) -> None:
    """Plot images.

    Parameters
    ----------
    *images : array_like
        Images to be plotted. Must be two or more dimensional.
        The last two axes are assumed to be the image axes.
        Other axes are averaged for display.
        Three-dimensional images with last axis size of three or four
        are plotted as RGB(A) images.
    percentile : float, optional
        The (q, 100-q) percentiles of image data are covered by colormaps.
        By default, the complete value range is covered.
        Does not apply to RGB images.
    columns : int, optional
        Number of columns in figure.
        By default, up to four columns are used.
    title : str, optional
        Figure title.
    labels : sequence of str, optional
        Labels for each image.
    show : bool, optional, default: True
        Display figure.
    **kwargs
        Additional arguments passed to :func:`matplotlib.pyplot.imshow`.

    Raises
    ------
    ValueError
        Percentile is out of range.

    """
    update_kwargs(
        kwargs, interpolation='nearest', location='right', shrink=0.5
    )
    cmap = kwargs.pop('cmap', None)
    figsize = kwargs.pop('figsize', None)
    subplot_kw = kwargs.pop('subplot_kw', {})
    location = kwargs['location']
    allrgb = True

    arrays = []
    shape = [1, 1]
    for image in images:
        image = numpy.asarray(image)
        if image.ndim < 2:
            raise ValueError(f'not an image {image.ndim=} < 2')
        if image.ndim == 3 and image.shape[2] in {3, 4}:
            # RGB(A)
            pass
        else:
            allrgb = False
            image = image.reshape(-1, *image.shape[-2:])
            if image.shape[0] == 1:
                image = image[0]
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    image = numpy.nanmean(image, axis=0)
            assert isinstance(image, numpy.ndarray)
            for i in (-1, -2):
                if image.shape[i] > shape[i]:
                    shape[i] = image.shape[i]
        arrays.append(image)

    if columns is None:
        n = len(arrays)
        if n < 3:
            columns = n
        elif n < 5:
            columns = 2
        elif n < 7:
            columns = 3
        else:
            columns = 4
    rows = int(numpy.ceil(len(arrays) / columns))

    vmin = None
    vmax = None
    if percentile is None:
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        if vmin is None:
            vmin = numpy.inf
            for image in images:
                vmin = min(vmin, numpy.nanmin(image))
            if vmin == numpy.inf:
                vmin = None
        if vmax is None:
            vmax = -numpy.inf
            for image in images:
                vmax = max(vmax, numpy.nanmax(image))
            if vmax == -numpy.inf:
                vmax = None

    # create figure with size depending on image aspect
    fig = pyplot.figure(layout='constrained', figsize=figsize)
    if figsize is None:
        # TODO: find optimal figure height as a function of
        # number of rows and columns, image shapes, labels, and colorbar
        # presence and placements.
        if allrgb:
            hadd = 0.0
        elif location == 'right':
            hadd = 0.5
        else:
            hadd = 1.2
        if labels is not None:
            hadd += 0.3 * rows
        w, h = fig.get_size_inches()
        aspect = min(1.0, max(0.5, shape[0] / shape[1]))
        fig.set_size_inches(
            w, h * 0.9 / columns * aspect * rows + h * 0.1 * aspect + hadd
        )
    gs = GridSpec(rows, columns, figure=fig)
    if title:
        fig.suptitle(title)

    axs = []
    for i, image in enumerate(arrays):
        ax = fig.add_subplot(gs[i // columns, i % columns], **subplot_kw)
        ax.set_anchor('C')
        axs.append(ax)
        pos = _imshow(
            ax,
            image,
            percentile=percentile,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            colorbar=percentile is not None,
            axis=i == 0 and not subplot_kw,
            title=None if labels is None else labels[i],
            **kwargs,
        )
    if not allrgb and percentile is None:
        fig.colorbar(pos, ax=axs, shrink=kwargs['shrink'], location=location)

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


def plot_histograms(
    *data: ArrayLike,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    labels: Sequence[str] | None = None,
    show: bool = True,
    **kwargs: Any,
) -> None:
    """Plot histograms of flattened data arrays.

    Parameters
    ----------
    data: array_like
        Data arrays to be plotted as histograms.
    title : str, optional
        Figure title.
    xlabel : str, optional
        Label for x-axis.
    ylabel : str, optional
        Label for y-axis.
    labels: sequence of str, optional
        Labels for each data array.
    show : bool, optional, default: True
        Display figure.
    **kwargs
        Additional arguments passed to :func:`matplotlib.pyplot.hist`.

    """
    ax = pyplot.subplots()[1]
    if kwargs.get('alpha') is None:
        ax.hist(
            [numpy.asarray(d).flatten() for d in data], label=labels, **kwargs
        )
    else:
        for d, label in zip(
            data, [None] * len(data) if labels is None else labels
        ):
            ax.hist(numpy.asarray(d).flatten(), label=label, **kwargs)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if labels is not None:
        ax.legend()
    pyplot.tight_layout()
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
    location = kwargs.pop('location', 'bottom')
    if image.ndim == 3 and image.shape[2] in {3, 4}:
        # RGB(A)
        vmin = None
        vmax = None
        percentile = None
        colorbar = False
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
        vmin, vmax = numpy.nanpercentile(image, percentile)
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
            fig.colorbar(pos, shrink=shrink, location=location, ticks=ticks)
    if title:
        ax.set_title(title)
    if not axis:
        ax.set_axis_off()
    # ax.set_anchor('C')
    return pos
