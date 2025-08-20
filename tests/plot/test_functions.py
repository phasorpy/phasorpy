"""Test higher level plot functions."""

import math

import numpy
import pytest
from matplotlib import pyplot
from xarray import DataArray

from phasorpy.plot import (
    plot_histograms,
    plot_image,
    plot_phasor,
    plot_phasor_image,
    plot_polar_frequency,
    plot_signal_image,
)

INTERACTIVE = False  # enable for interactive plotting


def test_plot_phasor():
    """Test plot_phasor function."""
    real, imag = numpy.random.multivariate_normal(
        [0.6, 0.4], [[3e-3, -1e-3], [-1e-3, 1e-3]], 32
    ).T
    plot_phasor(
        real,
        imag,
        frequency=80.0,
        color='tab:red',
        style='plot',
        title='plot',
        show=INTERACTIVE,
    )
    pyplot.close()

    _, ax = pyplot.subplots()
    real, imag = numpy.random.multivariate_normal(
        [0.6, 0.4], [[3e-3, -1e-3], [-1e-3, 1e-3]], (256, 256)
    ).T
    plot_phasor(
        real,
        imag,
        ax=ax,
        allquadrants=True,
        grid=False,
        cmap='Blues',
        title='hist2d',
        show=INTERACTIVE,
    )
    pyplot.close()

    plot_phasor(
        real,
        imag,
        levels=4,
        cmap='viridis',
        style='contour',
        title='contour',
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

    plot_signal_image(
        data,
        vmin=0,
        vmax=1,
        xlabel='xlabel',
        title='default',
        show=INTERACTIVE,
    )
    pyplot.close()
    plot_signal_image(data, axis=0, title='axis 0', show=INTERACTIVE)
    pyplot.close()
    plot_signal_image(data, axis=2, title='axis 2', show=INTERACTIVE)
    pyplot.close()
    plot_signal_image(
        data,
        percentile=(5, 95),
        cmap='hot',
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


def test_plot_plot_histograms():
    """Test plot_histograms function."""
    data = (numpy.random.normal(0, 1, 1000), numpy.random.normal(4, 2, 1000))
    plot_histograms(data[0], show=INTERACTIVE)
    pyplot.close()
    plot_histograms(*data, show=INTERACTIVE)
    pyplot.close()
    plot_histograms(*data, alpha=0.66, bins=50, show=INTERACTIVE)
    pyplot.close()
    plot_histograms(*data, alpha=0.66, title='Histograms', show=INTERACTIVE)
    pyplot.close()
    plot_histograms(*data, alpha=0.66, xlabel='X axis', show=INTERACTIVE)
    pyplot.close()
    plot_histograms(*data, alpha=0.66, ylabel='Y axis', show=INTERACTIVE)
    pyplot.close()
    plot_histograms(*data, alpha=0.66, labels=['A', 'B'], show=INTERACTIVE)
    pyplot.close()


@pytest.mark.parametrize('percentile', (None, 0.9))
@pytest.mark.parametrize('labels', (None, 'Label'))
@pytest.mark.parametrize('location', ('right', 'bottom'))
@pytest.mark.parametrize('aspect', (1.0, 0.75))
@pytest.mark.parametrize('nimages', (1, 2, 4, 5))
def test_plot_image(percentile, labels, location, aspect, nimages):
    """Test plot_image function."""
    images = numpy.random.normal(1.0, 0.2, (nimages, int(100 * aspect), 100))
    images[0] *= 2
    title = f'{nimages=}, {aspect=}, {percentile=}, {labels=}, {location=}'
    if labels is not None:
        labels = [labels] * nimages
    plot_image(
        *images,
        percentile=percentile,
        location=location,
        labels=labels,
        title=title,
        show=INTERACTIVE,
    )
    pyplot.close()


@pytest.mark.parametrize('columns', (None, 4))
@pytest.mark.parametrize('percentile', (None, 0.9))
def test_plot_image_shapes(columns, percentile):
    """Test plot_image function with images of different shapes."""
    images = [
        numpy.random.normal(0.5, 0.1, shape)
        for shape in (
            (100, 100, 3),
            (100, 100),
            (50, 100),
            (3, 100, 100),
            (100, 50, 3),
        )
    ]
    images[1][images[1] < 0.5] = numpy.nan
    plot_image(
        *images,
        columns=columns,
        percentile=percentile,
        labels=[f'{im.shape!r}' for im in images],
        title=f'{columns=} {percentile=}',
        show=INTERACTIVE,
    )
    pyplot.close()


def test_plot_image_other():
    """Test plot_image function with special cases."""
    images = [numpy.random.normal(0.5, 0.1, (100, 100, 3))] * 7
    plot_image(*images, title='RGB only', show=INTERACTIVE)
    pyplot.close()

    with pytest.warns(RuntimeWarning):
        plot_image(
            numpy.full((100, 100), numpy.nan),
            title='NaN only',
            show=INTERACTIVE,
        )
    pyplot.close()

    with pytest.raises(ValueError):
        plot_image(numpy.zeros(100), show=INTERACTIVE)


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
