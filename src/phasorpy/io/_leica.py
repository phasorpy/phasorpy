"""Read Leica image file formats."""

from __future__ import annotations

__all__ = ['lifetime_from_lif', 'phasor_from_lif', 'signal_from_lif']

from typing import TYPE_CHECKING

from .._utils import xarray_metadata

if TYPE_CHECKING:
    from .._typing import Any, DataArray, Literal, NDArray, PathLike

import numpy


def phasor_from_lif(
    filename: str | PathLike[Any],
    /,
    image: str | None = None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], dict[str, Any]]:
    """Return phasor coordinates and metadata from Leica image file.

    Leica image files may contain uncalibrated phasor coordinate images and
    metadata from the analysis of FLIM measurements.

    Parameters
    ----------
    filename : str or Path
        Name of Leica image file to read.
    image : str, optional
        Name of parent image containing phasor coordinates.

    Returns
    -------
    mean : ndarray
        Average intensity image.
    real : ndarray
        Image of real component of phasor coordinates.
    imag : ndarray
        Image of imaginary component of phasor coordinates.
    attrs : dict
        Select metadata:

        - ``'dims'`` (tuple of str):
          :ref:`Axes codes <axes>` for `mean` image dimensions.
        - ``'frequency'`` (float):
          Fundamental frequency of time-resolved phasor coordinates in MHz.
          May not be present in all files.
        - ``'flim_rawdata'`` (dict):
          Settings from SingleMoleculeDetection/RawData XML element.
        - ``'flim_phasor_channels'`` (list of dict):
          Settings from SingleMoleculeDetection/.../PhasorData/Channels XML
          elements.

    Raises
    ------
    liffile.LifFileError
        File is not a Leica image file.
    ValueError
        File or `image` do not contain phasor coordinates and metadata.

    Notes
    -----
    The implementation is based on the
    `liffile <https://github.com/cgohlke/liffile/>`__ library.

    Examples
    --------
    >>> mean, real, imag, attrs = phasor_from_lif(fetch('FLIM_testdata.lif'))
    >>> real.shape
    (1024, 1024)
    >>> attrs['dims']
    ('Y', 'X')
    >>> attrs['frequency']
    19.505

    """
    # TODO: read harmonic from XML if possible
    # TODO: get calibration settings from XML metadata, lifetime, or
    #   phasor plot images
    import liffile

    image = '' if image is None else f'.*{image}.*/'
    samples = 1

    with liffile.LifFile(filename) as lif:
        try:
            im = lif.images[image + 'Phasor Intensity$']
            dims = im.dims
            coords = im.coords
            # meta = image.attrs
            mean = im.asarray().astype(numpy.float32)
            real = lif.images[image + 'Phasor Real$'].asarray()
            imag = lif.images[image + 'Phasor Imaginary$'].asarray()
            # mask = lif.images[image + 'Phasor Mask$'].asarray()
        except Exception as exc:
            raise ValueError(
                f'{lif.filename!r} does not contain Phasor images'
            ) from exc

        attrs: dict[str, Any] = {'dims': dims, 'coords': coords}
        flim = im.parent_image
        if flim is not None and isinstance(flim, liffile.LifFlimImage):
            xml = flim.parent.xml_element
            frequency = xml.find('.//Dataset/RawData/LaserPulseFrequency')
            if frequency is not None and frequency.text is not None:
                attrs['frequency'] = float(frequency.text) * 1e-6
                clock_period = xml.find('.//Dataset/RawData/ClockPeriod')
                if clock_period is not None and clock_period.text is not None:
                    tmp = float(clock_period.text) * float(frequency.text)
                    samples = int(round(1.0 / tmp))
                    attrs['samples'] = samples
            channels = []
            for channel in xml.findall(
                './/Dataset/FlimData/PhasorData/Channels'
            ):
                ch = liffile.xml2dict(channel)['Channels']
                ch.pop('PhasorPlotShapes', None)
                channels.append(ch)
            attrs['flim_phasor_channels'] = channels
            attrs['flim_rawdata'] = flim.attrs.get('RawData', {})

    if samples > 1:
        mean /= samples
    return (
        mean,
        real.astype(numpy.float32),
        imag.astype(numpy.float32),
        attrs,
    )


def lifetime_from_lif(
    filename: str | PathLike[Any],
    /,
    image: str | None = None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], dict[str, Any]]:
    """Return lifetime image and metadata from Leica image file.

    Leica image files may contain fluorescence lifetime images and metadata
    from the analysis of FLIM measurements.

    Parameters
    ----------
    filename : str or Path
        Name of Leica image file to read.
    image : str, optional
        Name of parent image containing lifetime image.

    Returns
    -------
    lifetime : ndarray
        Fluorescence lifetime image in ns.
    intensity : ndarray
        Fluorescence intensity image.
    stddev : ndarray
        Standard deviation of fluorescence lifetimes in ns.
    attrs : dict
        Select metadata:

        - ``'dims'`` (tuple of str):
          :ref:`Axes codes <axes>` for `intensity` image dimensions.
        - ``'frequency'`` (float):
          Fundamental frequency of lifetimes in MHz.
          May not be present in all files.
        - ``'samples'`` (int):
          Number of bins in TCSPC histogram. May not be present in all files.
        - ``'flim_rawdata'`` (dict):
          Settings from SingleMoleculeDetection/RawData XML element.

    Raises
    ------
    liffile.LifFileError
        File is not a Leica image file.
    ValueError
        File or `image` does not contain lifetime coordinates and metadata.

    Notes
    -----
    The implementation is based on the
    `liffile <https://github.com/cgohlke/liffile/>`__ library.

    Examples
    --------
    >>> lifetime, intensity, stddev, attrs = lifetime_from_lif(
    ...     fetch('FLIM_testdata.lif')
    ... )
    >>> lifetime.shape
    (1024, 1024)
    >>> attrs['dims']
    ('Y', 'X')
    >>> attrs['frequency']
    19.505

    """
    import liffile

    image = '' if image is None else f'.*{image}.*/'

    with liffile.LifFile(filename) as lif:
        try:
            im = lif.images[image + 'Intensity$']
            dims = im.dims
            coords = im.coords
            # meta = im.attrs
            intensity = im.asarray()
            lifetime = lif.images[image + 'Fast Flim$'].asarray()
            stddev = lif.images[image + 'Standard Deviation$'].asarray()
        except Exception as exc:
            raise ValueError(
                f'{lif.filename!r} does not contain lifetime images'
            ) from exc

        attrs: dict[str, Any] = {'dims': dims, 'coords': coords}
        flim = im.parent_image
        if flim is not None and isinstance(flim, liffile.LifFlimImage):
            xml = flim.parent.xml_element
            frequency = xml.find('.//Dataset/RawData/LaserPulseFrequency')
            if frequency is not None and frequency.text is not None:
                attrs['frequency'] = float(frequency.text) * 1e-6
                clock_period = xml.find('.//Dataset/RawData/ClockPeriod')
                if clock_period is not None and clock_period.text is not None:
                    tmp = float(clock_period.text) * float(frequency.text)
                    samples = int(round(1.0 / tmp))
                    attrs['samples'] = samples
            attrs['flim_rawdata'] = flim.attrs.get('RawData', {})

    return (
        lifetime.astype(numpy.float32),
        intensity.astype(numpy.float32),
        stddev.astype(numpy.float32),
        attrs,
    )


def signal_from_lif(
    filename: str | PathLike[Any],
    /,
    *,
    image: int | str | None = None,
    dim: Literal['λ', 'Λ'] | str = 'λ',
) -> DataArray:
    """Return hyperspectral image and metadata from Leica image file.

    Leica image files may contain hyperspectral images and metadata from laser
    scanning microscopy measurements.

    Parameters
    ----------
    filename : str or Path
        Name of Leica image file to read.
    image : str or int, optional
        Index or regex pattern of image to return.
        By default, return the first image containing hyperspectral data.
    dim : str or None
        Character code of hyperspectral dimension.
        Either ``'λ'`` for emission (default) or ``'Λ'`` for excitation.

    Returns
    -------
    xarray.DataArray
        Hyperspectral image data.

        - ``coords['C']``: wavelengths in nm.
        - ``coords['T']``: time coordinates in s, if any.

    Raises
    ------
    liffile.LifFileError
        File is not a Leica image file.
    ValueError
        File is not a Leica image file or does not contain hyperspectral image.

    Notes
    -----
    The implementation is based on the
    `liffile <https://github.com/cgohlke/liffile/>`__ library.

    Reading of TCSPC histograms from FLIM measurements is not supported
    because the compression scheme is patent-pending.

    Examples
    --------
    >>> signal = signal_from_lif('ScanModesExamples.lif')  # doctest: +SKIP
    >>> signal.values  # doctest: +SKIP
    array(...)
    >>> signal.shape  # doctest: +SKIP
    (9, 128, 128)
    >>> signal.dims  # doctest: +SKIP
    ('C', 'Y', 'X')
    >>> signal.coords['C'].data  # doctest: +SKIP
    array([560, 580, 600, ..., 680, 700, 720])

    """
    import liffile

    with liffile.LifFile(filename) as lif:
        if image is None:
            # find image with excitation or emission dimension
            for im in lif.images:
                if dim in im.dims:
                    break
            else:
                raise ValueError(
                    f'{lif.filename!r} does not contain hyperspectral image'
                )
        else:
            im = lif.images[image]

        if dim not in im.dims or im.sizes[dim] < 4:
            raise ValueError(f'{im!r} does not contain spectral dimension')
        if 'C' in im.dims:
            raise ValueError(
                'hyperspectral image must not contain channel axis'
            )

        data = im.asarray()
        coords: dict[str, Any] = {
            ('C' if k == dim else k): (v * 1e9 if k == dim else v)
            for (k, v) in im.coords.items()
        }
        dims = tuple(('C' if d == dim else d) for d in im.dims)

        metadata = xarray_metadata(dims, im.shape, filename, **coords)

    from xarray import DataArray

    return DataArray(data, **metadata)
