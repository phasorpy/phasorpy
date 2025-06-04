"""Read other file formats."""

from __future__ import annotations

__all__ = [
    # 'signal_from_czi',
    'signal_from_flif',
    'phasor_from_ifli',
    'signal_from_imspector_tiff',
    'signal_from_lsm',
    'signal_from_ptu',
    'signal_from_sdt',
]

import os
from typing import TYPE_CHECKING
from xml.etree import ElementTree

from .._utils import parse_harmonic, squeeze_dims, xarray_metadata

if TYPE_CHECKING:
    from .._typing import (
        Any,
        DataArray,
        DTypeLike,
        Literal,
        NDArray,
        PathLike,
        Sequence,
        EllipsisType,
    )

import numpy


def signal_from_sdt(
    filename: str | PathLike[Any],
    /,
    *,
    index: int = 0,
) -> DataArray:
    """Return TCSPC histogram and metadata from Becker & Hickl SDT file.

    SDT files contain TCSPC measurement data and instrumentation parameters.

    Parameters
    ----------
    filename : str or Path
        Name of Becker & Hickl SDT file to read.
    index : int, optional, default: 0
        Index of dataset to read in case the file contains multiple datasets.

    Returns
    -------
    xarray.DataArray
        TCSPC histogram with :ref:`axes codes <axes>` ``'QCYXH'`` and
        type ``uint16``, ``uint32``, or ``float32``.
        Dimensions ``'Q'`` and ``'C'`` are optional detector channels.

        - ``coords['H']``: delay-times of histogram bins in ns.
        - ``attrs['frequency']``: repetition frequency in MHz.

    Raises
    ------
    ValueError
        File is not a SDT file containing TCSPC histogram.

    Notes
    -----
    The implementation is based on the
    `sdtfile <https://github.com/cgohlke/sdtfile/>`__ library.

    Examples
    --------
    >>> signal = signal_from_sdt(fetch('tcspc.sdt'))
    >>> signal.values
    array(...)
    >>> signal.dtype
    dtype('uint16')
    >>> signal.shape
    (128, 128, 256)
    >>> signal.dims
    ('Y', 'X', 'H')
    >>> signal.coords['H'].data
    array([0, ..., 12.45])
    >>> signal.attrs['frequency']  # doctest: +NUMBER
    79.99

    """
    import sdtfile

    with sdtfile.SdtFile(filename) as sdt:
        if (
            'SPC Setup & Data File' not in sdt.info.id
            and 'SPC FCS Data File' not in sdt.info.id
        ):
            # skip DLL data
            raise ValueError(
                f'{os.path.basename(filename)!r} '
                'is not an SDT file containing TCSPC data'
            )
        # filter block types?
        # sdtfile.BlockType(sdt.block_headers[index].block_type).contents
        # == 'PAGE_BLOCK'
        data = sdt.data[index]
        times = sdt.times[index] * 1e9

    # TODO: get spatial coordinates from scanner settings?
    metadata = xarray_metadata(
        'QCYXH'[-data.ndim :], data.shape, filename, H=times
    )
    metadata['attrs']['frequency'] = 1e3 / float(times[-1] + times[1])

    from xarray import DataArray

    return DataArray(data, **metadata)


def signal_from_ptu(
    filename: str | PathLike[Any],
    /,
    selection: Sequence[int | slice | EllipsisType | None] | None = None,
    *,
    trimdims: Sequence[Literal['T', 'C', 'H']] | str | None = None,
    dtype: DTypeLike | None = None,
    frame: int | None = None,
    channel: int | None = 0,
    dtime: int | None = 0,
    keepdims: bool = False,
) -> DataArray:
    """Return TCSPC histogram and metadata from PicoQuant PTU T3 mode file.

    PTU files contain TCSPC measurement data and instrumentation parameters,
    which are decoded to a multi-dimensional TCSPC histogram.

    Parameters
    ----------
    filename : str or Path
        Name of PTU file to read.
    selection : sequence of index types, optional
        Indices for all dimensions of image mode files:

        - ``None``: return all items along axis (default).
        - ``Ellipsis``: return all items along multiple axes.
        - ``int``: return single item along axis.
        - ``slice``: return chunk of axis.
          ``slice.step`` is a binning factor.
          If ``slice.step=-1``, integrate all items along axis.

    trimdims : str, optional, default: 'TCH'
        Axes to trim.
    dtype : dtype-like, optional, default: uint16
        Unsigned integer type of TCSPC histogram.
        Increase the bit depth to avoid overflows when integrating.
    frame : int, optional
        If < 0, integrate time axis, else return specified frame.
        Overrides `selection` for axis ``T``.
    channel : int, optional
        Index of channel to return.
        By default, return the first channel.
        If < 0, integrate channel axis.
        Overrides `selection` for axis ``C``.
    dtime : int, optional, default: 0
        Specifies number of bins in TCSPC histogram.
        If 0 (default), return the number of bins in one period.
        If < 0, integrate delay-time axis (image mode only).
        If > 0, return up to specified bin.
        Overrides `selection` for axis ``H``.
    keepdims : bool, optional, default: False
        If true, return reduced axes as size-one dimensions.

    Returns
    -------
    xarray.DataArray
        TCSPC histogram with :ref:`axes codes <axes>` ``'TYXCH'`` and
        type specified in ``dtype``:

        - ``coords['H']``: delay-times of histogram bins in ns.
        - ``attrs['frequency']``: repetition frequency in MHz.
        - ``attrs['ptu_tags']``: metadata read from PTU file.

        Size-one dimensions are prepended to point mode data to make them
        broadcastable to image data.

    Raises
    ------
    ptufile.PqFileError
        File is not a PicoQuant PTU file or is corrupted.
    ValueError
        File is not a PicoQuant PTU T3 mode file containing TCSPC data.

    Notes
    -----
    The implementation is based on the
    `ptufile <https://github.com/cgohlke/ptufile/>`__ library.

    Examples
    --------
    >>> signal = signal_from_ptu(fetch('hazelnut_FLIM_single_image.ptu'))
    >>> signal.values
    array(...)
    >>> signal.dtype
    dtype('uint16')
    >>> signal.shape
    (5, 256, 256, 132)
    >>> signal.dims
    ('T', 'Y', 'X', 'H')
    >>> signal.coords['H'].data
    array([0, ..., 12.7])
    >>> signal.attrs['frequency']  # doctest: +NUMBER
    78.02

    """
    import ptufile
    from xarray import DataArray

    with ptufile.PtuFile(filename, trimdims=trimdims) as ptu:
        if not ptu.is_t3:
            raise ValueError(f'{ptu.filename!r} is not a T3 mode PTU file')
        if ptu.is_image:
            data = ptu.decode_image(
                selection,
                dtype=dtype,
                frame=frame,
                channel=channel,
                dtime=dtime,
                keepdims=keepdims,
                asxarray=True,
            )
            assert isinstance(data, DataArray)
        elif ptu.measurement_submode == 1:
            # point mode IRF
            if dtime == -1:
                raise ValueError(f'{dtime=} not supported for point mode')
            data = ptu.decode_histogram(
                dtype=dtype, dtime=dtime, asxarray=True
            )
            assert isinstance(data, DataArray)
            if channel is not None:
                if keepdims:
                    data = data[channel : channel + 1]
                else:
                    data = data[channel]
            # prepend dimensions as needed to appear image-like
            data = data.expand_dims(dim={'Y': 1, 'X': 1})
            if keepdims:
                data = data.expand_dims(dim={'T': 1})
        else:
            raise ValueError(
                f'{ptu.filename!r} is not a point or image mode PTU file'
            )

        data.attrs['ptu_tags'] = ptu.tags
        data.attrs['frequency'] = ptu.frequency * 1e-6  # MHz
        data.coords['H'] = data.coords['H'] * 1e9

    return data


def signal_from_lsm(
    filename: str | PathLike[Any],
    /,
) -> DataArray:
    """Return hyperspectral image and metadata from Zeiss LSM file.

    LSM files contain multi-dimensional images and metadata from laser
    scanning microscopy measurements. The file format is based on TIFF.

    Parameters
    ----------
    filename : str or Path
        Name of Zeiss LSM file to read.

    Returns
    -------
    xarray.DataArray
        Hyperspectral image data.
        Usually, a 3-to-5-dimensional array of type ``uint8`` or ``uint16``.

        - ``coords['C']``: wavelengths in nm.
        - ``coords['T']``: time coordinates in s, if any.

    Raises
    ------
    tifffile.TiffFileError
        File is not a TIFF file.
    ValueError
        File is not an LSM file or does not contain hyperspectral image.

    Notes
    -----
    The implementation is based on the
    `tifffile <https://github.com/cgohlke/tifffile/>`__ library.

    Examples
    --------
    >>> signal = signal_from_lsm(fetch('paramecium.lsm'))
    >>> signal.values
    array(...)
    >>> signal.dtype
    dtype('uint8')
    >>> signal.shape
    (30, 512, 512)
    >>> signal.dims
    ('C', 'Y', 'X')
    >>> signal.coords['C'].data  # wavelengths
    array([423, ..., 713])

    """
    import tifffile

    with tifffile.TiffFile(filename) as tif:
        if not tif.is_lsm:
            raise ValueError(f'{tif.filename!r} is not an LSM file')

        page = tif.pages.first
        lsminfo = tif.lsm_metadata
        channels = page.tags[258].count

        if channels < 4 or lsminfo is None or lsminfo['SpectralScan'] != 1:
            raise ValueError(
                f'{tif.filename!r} does not contain hyperspectral image'
            )

        # TODO: contribute this to tifffile
        series = tif.series[0]
        data = series.asarray()
        dims = tuple(series.axes)
        coords = {}
        # channel wavelengths
        axis = dims.index('C')
        wavelengths = lsminfo['ChannelWavelength'].mean(axis=1)
        if wavelengths.size != data.shape[axis]:
            raise ValueError(
                f'{tif.filename!r} wavelengths do not match channel axis'
            )
        # stack may contain non-wavelength frame
        indices = wavelengths > 0
        wavelengths = wavelengths[indices]
        if wavelengths.size < 3:
            raise ValueError(
                f'{tif.filename!r} does not contain hyperspectral image'
            )
        wavelengths *= 1e9
        data = data.take(indices.nonzero()[0], axis=axis)
        coords['C'] = wavelengths
        # time stamps
        if 'T' in dims:
            coords['T'] = lsminfo['TimeStamps'] - lsminfo['TimeStamps'][0]
            if coords['T'].size != data.shape[dims.index('T')]:
                raise ValueError(
                    f'{tif.filename!r} timestamps do not match time axis'
                )
        # spatial coordinates
        for ax in 'ZYX':
            if ax in dims:
                size = data.shape[dims.index(ax)]
                coords[ax] = numpy.linspace(
                    lsminfo[f'Origin{ax}'],
                    size * lsminfo[f'VoxelSize{ax}'],
                    size,
                    endpoint=False,
                    dtype=numpy.float64,
                )
        metadata = xarray_metadata(series.axes, data.shape, filename, **coords)

    from xarray import DataArray

    return DataArray(data, **metadata)


def signal_from_imspector_tiff(
    filename: str | PathLike[Any],
    /,
) -> DataArray:
    """Return TCSPC histogram and metadata from ImSpector TIFF file.

    Parameters
    ----------
    filename : str or Path
        Name of ImSpector FLIM TIFF file to read.

    Returns
    -------
    xarray.DataArray
        TCSPC histogram with :ref:`axes codes <axes>` ``'HTZYX'`` and
        type ``uint16``.

        - ``coords['H']``: delay-times of histogram bins in ns.
        - ``attrs['frequency']``: repetition frequency in MHz.

    Raises
    ------
    tifffile.TiffFileError
        File is not a TIFF file.
    ValueError
        File is not an ImSpector FLIM TIFF file.

    Notes
    -----
    The implementation is based on the
    `tifffile <https://github.com/cgohlke/tifffile/>`__ library.

    Examples
    --------
    >>> signal = signal_from_imspector_tiff(fetch('Embryo.tif'))
    >>> signal.values
    array(...)
    >>> signal.dtype
    dtype('uint16')
    >>> signal.shape
    (56, 512, 512)
    >>> signal.dims
    ('H', 'Y', 'X')
    >>> signal.coords['H'].data  # dtime bins
    array([0, ..., 12.26])
    >>> signal.attrs['frequency']  # doctest: +NUMBER
    80.109

    """
    import tifffile

    with tifffile.TiffFile(filename) as tif:
        tags = tif.pages.first.tags
        omexml = tags.valueof(270, '')
        make = tags.valueof(271, '')

        if (
            make != 'ImSpector'
            or not omexml.startswith('<?xml version')
            or len(tif.series) != 1
            or not tif.is_ome
        ):
            raise ValueError(f'{tif.filename!r} is not an ImSpector TIFF file')

        series = tif.series[0]
        ndim = series.ndim
        axes = series.axes
        shape = series.shape

        if ndim < 3 or not axes.endswith('YX'):
            raise ValueError(
                f'{tif.filename!r} is not an ImSpector FLIM TIFF file'
            )

        data = series.asarray()

    attrs: dict[str, Any] = {}
    coords = {}
    physical_size = {}

    root = ElementTree.fromstring(omexml)
    ns = {
        '': 'http://www.openmicroscopy.org/Schemas/OME/2008-02',
        'ca': 'http://www.openmicroscopy.org/Schemas/CA/2008-02',
    }

    description = root.find('.//Description', ns)
    if (
        description is not None
        and description.text
        and description.text != 'not_specified'
    ):
        attrs['description'] = description.text

    pixels = root.find('.//Image/Pixels', ns)
    assert pixels is not None
    for ax in 'TZYX':
        attrib = 'TimeIncrement' if ax == 'T' else f'PhysicalSize{ax}'
        if ax not in axes or attrib not in pixels.attrib:
            continue
        size = float(pixels.attrib[attrib]) * shape[axes.index(ax)]
        physical_size[ax] = size
        coords[ax] = numpy.linspace(
            0.0,
            size,
            shape[axes.index(ax)],
            endpoint=False,
            dtype=numpy.float64,
        )

    axes_labels = root.find('.//Image/ca:CustomAttributes/AxesLabels', ns)
    if (
        axes_labels is None
        or 'X' not in axes_labels.attrib
        or 'TCSPC' not in axes_labels.attrib['X']
        or 'FirstAxis' not in axes_labels.attrib
        or 'SecondAxis' not in axes_labels.attrib
    ):
        raise ValueError(
            f'{tif.filename!r} is not an ImSpector FLIM TIFF file'
        )

    if axes_labels.attrib['FirstAxis'] == 'lifetime' or axes_labels.attrib[
        'FirstAxis'
    ].endswith('TCSPC T'):
        ax = axes[-3]
        assert axes_labels.attrib['FirstAxis-Unit'] == 'ns'
    elif ndim > 3 and (
        axes_labels.attrib['SecondAxis'] == 'lifetime'
        or axes_labels.attrib['SecondAxis'].endswith('TCSPC T')
    ):
        ax = axes[-4]
        assert axes_labels.attrib['SecondAxis-Unit'] == 'ns'
    else:
        raise ValueError(
            f'{tif.filename!r} is not an ImSpector FLIM TIFF file'
        )
    axes = axes.replace(ax, 'H')
    coords['H'] = coords[ax]
    del coords[ax]

    attrs['frequency'] = float(1000.0 / physical_size[ax])

    metadata = xarray_metadata(axes, shape, filename, attrs=attrs, **coords)

    from xarray import DataArray

    return DataArray(data, **metadata)


def phasor_from_ifli(
    filename: str | PathLike[Any],
    /,
    *,
    channel: int | None = 0,
    harmonic: int | Sequence[int] | Literal['all', 'any'] | str | None = None,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], dict[str, Any]]:
    """Return phasor coordinates and metadata from ISS IFLI file.

    ISS VistaVision IFLI files contain calibrated phasor coordinates for
    possibly several positions, wavelengths, time points, channels, slices,
    and frequencies from analog or digital frequency-domain fluorescence
    lifetime measurements.

    Parameters
    ----------
    filename : str or Path
        Name of ISS IFLI file to read.
    channel : int, optional
        Index of channel to return.
        By default, return the first channel.
        If None, return all channels.
    harmonic : int, sequence of int, or 'all', optional
        Harmonic(s) to return from file.
        If None (default), return the first harmonic stored in file.
        If `'all'`, return all harmonics of first frequency stored in file.
        If `'any'`, return all frequencies as stored in file, not necessarily
        harmonics of the first frequency.
        If a list, the first axes of the returned `real` and `imag` arrays
        contain specified harmonic(s).
        If an integer, the returned `real` and `imag` arrays are single
        harmonic and have the same shape as `mean`.
    **kwargs
        Additional arguments passed to :py:meth:`lfdfiles.VistaIfli.asarray`,
        for example ``memmap=True``.

    Returns
    -------
    mean : ndarray
        Average intensity image.
        May have up to 7 dimensions in ``'RETCZYX'`` order.
    real : ndarray
        Image of real component of phasor coordinates.
        Same shape as `mean`, except it may have a harmonic/frequency
        dimension prepended.
    imag : ndarray
        Image of imaginary component of phasor coordinates.
        Same shape as `real`.
    attrs : dict
        Select metadata:

        - ``'dims'`` (tuple of str):
          :ref:`Axes codes <axes>` for `mean` image dimensions.
        - ``'harmonic'`` (int or list of int):
          Harmonic(s) present in `real` and `imag`.
          If a scalar, `real` and `imag` are single harmonic and contain no
          harmonic axes.
          If a list, `real` and `imag` contain one or more harmonics in the
          first axis.
        - ``'frequency'`` (float):
          Fundamental frequency of time-resolved phasor coordinates in MHz.
        - ``'samples'`` (int):
            Number of samples per frequency.
        - ``'ifli_header'`` (dict):
          Metadata from IFLI file header.

    Raises
    ------
    lfdfiles.LfdFileError
        File is not an ISS IFLI file.
    IndexError
        Harmonic is not found in file.

    Notes
    -----
    The implementation is based on the
    `lfdfiles <https://github.com/cgohlke/lfdfiles/>`__ library.

    Examples
    --------
    >>> mean, real, imag, attr = phasor_from_ifli(
    ...     fetch('frequency_domain.ifli'), harmonic='all'
    ... )
    >>> mean.shape
    (256, 256)
    >>> real.shape
    (4, 256, 256)
    >>> attr['dims']
    ('Y', 'X')
    >>> attr['harmonic']
    [1, 2, 3, 5]
    >>> attr['frequency']  # doctest: +NUMBER
    80.33
    >>> attr['samples']
    64
    >>> attr['ifli_header']
    {'Version': 16, ... 'ModFrequency': (...), 'RefLifetime': (2.5,), ...}

    """
    import lfdfiles

    with lfdfiles.VistaIfli(filename) as ifli:
        assert ifli.axes is not None
        data = ifli.asarray(**kwargs)
        header = ifli.header
        axes = ifli.axes

    if channel is not None:
        data = data[:, :, :, channel]
        axes = axes[:3] + axes[4:]

    shape, dims, _ = squeeze_dims(data.shape, axes, skip='YXF')
    data = data.reshape(shape)
    data = numpy.moveaxis(data, -2, 0)  # move frequency to first axis
    mean = data[..., 0].mean(axis=0)  # average frequencies
    real = data[..., 1].copy()
    imag = data[..., 2].copy()
    dims = dims[:-2]
    del data

    samples = header['HistogramResolution']
    frequencies = header['ModFrequency']
    frequency = frequencies[0]
    harmonic_stored = [
        (
            int(round(f / frequency))
            if (0.99 < f / frequency % 1.0) < 1.01
            else None
        )
        for f in frequencies
    ]

    index: int | list[int]
    if harmonic is None:
        # return first harmonic in file
        keepdims = False
        harmonic = [1]
        index = [0]
    elif isinstance(harmonic, str) and harmonic in {'all', 'any'}:
        keepdims = True
        if harmonic == 'any':
            # return any frequency
            harmonic = [
                (frequencies[i] / frequency if h is None else h)
                for i, h in enumerate(harmonic_stored)
            ]
            index = list(range(len(harmonic_stored)))
        else:
            # return only harmonics of first frequency
            harmonic = [h for h in harmonic_stored if h is not None]
            index = [i for i, h in enumerate(harmonic_stored) if h is not None]
    else:
        # return specified harmonics
        harmonic, keepdims = parse_harmonic(
            harmonic, max(h for h in harmonic_stored if h is not None)
        )
        try:
            index = [harmonic_stored.index(h) for h in harmonic]
        except ValueError as exc:
            raise IndexError('harmonic not found') from exc

    real = real[index]
    imag = imag[index]
    if not keepdims:
        real = real[0]
        imag = imag[0]

    attrs = {
        'dims': tuple(dims),
        'harmonic': harmonic,
        'frequency': frequency * 1e-6,
        'samples': samples,
        'ifli_header': header,
    }

    return mean, real, imag, attrs


def signal_from_flif(
    filename: str | PathLike[Any],
    /,
) -> DataArray:
    """Return phase images and metadata from FlimFast FLIF file.

    FlimFast FLIF files contain phase images and metadata from full-field,
    frequency-domain fluorescence lifetime measurements.

    Parameters
    ----------
    filename : str or Path
        Name of FlimFast FLIF file to read.

    Returns
    -------
    xarray.DataArray
        Phase images with :ref:`axes codes <axes>` ``'THYX'`` and
        type ``uint16``:

        - ``coords['H']``: phases in radians.
        - ``attrs['frequency']``: repetition frequency in MHz.
        - ``attrs['ref_phase']``: measured phase of reference.
        - ``attrs['ref_mod']``: measured modulation of reference.
        - ``attrs['ref_tauphase']``: lifetime from phase of reference in ns.
        - ``attrs['ref_taumod']``: lifetime from modulation of reference in ns.

    Raises
    ------
    lfdfiles.LfdFileError
        File is not a FlimFast FLIF file.

    Notes
    -----
    The implementation is based on the
    `lfdfiles <https://github.com/cgohlke/lfdfiles/>`__ library.

    Examples
    --------
    >>> signal = signal_from_flif(fetch('flimfast.flif'))
    >>> signal.values
    array(...)
    >>> signal.dtype
    dtype('uint16')
    >>> signal.shape
    (32, 220, 300)
    >>> signal.dims
    ('H', 'Y', 'X')
    >>> signal.coords['H'].data
    array([0, ..., 6.087], dtype=float32)
    >>> signal.attrs['frequency']  # doctest: +NUMBER
    80.65

    """
    import lfdfiles

    with lfdfiles.FlimfastFlif(filename) as flif:
        nphases = int(flif.header.phases)
        data = flif.asarray()
        if data.shape[0] < nphases:
            raise ValueError(f'measured phases {data.shape[0]} < {nphases=}')
        if data.shape[0] % nphases != 0:
            data = data[: (data.shape[0] // nphases) * nphases]
        data = data.reshape(-1, nphases, data.shape[1], data.shape[2])
        if data.shape[0] == 1:
            data = data[0]
            axes = 'HYX'
        else:
            axes = 'THYX'
        # TODO: check if phases are ordered
        phases = numpy.radians(flif.records['phase'][:nphases])
        metadata = xarray_metadata(axes, data.shape, H=phases)
        attrs = metadata['attrs']
        attrs['frequency'] = float(flif.header.frequency)
        attrs['ref_phase'] = float(flif.header.measured_phase)
        attrs['ref_mod'] = float(flif.header.measured_mod)
        attrs['ref_tauphase'] = float(flif.header.ref_tauphase)
        attrs['ref_taumod'] = float(flif.header.ref_taumod)

    from xarray import DataArray

    return DataArray(data, **metadata)
