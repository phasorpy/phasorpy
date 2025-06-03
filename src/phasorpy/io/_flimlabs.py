"""Read FLIM LABS file formats."""

from __future__ import annotations

__all__ = ['phasor_from_flimlabs_json', 'signal_from_flimlabs_json']

import json
from typing import TYPE_CHECKING

from .._utils import parse_harmonic, xarray_metadata

if TYPE_CHECKING:
    from .._typing import (
        Any,
        DataArray,
        DTypeLike,
        Literal,
        NDArray,
        PathLike,
        Sequence,
    )

import numpy


def phasor_from_flimlabs_json(
    filename: str | PathLike[Any],
    /,
    channel: int | None = 0,
    harmonic: int | Sequence[int] | Literal['all'] | str | None = None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], dict[str, Any]]:
    """Return phasor coordinates and metadata from FLIM LABS JSON phasor file.

    FLIM LABS JSON files may contain calibrated phasor coordinates
    (possibly for multiple channels and harmonics) and metadata from
    digital frequency-domain measurements.

    Parameters
    ----------
    filename : str or Path
        Name of FLIM LABS JSON phasor file to read.
        The file name usually contains the string "_phasor".
    channel : int, optional
        Index of channel to return.
        By default, return the first channel.
        If None, return all channels.
    harmonic : int, sequence of int, or 'all', optional
        Harmonic(s) to return from file.
        If None (default), return the first harmonic stored in the file.
        If `'all'`, return all harmonics as stored in file.
        If a list, the first axes of the returned `real` and `imag` arrays
        contain specified harmonic(s).
        If an integer, the returned `real` and `imag` arrays are single
        harmonic and have the same shape as `mean`.

    Returns
    -------
    mean : ndarray
        Average intensity image.
        Zeroed if an intensity image is not present in file.
    real : ndarray
        Image of real component of phasor coordinates.
    imag : ndarray
        Image of imaginary component of phasor coordinates.
    attrs : dict
        Select metadata:

        - ``'dims'`` (tuple of str):
          :ref:`Axes codes <axes>` for `mean` image dimensions.
        - ``'harmonic'`` (int):
          Harmonic of `real` and `imag`.
        - ``'frequency'`` (float):
          Fundamental frequency of time-resolved phasor coordinates in MHz.
        - ``'flimlabs_header'`` (dict):
          FLIM LABS file header.

    Raises
    ------
    ValueError
        File is not a FLIM LABS JSON file containing phasor coordinates.
    IndexError
        Harmonic or channel not found in file.

    See Also
    --------
    phasorpy.io.signal_from_flimlabs_json

    Examples
    --------
    >>> mean, real, imag, attrs = phasor_from_flimlabs_json(
    ...     fetch('Convallaria_m2_1740751781_phasor_ch1.json'), harmonic='all'
    ... )
    >>> real.shape
    (3, 256, 256)
    >>> attrs['dims']
    ('Y', 'X')
    >>> attrs['harmonic']
    [1, 2, 3]
    >>> attrs['frequency']  # doctest: +NUMBER
    40.00

    """
    with open(filename, 'rb') as fh:
        try:
            data = json.load(fh)
        except Exception as exc:
            raise ValueError('not a valid JSON file') from exc

    if (
        'header' not in data
        or 'phasors_data' not in data
        or 'laser_period_ns' not in data['header']
        or 'file_id' not in data['header']
        # or data['header']['file_id'] != [73, 80, 71, 49]  # 'IPG1'
    ):
        raise ValueError(
            'not a FLIM LABS JSON file containing phasor coordinates'
        )

    header = data['header']
    phasor_data = data['phasors_data']

    harmonics = []
    channels = []  # 1-based
    for d in phasor_data:
        h = d['harmonic']
        if h not in harmonics:
            harmonics.append(h)
        c = d['channel']
        if c not in channels:
            channels.append(c)
    harmonics = sorted(harmonics)
    channels = sorted(channels)

    if channel is not None:
        if channel + 1 not in channels:
            raise IndexError(f'{channel=}')
        channel += 1  # 1-based index

    if isinstance(harmonic, str) and harmonic == 'all':
        harmonic = harmonics
        keep_harmonic_axis = True
    else:
        harmonic, keep_harmonic_axis = parse_harmonic(harmonic, harmonics[-1])
    if any(h not in harmonics for h in harmonic):
        raise IndexError(f'{harmonic=} not in {harmonics!r}')
    harmonic_index = {h: i for i, h in enumerate(harmonic)}

    nharmonics = len(harmonic)
    nchannels = len(channels) if channel is None else 1
    height = header['image_height']
    width = header['image_width']
    dtype = numpy.float32

    shape: tuple[int, ...] = nharmonics, nchannels, height, width
    axes: str = 'CYX'
    mean = numpy.zeros(shape[1:], dtype)
    real = numpy.zeros(shape, dtype)
    imag = numpy.zeros(shape, dtype)

    for d in phasor_data:
        h = d['harmonic']
        if h not in harmonic_index:
            continue
        h = harmonic_index[h]
        if channel is not None:
            if d['channel'] != channel:
                continue
            c = 0
        else:
            c = channels.index(d['channel'])

        real[h, c] = numpy.asarray(d['g_data'], dtype)
        imag[h, c] = numpy.asarray(d['s_data'], dtype)

    if 'intensities_data' in data:
        from .._phasorpy import _flimlabs_mean

        mean.shape = nchannels, height * width
        _flimlabs_mean(
            mean,
            data['intensities_data'],
            -1 if channel is None else channels.index(channel),
        )
        mean.shape = shape[1:]
        # JSON cannot store NaN values
        nan_mask = mean == 0
        real[:, nan_mask] = numpy.nan
        imag[:, nan_mask] = numpy.nan
        del nan_mask

    if nchannels == 1:
        axes = axes[1:]
        mean = mean[0]
        real = real[:, 0]
        imag = imag[:, 0]

    if not keep_harmonic_axis:
        real = real[0]
        imag = imag[0]

    attrs = {
        'dims': tuple(axes),
        'samples': 256,
        'harmonic': harmonic if keep_harmonic_axis else harmonic[0],
        'frequency': 1000.0 / header['laser_period_ns'],
        'flimlabs_header': header,
    }

    return mean, real, imag, attrs


def signal_from_flimlabs_json(
    filename: str | PathLike[Any],
    /,
    *,
    channel: int | None = 0,
    dtype: DTypeLike | None = None,
) -> DataArray:
    """Return TCSPC histogram and metadata from FLIM LABS JSON imaging file.

    FLIM LABS JSON imaging files contain encoded, multi-channel TCSPC
    histograms and metadata from digital frequency-domain measurements.

    Parameters
    ----------
    filename : str or Path
        Name of FLIM LABS JSON imaging file to read.
        The file name usually contains the string "_imaging" or "_phasor".
    channel : int, optional
        Index of channel to return.
        By default, return the first channel.
        If None, return all channels.
    dtype : dtype-like, optional, default: uint16
        Unsigned integer type of TCSPC histogram.
        Increase the bit-depth for high photon counts.

    Returns
    -------
    xarray.DataArray
        TCSPC histogram with :ref:`axes codes <axes>` ``'CYXH'`` and
        type specified in ``dtype``:

        - ``coords['H']``: delay-times of histogram bins in ns.
        - ``attrs['frequency']``: laser repetition frequency in MHz.
        - ``attrs['flimlabs_header']``: FLIM LABS file header.

    Raises
    ------
    ValueError
        File is not a FLIM LABS JSON file containing TCSPC histogram.
        `dtype` is not an unsigned integer.
    IndexError
        Channel out of range.

    See Also
    --------
    phasorpy.io.phasor_from_flimlabs_json

    Examples
    --------
    >>> signal = signal_from_flimlabs_json(
    ...     fetch('Convallaria_m2_1740751781_phasor_ch1.json')
    ... )
    >>> signal.values
    array(...)
    >>> signal.shape
    (256, 256, 256)
    >>> signal.dims
    ('Y', 'X', 'H')
    >>> signal.coords['H'].data
    array(...)
    >>> signal.attrs['frequency']  # doctest: +NUMBER
    40.00

    """
    with open(filename, 'rb') as fh:
        try:
            data = json.load(fh)
        except Exception as exc:
            raise ValueError('not a valid JSON file') from exc

    if (
        'header' not in data
        or 'laser_period_ns' not in data['header']
        or 'file_id' not in data['header']
        or ('data' not in data and 'intensities_data' not in data)
    ):
        raise ValueError(
            'not a FLIM LABS JSON file containing TCSPC histogram'
        )

    if dtype is None:
        dtype = numpy.uint16
    else:
        dtype = numpy.dtype(dtype)
        if dtype.kind != 'u':
            raise ValueError(f'{dtype=} is not an unsigned integer type')

    header = data['header']
    nchannels = len([c for c in header['channels'] if c])
    height = header['image_height']
    width = header['image_width']
    frequency = 1000.0 / header['laser_period_ns']

    if channel is not None:
        if channel >= nchannels or channel < 0:
            raise IndexError(f'{channel=} out of range[0, {nchannels=}]')
        nchannels = 1

    if 'data' in data:
        # file_id = [73, 77, 71, 49]  # 'IMG1'
        intensities_data = data['data']
    else:
        # file_id = [73, 80, 71, 49]  # 'IPG1'
        intensities_data = data['intensities_data']

    from .._phasorpy import _flimlabs_signal

    signal = numpy.zeros((nchannels, height * width, 256), dtype)
    _flimlabs_signal(
        signal,
        intensities_data,
        -1 if channel is None else channel,
    )

    if channel is None and nchannels > 1:
        signal.shape = (nchannels, height, width, 256)
        axes = 'CYXH'
    else:
        signal.shape = (height, width, 256)
        axes = 'YXH'

    coords: dict[str, Any] = {}
    coords['H'] = numpy.linspace(
        0.0, header['laser_period_ns'], 256, endpoint=False
    )
    if channel is None and nchannels > 1:
        coords['C'] = numpy.asarray(
            [i for i, c in enumerate(header['channels']) if c]
        )

    metadata = xarray_metadata(axes, signal.shape, filename, **coords)
    attrs = metadata['attrs']
    attrs['frequency'] = frequency
    attrs['flimlabs_header'] = header

    from xarray import DataArray

    return DataArray(signal, **metadata)
