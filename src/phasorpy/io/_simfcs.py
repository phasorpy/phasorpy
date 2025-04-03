"""SimFCS file readers and writers."""

from __future__ import annotations

__all__ = [
    'phasor_from_simfcs_referenced',
    'phasor_to_simfcs_referenced',
    'signal_from_b64',
    'signal_from_bh',
    'signal_from_bhz',
    'signal_from_z64',
]

import os
import struct
import zlib
from typing import TYPE_CHECKING

from .. import phasor_from_polar, phasor_to_polar
from .._utils import chunk_iter, parse_harmonic
from ._utils import xarray_metadata

if TYPE_CHECKING:
    from .._typing import (
        Any,
        ArrayLike,
        DataArray,
        Literal,
        NDArray,
        PathLike,
        Sequence,
    )

import numpy


def phasor_to_simfcs_referenced(
    filename: str | PathLike[Any],
    mean: ArrayLike,
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    size: int | None = None,
    dims: Sequence[str] | None = None,
) -> None:
    """Write phasor coordinate images to SimFCS referenced R64 file(s).

    SimFCS referenced R64 files store square-shaped (commonly 256x256)
    images of the average intensity, and the calibrated phasor coordinates
    (encoded as phase and modulation) of two harmonics as ZIP-compressed,
    single precision floating point arrays.
    The file format does not support any metadata.

    Images with more than two dimensions or larger than square size are
    chunked to square-sized images and saved to separate files with
    a name pattern, for example, "filename_T099_Y256_X000.r64".
    Images or chunks with less than two dimensions or smaller than square size
    are padded with NaN values.

    Parameters
    ----------
    filename : str or Path
        Name of SimFCS referenced R64 file to write.
        The file extension must be ``.r64``.
    mean : array_like
        Average intensity image.
    real : array_like
        Image of real component of calibrated phasor coordinates.
        Multiple harmonics, if any, must be in the first dimension.
        Harmonics must be starting at and increasing by one.
    imag : array_like
        Image of imaginary component of calibrated phasor coordinates.
        Multiple harmonics, if any, must be in the first dimension.
        Harmonics must be starting at and increasing by one.
    size : int, optional
        Size of X and Y dimensions of square-sized images stored in file.
        By default, ``size = min(256, max(4, sizey, sizex))``.
    dims : sequence of str, optional
        Character codes for `mean` dimensions used to format file names.

    See Also
    --------
    phasorpy.io.phasor_from_simfcs_referenced

    Examples
    --------
    >>> mean, real, imag = numpy.random.rand(3, 32, 32)
    >>> phasor_to_simfcs_referenced('_phasorpy.r64', mean, real, imag)

    """
    filename, ext = os.path.splitext(filename)
    if ext.lower() != '.r64':
        raise ValueError(f'file extension {ext} != .r64')

    # TODO: delay conversions to numpy arrays to inner loop
    mean = numpy.asarray(mean, numpy.float32)
    phi, mod = phasor_to_polar(real, imag, dtype=numpy.float32)
    del real
    del imag
    phi = numpy.rad2deg(phi)

    if phi.shape != mod.shape:
        raise ValueError(f'{phi.shape=} != {mod.shape=}')
    if mean.shape != phi.shape[-mean.ndim :]:
        raise ValueError(f'{mean.shape=} != {phi.shape[-mean.ndim:]=}')
    if phi.ndim == mean.ndim:
        phi = phi.reshape(1, *phi.shape)
        mod = mod.reshape(1, *mod.shape)
    nharmonic = phi.shape[0]

    if mean.ndim < 2:
        # not an image
        mean = mean.reshape(1, -1)
        phi = phi.reshape(nharmonic, 1, -1)
        mod = mod.reshape(nharmonic, 1, -1)

    # TODO: investigate actual size and harmonics limits of SimFCS
    sizey, sizex = mean.shape[-2:]
    if size is None:
        size = min(256, max(4, sizey, sizex))
    elif not 4 <= size <= 65535:
        raise ValueError(f'{size=} out of range [4, 65535]')

    harmonics_per_file = 2  # TODO: make this a parameter?
    chunk_shape = tuple(
        [max(harmonics_per_file, 2)] + ([1] * (phi.ndim - 3)) + [size, size]
    )
    multi_file = any(i / j > 1 for i, j in zip(phi.shape, chunk_shape))

    if dims is not None and len(dims) == phi.ndim - 1:
        dims = tuple(dims)
        dims = ('h' if dims[0].islower() else 'H',) + dims

    chunk = numpy.empty((size, size), dtype=numpy.float32)

    def rawdata_append(
        rawdata: list[bytes], a: NDArray[Any] | None = None
    ) -> None:
        if a is None:
            chunk[:] = numpy.nan
            rawdata.append(chunk.tobytes())
        else:
            sizey, sizex = a.shape[-2:]
            if sizey == size and sizex == size:
                rawdata.append(a.tobytes())
            elif sizey <= size and sizex <= size:
                chunk[:sizey, :sizex] = a[..., :sizey, :sizex]
                chunk[:sizey, sizex:] = numpy.nan
                chunk[sizey:, :] = numpy.nan
                rawdata.append(chunk.tobytes())
            else:
                raise RuntimeError  # should not be reached

    for index, label, _ in chunk_iter(
        phi.shape, chunk_shape, dims, squeeze=False, use_index=True
    ):
        rawdata = [struct.pack('I', size)]
        rawdata_append(rawdata, mean[index[1:]])
        phi_ = phi[index]
        mod_ = mod[index]
        for i in range(phi_.shape[0]):
            rawdata_append(rawdata, phi_[i])
            rawdata_append(rawdata, mod_[i])
        if phi_.shape[0] == 1:
            rawdata_append(rawdata)
            rawdata_append(rawdata)

        if not multi_file:
            label = ''
        with open(filename + label + ext, 'wb') as fh:
            fh.write(zlib.compress(b''.join(rawdata)))


def phasor_from_simfcs_referenced(
    filename: str | PathLike[Any],
    /,
    *,
    harmonic: int | Sequence[int] | Literal['all'] | str | None = None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], dict[str, Any]]:
    """Return phasor coordinates and metadata from SimFCS REF or R64 file.

    SimFCS referenced REF and R64 files contain phasor coordinate images
    (encoded as phase and modulation) for two harmonics.
    Phasor coordinates from lifetime-resolved signals are calibrated.

    Parameters
    ----------
    filename : str or Path
        Name of SimFCS REF or R64 file to read.
    harmonic : int or sequence of int, optional
        Harmonic(s) to include in returned phasor coordinates.
        By default, only the first harmonic is returned.

    Returns
    -------
    mean : ndarray
        Average intensity image.
    real : ndarray
        Image of real component of phasor coordinates.
        Multiple harmonics, if any, are in the first axis.
    imag : ndarray
        Image of imaginary component of phasor coordinates.
        Multiple harmonics, if any, are in the first axis.
    attrs : dict
        Select metadata:

        - ``'dims'`` (tuple of str):
          :ref:`Axes codes <axes>` for `mean` image dimensions.

    Raises
    ------
    lfdfiles.LfdfileError
        File is not a SimFCS REF or R64 file.

    See Also
    --------
    phasorpy.io.phasor_to_simfcs_referenced

    Notes
    -----
    The implementation is based on the
    `lfdfiles <https://github.com/cgohlke/lfdfiles/>`__ library.

    Examples
    --------
    >>> phasor_to_simfcs_referenced(
    ...     '_phasorpy.r64', *numpy.random.rand(3, 32, 32)
    ... )
    >>> mean, real, imag, _ = phasor_from_simfcs_referenced('_phasorpy.r64')
    >>> mean
    array([[...]], dtype=float32)

    """
    import lfdfiles

    ext = os.path.splitext(filename)[-1].lower()
    if ext == '.r64':
        with lfdfiles.SimfcsR64(filename) as r64:
            data = r64.asarray()
    elif ext == '.ref':
        with lfdfiles.SimfcsRef(filename) as ref:
            data = ref.asarray()
    else:
        raise ValueError(f'file extension must be .ref or .r64, not {ext!r}')

    harmonic, keep_harmonic_dim = parse_harmonic(harmonic, data.shape[0] // 2)

    mean = data[0].copy()
    real = numpy.empty((len(harmonic),) + mean.shape, numpy.float32)
    imag = numpy.empty_like(real)
    for i, h in enumerate(harmonic):
        h = (h - 1) * 2 + 1
        re, im = phasor_from_polar(numpy.deg2rad(data[h]), data[h + 1])
        real[i] = re
        imag[i] = im
    if not keep_harmonic_dim:
        real = real.reshape(mean.shape)
        imag = imag.reshape(mean.shape)

    return mean, real, imag, {'dims': ('Y', 'X')}


def signal_from_b64(
    filename: str | PathLike[Any],
    /,
) -> DataArray:
    """Return intensity image and metadata from SimFCS B64 file.

    B64 files contain one or more square intensity image(s), a carpet
    of lines, or a stream of intensity data. B64 files contain no metadata.

    Parameters
    ----------
    filename : str or Path
        Name of SimFCS B64 file to read.

    Returns
    -------
    xarray.DataArray
        Intensity image of type ``int16``.

    Raises
    ------
    lfdfiles.LfdFileError
        File is not a SimFCS B64 file.
    ValueError
        File does not contain an image stack.

    Notes
    -----
    The implementation is based on the
    `lfdfiles <https://github.com/cgohlke/lfdfiles/>`__ library.

    Examples
    --------
    >>> signal = signal_from_b64(fetch('simfcs.b64'))
    >>> signal.values
    array(...)
    >>> signal.dtype
    dtype('int16')
    >>> signal.shape
    (22, 1024, 1024)
    >>> signal.dtype
    dtype('int16')
    >>> signal.dims
    ('I', 'Y', 'X')

    """
    import lfdfiles

    with lfdfiles.SimfcsB64(filename) as b64:
        data = b64.asarray()
        if data.ndim != 3:
            raise ValueError(
                f'{os.path.basename(filename)!r} '
                'does not contain an image stack'
            )
        metadata = xarray_metadata(b64.axes, data.shape, filename)

    from xarray import DataArray

    return DataArray(data, **metadata)


def signal_from_z64(
    filename: str | PathLike[Any],
    /,
) -> DataArray:
    """Return image stack and metadata from SimFCS Z64 file.

    Z64 files commonly contain stacks of square images, such as intensity
    volumes or TCSPC histograms. Z64 files contain no metadata.

    Parameters
    ----------
    filename : str or Path
        Name of SimFCS Z64 file to read.

    Returns
    -------
    xarray.DataArray
        Image stack of type ``float32``.

    Raises
    ------
    lfdfiles.LfdFileError
        File is not a SimFCS Z64 file.

    Notes
    -----
    The implementation is based on the
    `lfdfiles <https://github.com/cgohlke/lfdfiles/>`__ library.

    Examples
    --------
    >>> signal = signal_from_z64(fetch('simfcs.z64'))
    >>> signal.values
    array(...)
    >>> signal.dtype
    dtype('float32')
    >>> signal.shape
    (256, 256, 256)
    >>> signal.dims
    ('Q', 'Y', 'X')

    """
    import lfdfiles

    with lfdfiles.SimfcsZ64(filename) as z64:
        data = z64.asarray()
        metadata = xarray_metadata(z64.axes, data.shape, filename)

    from xarray import DataArray

    return DataArray(data, **metadata)


def signal_from_bh(
    filename: str | PathLike[Any],
    /,
) -> DataArray:
    """Return TCSPC histogram and metadata from SimFCS B&H file.

    B&H files contain TCSPC histograms acquired from Becker & Hickl
    cards, or converted from other data sources. B&H files contain no metadata.

    Parameters
    ----------
    filename : str or Path
        Name of SimFCS B&H file to read.

    Returns
    -------
    xarray.DataArray
        TCSPC histogram with ref:`axes codes <axes>` ``'HYX'``,
        shape ``(256, 256, 256)``, and type ``float32``.

    Raises
    ------
    lfdfiles.LfdFileError
        File is not a SimFCS B&H file.

    Notes
    -----
    The implementation is based on the
    `lfdfiles <https://github.com/cgohlke/lfdfiles/>`__ library.

    Examples
    --------
    >>> signal = signal_from_bh(fetch('simfcs.b&h'))
    >>> signal.values
    array(...)
    >>> signal.dtype
    dtype('float32')
    >>> signal.shape
    (256, 256, 256)
    >>> signal.dims
    ('H', 'Y', 'X')

    """
    import lfdfiles

    with lfdfiles.SimfcsBh(filename) as bnh:
        assert bnh.axes is not None
        data = bnh.asarray()
        metadata = xarray_metadata(
            bnh.axes.replace('Q', 'H'), data.shape, filename
        )

    from xarray import DataArray

    return DataArray(data, **metadata)


def signal_from_bhz(
    filename: str | PathLike[Any],
    /,
) -> DataArray:
    """Return TCSPC histogram and metadata from SimFCS BHZ file.

    BHZ files contain TCSPC histograms acquired from Becker & Hickl
    cards, or converted from other data sources. BHZ files contain no metadata.

    Parameters
    ----------
    filename : str or Path
        Name of SimFCS BHZ file to read.

    Returns
    -------
    xarray.DataArray
        TCSPC histogram with ref:`axes codes <axes>` ``'HYX'``,
        shape ``(256, 256, 256)``, and type ``float32``.

    Raises
    ------
    lfdfiles.LfdFileError
        File is not a SimFCS BHZ file.

    Notes
    -----
    The implementation is based on the
    `lfdfiles <https://github.com/cgohlke/lfdfiles/>`__ library.

    Examples
    --------
    >>> signal = signal_from_bhz(fetch('simfcs.bhz'))
    >>> signal.values
    array(...)
    >>> signal.dtype
    dtype('float32')
    >>> signal.shape
    (256, 256, 256)
    >>> signal.dims
    ('H', 'Y', 'X')

    """
    import lfdfiles

    with lfdfiles.SimfcsBhz(filename) as bhz:
        assert bhz.axes is not None
        data = bhz.asarray()
        metadata = xarray_metadata(
            bhz.axes.replace('Q', 'H'), data.shape, filename
        )

    from xarray import DataArray

    return DataArray(data, **metadata)
