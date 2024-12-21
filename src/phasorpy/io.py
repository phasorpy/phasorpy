"""Read and write time-resolved and hyperspectral image file formats.

The ``phasorpy.io`` module provides functions to:

- read and write phasor coordinate images in OME-TIFF format, which can be
  imported in Bio-Formats and Fiji:

  - :py:func:`phasor_to_ometiff`
  - :py:func:`phasor_from_ometiff`

- read and write phasor coordinate images in SimFCS referenced R64 format:

  - :py:func:`phasor_to_simfcs_referenced`
  - :py:func:`phasor_from_simfcs_referenced`

- read time-resolved and hyperspectral image data and metadata (as relevant
  to phasor analysis) from many file formats used in bio-imaging:

  - :py:func:`read_imspector_tiff` - ImSpector FLIM TIFF
  - :py:func:`read_flimlabs_json` - FLIMLABS JSON
  - :py:func:`read_lsm` - Zeiss LSM
  - :py:func:`read_ifli` - ISS IFLI
  - :py:func:`read_sdt` - Becker & Hickl SDT
  - :py:func:`read_ptu` - PicoQuant PTU
  - :py:func:`read_fbd` - FLIMbox FBD
  - :py:func:`read_flif` - FlimFast FLIF
  - :py:func:`read_b64` - SimFCS B64
  - :py:func:`read_z64` - SimFCS Z64
  - :py:func:`read_bhz` - SimFCS BHZ
  - :py:func:`read_bh` - SimFCS B&H

  Support for other file formats is being considered:

  - OME-TIFF
  - Zeiss CZI
  - Leica LIF
  - Nikon ND2
  - Olympus OIB/OIF
  - Olympus OIR

The functions are implemented as minimal wrappers around specialized
third-party file reader libraries, currently
`tifffile <https://github.com/cgohlke/tifffile>`_,
`ptufile <https://github.com/cgohlke/ptufile>`_,
`sdtfile <https://github.com/cgohlke/sdtfile>`_, and
`lfdfiles <https://github.com/cgohlke/lfdfiles>`_.
For advanced or unsupported use cases, consider using these libraries directly.

The read functions typically have the following signature::

    read_ext(
        filename: str | PathLike,
        /,
        **kwargs
    ): -> xarray.DataArray

where ``ext`` indicates the file format and ``kwargs`` are optional arguments
passed to the underlying file reader library or used to select which data is
returned. The returned `xarray.DataArray
<https://docs.xarray.dev/en/stable/user-guide/data-structures.html>`_
contains an n-dimensional array with labeled coordinates, dimensions, and
attributes:

- ``data`` or ``values`` (*array_like*)

  Numpy array or array-like holding the array's values.

- ``dims`` (*tuple of str*)

  :ref:`Axes character codes <axes>` for each dimension in ``data``.
  For example, ``('T', 'C', 'Y', 'X')`` defines the dimension order in a
  4-dimensional array of a time-series of multi-channel images.

- ``coords`` (*dict_like[str, array_like]*)

  Coordinate arrays labelling each point in the data array.
  The keys are :ref:`axes character codes <axes>`.
  Values are 1-dimensional arrays of numbers or strings.
  For example, ``coords['C']`` could be an array of emission wavelengths.

- ``attrs`` (*dict[str, Any]*)

  Arbitrary metadata such as measurement or calibration parameters required to
  interpret the data values.
  For example, the laser repetition frequency of a time-resolved measurement.

.. _axes:

Axes character codes from the OME model and tifffile library are used as
``dims`` items and ``coords`` keys:

- ``'X'`` : width (OME)
- ``'Y'`` : height (OME)
- ``'Z'`` : depth (OME)
- ``'S'`` : sample (color components or phasor coordinates)
- ``'I'`` : sequence (of images, frames, or planes)
- ``'T'`` : time  (OME)
- ``'C'`` : channel (OME. Acquisition path or emission wavelength)
- ``'A'`` : angle (OME)
- ``'P'`` : phase (OME. In LSM, ``'P'`` maps to position)
- ``'R'`` : tile (OME. Region, position, or mosaic)
- ``'H'`` : lifetime histogram (OME)
- ``'E'`` : lambda (OME. Excitation wavelength)
- ``'F'`` : frequency (ISS)
- ``'Q'`` : other (OME. Harmonics in PhasorPy TIFF)
- ``'L'`` : exposure (FluoView)
- ``'V'`` : event (FluoView)
- ``'M'`` : mosaic (LSM 6)
- ``'J'`` : column (NDTiff)
- ``'K'`` : row (NDTiff)

"""

from __future__ import annotations

__all__ = [
    'phasor_from_ometiff',
    'phasor_from_simfcs_referenced',
    'phasor_to_ometiff',
    'phasor_to_simfcs_referenced',
    'read_b64',
    'read_bh',
    'read_bhz',
    # 'read_czi',
    'read_fbd',
    'read_flif',
    'read_flimlabs_json',
    'read_ifli',
    'read_imspector_tiff',
    # 'read_lif',
    'read_lsm',
    # 'read_nd2',
    # 'read_oif',
    # 'read_oir',
    # 'read_ometiff',
    'read_ptu',
    'read_sdt',
    'read_z64',
    '_squeeze_axes',
]

import logging
import os
import re
import struct
import zlib
from typing import TYPE_CHECKING

from ._utils import chunk_iter, parse_harmonic
from .phasor import phasor_from_polar, phasor_to_polar

if TYPE_CHECKING:
    from ._typing import (
        Any,
        ArrayLike,
        DataArray,
        DTypeLike,
        EllipsisType,
        Literal,
        NDArray,
        PathLike,
        Sequence,
    )

import numpy

logger = logging.getLogger(__name__)


def phasor_to_ometiff(
    filename: str | PathLike[Any],
    mean: ArrayLike,
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    frequency: float | None = None,
    harmonic: int | Sequence[int] | None = None,
    axes: str | None = None,
    dtype: DTypeLike | None = None,
    description: str | None = None,
    **kwargs: Any,
) -> None:
    """Write phasor coordinate images and metadata to OME-TIFF file.

    The OME-TIFF format is compatible with Bio-Formats and Fiji.

    By default, write phasor coordinates as single precision floating point
    values to separate image series.
    Write images larger than (1024, 1024) as (256, 256) tiles, datasets
    larger than 2 GB as BigTIFF, and datasets larger than 8 KB zlib-compressed.

    This file format is experimental and might be incompatible with future
    versions of this library. It is intended for temporarily exchanging
    phasor coordinates with other software, not as a long-term storage
    solution.

    Parameters
    ----------
    filename : str or Path
        Name of OME-TIFF file to write.
    mean : array_like
        Average intensity image. Write to image series named 'Phasor mean'.
    real : array_like
        Image of real component of phasor coordinates.
        Multiple harmonics, if any, must be in the first dimension.
        Write to image series named 'Phasor real'.
    imag : array_like
        Image of imaginary component of phasor coordinates.
        Multiple harmonics, if any, must be in the first dimension.
        Write to image series named 'Phasor imag'.
    frequency : float, optional
        Fundamental frequency of time-resolved phasor coordinates.
        Write to image series named 'Phasor frequency'.
    harmonic : int or sequence of int, optional
        Harmonics present in the first dimension of `real` and `imag`, if any.
        Write to image series named 'Phasor harmonic'.
        Only needed if harmonics are not starting at and increasing by one.
    axes : str, optional
        Character codes for `mean` image dimensions.
        By default, the last dimensions are assumed to be 'TZCYX'.
        If harmonics are present in `real` and `imag`, an "other" (``Q``)
        dimension is prepended to axes for those arrays.
        Refer to the OME-TIFF model for allowed axes and their order.
    dtype : dtype-like, optional
        Floating point data type used to store phasor coordinates.
        The default is ``float32``, which has 6 digits of precision
        and maximizes compatibility with other software.
    description : str, optional
        Plain-text description of dataset. Write as OME dataset description.
    **kwargs
        Additional arguments passed to :py:class:`tifffile.TiffWriter` and
        :py:meth:`tifffile.TiffWriter.write`.
        For example, ``compression=None`` writes image data uncompressed.

    See Also
    --------
    phasorpy.io.phasor_from_ometiff

    Notes
    -----
    Scalar or one-dimensional phasor coordinate arrays are written as images.

    The OME-TIFF format is specified in the
    `OME Data Model and File Formats Documentation
    <https://ome-model.readthedocs.io/>`_.

    The `6D, 7D and 8D storage
    <https://ome-model.readthedocs.io/en/latest/developers/6d-7d-and-8d-storage.html>`_
    extension is used to store multi-harmonic phasor coordinates.
    The modulo type for the first, harmonic dimension is "other".

    Examples
    --------
    >>> mean, real, imag = numpy.random.rand(3, 32, 32, 32)
    >>> phasor_to_ometiff(
    ...     '_phasorpy.ome.tif', mean, real, imag, axes='ZYX', frequency=80.0
    ... )

    """
    import tifffile

    from .version import __version__

    if dtype is None:
        dtype = numpy.float32
    dtype = numpy.dtype(dtype)
    if dtype.kind != 'f':
        raise ValueError(f'{dtype=} not a floating point type')

    mean = numpy.asarray(mean, dtype)
    real = numpy.asarray(real, dtype)
    imag = numpy.asarray(imag, dtype)
    datasize = mean.nbytes + real.nbytes + imag.nbytes

    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    if mean.shape != real.shape[-mean.ndim :]:
        raise ValueError(f'{mean.shape=} != {real.shape[-mean.ndim:]=}')
    has_harmonic_dim = real.ndim == mean.ndim + 1
    if mean.ndim == real.ndim or real.ndim == 0:
        nharmonic = 1
    else:
        nharmonic = real.shape[0]

    if mean.ndim < 2:
        # not an image
        mean = mean.reshape(1, -1)
        if has_harmonic_dim:
            real = real.reshape(real.shape[0], 1, -1)
            imag = imag.reshape(imag.shape[0], 1, -1)
        else:
            real = real.reshape(1, -1)
            imag = imag.reshape(1, -1)

    if harmonic is not None:
        harmonic, _ = parse_harmonic(harmonic)
        if len(harmonic) != nharmonic:
            raise ValueError('invalid harmonic')

    if frequency is not None:
        frequency_array = numpy.atleast_2d(frequency).astype(numpy.float64)
        if frequency_array.size > 1:
            raise ValueError('frequency must be scalar')

    if axes is None:
        axes = 'TZCYX'[-mean.ndim :]
    else:
        axes = ''.join(tuple(axes))  # accept dims tuple and str
    if len(axes) != mean.ndim:
        raise ValueError(f'{axes=} does not match {mean.ndim=}')
    axes_phasor = axes if mean.ndim == real.ndim else 'Q' + axes

    if 'photometric' not in kwargs:
        kwargs['photometric'] = 'minisblack'
    if 'compression' not in kwargs and datasize > 8192:
        kwargs['compression'] = 'zlib'
    if 'tile' not in kwargs and 'rowsperstrip' not in kwargs:
        if (
            axes.endswith('YX')
            and mean.shape[-1] > 1024
            and mean.shape[-2] > 1024
        ):
            kwargs['tile'] = (256, 256)

    mode = kwargs.pop('mode', None)
    bigtiff = kwargs.pop('bigtiff', None)
    if bigtiff is None:
        bigtiff = datasize > 2**31

    metadata = kwargs.pop('metadata', {})
    if 'Creator' not in metadata:
        metadata['Creator'] = f'PhasorPy {__version__}'

    dataset = metadata.pop('Dataset', {})
    if 'Name' not in dataset:
        dataset['Name'] = 'Phasor'
    if description:
        dataset['Description'] = description
    metadata['Dataset'] = dataset

    if has_harmonic_dim:
        metadata['TypeDescription'] = {'Q': 'Phasor harmonics'}

    with tifffile.TiffWriter(
        filename, bigtiff=bigtiff, mode=mode, ome=True
    ) as tif:
        metadata['Name'] = 'Phasor mean'
        metadata['axes'] = axes
        tif.write(mean, metadata=metadata, **kwargs)
        del metadata['Dataset']

        metadata['Name'] = 'Phasor real'
        metadata['axes'] = axes_phasor
        tif.write(real, metadata=metadata, **kwargs)

        metadata['Name'] = 'Phasor imag'
        tif.write(imag, metadata=metadata, **kwargs)

        if frequency is not None:
            tif.write(frequency_array, metadata={'Name': 'Phasor frequency'})

        if harmonic is not None:
            tif.write(
                numpy.atleast_2d(harmonic).astype(numpy.uint32),
                metadata={'Name': 'Phasor harmonic'},
            )


def phasor_from_ometiff(
    filename: str | PathLike[Any],
    /,
    *,
    harmonic: int | Sequence[int] | Literal['all'] | str | None = None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], dict[str, Any]]:
    """Return phasor images and metadata from OME-TIFF written by PhasorPy.

    Parameters
    ----------
    filename : str or Path
        Name of OME-TIFF file to read.
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
    real : ndarray
        Image of real component of phasor coordinates.
    imag : ndarray
        Image of imaginary component of phasor coordinates.
    attrs : dict
        Select metadata:

        - ``'axes'`` (str):
          Character codes for `mean` image dimensions.
        - ``'harmonic'`` (int or list of int):
          Harmonic(s) present in `real` and `imag`.
          If a scalar, `real` and `imag` are single harmonic and contain no
          harmonic axes.
          If a list, `real` and `imag` contain one or more harmonics in the
          first axis.
        - ``'frequency'`` (float, optional):
          Fundamental frequency of time-resolved phasor coordinates.
        - ``'description'`` (str, optional):
          OME dataset plain-text description.

    Raises
    ------
    tifffile.TiffFileError
        File is not a TIFF file.
    ValueError
        File is not an OME-TIFF containing phasor coordinates.
    IndexError
        Requested harmonic is not found in file.

    See Also
    --------
    phasorpy.io.phasor_to_ometiff

    Notes
    -----
    Scalar or one-dimensional phasor coordinates stored in the file are
    returned as two-dimensional images (three-dimensional if multiple
    harmonics are present).

    Examples
    --------
    >>> mean, real, imag = numpy.random.rand(3, 32, 32, 32)
    >>> phasor_to_ometiff(
    ...     '_phasorpy.ome.tif', mean, real, imag, axes='ZYX', frequency=80.0
    ... )
    >>> mean, real, imag, attrs = phasor_from_ometiff('_phasorpy.ome.tif')
    >>> mean
    array(...)
    >>> mean.dtype
    dtype('float32')
    >>> mean.shape
    (32, 32, 32)
    >>> attrs['axes']
    'ZYX'
    >>> attrs['frequency']
    80.0
    >>> attrs['harmonic']
    1

    """
    import tifffile

    name = os.path.basename(filename)

    with tifffile.TiffFile(filename) as tif:
        if (
            not tif.is_ome
            or len(tif.series) < 3
            or tif.series[0].name != 'Phasor mean'
            or tif.series[1].name != 'Phasor real'
            or tif.series[2].name != 'Phasor imag'
        ):
            raise ValueError(
                f'{name!r} is not an OME-TIFF containing phasor images'
            )

        attrs: dict[str, Any] = {'axes': tif.series[0].axes}

        # TODO: read coords from OME-XML
        ome_xml = tif.ome_metadata
        assert ome_xml is not None

        # TODO: parse OME-XML
        match = re.search(
            r'><Description>(.*)</Description><',
            ome_xml,
            re.MULTILINE | re.DOTALL,
        )
        if match is not None:
            attrs['description'] = (
                match.group(1)
                .replace('&amp;', '&')
                .replace('&gt;', '>')
                .replace('&lt;', '<')
            )

        has_harmonic_dim = tif.series[1].ndim > tif.series[0].ndim
        nharmonics = tif.series[1].shape[0] if has_harmonic_dim else 1
        harmonic_max = nharmonics
        for i in (3, 4):
            if len(tif.series) < i + 1:
                break
            series = tif.series[i]
            data = series.asarray().squeeze()
            if series.name == 'Phasor frequency':
                attrs['frequency'] = float(data.item(0))
            elif series.name == 'Phasor harmonic':
                if not has_harmonic_dim and data.size == 1:
                    attrs['harmonic'] = int(data.item(0))
                    harmonic_max = attrs['harmonic']
                elif has_harmonic_dim and data.size == nharmonics:
                    attrs['harmonic'] = data.tolist()
                    harmonic_max = max(attrs['harmonic'])
                else:
                    logger.warning(
                        f'harmonic={data} does not match phasor '
                        f'shape={tif.series[1].shape}'
                    )

        if 'harmonic' not in attrs:
            if has_harmonic_dim:
                attrs['harmonic'] = list(range(1, nharmonics + 1))
            else:
                attrs['harmonic'] = 1
        harmonic_stored = attrs['harmonic']

        mean = tif.series[0].asarray()
        if harmonic is None:
            # first harmonic in file
            if isinstance(harmonic_stored, list):
                attrs['harmonic'] = harmonic_stored[0]
            else:
                attrs['harmonic'] = harmonic_stored
            real = tif.series[1].asarray()
            if has_harmonic_dim:
                real = real[0].copy()
            imag = tif.series[2].asarray()
            if has_harmonic_dim:
                imag = imag[0].copy()
        elif isinstance(harmonic, str) and harmonic == 'all':
            # all harmonics as stored in file
            real = tif.series[1].asarray()
            imag = tif.series[2].asarray()
        else:
            # specified harmonics
            harmonic, keepdims = parse_harmonic(harmonic, harmonic_max)
            try:
                if isinstance(harmonic_stored, list):
                    index = [harmonic_stored.index(h) for h in harmonic]
                else:
                    index = [[harmonic_stored].index(h) for h in harmonic]
            except ValueError as exc:
                raise IndexError('harmonic not found') from exc

            if has_harmonic_dim:
                if keepdims:
                    attrs['harmonic'] = [harmonic_stored[i] for i in index]
                    real = tif.series[1].asarray()[index].copy()
                    imag = tif.series[2].asarray()[index].copy()
                else:
                    attrs['harmonic'] = harmonic_stored[index[0]]
                    real = tif.series[1].asarray()[index[0]].copy()
                    imag = tif.series[2].asarray()[index[0]].copy()
            elif keepdims:
                real = tif.series[1].asarray()
                real = real.reshape(1, *real.shape)
                imag = tif.series[2].asarray()
                imag = imag.reshape(1, *imag.shape)
                attrs['harmonic'] = [harmonic_stored]
            else:
                real = tif.series[1].asarray()
                imag = tif.series[2].asarray()

    if real.shape != imag.shape:
        logger.warning(f'{real.shape=} != {imag.shape=}')
    if real.shape[-mean.ndim :] != mean.shape:
        logger.warning(f'{real.shape[-mean.ndim:]=} != {mean.shape=}')

    return mean, real, imag, attrs


def phasor_to_simfcs_referenced(
    filename: str | PathLike[Any],
    mean: ArrayLike,
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    size: int | None = None,
    axes: str | None = None,
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
    axes : str, optional
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
        raise ValueError(f'{size=} out of range [4..65535]')

    harmonics_per_file = 2  # TODO: make this a parameter?
    chunk_shape = tuple(
        [max(harmonics_per_file, 2)] + ([1] * (phi.ndim - 3)) + [size, size]
    )
    multi_file = any(i / j > 1 for i, j in zip(phi.shape, chunk_shape))

    if axes is not None and len(axes) == phi.ndim - 1:
        axes = 'h' + axes

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
                chunk[sizey:, sizex:] = numpy.nan
                rawdata.append(chunk.tobytes())
            else:
                raise RuntimeError  # should not be reached

    for index, label, _ in chunk_iter(
        phi.shape, chunk_shape, axes, squeeze=False, use_index=True
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
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return phasor coordinate images from SimFCS referenced (REF, R64) file.

    SimFCS referenced REF and R64 files contain phasor coordinate images
    (encoded as phase and modulation) for two harmonics.
    Phasor coordinates from lifetime-resolved signals are calibrated.

    Parameters
    ----------
    filename : str or Path
        Name of REF or R64 file to read.
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

    Raises
    ------
    lfdfiles.LfdfileError
        File is not a SimFCS REF or R64 file.

    See Also
    --------
    phasorpy.io.phasor_to_simfcs_referenced

    Examples
    --------
    >>> phasor_to_simfcs_referenced(
    ...     '_phasorpy.r64', *numpy.random.rand(3, 32, 32)
    ... )
    >>> mean, real, imag = phasor_from_simfcs_referenced('_phasorpy.r64')
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

    return mean, real, imag


def read_lsm(
    filename: str | PathLike[Any],
    /,
) -> DataArray:
    """Return hyperspectral image and metadata from Zeiss LSM file.

    LSM files contain multi-dimensional images and metadata from laser
    scanning microscopy measurements. The file format is based on TIFF.

    Parameters
    ----------
    filename : str or Path
        Name of OME-TIFF file to read.

    Returns
    -------
    xarray.DataArray
        Hyperspectral image data.
        Usually, a 3-to-5-dimensional array of type ``uint8`` or ``uint16``.

    Raises
    ------
    tifffile.TiffFileError
        File is not a TIFF file.
    ValueError
        File is not an LSM file or does not contain hyperspectral image.

    Examples
    --------
    >>> data = read_lsm(fetch('paramecium.lsm'))
    >>> data.values
    array(...)
    >>> data.dtype
    dtype('uint8')
    >>> data.shape
    (30, 512, 512)
    >>> data.dims
    ('C', 'Y', 'X')
    >>> data.coords['C'].data  # wavelengths
    array(...)

    """
    import tifffile

    with tifffile.TiffFile(filename) as tif:
        if not tif.is_lsm:
            raise ValueError(f'{tif.filename} is not an LSM file')

        page = tif.pages.first
        lsminfo = tif.lsm_metadata
        channels = page.tags[258].count

        if channels < 4 or lsminfo is None or lsminfo['SpectralScan'] != 1:
            raise ValueError(
                f'{tif.filename} does not contain hyperspectral image'
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
                f'{tif.filename} wavelengths do not match channel axis'
            )
        # stack may contain non-wavelength frame
        indices = wavelengths > 0
        wavelengths = wavelengths[indices]
        if wavelengths.size < 3:
            raise ValueError(
                f'{tif.filename} does not contain hyperspectral image'
            )
        data = data.take(indices.nonzero()[0], axis=axis)
        coords['C'] = wavelengths
        # time stamps
        if 'T' in dims:
            coords['T'] = lsminfo['TimeStamps']
            if coords['T'].size != data.shape[dims.index('T')]:
                raise ValueError(
                    f'{tif.filename} timestamps do not match time axis'
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
        metadata = _metadata(series.axes, data.shape, filename, **coords)

    from xarray import DataArray

    return DataArray(data, **metadata)


def read_imspector_tiff(
    filename: str | PathLike[Any],
    /,
) -> DataArray:
    """Return FLIM image stack and metadata from ImSpector TIFF file.

    Parameters
    ----------
    filename : str or Path
        Name of ImSpector FLIM TIFF file to read.

    Returns
    -------
    xarray.DataArray
        TCSPC image stack.
        Usually, a 3-to-5-dimensional array of type ``uint16``.

        - ``coords['H']``: times of histogram bins.
        - ``attrs['frequency']``: repetition frequency in MHz.

    Raises
    ------
    tifffile.TiffFileError
        File is not a TIFF file.
    ValueError
        File is not an ImSpector FLIM TIFF file.

    Examples
    --------
    >>> data = read_imspector_tiff(fetch('Embryo.tif'))
    >>> data.values
    array(...)
    >>> data.dtype
    dtype('uint16')
    >>> data.shape
    (56, 512, 512)
    >>> data.dims
    ('H', 'Y', 'X')
    >>> data.coords['H'].data  # dtime bins
    array(...)
    >>> data.attrs['frequency']  # doctest: +NUMBER
    80.109

    """
    from xml.etree import ElementTree

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
            raise ValueError(f'{tif.filename} is not an ImSpector TIFF file')

        series = tif.series[0]
        ndim = series.ndim
        axes = series.axes
        shape = series.shape

        if ndim < 3 or not axes.endswith('YX'):
            raise ValueError(
                f'{tif.filename} is not an ImSpector FLIM TIFF file'
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
        raise ValueError(f'{tif.filename} is not an ImSpector FLIM TIFF file')

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
        raise ValueError(f'{tif.filename} is not an ImSpector FLIM TIFF file')
    axes = axes.replace(ax, 'H')
    coords['H'] = coords[ax]
    del coords[ax]

    attrs['frequency'] = float(1000.0 / physical_size[ax])

    metadata = _metadata(axes, shape, filename, attrs=attrs, **coords)

    from xarray import DataArray

    return DataArray(data, **metadata)


def read_flimlabs_json(
    filename: str | PathLike[Any],
    /,
    *,
    channel: int | None = None,
    dtype: DTypeLike | None = None,
) -> DataArray:
    """Return FLIM image stack and metadata from FLIMLABS JSON image file.

    Parameters
    ----------
    filename : str or Path
        Name of FLIMLABS JSON image file to read.
    channel : int, optional
        If None (default), return all channels, else return specified channel.
    dtype : dtype-like, optional, default: uint16
        Unsigned integer type of image histogram array.
        Increase the bit-depth for high photon counts.

    Returns
    -------
    xarray.DataArray
        TCSPC image stack.
        A 3 or 4-dimensional array of type `dtype`.

        - ``coords['H']``: times of histogram bins in ns.
        - ``attrs['frequency']``: laser repetition frequency in MHz.

    Raises
    ------
    ValueError
        File is not a FLIMLABS JSON image file.
        `dtype` is not an unsigned integer.

    Examples
    --------
    >>> data = read_flimlabs_json(
    ...     fetch('test03_1733492714_imaging.json')
    ... )  # doctest: +SKIP
    >>> data.values  # doctest: +SKIP
    array(...)
    >>> data.shape  # doctest: +SKIP
    (3, 256, 256, 256)
    >>> data.dims  # doctest: +SKIP
    ('C', 'Y', 'X', 'H')
    >>> data.coords['H'].data  # doctest: +SKIP
    array(...)
    >>> data.attrs['frequency']  # doctest: +SKIP
    80.0

    """
    import json

    with open(filename) as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError('not a FLIMLABS JSON file') from exc

    if (
        'data' not in data
        or 'header' not in data
        or 'laser_period_ns' not in data['header']
    ):
        raise ValueError('not a FLIMLABS JSON file')

    if dtype is None:
        dtype = numpy.uint16
    else:
        dtype = numpy.dtype(dtype)
        if dtype.kind != 'u':
            raise ValueError(f'{dtype=} is not an unsigned integer type')

    header = data['header']
    channels = len([c for c in header['channels'] if c])
    # TODO: how to use header['frames']?
    height = header['image_height']
    width = header['image_width']
    frequency = 1000.0 / header['laser_period_ns']

    if channel is None:
        histogram = numpy.zeros((channels, height * width, 256), dtype)
        axes = 'CYXH'
    else:
        histogram = numpy.zeros((height * width, 256), dtype)
        axes = 'YXH'

    if channel is None:
        for c, channel_ in enumerate(data['data']):
            for i, pixel in enumerate(channel_):
                hist = histogram[c, i]
                for index, count in pixel:
                    hist[index] = count
    else:
        for i, pixel in enumerate(data['data'][channel]):
            hist = histogram[i]
            for index, count in pixel:
                hist[index] = count

    if channel is None:
        histogram.shape = (channels, height, width, 256)
    else:
        histogram.shape = (height, width, 256)

    coords: dict[str, Any] = {}
    coords['H'] = numpy.linspace(
        0.0, header['laser_period_ns'], 256, endpoint=False
    )
    if channel is None:
        coords['C'] = numpy.asarray(
            [i for i, c in enumerate(header['channels']) if c]
        )

    metadata = _metadata(axes, histogram.shape, filename, **coords)
    attrs = metadata['attrs']
    attrs['frequency'] = frequency

    from xarray import DataArray

    return DataArray(histogram, **metadata)


def read_ifli(
    filename: str | PathLike[Any],
    /,
    *,
    channel: int = 0,
    **kwargs: Any,
) -> DataArray:
    """Return image and metadata from ISS IFLI file.

    ISS VistaVision IFLI files contain phasor coordinates for several
    positions, wavelengths, time points, channels, slices, and frequencies
    from analog or digital frequency-domain fluorescence lifetime measurements.

    Parameters
    ----------
    filename : str or Path
        Name of ISS IFLI file to read.
    channel : int, optional
        Index of channel to return. The first channel is returned by default.
    **kwargs
        Additional arguments passed to :py:meth:`lfdfiles.VistaIfli.asarray`,
        for example ``memmap=True``.

    Returns
    -------
    xarray.DataArray
        Average intensity and phasor coordinates.
        An array of up to 8 dimensions with :ref:`axes codes <axes>`
        ``'RCTZYXFS'`` and type ``float32``.
        The last dimension contains `mean`, `real`, and `imag` phasor
        coordinates.

        - ``coords['F']``: modulation frequencies.
        - ``coords['C']``: emission wavelengths, if any.
        - ``attrs['ref_tau']``: reference lifetimes.
        - ``attrs['ref_tau_frac']``: reference lifetime fractions.
        - ``attrs['ref_phasor']``: reference phasor coordinates for all
          frequencies.

    Raises
    ------
    lfdfiles.LfdFileError
        File is not an ISS IFLI file.

    Examples
    --------
    >>> data = read_ifli(fetch('frequency_domain.ifli'))
    >>> data.values
    array(...)
    >>> data.dtype
    dtype('float32')
    >>> data.shape
    (256, 256, 4, 3)
    >>> data.dims
    ('Y', 'X', 'F', 'S')
    >>> data.coords['F'].data  # doctest: +NUMBER
    array([8.033e+07, 1.607e+08, 2.41e+08, 4.017e+08])
    >>> data.coords['S'].data
    array(['mean', 'real', 'imag'], dtype='<U4')
    >>> data.attrs
    {'ref_tau': (2.5, 0.0), 'ref_tau_frac': (1.0, 0.0), 'ref_phasor': array...}

    """
    import lfdfiles

    with lfdfiles.VistaIfli(filename) as ifli:
        assert ifli.axes is not None
        # always return one acquisition channel to simplify metadata handling
        data = ifli.asarray(**kwargs)[:, channel : channel + 1].copy()
        shape, axes, _ = _squeeze_axes(data.shape, ifli.axes, skip='FYX')
        axes = axes.replace('E', 'C')  # spectral axis
        data = data.reshape(shape)
        header = ifli.header
        coords: dict[str, Any] = {}
        coords['S'] = ['mean', 'real', 'imag']
        coords['F'] = numpy.array(header['ModFrequency'])
        # TODO: how to distinguish time- from frequency-domain?
        # TODO: how to extract spatial coordinates?
        if 'T' in axes:
            coords['T'] = numpy.array(header['TimeTags'])
        if 'C' in axes:
            coords['C'] = numpy.array(header['SpectrumInfo'])
        # if 'Z' in axes:
        #     coords['Z'] = numpy.array(header[])
        metadata = _metadata(axes, shape, filename, **coords)
        attrs = metadata['attrs']
        attrs['ref_tau'] = (
            header['RefLifetime'][channel],
            header['RefLifetime2'][channel],
        )
        attrs['ref_tau_frac'] = (
            header['RefLifetimeFrac'][channel],
            1.0 - header['RefLifetimeFrac'][channel],
        )
        attrs['ref_phasor'] = numpy.array(header['RefDCPhasor'][channel])

    from xarray import DataArray

    return DataArray(data, **metadata)


def read_sdt(
    filename: str | PathLike[Any],
    /,
    *,
    index: int = 0,
) -> DataArray:
    """Return time-resolved image and metadata from Becker & Hickl SDT file.

    SDT files contain time-correlated single photon counting measurement data
    and instrumentation parameters.

    Parameters
    ----------
    filename : str or Path
        Name of SDT file to read.
    index : int, optional, default: 0
        Index of dataset to read in case the file contains multiple datasets.

    Returns
    -------
    xarray.DataArray
        Time correlated single photon counting image data with
        :ref:`axes codes <axes>` ``'YXH'`` and type ``uint16``, ``uint32``,
        or ``float32``.

        - ``coords['H']``: times of the histogram bins.
        - ``attrs['frequency']``: repetition frequency in MHz.

    Raises
    ------
    ValueError
        File is not an SDT file containing time-correlated single photon
        counting data.

    Examples
    --------
    >>> data = read_sdt(fetch('tcspc.sdt'))
    >>> data.values
    array(...)
    >>> data.dtype
    dtype('uint16')
    >>> data.shape
    (128, 128, 256)
    >>> data.dims
    ('Y', 'X', 'H')
    >>> data.coords['H'].data
    array(...)
    >>> data.attrs['frequency']  # doctest: +NUMBER
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
        times = sdt.times[index]

    # TODO: get spatial coordinates from scanner settings?
    metadata = _metadata('QYXH'[-data.ndim :], data.shape, filename, H=times)
    metadata['attrs']['frequency'] = 1e-6 / float(times[-1] + times[1])

    from xarray import DataArray

    return DataArray(data, **metadata)


def read_ptu(
    filename: str | PathLike[Any],
    /,
    selection: Sequence[int | slice | EllipsisType | None] | None = None,
    *,
    trimdims: Sequence[Literal['T', 'C', 'H']] | str | None = None,
    dtype: DTypeLike | None = None,
    frame: int | None = None,
    channel: int | None = None,
    dtime: int | None = 0,
    keepdims: bool = True,
) -> DataArray:
    """Return image histogram and metadata from PicoQuant PTU T3 mode file.

    PTU files contain time-correlated single photon counting measurement data
    and instrumentation parameters.

    Parameters
    ----------
    filename : str or Path
        Name of PTU file to read.
    selection : sequence of index types, optional
        Indices for all dimensions:

        - ``None``: return all items along axis (default).
        - ``Ellipsis``: return all items along multiple axes.
        - ``int``: return single item along axis.
        - ``slice``: return chunk of axis.
          ``slice.step`` is binning factor.
          If ``slice.step=-1``, integrate all items along axis.

    trimdims : str, optional, default: 'TCH'
        Axes to trim.
    dtype : dtype-like, optional, default: uint16
        Unsigned integer type of image histogram array.
        Increase the bit depth to avoid overflows when integrating.
    frame : int, optional
        If < 0, integrate time axis, else return specified frame.
        Overrides `selection` for axis ``T``.
    channel : int, optional
        If < 0, integrate channel axis, else return specified channel.
        Overrides `selection` for axis ``C``.
    dtime : int, optional, default: 0
        Specifies number of bins in image histogram.
        If 0 (default), return number of bins in one period.
        If < 0, integrate delay time axis.
        If > 0, return up to specified bin.
        Overrides `selection` for axis ``H``.
    keepdims : bool, optional, default: True
        If true (default), reduced axes are left as size-one dimension.

    Returns
    -------
    xarray.DataArray
        Decoded TTTR T3 records as up to 5-dimensional image array
        with :ref:`axes codes <axes>` ``'TYXCH'`` and type specified
        in ``dtype``:

        - ``coords['H']``: times of the histogram bins.
        - ``attrs['frequency']``: repetition frequency in MHz.

    Raises
    ------
    ptufile.PqFileError
        File is not a PicoQuant PTU file or is corrupted.
    ValueError
        File is not a PicoQuant PTU T3 mode file containing time-correlated
        single photon counting data.

    Examples
    --------
    >>> data = read_ptu(fetch('hazelnut_FLIM_single_image.ptu'))
    >>> data.values
    array(...)
    >>> data.dtype
    dtype('uint16')
    >>> data.shape
    (5, 256, 256, 1, 132)
    >>> data.dims
    ('T', 'Y', 'X', 'C', 'H')
    >>> data.coords['H'].data
    array(...)
    >>> data.attrs['frequency']  # doctest: +NUMBER
    78.02

    """
    import ptufile
    from xarray import DataArray

    with ptufile.PtuFile(filename, trimdims=trimdims) as ptu:
        if not ptu.is_t3 or not ptu.is_image:
            raise ValueError(
                f'{os.path.basename(filename)!r} '
                'is not a PTU file containing a T3 mode image'
            )
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
        data.attrs['frequency'] = ptu.frequency * 1e-6  # MHz

    return data


def read_flif(
    filename: str | PathLike[Any],
    /,
) -> DataArray:
    """Return frequency-domain image and metadata from FlimFast FLIF file.

    FlimFast FLIF files contain camera images and metadata from
    frequency-domain fluorescence lifetime measurements.

    Parameters
    ----------
    filename : str or Path
        Name of FlimFast FLIF file to read.

    Returns
    -------
    xarray.DataArray
        Frequency-domain phase images with :ref:`axes codes <axes>` ``'THYX'``
        and type ``uint16``:

        - ``coords['H']``: phases in radians.
        - ``attrs['frequency']``: repetition frequency in MHz.
        - ``attrs['ref_phase']``: measured phase of reference.
        - ``attrs['ref_mod']``: measured modulation of reference.
        - ``attrs['ref_tauphase']``: lifetime from phase of reference.
        - ``attrs['ref_taumod']``: lifetime from modulation of reference.

    Raises
    ------
    lfdfiles.LfdFileError
        File is not a FlimFast FLIF file.

    Examples
    --------
    >>> data = read_flif(fetch('flimfast.flif'))
    >>> data.values
    array(...)
    >>> data.dtype
    dtype('uint16')
    >>> data.shape
    (32, 220, 300)
    >>> data.dims
    ('H', 'Y', 'X')
    >>> data.coords['H'].data
    array(...)
    >>> data.attrs['frequency']  # doctest: +NUMBER
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
        metadata = _metadata(axes, data.shape, H=phases)
        attrs = metadata['attrs']
        attrs['frequency'] = float(flif.header.frequency)
        attrs['ref_phase'] = float(flif.header.measured_phase)
        attrs['ref_mod'] = float(flif.header.measured_mod)
        attrs['ref_tauphase'] = float(flif.header.ref_tauphase)
        attrs['ref_taumod'] = float(flif.header.ref_taumod)

    from xarray import DataArray

    return DataArray(data, **metadata)


def read_fbd(
    filename: str | PathLike[Any],
    /,
    *,
    frame: int | None = None,
    channel: int | None = None,
    keepdims: bool = True,
    laser_factor: float = -1.0,
) -> DataArray:
    """Return frequency-domain image and metadata from FLIMbox FBD file.

    FDB files contain encoded data from the FLIMbox device, which can be
    decoded to photon arrival windows, channels, and global times.
    The encoding scheme depends on the FLIMbox device's firmware.
    The FBD file format is undocumented.

    This function may fail to produce expected results when files use unknown
    firmware, do not contain image scans, settings were recorded incorrectly,
    scanner and FLIMbox frequencies were out of sync, or scanner settings were
    changed during acquisition.

    Parameters
    ----------
    filename : str or Path
        Name of FLIMbox FBD file to read.
    frame : int, optional
        If None (default), return all frames.
        If < 0, integrate time axis, else return specified frame.
    channel : int, optional
        If None (default), return all channels, else return specified channel.
    keepdims : bool, optional
        If true (default), reduced axes are left as size-one dimension.
    laser_factor : float, optional
        Factor to correct dwell_time/laser_frequency.

    Returns
    -------
    xarray.DataArray
        Frequency-domain image histogram with :ref:`axes codes <axes>`
        ``'TCYXH'`` and type ``uint16``:

        - ``coords['H']``: phases in radians.
        - ``attrs['frequency']``: repetition frequency in MHz.

    Raises
    ------
    lfdfiles.LfdFileError
        File is not a FLIMbox FBD file.

    Examples
    --------
    >>> data = read_fbd(fetch('convallaria_000$EI0S.fbd'))  # doctest: +SKIP
    >>> data.values  # doctest: +SKIP
    array(...)
    >>> data.dtype  # doctest: +SKIP
    dtype('uint16')
    >>> data.shape  # doctest: +SKIP
    (9, 2, 256, 256, 64)
    >>> data.dims  # doctest: +SKIP
    ('T', 'C', 'Y', 'X', 'H')
    >>> data.coords['H'].data  # doctest: +SKIP
    array(...)
    >>> data.attrs['frequency']  # doctest: +SKIP
    40.0

    """
    import lfdfiles

    integrate_frames = 0 if frame is None or frame >= 0 else 1

    with lfdfiles.FlimboxFbd(filename, laser_factor=laser_factor) as fbd:
        data = fbd.asimage(None, None, integrate_frames=integrate_frames)
        if integrate_frames:
            frame = None
        copy = False
        axes = 'TCYXH'
        if channel is None:
            if not keepdims and data.shape[1] == 1:
                data = data[:, 0]
                axes = 'TYXH'
        else:
            if channel < 0 or channel >= data.shape[1]:
                raise IndexError(f'{channel=} out of bounds')
            if keepdims:
                data = data[:, channel : channel + 1]
            else:
                data = data[:, channel]
                axes = 'TYXH'
            copy = True
        if frame is None:
            if not keepdims and data.shape[0] == 1:
                data = data[0]
                axes = axes[1:]
        else:
            if frame < 0 or frame > data.shape[0]:
                raise IndexError(f'{frame=} out of bounds')
            if keepdims:
                data = data[frame : frame + 1]
            else:
                data = data[frame]
                axes = axes[1:]
            copy = True
        if copy:
            data = data.copy()
        # TODO: return arrival window indices or micro-times as H coords?
        phases = numpy.linspace(
            0.0, numpy.pi * 2, data.shape[-1], endpoint=False
        )
        metadata = _metadata(axes, data.shape, H=phases)
        attrs = metadata['attrs']
        attrs['frequency'] = fbd.laser_frequency * 1e-6

    from xarray import DataArray

    return DataArray(data, **metadata)


def read_b64(
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
        Stack of square-sized intensity images of type ``int16``.

    Raises
    ------
    lfdfiles.LfdFileError
        File is not a SimFCS B64 file.
    ValueError
        File does not contain an image stack.

    Examples
    --------
    >>> data = read_b64(fetch('simfcs.b64'))
    >>> data.values
    array(...)
    >>> data.dtype
    dtype('int16')
    >>> data.shape
    (22, 1024, 1024)
    >>> data.dtype
    dtype('int16')
    >>> data.dims
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
        metadata = _metadata(b64.axes, data.shape, filename)

    from xarray import DataArray

    return DataArray(data, **metadata)


def read_z64(
    filename: str | PathLike[Any],
    /,
) -> DataArray:
    """Return image and metadata from SimFCS Z64 file.

    Z64 files contain stacks of square images such as intensity volumes
    or time-domain fluorescence lifetime histograms acquired from
    Becker & Hickl(r) TCSPC cards. Z64 files contain no metadata.

    Parameters
    ----------
    filename : str or Path
        Name of SimFCS Z64 file to read.

    Returns
    -------
    xarray.DataArray
        Single or stack of square-sized images of type ``float32``.

    Raises
    ------
    lfdfiles.LfdFileError
        File is not a SimFCS Z64 file.

    Examples
    --------
    >>> data = read_z64(fetch('simfcs.z64'))
    >>> data.values
    array(...)
    >>> data.dtype
    dtype('float32')
    >>> data.shape
    (256, 256, 256)
    >>> data.dims
    ('Q', 'Y', 'X')

    """
    import lfdfiles

    with lfdfiles.SimfcsZ64(filename) as z64:
        data = z64.asarray()
        metadata = _metadata(z64.axes, data.shape, filename)

    from xarray import DataArray

    return DataArray(data, **metadata)


def read_bh(
    filename: str | PathLike[Any],
    /,
) -> DataArray:
    """Return image and metadata from SimFCS B&H file.

    B&H files contain time-domain fluorescence lifetime histogram data,
    acquired from Becker & Hickl(r) TCSPC cards, or converted from other
    data sources. B&H files contain no metadata.

    Parameters
    ----------
    filename : str or Path
        Name of SimFCS B&H file to read.

    Returns
    -------
    xarray.DataArray
        Time-domain fluorescence lifetime histogram with axes ``'HYX'``,
        shape ``(256, 256, 256)``, and type ``float32``.

    Raises
    ------
    lfdfiles.LfdFileError
        File is not a SimFCS B&H file.

    Examples
    --------
    >>> data = read_bh(fetch('simfcs.b&h'))
    >>> data.values
    array(...)
    >>> data.dtype
    dtype('float32')
    >>> data.shape
    (256, 256, 256)
    >>> data.dims
    ('H', 'Y', 'X')

    """
    import lfdfiles

    with lfdfiles.SimfcsBh(filename) as bnh:
        assert bnh.axes is not None
        data = bnh.asarray()
        metadata = _metadata(bnh.axes.replace('Q', 'H'), data.shape, filename)

    from xarray import DataArray

    return DataArray(data, **metadata)


def read_bhz(
    filename: str | PathLike[Any],
    /,
) -> DataArray:
    """Return image and metadata from SimFCS BHZ file.

    BHZ files contain time-domain fluorescence lifetime histogram data,
    acquired from Becker & Hickl(r) TCSPC cards, or converted from other
    data sources. BHZ files contain no metadata.

    Parameters
    ----------
    filename : str or Path
        Name of SimFCS BHZ file to read.

    Returns
    -------
    xarray.DataArray
        Time-domain fluorescence lifetime histogram with axes ``'HYX'``,
        shape ``(256, 256, 256)``, and type ``float32``.

    Raises
    ------
    lfdfiles.LfdFileError
        File is not a SimFCS BHZ file.

    Examples
    --------
    >>> data = read_bhz(fetch('simfcs.bhz'))
    >>> data.values
    array(...)
    >>> data.dtype
    dtype('float32')
    >>> data.shape
    (256, 256, 256)
    >>> data.dims
    ('H', 'Y', 'X')

    """
    import lfdfiles

    with lfdfiles.SimfcsBhz(filename) as bhz:
        assert bhz.axes is not None
        data = bhz.asarray()
        metadata = _metadata(bhz.axes.replace('Q', 'H'), data.shape, filename)

    from xarray import DataArray

    return DataArray(data, **metadata)


def _metadata(
    dims: Sequence[str] | None,
    shape: tuple[int, ...],
    /,
    name: str | PathLike[Any] | None = None,
    attrs: dict[str, Any] | None = None,
    **coords: Any,
) -> dict[str, Any]:
    """Return xarray-style dims, coords, and attrs in a dict.

    >>> _metadata('SYX', (3, 2, 1), S=['0', '1', '2'])
    {'dims': ('S', 'Y', 'X'), 'coords': {'S': ['0', '1', '2']}, 'attrs': {}}

    """
    assert dims is not None
    dims = tuple(dims)
    if len(dims) != len(shape):
        raise ValueError(
            f'dims do not match shape {len(dims)} != {len(shape)}'
        )
    coords = {dim: coords[dim] for dim in dims if dim in coords}
    if attrs is None:
        attrs = {}
    metadata = {'dims': dims, 'coords': coords, 'attrs': attrs}
    if name:
        metadata['name'] = os.path.basename(name)
    return metadata


def _squeeze_axes(
    shape: Sequence[int],
    axes: str,
    /,
    skip: str = 'XY',
) -> tuple[tuple[int, ...], str, tuple[bool, ...]]:
    """Return shape and axes with length-1 dimensions removed.

    Remove unused dimensions unless their axes are listed in `skip`.

    Adapted from the tifffile library.

    Parameters
    ----------
    shape : tuple of ints
        Sequence of dimension sizes.
    axes : str
        Character codes for dimensions in `shape`.
    skip : str, optional
        Character codes for dimensions whose length-1 dimensions are
        not removed. The default is 'XY'.

    Returns
    -------
    shape : tuple of ints
        Sequence of dimension sizes with length-1 dimensions removed.
    axes : str
        Character codes for dimensions in output `shape`.
    squeezed : str
        Dimensions were kept (True) or removed (False).

    Examples
    --------
    >>> _squeeze_axes((5, 1, 2, 1, 1), 'TZYXC')
    ((5, 2, 1), 'TYX', (True, False, True, True, False))
    >>> _squeeze_axes((1,), 'Q')
    ((1,), 'Q', (True,))

    """
    if len(shape) != len(axes):
        raise ValueError(f'{len(shape)=} != {len(axes)=}')
    if not axes:
        return tuple(shape), axes, ()
    squeezed: list[bool] = []
    shape_squeezed: list[int] = []
    axes_squeezed: list[str] = []
    for size, ax in zip(shape, axes):
        if size > 1 or ax in skip:
            squeezed.append(True)
            shape_squeezed.append(size)
            axes_squeezed.append(ax)
        else:
            squeezed.append(False)
    if len(shape_squeezed) == 0:
        squeezed[-1] = True
        shape_squeezed.append(shape[-1])
        axes_squeezed.append(axes[-1])
    return tuple(shape_squeezed), ''.join(axes_squeezed), tuple(squeezed)
