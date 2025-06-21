"""Read and write OME-TIFF file format."""

from __future__ import annotations

__all__ = [
    # 'signal_from_ometiff',
    'phasor_from_ometiff',
    'phasor_to_ometiff',
]

import os
import re
from typing import TYPE_CHECKING

from .._utils import parse_harmonic
from ..utils import logger

if TYPE_CHECKING:
    from .._typing import (
        Any,
        ArrayLike,
        DTypeLike,
        Literal,
        NDArray,
        PathLike,
        Sequence,
    )

import numpy

from .. import __version__


def phasor_to_ometiff(
    filename: str | PathLike[Any],
    mean: ArrayLike,
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    frequency: float | None = None,
    harmonic: int | Sequence[int] | None = None,
    dims: Sequence[str] | None = None,
    dtype: DTypeLike | None = None,
    description: str | None = None,
    **kwargs: Any,
) -> None:
    """Write phasor coordinate images and metadata to OME-TIFF file.

    The OME-TIFF format is compatible with Bio-Formats and Fiji.

    By default, write phasor coordinates as single precision floating point
    values to separate image series.
    Write images larger than (1024, 1024) pixels as (256, 256) tiles, datasets
    larger than 2 GB as BigTIFF, and datasets larger than 8 KB using
    zlib compression.

    This file format is experimental and might be incompatible with future
    versions of this library. It is intended for temporarily exchanging
    phasor coordinates with other software, not as a long-term storage
    solution.

    Parameters
    ----------
    filename : str or Path
        Name of PhasorPy OME-TIFF file to write.
    mean : array_like
        Average intensity image. Write to an image series named 'Phasor mean'.
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
        Usually in units of MHz.
        Write to image series named 'Phasor frequency'.
    harmonic : int or sequence of int, optional
        Harmonics present in the first dimension of `real` and `imag`, if any.
        Write to image series named 'Phasor harmonic'.
        It is only needed if harmonics are not starting at and increasing by
        one.
    dims : sequence of str, optional
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

    The implementation is based on the
    `tifffile <https://github.com/cgohlke/tifffile/>`__ library.

    Examples
    --------
    >>> mean, real, imag = numpy.random.rand(3, 32, 32, 32)
    >>> phasor_to_ometiff(
    ...     '_phasorpy.ome.tif', mean, real, imag, dims='ZYX', frequency=80.0
    ... )

    """
    import tifffile

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

    axes = 'TZCYX'[-mean.ndim :] if dims is None else ''.join(tuple(dims))
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
    """Return phasor coordinates and metadata from PhasorPy OME-TIFF file.

    PhasorPy OME-TIFF files contain phasor mean intensity, real and imaginary
    components, along with frequency and harmonic information.

    Parameters
    ----------
    filename : str or Path
        Name of PhasorPy OME-TIFF file to read.
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

        - ``'dims'`` (tuple of str):
          :ref:`Axes codes <axes>` for `mean` image dimensions.
        - ``'harmonic'`` (int or list of int):
          Harmonic(s) present in `real` and `imag`.
          If a scalar, `real` and `imag` are single harmonic and contain no
          harmonic axes.
          If a list, `real` and `imag` contain one or more harmonics in the
          first axis.
        - ``'frequency'`` (float, optional):
          Fundamental frequency of time-resolved phasor coordinates.
          Usually in units of MHz.
        - ``'description'`` (str, optional):
          OME dataset plain-text description.

    Raises
    ------
    tifffile.TiffFileError
        File is not a TIFF file.
    ValueError
        File is not an OME-TIFF containing phasor coordinates.
    IndexError
        Harmonic is not found in file.

    See Also
    --------
    phasorpy.io.phasor_to_ometiff

    Notes
    -----
    Scalar or one-dimensional phasor coordinates stored in the file are
    returned as two-dimensional images (three-dimensional if multiple
    harmonics are present).

    The implementation is based on the
    `tifffile <https://github.com/cgohlke/tifffile/>`__ library.

    Examples
    --------
    >>> mean, real, imag = numpy.random.rand(3, 32, 32, 32)
    >>> phasor_to_ometiff(
    ...     '_phasorpy.ome.tif', mean, real, imag, dims='ZYX', frequency=80.0
    ... )
    >>> mean, real, imag, attrs = phasor_from_ometiff('_phasorpy.ome.tif')
    >>> mean
    array(...)
    >>> mean.dtype
    dtype('float32')
    >>> mean.shape
    (32, 32, 32)
    >>> attrs['dims']
    ('Z', 'Y', 'X')
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

        attrs: dict[str, Any] = {'dims': tuple(tif.series[0].axes)}

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
                    logger().warning(
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
        logger().warning(f'{real.shape=} != {imag.shape=}')
    if real.shape[-mean.ndim :] != mean.shape:
        logger().warning(f'{real.shape[-mean.ndim:]=} != {mean.shape=}')

    return mean, real, imag, attrs
