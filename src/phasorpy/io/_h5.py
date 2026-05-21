"""Read HDF5 histogram image files."""

from __future__ import annotations

__all__ = ['phasor_from_h5', 'signal_from_h5']

import operator
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

_DATASET = 'data'
_REFERENCE_DATASET = 'calibration/data/ref_common_delay_realigned'
_IRF_DATASET = 'calibration/data/irf_common_delay_realigned'
_H5_AXES = ('R', 'Z', 'Y', 'X', 'H', 'C')


def _normalize_h5_value(value: Any) -> Any:
    """Return HDF5 values converted to Python or NumPy-native values."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, numpy.ndarray):
        if value.ndim == 0:
            return _normalize_h5_value(value[()])
        if value.dtype.kind in 'SU':
            return [str(item) for item in value.tolist()]
        return value.tolist()
    if isinstance(value, numpy.generic):
        return value.item()
    return value


def _read_h5_attrs(obj: Any) -> dict[str, Any]:
    """Return HDF5 object attributes as plain Python values."""
    return {
        str(key): _normalize_h5_value(value)
        for key, value in obj.attrs.items()
    }


def _first_attr(
    *attrs: dict[str, Any],
    names: Sequence[str],
) -> Any | None:
    """Return first matching attribute from dictionaries."""
    for attr in attrs:
        for name in names:
            if name in attr:
                return attr[name]
    return None


def _as_index(index: int, size: int, name: str) -> int:
    """Return normalized index or raise IndexError."""
    try:
        index = operator.index(index)
    except TypeError as exc:
        msg = f'{name} must be an integer'
        raise TypeError(msg) from exc

    if index < 0:
        index += size
    if index < 0 or index >= size:
        msg = f'{name}={index!r} is out of bounds [0, {size - 1}]'
        raise IndexError(msg)
    return index


def _histogram_coordinates(data: Any, size: int) -> NDArray[Any]:
    """Return histogram coordinates from dataset attributes, if present."""
    for name in ('H', 'h', 'T', 't', 'time', 'times', 'histogram_bins'):
        if name not in data.attrs:
            continue
        coord = numpy.asarray(_normalize_h5_value(data.attrs[name]))
        if coord.ndim == 1 and coord.size == size:
            return coord
    return numpy.arange(size)


def signal_from_h5(
    filename: str | PathLike[Any],
    /,
    *,
    data_dataset: str = _DATASET,
    reference_dataset: str = _REFERENCE_DATASET,
    irf_dataset: str = _IRF_DATASET,
    reference: bool = False,
    irf: bool = False,
    repetition: int = 0,
    z: int = 0,
    channel: int = 0,
    dtype: DTypeLike | None = None,
) -> DataArray:
    """Return histogram image from HDF5 ``/data`` dataset.

    The supported HDF5 layout stores raw histogram images in a dataset named
    ``data`` by default. The dataset must have six dimensions in the order
    ``(repetition, z, y, x, time, channel)``. This function selects one
    repetition, one z plane, and one channel, and returns a three-dimensional
    ``xarray.DataArray`` with dimensions ``('Y', 'X', 'H')``.

    Parameters
    ----------
    filename : str or Path
        Name of HDF5 file to read.
    data_dataset : str, optional, default: 'data'
        HDF5 path of the raw histogram dataset.
    reference_dataset : str, optional
        HDF5 path of the reference histogram dataset.
    irf_dataset : str, optional
        HDF5 path of the instrument response function histogram dataset.
    reference : bool, optional, default: False
        Read reference histogram instead of image data. The reference dataset
        must have shape ``(time, channel)``. The selected channel is returned
        as a ``(1, 1, time)`` signal.
    irf : bool, optional, default: False
        Read IRF histogram instead of REF histogram when reading calibration
        data. If True, calibration data is read even if `reference` is False.
    repetition : int, optional, default: 0
        Repetition index to read.
    z : int, optional, default: 0
        Z-plane index to read.
    channel : int, optional, default: 0
        Channel index to read.
    dtype : dtype_like, optional
        Data type of returned signal.

    Returns
    -------
    xarray.DataArray
        Histogram image with dimensions ``('Y', 'X', 'H')``.

    Raises
    ------
    ValueError
        If ``data_dataset`` is not a dataset with shape ``RZYXHC``.
    IndexError
        If ``repetition``, ``z``, or ``channel`` is out of bounds.

    Notes
    -----
    The last HDF5 dimension is named ``H`` in the returned DataArray because
    PhasorPy uses ``H`` for lifetime histogram/sample axes. The original HDF5
    axes are stored in ``attrs['h5_axes']``.

    Examples
    --------
    >>> signal = signal_from_h5('_phasorpy.h5')  # doctest: +SKIP
    >>> signal.dims  # doctest: +SKIP
    ('Y', 'X', 'H')

    """
    import h5py

    with h5py.File(filename, 'r') as h5:
        file_attrs = _read_h5_attrs(h5)
        if reference or irf:
            calibration_dataset = irf_dataset if irf else reference_dataset
            calibration_name = 'irf' if irf else 'reference'
            try:
                data = h5[calibration_dataset]
            except KeyError as exc:
                msg = (
                    f'HDF5 {calibration_name} dataset '
                    f'{calibration_dataset!r} not found'
                )
                raise ValueError(msg) from exc

            if not isinstance(data, h5py.Dataset):
                msg = f'HDF5 object {calibration_dataset!r} is not a dataset'
                raise ValueError(msg)

            if data.ndim != 2:
                msg = (
                    f'HDF5 {calibration_name} dataset {data.name!r} has shape '
                    f'{data.shape!r}; expected (time, channel)'
                )
                raise ValueError(msg)

            shape = tuple(int(size) for size in data.shape)
            channel = _as_index(channel, shape[1], 'channel')

            signal = numpy.asarray(data[:, channel]).reshape(1, 1, shape[0])
            if dtype is not None:
                signal = signal.astype(dtype, copy=False)

            data_attrs = _read_h5_attrs(data)
            h_coord = _histogram_coordinates(data, shape[0])
            dataset_name = str(data.name)
            attrs: dict[str, Any] = {
                'h5_dataset': dataset_name,
                'h5_shape': shape,
                'h5_axes': ('H', 'C'),
                'h5_selection': {
                    'reference': True,
                    'irf': bool(irf),
                    'channel': channel,
                },
                'h5_attrs': {
                    'file': file_attrs,
                    calibration_name: data_attrs,
                },
                'samples': shape[0],
                'reference': True,
                'irf': bool(irf),
            }
            frequency = _first_attr(
                data_attrs,
                file_attrs,
                names=('frequency', 'Frequency'),
            )
            if frequency is not None:
                attrs['frequency'] = frequency

            metadata = xarray_metadata(
                'YXH',
                signal.shape,
                filename,
                attrs=attrs,
                H=h_coord,
            )

            from xarray import DataArray

            return DataArray(signal, **metadata)

        try:
            data = h5[data_dataset]
        except KeyError as exc:
            msg = f'HDF5 dataset {data_dataset!r} not found'
            raise ValueError(msg) from exc

        if not isinstance(data, h5py.Dataset):
            msg = f'HDF5 object {data_dataset!r} is not a dataset'
            raise ValueError(msg)

        if data.ndim != 6:
            msg = (
                f'HDF5 dataset {data.name!r} has shape {data.shape!r}; '
                'expected (repetition, z, y, x, time, channel)'
            )
            raise ValueError(msg)

        shape = tuple(int(size) for size in data.shape)
        repetition = _as_index(repetition, shape[0], 'repetition')
        z = _as_index(z, shape[1], 'z')
        channel = _as_index(channel, shape[5], 'channel')

        signal = numpy.asarray(data[repetition, z, :, :, :, channel])
        if dtype is not None:
            signal = signal.astype(dtype, copy=False)

        data_attrs = _read_h5_attrs(data)
        h_coord = _histogram_coordinates(data, shape[4])
        dataset_name = str(data.name)

    attrs: dict[str, Any] = {
        'h5_dataset': dataset_name,
        'h5_shape': shape,
        'h5_axes': _H5_AXES,
        'h5_selection': {
            'repetition': repetition,
            'z': z,
            'channel': channel,
        },
        'h5_attrs': {
            'file': file_attrs,
            'data': data_attrs,
        },
        'samples': shape[4],
    }

    frequency = _first_attr(
        data_attrs,
        file_attrs,
        names=('frequency', 'Frequency'),
    )
    if frequency is not None:
        attrs['frequency'] = frequency

    metadata = xarray_metadata(
        'YXH',
        signal.shape,
        filename,
        attrs=attrs,
        H=h_coord,
    )

    from xarray import DataArray

    return DataArray(signal, **metadata)


def phasor_from_h5(
    filename: str | PathLike[Any],
    /,
    *,
    data_dataset: str = _DATASET,
    reference_dataset: str = _REFERENCE_DATASET,
    irf_dataset: str = _IRF_DATASET,
    reference: bool = False,
    irf: bool = False,
    repetition: int = 0,
    z: int = 0,
    channel: int = 0,
    harmonic: int | Sequence[int] | Literal['all'] | str | None = None,
    dtype: DTypeLike | None = None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], dict[str, Any]]:
    """Return phasor coordinates from HDF5 ``/data`` histogram image.

    This is a convenience wrapper around :func:`signal_from_h5` and
    :func:`phasorpy.phasor.phasor_from_signal`. It reads one
    ``(Y, X, H)`` histogram image from an HDF5 dataset with shape
    ``(repetition, z, y, x, time, channel)`` and computes phasor coordinates
    along the histogram axis.

    Parameters
    ----------
    filename : str or Path
        Name of HDF5 file to read.
    data_dataset : str, optional, default: 'data'
        HDF5 path of the raw histogram dataset.
    reference_dataset : str, optional
        HDF5 path of the reference histogram dataset.
    irf_dataset : str, optional
        HDF5 path of the instrument response function histogram dataset.
    reference : bool, optional, default: False
        Read reference histogram instead of image data.
    irf : bool, optional, default: False
        Read IRF histogram instead of REF histogram when reading calibration
        data. If True, calibration data is read even if `reference` is False.
    repetition : int, optional, default: 0
        Repetition index to read.
    z : int, optional, default: 0
        Z-plane index to read.
    channel : int, optional, default: 0
        Channel index to read.
    harmonic : int, sequence of int, or 'all', optional
        Harmonic(s) to calculate.
    dtype : dtype_like, optional
        Data type of signal passed to phasor calculation.

    Returns
    -------
    mean : ndarray
        Average intensity image.
    real : ndarray
        Real component of phasor coordinates.
    imag : ndarray
        Imaginary component of phasor coordinates.
    attrs : dict
        Metadata copied from :func:`signal_from_h5`, plus ``'dims'`` and
        ``'harmonic'`` entries for compatibility with other phasor readers.

    """
    from ..phasor import phasor_from_signal

    signal = signal_from_h5(
        filename,
        data_dataset=data_dataset,
        reference_dataset=reference_dataset,
        irf_dataset=irf_dataset,
        reference=reference,
        irf=irf,
        repetition=repetition,
        z=z,
        channel=channel,
        dtype=dtype,
    )
    mean, real, imag = phasor_from_signal(
        signal,
        axis='H',
        harmonic=harmonic,
    )

    harmonics, has_harmonic_axis = parse_harmonic(
        harmonic,
        signal.shape[-1] // 2,
    )
    attrs = dict(signal.attrs)
    attrs['dims'] = tuple(dim for dim in signal.dims if dim != 'H')
    attrs['harmonic'] = harmonics if has_harmonic_axis else harmonics[0]
    attrs['signal_dims'] = tuple(signal.dims)

    return mean, real, imag, attrs
