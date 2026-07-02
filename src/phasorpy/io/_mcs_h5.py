"""Read BrightEyes MCS-H5 and generic HDF5 histogram image files."""

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

_DATASET = 'raw/spad'
_REFERENCE_DATASET = 'output/sum_reference_trace'
_IRF_DATASET = 'output/sum_irf_trace'
_GENERIC_DATASET = 'data'
_GENERIC_REFERENCE_DATASET = 'calibration/data/ref_common_delay_realigned'
_GENERIC_IRF_DATASET = 'calibration/data/irf_common_delay_realigned'
_MCS_H5_AXES = {
    5: ('repetition', 'z', 'y', 'x', 'time_bin'),
    6: ('repetition', 'z', 'y', 'x', 'time_bin', 'channel'),
    8: (
        'repetition',
        'z',
        'y',
        'x',
        'circular_repetition',
        'circular_point',
        'time_bin',
        'channel',
    ),
}
_GENERIC_H5_AXES = ('R', 'Z', 'Y', 'X', 'H', 'C')


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


def _read_first_h5_attrs(h5: Any, *paths: str) -> dict[str, Any]:
    """Return attributes from the first existing HDF5 object."""
    for path in paths:
        path = str(path).strip('/')
        if path not in h5:
            continue
        obj = h5[path]
        return _read_h5_attrs(obj)
    return {}


def _as_h5_dataset(h5: Any, path: str, name: str) -> Any:
    """Return HDF5 dataset from path."""
    path = str(path).strip('/')
    try:
        obj = h5[path]
    except KeyError as exc:
        msg = f'HDF5 {name} dataset {path!r} not found'
        raise ValueError(msg) from exc
    return obj


def _h5_path_exists(h5: Any, path: Any) -> bool:
    """Return whether path names an existing HDF5 object."""
    return isinstance(path, str) and bool(path.strip('/')) and path.strip('/') in h5


def _is_brighteyes_mcs_h5(h5: Any, file_attrs: dict[str, Any]) -> bool:
    """Return whether file advertises the BrightEyes MCS 0.0.6 schema."""
    if file_attrs.get('schema_name') == 'brighteyes_mcs_file':
        return True
    if 'raw/spad' in h5 and 'data' not in h5:
        return True
    if file_attrs.get('data_format_version') != '0.0.6':
        return False
    required_groups = ('raw', 'raw/metadata', 'raw/axes')
    if not all(path in h5 for path in required_groups):
        return False
    default = file_attrs.get('default')
    data_path = file_attrs.get('data_path')
    metadata_path = file_attrs.get('metadata_path')
    axes_path = file_attrs.get('axes_path')
    return (
        _h5_path_exists(h5, default)
        and data_path in (None, '/raw', 'raw')
        and metadata_path in (None, '/raw/metadata', 'raw/metadata')
        and axes_path in (None, '/raw/axes', 'raw/axes')
    )


def _first_existing_dataset(h5: Any, *paths: str) -> str:
    """Return the first path that points to an HDF5 dataset."""
    import h5py

    for path in paths:
        if not _h5_path_exists(h5, path):
            continue
        if isinstance(h5[str(path).strip('/')], h5py.Dataset):
            return str(path).strip('/')
    return str(paths[0]).strip('/')


def _mcs_default_data_dataset(h5: Any, requested: str) -> str:
    """Return default BrightEyes data dataset unless caller selected one."""
    requested = str(requested).strip('/')
    if requested not in {_DATASET, _GENERIC_DATASET}:
        return requested
    default = _normalize_h5_value(h5.attrs.get('default', ''))
    if _h5_path_exists(h5, default):
        return str(default).strip('/')
    if _h5_path_exists(h5, _DATASET):
        return _DATASET
    return requested


def _axis_coordinates(h5: Any, path: Any, size: int) -> NDArray[Any] | None:
    """Return one-dimensional axis coordinates if present."""
    if not isinstance(path, str):
        return None
    path = path.strip('/')
    if not path or path not in h5:
        return None
    coord = numpy.asarray(h5[path])
    if coord.ndim == 1 and coord.size == size:
        return coord
    return None


def _default_time_axis_paths(data: Any) -> tuple[str, ...]:
    """Return conventional time axis paths for a dataset."""
    data_path = str(data.name).strip('/')
    if data_path.startswith('raw/'):
        return ('raw/axes/digital_time_ns',)
    if data_path.startswith('calibration/'):
        return ('calibration/axes/time_ns',)
    if data_path.startswith('output/'):
        parts = data_path.split('/')
        if len(parts) > 1:
            return (f'output/{parts[1]}/axes/time_ns',)
    return ()


def _product_name(data: Any) -> str | None:
    """Return BrightEyes product name matching a dataset, if known."""
    attrs = _read_h5_attrs(data)
    product = attrs.get('product_name')
    if isinstance(product, str) and product:
        return product
    source_key = attrs.get('source_key') or attrs.get('source_data_key')
    if source_key == 'data':
        return 'spad'
    if source_key == 'data_channels_extra':
        return 'aux'

    data_path = str(data.name).strip('/')
    if data_path == 'raw/spad':
        return 'spad'
    if data_path == 'raw/aux':
        return 'aux'
    if data_path.endswith('/products/image_aux'):
        return 'aux'
    if data_path.endswith('/products/image'):
        return 'spad'
    return None


def _calibration_result_attrs(h5: Any, data: Any) -> dict[str, Any]:
    """Return attrs from the calibration result group containing a dataset."""
    parts = str(data.name).strip('/').split('/')
    if len(parts) >= 3 and parts[:2] == ['calibration', 'results']:
        return _read_first_h5_attrs(h5, '/'.join(parts[:3]))
    return {}


def _raw_calibration_attrs(h5: Any, data: Any) -> dict[str, Any]:
    """Return BrightEyes calibration attrs matching a raw data dataset."""
    product = _product_name(data)
    if product == 'spad':
        return _read_first_h5_attrs(h5, 'calibration/results/spad')
    if product == 'aux':
        return _read_first_h5_attrs(h5, 'calibration/results/aux')
    data_path = str(data.name).strip('/')
    return _read_first_h5_attrs(h5, f'calibration/{data_path}')


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


def _first_finite_float(value: Any) -> float | None:
    """Return first finite positive float from scalar or array-like value."""
    try:
        array = numpy.asarray(value, dtype=numpy.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    finite = array[numpy.isfinite(array) & (array > 0.0)]
    if finite.size == 0:
        return None
    return float(finite[0])


def _reference_lifetime_ns(
    h5: Any,
    *attrs: dict[str, Any],
    product: str | None = None,
) -> float | None:
    """Return BrightEyes reference lifetime in ns, if available."""
    value = _first_attr(
        *attrs,
        names=(
            'reference_lifetime_ns',
            'tau_reference_ns',
            'tau_ref_input_ns',
        ),
    )
    lifetime = _first_finite_float(value)
    if lifetime is not None:
        return lifetime

    products = [product] if product else []
    for candidate in ('spad', 'aux'):
        if candidate not in products:
            products.append(candidate)

    for product_name in products:
        path = f'calibration/results/{product_name}/fit/tau_reference_ns'
        if path not in h5:
            continue
        lifetime = _first_finite_float(h5[path][()])
        if lifetime is not None:
            return lifetime
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


def _histogram_coordinates(
    h5: Any,
    data: Any,
    size: int,
    *attrs: dict[str, Any],
) -> NDArray[Any]:
    """Return histogram coordinates from dataset attributes, if present."""
    for name in ('H', 'h', 'T', 't', 'time', 'times', 'histogram_bins'):
        if name not in data.attrs:
            continue
        coord = numpy.asarray(_normalize_h5_value(data.attrs[name]))
        if coord.ndim == 1 and coord.size == size:
            return coord
    data_attrs = _read_h5_attrs(data)
    for attr in (data_attrs, *attrs):
        for name in (
            'time_axis_path',
            'source_time_axis_path',
            'time_axis_source',
        ):
            coord = _axis_coordinates(h5, attr.get(name), size)
            if coord is not None:
                return coord
    for path in _default_time_axis_paths(data):
        coord = _axis_coordinates(h5, path, size)
        if coord is not None:
            return coord
    timebin = _first_attr(
        data_attrs,
        *attrs,
        names=(
            'time_bin_ns',
            'digital_time_bin_ns',
            'timebin_in_ns',
            'bin_width_ns',
        ),
    )
    if timebin is not None:
        timebin = float(timebin)
        if numpy.isfinite(timebin):
            return numpy.arange(size, dtype=numpy.float64) * timebin
    return numpy.arange(size)


def _generic_histogram_coordinates(data: Any, size: int) -> NDArray[Any]:
    """Return histogram coordinates from generic dataset attributes."""
    for name in ('H', 'h', 'T', 't', 'time', 'times', 'histogram_bins'):
        if name not in data.attrs:
            continue
        coord = numpy.asarray(_normalize_h5_value(data.attrs[name]))
        if coord.ndim == 1 and coord.size == size:
            return coord
    return numpy.arange(size)


def _signal_from_generic_h5(
    h5: Any,
    filename: str | PathLike[Any],
    *,
    data_dataset: str,
    reference_dataset: str,
    irf_dataset: str,
    reference: bool,
    irf: bool,
    repetition: int,
    z: int,
    channel: int | Literal['sum'],
    dtype: DTypeLike | None,
    file_attrs: dict[str, Any],
) -> DataArray:
    """Return histogram image from the legacy generic HDF5 convention."""
    import h5py

    if data_dataset == _DATASET:
        data_dataset = _GENERIC_DATASET
    if reference_dataset == _REFERENCE_DATASET:
        reference_dataset = _GENERIC_REFERENCE_DATASET
    if irf_dataset == _IRF_DATASET:
        irf_dataset = _GENERIC_IRF_DATASET

    if reference or irf:
        calibration_dataset = irf_dataset if irf else reference_dataset
        calibration_name = 'irf' if irf else 'reference'
        data = _as_h5_dataset(h5, calibration_dataset, calibration_name)

        if not isinstance(data, h5py.Dataset):
            msg = f'HDF5 object {calibration_dataset!r} is not a dataset'
            raise ValueError(msg)

        if data.ndim not in (1, 2):
            msg = (
                f'HDF5 {calibration_name} dataset {data.name!r} has shape '
                f'{data.shape!r}; expected (time,) or (time, channel)'
            )
            raise ValueError(msg)

        shape = tuple(int(size) for size in data.shape)
        if data.ndim == 1:
            if channel not in (0, 'sum', None):
                msg = f'{channel=} is out of bounds for unchannelled data'
                raise IndexError(msg)
            channel_selection: int | str | None = None
            signal = numpy.asarray(data[:]).reshape(1, 1, shape[0])
            h5_axes = ('H',)
        elif channel == 'sum':
            channel_selection = 'sum'
            signal = numpy.asarray(data[:, :]).sum(axis=1).reshape(
                1, 1, shape[0]
            )
            h5_axes = ('H', 'C')
        else:
            channel = _as_index(channel, shape[1], 'channel')
            channel_selection = channel
            signal = numpy.asarray(data[:, channel]).reshape(1, 1, shape[0])
            h5_axes = ('H', 'C')

        if dtype is not None:
            signal = signal.astype(dtype, copy=False)

        data_attrs = _read_h5_attrs(data)
        h_coord = _generic_histogram_coordinates(data, shape[0])
        attrs: dict[str, Any] = {
            'h5_dataset': str(data.name),
            'h5_shape': shape,
            'h5_axes': h5_axes,
            'h5_selection': {
                'reference': True,
                'irf': bool(irf),
                'channel': channel_selection,
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
            'YXH', signal.shape, filename, attrs=attrs, H=h_coord
        )

        from xarray import DataArray

        return DataArray(signal, **metadata)

    data = _as_h5_dataset(h5, data_dataset, 'data')

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
    if channel == 'sum':
        channel_selection: int | str = 'sum'
        signal = numpy.asarray(data[repetition, z, :, :, :, :]).sum(axis=-1)
    else:
        channel = _as_index(channel, shape[5], 'channel')
        channel_selection = channel
        signal = numpy.asarray(data[repetition, z, :, :, :, channel])
    if dtype is not None:
        signal = signal.astype(dtype, copy=False)

    data_attrs = _read_h5_attrs(data)
    h_coord = _generic_histogram_coordinates(data, shape[4])
    attrs = {
        'h5_dataset': str(data.name),
        'h5_shape': shape,
        'h5_axes': _GENERIC_H5_AXES,
        'h5_selection': {
            'repetition': repetition,
            'z': z,
            'channel': channel_selection,
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
        'YXH', signal.shape, filename, attrs=attrs, H=h_coord
    )

    from xarray import DataArray

    return DataArray(signal, **metadata)


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
    channel: int | Literal['sum'] = 0,
    dtype: DTypeLike | None = None,
) -> DataArray:
    """Return histogram image from BrightEyes MCS-H5 or generic HDF5.

    Files advertising the BrightEyes MCS 0.0.6 schema are read using the
    normalized ``/raw``, ``/calibration``, and ``/output`` paths. Other HDF5
    files fall back to the generic convention with a root ``/data`` dataset.

    Parameters
    ----------
    filename : str or Path
        Name of HDF5 file to read.
    data_dataset : str, optional, default: 'raw/spad'
        HDF5 path of the histogram dataset. For non-BrightEyes files, the
        default falls back to ``/data``.
    reference_dataset : str, optional
        HDF5 path of the common reference histogram dataset.
    irf_dataset : str, optional
        HDF5 path of the common instrument response function histogram dataset.
    reference : bool, optional, default: False
        Read reference histogram instead of image data. The reference dataset
        must have shape ``(time,)`` or ``(time, channel)``. The selected trace
        is returned as a ``(1, 1, time)`` signal.
    irf : bool, optional, default: False
        Read IRF histogram instead of REF histogram when reading calibration
        data. If True, calibration data is read even if `reference` is False.
    repetition : int, optional, default: 0
        Repetition index to read.
    z : int, optional, default: 0
        Z-plane index to read.
    channel : int or 'sum', optional, default: 0
        Channel index to read. If 'sum', integrate channel axis.
    dtype : dtype_like, optional
        Data type of returned signal.

    Returns
    -------
    xarray.DataArray
        Histogram image with dimensions ``('Y', 'X', 'H')`` for raster data
        or ``('Y', 'X', 'Q', 'P', 'H')`` for circular scan data.

    Raises
    ------
    ValueError
        If ``data_dataset`` is not a dataset with supported BrightEyes shape.
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
        if not _is_brighteyes_mcs_h5(h5, file_attrs):
            return _signal_from_generic_h5(
                h5,
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
                file_attrs=file_attrs,
            )

        data_dataset = _mcs_default_data_dataset(h5, data_dataset)
        reference_dataset = _first_existing_dataset(
            h5,
            reference_dataset,
            'calibration/results/spad/aligned/reference_trace',
            'output/sum_ref_001/products/trace',
        )
        irf_dataset = _first_existing_dataset(
            h5,
            irf_dataset,
            'calibration/results/spad/aligned/irf_trace',
            'output/sum_irf_001/products/trace',
        )
        if reference or irf:
            calibration_dataset = irf_dataset if irf else reference_dataset
            calibration_name = 'irf' if irf else 'reference'
            data = _as_h5_dataset(h5, calibration_dataset, calibration_name)

            if not isinstance(data, h5py.Dataset):
                msg = f'HDF5 object {calibration_dataset!r} is not a dataset'
                raise ValueError(msg)

            if data.ndim not in (1, 2):
                msg = (
                    f'HDF5 {calibration_name} dataset {data.name!r} has shape '
                    f'{data.shape!r}; expected (time,) or (time, channel)'
                )
                raise ValueError(msg)

            shape = tuple(int(size) for size in data.shape)
            if data.ndim == 1:
                if channel not in (0, 'sum', None):
                    msg = f'{channel=} is out of bounds for unchannelled data'
                    raise IndexError(msg)
                channel_selection: int | str | None = None
                signal = numpy.asarray(data[:]).reshape(1, 1, shape[0])
                h5_axes = ('H',)
            else:
                sum_channel = channel == 'sum'
                if sum_channel:
                    channel_selection = 'sum'
                    signal = numpy.asarray(data[:, :]).sum(axis=1).reshape(
                        1, 1, shape[0]
                    )
                else:
                    channel = _as_index(channel, shape[1], 'channel')
                    channel_selection = channel
                    signal = numpy.asarray(data[:, channel]).reshape(
                        1, 1, shape[0]
                    )
                h5_axes = ('H', 'C')

            if dtype is not None:
                signal = signal.astype(dtype, copy=False)

            data_attrs = _read_h5_attrs(data)
            parent_attrs = _read_h5_attrs(data.parent)
            calibration_attrs = _calibration_result_attrs(h5, data)
            calibration_metadata_attrs = _read_first_h5_attrs(
                h5,
                'calibration/metadata',
            )
            h_coord = _histogram_coordinates(
                h5,
                data,
                shape[0],
                parent_attrs,
                calibration_attrs,
                calibration_metadata_attrs,
            )
            dataset_name = str(data.name)
            attrs: dict[str, Any] = {
                'h5_dataset': dataset_name,
                'h5_shape': shape,
                'h5_axes': h5_axes,
                'h5_selection': {
                    'reference': True,
                    'irf': bool(irf),
                    'channel': channel_selection,
                },
                'h5_attrs': {
                    'file': file_attrs,
                    'calibration': calibration_attrs,
                    'calibration_metadata': calibration_metadata_attrs,
                    'group': parent_attrs,
                    calibration_name: data_attrs,
                },
                'samples': shape[0],
                'reference': True,
                'irf': bool(irf),
            }
            frequency = _first_attr(
                data_attrs,
                parent_attrs,
                calibration_attrs,
                calibration_metadata_attrs,
                file_attrs,
                names=('frequency', 'Frequency', 'laser_frequency_mhz'),
            )
            if frequency is not None:
                attrs['frequency'] = frequency
            lifetime = _reference_lifetime_ns(
                h5,
                data_attrs,
                parent_attrs,
                calibration_attrs,
                calibration_metadata_attrs,
                product='spad',
            )
            if lifetime is not None:
                attrs['reference_lifetime_ns'] = lifetime

            metadata = xarray_metadata(
                'YXH',
                signal.shape,
                filename,
                attrs=attrs,
                H=h_coord,
            )

            from xarray import DataArray

            return DataArray(signal, **metadata)

        data = _as_h5_dataset(h5, data_dataset, 'data')

        if not isinstance(data, h5py.Dataset):
            msg = f'HDF5 object {data_dataset!r} is not a dataset'
            raise ValueError(msg)

        if data.ndim not in _MCS_H5_AXES:
            msg = (
                f'HDF5 dataset {data.name!r} has shape {data.shape!r}; '
                'expected (repetition, z, y, x, time[, channel])'
            )
            raise ValueError(msg)

        shape = tuple(int(size) for size in data.shape)
        data_ndim = data.ndim
        repetition = _as_index(repetition, shape[0], 'repetition')
        z = _as_index(z, shape[1], 'z')

        if data.ndim == 5:
            if channel not in (0, 'sum'):
                msg = f'{channel=} is out of bounds for unchannelled data'
                raise IndexError(msg)
            channel_selection: int | str | None = None
            signal = numpy.asarray(data[repetition, z, :, :, :])
            samples = shape[4]
            signal_dims = ('Y', 'X', 'H')
        elif data.ndim == 6:
            if channel == 'sum':
                channel_selection = 'sum'
                signal = numpy.asarray(data[repetition, z, :, :, :, :]).sum(
                    axis=-1
                )
            else:
                channel = _as_index(channel, shape[5], 'channel')
                channel_selection = channel
                signal = numpy.asarray(data[repetition, z, :, :, :, channel])
            samples = shape[4]
            signal_dims = ('Y', 'X', 'H')
        else:
            if channel == 'sum':
                channel_selection = 'sum'
                signal = numpy.asarray(
                    data[repetition, z, :, :, :, :, :, :]
                ).sum(axis=-1)
            else:
                channel = _as_index(channel, shape[7], 'channel')
                channel_selection = channel
                signal = numpy.asarray(
                    data[repetition, z, :, :, :, :, :, channel]
                )
            samples = shape[6]
            signal_dims = ('Y', 'X', 'Q', 'P', 'H')
        if dtype is not None:
            signal = signal.astype(dtype, copy=False)

        data_attrs = _read_h5_attrs(data)
        parent_attrs = _read_h5_attrs(data.parent)
        calibration_attrs = _raw_calibration_attrs(h5, data)
        product_name = _product_name(data)
        metadata_attrs = _read_first_h5_attrs(
            h5,
            data_attrs.get('metadata_path', ''),
            parent_attrs.get('metadata_path', ''),
            'raw/metadata',
        )
        timing_attrs = _read_first_h5_attrs(
            h5,
            'raw/metadata/acquisition/timing',
        )
        h_coord = _histogram_coordinates(
            h5,
            data,
            samples,
            parent_attrs,
            calibration_attrs,
            metadata_attrs,
            timing_attrs,
        )
        dataset_name = str(data.name)
        h5_axes = (
            tuple(str(data_attrs['axis_order']).split(','))
            if data_attrs.get('axis_order') is not None
            else _MCS_H5_AXES[data_ndim]
        )
        reference_lifetime_ns = _reference_lifetime_ns(
            h5,
            data_attrs,
            parent_attrs,
            metadata_attrs,
            timing_attrs,
            calibration_attrs,
            file_attrs,
            product=product_name,
        )

    attrs: dict[str, Any] = {
        'h5_dataset': dataset_name,
        'h5_shape': shape,
        'h5_axes': h5_axes,
        'h5_selection': {
            'repetition': repetition,
            'z': z,
            'channel': channel_selection,
        },
        'h5_attrs': {
            'file': file_attrs,
            'group': parent_attrs,
            'metadata': metadata_attrs,
            'timing': timing_attrs,
            'calibration': calibration_attrs,
            'data': data_attrs,
        },
        'samples': samples,
    }

    frequency = _first_attr(
        data_attrs,
        parent_attrs,
        metadata_attrs,
        timing_attrs,
        calibration_attrs,
        file_attrs,
        names=('frequency', 'Frequency', 'laser_frequency_mhz'),
    )
    if frequency is not None:
        attrs['frequency'] = frequency
    if reference_lifetime_ns is not None:
        attrs['reference_lifetime_ns'] = reference_lifetime_ns

    metadata = xarray_metadata(
        signal_dims,
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
    channel: int | Literal['sum'] = 0,
    harmonic: int | Sequence[int] | Literal['all'] | str | None = None,
    dtype: DTypeLike | None = None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], dict[str, Any]]:
    """Return phasor coordinates from BrightEyes MCS-H5 histogram image.

    This is a convenience wrapper around :func:`signal_from_h5` and
    :func:`phasorpy.phasor.phasor_from_signal`. It reads one
    histogram image from an HDF5 dataset with BrightEyes channel-last layout
    and computes phasor coordinates along the histogram axis.

    Parameters
    ----------
    filename : str or Path
        Name of HDF5 file to read.
    data_dataset : str, optional, default: 'raw/spad'
        HDF5 path of the raw histogram dataset.
    reference_dataset : str, optional
        HDF5 path of the common reference histogram dataset.
    irf_dataset : str, optional
        HDF5 path of the common instrument response function histogram dataset.
    reference : bool, optional, default: False
        Read reference histogram instead of image data.
    irf : bool, optional, default: False
        Read IRF histogram instead of REF histogram when reading calibration
        data. If True, calibration data is read even if `reference` is False.
    repetition : int, optional, default: 0
        Repetition index to read.
    z : int, optional, default: 0
        Z-plane index to read.
    channel : int or 'sum', optional, default: 0
        Channel index to read. If 'sum', integrate channel axis.
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
