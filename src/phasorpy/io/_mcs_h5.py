"""Read BrightEyes-MCS HDF5 histogram image files."""

from __future__ import annotations

__all__ = ["signal_from_brighteyes_mcs"]

from typing import TYPE_CHECKING

from .._utils import xarray_metadata

if TYPE_CHECKING:
    from .._typing import Any, DataArray, Literal, PathLike


def signal_from_brighteyes_mcs(
    filename: str | PathLike[Any],
    /,
    *,
    dataset: str | None = None,
    time: int = 0,
    depth: int = 0,
    channel: int | Literal["sum"] = 0,
) -> DataArray:
    """Return histogram image from BrightEyes-MCS HDF5 file.

    Parameters
    ----------
    filename : str or Path
        Name of BrightEyes-MCS HDF5 file to read.
    dataset : str, optional
        Dataset to read. By default, read the BrightEyes-MCS default image
        dataset. The reserved aliases ``"reference"`` and ``"irf"`` read the
        default calibration reference and IRF traces. Other strings are
        interpreted as literal HDF5 dataset paths.
    time : int, optional, default: 0
        Index along the BrightEyes-MCS repetition/time axis.
    depth : int, optional, default: 0
        Index along the BrightEyes-MCS z/depth axis.
    channel : int or 'sum', optional, default: 0
        Channel index to read. If ``"sum"``, integrate the channel axis.

    Returns
    -------
    xarray.DataArray
        Histogram image with dimensions such as ``('Y', 'X', 'H')``.

    Raises
    ------
    ImportError
        If the ``brighteyes-mcs-reader`` package is not installed.
    ValueError
        If the file is not a supported BrightEyes-MCS HDF5 file.

    Notes
    -----
    The implementation is based on the
    `brighteyes-mcs-reader
    <https://github.com/VicidominiLab/BrightEyes-MCS-Reader/>`__ library.

    Examples
    --------
    >>> signal = signal_from_brighteyes_mcs('measurement.h5')  # doctest: +SKIP
    >>> signal.dims  # doctest: +SKIP
    ('Y', 'X', 'H')

    """
    try:
        from brighteyes_mcs_reader import read_signal
    except ImportError as exc:
        msg = (
            "the brighteyes-mcs-reader package is required to read "
            "BrightEyes-MCS HDF5 files"
        )
        raise ImportError(msg) from exc

    signal = read_signal(
        filename,
        dataset=dataset,
        time=time,
        depth=depth,
        channel=channel,
    )

    metadata = xarray_metadata(
        signal.dims,
        signal.data.shape,
        filename,
        attrs=signal.attrs,
        **signal.coords,
    )

    from xarray import DataArray

    return DataArray(signal.data, **metadata)
