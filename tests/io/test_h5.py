"""Test BrightEyes-MCS HDF5 file reader wrapper."""

from __future__ import annotations

import builtins

import numpy
import pytest
from numpy.testing import assert_array_equal

from phasorpy.io import signal_from_brighteyes_mcs

h5py = pytest.importorskip("h5py")


def _write_h5(filename):
    data = numpy.arange(2 * 3 * 4 * 5 * 8 * 2, dtype=numpy.uint16).reshape(
        2, 3, 4, 5, 8, 2
    )
    output = data.sum(axis=-1)
    reference = numpy.arange(8, dtype=numpy.float32)

    with h5py.File(filename, "w") as h5:
        h5.attrs["schema_name"] = "brighteyes_mcs_file"
        h5.attrs["data_format_version"] = "0.0.6"
        h5.attrs["default"] = "/raw/spad"

        raw = h5.create_group("raw")
        raw.attrs["metadata_path"] = "/raw/metadata"
        raw.attrs["axes_path"] = "/raw/axes"
        raw_data = raw.create_dataset("spad", data=data)
        raw_data.attrs["axis_order"] = (
            "repetition,z,y,x,time_bin,detector_channel"
        )
        raw_data.attrs["source_key"] = "data"
        raw_data.attrs["time_axis_path"] = "/raw/axes/digital_time_ns"
        raw_data.attrs["metadata_path"] = "/raw/metadata"

        metadata = raw.create_group("metadata")
        metadata.attrs["laser_frequency_mhz"] = 80.0
        timing = metadata.create_group("acquisition").create_group("timing")
        timing.attrs["digital_time_bin_ns"] = 12.5 / 8
        axes = raw.create_group("axes")
        axes.create_dataset(
            "digital_time_ns",
            data=numpy.linspace(0.0, 12.5, 8, endpoint=False),
        )

        calibration = h5.create_group("calibration")
        result = calibration.create_group("results").create_group("spad")
        result.attrs["laser_frequency_mhz"] = 80.0
        fit = result.create_group("fit")
        fit.create_dataset("tau_reference_ns", data=numpy.array([2.7, 2.7]))

        output_group = h5.create_group("output")
        output_group.attrs["default"] = "/output/sum_001/products/spad"
        output_group.attrs["default_run"] = "/output/sum_001"
        output_group.attrs["default_ref_trace_id"] = (
            "/output/sum_ref_001/products/trace"
        )

        run = output_group.create_group("sum_001")
        run.create_group("metadata").attrs["laser_frequency_mhz"] = 80.0
        run.create_group("axes").create_dataset(
            "time_ns", data=numpy.linspace(0.0, 12.5, 8, endpoint=False)
        )
        product = run.create_group("products").create_dataset(
            "spad", data=output
        )
        product.attrs["axis_order"] = "repetition,z,y,x,time_bin"
        product.attrs["time_axis_path"] = "/output/sum_001/axes/time_ns"
        product.attrs["metadata_path"] = "/output/sum_001/metadata"

        ref_run = output_group.create_group("sum_ref_001")
        ref_run.create_group("axes").create_dataset(
            "time_ns", data=numpy.linspace(0.0, 12.5, 8, endpoint=False)
        )
        ref = ref_run.create_group("products").create_dataset(
            "trace", data=reference
        )
        ref.attrs["output_type"] = "trace"
        ref.attrs["trace_kind"] = "sum_reference_trace"
        ref.attrs["time_axis_path"] = "/output/sum_ref_001/axes/time_ns"

    return output, reference


def test_signal_from_brighteyes_mcs(tmp_path):
    """Test wrapping a BrightEyes-MCS signal as xarray."""
    pytest.importorskip("brighteyes_mcs_reader")
    pytest.importorskip("xarray")
    filename = tmp_path / "histogram.h5"
    data, _ = _write_h5(filename)

    signal = signal_from_brighteyes_mcs(filename, time=1, depth=2)

    assert signal.dims == ("Y", "X", "H")
    assert signal.shape == (4, 5, 8)
    assert_array_equal(signal.values, data[1, 2])
    assert_array_equal(
        signal.coords["H"].values,
        numpy.linspace(0.0, 12.5, 8, endpoint=False),
    )
    assert signal.attrs["frequency"] == 80.0
    assert signal.attrs["reference_lifetime_ns"] == 2.7
    assert signal.attrs["h5_dataset"] == "/output/sum_001/products/spad"


def test_signal_from_brighteyes_mcs_reference(tmp_path):
    """Test reading a default BrightEyes-MCS reference trace."""
    pytest.importorskip("brighteyes_mcs_reader")
    pytest.importorskip("xarray")
    filename = tmp_path / "histogram.h5"
    _, reference = _write_h5(filename)

    signal = signal_from_brighteyes_mcs(filename, dataset="reference")

    assert signal.dims == ("Y", "X", "H")
    assert signal.shape == (1, 1, 8)
    assert_array_equal(signal.values, reference.reshape(1, 1, 8))
    assert signal.attrs["reference"] is True
    assert signal.attrs["irf"] is False


def test_signal_from_brighteyes_mcs_import_error(monkeypatch):
    """Test optional dependency error message."""
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "brighteyes_mcs_reader":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match="brighteyes-mcs-reader"):
        signal_from_brighteyes_mcs("missing.h5")
