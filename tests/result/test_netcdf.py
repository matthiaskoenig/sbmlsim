"""The compressed netCDF files of the results (#412)."""

import subprocess
import sys
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pytest
import xarray as xr

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.result import ScanResult
from sbmlsim.result.netcdf import (
    CHUNK_BYTES,
    STACK_MIN,
    STACK_PREFIX,
    chunk_shape,
    read_netcdf,
    write_netcdf,
)
from sbmlsim.sensitivity import SensitivityResult
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator

#: the HDF5 filter of zstd of hdf5plugin
ZSTD_FILTER = "32015"


@pytest.fixture(scope="module")
def results() -> dict[str, ScanResult]:
    """A result on a grid and a ragged one, with a status and a coordinate."""
    model = RoadrunnerSBMLModel(source=REPRESSILATOR_SBML)
    model.set_selections(["time", "PX", "PY", "PZ"])
    scan = Scan(
        Simulation(end=50),
        [Dimension("d", values={"n": np.linspace(1.5, 4.0, 20)})],
    )
    simulator = Simulator(n_workers=1)
    return {
        "grid": simulator.run(model, scan, time=np.linspace(0, 50, 51)),
        "ragged": simulator.run(model, scan, on_error="flag"),
    }


def _filters(path: Path, name: str) -> tuple[tuple[int, ...] | None, set[str]]:
    """Get the chunks and the filters of a variable of a file."""
    with h5py.File(path) as f:
        dataset = f[name]
        assert isinstance(dataset, h5py.Dataset)
        return dataset.chunks, set(dataset._filters)


@pytest.mark.parametrize("layout", ["grid", "ragged"])
@pytest.mark.parametrize("compression", ["zstd", "gzip", None])
def test_a_result_is_read_as_it_was_written(
    results: dict[str, ScanResult],
    layout: str,
    compression: str | None,
    tmp_path: Path,
) -> None:
    result = results[layout]
    path = tmp_path / "result.nc"
    result.to_netcdf(path, compression=compression)  # ty: ignore[invalid-argument-type]
    again = ScanResult.from_netcdf(path)
    xr.testing.assert_identical(again.ds, result.ds)
    chunks, filters = _filters(path, "PX")
    if compression is None:
        assert chunks is None
        assert not filters
    else:
        expected = ZSTD_FILTER if compression == "zstd" else "gzip"
        assert filters == {"shuffle", expected}
        assert chunks == chunk_shape(result.ds["PX"].shape, 8)


def test_the_default_is_zstd_and_smaller(
    results: dict[str, ScanResult], tmp_path: Path
) -> None:
    result = results["ragged"]
    result.to_netcdf(tmp_path / "zstd.nc")
    result.to_netcdf(tmp_path / "none.nc", compression=None)
    assert _filters(tmp_path / "zstd.nc", "time")[1] == {"shuffle", ZSTD_FILTER}
    # the status is a number as well, the labels of the coordinates are not
    assert _filters(tmp_path / "zstd.nc", "status")[1] == {"shuffle", ZSTD_FILTER}
    assert (tmp_path / "zstd.nc").stat().st_size < 0.8 * (
        (tmp_path / "none.nc").stat().st_size
    )


def test_a_read_result_is_written_with_the_compression_asked_for(
    results: dict[str, ScanResult], tmp_path: Path
) -> None:
    """The encoding of a file which was read is not kept."""
    results["grid"].to_netcdf(tmp_path / "zstd.nc")
    read = ScanResult.from_netcdf(tmp_path / "zstd.nc")
    read.to_netcdf(tmp_path / "none.nc", compression=None)
    assert _filters(tmp_path / "none.nc", "PX") == (None, set())


def test_an_unknown_compression_raises(
    results: dict[str, ScanResult], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="zstd, gzip or None"):
        results["grid"].to_netcdf(tmp_path / "r.nc", compression="lz4")  # ty: ignore[invalid-argument-type]


def test_a_file_without_the_attributes_raises(tmp_path: Path) -> None:
    xr.Dataset({"y": ("x", [1.0, 2.0])}).to_netcdf(
        tmp_path / "plain.nc", engine="h5netcdf"
    )
    with pytest.raises(ValueError, match="no result of sbmlsim"):
        ScanResult.from_netcdf(tmp_path / "plain.nc")
    with pytest.raises(ValueError, match="no result of sbmlsim"):
        SensitivityResult.from_netcdf(tmp_path / "plain.nc")


def test_a_new_process_reads_a_compressed_result(
    results: dict[str, ScanResult], tmp_path: Path
) -> None:
    """Reading registers the filter of zstd, which nothing imported before."""
    path = tmp_path / "result.nc"
    results["ragged"].to_netcdf(path)
    code = (
        "import sys\n"
        "from sbmlsim.result import ScanResult\n"
        "assert 'hdf5plugin' not in sys.modules\n"
        f"result = ScanResult.from_netcdf({str(path)!r})\n"
        "print(float(result.ds['PX'].sum()))"
    )
    run = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert run.returncode == 0, run.stderr
    expected = float(results["ragged"].ds["PX"].sum())
    assert float(run.stdout.strip().splitlines()[-1]) == expected


@pytest.mark.parametrize(
    ("shape", "chunks"),
    [
        # whole timecourses, as many as fit
        ((5000, 203), (CHUNK_BYTES // 8 // 203, 203)),
        # fewer points than fit
        ((20, 51), (20, 51)),
        # a timecourse larger than a chunk is cut
        ((3, 300_000), (1, CHUNK_BYTES // 8)),
        # the dimension before the last fills up first
        ((4, 50, 1000), (2, 50, 1000)),
        ((7,), (7,)),
    ],
)
def test_the_chunks_are_whole_timecourses(
    shape: tuple[int, ...], chunks: tuple[int, ...]
) -> None:
    assert chunk_shape(shape, 8) == chunks


@pytest.fixture(scope="module")
def wide() -> ScanResult:
    """A result of more than `STACK_MIN` variables, which are gathered (#607)."""
    model = RoadrunnerSBMLModel(source=REPRESSILATOR_SBML)
    scan = Scan(
        Simulation(end=50),
        [Dimension("d", values={"n": np.linspace(1.5, 4.0, 10)})],
    )
    return Simulator(n_workers=1).run(model, scan, time=np.linspace(0, 50, 51))


def _variables(path: Path) -> set[str]:
    """Get the names the file carries, the coordinates of the dimensions apart."""
    with h5py.File(path) as f:
        return {name for name in f if isinstance(f[name], h5py.Dataset)}


def test_the_variables_of_a_wide_result_are_gathered_into_one_array(
    wide: ScanResult, tmp_path: Path
) -> None:
    path = tmp_path / "wide.nc"
    assert len(wide.ds.data_vars) >= STACK_MIN

    wide.to_netcdf(path)

    names = _variables(path)
    assert f"{STACK_PREFIX}0" in names, names
    assert not {str(name) for name in wide.ds.data_vars} & names


@pytest.mark.parametrize("compression", ["zstd", "gzip", None])
def test_a_gathered_result_is_read_as_it_was_written(
    wide: ScanResult, compression: str | None, tmp_path: Path
) -> None:
    path = tmp_path / "wide.nc"
    wide.to_netcdf(path, compression=compression)  # ty: ignore[invalid-argument-type]

    again = ScanResult.from_netcdf(path)

    xr.testing.assert_identical(again.ds, wide.ds)
    # the variables come back in the order they were written in
    assert list(again.ds.data_vars) == list(wide.ds.data_vars)


def test_the_array_of_a_gathered_result_carries_the_compression(
    wide: ScanResult, tmp_path: Path
) -> None:
    path = tmp_path / "wide.nc"
    wide.to_netcdf(path)

    chunks, filters = _filters(path, f"{STACK_PREFIX}0")

    assert filters == {"shuffle", ZSTD_FILTER}
    stacked = (len(wide.ds.data_vars), *wide.ds["PX"].shape)
    assert chunks == chunk_shape(stacked, 8)
    # this result fits into one chunk; an array which does not is cut so that a
    # chunk of it is a chunk of a single variable, as it was before #607
    assert chunk_shape((3301, 1000, 101), 8)[0] == 1


def test_a_result_of_few_variables_keeps_its_variables_named(
    wide: ScanResult, tmp_path: Path
) -> None:
    few = wide.ds[[str(name) for name in list(wide.ds.data_vars)[: STACK_MIN - 1]]]
    few.attrs = wide.ds.attrs
    path = tmp_path / "few.nc"

    write_netcdf(few, path)

    assert _variables(path) >= {str(name) for name in few.data_vars}
    xr.testing.assert_identical(read_netcdf(path), few)


def test_gathering_does_not_make_the_file_larger(
    wide: ScanResult, tmp_path: Path
) -> None:
    gathered = tmp_path / "gathered.nc"
    one_by_one = tmp_path / "one_by_one.nc"
    wide.to_netcdf(gathered)
    # the layout of the releases before #607, which `STACK_MIN` of a variable
    # count below the result's reproduces
    with mock.patch("sbmlsim.result.netcdf.STACK_MIN", len(wide.ds.data_vars) + 1):
        wide.to_netcdf(one_by_one)

    assert _variables(one_by_one) >= {str(name) for name in wide.ds.data_vars}
    assert gathered.stat().st_size <= one_by_one.stat().st_size
