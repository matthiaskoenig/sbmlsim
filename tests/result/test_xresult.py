"""Results keep the time points of every simulation, interpolation is on request."""

import warnings

import numpy as np
import pytest

from sbmlsim.result import TimecourseResult, XResult
from sbmlsim.simulation import Dimension, ScanSim, Simulation
from sbmlsim.units import UnitsInformation, ureg


def _tc(times: list[float], values: list[float]) -> TimecourseResult:
    """Get a result of the columns `time` and `y`."""
    return TimecourseResult(
        columns=("time", "y"), values=np.column_stack([times, values]).astype(float)
    )


def _scan() -> ScanSim:
    """Get a scan of two simulations."""
    return ScanSim(
        Simulation(end=4),
        dimensions=[
            Dimension("d", index=np.arange(2), changes={"k": np.array([1.0, 2.0])})
        ],
    )


def test_a_single_result_has_no_scan_dimension() -> None:
    """One simulation is one dimension, its points."""
    xres = XResult.from_timecourses([_tc([0, 1, 3], [0, 1, 3])])
    assert xres["y"].dims == ("_point",)
    np.testing.assert_allclose(xres["time"].values, [0, 1, 3])


def test_a_ragged_scan_is_padded() -> None:
    """Simulations with different time points are padded with NaN."""
    xres = XResult.from_timecourses(
        [_tc([0, 4], [0, 4]), _tc([0, 1, 4], [0, 2, 8])], scan=_scan()
    )
    assert xres["y"].dims == ("_point", "d")
    assert xres["y"].shape == (3, 2)
    assert np.isnan(xres["y"].values[2, 0])
    assert np.isnan(xres["time"].values[2, 0])


def test_interpolate_onto_a_common_grid() -> None:
    """Every simulation is interpolated onto the given times."""
    xres = XResult.from_timecourses(
        [_tc([0, 4], [0, 4]), _tc([0, 1, 4], [0, 2, 8])], scan=_scan()
    )
    grid = xres.interpolate([0, 2, 4])
    assert grid["y"].dims == ("_time", "d")
    np.testing.assert_allclose(grid["y"].values, [[0, 0], [2, 4.0], [4, 8]])
    np.testing.assert_allclose(grid["_time"].values, [0, 2, 4])


def test_interpolate_outside_of_a_simulation_is_nan() -> None:
    """A time outside of a simulation has no value."""
    xres = XResult.from_timecourses([_tc([0, 2], [0, 2])])
    grid = xres.interpolate([1, 3])
    assert grid["y"].values[0] == pytest.approx(1.0)
    assert np.isnan(grid["y"].values[1])


def test_dim_mean_interpolates() -> None:
    """The mean over the scan is taken on the given times or on all time points."""
    uinfo = UnitsInformation(udict={"y": "mM", "time": "s"}, ureg=ureg)
    xres = XResult.from_timecourses(
        [_tc([0, 4], [0, 4]), _tc([0, 1, 4], [0, 2, 8])], scan=_scan(), uinfo=uinfo
    )
    mean = xres.dim_mean("y", times=[4])
    np.testing.assert_allclose(mean.magnitude, [6.0])
    assert str(mean.units) == "millimolar"
    mean_all = xres.dim_mean("y")
    np.testing.assert_allclose(mean_all.magnitude, [0.0, 1.5, 6.0])
    np.testing.assert_allclose(xres.dim_max("y", times=[1]).magnitude, [2.0])


def test_to_dataframe_drops_the_padding() -> None:
    """The rows of the padding are not rows of the data frame."""
    xres = XResult.from_timecourses(
        [_tc([0, 4], [0, 4]), _tc([0, 1, 4], [0, 2, 8])], scan=_scan()
    )
    df = xres.to_dataframe()
    assert len(df) == 5
    assert not df["time"].isna().any()


def test_mean_dataframe_and_mean_of_the_time() -> None:
    """The mean data frame has the time, the mean of the time is the grid."""
    uinfo = UnitsInformation(udict={"y": "mM", "time": "s"}, ureg=ureg)
    xres = XResult.from_timecourses(
        [_tc([0, 4], [0, 4]), _tc([0, 1, 4], [0, 2, 8])], scan=_scan(), uinfo=uinfo
    )
    with warnings.catch_warnings():
        # the data frame holds the magnitudes, the units are not stripped
        warnings.simplefilter("error")
        df = xres.to_mean_dataframe()
    np.testing.assert_allclose(df["time"], [0, 1, 4])
    np.testing.assert_allclose(df["y"], [0, 1.5, 6])
    np.testing.assert_allclose(xres.dim_mean("time").magnitude, [0, 1, 4])


def test_a_reduction_interpolates_only_its_variable() -> None:
    """`dim_mean` of one variable does not interpolate the others."""
    xres = XResult.from_timecourses(
        [_tc([0, 4], [0, 4]), _tc([0, 1, 4], [0, 2, 8])], scan=_scan()
    )
    grid = xres.interpolate([0, 4], keys=["y"])
    assert set(grid.xds.data_vars) == {"y", "time"}
