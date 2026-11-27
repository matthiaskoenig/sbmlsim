"""The chart builder of sbmlsim, `sbmlsim_core.chart` with the units of sbmlsim.

The builder and its tests are in sbmlsim-core; these tests hold what sbmlsim
adds: the names of `sbmlsim.plot.chart` and the quantities of its registry.
"""

import numpy as np
import pint
import pytest
import sbmlsim_core.chart
import xarray as xr

import sbmlsim.plot.chart
from sbmlsim.plot import Chart, ChartAxis
from sbmlsim.units import Q, define_unit, ureg


def test_the_names_of_sbmlsim_core() -> None:
    """`sbmlsim.plot.chart` is the chart builder of sbmlsim-core."""
    for name in sbmlsim.plot.chart.__all__:
        assert getattr(sbmlsim.plot.chart, name) is getattr(sbmlsim_core.chart, name)
    assert Chart is sbmlsim_core.chart.Chart
    assert "Chart" in dir(sbmlsim.plot.chart)
    with pytest.raises(AttributeError, match="no attribute 'Nothing'"):
        _ = sbmlsim.plot.chart.Nothing


def test_quantities_of_the_registry_of_sbmlsim() -> None:
    """A quantity of sbmlsim is converted by its registry, its own units included."""
    unit = define_unit("IU_chart_test = 0.5 * microgram")
    chart = Chart()
    plot = chart.plot(x=ChartAxis(label="time", unit="hr"), y=ChartAxis(unit="µg"))
    plot.curve(Q([30, 90], "min"), Q([2.0, 4.0], unit))
    plot.curve(Q([1.0], "hr"), Q([1.0], "count") * Q(1.0, "µg"))
    spec = chart.spec()["plots"][0]
    assert spec["elements"][0]["x"] == [0.5, 1.5]
    assert spec["elements"][0]["y"] == pytest.approx([1.0, 2.0])
    assert (spec["x"]["unit"], spec["y"]["unit"]) == ("h", "μg")
    with pytest.raises(pint.DimensionalityError, match="axis x"):
        plot.curve(Q([1.0], "mM"), [1.0])


def test_the_units_of_a_result() -> None:
    """A data array of a result gives an axis the short form of its unit."""
    chart = Chart()
    plot = chart.plot(y=ChartAxis(label="cp", unit="mM"))
    time = xr.DataArray(np.array([0.0, 60.0]), name="time", attrs={"units": "minute"})
    cp = xr.DataArray(np.array([1.0, 2.0]), attrs={"units": "micromole / liter"})
    plot.curve(time, cp)
    spec = chart.spec()["plots"][0]
    assert (spec["x"]["label"], spec["x"]["unit"]) == ("time", "min")
    assert spec["elements"][0]["y"] == pytest.approx([1e-3, 2e-3])
    assert str(ureg.Unit("mM")) == "millimolar"
