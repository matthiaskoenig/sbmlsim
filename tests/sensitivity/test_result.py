"""The result of a sensitivity analysis and finding the design of a result."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from sbmlsim.result import ScanResult
from sbmlsim.sensitivity.indices import design_of
from sbmlsim.sensitivity.result import PARAMETER, SensitivityResult
from sbmlsim.simulation import Dimension, Formula, Scan, Simulation, sampling
from sbmlsim.simulation.sampling import Uniform
from sbmlsim.simulator import Simulator
from tests.simulator.models import sbml


def _result() -> SensitivityResult:
    ds = xr.Dataset(
        {
            "auc.ST": ((PARAMETER, "dose"), np.array([[0.6, 0.7], [0.05, 0.3]])),
            "auc.S1": ((PARAMETER, "dose"), np.array([[0.5, 0.6], [0.04, 0.2]])),
            "c.ST": ((PARAMETER, "dose", "time"), np.ones((2, 2, 3))),
        },
        coords={PARAMETER: ["k1", "k2"], "dose": [0, 1], "time": [0.0, 1.0, 2.0]},
        attrs={
            "units": {
                "auc.ST": "dimensionless",
                "auc.S1": "dimensionless",
                "c.ST": "dimensionless",
            },
            "method": "sobol",
            "options": {"n": 8},
        },
    )
    return SensitivityResult(ds)


def test_a_result_has_its_variables_and_metadata() -> None:
    s = _result()
    assert s.method == "sobol"
    assert s.parameters == ["k1", "k2"]
    assert s.observables == ["auc", "c"]
    assert "auc.ST" in s and s["auc.ST"].dims == (PARAMETER, "dose")


def test_index_stacks_the_scalar_observables() -> None:
    stacked = _result().index("ST")
    assert stacked.dims == (PARAMETER, "observable", "dose")
    assert stacked["observable"].values.tolist() == ["auc"]
    with pytest.raises(KeyError, match="nope"):
        _result().index("nope")


def test_classify_and_to_dataframe() -> None:
    classes = _result().classify("auc.ST")
    assert classes.sel({PARAMETER: "k1", "dose": 0}).item() == "high"
    assert classes.sel({PARAMETER: "k2", "dose": 0}).item() == "negligible"
    df = _result().to_dataframe("auc.ST")
    assert list(df.index) == ["k1", "k2"]


def test_a_result_survives_netcdf(tmp_path: Path) -> None:
    s = _result()
    path = tmp_path / "s.nc"
    s.to_netcdf(path)
    again = SensitivityResult.from_netcdf(path)
    xr.testing.assert_equal(again.ds, s.ds)
    assert again.method == "sobol" and again.units == s.units


def test_design_of_finds_the_dimension_of_a_method() -> None:
    model = Simulator().load(sbml())
    design = sampling.sobol(
        {"k1": Uniform(0.5, 1.0), "k2": Uniform(0.5, 1.0)}, 8, seed=1
    )
    doses = Dimension("dose", values={"a0": [1.0, 2.0]})
    res = Simulator().run(
        model,
        Scan(Simulation(end=1, steps=2), [doses, design]),
        [Formula("k", "max(k1)")],
    )
    dim, record = design_of(res, {"sobol"})
    assert dim == "sobol" and record.method == "sobol"
    with pytest.raises(ValueError, match="morris"):
        design_of(res, {"morris"})


def test_two_designs_need_dim() -> None:
    model = Simulator().load(sbml())
    first = sampling.local(["k1"], 0.1, model=model, id="a")
    second = sampling.local(["k2"], 0.1, model=model, id="b")
    res = Simulator().run(
        model,
        Scan(Simulation(end=1, steps=2), [first, second]),
        [Formula("k", "max(k1)")],
    )
    with pytest.raises(ValueError, match=r"'a'.*'b'|dim="):
        design_of(res, {"local"})
    assert design_of(res, {"local"}, dim="b")[0] == "b"
    assert isinstance(res, ScanResult)


def test_a_selection_of_one_parameter_keeps_the_dimension() -> None:
    s = _result()
    one = s.sel(parameter="k1")
    assert one.parameters == ["k1"]
    assert one["auc.ST"].dims == (PARAMETER, "dose")
    assert s.isel(parameter=1).parameters == ["k2"]
    assert s.sel(parameter=["k2"]).parameters == ["k2"]
    # a variable selects a scalar as in xarray
    assert s["auc.ST"].sel(parameter="k1", dose=0).item() == 0.6


def test_index_raises_for_a_named_observable_it_cannot_stack() -> None:
    s = _result()
    with pytest.raises(ValueError, match=r"'c'.*time"):
        s.index("ST", observables=["auc", "c"])
    with pytest.raises(ValueError, match=r"'x' has no index 'ST'"):
        s.index("ST", observables=["x"])
    with pytest.raises(ValueError, match="no observable"):
        s.index("ST", observables=[])
    # a time point of the timecourse is a scalar observable
    stacked = s.sel(time=1.0).index("ST", observables=["auc", "c"])
    assert stacked["observable"].values.tolist() == ["auc", "c"]


def test_classify_needs_a_dimensionless_index() -> None:
    ds = _result().ds.copy()
    ds["auc.mu_star"] = ds["auc.ST"]
    ds.attrs = {**ds.attrs, "units": {**ds.attrs["units"], "auc.mu_star": "mM"}}
    with pytest.raises(ValueError, match=r"auc\.mu_star.*mM"):
        SensitivityResult(ds).classify("auc.mu_star")
