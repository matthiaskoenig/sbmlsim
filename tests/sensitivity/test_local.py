"""The local sensitivity analysis."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from matplotlib.colors import to_rgba

from sbmlsim import sensitivity
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.result import ScanResult
from sbmlsim.sensitivity.result import PARAMETER
from sbmlsim.simulation import Custom, Dimension, Formula, Scan, Simulation, sampling
from sbmlsim.simulator import Simulator
from sbmlsim.units import ureg
from tests.sensitivity.models import CHAIN, POWER_LAW, s2_end_fails_for_high_s1_and_k1
from tests.simulator.models import sbml, sbml_pk


@pytest.fixture(scope="module")
def model() -> RoadrunnerSBMLModel:
    return Simulator().load(sbml(POWER_LAW))


def test_the_normalized_sensitivities_are_the_exponents(
    model: RoadrunnerSBMLModel,
) -> None:
    design = sampling.local(["a", "b", "c"], 0.001, model=model)
    res = Simulator().run(
        model,
        Scan(Simulation(end=1, steps=1), [design]),
        [Formula("y_max", "max(y)")],
    )
    s = sensitivity.local(res)
    np.testing.assert_allclose(
        s["y_max.normalized"].values, [2.0, -1.0, 0.5], rtol=1e-5
    )
    assert s.parameters == ["a", "b", "c"]
    # raw: dy/dp = normalized * y / p
    y = 2.0**2 / 3.0 * 4.0**0.5
    np.testing.assert_allclose(
        s["y_max.raw"].values, [2.0 * y / 2.0, -y / 3.0, 0.5 * y / 4.0], rtol=1e-5
    )
    assert s.method == "local"


def test_the_design_dimension_need_not_be_first(model: RoadrunnerSBMLModel) -> None:
    design = sampling.local(["b"], 0.001, model=model)
    scale = Dimension("scale", values={"c": [4.0, 16.0]})
    res = Simulator().run(
        model,
        Scan(Simulation(end=1, steps=1), [scale, design]),
        [Formula("y_max", "max(y)")],
    )
    s = sensitivity.local(res)
    assert s["y_max.normalized"].dims == (PARAMETER, "scale")
    np.testing.assert_allclose(s["y_max.normalized"].values, [[-1.0, -1.0]], rtol=1e-5)


def test_a_timecourse_has_indices_per_time(model: RoadrunnerSBMLModel) -> None:
    design = sampling.local(["a"], 0.001, model=model)
    res = Simulator().run(
        model, Scan(Simulation(end=1, steps=2), [design]), [Formula("yt", "y")]
    )
    s = sensitivity.local(res)
    assert s["yt.normalized"].dims == (PARAMETER, "time")
    np.testing.assert_allclose(s["yt.normalized"].values, 2.0, rtol=1e-5)


def test_a_ragged_timecourse_raises() -> None:
    model = Simulator().load(sbml())
    design = sampling.local(["k1"], 0.1, model=model)
    res = Simulator().run(
        model, Scan(Simulation(end=1), [design]), [Formula("a", "[A]")]
    )
    with pytest.raises(ValueError, match="time="):
        sensitivity.local(res)


def _pk_result(targets: list[str]) -> ScanResult:
    model = Simulator().load(sbml_pk())
    design = sampling.local(targets, 0.01, model=model)
    return Simulator().run(
        model,
        Scan(Simulation(end=4, steps=4), [design]),
        [Formula("cmax", "max([C])"), Formula("c_end", "[C]")],
    )


def test_the_units_of_the_indices() -> None:
    res = _pk_result(["ka", "ke"])
    s = sensitivity.local(res)
    assert res.units["cmax"] == "milligram / liter"
    assert s.units["cmax.normalized"] == "dimensionless"
    # the short form of the unit of the observable per unit of the parameter
    assert s.units["cmax.raw"] == "h*mg/l"
    assert ureg.Unit(s.units["cmax.raw"]) == ureg.Unit("mg/l") / ureg.Unit("1/hr")
    # the coordinates keep their units
    assert s.units["time"] == res.units["time"] == "hr"
    assert s.units[PARAMETER] == ""
    mixed = sensitivity.local(_pk_result(["ke", "PODOSE"]), observables=["cmax"])
    assert mixed.units["cmax.raw"] == ""
    assert mixed.units["cmax.normalized"] == "dimensionless"


def test_the_coordinates_of_other_dimensions_are_kept(
    model: RoadrunnerSBMLModel,
) -> None:
    design = sampling.local(["b"], 0.001, model=model)
    scale = Dimension("scale", values={"c": [4.0, 16.0]}, labels=["low", "high"])
    res = Simulator().run(
        model,
        Scan(Simulation(end=1, steps=1), [scale, design]),
        [Formula("y_max", "max(y)")],
    )
    s = sensitivity.local(res)
    assert s.ds["c"].dims == ("scale",)
    assert s.ds["c"].values.tolist() == [4.0, 16.0]
    assert s.units["c"] == res.units["c"]
    assert s.units["scale"] == ""
    # the targets of the design are no coordinates of the indices
    assert "b" not in s.ds.coords


def test_a_zero_observable_has_no_normalized_sensitivity() -> None:
    model = Simulator().load(sbml(POWER_LAW))
    design = sampling.local(["a"], 0.1, model=model)
    res = Simulator().run(
        model,
        Scan(Simulation(end=1, steps=1), [design]),
        [Formula("z", "0 * y"), Formula("y_max", "max(y)")],
    )
    s = sensitivity.local(res)
    assert np.isnan(s["z.normalized"].values).all()
    np.testing.assert_allclose(s["z.raw"].values, 0.0)
    assert np.isfinite(s["y_max.normalized"].values).all()


def test_a_zero_parameter_has_no_sensitivity() -> None:
    """A parameter whose reference is zero is never moved: its indices are undefined."""
    model = Simulator().load(
        sbml("model zero\n  a = 0; b = 3\n  y := (1 + a) * b\nend\n")
    )
    design = sampling.local(["a", "b"], 0.1, model=model)
    res = Simulator().run(
        model, Scan(Simulation(end=1, steps=1), [design]), [Formula("y_max", "max(y)")]
    )
    s = sensitivity.local(res)
    for index in ("raw", "normalized"):
        assert np.isnan(s[f"y_max.{index}"].sel(parameter="a").item())
        assert np.isfinite(s[f"y_max.{index}"].sel(parameter="b").item())
    assert s.classify("y_max.normalized").values.tolist() == ["", "high"]
    figure = sensitivity.plot_heatmap(s, "normalized")
    (ax,) = [ax for ax in figure.axes if ax.get_label() == "heatmap"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b"]
    grey = [
        p for p in ax.patches if np.allclose(p.get_facecolor(), to_rgba("lightgrey"))
    ]
    assert len(grey) == 1


def test_observables_restrict_the_variables(model: RoadrunnerSBMLModel) -> None:
    design = sampling.local(["a"], 0.001, model=model)
    res = Simulator().run(
        model,
        Scan(Simulation(end=1, steps=1), [design]),
        [Formula("y_max", "max(y)"), Formula("y_min", "min(y)")],
    )
    s = sensitivity.local(res, observables=["y_min"])
    assert set(s.ds.data_vars) == {"y_min.raw", "y_min.normalized"}


def test_an_unknown_observable_raises(model: RoadrunnerSBMLModel) -> None:
    design = sampling.local(["a"], 0.001, model=model)
    res = Simulator().run(
        model, Scan(Simulation(end=1, steps=1), [design]), [Formula("y_max", "max(y)")]
    )
    with pytest.raises(ValueError, match="nope"):
        sensitivity.local(res, observables=["nope"])


def test_dim_chooses_one_of_two_designs(model: RoadrunnerSBMLModel) -> None:
    first = sampling.local(["a"], 0.001, model=model, id="first")
    second = sampling.local(["b"], 0.001, model=model, id="second")
    res = Simulator().run(
        model,
        Scan(Simulation(end=1, steps=1), [first, second]),
        [Formula("y_max", "max(y)")],
    )
    with pytest.raises(ValueError, match="dim="):
        sensitivity.local(res)
    s = sensitivity.local(res, dim="second")
    assert s.parameters == ["b"]
    assert s["y_max.normalized"].dims == (PARAMETER, "first")
    np.testing.assert_allclose(s["y_max.normalized"].values, [[-1.0] * 3], rtol=1e-5)
    with pytest.raises(ValueError, match="third"):
        sensitivity.local(res, dim="third")


def _failing_chain(design: Dimension) -> ScanResult:
    """Run the chain under two conditions; a point of the high one fails for k1 > 1."""
    model = Simulator().load(sbml(CHAIN))
    conditions = Dimension(
        "S1_0", values={"[S1]": [1.0, 10.0]}, labels=["reference", "high"]
    )
    return Simulator(n_workers=1).run(
        model,
        Scan(Simulation(end=2, steps=2), [conditions, design]),
        [
            Formula("s2_max", "max([S2])"),
            Custom(
                "s2_end",
                s2_end_fails_for_high_s1_and_k1,
                "dimensionless",
                symbols=["[S1]", "[S2]", "k1"],
            ),
        ],
        on_error="flag",
    )


def test_a_failed_point_gives_nan_indices_and_one_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = Simulator().load(sbml(CHAIN))
    res = _failing_chain(sampling.local(["k1", "k2"], 0.01, model=model))
    # only the point k1+ of the high condition failed
    assert res["status"].values.sum() == 1
    caplog.clear()
    s = sensitivity.local(res)
    for name in ("s2_max", "s2_end"):
        for index in ("raw", "normalized"):
            values = s[f"{name}.{index}"]
            assert np.isnan(values.sel(parameter="k1", S1_0="high").item())
            assert np.isfinite(values.sel(parameter="k2", S1_0="high").item())
            assert np.isfinite(values.sel(S1_0="reference").values).all()
    records = [r for r in caplog.records if r.name == "sbmlsim.sensitivity.indices"]
    assert len(records) == 1
    assert records[0].levelname == "WARNING"
    assert records[0].getMessage().startswith("2 elements")


def test_a_cut_result_raises(model: RoadrunnerSBMLModel) -> None:
    design = sampling.local(["a", "b"], 0.001, model=model)
    res = Simulator().run(
        model, Scan(Simulation(end=1, steps=1), [design]), [Formula("y_max", "max(y)")]
    )
    with pytest.raises(ValueError, match="was cut"):
        sensitivity.local(res.isel(local=[0, 1, 2]))


def test_a_stored_result_is_analysed(
    model: RoadrunnerSBMLModel, tmp_path: Path
) -> None:
    design = sampling.local(["a", "b", "c"], 0.001, model=model)
    res = Simulator().run(
        model, Scan(Simulation(end=1, steps=1), [design]), [Formula("y_max", "max(y)")]
    )
    path = tmp_path / "r.nc"
    res.to_netcdf(path)
    again = sensitivity.local(ScanResult.from_netcdf(path))
    xr.testing.assert_allclose(again.ds, sensitivity.local(res).ds)
    assert again.units == sensitivity.local(res).units
