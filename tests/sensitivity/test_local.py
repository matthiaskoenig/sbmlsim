"""The local sensitivity analysis."""

import numpy as np
import pytest

from sbmlsim import sensitivity
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.result import ScanResult
from sbmlsim.sensitivity.result import PARAMETER
from sbmlsim.simulation import Dimension, Formula, Scan, Simulation, sampling
from sbmlsim.simulator import Simulator
from sbmlsim.units import ureg
from tests.sensitivity.models import POWER_LAW
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
    s = sensitivity.local(res, observables=["cmax"])
    unit = res.units["cmax"]
    assert unit
    assert s.units["cmax.normalized"] == "dimensionless"
    assert s.units["cmax.raw"] == str(ureg.Unit(unit) / ureg.Unit("1/hr"))
    mixed = sensitivity.local(_pk_result(["ke", "PODOSE"]), observables=["cmax"])
    assert mixed.units["cmax.raw"] == ""
    assert mixed.units["cmax.normalized"] == "dimensionless"


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


def test_a_zero_parameter_has_no_raw_sensitivity() -> None:
    model = Simulator().load(sbml("model zero\n  a = 0; b = 3\n  y := a + b\nend\n"))
    design = sampling.local(["a", "b"], 0.1, model=model)
    res = Simulator().run(
        model, Scan(Simulation(end=1, steps=1), [design]), [Formula("y_max", "max(y)")]
    )
    s = sensitivity.local(res)
    assert np.isnan(s["y_max.raw"].sel(parameter="a").values).all()
    assert np.isfinite(s["y_max.raw"].sel(parameter="b").values).all()


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
