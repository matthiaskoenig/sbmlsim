"""The local sensitivity analysis."""

import numpy as np
import pytest

from sbmlsim import sensitivity
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.sensitivity.result import PARAMETER
from sbmlsim.simulation import Dimension, Formula, Scan, Simulation, sampling
from sbmlsim.simulator import Simulator
from tests.sensitivity.models import POWER_LAW
from tests.simulator.models import sbml


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
