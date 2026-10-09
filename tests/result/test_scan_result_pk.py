"""A result hands a PK analysis and a timecourse over to pkpdutils."""

import numpy as np
import pkpdutils as pk
import pytest

from sbmlsim import Q
from sbmlsim.simulation import PK, Change, Dimension, Formula, Scan, Simulation
from sbmlsim.simulator import Simulator
from tests.simulator.models import sbml_pk

OBSERVABLES = [Formula("conc", "[C]"), PK("c", "[C]", dose="PODOSE", route="oral")]


@pytest.fixture(scope="module")
def pk_sbml() -> str:
    return sbml_pk()


def simulation(steps: int | None = 480) -> Simulation:
    return Simulation(
        end=48, steps=steps, changes=[Change(0, {"PODOSE": Q(100, "mg")})]
    )


def scan(steps: int | None = 480) -> Scan:
    return Scan(
        simulation(steps),
        [Dimension("dose", values={"PODOSE": Q([50.0, 100.0], "mg")})],
    )


def test_nca_is_the_analysis_of_pkpdutils(pk_sbml: str) -> None:
    res = Simulator().run(pk_sbml, scan(), OBSERVABLES)
    nca = res.nca("c")
    assert isinstance(nca, pk.NCAResult)
    assert nca.sample_dims == ("dose",)
    np.testing.assert_array_equal(nca["cmax"].values, res["c.cmax"].values)
    assert nca.units("cmax") == res.units["c.cmax"]
    assert nca["flags"].dtype.kind == "i"


def test_to_timecourses_hands_over_a_timecourse(pk_sbml: str) -> None:
    res = Simulator().run(pk_sbml, scan(), OBSERVABLES)
    timecourses = res.to_timecourses(
        "conc", dose={"amount": np.array([50.0, 100.0]), "unit": "mg"}, route="oral"
    )
    again = pk.nca(timecourses)
    np.testing.assert_allclose(
        again["auc_inf_obs"].values, res["c.auc_inf_obs"].values, rtol=1e-9
    )


def test_a_ragged_result_and_a_single_simulation(pk_sbml: str) -> None:
    ragged = Simulator().run(pk_sbml, scan(steps=None), OBSERVABLES)
    assert ragged.ragged
    timecourses = ragged.to_timecourses("conc")
    assert pk.nca(timecourses)["cmax"].shape == (2,)
    single = Simulator().run(pk_sbml, simulation(), OBSERVABLES)
    assert single.nca("c").sample_dims == ()
    assert float(pk.nca(single.to_timecourses("conc"))["cmax"]) == pytest.approx(
        float(single["c.cmax"])
    )


def test_nca_needs_the_flags_and_to_timecourses_a_timecourse(pk_sbml: str) -> None:
    res = Simulator().run(pk_sbml, scan(), OBSERVABLES, keep=["conc", "c.cmax"])
    with pytest.raises(KeyError, match=r"c\.flags"):
        res.nca("c")
    with pytest.raises(ValueError, match=r"c\.cmax"):
        res.to_timecourses("c.cmax")
