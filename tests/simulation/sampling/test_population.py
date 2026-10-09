"""Virtual populations."""

from typing import Literal

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.simulation import Dimension, Formula, Scan, Simulation
from sbmlsim.simulation.sampling import Normal, Truncated, population
from sbmlsim.simulator import Simulator
from tests.simulator.models import (
    clearance_of,
    covariate_as_target,
    no_mapping,
    sbml,
    wrong_length,
)

COVARIATES = {
    "BW": Truncated(Normal(Q(70.0, "kg"), Q(15.0, "kg")), lower=Q(40.0, "kg"))
}


def test_a_population_maps_its_covariates_to_targets() -> None:
    dimension = population(clearance_of, COVARIATES, 200, seed=1)
    bw = np.asarray(dimension.coordinates["BW"].magnitude)
    assert bw.min() >= 40.0
    np.testing.assert_allclose(dimension.values["k1"], 0.8 * (bw / 70.0) ** 0.75)
    design = dimension.design
    assert design is not None
    assert design.method == "population"
    assert design.options["function"] == "tests.simulator.models:clearance_of"
    assert design.options["covariates"] == ["BW"]
    lhs = population(clearance_of, COVARIATES, 50, seed=1, method="lhs").design
    assert lhs is not None
    assert lhs.options["method"] == "lhs"


def test_the_result_has_the_covariates_as_coordinates() -> None:
    dimension = population(clearance_of, COVARIATES, 5, seed=2)
    res = Simulator().run(
        sbml(), Scan(Simulation(end=1, steps=2), [dimension]), [Formula("k", "k1")]
    )
    assert res.ds["BW"].dims == ("population",)
    assert res.units["BW"] == "kilogram"


def test_a_wrong_population_function_raises() -> None:
    with pytest.raises(ValueError, match="mapping"):
        population(no_mapping, COVARIATES, 5, seed=1)
    with pytest.raises(ValueError, match="length"):
        population(wrong_length, COVARIATES, 5, seed=1)
    with pytest.raises(ValueError, match="covariate"):
        population(covariate_as_target, COVARIATES, 5, seed=1)
    with pytest.raises(ValueError, match="module"):
        population(lambda c: {"k1": c["BW"]}, COVARIATES, 5, seed=1)
    with pytest.raises(ValueError, match="method"):
        population(clearance_of, COVARIATES, 5, seed=1, method="sobol")


def _hiding_cases() -> dict[str, list[Dimension]]:
    ones = np.ones(2)
    return {
        "a selection": [Dimension("d", values={"k2": ones}, coordinates={"k1": ones})],
        "a dimension id": [
            Dimension("d", values={"k2": ones}, coordinates={"e": ones}),
            Dimension("e", values={"k1": ones}),
        ],
        "a changed target": [
            Dimension("d", values={"k2": ones}),
            Dimension("e", values={"k1": ones}, coordinates={"k2": ones}),
        ],
        "time": [Dimension("d", values={"k2": ones}, coordinates={"time": ones})],
        "status": [Dimension("d", values={"k2": ones}, coordinates={"status": ones})],
        "_point": [Dimension("d", values={"k2": ones}, coordinates={"_point": ones})],
        "a coordinate": [
            Dimension("d", values={"k2": ones}, coordinates={"w": ones}),
            Dimension("e", values={"k1": ones}, coordinates={"w": ones}),
        ],
    }


@pytest.mark.parametrize("case", list(_hiding_cases()))
@pytest.mark.parametrize("on_error", ["raise", "flag"])
def test_a_coordinate_never_hides_or_is_hidden_by_a_name(
    case: str, on_error: Literal["raise", "flag"]
) -> None:
    scan = Scan(Simulation(end=1, steps=2), _hiding_cases()[case])
    with pytest.raises(ValueError, match="hide"):
        Simulator().run(sbml(), scan, on_error=on_error)
