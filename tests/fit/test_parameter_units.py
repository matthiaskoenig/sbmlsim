"""A parameter in a unit id of the model through a fit and its designs (#268).

`per_min` is a unit id of the HCTZ model, `1/min`. The parameter keeps the
unit it was defined with, the values of the fit (the parameter sets, the
Fisher information, the profiles) carry the unit of pint, which the designs
of `sbmlsim.simulation.sampling` read without a model.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from sbmlsim import Q
from sbmlsim.fit import FitParameter, FitSettings
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.fit.fisher import fisher_information
from sbmlsim.fit.identifiability import ProfileSettings, profile_likelihood
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.fit.runner import run_optimization
from sbmlsim.simulation.sampling import fit_parameters, fit_repeats, profile_parameters

#: the unit of pint of `per_min`
PER_MINUTE = Q(1.0, "1/min").units


@pytest.fixture
def problem_per_min(
    definition_hctz_pk: FitDefinition, fit_settings: FitSettings
) -> OptimizationProblem:
    """Get the initialized PK problem with `Ka_dis_hctz` in `per_min`."""
    ka = FitParameter(
        pid="Ka_dis_hctz",
        start_value=0.01,
        lower_bound=1e-4,
        upper_bound=10,
        unit="per_min",
    )
    definition = replace(
        definition_hctz_pk, parameters=[ka, *definition_hctz_pk.parameters[1:]]
    )
    problem = definition.problem(opid="hctz_per_min")
    problem.initialize(fit_settings)
    return problem


def test_the_sets_of_a_problem_have_the_units_of_pint(
    problem_per_min: OptimizationProblem,
) -> None:
    """The units of the values are units of pint, the parameter keeps its unit."""
    assert problem_per_min.parameter_units["Ka_dis_hctz"] == "1/min"
    assert problem_per_min.parameter_set_model().units["Ka_dis_hctz"] == "1/min"
    # the unit as it was defined, for the display and the export
    assert problem_per_min.parameters[0].unit == "per_min"


def test_the_fisher_design_takes_a_unit_id_of_the_model(
    problem_per_min: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """`fit_parameters` draws in the unit of pint of a parameter in a unit id."""
    fisher = fisher_information(
        problem_per_min, fit_settings, problem_per_min.parameter_set_model()
    )
    assert fisher.units[0] == "1/min"
    dimension = fit_parameters(fisher, 4, seed=1)
    assert dimension.values["Ka_dis_hctz"].units == PER_MINUTE


def test_the_repeats_design_takes_a_unit_id_of_the_model(
    problem_per_min: OptimizationProblem,
    fit_settings: FitSettings,
    short_fit: dict[str, Any],
    tmp_path: Path,
) -> None:
    """`fit_repeats` reads the units of the result, also of one read back."""
    result = run_optimization(
        problem=problem_per_min,
        settings=fit_settings,
        size=1,
        serial=True,
        show_progress=False,
        runs_dir=tmp_path,
        **short_fit,
    )
    assert result.units["Ka_dis_hctz"] == "1/min"
    assert result.parameter_set().units["Ka_dis_hctz"] == "1/min"
    dimension = fit_repeats(result, 1)
    assert dimension.values["Ka_dis_hctz"].units == PER_MINUTE

    stored = OptimizationResult.from_directory(tmp_path)
    assert stored.units["Ka_dis_hctz"] == "1/min"
    assert fit_repeats(stored, 1).values["Ka_dis_hctz"].units == PER_MINUTE


def test_the_profile_design_takes_a_unit_id_of_the_model(
    problem_per_min: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """`profile_parameters` draws in the unit of pint of a parameter in a unit id."""
    identifiability = profile_likelihood(
        problem=problem_per_min,
        settings=fit_settings,
        parameter_set=problem_per_min.parameter_set_model(),
        profile_settings=ProfileSettings(
            reoptimize=False, initial_step=0.5, min_step=0.1, max_step=2.0, max_points=6
        ),
        pids=["Ka_dis_hctz"],
        serial=True,
        show_progress=False,
    )
    assert identifiability.parameter_set.units["Ka_dis_hctz"] == "1/min"
    dimension = profile_parameters(identifiability, 4, seed=1)
    assert dimension.values["Ka_dis_hctz"].units == PER_MINUTE
    assert dimension.design is not None
    assert dimension.design.references["Ka_dis_hctz"]["unit"] == "1/min"
