"""A parameter in a unit id of the model through a fit and its designs (#268).

`per_min` is a unit id of the HCTZ model, `1/min`. The parameter keeps the
unit it was defined with, the values of the fit (the parameter sets, the
Fisher information, the profiles) carry the unit of pint, which the designs
of `sbmlsim.simulation.sampling` read without a model.
"""

import re
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from sbmlsim import Q
from sbmlsim.fit import FitParameter, FitSettings
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.fit.fisher import fisher_information
from sbmlsim.fit.identifiability import ProfileSettings, profile_likelihood
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.parameters import ParameterSet, ParameterSets
from sbmlsim.fit.petab_v2.likelihood import log_prior
from sbmlsim.fit.report import FitReport
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


def test_the_tables_of_a_report_show_the_unit_as_defined(
    problem_per_min: OptimizationProblem, fit_settings: FitSettings, tmp_path: Path
) -> None:
    """Every table of a report shows `per_min`, the values keep `1/min`."""
    pset = problem_per_min.parameter_set_model()
    identifiability = profile_likelihood(
        problem=problem_per_min,
        settings=fit_settings,
        parameter_set=pset,
        profile_settings=ProfileSettings(
            reoptimize=False, initial_step=0.5, min_step=0.1, max_step=2.0, max_points=6
        ),
        pids=["Ka_dis_hctz"],
        serial=True,
        show_progress=False,
    )
    fisher = fisher_information(problem_per_min, fit_settings, pset)
    report = FitReport(
        problem=problem_per_min,
        settings=fit_settings,
        parameter_sets=ParameterSets([pset]),
        identifiability=identifiability,
        fisher=fisher,
        mapping_figures=False,
    )
    results_dir = report.create(output_dir=tmp_path, name="report")

    # the parameters and the profiles of the text report
    text = (results_dir / "report.txt").read_text(encoding="utf-8")
    assert re.search(r"Ka_dis_hctz\s+per_min\s", text)
    assert re.search(r"Ka_dis_hctz\s+\S+\s+per_min\s", text)
    assert not any(
        "1/min" in line for line in text.splitlines() if "Ka_dis_hctz" in line
    )
    for name in ["fisher.tsv", "identifiability.tsv"]:
        table = pd.read_csv(results_dir / name, sep="\t").set_index("parameter")
        assert table.loc["Ka_dis_hctz", "unit"] == "per_min", name
    # the parameters, the Fisher information, the correlation and the
    # profiles of the HTML report, the correlation has no unit
    html = (results_dir / "index.html").read_text(encoding="utf-8")
    rows = [
        row
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)
        if re.search(r"<td[^>]*>Ka_dis_hctz</td>", row)
    ]
    assert len(rows) == 4
    assert sum("<td>per_min</td>" in row for row in rows) == 3
    assert not any("1/min" in row for row in rows)
    # the values carry the units of pint
    assert fisher.units[0] == "1/min"
    assert identifiability.parameter_set.units["Ka_dis_hctz"] == "1/min"


def test_the_prior_of_an_uninitialized_problem_takes_the_sets_of_a_fit(
    problem_per_min: OptimizationProblem,
) -> None:
    """A set in `1/min` is in the unit of a parameter in `per_min`.

    An uninitialized problem has not resolved the unit ids of its models, the
    sets of a fit carry the resolved unit of every parameter.
    """
    uninitialized = deepcopy(problem_per_min)
    assert not uninitialized.is_initialized
    pset = problem_per_min.parameter_set_model()
    assert pset.units["Ka_dis_hctz"] == "1/min"
    expected = log_prior(problem_per_min, pset)
    assert log_prior(uninitialized, pset) == expected
    # the unit as defined and another unit of pint of the other parameters
    defined = ParameterSet(
        "defined",
        {**pset.values, "GU__HCTZABS_k": 60.0 * pset.values["GU__HCTZABS_k"]},
        {**pset.units, "Ka_dis_hctz": "per_min", "GU__HCTZABS_k": "1/hr"},
    )
    assert log_prior(uninitialized, defined) == pytest.approx(expected)
