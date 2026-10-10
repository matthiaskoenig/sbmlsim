"""The PEtab export refuses scalar observables and scan tasks."""

from pathlib import Path

import pytest

from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import FitMappingCollection, FitParameter
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings, ParameterScaleType
from sbmlsim.fit.petab_v2 import PetabExporter
from sbmlsim.fit.petab_v2.gaps import gap_mappings, gaps_of_problem
from tests.fit.scalar_experiment import DoseStudy, ScalarStudy


def _problem(
    experiment: type[SimulationExperiment], mapping: str, tmp_path: Path
) -> OptimizationProblem:
    problem = OptimizationProblem(
        "scalar",
        [FitMappingCollection(experiment=experiment, mappings=[mapping])],
        [
            FitParameter(
                pid="V", lower_bound=1.0, upper_bound=100.0, start_value=30.0, unit="l"
            )
        ],
        base_path=tmp_path,
        data_path=tmp_path,
    )
    problem.initialize(FitSettings(parameter_scale=ParameterScaleType.LINEAR))
    return problem


def test_scalar_observable_gap(tmp_path: Path) -> None:
    problem = _problem(ScalarStudy, "fm_cmax", tmp_path)
    ids = {gap.id for gap in gaps_of_problem(problem)}
    assert "scalar-observable" in ids
    assert "scan-task" not in ids
    assert "x-observable" not in ids
    assert gap_mappings(problem)["scalar-observable"] == ["ScalarStudy.fm_cmax"]


def test_scan_task_gap(tmp_path: Path) -> None:
    problem = _problem(DoseStudy, "fm_dose", tmp_path)
    ids = {gap.id for gap in gaps_of_problem(problem)}
    assert {"scalar-observable", "scan-task"} <= ids
    assert "x-observable" not in ids
    assert gap_mappings(problem)["scan-task"] == ["DoseStudy.fm_dose"]


def test_export_names_the_mappings(tmp_path: Path) -> None:
    problem = _problem(ScalarStudy, "fm_cmax", tmp_path)
    with pytest.raises(ValueError) as error:
        PetabExporter(problem).to_problem()
    message = str(error.value)
    for text in ("scalar-observable", "fm_cmax", "ScalarStudy"):
        assert text in message


def test_hctz_has_no_new_gap(op_hctz_pk: OptimizationProblem) -> None:
    op_hctz_pk.initialize(FitSettings())
    ids = {gap.id for gap in gaps_of_problem(op_hctz_pk)}
    assert not ids & {"scalar-observable", "scan-task"}
    PetabExporter(op_hctz_pk).check()
