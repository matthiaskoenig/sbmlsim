"""The PEtab export refuses scalar observables and scan tasks."""

from pathlib import Path

import pytest

from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import FitData, FitMapping, FitMappingCollection, FitParameter
from sbmlsim.fit.objects import MappingKind
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings, ParameterScaleType
from sbmlsim.fit.petab_v2 import PetabExporter
from sbmlsim.fit.petab_v2.gaps import gap_mappings, gaps_of_problem
from tests.fit.scalar_experiment import DoseStudy, MixedStudy, ScalarStudy


class ScanPointStudy(MixedStudy):
    """The timecourse of the point of 100 mg of the scan over the doses."""

    def fit_mappings(self) -> dict[str, FitMapping]:
        mappings = super().fit_mappings()
        mappings["fm_tc_scan"] = FitMapping(
            self,
            reference=FitData(self, dataset="tc", xid="time", yid="C"),
            observable=FitData(
                self, task="task_doses", xid="time", yid="[C]", sel={"dose": 1}
            ),
        )
        return mappings


def _problem(
    experiment: type[SimulationExperiment], mapping: str, tmp_path: Path
) -> OptimizationProblem:
    return _problem_of(
        [FitMappingCollection(experiment=experiment, mappings=[mapping])], tmp_path
    )


def _problem_of(
    collections: list[FitMappingCollection], tmp_path: Path
) -> OptimizationProblem:
    problem = OptimizationProblem(
        "scalar",
        collections,
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


def test_a_timecourse_of_a_scan_point_is_a_scan_task(tmp_path: Path) -> None:
    problem = _problem(ScanPointStudy, "fm_tc_scan", tmp_path)
    ids = {gap.id for gap in gaps_of_problem(problem)}
    assert "scan-task" in ids
    assert "scalar-observable" not in ids
    with pytest.raises(ValueError, match="scan-task") as error:
        PetabExporter(problem).check()
    assert "ScanPointStudy.fm_tc_scan" in str(error.value)


def test_export_names_every_mapping_of_a_gap(tmp_path: Path) -> None:
    problem = _problem_of(
        [
            FitMappingCollection(experiment=ScalarStudy, mappings=["fm_cmax"]),
            FitMappingCollection(experiment=DoseStudy, mappings=["fm_dose"]),
        ],
        tmp_path,
    )
    with pytest.raises(ValueError) as error:
        PetabExporter(problem).check()
    message = str(error.value)
    assert "mappings: ScalarStudy.fm_cmax, DoseStudy.fm_dose" in message
    assert "mappings: DoseStudy.fm_dose" in message


def test_the_gaps_of_an_export_are_the_ones_of_its_kinds(tmp_path: Path) -> None:
    """A scalar mapping which an export of the training data leaves out is no gap."""
    problem = _problem_of(
        [
            FitMappingCollection(experiment=MixedStudy, mappings=["fm_tc"]),
            FitMappingCollection(
                experiment=ScalarStudy,
                mappings=["fm_cmax"],
                kind=MappingKind.VALIDATION,
            ),
        ],
        tmp_path,
    )
    training = PetabExporter(problem, kinds={MappingKind.TRAINING})
    assert "scalar-observable" not in {gap.id for gap in training.gaps}
    training.check()
    with pytest.raises(ValueError, match=r"ScalarStudy\.fm_cmax"):
        PetabExporter(problem).check()


def test_the_selections_gap_counts_the_exported_experiments(tmp_path: Path) -> None:
    """An export of the training data of one experiment has no selections gap."""
    problem = _problem_of(
        [
            FitMappingCollection(experiment=MixedStudy, mappings=["fm_tc"]),
            FitMappingCollection(
                experiment=ScalarStudy,
                mappings=["fm_cmax"],
                kind=MappingKind.VALIDATION,
            ),
        ],
        tmp_path,
    )
    training = PetabExporter(problem, kinds={MappingKind.TRAINING})
    assert "selections" not in {gap.id for gap in training.gaps}
    assert "selections" in {gap.id for gap in PetabExporter(problem).gaps}
