"""The PEtab export refuses scalar observables and scan tasks."""

from pathlib import Path
from typing import ClassVar

import petab.v2 as petab_v2
import pytest

from sbmlsim import Q
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import FitData, FitMapping, FitMappingCollection, FitParameter
from sbmlsim.fit.objects import MappingKind
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings, ParameterScaleType
from sbmlsim.fit.petab_v2 import PetabExporter, from_petab, to_petab_problems
from sbmlsim.fit.petab_v2.export import YAML_FILE
from sbmlsim.fit.petab_v2.extension import extension_of
from sbmlsim.fit.petab_v2.gaps import gap_mappings, gaps_of_problem
from sbmlsim.fit.petab_v2.readme import README_FILE
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import Change, Scan, Simulation
from sbmlsim.uncertainty import Uncertain
from tests.fit.scalar_experiment import DoseStudy, MixedStudy, ScalarStudy
from tests.simulator.models import sbml_pk


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


class UncertainDoseStudy(ScalarStudy):
    """`ScalarStudy` with an uncertain dose."""

    def simulations(self) -> dict[str, Simulation | Scan]:
        simulations = super().simulations()
        simulations["single"] = Simulation(
            end=48,
            steps=480,
            changes=[Change(0, {"PODOSE": Uncertain(Q(100, "mg"), cv=0.05)})],
        )
        return simulations


def test_an_uncertain_value_is_a_gap(tmp_path: Path) -> None:
    """The nominal value of an uncertain value is written, the gap says so."""
    problem = _problem(UncertainDoseStudy, "fm_cmax", tmp_path)
    assert "uncertain-value" in {gap.id for gap in gaps_of_problem(problem)}
    plain = _problem(ScalarStudy, "fm_cmax", tmp_path)
    assert "uncertain-value" not in {gap.id for gap in gaps_of_problem(plain)}


class _FileModel:
    """The model of a study as a file, which the export writes."""

    #: the file of the model, written by `_write_model`
    MODEL: ClassVar[Path | None] = None

    def models(self) -> dict[str, AbstractModel | Path]:
        assert _FileModel.MODEL is not None
        return {"m": AbstractModel(source=_FileModel.MODEL)}


class FileMixedStudy(_FileModel, MixedStudy):
    """`MixedStudy` with its model as a file."""


class FileDoseStudy(_FileModel, DoseStudy):
    """`DoseStudy` with its model as a file."""


class FileScalarStudy(_FileModel, ScalarStudy):
    """`ScalarStudy` with its model as a file."""


def _file_problem(tmp_path: Path) -> OptimizationProblem:
    """Get a problem of a timecourse, a PK parameter and a scan, its model a file."""
    model = tmp_path / "pk.xml"
    model.write_text(sbml_pk(), encoding="utf-8")
    _FileModel.MODEL = model
    return _problem_of(
        [
            FitMappingCollection(
                experiment=FileMixedStudy, mappings=["fm_tc", "fm_cmax"]
            ),
            FitMappingCollection(experiment=FileDoseStudy, mappings=["fm_dose"]),
            FitMappingCollection(
                experiment=FileScalarStudy,
                mappings=["fm_cmax"],
                kind=MappingKind.VALIDATION,
            ),
        ],
        tmp_path,
    )


def test_skip_leaves_out_the_unsupported_mappings(tmp_path: Path) -> None:
    problem = _problem_of(
        [
            FitMappingCollection(experiment=MixedStudy, mappings=["fm_tc", "fm_cmax"]),
            FitMappingCollection(experiment=DoseStudy, mappings=["fm_dose"]),
        ],
        tmp_path,
    )
    with pytest.raises(ValueError, match="scalar-observable"):
        PetabExporter(problem).check()
    exporter = PetabExporter(problem, unsupported="skip")
    assert exporter.skipped == {
        "scalar-observable": ["MixedStudy.fm_cmax", "DoseStudy.fm_dose"],
        "scan-task": ["DoseStudy.fm_dose"],
    }
    assert [problem.mapping_keys[k] for k in exporter.indices] == ["fm_tc"]
    exporter.check()
    ids = {gap.id for gap in exporter.gaps}
    assert not ids & {"scalar-observable", "scan-task"}


def test_skip_of_every_mapping_raises(tmp_path: Path) -> None:
    problem = _problem(ScalarStudy, "fm_cmax", tmp_path)
    with pytest.raises(ValueError, match="nothing to export"):
        PetabExporter(problem, unsupported="skip")


def test_an_unknown_mode_raises(tmp_path: Path) -> None:
    problem = _problem(ScalarStudy, "fm_cmax", tmp_path)
    with pytest.raises(ValueError, match="`unsupported` is one of"):
        PetabExporter(problem, unsupported="ignore")  # ty: ignore[invalid-argument-type]


def test_skip_names_the_mappings_in_the_extension_and_the_readme(
    tmp_path: Path,
) -> None:
    problem = _file_problem(tmp_path)
    exporters = to_petab_problems(problem, tmp_path / "petab", unsupported="skip")
    # the validation data is a PK parameter only: no validation problem
    assert list(exporters) == ["training"]
    directory = tmp_path / "petab" / "training"
    extension = extension_of(petab_v2.Problem.from_yaml(directory / YAML_FILE).config)
    assert extension is not None
    assert extension.skipped == {
        "scalar-observable": ["FileMixedStudy.fm_cmax", "FileDoseStudy.fm_dose"],
        "scan-task": ["FileDoseStudy.fm_dose"],
    }
    readme = (directory / README_FILE).read_text(encoding="utf-8")
    assert "PEtab v2 cannot express 2 fit mappings of the fit" in readme
    assert "### `scalar-observable`, 2 fit mappings" in readme
    assert "- `FileDoseStudy`: `fm_dose`" in readme
    assert "PEtab v2 cannot express any validation data of this fit" in readme
    assert "- `FileScalarStudy`: `fm_cmax`" in readme
    # read back: the problem of the timecourse
    read, settings = from_petab(directory / YAML_FILE)
    read.initialize(settings)
    assert read.mapping_keys == ["fm_tc"]


def test_a_fit_without_validation_data_has_the_training_problem(
    tmp_path: Path,
) -> None:
    model = tmp_path / "pk.xml"
    model.write_text(sbml_pk(), encoding="utf-8")
    _FileModel.MODEL = model
    problem = _problem_of(
        [FitMappingCollection(experiment=FileMixedStudy, mappings=["fm_tc"])],
        tmp_path,
    )
    exporters = to_petab_problems(problem, tmp_path / "petab")
    assert list(exporters) == ["training"]
    readme = (tmp_path / "petab" / "training" / README_FILE).read_text(encoding="utf-8")
    assert "this fit has no validation data" in readme
    assert "Every fit mapping of the fit is written" in readme
