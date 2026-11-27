"""The training and the validation problem of a fit and their READMEs."""

import filecmp
from pathlib import Path

import numpy as np
import petab.v2 as petab_v2
import pytest

from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS
from sbmlsim.fit import FitSettings
from sbmlsim.fit.metrics import FitMetrics
from sbmlsim.fit.objects import MappingKind
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.fit.petab_v2 import PetabExporter, from_petab, to_petab_problems
from sbmlsim.fit.petab_v2.extension import extension_of
from sbmlsim.fit.petab_v2.readme import README_FILE


@pytest.fixture(scope="module")
def problem() -> OptimizationProblem:
    """Get the initialized HCTZ PK problem, with training, validation and outliers."""
    definition = FIT_DEFINITIONS["PK"]
    problem = definition.problem(opid="hctz_pk_problems")
    problem.initialize(definition.settings)
    return problem


@pytest.fixture(scope="module")
def petab_dir(
    tmp_path_factory: pytest.TempPathFactory, problem: OptimizationProblem
) -> Path:
    """Write the training and the validation problem once for the module."""
    output_dir = tmp_path_factory.mktemp("problems")
    exporters = to_petab_problems(problem, output_dir, origin="the start values")
    assert list(exporters) == ["training", "validation"]
    return output_dir


def _x0(problem: OptimizationProblem) -> np.ndarray:
    return np.asarray(problem.x0, dtype=float)


def test_the_problems_have_the_data_of_their_role(
    petab_dir: Path, problem: OptimizationProblem
) -> None:
    for role, kind in (
        ("training", MappingKind.TRAINING),
        ("validation", MappingKind.VALIDATION),
    ):
        yaml_file = petab_dir / role / "problem.yaml"
        petab_problem = petab_v2.Problem.from_yaml(yaml_file)
        extension = extension_of(petab_problem.config)
        assert extension is not None
        assert extension.role == role
        assert {info["kind"] for info in extension.observables.values()} == {kind.value}
        assert len(extension.observables) == len(problem.indices(kind))
        assert (petab_dir / role / README_FILE).is_file()


def test_the_validation_problem_fixes_the_parameters(
    petab_dir: Path, problem: OptimizationProblem
) -> None:
    training = petab_v2.Problem.from_yaml(petab_dir / "training" / "problem.yaml")
    validation = petab_v2.Problem.from_yaml(petab_dir / "validation" / "problem.yaml")
    assert all(p.estimate for p in training.parameters)
    assert not any(p.estimate for p in validation.parameters)
    values = dict(zip(problem.pids, _x0(problem), strict=True))
    assert {p.id: p.nominal_value for p in validation.parameters} == values


def test_the_training_problem_reads_back_with_the_cost_of_the_fit(
    petab_dir: Path, problem: OptimizationProblem
) -> None:
    read, settings = from_petab(petab_dir / "training" / "problem.yaml")
    read.initialize(settings)
    assert read.mapping_counts() == {
        MappingKind.TRAINING: len(problem.indices(MappingKind.TRAINING))
    }
    cost = problem.cost_least_square(problem.to_scale(_x0(problem)))
    # up to the selections of the tasks, see the gap `selections`
    assert read.cost_least_square(read.to_scale(_x0(read))) == pytest.approx(
        cost, rel=1e-4
    )


def test_the_validation_problem_is_the_cost_of_the_validation_data(
    petab_dir: Path, problem: OptimizationProblem
) -> None:
    """The validation problem read back has the cost of the validation data."""
    read, settings = from_petab(petab_dir / "validation" / "problem.yaml")
    read.initialize(settings)
    # the parameters of the fit at their values, the data of its objective
    assert read.pids == problem.pids
    np.testing.assert_allclose(_x0(read), _x0(problem), rtol=1e-12)
    assert read.mapping_counts() == {
        MappingKind.TRAINING: len(problem.indices(MappingKind.VALIDATION))
    }

    parameter_set = ParameterSet.from_fit_parameters(
        problem.parameters, _x0(problem), sid="x0", units=problem.parameter_units
    )
    points = FitMetrics(problem, parameter_set).datapoints_df()
    validation = points[points.kind == MappingKind.VALIDATION.value]
    cost = 0.5 * float(np.sum(np.square(validation.IWRES)))
    assert read.cost_least_square(read.to_scale(_x0(read))) == pytest.approx(
        cost, rel=1e-4
    )


def test_the_readmes_name_what_is_left_out(
    petab_dir: Path, problem: OptimizationProblem
) -> None:
    training = (petab_dir / "training" / README_FILE).read_text(encoding="utf-8")
    validation = (petab_dir / "validation" / README_FILE).read_text(encoding="utf-8")
    assert training.startswith("# PEtab v2 training problem of `hctz_pk_problems`")
    assert validation.startswith("# PEtab v2 validation problem of `hctz_pk_problems`")
    assert "the validation problem `../validation`" in training
    assert "the training problem `../training`" in validation
    assert "| parameter values | the start values |" in training
    # the outliers are in neither problem, the other kind in the other one
    outliers = problem.indices(MappingKind.OUTLIER)
    assert outliers
    for k in outliers:
        mapping = problem.mapping_keys[k]
        assert f"`{mapping}`" in training
        assert f"`{mapping}`" in validation
    assert "are the outliers" in training
    assert "in the validation problem `../validation`" in training
    assert "in the training problem `../training`" in validation
    for section in (
        "## The fit problem",
        "## Files",
        "## Data",
        "## Data which is not written",
        "## What PEtab does not express",
        "## Measurements at the time of a change",
        "## Reading the problem",
    ):
        assert section in training
        assert section in validation
    assert "`estimate=false`" in validation
    assert "### selections" in training


def test_an_export_is_deterministic(
    tmp_path: Path, petab_dir: Path, problem: OptimizationProblem
) -> None:
    """The same problem writes the same files, its README included."""
    to_petab_problems(problem, tmp_path, origin="the start values")
    for role in ("training", "validation"):
        names = sorted(path.name for path in (petab_dir / role).iterdir())
        assert names == sorted(path.name for path in (tmp_path / role).iterdir())
        _, mismatch, errors = filecmp.cmpfiles(
            petab_dir / role, tmp_path / role, names, shallow=False
        )
        assert not mismatch
        assert not errors


def test_a_role_takes_no_kinds(problem: OptimizationProblem) -> None:
    with pytest.raises(ValueError, match="takes no `kinds`"):
        PetabExporter(problem, role="training", kinds={MappingKind.TRAINING})
    with pytest.raises(ValueError, match="`role` is one of"):
        PetabExporter(problem, role="test")  # ty: ignore[invalid-argument-type]


def test_the_readme_of_a_fit_names_its_kinds(
    tmp_path: Path, problem: OptimizationProblem
) -> None:
    """The one problem of every kind says that PEtab has no kind."""
    exporter = PetabExporter(problem)
    exporter.write(tmp_path)
    readme = (tmp_path / README_FILE).read_text(encoding="utf-8")
    assert readme.startswith("# PEtab v2 problem `hctz_pk_problems`")
    assert "PEtab has no kind of a measurement" in readme
    assert "### mapping-kind" in readme
    # the outliers are written, with their kind
    assert "| outlier |" in readme
    assert "## Data which is not written" in readme


def test_to_petab_writes_no_readme_on_request(
    tmp_path: Path, problem: OptimizationProblem
) -> None:
    from sbmlsim.fit.petab_v2 import to_petab

    to_petab(problem, tmp_path, readme=False)
    assert not (tmp_path / README_FILE).exists()
    assert (tmp_path / "problem.yaml").exists()


def test_settings_are_the_ones_of_the_fit(petab_dir: Path) -> None:
    _, settings = from_petab(petab_dir / "validation" / "problem.yaml")
    assert settings == FIT_DEFINITIONS["PK"].settings
    assert isinstance(settings, FitSettings)
