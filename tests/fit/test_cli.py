"""Test the general fit runner and its command line tools."""

from pathlib import Path

import pytest

from sbmlsim.fit import FitSettings
from sbmlsim.fit.cli import (
    ALGORITHMS,
    FitDefinition,
    fit_cli,
    load_parameter_sets,
    report_cli,
    run_fit,
)
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import OptimizationStrategy


def test_definition_experiments(definition_hctz_pkiv: FitDefinition) -> None:
    """The definition creates the fit experiments of its problem."""
    experiments = definition_hctz_pkiv.experiments()
    assert experiments
    assert all(e.mappings for e in experiments)


def test_definition_experiments_selected(definition_hctz_pkiv: FitDefinition) -> None:
    """A subset of the experiments is selected by their ids."""
    all_ids = list(definition_hctz_pkiv.fit_experiments())
    experiments = definition_hctz_pkiv.experiments(study_ids=all_ids[:1])
    assert len(experiments) == 1


def test_definition_unknown_experiment(definition_hctz_pkiv: FitDefinition) -> None:
    """An unknown experiment id is reported."""
    with pytest.raises(KeyError, match="Unknown experiments"):
        definition_hctz_pkiv.experiments(study_ids=["Nonexistent1999"])


def test_definition_problem(definition_hctz_pkiv: FitDefinition) -> None:
    """The definition creates the optimization problem."""
    problem = definition_hctz_pkiv.problem(opid="test")
    assert isinstance(problem, OptimizationProblem)
    assert problem.opid == "test"
    assert problem.pids == [p.pid for p in definition_hctz_pkiv.parameters]
    assert not problem.is_initialized


def test_run_fit_all(definition_hctz_pkiv: FitDefinition) -> None:
    """The `ALL` strategy fits the experiments together."""
    runs = run_fit(
        definition=definition_hctz_pkiv,
        opid="pkiv",
        strategy=OptimizationStrategy.ALL,
        size=1,
        n_cores=1,
        seed=1234,
    )
    assert list(runs) == ["pkiv"]
    run = runs["pkiv"]
    assert run.problem.opid == "pkiv"
    assert run.result.size == 1
    assert run.result.opid == "pkiv"
    # the settings travel with the result, the report reads them back
    assert run.result.settings == definition_hctz_pkiv.settings


def test_run_fit_single(definition_hctz_pkiv: FitDefinition) -> None:
    """The `SINGLE` strategy fits every experiment on its own."""
    experiments = definition_hctz_pkiv.experiments()
    runs = run_fit(
        definition=definition_hctz_pkiv,
        strategy=OptimizationStrategy.SINGLE,
        size=1,
        n_cores=1,
        seed=1234,
    )
    assert list(runs) == [e.experiment_class.__name__ for e in experiments]


def test_run_fit_report(tmp_path: Path, definition_hctz_pkiv: FitDefinition) -> None:
    """A finished fit creates its report."""
    runs = run_fit(
        definition=definition_hctz_pkiv, opid="pkiv", size=1, n_cores=1, seed=1234
    )
    results_dir = runs["pkiv"].report(output_dir=tmp_path)
    assert (results_dir / "index.html").exists()
    assert (results_dir / "parameters.json").exists()
    assert (results_dir / "metrics.tsv").exists()


def test_fit_cli(tmp_path: Path) -> None:
    """The fit tool runs a fit of a definition and reports it."""
    from examples.hctz.fitting.fitting import FIT_DEFINITIONS

    runs = fit_cli(
        FIT_DEFINITIONS,
        args=[
            "--subset=PKIV",
            "--runs=1",
            "--cores=1",
            "--seed=1234",
            "--method=LSQ",
            "--strategy=ALL",
            "--name=cli",
            f"--output_dir={tmp_path}",
        ],
    )
    assert list(runs) == ["PKIV"]
    assert (tmp_path / "cli" / "index.html").exists()
    assert (tmp_path / "cli" / "parameters.json").exists()


def test_report_cli(tmp_path: Path, definition_hctz_pkiv: FitDefinition) -> None:
    """The report tool reports stored parameters without optimizing."""
    from examples.hctz.fitting.fitting import FIT_DEFINITIONS

    runs = run_fit(
        definition=definition_hctz_pkiv, opid="pkiv", size=1, n_cores=1, seed=1234
    )
    parameters_path = tmp_path / "parameters.json"
    runs["pkiv"].result.parameter_sets(size=1).to_json(path=parameters_path)

    results_dir = report_cli(
        FIT_DEFINITIONS,
        args=[
            str(parameters_path),
            "--subset=PKIV",
            "--name=stored",
            f"--output_dir={tmp_path / 'report'}",
        ],
    )
    assert (results_dir / "index.html").exists()
    assert (results_dir / "metrics.tsv").exists()


@pytest.mark.parametrize("f_cli", [fit_cli, report_cli])
def test_cli_requires_definitions(f_cli) -> None:
    """A tool without a fit definition is an error."""
    with pytest.raises(ValueError, match="At least one FitDefinition"):
        f_cli({}, args=[])


def test_algorithms() -> None:
    """The short names of the algorithms are the choices of the tool."""
    assert set(ALGORITHMS) == {"LSQ", "DE"}


def test_load_parameter_sets(tmp_path: Path, fit_settings: FitSettings) -> None:
    """Sets of several files are combined, duplicate ids are made unique."""
    from sbmlsim.fit import ParameterSet, ParameterSets

    paths = []
    for name in ["run1", "run2"]:
        run_dir = tmp_path / name
        run_dir.mkdir()
        path = run_dir / "parameters.json"
        ParameterSets([ParameterSet(sid="fit", values={"p1": 1.0})]).to_json(path=path)
        paths.append(path)

    psets = load_parameter_sets(paths)
    assert len(psets) == 2
    assert [p.sid for p in psets] == ["fit", "run2_fit"]
