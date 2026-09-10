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
from sbmlsim.fit.objects import MappingKind
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import OptimizationStrategy


def test_definition_collections(definition_hctz_pk: FitDefinition) -> None:
    """The definition creates the fit mapping collections of its problem."""
    collections = definition_hctz_pk.collections()
    assert collections
    assert all(collection.mappings for collection in collections)
    # every collection has an id, which names the experiments of a PEtab problem
    assert all(collection.sid for collection in collections)


def test_definition_collections_selected(
    definition_hctz_pk: FitDefinition,
) -> None:
    """A subset of the studies is selected by their ids."""
    # the callable of the definition creates the collections by study id
    study_ids = list(definition_hctz_pk.mapping_collections())
    collections = definition_hctz_pk.collections(study_ids=study_ids[:1])
    # a study has one collection per kind of its data
    assert collections
    assert {c.experiment_class.__name__ for c in collections} == {study_ids[0]}


def test_definition_unknown_study(definition_hctz_pk: FitDefinition) -> None:
    """An unknown study id is reported."""
    with pytest.raises(KeyError, match="Unknown studies"):
        definition_hctz_pk.collections(study_ids=["Nonexistent1999"])


def test_definition_problem(definition_hctz_pk: FitDefinition) -> None:
    """The definition creates the optimization problem."""
    problem = definition_hctz_pk.problem(opid="test")
    assert isinstance(problem, OptimizationProblem)
    assert problem.opid == "test"
    assert problem.pids == [p.pid for p in definition_hctz_pk.parameters]
    assert not problem.is_initialized


def test_run_fit_all(definition_hctz_pk: FitDefinition) -> None:
    """The `ALL` strategy fits the experiments together."""
    runs = run_fit(
        definition=definition_hctz_pk,
        opid="pk",
        strategy=OptimizationStrategy.ALL,
        size=1,
        n_cores=1,
        seed=1234,
    )
    assert list(runs) == ["pk"]
    run = runs["pk"]
    assert run.problem.opid == "pk"
    assert run.result.size == 1
    assert run.result.opid == "pk"
    # the settings travel with the result, the report reads them back
    assert run.result.settings == definition_hctz_pk.settings


def test_run_fit_single(definition_hctz_pk: FitDefinition) -> None:
    """The `SINGLE` strategy fits every collection of training data on its own.

    A collection which is not fitted is not a problem of its own: the
    validation data is evaluated with the training data of a fit, and the
    outliers and the excluded data are not used at all.
    """
    collections = definition_hctz_pk.collections()
    training = [
        collection
        for collection in collections
        if collection.kind is MappingKind.TRAINING
    ]
    assert len(training) < len(collections), (
        "the fixture should have a collection which is not fitted"
    )

    runs = run_fit(
        definition=definition_hctz_pk,
        opid="the_fit",
        strategy=OptimizationStrategy.SINGLE,
        size=1,
        n_cores=1,
        seed=1234,
    )
    # every collection of training data gets its own problem, they share the
    # id of the fit
    assert list(runs) == [
        f"{collection.experiment_class.__name__}_the_fit" for collection in training
    ]


def test_run_fit_report(tmp_path: Path, definition_hctz_pk: FitDefinition) -> None:
    """A finished fit creates its report."""
    runs = run_fit(
        definition=definition_hctz_pk, opid="pk", size=1, n_cores=1, seed=1234
    )
    results_dir = runs["pk"].report(output_dir=tmp_path)
    assert (results_dir / "index.html").exists()
    assert (results_dir / "parameters.json").exists()
    assert (results_dir / "metrics.tsv").exists()


def test_fit_cli(tmp_path: Path) -> None:
    """The fit tool runs a fit of a definition and reports it."""
    from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS

    runs = fit_cli(
        FIT_DEFINITIONS,
        args=[
            "--subset=PK",
            "--runs=1",
            "--cores=1",
            "--seed=1234",
            "--method=LSQ",
            "--strategy=ALL",
            "--name=cli",
            f"--output_dir={tmp_path}",
        ],
    )
    # the fit gets an id of its own, which names its problem and its result
    opid = next(iter(runs))
    assert opid.startswith("PK_")
    assert runs[opid].result.sid == opid
    assert (tmp_path / "cli" / "index.html").exists()
    assert (tmp_path / "cli" / "parameters.json").exists()


def test_report_cli(tmp_path: Path, definition_hctz_pk: FitDefinition) -> None:
    """The report tool reports stored parameters without optimizing."""
    from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS

    runs = run_fit(
        definition=definition_hctz_pk, opid="pk", size=1, n_cores=1, seed=1234
    )
    parameters_path = tmp_path / "parameters.json"
    runs["pk"].result.parameter_sets(size=1).to_json(path=parameters_path)

    results_dir = report_cli(
        FIT_DEFINITIONS,
        args=[
            str(parameters_path),
            "--subset=PK",
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


def test_fit_id_is_unique_and_sortable() -> None:
    """The id of a fit carries its name, the time and a hash."""
    from sbmlsim.fit.result import fit_id

    first = fit_id("PK")
    second = fit_id("PK")
    assert first.startswith("PK_")
    assert first != second
    # <name>_<date>_<time>__<hash>
    assert len(first.split("_")) == 5
    assert fit_id().count("__") == 1


def test_fit_id_is_used_everywhere(
    tmp_path: Path, definition_hctz_pk: FitDefinition
) -> None:
    """The id created for a fit is the id of its problem, result and report."""
    runs = run_fit(
        definition=definition_hctz_pk, opid="the_fit", size=1, n_cores=1, seed=1234
    )
    run = runs["the_fit"]
    assert run.problem.opid == "the_fit"
    assert run.result.opid == "the_fit"
    assert run.result.sid == "the_fit"
    # the report is written into a directory named after the fit
    assert run.report(output_dir=tmp_path).name == "the_fit"
    # and the parameter sets of the result carry it
    assert run.result.parameter_set().sid.startswith("the_fit")


def test_run_fit_creates_an_id(definition_hctz_pk: FitDefinition) -> None:
    """A fit without an id gets one."""
    runs = run_fit(definition=definition_hctz_pk, size=1, n_cores=1, seed=1234)
    opid = next(iter(runs))
    assert "__" in opid
    assert runs[opid].result.sid == opid
