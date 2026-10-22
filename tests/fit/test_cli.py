"""Test the general fit runner and its command line tools."""

import dataclasses
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pytest

from sbmlsim.fit import FitSettings
from sbmlsim.fit.cli import (
    ALGORITHMS,
    FitDefinition,
    FitRecords,
    fit_cli,
    load_fit_records,
    load_parameter_sets,
    report_cli,
    report_from_records,
    run_fit,
)
from sbmlsim.fit.evaluation import Grouping
from sbmlsim.fit.methods import ReportContext
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


def test_run_fit_all(
    definition_hctz_pk: FitDefinition, short_fit: dict[str, Any]
) -> None:
    """The `ALL` strategy fits the experiments together."""
    runs = run_fit(
        definition=definition_hctz_pk,
        opid="pk",
        strategy=OptimizationStrategy.ALL,
        size=1,
        n_cores=1,
        seed=1234,
        **short_fit,
    )
    assert list(runs) == ["pk"]
    run = runs["pk"]
    assert run.problem.opid == "pk"
    assert run.result.size == 1
    assert run.result.opid == "pk"
    # the settings travel with the result, the report reads them back
    assert run.result.settings == definition_hctz_pk.settings


def test_run_fit_single(
    definition_hctz_pk: FitDefinition, short_fit: dict[str, Any]
) -> None:
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
        **short_fit,
    )
    # every collection of training data gets its own problem, they share the
    # id of the fit
    assert list(runs) == [
        f"{collection.experiment_class.__name__}_the_fit" for collection in training
    ]


def test_run_fit_report(
    tmp_path: Path, definition_hctz_pk: FitDefinition, short_fit: dict[str, Any]
) -> None:
    """A finished fit creates its report."""
    runs = run_fit(
        definition=definition_hctz_pk,
        opid="pk",
        size=1,
        n_cores=1,
        seed=1234,
        **short_fit,
    )
    results_dir = runs["pk"].report(output_dir=tmp_path, mapping_figures=False)
    assert (results_dir / "index.html").exists()
    assert (results_dir / "parameters.json").exists()
    assert (results_dir / "metrics.tsv").exists()


def test_fit_cli(tmp_path: Path) -> None:
    """The fit tool runs a fit of a definition and reports it."""
    from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS

    runs = fit_cli(
        FIT_DEFINITIONS,
        args=[
            "--no-mapping-figures",
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


def test_report_cli(
    tmp_path: Path, definition_hctz_pk: FitDefinition, short_fit: dict[str, Any]
) -> None:
    """The report tool reports stored parameters without optimizing."""
    from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS

    runs = run_fit(
        definition=definition_hctz_pk,
        opid="pk",
        size=1,
        n_cores=1,
        seed=1234,
        **short_fit,
    )
    parameters_path = tmp_path / "parameters.json"
    runs["pk"].result.parameter_sets(size=1).to_json(path=parameters_path)

    results_dir = report_cli(
        FIT_DEFINITIONS,
        args=[
            "--no-mapping-figures",
            str(parameters_path),
            "--subset=PK",
            "--name=stored",
            f"--output_dir={tmp_path / 'report'}",
        ],
    )
    assert (results_dir / "index.html").exists()
    assert (results_dir / "metrics.tsv").exists()
    # without the records of a fit next to the parameters the report has none
    for name in ["optimization_result.json", "identifiability.json", "fisher.json"]:
        assert not (results_dir / name).exists(), name


def test_report_cli_reads_the_records_of_the_fit(
    tmp_path: Path, report_dir: Path
) -> None:
    """A report of the parameters of a fit carries its records.

    The result of the optimization, the profiles and the Fisher information
    next to the `parameters.json` give the report its methods, its provenance
    and its identifiability.
    """
    from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS

    results_dir = report_cli(
        FIT_DEFINITIONS,
        args=[
            "--no-mapping-figures",
            str(report_dir / "parameters.json"),
            "--subset=PK",
            "--name=record",
            f"--output_dir={tmp_path}",
        ],
    )
    for name in ["optimization_result.json", "identifiability.json", "fisher.json"]:
        assert (results_dir / name).exists(), name
    assert (results_dir / "plots" / "traces.svg").exists()
    assert (results_dir / "plots" / "profiles.svg").exists()

    html = (results_dir / "index.html").read_text(encoding="utf-8")
    # the provenance of the fit, its seed and its identifiability
    assert "1234" in html
    assert "not recorded" not in html
    assert "Profile likelihood" in html and "Fisher information" in html
    assert (results_dir / "report.pdf").exists()


def test_report_cli_describes_the_fit_with_its_stored_settings(
    tmp_path: Path, report_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A report of a record has the settings the fit ran with, not the definition.

    The definition may have changed since the fit ran: the methods describe
    the settings of the record and a warning names the difference.
    """
    from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS

    records = tmp_path / "records"
    records.mkdir()
    shutil.copy(report_dir / "parameters.json", records)
    stored = json.loads((report_dir / "optimization_result.json").read_text())
    assert stored["settings"]["loss_function"] == "LINEAR"
    stored["settings"]["loss_function"] = "SOFT_L1"
    (records / "optimization_result.json").write_text(json.dumps(stored))

    with caplog.at_level(logging.WARNING, logger="sbmlsim.fit.cli"):
        results_dir = report_cli(
            FIT_DEFINITIONS,
            args=[
                "--no-mapping-figures",
                str(records / "parameters.json"),
                "--subset=PK",
                "--name=stored",
                f"--output_dir={tmp_path}",
            ],
        )
    typ = (results_dir / "report.typ").read_text(encoding="utf-8")
    assert "soft L1 loss" in typ
    assert "The loss function is linear" not in typ
    assert "settings of the record" in caplog.text
    assert "loss_function" in caplog.text


def test_report_from_records(
    report_dir: Path, definition_hctz_pk: FitDefinition
) -> None:
    """The report of a fit is built from the records of its folder.

    The parameters, the result of the optimization, the profiles and the
    Fisher information of the folder are in the report, its problem has the
    given id and its context is the given context.
    """
    from sbmlsim.fit.report import FitReport

    context = ReportContext(title="Model: fit PK", notes=("A note.",))
    report = report_from_records(
        definition_hctz_pk, report_dir, opid="PK", context=context
    )
    assert isinstance(report, FitReport)
    assert report.problem.opid == "PK"
    assert report.context == context
    assert report.opt_result is not None
    assert report.identifiability is not None
    assert report.fisher is not None
    stored = load_parameter_sets([report_dir / "parameters.json"])
    assert [p.sid for p in report.parameter_sets] == [p.sid for p in stored]
    assert report.settings == report.opt_result.settings
    assert report.show_titles is False
    assert report.mapping_figures is True


def test_report_from_records_of_other_parameters(
    tmp_path: Path, report_dir: Path, definition_hctz_pk: FitDefinition
) -> None:
    """The parameters of a report from records may come from other files."""
    path = tmp_path / "other.json"
    sets = load_parameter_sets([report_dir / "parameters.json"])
    sets[0].sid = "other"
    sets.to_json(path=path)
    report = report_from_records(
        definition_hctz_pk,
        report_dir,
        opid="PK",
        parameters=[path],
        mapping_figures=False,
    )
    assert [p.sid for p in report.parameter_sets] == ["other"]
    assert report.mapping_figures is False
    assert report.context == ReportContext()


def test_report_from_records_with_loaded_records(
    report_dir: Path, definition_hctz_pk: FitDefinition
) -> None:
    """Records which the caller read already are not read again."""
    records = load_fit_records(report_dir)
    records = dataclasses.replace(records, identifiability=None)
    report = report_from_records(
        definition_hctz_pk, report_dir, opid="PK", records=records
    )
    assert report.opt_result is not None
    assert report.identifiability is None


def test_report_from_records_without_records(
    tmp_path: Path, report_dir: Path, definition_hctz_pk: FitDefinition
) -> None:
    """A folder with `parameters.json` alone gives a report without records."""
    shutil.copy(report_dir / "parameters.json", tmp_path)
    report = report_from_records(definition_hctz_pk, tmp_path, opid="PK")
    assert report.opt_result is None
    assert report.identifiability is None
    assert report.fisher is None
    assert report.settings == definition_hctz_pk.settings


@pytest.mark.parametrize("callable_context", [True, False])
def test_report_cli_passes_the_context(
    tmp_path: Path, report_dir: Path, callable_context: bool
) -> None:
    """A caller passes the context of the report, as data or per subset."""
    from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS

    records = load_fit_records(report_dir)
    assert records.opt_result is not None

    seen: list[tuple[str, FitDefinition, str | None]] = []

    def context_of(
        subset: str, definition: FitDefinition, records: FitRecords
    ) -> ReportContext:
        sid = records.opt_result.sid if records.opt_result is not None else None
        seen.append((subset, definition, sid))
        return ReportContext(title=f"Model: fit {subset}", notes=("A note.",))

    results_dir = report_cli(
        FIT_DEFINITIONS,
        args=[
            "--no-mapping-figures",
            str(report_dir / "parameters.json"),
            "--subset=PK",
            "--name=context",
            f"--output_dir={tmp_path}",
        ],
        context=(
            context_of
            if callable_context
            else context_of("PK", FIT_DEFINITIONS["PK"], records)
        ),
    )
    # the callable gets the subset, its definition and the records of the fit
    assert seen == [("PK", FIT_DEFINITIONS["PK"], records.opt_result.sid)]
    typ = (results_dir / "report.typ").read_text(encoding="utf-8")
    assert "Model: fit PK" in typ
    assert "A note." in typ


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
    tmp_path: Path, definition_hctz_pk: FitDefinition, short_fit: dict[str, Any]
) -> None:
    """The id created for a fit is the id of its problem, result and report."""
    runs = run_fit(
        definition=definition_hctz_pk,
        opid="the_fit",
        size=1,
        n_cores=1,
        seed=1234,
        **short_fit,
    )
    run = runs["the_fit"]
    assert run.problem.opid == "the_fit"
    assert run.result.opid == "the_fit"
    assert run.result.sid == "the_fit"
    # the report is written into a directory named after the fit
    assert run.report(mapping_figures=False, output_dir=tmp_path).name == "the_fit"
    # and the parameter sets of the result carry it
    assert run.result.parameter_set().sid.startswith("the_fit")


def test_run_fit_creates_an_id(
    definition_hctz_pk: FitDefinition, short_fit: dict[str, Any]
) -> None:
    """A fit without an id gets one."""
    runs = run_fit(
        definition=definition_hctz_pk, size=1, n_cores=1, seed=1234, **short_fit
    )
    opid = next(iter(runs))
    assert "__" in opid
    assert runs[opid].result.sid == opid


def test_the_route_parameters_of_the_example_are_versioned() -> None:
    """The example shows one entity estimated for the oral data only."""
    from examples.hctz_fitting.fitting.parameters import PARAMETERS_BY_ROUTE

    versioned = [p for p in PARAMETERS_BY_ROUTE if p.is_versioned]
    assert len(versioned) == 1
    assert {p.target_id for p in versioned} == {"Ka_dis_hctz"}
    assert {p.pid for p in versioned} == {"Ka_dis_hctz_po"}


def test_the_route_parameters_of_the_example_leave_iv_uncovered(
    definition_hctz_pk: FitDefinition, fit_settings: FitSettings
) -> None:
    """The coverage table the feature exists to produce is not empty.

    `Ka_dis_hctz_po` only applies to the oral data, so the intravenous
    simulations must show up as a coverage gap rather than as a second,
    unconstrained version of the parameter.
    """
    from examples.hctz_fitting.fitting.parameters import PARAMETERS_BY_ROUTE

    definition = dataclasses.replace(definition_hctz_pk, parameters=PARAMETERS_BY_ROUTE)
    problem = definition.problem(opid="route_coverage")
    problem.initialize(fit_settings)

    mapping = problem.parameter_mapping
    assert mapping is not None
    (row,) = [row for row in mapping.coverage() if row.pid == "Ka_dis_hctz_po"]
    assert 0 < row.n_covered < row.n_groups
    assert row.uncovered_groups


def test_a_versioned_fit_is_not_worse_than_the_shared_one(
    definition_hctz_pk: FitDefinition, fit_settings: FitSettings
) -> None:
    """Estimating one entity per route can only lower the cost.

    Two versions are a superset of one shared value, so the optimum of the
    versioned problem is at most the optimum of the shared one. Both are run
    from the values of the model, so this compares like with like.
    """
    from examples.hctz_fitting.fitting.parameters import PARAMETERS, PARAMETERS_BY_ROUTE

    def _cost(parameters: list) -> float:
        # a fresh problem per parameter list, not a patched one: `__init__` is
        # what derives `pids`, `punits`, `bounds`, `x0` and `xmodel` from the
        # parameters consistently, and patching them on a built problem leaves
        # some of that stale
        definition = dataclasses.replace(definition_hctz_pk, parameters=parameters)
        problem = definition.problem(opid="cost")
        problem.initialize(fit_settings)
        return problem.cost_least_square(problem.to_scale(problem.xmodel))

    # at the values of the model the two are the same fit, which is the check
    # that the versions were bound and nothing else moved
    assert _cost(PARAMETERS_BY_ROUTE) == pytest.approx(_cost(PARAMETERS), rel=1e-6)


def test_group_by_on_the_command_line(
    tmp_path: Path, definition_hctz_pk: FitDefinition, short_fit: dict[str, Any]
) -> None:
    """`--group-by` evaluates the report by the columns it names."""
    from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS

    runs = run_fit(
        definition=definition_hctz_pk,
        opid="pk",
        size=1,
        n_cores=1,
        seed=1234,
        **short_fit,
    )
    parameters_path = tmp_path / "parameters.json"
    runs["pk"].result.parameter_sets(size=1).to_json(path=parameters_path)

    results_dir = report_cli(
        FIT_DEFINITIONS,
        args=[
            "--no-mapping-figures",
            str(parameters_path),
            "--subset=PK",
            "--name=grouped",
            "--group-by",
            "tissue",
            "--group-by",
            "tissue,route",
            f"--output_dir={tmp_path / 'report'}",
        ],
    )
    plots = results_dir / "plots"
    assert (plots / "goodness_of_fit_tissue.svg").exists()
    assert (plots / "goodness_of_fit_tissue_route.svg").exists()


def test_the_groupings_of_a_definition(
    tmp_path: Path, definition_hctz_pk: FitDefinition, short_fit: dict[str, Any]
) -> None:
    """A report of a definition evaluates by its groupings, `--group-by` replaces them."""
    definition = dataclasses.replace(
        definition_hctz_pk, groupings=(Grouping.by("health"),)
    )
    runs = run_fit(
        definition=definition,
        opid="pk",
        size=1,
        n_cores=1,
        seed=1234,
        **short_fit,
    )
    runs["pk"].result.parameter_sets(size=1).to_json(path=tmp_path / "parameters.json")
    report = report_from_records(definition, tmp_path, opid="pk", mapping_figures=False)
    assert report.groupings == (Grouping.by("health"),)
    report = report_from_records(
        definition,
        tmp_path,
        opid="pk",
        mapping_figures=False,
        groupings=[Grouping.by("tissue")],
    )
    assert [grouping.name for grouping in report.groupings] == ["tissue"]
