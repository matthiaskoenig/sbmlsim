"""The data points of the adopted fits of a model with the kinds of their mappings.

The kind of a fit mapping (`training`, `validation`, `outlier`) decides what its
data can be evidence of: a point which was fitted is no validation. The kinds
are the ones the record of an adopted fit stores (`metrics_mappings.tsv`, the
kind every mapping had when the fit ran), and the current selection of the fit
definition must have the same kinds: a split changed after the adoption without
a refit would otherwise let fitted data count as validation. The remedy of a
difference is a refit and a new adoption, never an override.

`evaluate_fits` evaluates the adopted fits on the current model and data into
the table of their data points; `mapping_table` is the same table with one row
per mapping, without a simulation, for the static check of a declaration.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import pandas as pd

from sbmlsim.fit.helpers import FitMappings
from sbmlsim.fit.metrics import metadata_dict
from sbmlsim.fit.objects import EVALUATED_KINDS, FitMapping
from sbmlsim.fit.result import fit_subset

if TYPE_CHECKING:
    from pathlib import Path

    from sbmlsim.experiment import SimulationExperiment
    from sbmlsim.fit.cli import FitDefinition
    from sbmlsim.fit.report import FitReport
    from sbmlsim.project import Project

__all__ = [
    "FitTable",
    "KindMismatchError",
    "MappingLookup",
    "adopted_ids",
    "evaluate_fits",
    "kind_differences",
    "mapping_table",
    "recorded_kinds",
]

logger = logging.getLogger(__name__)

#: the record of the metrics of the mappings of an adopted fit, with their kinds
METRICS_MAPPINGS: Final = "metrics_mappings.tsv"

#: the columns of a table of the fits before the metadata of the mappings
KEY_COLUMNS: Final = ("fit", "experiment", "mapping", "kind", "observable")

#: the remedy of a kind which differs from the recorded one
KIND_REMEDY: Final = (
    "Refit and adopt the fit again; the kinds of an adopted fit are never overridden."
)

#: gets the fit mapping of an experiment by its key
type MappingLookup = Callable[[str, str], FitMapping]

#: the kind of a mapping by the subset of its fit, its experiment and its key
type Kinds = Mapping[tuple[str, str, str], str]


@dataclass(frozen=True, kw_only=True, eq=False)
class FitTable:
    """The data points of the adopted fits of a model.

    Attributes:
        table: one row per data point (`evaluate_fits`) or per mapping
            (`mapping_table`), with the columns `fit` (the subset of the fit),
            `experiment`, `mapping`, `kind`, `observable` and the fields of the
            metadata of the mappings; the data points have the columns of
            `sbmlsim.fit.metrics.FitMetrics.datapoints_df` as well.
        lookup: the fit mapping of an experiment and a key.
        recorded_kinds: the kinds the records of the adopted fits store, by
            the subset of the fit, the experiment and the key of the mapping.
        reports: the reports the data points were evaluated with, by the
            subset of the fit (`evaluate_fits`), which the verification
            evaluates again with other settings; none for `mapping_table`.
    """

    table: pd.DataFrame
    lookup: MappingLookup
    recorded_kinds: Kinds
    reports: Mapping[str, FitReport] = field(
        default_factory=lambda: MappingProxyType({})
    )


class KindMismatchError(ValueError):
    """The kinds of the fit mappings differ from the kinds the adopted fits recorded.

    Attributes:
        problems: every mapping whose kind differs.
    """

    def __init__(self, problems: list[str]) -> None:
        """Create the error.

        Args:
            problems: every mapping whose kind differs.
        """
        self.problems = list(problems)
        super().__init__(
            "The kinds of the fit mappings differ from the kinds the adopted fits "
            "recorded:\n"
            + "\n".join(f"- {problem}" for problem in self.problems)
            + f"\n{KIND_REMEDY}"
        )


@dataclass(frozen=True)
class _Lookup:
    """The fit mappings of instantiated experiments, see `MappingLookup`.

    Attributes:
        experiments: the experiments by id.
    """

    experiments: Mapping[str, SimulationExperiment]

    def __call__(self, experiment: str, mapping: str) -> FitMapping:
        """Get the fit mapping of an experiment by its key."""
        return self.experiments[experiment]._fit_mappings[mapping]


def recorded_kinds(record: Path, fit: str) -> dict[tuple[str, str, str], str]:
    """Get the kinds of the mappings which the record of an adopted fit stores.

    Args:
        record: the directory of the record of the adopted fit.
        fit: the subset of the fit, the first item of every key.

    Returns:
        The kind by the subset, the experiment and the key of the mapping.

    Raises:
        FileNotFoundError: if the record has no metrics of its mappings.
        ValueError: if the metrics lack the experiment, the mapping or the kind,
            or store two kinds of one mapping.
    """
    path = record / METRICS_MAPPINGS
    if not path.exists():
        raise FileNotFoundError(
            f"The record '{record}' of the adopted fit has no {METRICS_MAPPINGS} "
            f"with the kinds of its mappings; adopt the fit again."
        )
    table = pd.read_csv(path, sep="\t", dtype=str)
    missing = [c for c in ("experiment", "mapping", "kind") if c not in table.columns]
    if missing:
        raise ValueError(
            f"The {METRICS_MAPPINGS} of the record '{record}' of the adopted fit "
            f"lacks the columns {missing}; adopt the fit again."
        )
    found: dict[tuple[str, str, str], set[str]] = {}
    for experiment, mapping, kind in zip(
        table["experiment"], table["mapping"], table["kind"], strict=True
    ):
        found.setdefault((fit, experiment, mapping), set()).add(kind)
    kinds: dict[tuple[str, str, str], str] = {}
    for key, values in found.items():
        if len(values) > 1:
            raise ValueError(
                f"The record '{record}' stores the kinds {sorted(values)} of the "
                f"mapping {key[1]}.{key[2]}, a mapping has one kind."
            )
        (kinds[key],) = values
    return kinds


def kind_differences(table: pd.DataFrame, recorded: Kinds) -> list[str]:
    """Compare the kinds of the mappings of a table with the recorded kinds.

    Args:
        table: a table of the fits with the columns `fit`, `experiment`,
            `mapping` and `kind`, see `FitTable`.
        recorded: the recorded kinds, see `recorded_kinds`.

    Returns:
        One message per mapping whose kind differs from the recorded one, or
        which only the table or only the record has; empty if the kinds agree.
    """
    current: dict[tuple[str, str, str], set[str]] = {}
    for fit, experiment, mapping, kind in zip(
        table["fit"], table["experiment"], table["mapping"], table["kind"], strict=True
    ):
        current.setdefault((str(fit), str(experiment), str(mapping)), set()).add(
            str(kind)
        )
    messages: list[str] = []
    for key in sorted(current.keys() | set(recorded)):
        fit, experiment, mapping = key
        name = f"{fit}: {experiment}.{mapping}"
        kinds = current.get(key)
        if kinds is None:
            messages.append(
                f"{name} is {recorded[key]} in the record of the adopted fit, "
                f"but not selected now"
            )
        elif key not in recorded:
            messages.append(
                f"{name} is {' and '.join(sorted(kinds))} now, but not in the "
                f"record of the adopted fit"
            )
        elif kinds != {recorded[key]}:
            messages.append(
                f"{name} is {' and '.join(sorted(kinds))} now, but "
                f"{recorded[key]} in the record of the adopted fit"
            )
    return messages


def adopted_ids(project: Project) -> list[str]:
    """Get the ids of the adopted fits of a project, in the order of their subsets.

    Raises:
        ValueError: if the project has no adopted fits or two adopted fits
            have one subset.
    """
    fit_ids = list(project.adopted_fits().values())
    if not fit_ids:
        raise ValueError(f"The model '{project.name}' has no adopted fits.")
    subsets = [fit_subset(fit_id) for fit_id in fit_ids]
    repeated = sorted({s for s in subsets if subsets.count(s) > 1})
    if repeated:
        raise ValueError(
            f"The adopted fits of '{project.name}' have the subsets {repeated} "
            f"twice, a subset has one adopted fit."
        )
    return fit_ids


def _adopted_fits(project: Project) -> Iterator[tuple[str, Path, FitDefinition]]:
    """Get the subset, the record and the definition of every adopted fit.

    Raises:
        ValueError: if the project has no adopted fits, two adopted fits have
            one subset or a subset has no fit definition.
        FileNotFoundError: if the record of a fit has no parameters.
    """
    fit_ids = adopted_ids(project)
    definitions = project.load_fit_definitions()
    for fit_id in fit_ids:
        record = project.adopted_record(fit_id)
        yield fit_subset(fit_id), record, definitions[fit_subset(fit_id)]


def _observable(fit_mapping: FitMapping) -> str:
    """Get the observable of a fit mapping as the data points name it.

    The y of the observable as `OptimizationProblem` resolves it: the selection,
    e.g. `[A]` for a concentration, or the name of an observable model.
    """
    y = fit_mapping.observable.y
    return y.index if fit_mapping.observable_model is not None else y.selection


def evaluate_fits(project: Project) -> FitTable:
    """Evaluate the adopted fits of a project into the table of their data points.

    Every fit is evaluated fresh on the current model and data with the settings
    it ran with (`sbmlsim.fit.cli.report_from_records`), at the reference
    parameter set of its record; the kinds of its mappings must be the recorded
    ones.

    Args:
        project: the project, its adopted fits are evaluated.

    Returns:
        The data points of every fit with the column `fit`, the subset, and the
        report of every fit.

    Raises:
        ValueError: if the project has no adopted fits, see `adopted_ids`.
        FileNotFoundError: if a record has no parameters or no kinds.
        KindMismatchError: if a kind differs from the recorded one.
    """
    from sbmlsim.fit.cli import report_from_records

    tables: list[pd.DataFrame] = []
    experiments: dict[str, SimulationExperiment] = {}
    recorded: dict[tuple[str, str, str], str] = {}
    reports: dict[str, FitReport] = {}
    for subset, record, definition in _adopted_fits(project):
        recorded.update(recorded_kinds(record, subset))
        logger.info("Evaluate the adopted fit '%s' of '%s'", subset, project.name)
        report = report_from_records(
            definition, record, opid=subset, mapping_figures=False
        )
        points = report.metrics(report.reference_set).datapoints_df()
        points.insert(0, "fit", subset)
        tables.append(points)
        reports[subset] = report
        # two fits of one experiment hold equal mappings, either serves
        experiments.update(report.problem.runner_initialized.experiments)
    table = pd.concat(tables, ignore_index=True)
    differences = kind_differences(table, recorded)
    if differences:
        raise KindMismatchError(differences)
    return FitTable(
        table=table,
        lookup=_Lookup(experiments),
        recorded_kinds=recorded,
        reports=MappingProxyType(reports),
    )


def mapping_table(project: Project) -> FitTable:
    """Get the mappings of the adopted fits of a project without a simulation.

    The kinds are the ones of the current selection of every fit definition;
    the excluded mappings, which a fit does not evaluate, are no rows. The
    recorded kinds are compared by the caller, see `kind_differences`.

    Args:
        project: the project, its adopted fits are listed.

    Returns:
        One row per fit and mapping with the columns of `KEY_COLUMNS` and the
        fields of the metadata.

    Raises:
        ValueError: if the project has no adopted fits, see `adopted_ids`.
        FileNotFoundError: if a record has no parameters or no kinds.
    """
    # the subset, the experiment, the keys (empty for all) and the kind
    selected: list[tuple[str, str, list[str], str]] = []
    classes: dict[str, type[SimulationExperiment]] = {}
    recorded: dict[tuple[str, str, str], str] = {}
    for subset, record, definition in _adopted_fits(project):
        recorded.update(recorded_kinds(record, subset))
        for collection in definition.collections():
            if collection.kind not in EVALUATED_KINDS:
                continue
            experiment = collection.experiment_class.__name__
            classes[experiment] = collection.experiment_class
            selected.append(
                (subset, experiment, list(collection.mappings), collection.kind.value)
            )
    fit_mappings = FitMappings(
        experiment_classes=classes.values(),
        base_path=project.base_path or project.root,
        data_path=project.data_path,
    )
    lookup = _Lookup(fit_mappings.experiments)
    rows: list[dict[str, str]] = []
    for subset, experiment, keys, kind in selected:
        # an empty list of keys is every mapping of the experiment
        for key in keys or fit_mappings.keys[experiment]:
            fit_mapping = lookup(experiment, key)
            rows.append(
                {
                    "fit": subset,
                    "experiment": experiment,
                    "mapping": key,
                    "kind": kind,
                    "observable": _observable(fit_mapping),
                    **metadata_dict(fit_mapping.metadata, f"{experiment}.{key}"),
                }
            )
    # the fields of the metadata of all mappings, in the order they appear,
    # empty for a mapping without the field as in the data points
    columns = list(dict.fromkeys(key for row in rows for key in row))
    table = pd.DataFrame(
        [{column: row.get(column, "") for column in columns} for row in rows],
        columns=pd.Index(list(dict.fromkeys([*KEY_COLUMNS, *columns]))),
    )
    return FitTable(table=table, lookup=lookup, recorded_kinds=recorded)
