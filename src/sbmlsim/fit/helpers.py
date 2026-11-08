"""Selection of the data of a fit.

A fit selects its data from the fit mappings of its simulation experiments.
`FitMappings` is the complete list of these mappings, i.e., every curve which
is mapped to a simulation, and `FitMappings.select` decides what a fit does
with every one of them in three steps, each setting a `MappingKind`:

1. **The filters select the training data.** A mapping which passes every
   filter is training data of the fit. A mapping which fails a filter is
   `EXCLUDED`: the fit does not use it at all, e.g. the data of a route the fit
   is not about or an arm with a coadministration the model does not describe.
2. **The outliers are named by their keys.** An outlier is training data whose
   values are not usable, e.g. a curve which contradicts the rest of the data,
   so it is tagged once for the complete list of mappings and not per fit. It
   is `OUTLIER` in every fit whose filters select it: it is not fitted, but
   simulated and evaluated so that a report shows where it sits relative to the
   model. An outlier the filters exclude stays excluded.
3. **Part of the training data is the validation data.** The validation data
   is selected from what is left by its keys or by a filter, it is `VALIDATION`:
   not fitted, but simulated and evaluated so that a report shows how the fit
   describes data it was not fitted on. Everything else is `TRAINING` and enters
   the cost.

The three steps are ordered, so a mapping which hits several has one kind:
excluded beats outlier, outlier beats validation and validation beats training.
The kind belongs to the selection of the data, not to the fit mappings of a
simulation experiment: the `MappingMetaData` of a mapping describes its curve,
and the same curve is training data of one fit and validation data of another.

The filters of the mappings are the ones of `sbmlsim.fit.filters` (by the
metadata, the observable, the key and the study), which this module exports
as well; `select_mappings` selects the data of a fit of a
`sbmlsim.project.Project` with its paths.
"""

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from sbmlsim.engine import JitOptimization
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.fit import display
from sbmlsim.fit.filters import (
    KeyFilter,
    MetadataFilter,
    NamedFilter,
    NamedMappings,
    ObservableFilter,
    filter_keys,
    metadata_filter,
    named,
    named_filter,
    observable_filter,
    observable_id,
    study,
)
from sbmlsim.fit.objects import (
    FitMapping,
    FitMappingCollection,
    MappingKind,
    MappingMetaData,
)

if TYPE_CHECKING:
    from sbmlsim.project import Project

__all__ = [
    "FitMappings",
    "KeyFilter",
    "MappingFilter",
    "MappingSelection",
    "MetadataFilter",
    "NamedFilter",
    "NamedMappings",
    "ObservableFilter",
    "filter_keys",
    "metadata_filter",
    "named",
    "named_filter",
    "observable_filter",
    "observable_id",
    "select_mapping_collections",
    "select_mappings",
    "study",
]

logger = logging.getLogger(__name__)

#: filter of fit mappings, called with the key of a fit mapping and the mapping
MappingFilter = Callable[[str, FitMapping], bool]


def _filters(filters: MappingFilter | Iterable[MappingFilter]) -> list[MappingFilter]:
    """Get a single filter or an iterable of filters as a list."""
    if isinstance(filters, Iterable):
        return list(filters)
    return [filters]


@dataclass
class MappingSelection:
    """The kind of every fit mapping, i.e., what a fit does with the data.

    A selection is created by `FitMappings.select`. It is the result of the
    three steps of the selection for every fit mapping of every experiment.

    Attributes:
        kinds: the `MappingKind` of every fit mapping by experiment id and key.
        collections: the `FitMappingCollection` per experiment and kind, in the
            order of `MappingKind`, which is what a `FitDefinition` is defined
            with. Every fit mapping is in exactly one collection.
        df: one row per fit mapping with `experiment`, `fm_key`, `yid` (the
            observable), `kind` and the fields of its `MappingMetaData`; the
            overview `display.print_data` prints.
    """

    kinds: dict[str, dict[str, MappingKind]]
    collections: dict[str, list[FitMappingCollection]] = field(repr=False)
    df: pd.DataFrame = field(repr=False)

    def kind(self, experiment_id: str, key: str) -> MappingKind:
        """Get the kind of a fit mapping.

        Args:
            experiment_id: id of the simulation experiment.
            key: key of the fit mapping.
        """
        return self.kinds[experiment_id][key]

    def kinds_of(self, experiment_id: str) -> dict[str, MappingKind]:
        """Get the kinds of the fit mappings of an experiment.

        Args:
            experiment_id: id of the simulation experiment.
        """
        return self.kinds[experiment_id]

    def print(self, detail: bool = True) -> None:
        """Print the overview of the selected data.

        Args:
            detail: list the single fit mappings, not only the counts.
        """
        display.print_data(self.df, detail=detail)


class FitMappings:
    """The fit mappings of simulation experiments, i.e., the data a fit selects from.

    The experiments are instantiated once, which loads their models and their
    datasets and is the expensive part, and are selected from several times,
    once per fit problem.

    Attributes:
        runner: runner with the instantiated simulation experiments.
        keys: the keys of the fit mappings of every experiment by experiment id.
    """

    def __init__(
        self,
        experiment_classes: Iterable[type[SimulationExperiment]],
        base_path: Path,
        data_path: Path | Iterable[Path] | None,
        jit_optimization: JitOptimization | str | None = None,
    ):
        """Instantiate the simulation experiments.

        Args:
            experiment_classes: simulation experiment classes with fit mappings.
            base_path: base path of the simulation experiments.
            data_path: path or paths of the datasets of the simulation
                experiments, none without datasets.
            jit_optimization: the optimization level the models of the
                experiments are compiled at, see `ExperimentRunner`; the level
                of a project is `SimulatorSettings.jit_optimization`.
        """
        self.runner = ExperimentRunner(
            experiment_classes=list(experiment_classes),
            base_path=base_path,
            data_path=data_path,
            # a fit needs every experiment, its definition errors are raised
            on_error="raise",
            jit_optimization=jit_optimization,
        )
        self.keys: dict[str, list[str]] = {
            experiment_id: list(experiment._fit_mappings)
            for experiment_id, experiment in self.runner.experiments.items()
        }

    @property
    def experiments(self) -> dict[str, SimulationExperiment]:
        """The instantiated simulation experiments by id."""
        return self.runner.experiments

    def _check_keys(self, keys: Iterable[str], what: str) -> set[str]:
        """Check that the keys are fit mappings of some experiment.

        Raises:
            ValueError: for a key which is no fit mapping of any experiment.
        """
        selected = set(keys)
        known = {key for keys in self.keys.values() for key in keys}
        unknown = sorted(selected - known)
        if unknown:
            raise ValueError(
                f"The {what} '{unknown}' are no fit mappings of the experiments "
                f"'{sorted(self.keys)}'."
            )
        return selected

    def select(
        self,
        filters: MappingFilter | Iterable[MappingFilter] = (),
        outliers: Iterable[str] = (),
        validation: Iterable[str] | MappingFilter = (),
        print_info: bool = True,
    ) -> MappingSelection:
        """Select the data of a fit, see the module for the three steps.

        Args:
            filters: filters of the training data. A mapping which passes every
                filter is training data, a mapping which fails one is excluded.
                No filters select every mapping.
            outliers: keys of the outliers, i.e., of the training data which is
                not usable and not fitted. An outlier is a decision about the
                data, so it is named for the complete list of mappings, not per
                fit; an outlier the filters exclude stays excluded.
            validation: the validation data, i.e., the training data which is
                not fitted but evaluated, as the keys of the mappings or as a
                filter of the training data which is left after the outliers.
            print_info: print the overview of the selected data.

        Returns:
            The selection with the kind of every fit mapping and the collections
            a fit is defined with.

        Raises:
            ValueError: for an outlier or validation key which is no fit mapping
                of any experiment.
        """
        training_filters = _filters(filters)
        outlier_keys = self._check_keys(outliers, "outliers")
        validation_filters: list[MappingFilter]
        if isinstance(validation, Iterable):
            validation_keys = self._check_keys(validation, "validation mappings")
            validation_filters = [KeyFilter(frozenset(validation_keys))]
        else:
            validation_filters = [validation]

        kinds: dict[str, dict[str, MappingKind]] = {}
        collections: dict[str, list[FitMappingCollection]] = {}
        rows: list[dict[str, Any]] = []

        for experiment_id, experiment in self.runner.experiments.items():
            experiment_kinds: dict[str, MappingKind] = {}
            for key, fit_mapping in experiment._fit_mappings.items():
                # 1. the filters select the training data
                if not all(f(key, fit_mapping) for f in training_filters):
                    kind = MappingKind.EXCLUDED
                # 2. the outliers are part of the training data
                elif key in outlier_keys:
                    kind = MappingKind.OUTLIER
                # 3. the validation data is selected from the rest
                elif all(f(key, fit_mapping) for f in validation_filters):
                    kind = MappingKind.VALIDATION
                else:
                    kind = MappingKind.TRAINING
                experiment_kinds[key] = kind
                rows.append(_row(experiment_id, key, fit_mapping, kind))

            kinds[experiment_id] = experiment_kinds
            collections[experiment_id] = [
                FitMappingCollection(
                    experiment=type(experiment),
                    mappings=[key for key, k in experiment_kinds.items() if k is kind],
                    weights=None,
                    use_mapping_weights=True,
                    kind=kind,
                )
                for kind in MappingKind
                if kind in experiment_kinds.values()
            ]

        selection = MappingSelection(
            kinds=kinds, collections=collections, df=pd.DataFrame(rows)
        )
        if print_info:
            selection.print()
        return selection


def _row(
    experiment_id: str, key: str, fit_mapping: FitMapping, kind: MappingKind
) -> dict[str, Any]:
    """Get the row of a fit mapping for the overview of the data."""
    row: dict[str, Any] = {
        "experiment": experiment_id,
        "fm_key": key,
        "yid": fit_mapping.observable.y.selection,
        "kind": kind.value,
    }
    metadata: MappingMetaData | None = fit_mapping.metadata
    if metadata is not None:
        row.update(metadata.to_dict())
    return row


def select_mapping_collections(
    experiment_classes: Iterable[type[SimulationExperiment]],
    base_path: Path,
    data_path: Path | Iterable[Path],
    filters: MappingFilter | Iterable[MappingFilter] = (),
    outliers: Iterable[str] = (),
    validation: Iterable[str] | MappingFilter = (),
    print_info: bool = True,
    jit_optimization: JitOptimization | str | None = None,
) -> dict[str, list[FitMappingCollection]]:
    """Select the data of a fit in one call, see `FitMappings.select`.

    This instantiates the experiments and selects from them once, which is
    what the `mapping_collections` of a `FitDefinition` does.

    Args:
        experiment_classes: simulation experiment classes with fit mappings.
        base_path: base path of the simulation experiments.
        data_path: path or paths of the datasets of the simulation experiments.
        filters: filters of the training data.
        outliers: keys of the outliers.
        validation: keys or filter of the validation data.
        print_info: print the overview of the selected data.
        jit_optimization: the optimization level the models of the experiments
            are compiled at, see `FitMappings`.

    Returns:
        The fit mapping collections of all kinds by experiment id.
    """
    fit_mappings = FitMappings(
        experiment_classes=experiment_classes,
        base_path=base_path,
        data_path=data_path,
        jit_optimization=jit_optimization,
    )
    return fit_mappings.select(
        filters=filters,
        outliers=outliers,
        validation=validation,
        print_info=print_info,
    ).collections


def select_mappings(
    project: Project,
    experiment_classes: Iterable[type[SimulationExperiment]],
    filters: MappingFilter | Iterable[MappingFilter] = (),
    outliers: Iterable[str] = (),
    validation: Iterable[str] | MappingFilter = (),
) -> dict[str, list[FitMappingCollection]]:
    """Select the data of a fit of a project, see `FitMappings.select`.

    The experiments are created with the paths of the project (its
    `base_path`, else its `root`, and its `data_path`) and its models compiled
    at its level (`SimulatorSettings.jit_optimization`); the counts of the
    selected mappings per experiment and kind are printed.

    Args:
        project: the project.
        experiment_classes: the experiments with the fit mappings.
        filters: the filters of the training data.
        outliers: the keys of the training data which is not usable.
        validation: the keys or a filter of the validation data.

    Returns:
        The fit mapping collections of all kinds by experiment id.
    """
    fit_mappings = FitMappings(
        experiment_classes=experiment_classes,
        base_path=project.base_path or project.root,
        data_path=project.data_path,
        jit_optimization=project.simulator.jit_optimization,
    )
    selection = fit_mappings.select(
        filters=filters, outliers=outliers, validation=validation, print_info=False
    )
    selection.print(detail=False)
    return selection.collections
