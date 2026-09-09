"""Helper functions for fitting."""

import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.fit import display
from sbmlsim.fit.objects import (
    FitMapping,
    FitMappingCollection,
    MappingKind,
    MappingMetaData,
)

logger = logging.getLogger(__name__)

MappingFilter = Callable[[str, FitMapping], bool]


def filtered_mapping_collections(
    experiment_classes: list[type[SimulationExperiment]],
    metadata_filters: MappingFilter | Iterable[MappingFilter],
    base_path: Path,
    data_path: Path,
    kind: MappingKind = MappingKind.TRAINING,
) -> tuple[dict[str, list[FitMappingCollection]], pd.DataFrame]:
    """Create fit experiments from the fit mappings which pass all filters.

    Every filter is called with the key of a fit mapping and the `FitMapping`; a
    mapping is used if all filters accept it. The fit experiments use the weights
    of the mappings (`use_mapping_weights=True`).

    The `kind` classifies the selected data: the training data of a fit, the
    validation data it is evaluated on, or the outliers which are not used. The
    selection and its classification happen here, the fit mappings of the
    simulation experiments only describe the curves.

    Args:
        experiment_classes: simulation experiment classes to filter.
        metadata_filters: a single filter or an iterable of filters.
        base_path: base path of the simulation experiments.
        data_path: path of the datasets of the simulation experiments.
        kind: what a fit does with the selected mappings.

    Returns:
        Tuple of the fit experiments by experiment id and a DataFrame with the
        metadata of the accepted mappings.
    """
    # instantiate objects for filtering of fit mappings
    runner = ExperimentRunner(
        experiment_classes=experiment_classes,
        base_path=base_path,
        data_path=data_path,
    )
    return _filter_mapping_collections(runner, metadata_filters, kind)


def _filter_mapping_collections(
    runner: ExperimentRunner,
    metadata_filters: MappingFilter | Iterable[MappingFilter],
    kind: MappingKind,
) -> tuple[dict[str, list[FitMappingCollection]], pd.DataFrame]:
    """Select the fit mappings of initialized experiments, see `filtered_mapping_collections`.

    Initializing the experiments loads their models and datasets, which is the
    expensive part of the selection, so a runner is filtered several times.

    Args:
        runner: runner with the initialized simulation experiments.
        metadata_filters: a single filter or an iterable of filters.
        kind: what a fit does with the selected mappings.

    Returns:
        Tuple of the fit experiments by experiment id and a DataFrame with the
        metadata of the accepted mappings.
    """
    filters: list[MappingFilter] = (
        list(metadata_filters)
        if isinstance(metadata_filters, Iterable)
        else [metadata_filters]
    )

    mapping_collections: dict[str, list[FitMappingCollection]] = {}
    all_info: list[dict[str, Any]] = []

    for experiment_name, experiment in runner.experiments.items():
        experiment_class = type(experiment)

        # filter mappings by metadata
        mappings: list[str] = []
        for fm_key, fit_mapping in experiment._fit_mappings.items():
            if not all(f(fm_key, fit_mapping) for f in filters):
                continue

            mappings.append(fm_key)

            # collect information
            metadata: MappingMetaData | None = fit_mapping.metadata
            if metadata is None:
                continue
            try:
                yid = "__".join(fit_mapping.observable.y.sid.split("__")[1:])
                all_info.append(
                    {
                        "experiment": experiment_name,
                        "fm_key": fm_key,
                        "yid": yid,
                        "kind": kind.value,
                        **metadata.to_dict(),
                    }
                )
            except Exception as err:
                logger.error(
                    "Error in metadata for experiment '%s', fm_key='%s'",
                    experiment_name,
                    fm_key,
                )
                raise err

        if mappings:
            # add fit experiment from filtered mappings
            mapping_collections[experiment_name] = [
                FitMappingCollection(
                    experiment=experiment_class,
                    mappings=mappings,
                    weights=None,
                    use_mapping_weights=True,
                    kind=kind,
                )
            ]

    return mapping_collections, pd.DataFrame(all_info)


def f_collection(
    experiment_classes: list[type[SimulationExperiment]],
    metadata_filters: MappingFilter | Iterable[MappingFilter],
    base_path: Path,
    data_path: Path,
    print_info: bool = True,
    kind: MappingKind = MappingKind.TRAINING,
) -> dict[str, list[FitMappingCollection]]:
    """Get the filtered fit experiments and print the metadata of the mappings.

    See `filtered_mapping_collections`, this only drops the metadata DataFrame.
    """
    mapping_collections, df = filtered_mapping_collections(
        experiment_classes,
        metadata_filters=metadata_filters,
        base_path=base_path,
        data_path=data_path,
        kind=kind,
    )
    if print_info:
        display.print_data(df)

    return mapping_collections


def mapping_kinds_info(df: pd.DataFrame) -> str:
    """Summarize how the fit mappings of a metadata table are used.

    Args:
        df: metadata table of `filtered_mapping_collections`.

    Returns:
        One line with the number of mappings per `MappingKind`.
    """
    if "kind" not in df.columns:
        return f"{'mappings':<12}: {len(df)}"

    counts = df["kind"].value_counts()
    parts = [
        f"{int(counts[kind.value])} {kind.value}"
        for kind in MappingKind
        if kind.value in counts
    ]
    return f"{'mappings':<12}: {len(df)} ({', '.join(parts)})"


def filter_empty(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Accept all fit mappings."""
    return True


def filter_keys(keys: Iterable[str]) -> MappingFilter:
    """Create a filter which accepts the fit mappings with the given keys.

    This selects the data of a fit by name, e.g., the curves which are kept out
    of the fit as validation data or dropped as outliers.

    Args:
        keys: keys of the fit mappings to accept.

    Returns:
        Filter for `filtered_mapping_collections`.
    """
    selected = set(keys)

    def f_filter(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
        return fit_mapping_key in selected

    return f_filter


def filter_not_keys(keys: Iterable[str]) -> MappingFilter:
    """Create a filter which rejects the fit mappings with the given keys.

    This is the complement of `filter_keys`, i.e., the data which stays in the
    fit.

    Args:
        keys: keys of the fit mappings to reject.

    Returns:
        Filter for `filtered_mapping_collections`.
    """
    excluded = set(keys)

    def f_filter(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
        return fit_mapping_key not in excluded

    return f_filter


def mapping_collections_by_kind(
    experiment_classes: list[type[SimulationExperiment]],
    base_path: Path,
    data_path: Path,
    filters_by_kind: dict[MappingKind, MappingFilter | Iterable[MappingFilter]],
    print_info: bool = True,
) -> dict[str, list[FitMappingCollection]]:
    """Select the data of a fit and classify it in one step.

    Every kind gets its own filters, so the data of a fit is split into the
    training data, the validation data it is evaluated on and the outliers
    which are not used. One overview of all selected mappings is printed.

    Args:
        experiment_classes: simulation experiment classes to filter.
        base_path: base path of the simulation experiments.
        data_path: path of the datasets of the simulation experiments.
        filters_by_kind: filters of every kind, see `filtered_mapping_collections`.
        print_info: print the overview of the selected data.

    Returns:
        The fit experiments of all kinds by experiment id.
    """
    # the experiments are initialized once and filtered for every kind
    runner = ExperimentRunner(
        experiment_classes=experiment_classes,
        base_path=base_path,
        data_path=data_path,
    )
    experiments: list[dict[str, list[FitMappingCollection]]] = []
    frames: list[pd.DataFrame] = []
    for kind, filters in filters_by_kind.items():
        mapping_collections, df = _filter_mapping_collections(runner, filters, kind)
        experiments.append(mapping_collections)
        frames.append(df)

    if print_info:
        display.print_data(pd.concat(frames, ignore_index=True))

    return merge_mapping_collections(*experiments)


def merge_mapping_collections(
    *mapping_collections: dict[str, list[FitMappingCollection]],
) -> dict[str, list[FitMappingCollection]]:
    """Combine the fit experiments of several selections.

    A fit is built from the selections of its kinds, e.g., its training data
    and its validation data, which are combined here.

    Args:
        mapping_collections: fit experiments by experiment id.

    Returns:
        The fit experiments of all selections by experiment id.
    """
    merged: dict[str, list[FitMappingCollection]] = {}
    for experiments in mapping_collections:
        for sid, exps in experiments.items():
            merged.setdefault(sid, []).extend(exps)
    return merged
