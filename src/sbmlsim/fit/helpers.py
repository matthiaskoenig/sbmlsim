"""Helper functions for fitting."""

import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from sbmlsim.console import console
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.fit.objects import FitExperiment, FitMapping, MappingMetaData

logger = logging.getLogger(__name__)

MappingFilter = Callable[[str, FitMapping], bool]


def filtered_fit_experiments(
    experiment_classes: list[type[SimulationExperiment]],
    metadata_filters: MappingFilter | Iterable[MappingFilter],
    base_path: Path,
    data_path: Path,
) -> tuple[dict[str, list[FitExperiment]], pd.DataFrame]:
    """Create fit experiments from the fit mappings which pass all filters.

    Every filter is called with the key of a fit mapping and the `FitMapping`; a
    mapping is used if all filters accept it. The fit experiments use the weights
    of the mappings (`use_mapping_weights=True`).

    Args:
        experiment_classes: simulation experiment classes to filter.
        metadata_filters: a single filter or an iterable of filters.
        base_path: base path of the simulation experiments.
        data_path: path of the datasets of the simulation experiments.

    Returns:
        Tuple of the fit experiments by experiment id and a DataFrame with the
        metadata of the accepted mappings.
    """
    filters: list[MappingFilter] = (
        list(metadata_filters)
        if isinstance(metadata_filters, Iterable)
        else [metadata_filters]
    )

    # instantiate objects for filtering of fit mappings
    runner = ExperimentRunner(
        experiment_classes=experiment_classes,
        base_path=base_path,
        data_path=data_path,
    )

    fit_experiments: dict[str, list[FitExperiment]] = {}
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
            fit_experiments[experiment_name] = [
                FitExperiment(
                    experiment=experiment_class,
                    mappings=mappings,
                    weights=None,
                    use_mapping_weights=True,
                )
            ]

    return fit_experiments, pd.DataFrame(all_info)


def f_fitexp(
    experiment_classes: list[type[SimulationExperiment]],
    metadata_filters: MappingFilter | Iterable[MappingFilter],
    base_path: Path,
    data_path: Path,
    print_info: bool = True,
) -> dict[str, list[FitExperiment]]:
    """Get the filtered fit experiments and print the metadata of the mappings.

    See `filtered_fit_experiments`, this only drops the metadata DataFrame.
    """
    fit_experiments, df = filtered_fit_experiments(
        experiment_classes,
        metadata_filters=metadata_filters,
        base_path=base_path,
        data_path=data_path,
    )
    if print_info:
        console.print(df.to_string())

    return fit_experiments


def filter_empty(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Accept all fit mappings."""
    return True


def filter_outlier(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Accept the fit mappings which are not marked as outliers."""
    return fit_mapping.metadata is None or not fit_mapping.metadata.outlier
