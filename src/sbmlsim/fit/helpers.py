"""Helper functions for fitting."""

from pathlib import Path

from sbmlsim.fit import FitExperiment, FitMapping
import pandas as pd
from pymetadata.console import console
from pymetadata import log


from sbmlsim.experiment import ExperimentRunner, SimulationExperiment

from typing import Type, Union, Callable, Iterable, Tuple, Any

from sbmlsim.fit.objects import MappingMetaData

logger = log.get_logger(__name__)


def filtered_fit_experiments(
    experiment_classes: list[Type[SimulationExperiment]],
    metadata_filters: Union[Callable, Iterable[Callable]],
    base_path: Path,
    data_path: Path,
) -> Tuple[dict[str, list[FitExperiment]], pd.DataFrame]:
    """Fit experiments based on MappingMetaData.

    :param experiment_classes: List of SimulationExperiment class definition
    :param metadata_filter:
    """
    filters = (
        [metadata_filters]
        if isinstance(metadata_filters, Callable)
        else metadata_filters
    )

    # instantiate objects for filtering of fit mappings
    runner = ExperimentRunner(
        experiment_classes=experiment_classes,
        base_path=base_path,
        data_path=data_path,
    )

    fit_experiments: dict[str, list[FitExperiment]] = {}
    all_info: list[dict] = []

    for k, experiment_name in enumerate(runner.experiments):
        # print(experiment_name)
        experiment_class = experiment_classes[k]
        experiment = runner.experiments[experiment_name]

        # filter mappings by metadata
        mappings = []
        for fm_key, fit_mapping in experiment.fit_mappings().items():
            # tests all the filters
            accept = True
            for filter in filters:
                if not filter(fm_key, fit_mapping):
                    accept = False
                    break

            if accept:
                mappings.append(fm_key)

                # collect information
                try:
                    metadata: MappingMetaData = fit_mapping.metadata
                    yid = "__".join(fit_mapping.observable.y.sid.split("__")[1:])
                    info: dict[str, Any] = {
                        "experiment": experiment_name,
                        "fm_key": fm_key,
                        "yid": yid,
                        **metadata.to_dict(),
                    }
                    all_info.append(info)
                except Exception as err:
                    logger.error(
                        f"Error in metadata for experiment '{experiment_name}', {fm_key=}"
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

    df = pd.DataFrame(all_info)

    return fit_experiments, df


def f_fitexp(
    experiment_classes: list[Type[SimulationExperiment]],
    metadata_filters: Union[Callable, Iterable[Callable]],
    base_path: Path,
    data_path: Path,
):
    """Generic function to get fit experiments for filter."""
    fit_experiments, df = filtered_fit_experiments(
        experiment_classes,
        metadata_filters=metadata_filters,
        base_path=base_path,
        data_path=data_path,
    )
    console.print(df.to_string())

    return fit_experiments


def filter_empty(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Return all experiments/mappings."""
    return True


def filter_outlier(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Return non outlier experiments."""
    return not fit_mapping.metadata.outlier
