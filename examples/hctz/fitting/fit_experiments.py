"""Parameter fit problems for HCTZ.

The fit experiments are built from the fit mappings of the studies which pass a
set of filters on the metadata of the mappings.
"""

from examples.hctz import DATA_PATH, HCTZ_PATH
from examples.hctz.experiments.metadata import (
    Coadministration,
    Fasting,
    HCTZMappingMetaData,
    Route,
)
from examples.hctz.experiments.studies import Beermann1976, Patel1984
from sbmlsim.console import console
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import FitExperiment, FitMapping
from sbmlsim.fit.helpers import MappingFilter, f_fitexp, filter_empty

# observables of the pharmacokinetics
PK_OBSERVABLES = {"Afeces_hctz", "Aurine_hctz", "Cve_hctz", "KI__HCTZEX"}

EXPERIMENT_CLASSES: list[type[SimulationExperiment]] = [
    Beermann1976,
    Patel1984,
]


def fit_experiments(
    metadata_filters: MappingFilter | list[MappingFilter],
) -> dict[str, list[FitExperiment]]:
    """Get the fit experiments of the studies for the given filters."""
    return f_fitexp(
        experiment_classes=EXPERIMENT_CLASSES,
        metadata_filters=metadata_filters,
        base_path=HCTZ_PATH,
        data_path=DATA_PATH,
    )


def _metadata(fit_mapping: FitMapping) -> HCTZMappingMetaData:
    """Get the metadata of a fit mapping."""
    metadata = fit_mapping.metadata
    if not isinstance(metadata, HCTZMappingMetaData):
        raise ValueError(f"HCTZMappingMetaData required on: '{fit_mapping}'")
    return metadata


def _yid(fit_mapping: FitMapping) -> str:
    """Get the observable of a fit mapping."""
    return "__".join(fit_mapping.observable.y.sid.split("__")[1:])


def filter_control(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Return control experiments/mappings."""
    metadata = _metadata(fit_mapping)

    # only PO and IV (no SL, MU, RE)
    if metadata.route not in {Route.PO, Route.IV}:
        return False

    # remove not fasted
    if metadata.fasting not in {Fasting.NR, Fasting.FASTED}:
        return False

    # remove coadministration
    if metadata.coadministration != Coadministration.NONE:
        return False

    # remove outliers
    return not metadata.outlier


def filter_iv(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Only IV application data."""
    return _metadata(fit_mapping).route == Route.IV


def filter_pk(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Only HCTZ PK data."""
    return _yid(fit_mapping) in PK_OBSERVABLES


def filter_pd(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Only HCTZ PD data."""
    return _yid(fit_mapping) not in PK_OBSERVABLES


def f_fitexp_all() -> dict[str, list[FitExperiment]]:
    """All data."""
    return fit_experiments(filter_empty)


def f_fitexp_control() -> dict[str, list[FitExperiment]]:
    """Control data."""
    return fit_experiments([filter_control])


def f_fitexp_pk() -> dict[str, list[FitExperiment]]:
    """HCTZ pharmacokinetics data."""
    return fit_experiments([filter_control, filter_pk])


def f_fitexp_pkiv() -> dict[str, list[FitExperiment]]:
    """HCTZ iv pharmacokinetics data."""
    return fit_experiments([filter_control, filter_pk, filter_iv])


def f_fitexp_pd() -> dict[str, list[FitExperiment]]:
    """HCTZ pharmacodynamics data."""
    return fit_experiments([filter_control, filter_pd])


if __name__ == "__main__":
    for f in [
        f_fitexp_all,
        f_fitexp_control,
        f_fitexp_pk,
        f_fitexp_pkiv,
        f_fitexp_pd,
    ]:
        console.rule(title=f.__name__, align="left", style="white")
        for sid, fitexps in f().items():
            console.print(f"{sid}: {fitexps}")
