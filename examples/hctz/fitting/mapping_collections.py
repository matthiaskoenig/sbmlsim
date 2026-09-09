"""Parameter fit problems for HCTZ.

The fit experiments are built from the fit mappings of the studies which pass a
set of filters on the metadata of the mappings. The metadata of a mapping
describes its curve, what a fit does with the curve is decided here: every
selection gets a `MappingKind`, i.e., it is the training data of the fit, the
validation data it is evaluated on, or an outlier which is not used.
"""

from examples.hctz import DATA_PATH, HCTZ_PATH
from examples.hctz.experiments.metadata import (
    Coadministration,
    Fasting,
    HCTZMappingMetaData,
    Route,
)
from examples.hctz.experiments.studies import Beermann1976, Patel1984, Weir1998
from sbmlsim.console import console
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import FitMapping, FitMappingCollection, MappingKind
from sbmlsim.fit.helpers import (
    MappingFilter,
    f_collection,
    filter_empty,
    filter_keys,
    filter_not_keys,
    mapping_collections_by_kind,
)

# observables of the pharmacokinetics
PK_OBSERVABLES = {"Afeces_hctz", "Aurine_hctz", "Cve_hctz", "KI__HCTZEX"}

EXPERIMENT_CLASSES: list[type[SimulationExperiment]] = [
    Beermann1976,
    Patel1984,
    Weir1998,
]

#: mappings which are not used, the data is not usable
OUTLIER_MAPPINGS: set[str] = {
    "fm_hctz5po_4",
    "fm_excretion_hctz5po_4",
}

#: mappings which are kept out of the fits, the fits are evaluated on them.
#: The highest oral dose of Patel1984 checks how the parameters extrapolate,
#: and Weir1998 checks the multiple dosing: the fits are made on single doses,
#: so the accumulation over eleven doses every 12 hours is a prediction.
#: The `_kombi` arms of Weir1998 are hydrochlorothiazide with diltiazem,
#: which the filter of the coadministration removes before this.
VALIDATION_MAPPINGS: set[str] = {
    "fm_200_tab_urine",
    "fm_200_sus_urine",
    "fm_Fig2_hctz25",
    "fm_Fig3_amount_cumulative_hctz25",
    "fm_Tab4_excretion_hctz25",
}


def mapping_collections(
    metadata_filters: MappingFilter | list[MappingFilter],
    kind: MappingKind = MappingKind.TRAINING,
) -> dict[str, list[FitMappingCollection]]:
    """Get the fit experiments of the studies for the given filters."""
    return f_collection(
        experiment_classes=EXPERIMENT_CLASSES,
        metadata_filters=metadata_filters,
        base_path=HCTZ_PATH,
        data_path=DATA_PATH,
        kind=kind,
    )


def classified_mapping_collections(
    metadata_filters: list[MappingFilter],
) -> dict[str, list[FitMappingCollection]]:
    """Split a selection of mappings into training, validation and outliers.

    The outliers and the validation data are named in `OUTLIER_MAPPINGS` and
    `VALIDATION_MAPPINGS`, everything else the filters accept is fitted.

    Args:
        metadata_filters: filters which select the data of the fit.

    Returns:
        The fit experiments of the three kinds by experiment id.
    """
    excluded = OUTLIER_MAPPINGS | VALIDATION_MAPPINGS
    return mapping_collections_by_kind(
        experiment_classes=EXPERIMENT_CLASSES,
        base_path=HCTZ_PATH,
        data_path=DATA_PATH,
        filters_by_kind={
            MappingKind.TRAINING: [*metadata_filters, filter_not_keys(excluded)],
            MappingKind.VALIDATION: [
                *metadata_filters,
                filter_keys(VALIDATION_MAPPINGS),
            ],
            MappingKind.OUTLIER: [*metadata_filters, filter_keys(OUTLIER_MAPPINGS)],
        },
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
    return metadata.coadministration == Coadministration.NONE


def filter_iv(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Only IV application data."""
    return _metadata(fit_mapping).route == Route.IV


def filter_pk(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Only HCTZ PK data."""
    return _yid(fit_mapping) in PK_OBSERVABLES


def filter_pd(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Only HCTZ PD data."""
    return _yid(fit_mapping) not in PK_OBSERVABLES


def f_collections_all() -> dict[str, list[FitMappingCollection]]:
    """All data."""
    return mapping_collections(filter_empty)


def f_collections_control() -> dict[str, list[FitMappingCollection]]:
    """Control data."""
    return mapping_collections([filter_control])


def f_collections_pk() -> dict[str, list[FitMappingCollection]]:
    """HCTZ pharmacokinetics data, split into training, validation and outliers."""
    return classified_mapping_collections([filter_control, filter_pk])


def f_collections_pkiv() -> dict[str, list[FitMappingCollection]]:
    """HCTZ iv pharmacokinetics data."""
    return classified_mapping_collections([filter_control, filter_pk, filter_iv])


def f_collections_pd() -> dict[str, list[FitMappingCollection]]:
    """HCTZ pharmacodynamics data."""
    return mapping_collections([filter_control, filter_pd])


if __name__ == "__main__":
    for f in [
        f_collections_all,
        f_collections_control,
        f_collections_pk,
        f_collections_pkiv,
        f_collections_pd,
    ]:
        console.rule(title=f.__name__, align="left", style="white")
        for sid, collections in f().items():
            console.print(f"{sid}: {collections}")
