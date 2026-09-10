"""Parameter fit problems for HCTZ.

The data of a fit is selected from the fit mappings of the studies in three
steps, see `sbmlsim.fit.helpers`: the filters on the metadata of the mappings
select the training data of the fit and exclude the rest, the outliers are
named once for all mappings and are the training data which is not usable, and
the validation data is the part of the training data a fit is evaluated on but
not fitted to. The metadata of a mapping describes its curve, what a fit does
with the curve is decided here.
"""

import sys
from pathlib import Path

# run as a script (`python examples/hctz_fitting/fitting/mapping_collections.py`,
# the "run file" of an IDE) the repository is not on `sys.path`, so the
# `examples` package is not found
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.hctz_fitting import DATA_PATH, HCTZ_PATH
from examples.hctz_fitting.experiments.metadata import (
    Coadministration,
    HCTZMappingMetaData,
    Route,
)
from examples.hctz_fitting.experiments.studies import Beermann1976, Patel1984, Weir1998
from sbmlsim.console import console
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import FitMapping, FitMappingCollection
from sbmlsim.fit.helpers import MappingFilter, select_mapping_collections

# observables of the pharmacokinetics
PK_OBSERVABLES = {"Afeces_hctz", "Aurine_hctz", "Cve_hctz", "KI__HCTZEX"}

EXPERIMENT_CLASSES: list[type[SimulationExperiment]] = [
    Beermann1976,
    Patel1984,
    Weir1998,
]

#: outliers: the data is not usable. Tagged once for all mappings, an outlier
#: is not fitted in any fit whose filters select it, but simulated and
#: evaluated so that the decision can be checked against the model
OUTLIER_MAPPINGS: set[str] = {
    "fm_hctz5po_4",
    "fm_excretion_hctz5po_4",
}

#: validation data: kept out of the fits, the fits are evaluated on it.
#: The highest oral dose of Patel1984 checks how the parameters extrapolate,
#: and Weir1998 checks the multiple dosing: the fits are made on single doses,
#: so the accumulation over eleven doses every 12 hours is a prediction.
VALIDATION_MAPPINGS: set[str] = {
    "fm_200_tab_urine",
    "fm_200_sus_urine",
    "fm_Fig2_hctz25",
    "fm_Fig3_amount_cumulative_hctz25",
    "fm_Tab4_excretion_hctz25",
}


def mapping_collections(
    filters: MappingFilter | list[MappingFilter],
    validation: set[str] | MappingFilter = VALIDATION_MAPPINGS,
) -> dict[str, list[FitMappingCollection]]:
    """Select the data of a fit from the fit mappings of the studies.

    The filters select the training data, the mappings which fail them are
    excluded, e.g. the `_kombi` arms of Weir1998, which are hydrochlorothiazide
    with diltiazem and the model has no interaction for it. The outliers are
    `OUTLIER_MAPPINGS`, and the validation data is `VALIDATION_MAPPINGS` by
    default.

    Args:
        filters: filters of the training data.
        validation: keys or filter of the validation data.

    Returns:
        The fit mapping collections of all kinds by experiment id.
    """
    return select_mapping_collections(
        experiment_classes=EXPERIMENT_CLASSES,
        base_path=HCTZ_PATH,
        data_path=DATA_PATH,
        filters=filters,
        outliers=OUTLIER_MAPPINGS,
        validation=validation,
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


def filter_coadministration(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Only data without coadministration."""
    return _metadata(fit_mapping).coadministration == Coadministration.NONE


def filter_iv_po(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Only iv and po data."""
    return _metadata(fit_mapping).route in {Route.IV, Route.PO}


def filter_pk(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Only HCTZ PK data."""
    return _yid(fit_mapping) in PK_OBSERVABLES


def filter_pd(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Only HCTZ PD data."""
    return _yid(fit_mapping) not in PK_OBSERVABLES


def f_collections_all() -> dict[str, list[FitMappingCollection]]:
    """All data."""
    return mapping_collections([])


def f_collections_pk() -> dict[str, list[FitMappingCollection]]:
    """HCTZ pharmacokinetics data."""
    return mapping_collections([filter_coadministration, filter_iv_po, filter_pk])


def f_collections_pd() -> dict[str, list[FitMappingCollection]]:
    """HCTZ pharmacodynamics data."""
    return mapping_collections([filter_coadministration, filter_iv_po, filter_pd])


if __name__ == "__main__":
    for f in [
        f_collections_all,
        f_collections_pk,
        f_collections_pd,
    ]:
        console.rule(title=f.__name__, align="left", style="white")
        f()
