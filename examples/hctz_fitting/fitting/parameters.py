"""Parameters to optimize."""

from examples.hctz_fitting.experiments.metadata import Route
from examples.hctz_fitting.fitting.mapping_collections import _metadata
from sbmlsim.fit import FitMapping, FitParameter

PARAMETERS: list[FitParameter] = [
    # absorption
    FitParameter(
        pid="Ka_dis_hctz",
        start_value=1.0,
        lower_bound=1e-4,
        upper_bound=10,
        unit="1/hr",
    ),
    FitParameter(
        pid="GU__HCTZABS_k",
        lower_bound=1e-4,
        start_value=0.02,
        upper_bound=10,
        unit="1/min",
    ),
    FitParameter(
        pid="KI__HCTZEX_k",
        lower_bound=1e-10,
        start_value=1e-6,
        upper_bound=1,
        unit="1/ml",
    ),
]


def is_oral(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Select the oral data, which is where a dissolution rate applies."""
    return _metadata(fit_mapping).route == Route.PO


def is_intravenous(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Select the intravenous data."""
    return _metadata(fit_mapping).route == Route.IV


#: the parameters with the dissolution estimated per route. A selector must be
#: a module level function: the workers of a parallel fit unpickle it
PARAMETERS_BY_ROUTE: list[FitParameter] = [
    FitParameter(
        pid="Ka_dis_hctz_po",
        start_value=0.35,
        lower_bound=0.01,
        upper_bound=10.0,
        unit="1/hr",
        target="Ka_dis_hctz",
        mappings=is_oral,
    ),
    FitParameter(
        pid="Ka_dis_hctz_iv",
        start_value=0.35,
        lower_bound=0.01,
        upper_bound=10.0,
        unit="1/hr",
        target="Ka_dis_hctz",
        mappings=is_intravenous,
    ),
    *[p for p in PARAMETERS if p.pid != "Ka_dis_hctz"],
]
