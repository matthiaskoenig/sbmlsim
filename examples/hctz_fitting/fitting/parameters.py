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


#: the dissolution estimated for the oral data only. A selector must be a
#: module level function: the workers of a parallel fit unpickle it.
#:
#: There is no intravenous version: `Ka_dis_hctz` is the dissolution rate of
#: the oral dose, and an intravenous dose has nothing to dissolve, so it has
#: no effect on the intravenous curves at all
#: (`tests/fit/test_parameter_mapping.py::test_the_versions_reach_their_own_simulations`
#: measures this directly). Versioning it for the intravenous data too, as an
#: earlier revision of this example did, would add a second parameter no
#: curve constrains -- an objective flat in it. Leaving the intravenous
#: mappings unversioned instead keeps them on the model's shared value, which
#: is the correct value for them, and is exactly what
#: `problem.parameter_mapping.coverage()` reports as uncovered.
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
    *[p for p in PARAMETERS if p.pid != "Ka_dis_hctz"],
]
