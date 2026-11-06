"""HCTZ parameter fitting.

The fit problems of the HCTZ model: which fit mapping collections enter a fit
and which parameters are adjusted. Everything else is the general fit runner of
`sbmlsim.fit.cli`.

    python -m examples.hctz_fitting.fitting.fitting --subset=PK --runs=4 --cores=2 \
        --seed=1234 --method=LSQ --strategy=ALL --name=PK_LSQ_ALL

The results, figures and the HTML report are written into `results/fit` in the
working directory.
"""

import sys
from pathlib import Path

# run as a script (`python examples/hctz_fitting/fitting/fitting.py`, the "run file" of an
# IDE) the repository is not on `sys.path`, so the `examples` package is not found
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.hctz_fitting import DATA_PATH, HCTZ_PATH
from examples.hctz_fitting.experiments.metadata import HCTZMappingMetaData, Route
from examples.hctz_fitting.experiments.studies import Beermann1976
from examples.hctz_fitting.fitting.mapping_collections import (
    f_collections_pk,
)
from examples.hctz_fitting.fitting.parameters import PARAMETERS
from sbmlsim.fit import FitMapping, FitMappingCollection, FitSettings
from sbmlsim.fit.cli import FitDefinition, fit_cli
from sbmlsim.fit.evaluation import Grouping
from sbmlsim.fit.helpers import select_mapping_collections
from sbmlsim.fit.options import (
    LossFunctionType,
    ResidualType,
    WeightingCurvesType,
    WeightingPointsType,
)

#: settings of the fits, stored with the result and read back by the report
FIT_SETTINGS = FitSettings(
    residual=ResidualType.NORMALIZED,
    loss_function=LossFunctionType.LINEAR,
    weighting_curves=(
        WeightingCurvesType.MAPPING,  # user defined weights
        WeightingCurvesType.POINTS,  # number of points
    ),
    # mappings without errors are weighted with CV=0.5
    weighting_points=WeightingPointsType.ERROR_WEIGHTING,
    variable_step_size=True,
    relative_tolerance=1e-6,
    absolute_tolerance=1e-6,
)


def filter_iv(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Only the iv data.

    A module level function, not a lambda, because a filter is called in the
    workers of a parallel fit, which get the definition pickled.
    """
    metadata = fit_mapping.metadata
    return isinstance(metadata, HCTZMappingMetaData) and metadata.route is Route.IV


def f_collections_iv() -> dict[str, list[FitMappingCollection]]:
    """The iv data of Beermann1976, the rest is excluded."""
    return select_mapping_collections(
        experiment_classes=[Beermann1976],
        base_path=HCTZ_PATH,
        data_path=DATA_PATH,
        filters=[filter_iv],
        print_info=False,
    )


#: the fit problems of the HCTZ model, the keys are the `--subset` argument,
#: the first one is its default
FIT_DEFINITIONS: dict[str, FitDefinition] = {
    # all pharmacokinetics data
    "PK": FitDefinition(
        mapping_collections=f_collections_pk,
        parameters=PARAMETERS,
        base_path=HCTZ_PATH,
        data_path=DATA_PATH,
        settings=FIT_SETTINGS,
        # the reports evaluate the data per tissue: plasma, urine and feces
        groupings=(Grouping.by("tissue"),),
    ),
    # the iv data of Beermann1976, a small problem of four training mappings
    "IV": FitDefinition(
        mapping_collections=f_collections_iv,
        parameters=PARAMETERS,
        base_path=HCTZ_PATH,
        data_path=DATA_PATH,
        settings=FIT_SETTINGS,
    ),
}


def main() -> None:
    """Run a fit of the HCTZ model."""
    fit_cli(FIT_DEFINITIONS)


if __name__ == "__main__":
    main()
