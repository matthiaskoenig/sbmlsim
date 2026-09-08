"""HCTZ parameter fitting.

The fit problems of the HCTZ model: which fit experiments enter a fit and which
parameters are adjusted. Everything else is the general fit runner of
`sbmlsim.fit.cli`.

    python -m examples.hctz.fitting.fitting --subset=PK --runs=4 --cores=2 \
        --seed=1234 --method=LSQ --strategy=ALL --name=PK_LSQ_ALL

The results, figures and the HTML report are written into `results/fit` in the
working directory.
"""

import sys
from pathlib import Path

# run as a script (`python examples/hctz/fitting/fitting.py`, the "run file" of an
# IDE) the repository is not on `sys.path`, so the `examples` package is not found
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.hctz import DATA_PATH, HCTZ_PATH
from examples.hctz.fitting.fit_experiments import f_fitexp_pk, f_fitexp_pkiv
from examples.hctz.fitting.parameters import parameters_pk
from sbmlsim.fit import FitSettings
from sbmlsim.fit.cli import FitDefinition, fit_cli
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

#: the fit problems of the HCTZ model, the keys are the `--subset` argument
FIT_DEFINITIONS: dict[str, FitDefinition] = {
    # all pharmacokinetics data
    "PK": FitDefinition(
        fit_experiments=f_fitexp_pk,
        parameters=parameters_pk,
        base_path=HCTZ_PATH,
        data_path=DATA_PATH,
        settings=FIT_SETTINGS,
    ),
    # pharmacokinetics data of the iv studies
    "PKIV": FitDefinition(
        fit_experiments=f_fitexp_pkiv,
        parameters=parameters_pk,
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
