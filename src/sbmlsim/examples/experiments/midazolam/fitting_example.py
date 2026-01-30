from typing import Callable

from sbmlsim.examples.experiments.midazolam import MIDAZOLAM_PATH
from sbmlsim.examples.experiments.midazolam.fitting_problems import (
    op_kupferschmidt1995,
    op_mandema1992,
    op_mid1oh_iv,
)
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import (
    OptimizationAlgorithmType,
    ResidualType,
    WeightingCurvesType,
    WeightingPointsType,
)
from sbmlsim.fit.runner import run_optimization


def fitting_example(op_factory: Callable, size: int = 10, n_cores: int = 10) -> None:
    """Demonstrate fitting functionality."""

    fit_kwargs = {
        "seed": 1234,
        "residual": ResidualType.NORMALIZED,
        "weighting_curves": [WeightingCurvesType.POINTS],
        "weighting_points": WeightingPointsType.ERROR_WEIGHTING,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-6,
    }

    for alg_key, algorithm in [
        ("lsq", OptimizationAlgorithmType.LEAST_SQUARE),
        ("de", OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION),
    ]:
        op: OptimizationProblem = op_factory()
        fit_path = MIDAZOLAM_PATH / "results_fit" / op.opid / alg_key
        if not fit_path.exists():
            fit_path.mkdir(parents=True)

        run_optimization(
            problem=op, algorithm=algorithm, size=size, n_cores=n_cores, **fit_kwargs
        )
        # OptimizationAnalysis(opt_result=opt_result, op=op)


if __name__ == "__main__":
    for op_factory in [op_mandema1992, op_kupferschmidt1995, op_mid1oh_iv]:
        fitting_example(op_factory=op_factory)
