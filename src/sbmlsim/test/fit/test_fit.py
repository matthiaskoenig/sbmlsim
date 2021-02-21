import pytest

from sbmlsim.examples.experiments.midazolam import fitting_parallel, fitting_serial
from sbmlsim.examples.experiments.midazolam.fitting_problems import (
    op_kupferschmidt1995,
    op_mandema1992,
    op_mid1oh_iv,
)
from sbmlsim.fit.fit import process_optimization_result
from sbmlsim.fit.optimization import FittingType, ResidualType, WeightingLocalType


fit_kwargs_testdata = [
    # fitting_type
    {
        "fitting_type": FittingType.ABSOLUTE_VALUES,
        "residual_type": ResidualType.ABSOLUTE_NORMED_RESIDUALS,
        "weighting_local": WeightingLocalType.ABSOLUTE_ONE_OVER_WEIGHTING,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-6,
    },
    {
        "fitting_type": FittingType.ABSOLUTE_CHANGES_BASELINE,
        "residual_type": ResidualType.ABSOLUTE_NORMED_RESIDUALS,
        "weighting_local": WeightingLocalType.ABSOLUTE_ONE_OVER_WEIGHTING,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-6,
    },
    {
        "fitting_type": FittingType.RELATIVE_CHANGES_BASELINE,
        "residual_type": ResidualType.ABSOLUTE_NORMED_RESIDUALS,
        "weighting_local": WeightingLocalType.ABSOLUTE_ONE_OVER_WEIGHTING,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-6,
    },
    # residual_type
    {
        "fitting_type": FittingType.ABSOLUTE_VALUES,
        "residual_type": ResidualType.ABSOLUTE_RESIDUALS,
        "weighting_local": WeightingLocalType.ABSOLUTE_ONE_OVER_WEIGHTING,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-6,
    },
    {
        "fitting_type": FittingType.ABSOLUTE_VALUES,
        "residual_type": ResidualType.RELATIVE_RESIDUALS,
        "weighting_local": WeightingLocalType.ABSOLUTE_ONE_OVER_WEIGHTING,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-6,
    },
    # weighing local FIXME: rename to weighting_points
    #     NO_WEIGHTING
    #     ABSOLUTE_ONE_OVER_WEIGHTING
    #     RELATIVE_ONE_OVER_WEIGHTING
    {
        "fitting_type": FittingType.ABSOLUTE_VALUES,
        "residual_type": ResidualType.ABSOLUTE_NORMED_RESIDUALS,
        "weighting_local": WeightingLocalType.NO_WEIGHTING,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-6,
    },
    {
        "fitting_type": FittingType.ABSOLUTE_VALUES,
        "residual_type": ResidualType.ABSOLUTE_NORMED_RESIDUALS,
        "weighting_local": WeightingLocalType.RELATIVE_ONE_OVER_WEIGHTING,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-6,
    },
]


@pytest.mark.parametrize("fit_kwargs", fit_kwargs_testdata)
def test_fit_settings(fit_kwargs):
    """Test various arguments to optimization problem."""
    opt_res = fitting_serial.fit_lsq(problem_factory=op_mid1oh_iv, **fit_kwargs)
    assert opt_res is not None


fit_kwargs_default = {
    "fitting_type": FittingType.ABSOLUTE_VALUES,
    "residual_type": ResidualType.ABSOLUTE_NORMED_RESIDUALS,
    "weighting_local": WeightingLocalType.ABSOLUTE_ONE_OVER_WEIGHTING,
    "absolute_tolerance": 1e-6,
    "relative_tolerance": 1e-6,
}


def test_process_optimization(tmp_path):
    opt_res, problem = fitting_serial.fit_lsq(
        problem_factory=op_mid1oh_iv, **fit_kwargs_default
    )
    process_optimization_result(
        opt_res, problem=problem, output_path=tmp_path, **fit_kwargs_default
    )


def test_fit_lsq_serial():
    opt_res = fitting_serial.fit_lsq(problem_factory=op_mid1oh_iv, **fit_kwargs_default)
    assert opt_res is not None


def test_fit_de_serial():
    opt_res = fitting_serial.fit_de(problem_factory=op_mid1oh_iv, **fit_kwargs_default)
    assert opt_res is not None


def test_fit_lsq_parallel():
    opt_res = fitting_parallel.fit_lsq(
        problem_factory=op_mid1oh_iv, **fit_kwargs_default
    )
    assert opt_res is not None


def test_fit_de_parallel():
    opt_res = fitting_parallel.fit_de(
        problem_factory=op_mid1oh_iv, **fit_kwargs_default
    )
    assert opt_res is not None
