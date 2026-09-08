"""Test fit."""

from typing import Any

import pytest

from sbmlsim.fit import FitSettings
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import (
    LossFunctionType,
    OptimizationAlgorithmType,
    ResidualType,
    WeightingCurvesType,
    WeightingPointsType,
)
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.fit.runner import run_optimization

settings_testdata: list[FitSettings] = [
    FitSettings(
        residual=residual_type,
        weighting_curves=weighting_curves,
        weighting_points=weighting_points,
    )
    for residual_type in [
        ResidualType.ABSOLUTE,
        ResidualType.NORMALIZED,
        # ResidualType.ABSOLUTE_TO_BASELINE,
        # ResidualType.NORMALIZED_TO_BASELINE,
    ]
    for weighting_curves in [
        (),
        (WeightingCurvesType.POINTS,),
        (WeightingCurvesType.MAPPING,),
        (WeightingCurvesType.POINTS, WeightingCurvesType.MAPPING),
    ]
    for weighting_points in [
        WeightingPointsType.NO_WEIGHTING,
        WeightingPointsType.ERROR_WEIGHTING,
    ]
]


@pytest.mark.parametrize("settings", settings_testdata)
def test_fit_settings(settings: FitSettings, op_hctz_pkiv: OptimizationProblem) -> None:
    """Test various settings of the optimization problem."""
    op = op_hctz_pkiv
    opt_result: OptimizationResult = run_optimization(
        problem=op,
        settings=settings,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        n_cores=1,
        serial=True,
    )

    assert opt_result is not None
    assert op.settings == settings
    assert op.residual == settings.residual
    assert op.weighting_curves == settings.weighting_curves
    assert op.weighting_points == settings.weighting_points
    # the settings travel with the result, the report needs them
    assert opt_result.settings == settings


def test_settings_round_trip() -> None:
    """The settings serialize to a dictionary and back."""
    settings = FitSettings(
        residual=ResidualType.NORMALIZED,
        loss_function=LossFunctionType.SOFT_L1,
        weighting_curves=(WeightingCurvesType.POINTS,),
        weighting_points=WeightingPointsType.ERROR_WEIGHTING,
    )
    assert FitSettings.from_dict(settings.to_dict()) == settings


def test_settings_normalize_weighting_curves() -> None:
    """A list of weightings is normalized to a tuple, so the settings compare."""
    settings = FitSettings(weighting_curves=[WeightingCurvesType.POINTS])
    assert settings.weighting_curves == (WeightingCurvesType.POINTS,)
    assert settings == FitSettings(weighting_curves=(WeightingCurvesType.POINTS,))


def test_initialize_requires_settings(op_hctz_pkiv: OptimizationProblem) -> None:
    """The problem is initialized with `FitSettings`."""
    with pytest.raises(TypeError, match="FitSettings"):
        op_hctz_pkiv.initialize(settings={"residual": "ABSOLUTE"})  # ty: ignore[invalid-argument-type]


def test_initialize_is_idempotent(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Initializing again with the same settings does not repeat the work."""
    op = op_hctz_pkiv
    op.initialize(fit_settings)
    n_mappings = len(op.mapping_keys)
    assert n_mappings > 0
    runner = op.runner

    op.initialize(fit_settings)
    assert len(op.mapping_keys) == n_mappings
    # the experiments were not created again
    assert op.runner is runner


def test_initialize_new_settings(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Initializing with other settings resolves the mappings again."""
    op = op_hctz_pkiv
    op.initialize(fit_settings)
    n_mappings = len(op.mapping_keys)

    other = FitSettings(
        residual=ResidualType.ABSOLUTE,
        weighting_points=WeightingPointsType.NO_WEIGHTING,
    )
    op.initialize(other)
    assert op.settings == other
    assert len(op.mapping_keys) == n_mappings


@pytest.mark.parametrize(
    "loss_function",
    [
        LossFunctionType.LINEAR,
        LossFunctionType.SOFT_L1,
        LossFunctionType.CAUCHY,
        LossFunctionType.ARCTAN,
    ],
)
def test_loss_function(
    loss_function: LossFunctionType,
    op_hctz_pkiv: OptimizationProblem,
    fit_settings: FitSettings,
) -> None:
    """Test the various loss functions."""
    op = op_hctz_pkiv
    settings = FitSettings(
        residual=fit_settings.residual,
        loss_function=loss_function,
        weighting_curves=fit_settings.weighting_curves,
        weighting_points=fit_settings.weighting_points,
    )
    opt_result: OptimizationResult = run_optimization(
        problem=op,
        settings=settings,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        n_cores=1,
        serial=True,
    )
    assert opt_result
    assert op.loss_function == loss_function
    # the loss functions are finite on signed residuals
    assert all(fit.cost >= 0.0 for fit in opt_result.fits)
    assert opt_result.df_fits.cost.notna().all()


def test_fit_lsq_serial(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Test serial least square fit."""
    opt_result: OptimizationResult = run_optimization(
        problem=op_hctz_pkiv,
        settings=fit_settings,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        n_cores=1,
        serial=True,
    )
    assert opt_result is not None


def test_fit_de_serial(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Test serial differential evolution fit."""
    opt_result: OptimizationResult = run_optimization(
        problem=op_hctz_pkiv,
        settings=fit_settings,
        algorithm=OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION,
        size=1,
        n_cores=1,
        serial=True,
        maxiter=2,
    )
    assert opt_result is not None


@pytest.mark.parametrize("show_progress", [True, False])
def test_fit_lsq_parallel(
    show_progress: bool,
    op_hctz_pkiv: OptimizationProblem,
    fit_settings: FitSettings,
) -> None:
    """Test parallel least square fit, with and without the progress display."""
    opt_result: OptimizationResult = run_optimization(
        problem=op_hctz_pkiv,
        settings=fit_settings,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=2,
        n_cores=2,
        serial=False,
        show_progress=show_progress,
    )
    assert opt_result is not None
    assert opt_result.size == 2


def test_deprecated_arguments(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The removed arguments are reported."""
    kwargs: dict[str, Any] = {"weighting_local": WeightingPointsType.NO_WEIGHTING}
    with pytest.raises(ValueError, match="weighting_local"):
        run_optimization(
            problem=op_hctz_pkiv, settings=fit_settings, serial=True, **kwargs
        )
