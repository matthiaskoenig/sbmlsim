"""Test fit."""

from pathlib import Path
from typing import Any

import numpy as np
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
from sbmlsim.fit.sampling import SamplingType

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
def test_fit_settings(
    settings: FitSettings, op_hctz_pk: OptimizationProblem, short_fit: dict[str, Any]
) -> None:
    """Test various settings of the optimization problem."""
    op = op_hctz_pk
    opt_result: OptimizationResult = run_optimization(
        problem=op,
        settings=settings,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        n_cores=1,
        serial=True,
        **short_fit,
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


def test_initialize_requires_settings(op_hctz_pk: OptimizationProblem) -> None:
    """The problem is initialized with `FitSettings`."""
    with pytest.raises(TypeError, match="FitSettings"):
        op_hctz_pk.initialize(settings={"residual": "ABSOLUTE"})  # ty: ignore[invalid-argument-type]


def test_initialize_is_idempotent(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Initializing again with the same settings does not repeat the work."""
    op = op_hctz_pk
    op.initialize(fit_settings)
    n_mappings = len(op.mapping_keys)
    assert n_mappings > 0
    runner = op.runner

    op.initialize(fit_settings)
    assert len(op.mapping_keys) == n_mappings
    # the experiments were not created again
    assert op.runner is runner


def test_initialize_new_settings(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Initializing with other settings resolves the mappings again."""
    op = op_hctz_pk
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
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    short_fit: dict[str, Any],
) -> None:
    """Test the various loss functions."""
    op = op_hctz_pk
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
        **short_fit,
    )
    assert opt_result
    assert op.loss_function == loss_function
    # the loss functions are finite on signed residuals
    assert all(fit.cost >= 0.0 for fit in opt_result.fits)
    assert opt_result.df_fits.cost.notna().all()


def test_fit_lsq_serial(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Test serial least square fit."""
    opt_result: OptimizationResult = run_optimization(
        problem=op_hctz_pk,
        settings=fit_settings,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        n_cores=1,
        serial=True,
    )
    assert opt_result is not None


def test_fit_de_serial(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Test serial differential evolution fit."""
    opt_result: OptimizationResult = run_optimization(
        problem=op_hctz_pk,
        settings=fit_settings,
        algorithm=OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION,
        size=1,
        n_cores=1,
        serial=True,
        # the smallest population and no polishing by least squares, which
        # would be most of the evaluations
        maxiter=2,
        popsize=2,
        polish=False,
    )
    assert opt_result is not None


@pytest.mark.parametrize("show_progress", [True, False])
def test_fit_lsq_parallel(
    show_progress: bool,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    short_fit: dict[str, Any],
) -> None:
    """Test parallel least square fit, with and without the progress display."""
    opt_result: OptimizationResult = run_optimization(
        problem=op_hctz_pk,
        settings=fit_settings,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=2,
        n_cores=2,
        serial=False,
        show_progress=show_progress,
        **short_fit,
    )
    assert opt_result is not None
    assert opt_result.size == 2


def test_one_worker_runs_without_a_pool(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    short_fit: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fit with one worker is serial and fits what a serial fit fits."""
    from sbmlsim.fit import runner

    serial = runner.run_optimization(
        problem=op_hctz_pk,
        settings=fit_settings,
        size=2,
        seed=1234,
        serial=True,
        show_progress=False,
        **short_fit,
    )

    def no_pool(**kwargs: Any) -> None:
        raise AssertionError("a fit with one worker started a pool")

    monkeypatch.setattr(runner, "_run_optimization_parallel", no_pool)
    one = runner.run_optimization(
        problem=op_hctz_pk,
        settings=fit_settings,
        size=2,
        seed=1234,
        n_cores=1,
        show_progress=False,
        **short_fit,
    )
    np.testing.assert_allclose(one.xopt, serial.xopt)


@pytest.mark.parametrize(
    ("algorithm", "kwargs", "unknown"),
    [
        # removed arguments of an earlier version
        (
            OptimizationAlgorithmType.LEAST_SQUARE,
            {"weighting_local": WeightingPointsType.NO_WEIGHTING},
            "weighting_local",
        ),
        (OptimizationAlgorithmType.LEAST_SQUARE, {"fitting_type": "x"}, "fitting_type"),
        # an argument of the other optimizer
        (OptimizationAlgorithmType.LEAST_SQUARE, {"maxiter": 2}, "maxiter"),
        (OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION, {"max_nfev": 3}, "max_nfev"),
        # what sbmlsim sets itself
        (OptimizationAlgorithmType.LEAST_SQUARE, {"bounds": ([0.0], [1.0])}, "bounds"),
    ],
)
def test_unknown_optimizer_arguments(
    algorithm: OptimizationAlgorithmType,
    kwargs: dict[str, Any],
    unknown: str,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    tmp_path: Path,
) -> None:
    """An argument the optimizer does not accept is rejected before any run."""
    with pytest.raises(TypeError, match=unknown) as err:
        run_optimization(
            problem=op_hctz_pk,
            settings=fit_settings,
            algorithm=algorithm,
            serial=True,
            runs_dir=tmp_path,
            **kwargs,
        )
    assert algorithm.name in str(err.value)
    # no repeat ran
    assert not list(tmp_path.iterdir())


def test_optimizer_arguments_reach_the_optimizer(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    short_fit: dict[str, Any],
) -> None:
    """An argument of the optimizer is accepted and limits the run."""
    assert short_fit == {"max_nfev": 3}
    opt_result = run_optimization(
        problem=op_hctz_pk,
        settings=fit_settings,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        serial=True,
        **short_fit,
    )
    (fit,) = opt_result.fits
    # a run which failed has no evaluations
    assert 0 < fit.nfev <= 3


def test_sampling_is_not_an_optimizer_argument(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    short_fit: dict[str, Any],
) -> None:
    """The sampling of the start values is an argument of the fit, not of scipy."""
    opt_result = run_optimization(
        problem=op_hctz_pk,
        settings=fit_settings,
        size=1,
        serial=True,
        sampling=SamplingType.START,
        **short_fit,
    )
    (fit,) = opt_result.fits
    # the run started from the start values of the parameters
    assert list(fit.x0) == pytest.approx(op_hctz_pk.x0)


def test_estimate_total_time() -> None:
    """The estimate is the time per batch times the number of batches."""
    from sbmlsim.fit.runner import estimate_total_time

    # nothing done yet, no estimate
    assert estimate_total_time(elapsed=10.0, completed=0, total=8) is None
    assert estimate_total_time(elapsed=0.0, completed=1, total=8) is None
    # a single worker: the mean time per run times the runs
    assert estimate_total_time(elapsed=20.0, completed=2, total=8) == 80.0
    # four workers process four runs at once, so the first run which is done
    # is the first batch and the estimate is not four times too large
    assert estimate_total_time(elapsed=10.0, completed=1, total=8, workers=4) == 20.0
    assert estimate_total_time(elapsed=10.0, completed=4, total=8, workers=4) == 20.0
    assert estimate_total_time(elapsed=22.0, completed=5, total=8, workers=4) == 22.0
    # the estimate is the elapsed time once everything is done
    assert estimate_total_time(elapsed=42.0, completed=8, total=8, workers=4) == 42.0


def test_total_time_column() -> None:
    """The progress shows the estimated total runtime."""
    from rich.progress import Progress

    from sbmlsim.fit.runner import TotalTimeColumn

    clock = [0.0]
    progress = Progress(TotalTimeColumn(), get_time=lambda: clock[0])
    task_id = progress.add_task("optimizing", total=4, workers=2)
    task = progress.tasks[0]
    column = TotalTimeColumn()

    assert str(column.render(task)) == "~ -:--:-- total"
    clock[0] = 90.0
    progress.update(task_id, completed=1)
    # one of two batches done after 90 seconds
    assert str(column.render(task)) == "~ 0:03:00 total"
    clock[0] = 200.0
    progress.update(task_id, completed=4)
    assert str(column.render(task)) == "~ 0:03:20 total"
