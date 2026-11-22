"""Tests of the Fisher information of a fit."""

import json
import logging
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from sbmlsim import progress
from sbmlsim.engine import Engine
from sbmlsim.fit import FitSettings
from sbmlsim.fit.fisher import (
    CENTRAL_DIFFERENCES,
    DEFAULT_STEP,
    SENSITIVITIES,
    FisherInformation,
    default_step,
    evaluator,
    fisher_information,
    jacobian,
)
from sbmlsim.fit.metrics import aic, bic, bic_from_mse
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ParameterScaleType


def test_perfect_identifiable_fit_has_zero_errors_and_exact_intervals() -> None:
    information = FisherInformation(
        "perfect",
        "fit",
        ["p"],
        np.array([2.0]),
        ParameterScaleType.LINEAR,
        np.array([[1.0]]),
        cost=0.0,
        n=10,
    )
    df = information.summary_df
    assert df.se[0] == 0.0
    assert df.ci_lower[0] == df.ci_upper[0] == 2.0


@pytest.mark.parametrize("view", [False, True])
def test_fisher_reports_reuse_decompositions_and_track_matrix_changes(
    view: bool,
) -> None:
    matrix = np.eye(4)[::2, ::2] if view else np.eye(2)
    information = FisherInformation(
        "report",
        "fit",
        ["a", "b"],
        np.array([1.0, 2.0]),
        ParameterScaleType.LINEAR,
        matrix,
        cost=1.0,
        n=10,
    )
    with (
        patch("numpy.linalg.eigvalsh", wraps=np.linalg.eigvalsh) as eigenvalues,
        patch("numpy.linalg.pinv", wraps=np.linalg.pinv) as inverse,
    ):
        _ = information.summary_df
        _ = information.correlation
        assert eigenvalues.call_count == inverse.call_count == 1
        information.matrix[0, 0] = 2.0
        np.testing.assert_allclose(
            information.covariance, information.sigma2 * np.diag([0.5, 1.0])
        )
        assert eigenvalues.call_count == inverse.call_count == 2
        information.rank_tolerance = 0.75
        assert not information.is_identifiable
        _ = information.covariance
        assert eigenvalues.call_count == inverse.call_count == 3
        information.cost = 2.0
        np.testing.assert_allclose(
            information.covariance, information.sigma2 * np.diag([0.5, 1.0])
        )
        assert eigenvalues.call_count == inverse.call_count == 3


def test_bic_penalizes_more_than_aic() -> None:
    """The BIC charges `ln(n)` per parameter where the AIC charges `2`."""
    residuals = np.array([0.1, -0.2, 0.15, 0.05, -0.1, 0.2, -0.05, 0.12])
    # ln(8) = 2.08 > 2, so the BIC is the larger of the two
    assert bic(residuals, k=2) > aic(residuals, k=2)
    # and the difference is exactly the difference of the penalties
    assert bic(residuals, k=2) - aic(residuals, k=2) == pytest.approx(
        2 * (np.log(8) - 2)
    )


def test_bic_of_a_smaller_model() -> None:
    """A parameter which does not improve the fit costs `ln(n)`."""
    assert bic_from_mse(mse=0.01, n=100, k=3) - bic_from_mse(
        mse=0.01, n=100, k=2
    ) == pytest.approx(np.log(100))


def test_bic_requires_a_non_negative_mse() -> None:
    """A perfect fit has the best possible BIC, a negative MSE is an error."""
    assert bic_from_mse(mse=0.0, n=10, k=2) == float("-inf")
    with pytest.raises(ValueError, match="non-negative MSE"):
        bic_from_mse(mse=-1.0, n=10, k=2)
    with pytest.raises(ValueError, match="positive number of points"):
        bic_from_mse(mse=1.0, n=0, k=2)


def test_the_jacobian_has_a_column_per_parameter(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The jacobian of the residuals is one column per parameter."""
    problem = op_hctz_iv
    problem.initialize(fit_settings)
    x = problem.to_scale(np.asarray(problem.x0, dtype=float))

    jac = jacobian(problem, x)
    residuals = np.asarray(problem.residuals(x), dtype=float)
    assert jac.shape == (residuals.size, len(problem.pids))
    assert np.all(np.isfinite(jac))

    # the data of the problem is an intravenous dose, which says nothing about
    # the absorption of an oral one: the column of that parameter is zero
    sensitivity = dict(zip(problem.pids, np.max(np.abs(jac), axis=0), strict=True))
    assert sensitivity["GU__HCTZABS_k"] == 0.0
    # and it says a lot about the renal excretion
    assert sensitivity["KI__HCTZEX_k"] > 0.0


def test_the_information_of_another_engine_takes_the_differences(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A problem of roadrunner has no sensitivities, so python differentiates (#556)."""
    problem = op_hctz_pk
    # the differences are the python path, which a problem of the engine core
    # does not take: it integrates the sensitivities with its states
    problem.engine = Engine.ROADRUNNER
    problem.initialize(fit_settings)

    fim = fisher_information(problem, fit_settings, problem.parameter_set_model())

    assert fim.jacobian_of == CENTRAL_DIFFERENCES
    assert fim.step == pytest.approx(default_step(fit_settings.relative_tolerance))


def test_the_sensitivities_of_the_engine_can_be_required(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A required jacobian of the engine names what keeps the problem from it."""
    problem = op_hctz_pk
    # a problem of another engine than the core, so there is a gap to name
    problem.engine = Engine.ROADRUNNER
    problem.initialize(fit_settings)

    with pytest.raises(ValueError, match="engine: the problem simulates with"):
        fisher_information(
            problem,
            fit_settings,
            problem.parameter_set_model(),
            engine_jacobian=True,
        )


def test_the_fisher_information_of_the_example(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The information of the parameters the model starts from.

    The problem runs on `sbmlsim.engine.DEFAULT_ENGINE`, the engine core,
    which integrates the forward sensitivities of the residuals with their
    states: they are the jacobian of the information, exactly and without a
    step, where the central differences of python take a step of the noise of
    the integrator (`test_the_information_of_another_engine_takes_the_differences`).
    """
    problem = op_hctz_pk
    problem.initialize(fit_settings)
    fim = fisher_information(problem, fit_settings, problem.parameter_set_model())

    # the exact derivatives of the engine, which take no step
    assert fim.jacobian_of == SENSITIVITIES
    assert fim.step == 0.0
    assert fim.k == len(problem.pids)
    assert fim.targets == [p.target_id for p in problem.parameters]
    assert fim.matrix.shape == (fim.k, fim.k)
    # `J' J` is symmetric and positive semi-definite
    assert np.allclose(fim.matrix, fim.matrix.T)
    assert np.all(fim.eigenvalues >= -1e-8 * fim.eigenvalues[0])
    assert fim.eigenvalues[0] >= fim.eigenvalues[-1]
    assert fim.condition_number >= 1.0
    assert 0 <= fim.rank <= fim.k


def test_a_parameter_the_data_cannot_determine(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The information is rank deficient when the data misses a parameter.

    The intravenous data of the example determines the renal excretion and
    says nothing about the absorption of an oral dose, so the fit of these
    three parameters on this data is not identifiable.
    """
    problem = op_hctz_iv
    problem.initialize(fit_settings)
    fim = fisher_information(problem, fit_settings, problem.parameter_set_model())

    assert not fim.is_identifiable
    assert fim.rank < fim.k
    # a direction which the data does not constrain has no curvature
    assert fim.eigenvalues[-1] < fim.eigenvalues[0] * fim.rank_tolerance
    assert fim.condition_number > 1e8


def test_the_rank_deficiency_is_logged_once(
    caplog: pytest.LogCaptureFixture,
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
) -> None:
    """A report reads the covariance several times and warns about it once."""
    problem = op_hctz_iv
    problem.initialize(fit_settings)
    fim = fisher_information(problem, fit_settings, problem.parameter_set_model())
    assert not fim.is_identifiable

    with caplog.at_level(logging.WARNING, logger="sbmlsim.fit.fisher"):
        # every reader of the covariance, i.e. what a report evaluates
        assert fim.standard_errors is not None
        assert fim.correlation is not None
        assert fim.confidence_intervals() is not None
        assert fim.summary_df is not None

    warnings = [r for r in caplog.records if "rank" in r.getMessage()]
    assert len(warnings) == 1


def test_the_errors_and_the_intervals(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A parameter which the data determines has an interval around it."""
    problem = op_hctz_pk
    problem.initialize(fit_settings)
    fim = fisher_information(problem, fit_settings, problem.parameter_set_model())
    if not fim.is_identifiable:
        pytest.skip("the problem is not locally identifiable at these parameters")

    lower, upper = fim.confidence_intervals()
    assert np.all(np.isfinite(fim.standard_errors))
    # the interval contains the value, and on a logarithmic scale it is not
    # symmetric around it
    assert np.all(lower < fim.values)
    assert np.all(fim.values < upper)

    df = fim.summary_df
    assert list(df.parameter) == problem.pids
    assert np.allclose(df.value, fim.values)

    # the correlation of a parameter with itself is one
    corr = fim.correlation
    assert np.allclose(np.diag(corr.values), 1.0)
    assert np.all(np.abs(corr.values) <= 1.0 + 1e-8)


def test_the_information_is_json_serializable(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The matrix is stored with the parameters it belongs to."""
    problem = op_hctz_pk
    problem.initialize(fit_settings)
    fim = fisher_information(problem, fit_settings, problem.parameter_set_model())

    import json

    data = json.loads(json.dumps(fim.to_dict()))
    assert data["pids"] == problem.pids
    assert data["scale"] == fim.scale.name
    assert len(data["matrix"]) == fim.k


@pytest.mark.parametrize(
    ("scale", "values", "errors", "expected"),
    [
        # linear: se / |value|
        (ParameterScaleType.LINEAR, [2.0, 0.5], [0.2, 0.1], [0.1, 0.2]),
        # natural logarithm: se of ln(p)
        (ParameterScaleType.LOG, [2.0, 1e-4], [0.2, 0.1], [0.2, 0.1]),
        # decadic logarithm: ln(10) se of log10(p), whatever the value; the
        # value 1.15 (log10 0.06) and 1.85e-4 gave 134 % and 2.7 % before
        (
            ParameterScaleType.LOG10,
            [1.15, 1.85e-4],
            [0.0825, 0.104],
            [np.log(10) * 0.0825, np.log(10) * 0.104],
        ),
    ],
)
def test_the_cv_is_the_relative_error_of_the_parameter(
    scale: ParameterScaleType,
    values: list[float],
    errors: list[float],
    expected: list[float],
) -> None:
    """The cv does not depend on the unit or the magnitude of a parameter."""
    # a diagonal information with sigma2 = 1 has the errors 1/sqrt(diagonal)
    n, cost = 12, 5.0
    matrix = np.diag(1 / np.square(errors))
    information = FisherInformation(
        "cv", "fit", ["a", "b"], np.array(values), scale, matrix, cost=cost, n=n
    )
    np.testing.assert_allclose(information.standard_errors, errors)
    np.testing.assert_allclose(information.relative_errors, expected)
    np.testing.assert_allclose(
        information.summary_df.cv.to_numpy(), 100 * np.asarray(expected)
    )


def test_fisher_from_dict_round_trip(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The information read back from its record is the information."""
    import json

    problem = op_hctz_pk
    problem.initialize(fit_settings)
    fisher = fisher_information(
        problem, fit_settings, problem.parameter_set_model(), step=1e-4
    )
    fisher.rank_tolerance = 1e-3
    assert fisher.step == 1e-4

    record = json.loads(json.dumps(fisher.to_dict()))
    again = FisherInformation.from_dict(record)
    assert again.step == 1e-4
    assert again.jacobian_of == CENTRAL_DIFFERENCES
    del record["step"]
    del record["jacobian_of"]
    old = FisherInformation.from_dict(record)
    assert old.step == DEFAULT_STEP, "an old record"
    assert old.jacobian_of == CENTRAL_DIFFERENCES, "an old record"
    assert again.pids == fisher.pids
    assert again.scale == fisher.scale
    assert again.parameter_scales == fisher.parameter_scales
    assert again.units == fisher.units
    assert again.targets == fisher.targets
    assert (again.opid, again.sid, again.n, again.alpha, again.cost) == (
        fisher.opid,
        fisher.sid,
        fisher.n,
        fisher.alpha,
        fisher.cost,
    )
    np.testing.assert_allclose(again.values, fisher.values)
    np.testing.assert_allclose(again.matrix, fisher.matrix)
    assert again.rank_tolerance == 1e-3
    assert again.rank == fisher.rank
    assert again.condition_number == fisher.condition_number
    np.testing.assert_allclose(again.standard_errors, fisher.standard_errors)


def test_the_step_balances_the_noise_of_the_integrator() -> None:
    """A central difference divides the noise of the integrator by its step."""
    assert default_step(1e-6) == pytest.approx(1e-2)
    assert default_step(1e-9) == pytest.approx(1e-3)


def test_the_jacobian_on_workers_is_the_jacobian_of_this_process(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The points of a jacobian on a pool give the values of this process.

    The two points of every parameter are independent, so they are evaluated
    on the workers of a fit; the step of a point is the same either way, so
    the jacobian is the same bitwise.
    """
    problem = op_hctz_iv
    problem.initialize(fit_settings)
    x = problem.to_scale(np.asarray(problem.x0, dtype=float))

    serial = jacobian(problem, x)
    with evaluator(problem, fit_settings, n_cores=2) as evaluate:
        assert evaluate is not None
        on_workers = jacobian(problem, x, evaluate=evaluate)
    assert np.array_equal(serial, on_workers)


def test_the_jacobian_of_the_engine_core_is_one_batch_in_one_worker(
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The points of a jacobian of the engine core run on the threads of a worker.

    The engine simulates them at once, so the cores of the jacobian are the
    threads of one worker and the jacobian is the one of this process, see
    `sbmlsim.fit.runner.FitPool.threads`.
    """
    from sbmlsim.fit import runner

    problem = OptimizationProblem(
        opid=op_hctz_iv.opid,
        mapping_collections=op_hctz_iv.mapping_collections,
        fit_parameters=op_hctz_iv.parameters,
        base_path=op_hctz_iv.base_path,
        data_path=op_hctz_iv.data_path,
        engine=Engine.CORE,
    )
    problem.initialize(fit_settings)
    x = problem.to_scale(np.asarray(problem.x0, dtype=float))
    serial = jacobian(problem, x)
    sized: list[tuple[int, int]] = []
    real = runner.worker_pool

    def worker_pool(problem: Any, settings: Any, n_cores: int, threads: int = 1) -> Any:
        sized.append((n_cores, threads))
        return real(problem, settings, n_cores, threads)

    monkeypatch.setattr(runner, "worker_pool", worker_pool)
    with evaluator(problem, fit_settings, n_cores=2) as evaluate:
        assert evaluate is not None
        batched = jacobian(problem, x, evaluate=evaluate)

    assert sized == [(1, 2)]
    assert np.array_equal(serial, batched)


def test_the_points_of_a_jacobian_report_their_progress(
    recorder, op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The points of a jacobian in this process are the steps of an activity."""
    problem = op_hctz_iv
    problem.initialize(fit_settings)
    x = problem.to_scale(np.asarray(problem.x0, dtype=float))

    with progress.reporter(recorder):
        jacobian(problem, x)
    jacobians = [event for event in recorder.events if event.kind == "jacobian"]
    assert jacobians[0].total == 2 * len(problem.pids)
    assert jacobians[0].label == "Fisher information"
    assert jacobians[-1].completed == 2 * len(problem.pids)


def _information(matrix: np.ndarray) -> FisherInformation:
    """Get a Fisher information of a matrix, for the tests of its decomposition."""
    return FisherInformation(
        opid="op",
        sid="set",
        pids=[f"p{k}" for k in range(matrix.shape[0])],
        values=np.ones(matrix.shape[0]),
        scale=ParameterScaleType.LOG10,
        matrix=matrix,
        cost=1.25,
        n=40,
    )


def _sloppy_matrix() -> np.ndarray:
    """Get a Fisher information of a fit whose parameters are nearly collinear."""
    rng = np.random.default_rng(7)
    jacobian = rng.normal(size=(40, 4))
    jacobian[:, 2] = jacobian[:, 0] * 1.000001 + 1e-6 * rng.normal(size=40)
    return np.asarray(jacobian.T @ jacobian)


def test_the_decomposition_of_the_matrix_is_stored_with_the_information() -> None:
    """A record carries the pseudo-inverse and the eigenvalues of its matrix.

    The SVD and the eigenvalue decomposition of LAPACK depend on the kernels
    of the CPU, so an information which computed them at every read gave
    standard errors which differ in their last digits on another machine, and
    a model whose factory writes them into its SBML did not rebuild byte for
    byte (#523).
    """
    information = _information(_sloppy_matrix())
    stored = information.to_dict()
    assert np.shape(stored["inverse"]) == (4, 4)
    assert np.shape(stored["spectrum"]) == (4,)

    back = FisherInformation.from_dict(json.loads(json.dumps(stored)))
    np.testing.assert_array_equal(back.standard_errors, information.standard_errors)
    np.testing.assert_array_equal(back.eigenvalues, information.eigenvalues)
    np.testing.assert_array_equal(
        back.correlation.to_numpy(), information.correlation.to_numpy()
    )
    assert back.rank == information.rank


def test_a_record_of_the_decomposition_does_not_decompose_again() -> None:
    """The stored decomposition is read and not computed, which is the point of it.

    A reader on another machine must answer the numbers of the record and not
    the ones its own LAPACK computes for the same matrix.
    """
    stored = json.loads(json.dumps(_information(_sloppy_matrix()).to_dict()))
    back = FisherInformation.from_dict(stored)
    with (
        patch("numpy.linalg.eigvalsh", wraps=np.linalg.eigvalsh) as eigenvalues,
        patch("numpy.linalg.pinv", wraps=np.linalg.pinv) as inverse,
    ):
        _ = back.summary_df
        _ = back.correlation
        _ = back.standard_errors
        _ = back.rank
        assert eigenvalues.call_count == inverse.call_count == 0


def test_a_record_written_before_the_decomposition_computes_it() -> None:
    """A record of an earlier release has no decomposition and computes it as before."""
    stored = _information(_sloppy_matrix()).to_dict()
    legacy = {k: v for k, v in stored.items() if k not in ("inverse", "spectrum")}
    back = FisherInformation.from_dict(legacy)
    assert back.inverse is None
    assert back.spectrum is None
    np.testing.assert_allclose(
        back.standard_errors, _information(_sloppy_matrix()).standard_errors
    )


def test_a_matrix_changed_after_the_fact_drops_the_stored_decomposition() -> None:
    """The decomposition belongs to the matrix it was computed from."""
    stored = json.loads(json.dumps(_information(np.diag([1.0, 4.0])).to_dict()))
    back = FisherInformation.from_dict(stored)
    np.testing.assert_allclose(back.covariance, back.sigma2 * np.diag([1.0, 0.25]))
    back.matrix = np.diag([4.0, 16.0])
    np.testing.assert_allclose(back.covariance, back.sigma2 * np.diag([0.25, 0.0625]))
    assert back.inverse is None
    assert back.spectrum is None


def test_a_stored_inverse_of_another_shape_is_refused() -> None:
    """A decomposition which is not of the matrix is an error and not a wrong number."""
    with pytest.raises(ValueError, match="inverse"):
        FisherInformation(
            opid="op",
            sid="set",
            pids=["a", "b"],
            values=np.ones(2),
            scale=ParameterScaleType.LOG10,
            matrix=np.diag([1.0, 4.0]),
            cost=1.0,
            n=10,
            inverse=np.ones((3, 3)),
        )


def test_stored_eigenvalues_of_another_shape_are_refused() -> None:
    """One eigenvalue per parameter, or the rank of the record is not its rank."""
    with pytest.raises(ValueError, match="eigenvalues"):
        FisherInformation(
            opid="op",
            sid="set",
            pids=["a", "b"],
            values=np.ones(2),
            scale=ParameterScaleType.LOG10,
            matrix=np.diag([1.0, 4.0]),
            cost=1.0,
            n=10,
            spectrum=np.ones(3),
        )
