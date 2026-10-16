"""Tests of the Fisher information of a fit."""

import logging
from unittest.mock import patch

import numpy as np
import pytest

from sbmlsim.fit import FitSettings
from sbmlsim.fit.fisher import FisherInformation, fisher_information, jacobian
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


def test_the_fisher_information_of_the_example(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The information of the parameters the model starts from."""
    problem = op_hctz_pk
    problem.initialize(fit_settings)
    fim = fisher_information(problem, fit_settings, problem.parameter_set_model())

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
    fisher = fisher_information(problem, fit_settings, problem.parameter_set_model())

    again = FisherInformation.from_dict(json.loads(json.dumps(fisher.to_dict())))
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
