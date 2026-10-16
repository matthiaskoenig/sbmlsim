"""Metrics of a fit.

The metrics are calculated for a parameter set on an initialized
`OptimizationProblem`, i.e., from the data of the fit mappings and the
predictions of the model for the parameters, see `FitMetrics`. The functions
which do the arithmetic work on plain arrays and are used on their own as well.

The names of the columns follow the convention of population pharmacokinetics:

    DV      the measured value (dependent variable)
    PRED    prediction of the population parameters, i.e., the parameter set
            which is shared by all fit mappings
    IPRED   prediction of the individual parameters, i.e., the parameter set of
            the fit mapping. Without individual parameters IPRED is PRED
    RES     DV - PRED
    IRES    DV - IPRED
    NRES    IRES / mean(DV) of the fit mapping, the normalized residual, so the
            residuals of curves of different magnitude are comparable
    IWRES   the residual of the cost of the optimization problem, i.e., IRES
            as the `ResidualType` of the settings defines it, weighted and
            with the loss function applied. The cost of the training data is
            `0.5 * sum(IWRES**2)`

The data of a fit spans orders of magnitude, so the absolute metrics `MSE`,
`RMSE` and `R2` are dominated by the curves with the largest values and a
curve of small values is missed by a factor of 20 at an absolute error which is
still tiny. `NRMSE`, the root mean square of `NRES`, is the scale free metric
which reads the same for every curve, and `RMSE_w` is the root mean square of
`IWRES`, i.e., what the fit minimizes. `within_2fold` is the percentage of the
data points whose prediction lies within twofold of the data, the criterion of
pharmacokinetic predictions, which reads the same for every curve as well; it
counts only the data points with a positive value and prediction, the others
have no ratio.

Note the sign: the residuals of `OptimizationProblem.residuals` are
`prediction - data`, the residuals here are `data - prediction`.
"""

from __future__ import annotations

import logging
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from sbmlsim.fit.objects import EVALUATED_KINDS, MappingKind
from sbmlsim.fit.parameters import ParameterSet

if TYPE_CHECKING:
    from sbmlsim.fit.optimization import OptimizationProblem

logger = logging.getLogger(__name__)


def _as_array(values: ArrayLike) -> np.ndarray:
    """Convert to a flat array of floats."""
    return np.asarray(values, dtype=float).ravel()


def sse(residuals: ArrayLike) -> float:
    """Sum of Squared Errors (SSE) of the residuals.

    Args:
        residuals: residuals of the fit.

    Returns:
        Sum of the squared residuals.
    """
    return float(np.sum(np.square(_as_array(residuals))))


def mse(residuals: ArrayLike) -> float:
    """Mean Squared Error (MSE) of the residuals.

    Args:
        residuals: residuals of the fit.

    Returns:
        Mean of the squared residuals.

    Raises:
        ValueError: if no residuals are given.
    """
    res = _as_array(residuals)
    if res.size == 0:
        raise ValueError("MSE requires at least one residual.")
    return float(np.mean(np.square(res)))


def rmse(residuals: ArrayLike) -> float:
    """Root Mean Squared Error (RMSE) of the residuals.

    Args:
        residuals: residuals of the fit.

    Returns:
        Square root of the mean squared error.
    """
    return rmse_from_mse(mse(residuals))


def rmse_from_mse(mse: float) -> float:
    """Root Mean Squared Error (RMSE) from the mean squared error.

    Args:
        mse: mean squared error of the fit.

    Returns:
        Square root of the mean squared error.

    Raises:
        ValueError: if the mean squared error is negative.
    """
    if mse < 0.0:
        raise ValueError(f"RMSE requires a non-negative MSE, but '{mse}' given.")
    return float(np.sqrt(mse))


def aic(residuals: ArrayLike, k: int) -> float:
    """Akaike Information Criterion (AIC) of the residuals.

    Args:
        residuals: residuals of the fit.
        k: number of fitted parameters.

    Returns:
        Akaike information criterion.
    """
    res = _as_array(residuals)
    return aic_from_mse(mse=mse(res), n=res.size, k=k)


def aic_from_mse(mse: float, n: int, k: int) -> float:
    """Akaike Information Criterion (AIC) from the mean squared error.

    The AIC is calculated for a least squares fit with normally distributed
    residuals, i.e., `AIC = n * ln(MSE) + 2 * k` up to an additive constant.
    Only differences of the AIC between models fitted on the same data are
    meaningful. A perfect fit, an MSE of 0, has the AIC `-inf`, the best
    possible value.

    Args:
        mse: mean squared error of the fit.
        n: number of data points.
        k: number of fitted parameters.

    Returns:
        Akaike information criterion, `-inf` for an MSE of 0.

    Raises:
        ValueError: if the mean squared error is negative or the number of data
            points is not positive.
    """
    if mse < 0.0:
        raise ValueError(f"AIC requires a non-negative MSE, but '{mse}' given.")
    if n <= 0:
        raise ValueError(f"AIC requires a positive number of points, but '{n}' given.")
    if mse == 0.0:
        return float("-inf")
    return float(n * np.log(mse) + 2 * k)


def bic(residuals: ArrayLike, k: int) -> float:
    """Bayesian Information Criterion (BIC) of the residuals.

    Args:
        residuals: residuals of the fit.
        k: number of fitted parameters.

    Returns:
        Bayesian information criterion.
    """
    res = _as_array(residuals)
    return bic_from_mse(mse=mse(res), n=res.size, k=k)


def bic_from_mse(mse: float, n: int, k: int) -> float:
    """Bayesian Information Criterion (BIC) from the mean squared error.

    The BIC is calculated for a least squares fit with normally distributed
    residuals, i.e., `BIC = n * ln(MSE) + k * ln(n)` up to the same additive
    constant as the AIC, so only differences between models fitted on the same
    data are meaningful.

    The BIC penalizes a parameter with `ln(n)` where the AIC penalizes it with
    `2`, i.e. it prefers the smaller model as soon as there are more than seven
    data points and increasingly so with more of them.

    Args:
        mse: mean squared error of the fit.
        n: number of data points.
        k: number of fitted parameters.

    Returns:
        Bayesian information criterion, `-inf` for an MSE of 0, a perfect fit.

    Raises:
        ValueError: if the mean squared error is negative or the number of data
            points is not positive.
    """
    if mse < 0.0:
        raise ValueError(f"BIC requires a non-negative MSE, but '{mse}' given.")
    if n <= 0:
        raise ValueError(f"BIC requires a positive number of points, but '{n}' given.")
    if mse == 0.0:
        return float("-inf")
    return float(n * np.log(mse) + k * np.log(n))


def r_squared(y_observed: ArrayLike, y_predicted: ArrayLike) -> float:
    """Coefficient of determination (R²) of a prediction.

    `R² = 1 - SSE / SST` with `SST` the total sum of squares of the data. The
    predictions of a non-linear model are not a linear regression of the data,
    so R² is not the square of a correlation and can be negative: a negative R²
    means the prediction is worse than the mean of the data.

    R² of fewer than two data points is `nan` without a message, e.g. a value
    per simulation against the one row of a study group; a warning is logged for
    several data points without variance.

    Args:
        y_observed: measured values.
        y_predicted: predicted values.

    Returns:
        Coefficient of determination, `nan` for fewer than two data points or
        data without variance.

    Raises:
        ValueError: if the data and the prediction have different lengths.
    """
    y_obs = _as_array(y_observed)
    y_pred = _as_array(y_predicted)
    if y_obs.size != y_pred.size:
        raise ValueError(
            f"R² requires as many predictions as data points, but got "
            f"'{y_pred.size}' for '{y_obs.size}'."
        )
    if y_obs.size < 2:
        return float("nan")
    sst = float(np.sum(np.square(y_obs - np.mean(y_obs))))
    if sst == 0.0:
        logger.warning("R² is undefined for data without variance.")
        return float("nan")
    return 1.0 - sse(y_obs - y_pred) / sst


def fold_counts(
    y_observed: ArrayLike, y_predicted: ArrayLike, fold: float = 2.0
) -> tuple[int, int]:
    """Count the data points whose prediction is within a fold factor.

    A data point is within the fold factor if `1/fold <= f / y <= fold` for its
    value `y` and its prediction `f`, the bounds included. Only the data points
    with a positive and finite value and prediction have a ratio, the others
    are left out, i.e. the count is of the `n_+` points with a ratio.

    Args:
        y_observed: measured values.
        y_predicted: predicted values.
        fold: the fold factor, larger than 1.

    Returns:
        The number of points within the fold factor and the number `n_+` of
        the points with a ratio, the denominator of `within_fold`.

    Raises:
        ValueError: if the data and the prediction have different lengths or
            the fold factor is not larger than 1.
    """
    if not fold > 1.0:
        raise ValueError(f"The fold factor must be larger than 1, but '{fold}' given.")
    y_obs = _as_array(y_observed)
    y_pred = _as_array(y_predicted)
    if y_obs.size != y_pred.size:
        raise ValueError(
            f"The fold criterion requires as many predictions as data points, "
            f"but got '{y_pred.size}' for '{y_obs.size}'."
        )
    mask = np.isfinite(y_obs) & np.isfinite(y_pred) & (y_obs > 0) & (y_pred > 0)
    y_obs, y_pred = y_obs[mask], y_pred[mask]
    # the bounds as products, so that twofold is exact in floating point
    inside = (y_pred >= y_obs / fold) & (y_pred <= y_obs * fold)
    return int(np.sum(inside)), int(np.sum(mask))


def within_fold(
    y_observed: ArrayLike, y_predicted: ArrayLike, fold: float = 2.0
) -> float:
    """Percentage of the data points whose prediction is within a fold factor.

    The percentage of the points of `fold_counts`, i.e. of the `n_+` points
    with a positive and finite value and prediction, whose ratio `f / y` is
    within `1/fold` and `fold`, the bounds included. The criterion of the
    predictions of pharmacokinetics is twofold, the default.

    Args:
        y_observed: measured values.
        y_predicted: predicted values.
        fold: the fold factor, larger than 1.

    Returns:
        Percentage (0 to 100) of the points with a ratio within the fold
        factor, `nan` if no point has a ratio.

    Raises:
        ValueError: if the data and the prediction have different lengths or
            the fold factor is not larger than 1.
    """
    inside, n_ratio = fold_counts(y_observed, y_predicted, fold=fold)
    if n_ratio == 0:
        return float("nan")
    return 100.0 * inside / n_ratio


@dataclass
class FitMetrics:
    """Metrics of a parameter set on an optimization problem.

    The metrics are calculated from the data of the fit mappings and the
    predictions of the model, see the module docstring for the names.

    Attributes:
        problem: initialized optimization problem, it provides the data.
        parameter_set: parameters the predictions are calculated for, they give
            IPRED.
        population_parameter_set: parameters shared by all fit mappings, they
            give PRED. Without them PRED is IPRED, which is the case of a
            deterministic fit of a single parameter set.
        evaluation: a keyword only argument and no field, the complete
            residual data of the problem at the parameter set, i.e.
            `OptimizationProblem.residuals` with `complete_data=True`, which
            the metrics use instead of simulating the set again, e.g. the
            evaluation a report keeps of its sets.
    """

    problem: OptimizationProblem
    parameter_set: ParameterSet
    population_parameter_set: ParameterSet | None = None
    evaluation: InitVar[dict[str, list[Any]] | None] = field(default=None, kw_only=True)
    # the evaluation with the values of the parameter set it is the one of
    _evaluation: tuple[bytes, dict[str, list[Any]]] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self, evaluation: dict[str, list[Any]] | None) -> None:
        """Check that the problem was initialized and keep the evaluation.

        Args:
            evaluation: the complete residual data at the parameter set.

        Raises:
            ValueError: if the problem is not initialized.
        """
        if not self.problem.is_initialized:
            raise ValueError(
                f"The metrics require an initialized OptimizationProblem, "
                f"'{self.problem.opid}' is not initialized."
            )
        if evaluation is not None:
            values = self.problem.bind_parameter_set(self.parameter_set).x(
                self.problem.pids
            )
            self._evaluation = (values.tobytes(), evaluation)

    @property
    def n_parameters(self) -> int:
        """Number of fitted parameters, the `k` of the AIC."""
        return len(self.problem.parameters)

    def _residual_data(self, pset: ParameterSet) -> dict[str, list[Any]]:
        """Get the complete residual data of every mapping for a parameter set."""
        values = self.problem.bind_parameter_set(pset).x(self.problem.pids)
        if self._evaluation is not None and self._evaluation[0] == values.tobytes():
            return self._evaluation[1]
        return self.problem.residuals(  # ty: ignore[invalid-return-type]
            xlog=self.problem.to_scale(values),
            complete_data=True,
        )

    def _predictions(self, pset: ParameterSet) -> list[np.ndarray]:
        """Get the prediction at the data points of every mapping."""
        res_data = self._residual_data(pset)
        return [np.asarray(y, dtype=float) for y in res_data["y_obsip"]]

    def datapoints_df(self) -> pd.DataFrame:
        """Get the table of the data points with their predictions.

        Returns:
            DataFrame with one row per data point and the columns `experiment`,
            `mapping`, `kind`, `x`, `DV`, `PRED`, `IPRED`, `RES`, `IRES`,
            `NRES` and `IWRES`, see the module.
        """
        res_data = self._residual_data(self.parameter_set)
        ipred_all = [np.asarray(y, dtype=float) for y in res_data["y_obsip"]]
        pred_all = (
            self._predictions(self.population_parameter_set)
            if self.population_parameter_set is not None
            else ipred_all
        )

        data: list[dict[str, Any]] = []
        for k, mapping in enumerate(self.problem.mapping_keys):
            dv = np.asarray(self.problem.y_references[k], dtype=float)
            pred = pred_all[k]
            ipred = ipred_all[k]
            # the residual of the cost, with the sign of `data - prediction`
            iwres = -np.asarray(res_data["residuals_weighted"][k], dtype=float)
            dv_mean = float(np.mean(dv))
            for ix in range(dv.size):
                ires = dv[ix] - ipred[ix]
                data.append(
                    {
                        "experiment": self.problem.experiment_keys[k],
                        "mapping": mapping,
                        "kind": self.problem.mapping_kinds[k].value,
                        "x": self.problem.x_references[k][ix],
                        "DV": dv[ix],
                        "PRED": pred[ix],
                        "IPRED": ipred[ix],
                        "RES": dv[ix] - pred[ix],
                        "IRES": ires,
                        "NRES": ires / dv_mean if dv_mean else float("nan"),
                        "IWRES": iwres[ix],
                    }
                )

        return pd.DataFrame(
            data,
            columns=pd.Index(
                [
                    "experiment",
                    "mapping",
                    "kind",
                    "x",
                    "DV",
                    "PRED",
                    "IPRED",
                    "RES",
                    "IRES",
                    "NRES",
                    "IWRES",
                ]
            ),
        )

    def mappings_df(self) -> pd.DataFrame:
        """Get the metrics of every fit mapping.

        Returns:
            DataFrame with one row per fit mapping and the columns `experiment`,
            `mapping`, `kind`, `n`, `MSE`, `RMSE`, `NRMSE`, `RMSE_w`, `R2` and
            `within_2fold`, the percentage of the points within twofold.
        """
        return self._mappings_df(self.datapoints_df())

    def _mappings_df(self, dp: pd.DataFrame) -> pd.DataFrame:
        """Compute mapping metrics from one evaluation's data points."""
        data: list[dict[str, Any]] = []
        for k, mapping in enumerate(self.problem.mapping_keys):
            # the mappings of two experiments may have the same key
            experiment = self.problem.experiment_keys[k]
            of_mapping = dp[(dp.experiment == experiment) & (dp.mapping == mapping)]
            mse_value = mse(of_mapping.IRES)
            data.append(
                {
                    "experiment": experiment,
                    "mapping": mapping,
                    "kind": self.problem.mapping_kinds[k].value,
                    "n": len(of_mapping),
                    "MSE": mse_value,
                    "RMSE": rmse_from_mse(mse_value),
                    "NRMSE": rmse(of_mapping.NRES),
                    "RMSE_w": rmse(of_mapping.IWRES),
                    "R2": r_squared(of_mapping.DV, of_mapping.IPRED),
                    "within_2fold": within_fold(of_mapping.DV, of_mapping.IPRED),
                }
            )

        return pd.DataFrame(
            data,
            columns=pd.Index(
                [
                    "experiment",
                    "mapping",
                    "kind",
                    "n",
                    "MSE",
                    "RMSE",
                    "NRMSE",
                    "RMSE_w",
                    "R2",
                    "within_2fold",
                ]
            ),
        )

    def summary(self, kind: MappingKind | None = None) -> dict[str, Any]:
        """Get the metrics over the data points of the problem.

        `MSE`, `RMSE`, `R2`, `AIC` and `BIC` are the unweighted metrics of the
        data and the predictions, i.e., they are dominated by the fit mappings
        with the largest values. The two information criteria differ in how
        they penalize a parameter, `2` against `ln(n)`, so the BIC prefers the
        smaller model of two which describe the data equally well. `NRMSE` is
        the root mean square of the residuals normalized by the mean of their
        curve, so every curve counts the same whatever its magnitude, and
        `RMSE_w` is the root mean square of the residuals of the cost, so a
        parameter set can have a larger RMSE and smaller weighted residuals
        than another one, which is what the weighting is for. `within_2fold`
        is the percentage of the data points with a positive value and
        prediction whose prediction is within twofold of the data, see
        `within_fold`, `nan` if no point has a ratio.

        `cost` is the objective the optimization minimizes. It is defined on the
        training data alone, so it is only reported for the training data and is
        `nan` for the other kinds.

        Args:
            kind: only the data points of the fit mappings of this kind, all
                data points if `None`.

        Returns:
            Dictionary with the id of the parameter set, the `kind`, the number
            of data points `n`, the number of parameters `k`, the `cost`,
            `MSE`, `RMSE`, `NRMSE`, `RMSE_w`, `R2`, `within_2fold`, `AIC` and
            `BIC`.

        Raises:
            ValueError: if the problem has no data points of the kind.
        """
        return self._summary(self.datapoints_df(), kind)

    def _summary(
        self, dp: pd.DataFrame, kind: MappingKind | None = None
    ) -> dict[str, Any]:
        """Compute summary metrics from one evaluation's data points."""
        if kind is not None:
            dp = dp[dp.kind == kind.value]
        if len(dp) == 0:
            raise ValueError(
                f"'{self.problem.opid}' has no data points of kind '{kind}'."
            )

        # the cost is the objective of the optimization, which is defined on
        # the training data
        kinds = list(self.problem.mapping_counts())
        is_training = kind is MappingKind.TRAINING or kinds == [MappingKind.TRAINING]
        cost = float("nan")
        if is_training:
            cost = 0.5 * sse(dp.IWRES) if np.isfinite(dp.IPRED).all() else float("inf")

        mse_value = mse(dp.IRES)
        return {
            "parameter_set": self.parameter_set.sid,
            "kind": kind.value if kind else "all",
            "n": len(dp),
            "k": self.n_parameters,
            "cost": cost,
            "MSE": mse_value,
            "RMSE": rmse_from_mse(mse_value),
            "NRMSE": rmse(dp.NRES),
            "RMSE_w": rmse(dp.IWRES),
            "R2": r_squared(dp.DV, dp.IPRED),
            "within_2fold": within_fold(dp.DV, dp.IPRED),
            "AIC": aic_from_mse(mse=mse_value, n=len(dp), k=self.n_parameters),
            "BIC": bic_from_mse(mse=mse_value, n=len(dp), k=self.n_parameters),
        }

    def summary_df(self) -> pd.DataFrame:
        """Get the metrics per kind of fit mapping.

        A fit is evaluated on its training data, on its validation data and on
        the outliers it dropped, so there is a row for every kind the problem
        has. There is no row over all data points: it pools the data a fit was
        fitted on with the data it dropped, which is not a number to read.
        `summary()` gives the metrics over all data points where they are
        wanted.

        Returns:
            DataFrame with one row per kind, see `summary`.
        """
        return self._summary_df(self.datapoints_df())

    def _summary_df(self, dp: pd.DataFrame) -> pd.DataFrame:
        """Compute the summaries from one evaluation's data points."""
        kinds = [
            kind for kind in self.problem.mapping_counts() if kind in EVALUATED_KINDS
        ]
        return pd.DataFrame([self._summary(dp, kind=kind) for kind in kinds])

    def cost(self) -> float:
        """Get the cost of the parameter set, i.e., the objective of the fit."""
        values = self.problem.bind_parameter_set(self.parameter_set).x(
            self.problem.pids
        )
        if self._evaluation is not None and self._evaluation[0] == values.tobytes():
            return float(
                self._summary(self.datapoints_df(), MappingKind.TRAINING)["cost"]
            )
        return self.problem.cost_least_square(self.problem.to_scale(values))

    def report(self) -> str:
        """Get the metrics as text, per kind of fit mapping and per mapping."""
        dp = self.datapoints_df()
        info = [
            "-" * 80,
            f"Metrics: {self.parameter_set.sid}",
            "-" * 80,
            self._summary_df(dp).to_string(index=False),
            "",
            self._mappings_df(dp).to_string(index=False),
            "-" * 80,
        ]
        return "\n".join(info)
