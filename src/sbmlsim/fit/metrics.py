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
    IWRES   IRES weighted with the weights of the optimization problem

Note the sign: the residuals of `OptimizationProblem.residuals` are
`prediction - data`, the residuals here are `data - prediction`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

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
    meaningful.

    Args:
        mse: mean squared error of the fit.
        n: number of data points.
        k: number of fitted parameters.

    Returns:
        Akaike information criterion.

    Raises:
        ValueError: if the mean squared error or the number of data points is
            not positive.
    """
    if mse <= 0.0:
        raise ValueError(f"AIC requires a positive MSE, but '{mse}' given.")
    if n <= 0:
        raise ValueError(f"AIC requires a positive number of points, but '{n}' given.")
    return float(n * np.log(mse) + 2 * k)


def r_squared(y_observed: ArrayLike, y_predicted: ArrayLike) -> float:
    """Coefficient of determination (R²) of a prediction.

    `R² = 1 - SSE / SST` with `SST` the total sum of squares of the data. The
    predictions of a non-linear model are not a linear regression of the data,
    so R² is not the square of a correlation and can be negative: a negative R²
    means the prediction is worse than the mean of the data.

    Args:
        y_observed: measured values.
        y_predicted: predicted values.

    Returns:
        Coefficient of determination, `nan` if the data has no variance.

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
    sst = float(np.sum(np.square(y_obs - np.mean(y_obs))))
    if sst == 0.0:
        logger.warning("R² is undefined for data without variance.")
        return float("nan")
    return 1.0 - sse(y_obs - y_pred) / sst


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
    """

    problem: OptimizationProblem
    parameter_set: ParameterSet
    population_parameter_set: ParameterSet | None = None

    def __post_init__(self) -> None:
        """Check that the problem was initialized.

        Raises:
            ValueError: if the problem is not initialized.
        """
        if not self.problem.is_initialized:
            raise ValueError(
                f"The metrics require an initialized OptimizationProblem, "
                f"'{self.problem.opid}' is not initialized."
            )

    @property
    def n_parameters(self) -> int:
        """Number of fitted parameters, the `k` of the AIC."""
        return len(self.problem.parameters)

    def _predictions(self, pset: ParameterSet) -> list[np.ndarray]:
        """Get the prediction at the data points of every mapping."""
        res_data: dict[str, list[Any]] = self.problem.residuals(  # ty: ignore[invalid-assignment]
            xlog=np.log10(pset.x(self.problem.pids)), complete_data=True
        )
        return [np.asarray(y, dtype=float) for y in res_data["y_obsip"]]

    def datapoints_df(self) -> pd.DataFrame:
        """Get the table of the data points with their predictions.

        Returns:
            DataFrame with one row per data point and the columns `experiment`,
            `mapping`, `x`, `DV`, `PRED`, `IPRED`, `RES`, `IRES` and `IWRES`.
        """
        ipred_all = self._predictions(self.parameter_set)
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
            # the weights of the problem, they define the cost
            weights = np.sqrt(np.asarray(self.problem.weights[k], dtype=float))
            for ix in range(dv.size):
                data.append(
                    {
                        "experiment": self.problem.experiment_keys[k],
                        "mapping": mapping,
                        "x": self.problem.x_references[k][ix],
                        "DV": dv[ix],
                        "PRED": pred[ix],
                        "IPRED": ipred[ix],
                        "RES": dv[ix] - pred[ix],
                        "IRES": dv[ix] - ipred[ix],
                        "IWRES": (dv[ix] - ipred[ix]) * weights[ix],
                    }
                )

        return pd.DataFrame(
            data,
            columns=pd.Index(
                [
                    "experiment",
                    "mapping",
                    "x",
                    "DV",
                    "PRED",
                    "IPRED",
                    "RES",
                    "IRES",
                    "IWRES",
                ]
            ),
        )

    def mappings_df(self) -> pd.DataFrame:
        """Get the metrics of every fit mapping.

        Returns:
            DataFrame with one row per fit mapping and the columns `experiment`,
            `mapping`, `n`, `MSE`, `RMSE` and `R2`.
        """
        ipred_all = self._predictions(self.parameter_set)

        data: list[dict[str, Any]] = []
        for k, mapping in enumerate(self.problem.mapping_keys):
            dv = np.asarray(self.problem.y_references[k], dtype=float)
            ipred = ipred_all[k]
            ires = dv - ipred
            mse_value = mse(ires)
            data.append(
                {
                    "experiment": self.problem.experiment_keys[k],
                    "mapping": mapping,
                    "n": dv.size,
                    "MSE": mse_value,
                    "RMSE": rmse_from_mse(mse_value),
                    "R2": r_squared(dv, ipred),
                }
            )

        return pd.DataFrame(
            data,
            columns=pd.Index(["experiment", "mapping", "n", "MSE", "RMSE", "R2"]),
        )

    def summary(self) -> dict[str, Any]:
        """Get the metrics over all data points of the problem.

        `MSE`, `RMSE`, `R2` and `AIC` are the unweighted metrics of the data and
        the predictions, i.e., they are dominated by the fit mappings with the
        largest values. `cost` is the objective the optimization minimizes and
        `RMSE_w` the root mean square of the weighted residuals, both of which
        use the weighting of the settings; a parameter set can therefore have a
        lower cost and a larger RMSE than another one.

        Returns:
            Dictionary with the id of the parameter set, the number of data
            points `n`, the number of parameters `k`, the `cost`, `MSE`,
            `RMSE`, `RMSE_w`, `R2` and `AIC`.
        """
        dp = self.datapoints_df()
        mse_value = mse(dp.IRES)
        return {
            "parameter_set": self.parameter_set.sid,
            "n": len(dp),
            "k": self.n_parameters,
            "cost": self.cost(),
            "MSE": mse_value,
            "RMSE": rmse_from_mse(mse_value),
            "RMSE_w": rmse(dp.IWRES),
            "R2": r_squared(dp.DV, dp.IPRED),
            "AIC": aic_from_mse(mse=mse_value, n=len(dp), k=self.n_parameters),
        }

    def summary_df(self) -> pd.DataFrame:
        """Get the metrics over all data points as a single row DataFrame."""
        return pd.DataFrame([self.summary()])

    def cost(self) -> float:
        """Get the cost of the parameter set, i.e., the objective of the fit."""
        return self.problem.cost_least_square(
            np.log10(self.parameter_set.x(self.problem.pids))
        )

    def report(self) -> str:
        """Get the metrics as text."""
        summary = self.summary()
        info = [
            "-" * 80,
            f"Metrics: {self.parameter_set.sid}",
            "-" * 80,
            "\n".join(f"\t{key}: {value}" for key, value in summary.items()),
            "",
            self.mappings_df().to_string(index=False),
            "-" * 80,
        ]
        return "\n".join(info)
