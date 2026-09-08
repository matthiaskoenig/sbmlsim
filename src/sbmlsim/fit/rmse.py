"""Statistics of fits.

Metrics which summarize the quality of a fit from the mean squared error of the
residuals, the number of data points and the number of fitted parameters.
"""

import numpy as np


def mse(residuals: np.ndarray) -> float:
    """Mean Squared Error (MSE) of the residuals.

    Args:
        residuals: residuals of the fit, i.e., `f(x_i) - y_i`.

    Returns:
        Mean of the squared residuals.

    Raises:
        ValueError: if no residuals are given.
    """
    residuals = np.asarray(residuals, dtype=float)
    if residuals.size == 0:
        raise ValueError("MSE requires at least one residual.")
    return float(np.mean(np.square(residuals)))


def rmse(mse: float) -> float:
    """Root Mean Squared Error (RMSE).

    Args:
        mse: mean squared error of the fit.

    Returns:
        Square root of the mean squared error.
    """
    return float(np.sqrt(mse))


def aic(mse: float, n: int, k: int) -> float:
    """Akaike Information Criterion (AIC).

    The AIC is calculated for a least squares fit with normally distributed
    residuals, i.e., `AIC = n * ln(MSE) + 2 * k` up to an additive constant. Only
    differences of the AIC between models fitted on the same data are meaningful.

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
