"""Main options for the parameter fitting.

In the optimization the cost is minimized for Nk curves with every curve having Nki
data points.

sum(Nk)( w{k}^2 * sum(NKi) (w{i,k}^2 * res{i,k}))
"""

from enum import Enum


class OptimizationAlgorithmType(Enum):
    """Type of optimization.

    `least square` : Least square is a local optimization method and works well in
    combination with many start values, i.e., many repeats of the optimization problem.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
    for more information.

    `differential evolution` : Differential evolution is a global optimization method
    and normally is run with a limited number of repeats. See
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html#scipy.optimize.differential_evolution
    for more information.
    """

    LEAST_SQUARE = 1
    DIFFERENTIAL_EVOLUTION = 2


class ResidualType(Enum):
    """Handling of the residuals.

    How are the residuals calculated? Are the absolute residuals used,
    or are the residuals normalized based on the data points, i.e., relative
    residuals.

    `absolute` (default) : uses the absolute data values for calculation of the
    residuals::

        r(i,k) = y(i,k) - f(xi,k)

    `normalized` : normalizing residuals of curve with 1/mean of the timecourse::

        r(i,k) = (y(i,k) - f(xi,k))/mean(y(k))

    This allows to use time courses with very different absolute values in a single
    optimization problem.

    `absolute_to_baseline` (experimental) : uses the absolute changes to
    baseline with baseline being the first data point. This requires appropriate
    pre-simulations for the model to reach a baseline.
    Data and fits have to be checked carefully.
    Residuals are calculated as::

        r(i,k) = (y(i,k) - ybase(k)) - (f(xi,k) - fbase(k))

    `normalized changes baseline` (experimental): Uses the normalized changes to
    baseline with baseline being the first data point.
    This requires appropriate pre-simulations for the model to reach a baseline.
    Data and fits have to be checked carefully.
    The residuals are calculated as::

        r(i,k) = (y(i,k) - ybase(k)) - (f(xi,k) - fbase(k))/mean(y(k))
    """

    ABSOLUTE = 1
    NORMALIZED = 2
    ABSOLUTE_TO_BASELINE = 3
    NORMALIZED_TO_BASELINE = 4


class LossFunctionType(Enum):
    """Determines the loss function.

    minimize F(x) = 0.5 * sum(rho(residuals_weighted(x)**2)

    The following loss functions are supported are allowed:

    ‘linear’ (default) : rho(z) = z. Gives a standard least-squares problem.

    ‘soft_l1’ : rho(z) = 2 * ((1 + z)**0.5 - 1). The smooth approximation of l1
    (absolute value) loss. Usually a good choice for robust least squares.

    ‘cauchy’ : rho(z) = ln(1 + z). Severely weakens outliers influence,
    but may cause difficulties in optimization process.

    ‘arctan’ : rho(z) = arctan(z). Limits a maximum loss on a single residual,
    has properties similar to ‘cauchy’.
    """

    LINEAR = 1
    SOFT_L1 = 2
    CAUCHY = 3
    ARCTAN = 4


class WeightingCurvesType(Enum):
    """Weighting w_{k} of the curves k.

    Users can provide set of weightings for the individual curves. By default no
    weightings are applied, i.e. all curves are weighted equally if no weighting
    option is provided::

      w_{k} = 1.0

    `mapping` : curves k are weighted with the provided user weights in the fit
    mappings, e.g., counts::

        w_{k} = wu_{k}

    `points` : weighting with the number of data points. Often time courses
    contain different number of data points. The residuals should contribute
    equally per data point::

        w_{k} = 1.0/count{k}


    The various options can be combined, e.g. mapping and points results in::

        w_{k} = wu_{k}/count{k}/mean{y(k)}
    """

    MAPPING = 1
    POINTS = 2


class WeightingPointsType(Enum):
    """Weighting w_{i,k} of the data points i within a single fit mapping k.

    This decides how the data points within a single fit mapping are
    weighted.

    `no weighting` (default) : all data points are weighted equally::

        w_{i,k} = 1.0

    `error_weighting`: data points are weighted as ~1/error

      # FIXME: update documentation, These must probably be normalized also.
      if yerr{i,k}:
        w_{i,k} = 1.0/yerr{i,k}
      else:
        w_{i,k} = 1.0/yerr{i,k}
    """

    NO_WEIGHTING = 1
    ERROR_WEIGHTING = 2
