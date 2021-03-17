"""Main options for the parameter fitting.

In the optimization the cost is minimized for Nk curves with every curve having Nki
data points.

sum(Nk)( w{k}^2 * sum(NKi) (w{i,k}^2 * res{i,k}))


"""

from enum import Enum


class OptimizationAlgorithmType(Enum):
    """Type of optimization.

    **least square**
      Least square is a local optimization method and works well in combination
      with many start values, i.e., many repeats of the optimization problem. See
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
      for more information.

    **differential evolution**
      Differential evolution is a global optimization method and normally is run
      with a limited number of repeats. See
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

    **absolute (default)**
      uses the absolute data values for calculation of the residuals

      r(i,k) = y(i,k) - f(xi,k)

    # FIXME: deprecate, does not work due to large relative errors for small data points
    **relative residuals**
      normalize every residual by the corresponding value.
      Normalized residuals make different fit mappings comparable.
      zero/inf valus are filtered out

      if y(i,k) != 0:
        r(i,k) = (y(i,k) - f(xi,k))/y(i,k)
      else:
        r(i,k) = 0

    **normalized residuals**
      normalizing with 1/mean of the timecourse:
        r(i,k) = (y(i,k) - f(xi,k))/mean(y(k))
      This makes timecourses comparable.

    **absolute changes baseline**
      uses the absolute changes to baseline with baseline being the first data point.
      This requires appropriate pre-simulations for the model to reach a baseline.
      Data and fits have to be checked carefully.

      r(i,k) = (y(i,k) - ybase(k)) - (f(xi,k) - fbase(k))

     **normalized changes baseline**
     # FIXME:update

      uses the relative changes to baseline with baseline being the first data point.
      This requires appropriate pre-simulations for the model to reach a baseline.
      Data and fits have to be checked carefully.
      zero/inf valus are filtered out

      if y(i,k) != 0:
        r(i,k) = (y(i,k) - ybase(k)) - (f(xi,k) - fbase(k))/y(i,k)
      else:
        r(i,k) = 0

    """

    ABSOLUTE = 1
    # RELATIVE = 2
    NORMALIZED = 3
    ABSOLUTE_CHANGES_BASELINE = 4
    RELATIVE_CHANGES_BASELINE = 5


class WeightingCurvesType(Enum):
    """Weighting w_{k} of the curves k.

    Users can provide set of weightings for the individual curves.

    **no weighting (default)**
      no weighting option is provided, all curves weighted equally:

      w_{k} = 1.0

    **mapping**
      curves k are weighted with the provided user weights in the fit mappings,
      e.g., counts

      w_{k} = wu_{k}

    **points**
      weighting with the number of data points:

      w_{k} = 1.0/count{k}

    The various options can be combined, i.e.,
      mapping, points, mean
    results in

      w_{k} = wu_{k}/count{k}/mean{y(k)}

    """

    MAPPING = 1
    POINTS = 2
    # MEAN = 3


class WeightingPointsType(Enum):
    """Weighting w_{i,k} of the data points i within a single fit mapping k.

    This decides how the data points within a single fit mapping are
    weighted.

    **no weighting**
      all data points are weighted equally

      w_{i,k} = 1.0

    **error weighting**
      data points are weighted as ~1/error

      # FIXME: update documentation
      if yerr{i,k}:
        w_{i,k} = 1.0/yerr{i,k}
      else:
        w_{i,k} = 1.0/yerr{i,k}

    """

    NO_WEIGHTING = 1
    ERROR_WEIGHTING = 2
