"""
Main options for the parameter fitting.
"""

from enum import Enum

# FIXME: check the combination of the parameters and simplify strategies


class OptimizationAlgorithmType(Enum):
    """Type of optimization.

    Least square is a local optimization method and works well in combination
    with many start values, i.e., many repeats of the optimization problem.

    Differential evolution is a global optimization method and normally is run
    with a limited number of repeats.
    """

    LEAST_SQUARE = 1
    DIFFERENTIAL_EVOLUTION = 2


class FittingStrategyType(Enum):
    """Type of fitting strategy (absolute changes or relative changes to baseline).

    Decides how to fit the data. If the various datasets have large offsets
    a fitting of the relative changes to baseline can work.
    As baseline the simulations should contain a pre-simulation.

    **absolute values**
      uses the absolute data values for fitting

    **absolute changes baseline**
      uses the absolute changes to baseline with baseline being the first data point.
      This requires appropriate pre-simulations for the model to reach a baseline.
      Data and fits have to be checked carefully.

    **relative changes baseline**
      uses the relative changes to baseline with baseline being the first data point.
      This requires appropriate pre-simulations for the model to reach a baseline.
      Data and fits have to be checked carefully.

    The default is using the absolute values in the fitting strategy,
    """

    ABSOLUTE_VALUES = 1
    ABSOLUTE_CHANGES_BASELINE = 2
    RELATIVE_CHANGES_BASELINE = 3


class ResidualType(Enum):
    """Handling of the residuals.

    How are the residuals calculated? Are the absolute residuals used,
    or are the residuals normalized based on the data points, i.e., relative
    residuals.

    **absolute residuals (default)**
      uses absolute residuals
    **relative residuals**
      (local effects) induces local weighting by normalizing every residual by
      relative value. Relative residuals make different fit mappings comparable.
    **absolute normed residuals**
      uses residuals normed per reference data
    """
    ABSOLUTE_RESIDUALS = 1
    RELATIVE_RESIDUALS = 2
    ABSOLUTE_NORMED_RESIDUALS = 3


class WeightingPointsType(Enum):
    """Weighting of the data points within a single fit mapping.

    This decides how the data points within a single fit mapping are
    weighted.

    **no weighting**
      all data points are weighted equally

    **absolute one over weighting**
      data points are weighted as ~1/error

    **relative one over weighting**
      data points are weighed relative to baseline
    """

    NO_WEIGHTING = 1  # data points are weighted equally
    ABSOLUTE_ONE_OVER_WEIGHTING = 2  # data points are weighted as 1/(error-min(error))
    RELATIVE_ONE_OVER_WEIGHTING = 3  # FIXME: check that this is working and documented

