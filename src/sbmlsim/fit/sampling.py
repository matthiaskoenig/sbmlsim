"""Sampling of parameter values."""

import logging
from enum import Enum

import numpy as np
import pandas as pd
from scipy.stats import qmc

from sbmlsim.fit.objects import FitParameter
from sbmlsim.fit.options import ParameterScaleType

logger = logging.getLogger(__name__)


class SamplingType(Enum):
    """Type of sampling used.

    The LHS options are latin hypercube sampling types. `START` does not
    sample: every run starts from the start values of the parameters, which is
    what a fit needs when the start values are a solution to improve, e.g. the
    nominal values of a model or the trained network of a hybrid problem,
    instead of a point of the bounds. The runs are identical then, so one run is
    enough for a local optimizer.
    """

    LOGUNIFORM = 1
    UNIFORM = 2
    LOGUNIFORM_LHS = 3
    UNIFORM_LHS = 4
    START = 5

    @property
    def is_log(self) -> bool:
        """Check if the samples are drawn in logarithmic space."""
        return self in {SamplingType.LOGUNIFORM, SamplingType.LOGUNIFORM_LHS}

    @property
    def is_lhs(self) -> bool:
        """Check if latin hypercube sampling is used."""
        return self in {SamplingType.UNIFORM_LHS, SamplingType.LOGUNIFORM_LHS}


def create_samples(
    parameters: list[FitParameter],
    size: int,
    sampling: SamplingType = SamplingType.LOGUNIFORM,
    seed: int | None = None,
    min_bound: float = 1e-10,
) -> pd.DataFrame:
    """Create samples of start values from the bounds of the parameters.

    With `SamplingType.START` every sample is the vector of the start values
    of the parameters and the bounds and the seed are not used.

    A parameter with an infinite bound has no interval to sample: it starts
    from its start value in every sample, so the samples differ in the
    parameters with finite bounds. The samples of these do not depend on the
    parameters which are not sampled.

    A parameter on the linear scale (`FitParameter.scale`), e.g. an element
    of a neural network with negative bounds, is sampled uniformly in its
    bounds whatever the sampling type; a parameter without a scale of its own
    follows the sampling type. Logarithmic sampling requires positive bounds,
    non-positive lower bounds are replaced by `min_bound`.

    Args:
        parameters: parameters to sample, the bounds define the sampled interval.
        size: number of samples.
        sampling: type of sampling.
        seed: seed of the random number generator, for reproducible samples.
        min_bound: hard lower bound, replaces a non-positive bound of a
            logarithmic sampling.

    Returns:
        DataFrame with one row per sample and one column per parameter.

    Raises:
        ValueError: if the sampling type is unsupported, the bounds are
            invalid, or a parameter with an infinite bound has no start value
            (with `SamplingType.START` a parameter without a start value).
    """
    if size < 1:
        raise ValueError(f"'size' must be a positive integer, but '{size}' given.")
    if not parameters:
        raise ValueError("'parameters' must not be empty.")

    if sampling is SamplingType.START:
        return _start_samples(parameters, size)

    rng = np.random.default_rng(seed)

    # get samples in the unit hypercube [0, 1)
    x: np.ndarray
    if sampling.is_lhs:
        # Latin-Hypercube sampling
        sampler = qmc.LatinHypercube(d=len(parameters), rng=rng)
        x = sampler.random(n=size)
    elif sampling in {SamplingType.UNIFORM, SamplingType.LOGUNIFORM}:
        x = rng.random(size=(size, len(parameters)))
    else:
        raise ValueError(f"Unsupported SamplingType: '{sampling}'")

    for k, p in enumerate(parameters):
        if np.isinf(p.lower_bound) or np.isinf(p.upper_bound):
            if p.start_value is None:
                raise ValueError(
                    f"'{p.pid}': a parameter with the infinite bounds "
                    f"[{p.lower_bound} - {p.upper_bound}] is not sampled and "
                    f"requires a 'start_value'."
                )
            x[:, k] = p.start_value
            continue
        is_log = sampling.is_log and p.scale is not ParameterScaleType.LINEAR
        lb, ub = _sampling_bounds(parameter=p, is_log=is_log, min_bound=min_bound)

        # stretch sampling dimension from [0, 1) to [lb, ub)
        if is_log:
            lb_log = np.log10(lb)
            ub_log = np.log10(ub)
            # samples are in log space, parameter values in real space
            x[:, k] = np.power(10, lb_log + x[:, k] * (ub_log - lb_log))
        else:
            x[:, k] = lb + x[:, k] * (ub - lb)

    return pd.DataFrame(x, columns=pd.Index([p.pid for p in parameters]))


def _start_samples(parameters: list[FitParameter], size: int) -> pd.DataFrame:
    """Repeat the start values of the parameters `size` times.

    Raises:
        ValueError: if a parameter has no start value.
    """
    start: list[float] = []
    missing: list[str] = []
    for p in parameters:
        if p.start_value is None:
            missing.append(p.pid)
        else:
            start.append(float(p.start_value))
    if missing:
        raise ValueError(
            f"The parameters {missing} have no 'start_value', which the sampling "
            f"'START' starts every run from."
        )
    return pd.DataFrame(
        np.tile(start, (size, 1)), columns=pd.Index([p.pid for p in parameters])
    )


def _sampling_bounds(
    parameter: FitParameter,
    is_log: bool,
    min_bound: float,
) -> tuple[float, float]:
    """Resolve the finite bounds of a parameter to the interval which is sampled.

    Args:
        parameter: the parameter with finite bounds.
        is_log: whether the parameter is sampled in logarithmic space.
        min_bound: replaces a non-positive lower bound of a logarithmic
            sampling.

    Returns:
        The lower and the upper bound of the sampled interval.

    Raises:
        ValueError: if the bounds are not an interval, or if the upper bound
            of a logarithmic sampling is not positive.
    """
    pid = parameter.pid
    lb = float(parameter.lower_bound)
    ub = float(parameter.upper_bound)

    if is_log:
        # logarithmic sampling requires positive bounds
        if lb <= 0.0:
            logger.warning("'%s': non-positive lower bound set to '%s'", pid, min_bound)
            lb = min_bound
        if ub <= 0.0:
            raise ValueError(
                f"'{pid}': logarithmic sampling requires a positive upper bound, "
                f"but '{ub}' given."
            )

    if lb > ub:
        raise ValueError(f"'{pid}': lower bound '{lb}' is larger than upper '{ub}'.")

    return lb, ub
