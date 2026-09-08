"""Sampling of parameter values."""

import logging
from enum import Enum

import numpy as np
import pandas as pd
from scipy.stats import qmc

from sbmlsim.fit.objects import FitParameter

logger = logging.getLogger(__name__)


class SamplingType(Enum):
    """Type of sampling used.

    The LHS options are latin hypercube sampling types.
    """

    LOGUNIFORM = 1
    UNIFORM = 2
    LOGUNIFORM_LHS = 3
    UNIFORM_LHS = 4

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
    max_bound: float = 1e10,
) -> pd.DataFrame:
    """Create samples of start values from the bounds of the parameters.

    Infinite bounds are replaced by the hard bounds `min_bound` and `max_bound`.
    Logarithmic sampling requires positive bounds, non-positive lower bounds are
    replaced by `min_bound`.

    Args:
        parameters: parameters to sample, the bounds define the sampled interval.
        size: number of samples.
        sampling: type of sampling.
        seed: seed of the random number generator, for reproducible samples.
        min_bound: hard lower bound, replaces an infinite or non-positive bound.
        max_bound: hard upper bound, replaces an infinite bound.

    Returns:
        DataFrame with one row per sample and one column per parameter.

    Raises:
        ValueError: if the sampling type is unsupported or the bounds are invalid.
    """
    if size < 1:
        raise ValueError(f"'size' must be a positive integer, but '{size}' given.")
    if not parameters:
        raise ValueError("'parameters' must not be empty.")

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
        lb, ub = _sampling_bounds(
            parameter=p,
            sampling=sampling,
            min_bound=min_bound,
            max_bound=max_bound,
        )

        # stretch sampling dimension from [0, 1) to [lb, ub)
        if sampling.is_log:
            lb_log = np.log10(lb)
            ub_log = np.log10(ub)
            # samples are in log space, parameter values in real space
            x[:, k] = np.power(10, lb_log + x[:, k] * (ub_log - lb_log))
        else:
            x[:, k] = lb + x[:, k] * (ub - lb)

    return pd.DataFrame(x, columns=pd.Index([p.pid for p in parameters]))


def _sampling_bounds(
    parameter: FitParameter,
    sampling: SamplingType,
    min_bound: float,
    max_bound: float,
) -> tuple[float, float]:
    """Resolve the bounds of a parameter to the finite interval which is sampled."""
    pid = parameter.pid
    lb = float(parameter.lower_bound)
    ub = float(parameter.upper_bound)

    if np.isinf(lb):
        lb = -max_bound if lb < 0 else max_bound
        logger.warning("'%s': infinite lower bound set to '%s'", pid, lb)
    if np.isinf(ub):
        ub = max_bound if ub > 0 else -max_bound
        logger.warning("'%s': infinite upper bound set to '%s'", pid, ub)

    if sampling.is_log:
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
