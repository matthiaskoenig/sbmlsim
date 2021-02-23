"""Sampling of parameter values."""
import logging
from enum import Enum
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyDOE import lhs

from sbmlsim.fit.objects import FitParameter
from sbmlsim.utils import timeit


logger = logging.getLogger(__name__)


class SamplingType(Enum):
    """Type of sampling used.

    The LHS options are latin hypercube sampling types.
    """

    LOGUNIFORM = 1
    UNIFORM = 2
    LOGUNIFORM_LHS = 3
    UNIFORM_LHS = 4


def create_samples(
    parameters: Iterable[FitParameter],
    size,
    sampling=SamplingType.LOGUNIFORM,
    seed=None,
    min_bound=1e-10,
    max_bound=1e10,
) -> pd.DataFrame:
    """Create samples from given parameter information.

    :param parameters:
    :param size:
    :param sampling:
    :param seed:
    :param min_bound: hard lower bound
    :param min_bound: hard upper bound
    :return:
    """
    # TODO: add option to get current model parameter values as start values for local gradient descent

    # seed for reproducibility
    if seed:
        np.random.seed(seed)

    # get samples between [0, 1)
    if sampling in {SamplingType.UNIFORM, SamplingType.LOGUNIFORM}:
        # samples = np.random.uniform(0, 1, size=size)
        x = np.random.rand(size, len(parameters))
        # print(type(x), x.shape)

    elif sampling in {SamplingType.UNIFORM_LHS, SamplingType.LOGUNIFORM_LHS}:
        # Latin-Hypercube sampling
        # https://pythonhosted.org/pyDOE/randomized.html#latin-hypercube
        # “maximin” or “m”: maximize the minimum distance between points, but place the point in a randomized location within its interval
        x = lhs(n=len(parameters), samples=size)  # criterion="maximin"
    else:
        raise ValueError(f"Unsupported SamplingType: '{sampling}'")

    for k, p in enumerate(parameters):
        # handle bounds
        lb = p.lower_bound
        if np.isinf(lb):
            logger.warning(f"infinite lower bound set to '{-max_bound}'")
            lb = -max_bound
            if sampling in [SamplingType.LOGUNIFORM, SamplingType.LOGUNIFORM_LHS]:
                if lb <= 0.0:
                    logger.warning(f"negative lower bound set to '{min_bound}'")
                    lb = min_bound

        ub = p.upper_bound
        if np.isinf(ub):
            logger.warning(f"infinite upper bound set to '{max_bound}'")
            ub = min_bound

        # stretch sampling dimension from [0, 1) to [lb, ub)
        if sampling in {SamplingType.UNIFORM, SamplingType.UNIFORM_LHS}:
            x[:, k] = lb + x[:, k] * (ub - lb)
        elif sampling in {SamplingType.LOGUNIFORM, SamplingType.LOGUNIFORM_LHS}:
            lb_log = np.log10(lb)
            ub_log = np.log10(ub)
            # samples are in log space
            values_log = lb_log + x[:, k] * (ub_log - lb_log)
            # parameter values in real space
            x[:, k] = np.power(10, values_log)

    # print(type(x), x.shape)
    return pd.DataFrame(x, columns=[p.pid for p in parameters])


def plot_samples(samples):
    """Plot samples."""
    df = list(samples.values())[0]
    pids = df.columns

    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, ncols=2, figsize=(10, 10))
    axes = (ax1, ax2, ax3, ax4)
    for k, key in enumerate(samples.keys()):
        ax = axes[k]
        ax.set_xlabel(pids[0])
        ax.set_ylabel(pids[1])

        # start point
        df = samples[key]
        ax.set_title(key)
        ax.plot(
            df[pids[0]],
            df[pids[1]],
            markersize=10,
            alpha=0.9,
            label=key,
            linestyle="None",
            marker="s",
            color="black",
        )

        # ax.legend()
        ax.set_xscale("log")
        ax.set_yscale("log")

    plt.show()


def example_sampling():
    """Examples howing the use of the sampling.

    :return:
    """
    parameters: List[FitParameter] = [
        FitParameter(pid="p1", lower_bound=10, upper_bound=1e4),
        FitParameter(pid="p2", lower_bound=1, upper_bound=1e3),
        FitParameter(pid="p3", lower_bound=1, upper_bound=1e3),
    ]
    samples: Dict[str, pd.DataFrame] = {}
    for sampling in [
        SamplingType.UNIFORM,
        SamplingType.UNIFORM_LHS,
        SamplingType.LOGUNIFORM,
        SamplingType.LOGUNIFORM_LHS,
    ]:
        print(f"* {sampling.name} *")
        df = create_samples(
            parameters=parameters, size=10, sampling=sampling, seed=1234
        )
        samples[sampling.name] = df

    print(samples)
    plot_samples(samples)


if __name__ == "__main__":
    example_sampling()
