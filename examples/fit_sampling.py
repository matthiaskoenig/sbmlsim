"""Example showing the sampling of parameter start values for a fit.

The four sampling types of `sbmlsim.fit.sampling` are compared on three parameters
with logarithmic bounds. The figure is written to `fit_sampling.png` in the working
directory.
"""

import pandas as pd
from matplotlib import pyplot as plt

from sbmlsim.console import console
from sbmlsim.fit import FitParameter
from sbmlsim.fit.sampling import SamplingType, create_samples


def plot_samples(samples: dict[str, pd.DataFrame], path: str) -> None:
    """Plot the first two parameters of the samples for every sampling type."""
    pids = next(iter(samples.values())).columns

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), layout="constrained")
    for ax, (key, df) in zip(axes.flatten(), samples.items(), strict=False):
        ax.set_title(key)
        ax.set_xlabel(pids[0])
        ax.set_ylabel(pids[1])
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
        ax.set_xscale("log")
        ax.set_yscale("log")

    fig.savefig(path, bbox_inches="tight")


def run_sampling_example() -> None:
    """Create and plot samples for all sampling types."""
    parameters: list[FitParameter] = [
        FitParameter(pid="p1", lower_bound=10, upper_bound=1e4, unit="dimensionless"),
        FitParameter(pid="p2", lower_bound=1, upper_bound=1e3, unit="dimensionless"),
        FitParameter(pid="p3", lower_bound=1, upper_bound=1e3, unit="dimensionless"),
    ]
    samples: dict[str, pd.DataFrame] = {}
    for sampling in SamplingType:
        console.rule(title=sampling.name, align="left", style="white")
        df = create_samples(
            parameters=parameters, size=10, sampling=sampling, seed=1234
        )
        console.print(df)
        samples[sampling.name] = df

    plot_samples(samples, path="fit_sampling.png")


if __name__ == "__main__":
    run_sampling_example()
