"""Example showing the sampling of parameter start values for a fit.

The sampling types of `sbmlsim.fit.sampling` are compared on three parameters with
logarithmic bounds; `START` does not sample and shows the one point every run starts
from. The figure is written to `fit_sampling.png` in the working directory.
"""

import pandas as pd
from matplotlib import pyplot as plt

from sbmlsim.console import console
from sbmlsim.fit import FitParameter
from sbmlsim.fit.sampling import SamplingType, create_samples

PAD = 1.4  # factor between a bound and the edge of a panel, so no marker is cut


def plot_samples(
    samples: dict[str, pd.DataFrame], parameters: list[FitParameter], path: str
) -> None:
    """Plot the first two parameters of the samples for every sampling type.

    Every panel spans the bounds of the two parameters, so the panels compare.
    """
    p_x, p_y = parameters[0], parameters[1]

    ncols = 3
    nrows = -(-len(samples) // ncols)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), layout="constrained"
    )
    for ax in axes.flatten()[len(samples) :]:
        ax.set_visible(False)
    for ax, (key, df) in zip(axes.flatten(), samples.items(), strict=False):
        ax.set_title(key)
        ax.set_xlabel(p_x.pid)
        ax.set_ylabel(p_y.pid)
        ax.plot(
            df[p_x.pid],
            df[p_y.pid],
            markersize=10,
            alpha=0.9,
            label=key,
            linestyle="None",
            marker="s",
            color="black",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(p_x.lower_bound / PAD, p_x.upper_bound * PAD)
        ax.set_ylim(p_y.lower_bound / PAD, p_y.upper_bound * PAD)

    fig.savefig(path, bbox_inches="tight")


def run_sampling_example() -> None:
    """Create and plot samples for all sampling types."""
    parameters: list[FitParameter] = [
        FitParameter(
            pid="p1",
            start_value=100.0,
            lower_bound=10,
            upper_bound=1e4,
            unit="dimensionless",
        ),
        FitParameter(
            pid="p2",
            start_value=10.0,
            lower_bound=1,
            upper_bound=1e3,
            unit="dimensionless",
        ),
        FitParameter(
            pid="p3",
            start_value=10.0,
            lower_bound=1,
            upper_bound=1e3,
            unit="dimensionless",
        ),
    ]
    samples: dict[str, pd.DataFrame] = {}
    for sampling in SamplingType:
        console.rule(title=sampling.name, align="left", style="white")
        df = create_samples(
            parameters=parameters, size=10, sampling=sampling, seed=1234
        )
        console.print(df)
        samples[sampling.name] = df

    plot_samples(samples, parameters=parameters, path="fit_sampling.png")


if __name__ == "__main__":
    run_sampling_example()
