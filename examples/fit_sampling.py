"""Example showing the sampling of parameter start values for a fit.

The sampling types of `sbmlsim.fit.sampling` are compared on three parameters with
logarithmic bounds; `START` does not sample and shows the one point every run starts
from. The figure is written to `fit_sampling.svg` in the working directory.
"""

import pandas as pd

from sbmlsim.console import console
from sbmlsim.fit import FitParameter
from sbmlsim.fit.sampling import SamplingType, create_samples
from sbmlsim.plot import Chart, ChartAxis

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
    chart = Chart(width=360 * ncols, height=360 * nrows, rows=nrows, columns=ncols)
    for k, (key, df) in enumerate(samples.items()):
        plot = chart.plot(
            k // ncols,
            k % ncols,
            title=key,
            x=ChartAxis(
                label=p_x.pid,
                scale="log",
                min=p_x.lower_bound / PAD,
                max=p_x.upper_bound * PAD,
            ),
            y=ChartAxis(
                label=p_y.pid,
                scale="log",
                min=p_y.lower_bound / PAD,
                max=p_y.upper_bound * PAD,
            ),
            legend=False,
        )
        plot.curve(
            df[p_x.pid].to_numpy(),
            df[p_y.pid].to_numpy(),
            line="none",
            marker="square",
            size=8,
            opacity=0.9,
            color="#000000",
        )

    chart.save(path)


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

    plot_samples(samples, parameters=parameters, path="fit_sampling.svg")


if __name__ == "__main__":
    run_sampling_example()
