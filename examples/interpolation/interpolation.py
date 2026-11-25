"""Interpolation of data points as an SBML model.

`sbmlsim.interpolation` turns a table of data points into an SBML model
which evaluates the interpolation, so that the data can be used inside a
simulation. The example interpolates the same two data series with all three
methods and simulates the resulting models with sbmlsim.

Run it from the root of the repository:

```bash
python -m examples.interpolation.interpolation
```

The figure is written into the current working directory.
"""

import logging
import tempfile
from pathlib import Path

import pandas as pd

from sbmlsim import interpolation as ip
from sbmlsim.plot import Chart, ChartAxis
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator

logger = logging.getLogger(__name__)


def interpolation_example() -> Chart:
    """Interpolate two data series with all interpolation methods.

    Returns:
        The chart with the data points and the simulated interpolations. The
        caller writes it to a file, nothing is shown, so that the example does
        not open a window when it runs unattended.
    """
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    y = [0.0, 2.0, 1.0, 1.5, 2.5, 3.5]
    z = [10.0, 5.0, 2.5, 1.25, 0.6, 0.3]
    data1 = pd.DataFrame({"time": x, "y": y, "z": z})

    chart = Chart(width=640, height=400, title="Interpolation Example")
    ax1 = chart.plot(x=ChartAxis(label="time [AU]"), y=ChartAxis(label="data [AU]"))
    ax1.curve(x, y, line="none", marker="circle", color="#000000", label="y")
    ax1.curve(x, z, line="none", marker="square", color="#000000", label="z")

    simulator = Simulator(n_workers=1)
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    for k, method in enumerate(
        [
            ip.INTERPOLATION_CONSTANT,
            ip.INTERPOLATION_LINEAR,
            ip.INTERPOLATION_CUBIC_SPLINE,
        ]
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_f = Path(tmpdir, "interpolation.xml")

            interpolation = ip.Interpolation(data=data1, method=method)
            interpolation.write_sbml_to_file(tmp_f)

            loaded = simulator.load(tmp_f)
            loaded.set_selections(["time", "y", "z"])
            s = simulator.simulate(loaded, Simulation(end=5, steps=50))
            ax1.curve(s["time"], s["y"], label=f"y {method}", color=colors[k])
            ax1.curve(
                s["time"], s["z"], label=f"z {method}", color=colors[k], line="dashed"
            )

    return chart


if __name__ == "__main__":
    from sbmlsim import log

    log.enable_rich_logging()

    fig_path = interpolation_example().save(Path.cwd() / "interpolation.svg")
    logger.info("Figure written to '%s'", fig_path)
