"""Sensitivity analyses of a simple chain: local, Sobol, FAST and Morris.

The chain S1 -> S2 -> S3 with the rates k1 and k2 is simulated for three
initial concentrations of S1 (a dimension of the scan); the observables are
the mean concentration of every species and the maximum of S2. Every analysis
is a design of the sampler, a run and the indices on the result.
"""

import argparse
from pathlib import Path

from sbmlsim import sensitivity
from sbmlsim.simulation import Dimension, Formula, Scan, Simulation, sampling
from sbmlsim.simulator import Simulator

MODEL = Path(__file__).parent / "simple_chain.xml"

OBSERVABLES = [
    Formula("S1_mean", "mean([S1])"),
    Formula("S2_mean", "mean([S2])"),
    Formula("S3_mean", "mean([S3])"),
    Formula("S2_max", "max([S2])"),
]


def run(quick: bool, cores: int | None) -> dict[str, sensitivity.SensitivityResult]:
    """Run the four analyses and draw their figures into the working directory."""
    simulator = Simulator(n_workers=cores)
    model = simulator.load(MODEL)
    simulation = Simulation(end=1000, steps=1000)
    conditions = Dimension(
        "S1_0", values={"[S1]": [0.1, 1.0, 10.0]}, labels=["low", "reference", "high"]
    )
    parameters = sampling.parameters_of(model)
    bounds = {pid: sampling.Uniform(relative=0.15) for pid in parameters}
    n = 32 if quick else 1024

    designs = {
        "local": (sampling.local(parameters, 0.01, model=model), sensitivity.local),
        "sobol": (sampling.sobol(bounds, n, seed=1, model=model), sensitivity.sobol),
        "fast": (
            sampling.fast(bounds, max(n, 65), seed=1, model=model),
            sensitivity.fast,
        ),
        "morris": (
            sampling.morris(bounds, 10 if quick else 100, seed=1, model=model),
            sensitivity.morris,
        ),
    }
    results = {}
    for name, (design, analysis) in designs.items():
        scan = Scan(simulation, [conditions, design])
        results[name] = analysis(simulator.run(model, scan, OBSERVABLES))

    dpi = 72 if quick else 300
    reference = conditions.values["[S1]"][list(conditions.labels).index("reference")]
    condition = f"[S1] = {reference:g}"
    sensitivity.plot_heatmap(
        results["local"],
        "normalized",
        title=f"normalized, {condition}",
        S1_0="reference",
        path=Path("local.png"),
        dpi=dpi,
    )
    sensitivity.plot_indices(
        results["sobol"],
        "S2_mean",
        title=f"S2_mean, Sobol, {condition}",
        S1_0="reference",
        path=Path("sobol_S2_mean.png"),
        dpi=dpi,
    )
    sensitivity.plot_morris(
        results["morris"],
        "S2_mean",
        title=f"S2_mean, Morris, {condition}",
        S1_0="reference",
        path=Path("morris_S2_mean.png"),
        dpi=dpi,
    )
    shown = {"local": "normalized", "sobol": "ST", "fast": "ST", "morris": "mu_star"}
    for name, result in results.items():
        print(f"{name}: {shown[name]} at {condition}")
        print(result.index(shown[name]).sel(S1_0="reference").to_pandas().round(5))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="small designs and figures, for the tests"
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help="the workers of the simulator, all from 256 points by default",
    )
    options = parser.parse_args()
    run(options.quick, options.cores)
