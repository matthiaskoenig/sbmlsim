"""Sensitivity analysis on the result of a scan.

An analysis is a design of the sampler, a run and the indices on the result.
The design of [`sbmlsim.simulation.sampling`](../api/simulation.sampling.md)
(`local`, `sobol`, `fast` or `morris`) is a dimension of a `Scan`,
`Simulator.run` simulates it, and `local`, `sobol`, `fast` and `morris` read
the record of the design from the result and compute the indices of every
observable for every label of the other dimensions and every time point of a
timecourse on a grid.

- [`sensitivity.indices`](../api/sensitivity.indices.md): the four analyses.
- [`sensitivity.result`](../api/sensitivity.result.md): the `SensitivityResult`
  with the indices, their units, `to_dataframe`, `classify` and netCDF.
- [`sensitivity.plots`](../api/sensitivity.plots.md): the heatmap, the bars of
  the indices and the plane of Morris.
- [`sensitivity.classification`](../api/sensitivity.classification.md): the
  classification of sensitivities.
- [`sensitivity.uncertainty`](../api/sensitivity.uncertainty.md): the bands and
  distributions of a scan over draws.
"""

from .indices import fast, local, morris, sobol
from .plots import plot_heatmap, plot_indices, plot_morris
from .result import SensitivityResult

__all__ = [
    "SensitivityResult",
    "fast",
    "local",
    "morris",
    "plot_heatmap",
    "plot_indices",
    "plot_morris",
    "sobol",
]
