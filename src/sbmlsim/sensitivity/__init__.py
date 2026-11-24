"""Sensitivity analysis on the result of a scan.

An analysis is a design of the sampler, a run and the indices on the result.
The design of [`sbmlsim.simulation.sampling`](../api/simulation.sampling.md)
(`local`, `sobol`, `fast`, `morris`, or `random` or `lhs` for an expansion) is
a dimension of a `Scan`, `Simulator.run` simulates it, and `local`, `sobol`,
`fast`, `morris` and `pce` read the record of the design from the result and
compute the indices of every observable for every label of the other
dimensions and every time point of a timecourse on a grid.

- [`sensitivity.indices`](../api/sensitivity.indices.md): the four analyses of
  a design of its own.
- [`sensitivity.forward`](../api/sensitivity.forward.md): `forward`, the local
  sensitivities of a simulation from the forward sensitivities of the engine
  core, without a design and without a step.
- [`sensitivity.estimators`](../api/sensitivity.estimators.md): the
  estimators of the three global methods, over every element of an
  observable at once.
- [`sensitivity.chaos`](../api/sensitivity.chaos.md): `pce`, the indices of a
  polynomial chaos expansion of a sample, the expansion as a surrogate, and
  `choose_truncation`, the basis a number of points supports, and
  `supported`, whether a number of points supports a basis of a size.
- [`sensitivity.result`](../api/sensitivity.result.md): the `SensitivityResult`
  with the indices, their units, `to_dataframe`, `classify` and netCDF.
- [`sensitivity.plots`](../api/sensitivity.plots.md): the heatmap, the bars of
  the indices and the plane of Morris.
- [`sensitivity.classification`](../api/sensitivity.classification.md): the
  classification of sensitivities.
- [`sensitivity.uncertainty`](../api/sensitivity.uncertainty.md): the bands and
  distributions of a scan over draws.
- [`sensitivity.parameters`](../api/sensitivity.parameters.md): the table of
  the parameters of an analysis, as sbmlsim 0.8 wrote it.
- [`sensitivity.runner`](../api/sensitivity.runner.md): the sensitivity
  analyses of a project over the groups of its simulations, with the files of
  sbmlsim 0.8.5, the kind `sensitivity` of `Project.sensitivities`; not
  imported by this package.
"""

from .chaos import PolynomialChaos, choose_truncation, fit_chaos, pce, supported
from .forward import forward
from .indices import fast, local, morris, sobol
from .plots import plot_heatmap, plot_indices, plot_morris
from .result import SensitivityResult

__all__ = [
    "PolynomialChaos",
    "SensitivityResult",
    "choose_truncation",
    "fast",
    "fit_chaos",
    "forward",
    "local",
    "morris",
    "pce",
    "plot_heatmap",
    "plot_indices",
    "plot_morris",
    "sobol",
    "supported",
]
