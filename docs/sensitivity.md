# Sensitivity analysis

Sensitivity analysis quantifies how the outputs of a model depend on its parameters. An analysis of `sbmlsim.sensitivity` is three steps: a design of the sampler, a run and the indices on the result. The design (`local`, `sobol`, `fast` or `morris` of `sbmlsim.simulation.sampling`, see [Sampling and uncertainty](sampling.md)) is a dimension of a `Scan`, `Simulator.run` simulates its points with the observables, and `sensitivity.local`, `sensitivity.sobol`, `sensitivity.fast` and `sensitivity.morris` read the record of the design from the result and compute the indices of the global methods of [SALib](https://salib.readthedocs.io), see [References](references.md#sensitivity-analysis).

The indices are computed for every observable and for every label of the other dimensions of the scan, e.g. a dose or a condition, and for every time point of a timecourse on a grid (`time=` of the run). A scan can combine a design with any other dimensions, so one run gives the sensitivities for all conditions. A timecourse with its native time points (ragged) has no common time points to compute indices on and raises, ask for a grid with `time=`.

The examples below use the repressilator and three of its parameters.

```python
from sbmlsim import sensitivity
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Formula, Scan, Simulation, sampling
from sbmlsim.simulation.sampling import Uniform
from sbmlsim.simulator import Simulator

simulator = Simulator(n_workers=1)
model = simulator.load(REPRESSILATOR_SBML)
simulation = Simulation(end=200, steps=200)
parameters = ["n", "tau_prot", "ps_a"]
observables = [Formula("px_max", "max(PX)"), Formula("px_mean", "mean(PX)")]
bounds = {pid: Uniform(relative=0.2) for pid in parameters}
```

## Local sensitivities

The local design changes every parameter alone to `1 + delta` and `1 - delta` times its reference, and the analysis takes central differences. `raw` is the change of the observable per unit of the parameter, `normalized` the relative change of the observable per relative change of the parameter, which is comparable between parameters and observables.

```python
design = sampling.local(parameters, 0.01, model=model)
result = simulator.run(model, Scan(simulation, [design]), observables)
local = sensitivity.local(result)
print(local.index("normalized").to_pandas().round(3))
```

A normalized index is `NaN` for a zero reference of the parameter or of the observable.

## Sobol indices

The Sobol design of Saltelli has `n (d + 2)` points for `d` parameters, `n` a power of two. `S1` is the first order index, the share of the variance of the observable which one parameter explains alone, `ST` the total index with all its interactions, and `S2` (`second_order=True` of the design) the share of a pair. Every index has an interval, `S1_conf` (`conf_level`, `num_resamples`).

```python
design = sampling.sobol(bounds, 64, seed=1, model=model)
result = simulator.run(model, Scan(simulation, [design]), observables)
sobol = sensitivity.sobol(result)
print(sobol.index("ST").to_pandas().round(3))
```

## FAST and Morris

The extended FAST (`S1` and `ST` with intervals) needs more than `4 m^2` points per parameter (`m=4`). The Morris screening is the cheapest, with the mean `mu`, the mean of the absolute effects `mu_star`, the standard deviation `sigma` of the elementary effects of the trajectories and the interval `mu_star_conf`; a large `sigma` against `mu_star` points to interactions or nonlinear effects.

```python
fast = sensitivity.fast(
    simulator.run(
        model,
        Scan(simulation, [sampling.fast(bounds, 65, seed=1, model=model)]),
        observables,
    )
)
morris = sensitivity.morris(
    simulator.run(
        model,
        Scan(simulation, [sampling.morris(bounds, 10, seed=1, model=model)]),
        observables,
    )
)
print(fast.index("ST").to_pandas().round(3))
print(morris.index("mu_star").to_pandas().round(3))
```

A simulation of a point which fails gives `NaN` for the point (`on_error="flag"` of the run), and the indices of the elements which contain it are `NaN`, with one warning which counts them. The indices of a constant observable are `NaN` for Sobol and FAST and zero for the local and the Morris analysis.

## The result

A `SensitivityResult` wraps an `xarray.Dataset` (`ds`) with a variable per observable and index, `<observable>.<index>`, over `(parameter, *dimensions, [time])`, the unit of every observable in `attrs["units"]` and the method, its options and the provenance of the scan in `attrs`. Second order indices are over `(parameter, parameter_2, ...)`.

```python
print(sorted(sobol.ds.data_vars))
print(sobol.method, sobol.parameters, sobol.observables)
print(sobol["px_max.ST"].sel(parameter="n").item())
df = sobol.to_dataframe("px_max.ST")
stacked = sobol.index("ST")
print(df.shape, stacked.dims)
```

`index(name)` stacks an index of the scalar observables into one array over `(parameter, observable, ...)`, `sel` and `isel` select labels, `to_dataframe` gives a table and `to_netcdf`/`from_netcdf` write and read the result. `classify(name)` applies the classification of sensitivities of the IPCS to every value of an index, e.g. of the normalized local sensitivities, as `SensitivityClassification` values, see `sensitivity.classification`.

```python
print(local.classify("px_max.normalized").values)
sobol.to_netcdf("sobol.nc")
print(sensitivity.SensitivityResult.from_netcdf("sobol.nc").method)
```

## Plots

The plots are functions which return a figure and save it only with a `path`. The labels of the other dimensions of the scan are selected by keyword, e.g. `dose=0`.

```python
sensitivity.plot_heatmap(local, "normalized", cutoff=None, path="local.png", dpi=72)
sensitivity.plot_indices(sobol, "px_max", path="sobol.png", dpi=72)
sensitivity.plot_morris(morris, "px_max", path="morris.png", dpi=72)
```

| plot | shows |
| --- | --- |
| `plot_heatmap(result, index, ...)` | an index of all scalar observables over the parameters, clustered, with a `cutoff` for the parameters without an effect; for `raw` and `normalized` on a diverging color map around 0 |
| `plot_indices(result, observable)` | the bars of `S1` and `ST` of a Sobol or FAST analysis with their intervals |
| `plot_morris(result, observable)` | `mu_star` against `sigma` with a point per parameter |

The uncertainty of the outputs of a scan over draws is analysed with the bands and distributions of `sbmlsim.sensitivity.uncertainty`, see [Sampling and uncertainty](sampling.md#uncertainty).

The complete example with all four methods, a dimension of conditions and the figures is `examples/sensitivity/sensitivity_example.py`; `--cores` sets the number of workers of the simulator and `--quick` runs it with small designs and figures.
