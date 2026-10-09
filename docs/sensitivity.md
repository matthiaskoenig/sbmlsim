# Sensitivity analysis

Sensitivity analysis quantifies how the outputs of a model depend on its parameters. An analysis of `sbmlsim.sensitivity` is three steps: a design of the sampler, a run and the indices on the result. The design (`local`, `sobol`, `fast` or `morris` of `sbmlsim.simulation.sampling`, see [Sampling and uncertainty](sampling.md)) is a dimension of a `Scan`, `Simulator.run` simulates its points with the observables, and `sensitivity.local`, `sensitivity.sobol`, `sensitivity.fast` and `sensitivity.morris` read the record of the design from the result and compute the local indices and the global indices of [SALib](https://salib.readthedocs.io), see [References](references.md#sensitivity-analysis).

The indices are computed for every observable and for every label of the other dimensions of the scan, e.g. a dose or a condition, and for every time point of a timecourse on a common time grid (`time=` of the run, or a simulation with `steps` or `times`). A scan can combine a design with any other dimensions, so one run gives the sensitivities for all conditions, see [Conditions and time points](#conditions-and-time-points). A timecourse with its native time points (ragged) has no common time points to compute indices on and raises, ask for a grid with `time=` or a simulation with `steps` or `times`. `observables=` restricts an analysis to some observables of the result, and `dim=` chooses the design when the result has several of the method.

The examples below use the repressilator and three of its parameters.

```python
from sbmlsim import sensitivity
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Dimension, Formula, Scan, Simulation, sampling
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

The local design changes every parameter alone to `1 + delta` and `1 - delta` times its reference, and the analysis takes central differences. `raw` is the change of the observable per change of the parameter, in the unit of the observable per unit of the parameter (no unit when the parameters have different units), `normalized` the relative change of the observable per relative change of the parameter, which is dimensionless and comparable between parameters and observables.

```python
design = sampling.local(parameters, 0.01, model=model)
result = simulator.run(model, Scan(simulation, [design]), observables)
local = sensitivity.local(result)
print(local.index("normalized").to_pandas().round(3))
```

A normalized index is `NaN` for a zero reference of the observable. A parameter whose reference is zero is not moved by the design, so both of its indices are `NaN`: undefined, not insensitive, `classify` gives `""` for it and the heatmap keeps its row in grey.

## Sobol indices

The Sobol design of Saltelli has `n (d + 2)` points for `d` parameters, `n` a power of two. `S1` is the first order index, the share of the variance of the observable which one parameter explains alone, `ST` the total index with all its interactions, and `S2` (`second_order=True` of the design) the share of a pair. Every index has an interval, `S1_conf` (`conf_level`, `num_resamples`). The indices are dimensionless.

```python
design = sampling.sobol(bounds, 64, seed=1, model=model)
result = simulator.run(model, Scan(simulation, [design]), observables)
sobol = sensitivity.sobol(result)
print(sobol.index("ST").to_pandas().round(3))
```

## FAST and Morris

The extended FAST (`S1` and `ST` with intervals) needs more than `4 m^2` points per parameter (`m=4`); SALib documents its bootstrap intervals as unreliable, so `S1_conf` and `ST_conf` of FAST are indicative only. The Morris screening is the cheapest, with the mean `mu`, the mean of the absolute effects `mu_star`, the standard deviation `sigma` of the elementary effects of the trajectories and the interval `mu_star_conf`; a large `sigma` against `mu_star` points to interactions or nonlinear effects.

The elementary effects have the unit of the observable. SALib divides the change of the observable by its jump `levels / (2 (levels - 1))` on its grid of levels, and one step of a trajectory moves the probability of one parameter by `1/2`, so an effect is `2 (levels - 1) / levels` times the change of the observable over one step: `1.5` times for the default of four levels.

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

## Failed points and constant elements

A simulation of a point which fails gives `NaN` for the point (`on_error="flag"` of the run), and the indices which use it are `NaN`, with one warning which counts the elements: every index of the element for Sobol, FAST and Morris, the indices of the parameter whose point failed (or of every parameter for the reference) for the local analysis.

An element whose values vary by no more than the error of the integrator is constant: its range is at most `tolerance * max|y|`. A constant element, e.g. a saturated or a conserved value or a steady state, has `NaN` Sobol and FAST indices, since it has no variance to share out, and zero Morris effects. The tolerance is 1000 times the relative tolerance of the integrator, which the result records in `attrs["integrator_settings"]` (the error of a value is up to about 50 relative tolerances): `1e-7` for the default `1e-10` of `Simulator`. Below `1e-10` the error of the integrator does not decrease any more, so the tolerance stays `1e-7`, and a result without the record takes the default. `tolerance=` of `sobol`, `fast` and `morris` sets it, `0` keeps every element which is not exactly constant; the options of a result record the one it used. A value which has decayed to the level of the absolute tolerance of the integrator varies by its error relative to itself and is not caught, leave such time points out. The local indices of a constant element are zero within the error of the integrator.

## Conditions and time points

A dimension of conditions next to the design gives the indices of every condition, and a timecourse on a grid the indices of every time point. The plots select one label of every other dimension by keyword.

```python
leak = Dimension("leak", values={"ps_0": [0.0005, 0.005]}, labels=["low", "high"])
result = simulator.run(
    model,
    Scan(simulation, [leak, sampling.local(parameters, 0.01, model=model)]),
    [*observables, Formula("px", "PX")],
)
by_condition = sensitivity.local(result)
print(by_condition["px_max.normalized"].to_pandas().round(3))
print(by_condition["px.normalized"].dims, by_condition.units["time"])
sensitivity.plot_heatmap(
    by_condition, "normalized", leak="high", path="local_high.png", dpi=72
)
at_100 = by_condition.sel(time=100.0)
print(at_100.index("normalized", observables=["px", "px_max"]).sel(leak="low").values)
```

The result keeps the labels of the other dimensions, the values a dimension changes along them (`ps_0` over `leak`) and the time, with their units in `attrs["units"]`. An analysis calls SALib once per element, about 10 ms for a Sobol design of `n = 1024` and five parameters, so a timecourse of 1000 time points under three conditions takes about half a minute per observable; ask the run for the time points the analysis needs with `time=`.

## The result

A `SensitivityResult` wraps an `xarray.Dataset` (`ds`) with a variable per observable and index, `<observable>.<index>`, over `(parameter, *dimensions, [time])`, the unit of every variable and coordinate in `attrs["units"]` and the method, its options and the provenance of the scan in `attrs`. Second order indices are over `(parameter, parameter_2, ...)`.

```python
print(sorted(sobol.ds.data_vars))
print(sobol.method, sobol.parameters, sobol.observables)
print(sobol["px_max.ST"].sel(parameter="n").item())
print(sobol.sel(parameter="n").parameters)
df = sobol.to_dataframe("px_max.ST")
stacked = sobol.index("ST")
print(df.shape, stacked.dims)
```

`index(name)` stacks an index of the scalar observables into one array over `(parameter, observable, ...)`; a timecourse named in `observables=` raises, it is stacked at one time point after `sel(time=...)`. `sel` and `isel` select labels, a single label of `parameter` keeps the dimension, `to_dataframe` gives a table and `to_netcdf`/`from_netcdf` write and read the result. `classify(name)` applies the classification of sensitivities of the IPCS to every value of an index, as `SensitivityClassification` values, see `sensitivity.classification`; its thresholds are the ones of normalized sensitivities, so it takes the dimensionless indices (`normalized`, `S1`, `ST`, ...) and raises for an index with a unit.

```python
print(local.classify("px_max.normalized").values)
sobol.to_netcdf("sobol.nc")
print(sensitivity.SensitivityResult.from_netcdf("sobol.nc").method)
```

## Plots

The plots are functions which return a figure and save it only with a `path`. The labels of the other dimensions of the scan are selected by keyword, e.g. `dose=0`.

```python
sensitivity.plot_heatmap(local, "normalized", path="local.png", dpi=72)
sensitivity.plot_heatmap(morris, "mu_star", path="morris_mu_star.png", dpi=72)
sensitivity.plot_indices(sobol, "px_max", path="sobol.png", dpi=72)
sensitivity.plot_morris(morris, "px_max", path="morris.png", dpi=72)
```

| plot | shows |
| --- | --- |
| `plot_heatmap(result, index, ...)` | an index of all scalar observables over the parameters, clustered, with its value in every cell and a colorbar labelled with the index and its unit; the signed indices (`raw`, `normalized` and `mu`) on a diverging color map around 0 |
| `plot_indices(result, observable)` | the bars of `S1` and `ST` of a Sobol or FAST analysis with their intervals |
| `plot_morris(result, observable)` | `mu_star` against `sigma` with a point per parameter, both axes from 0; a note names a parameter with undefined effects |

The `cutoff` of the heatmap leaves out the parameters whose values are all below it, absolute in the unit of the index: by default `0.1` for the dimensionless indices (`normalized`, `S1`, `ST`) and none for the indices with a unit (`raw`, `mu`, `mu_star`, `sigma`), `cutoff=0` keeps every parameter. A heatmap in which no parameter reaches the cutoff raises and names the largest value; an undefined value is never left out, its cell is grey. All observables of a heatmap share one color scale, so for an index with a unit the observables with the largest values dominate; `normalized` compares observables of different units.

The uncertainty of the outputs of a scan over draws is analysed with the bands and distributions of `sbmlsim.sensitivity.uncertainty`, see [Sampling and uncertainty](sampling.md#uncertainty).

The complete example with all four methods, a dimension of conditions and the figures is `examples/sensitivity/sensitivity_example.py`; `--cores` sets the number of workers of the simulator and `--quick` runs it with small designs and figures.
