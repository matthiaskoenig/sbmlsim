# Observables

An observable is what a scan computes from every simulation: a timecourse, such as a concentration normalized to its maximum, or a value per simulation, such as the maximum itself or the area under the curve. `Simulator.run(model, scan, observables, keep=...)` evaluates them in the workers on the native solution of every simulation, before any interpolation onto a grid, so they are as exact as the output of the simulation. There are three kinds: a `Formula` of the math of PEtab with reductions over the time, the non-compartmental analysis of pkpdutils (`PK`) and a `Custom` function; `keep` chooses which of them the result keeps.

## Formulas and reductions

A `Formula` reads the selections of the model (`S` an amount, `[S]` a concentration, `time`, parameters, reactions) and other observables by their id:

```python
import numpy as np

from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Dimension, Formula, Scan, Simulation
from sbmlsim.simulator import Simulator

scan = Scan(
    Simulation(end=500, steps=500),
    [Dimension("hill", values={"n": np.linspace(1.5, 4.0, 6)})],
)
observables = [
    Formula("px", "PX"),
    Formula("px_max", "max(px)"),
    Formula("px_mean", "mean(px)"),
    Formula("px_250", "at(px, 250)"),
    Formula("px_rel", "px / px_max"),
]
res = Simulator().run(REPRESSILATOR_SBML, scan, observables)
print(res["px_max"].values)  # a value per point of the scan
print(res["px_rel"].dims)  # ('hill', 'time')
```

Four reductions reduce the time of every simulation on its own:

| reduction | value |
| --- | --- |
| `max(x)`, `min(x)` | the largest and the smallest value, ignoring `NaN` |
| `mean(x)` | the time weighted mean: the trapezoidal integral divided by the time between the first and the last time point, which does not depend on the steps of the integrator |
| `at(x, t)` | the value at the time `t`, a number in the time unit of the model or a quantity of a time unit, interpolated linearly; at the time of a change the value after it, outside of the simulated times `NaN` |

A formula which reads only values per simulation is a value per simulation, otherwise a timecourse; a value per simulation in a timecourse is the same at every time, which is how `px / px_max` normalizes every simulation to its own maximum and `ins / at(ins, 0)` to its baseline. `max` and `min` with two or more arguments are the elementwise maximum and minimum of PEtab. The padding of a ragged scan and the steady state after the end are no time points of a reduction.

## Units

The unit of a formula is derived from the units of what it reads, the reductions keep the unit of their argument. A formula with `unit=` is converted into it, and only the result is converted: inside of the run every observable keeps its natural unit, the derived one, which is the unit a `PK` and a `Custom` see. A formula which mixes units of one dimension at different scales, e.g. ng/ml and mg/l, raises when the scan is compiled, while a comparison of mixed scales is not detected:

```python
from sbmlsim import Q
from sbmlsim.resources import MIDAZOLAM_SBML
from sbmlsim.simulation import PK, Change, Custom

simulation = Simulation(
    time_unit="hr",
    end=24,
    steps=480,
    changes=[Change(0, {"PODOSE_mid": Q(7.5, "mg")})],
)
mass = Formula("mid", "[Cve_mid] * Mr_mid", unit="ng/ml")
res = Simulator().run(
    MIDAZOLAM_SBML, simulation, [mass, Formula("mid_max", "max(mid)")]
)
print(res.units["mid"], res.units["mid_max"])
```

Where pint cannot derive a unit, e.g. for `piecewise` or a comparison, the formula needs `unit=`, which is then taken as declared: `Formula("high", "piecewise(1, mid > 100, 0)", unit="dimensionless")`. The same holds for a formula which derives a dimensionless unit from symbols which have no unit or from numbers only. The compile step of a scan raises for a formula without a unit it needs, for a unit which cannot be converted, for a symbol which is neither a selection nor an observable and for observables which read each other in a cycle, before any simulation runs.

## PK

`PK` is the non-compartmental analysis of a timecourse with pkpdutils, every parameter a value per simulation `<id>.<parameter>`. The analysis runs on the natural unit of the timecourse, so the parameters of the example below are in `g*mmol/l/mol` (mg/l) and minutes:

```python
doses = Scan(
    simulation,
    [Dimension("dose", values={"PODOSE_mid": Q([5.0, 7.5, 15.0], "mg")})],
)
observables = [
    mass,
    PK("pk", "mid", dose="PODOSE_mid", route="oral"),
    Formula("auc_per_cmax", "pk.auc_inf_obs / pk.cmax"),
]
res = Simulator().run(MIDAZOLAM_SBML, doses, observables)
print(res["pk.cmax"].values, res.units["pk.cmax"])
print(res["auc_per_cmax"].values, res.units["auc_per_cmax"])
nca = res.nca("pk")  # the NCAResult of pkpdutils
timecourses = res.to_timecourses("mid")  # the Timecourses of pkpdutils
```

The doses are read from every point of the scan: the values the simulation assigns to the dose target are the doses and the times of these values their times, so a dose of a dimension and a multiple dosing by `Change([0, 12], ...)` need no further input; a value before the initialization is a dose at the start. A quantity is a fixed dose at the start, and without a dose the parameters which need one (clearance, volume) are left out. The parameters are those pkpdutils derives for the dosing, e.g. `pk.cmax`, `pk.tmax`, `pk.auc_inf_obs`, `pk.thalf`, `pk.cl_f`, and `pk.flags`, the flags of the analysis; `parameters=[...]` keeps a subset and `options=pkpdutils.NCAOptions(...)` sets the analysis. A formula reads a parameter by its name. The points are analysed in groups of the same number of doses and of time points, so no row is padded and a parameter does not depend on the chunking of the scan. The analysis is as exact as the time points of the simulation: the steps of the integrator can be coarse in a smooth elimination phase and make many small groups, so a non-compartmental analysis is best run on `steps` or `times`.

## Custom functions

A `Custom` observable is a function of a module, `function(time, values)`, called once per simulation with its time points in the time unit of the model and the values of its symbols in their natural unit; it returns a float, or an array of the length of `time` with `kind=ObservableKind.TIMECOURSE`. The unit of the result is given by the observable. The example function reads the concentration `mid`, whose natural unit is mg/l, and compares it with a threshold of 25 ng/ml:

```python
from examples.observables import time_above

res = Simulator().run(
    MIDAZOLAM_SBML,
    doses,
    [mass, Custom("t_above", time_above, "min", symbols=["mid"])],
    keep=["t_above"],
)
print(res["t_above"].values)
```

The function must be defined at the top level of a module, so that it pickles for the workers of a scan; a lambda or a closure raises.

## keep and memory

`keep` is the list of observables in the result, all by default; the id of a PK observable keeps all its parameters. The other observables are evaluated where a kept one needs them and dropped, and an observable no kept one needs is not evaluated at all:

```python
res = Simulator().run(MIDAZOLAM_SBML, doses, observables, keep=["pk"])
print(sorted(res.ds.data_vars)[:3], "time" in res.ds.dims)
```

`keep` is the control of the memory of a large scan: 1e5 points with 481 time points and two timecourses are 0.8 GB, the same points with ten values per simulation 8 MB. A result whose kept observables are all values per simulation has no time dimension.

## Errors

With `on_error="flag"` every observable of a point which fails is `NaN` and the variable `status` marks the point, see [Parameter scans](scans.md). The observables are evaluated on a chunk of points at once; after a failure the points are evaluated one at a time, so a point fails exactly when its own simulation or its own observables fail, whichever observable it is. With `on_error="raise"` the first failing point in the order of the scan is raised, an observable failure of an earlier point before a failure of the simulation of a later one.
