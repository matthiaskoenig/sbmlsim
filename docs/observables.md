# Observables

An observable is a quantity computed from every simulation of a run: a timecourse, such as a concentration in ng/ml or normalized to its peak, or one value per simulation, such as the peak itself or the area under the curve. You pass them to `Simulator.run(model, scan, observables)`. There are three types:

- `Formula`: an expression of model variables, with reductions over time such as `max` and `mean`.
- `PK`: the non-compartmental analysis of a concentration with [pkpdutils](https://github.com/matthiaskoenig/pkpdutils), e.g. `cmax`, `auc` and the half-life.
- `Custom`: your own python function.

Observables are evaluated on the full solution of every simulation, before the result is interpolated onto a time grid, so they are as exact as the simulation itself.

## Formulas and reductions

A `Formula` uses the math of PEtab. It reads the variables of the model (`S` is the amount of a species, `[S]` its concentration, plus `time`, parameters and reactions) and other observables by their id. Here the repressilator runs for six values of the Hill coefficient `n`:

```python
import matplotlib.pyplot as plt
import numpy as np

from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Dimension, Formula, Scan, Simulation
from sbmlsim.simulator import Simulator

simulator = Simulator()
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
res = simulator.run(REPRESSILATOR_SBML, scan, observables)
print(res["px_max"].values.round(0))
print(res["px_rel"].dims)
```

```text title="Output"
[ 731. 2363. 5569. 7177. 7860. 8175.]
('hill', 'time')
```

`px_max` has one value per point of the scan. `px_rel` divides a timecourse by a value per simulation, so every simulation is normalized to its own maximum:

```python
fig, ax = plt.subplots(figsize=(7, 3.5))
for n, curve in zip(res["n"].values, res["px_rel"].values):
    ax.plot(res["time"], curve, label=f"n = {n:.1f}")
ax.set(xlabel="time [min]", ylabel="PX / max(PX)")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout()
```

![Output figure](images/outputs/observables-02.png)

A reduction turns the timecourse of each simulation into one value:

| reduction | value |
| --- | --- |
| `max(x)`, `min(x)` | the largest and the smallest value, ignoring `NaN` |
| `mean(x)` | the time-weighted mean: the integral (trapezoidal rule) divided by the simulated time span |
| `at(x, t)` | the value at time `t`, interpolated linearly; `NaN` outside of the simulated times |

A formula that reads only values per simulation is itself a value per simulation, otherwise it is a timecourse. A value per simulation inside a timecourse is the same at every time point, which is how `px / px_max` normalizes to the maximum and `ins / at(ins, 0)` to the baseline. With two or more arguments, `max` and `min` are the elementwise maximum and minimum of PEtab.

The `t` of `at` is a number in the time unit of the model, or a value per simulation with that unit, such as `pk.tmax`. At the time of a change or an event, `at` gives the value after it. Neither the padding of a ragged scan nor the steady state after the end count as time points of a reduction.

## Units

The unit of a formula is derived from the units of what it reads, and a reduction keeps the unit of its argument. With `unit=` the formula is converted into that unit, and the observable has this unit everywhere: in the result and in every observable that reads it.

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
res = simulator.run(MIDAZOLAM_SBML, simulation, [mass, Formula("mid_max", "max(mid)")])
print(res.units["mid"], res.units["mid_max"])
```

```text title="Output"
nanogram / milliliter nanogram / milliliter
```

`[Cve_mid]` is the molar concentration of midazolam in the venous plasma and `Mr_mid` its molar mass, so `mid` is the mass concentration in ng/ml.

Some rules for units:

- A formula that mixes units of one dimension at different scales raises when the scan is compiled, e.g. `mid - [Cve_mid] * Mr_mid` with `mid` in ng/ml and the product in mg/l. Declare one of them as a `Formula` with `unit=`; a factor in the formula does not convert it.
- pint cannot derive a unit for a comparison or a `piecewise`. Such a formula needs `unit=`, which is taken as given: `Formula("high", "piecewise(1, mid > 100, 0)", unit="dimensionless")`. Mixed scales inside it are not checked, so `piecewise(mid, time < 600, [Cve_mid] * Mr_mid)` runs and is off by a factor of 1000 in one branch.
- A formula without a unit it needs, a unit that cannot be converted, an unknown symbol and observables that read each other in a cycle all raise before any simulation runs.

## PK

`PK` runs a non-compartmental analysis of a timecourse with pkpdutils. Every PK parameter is one value per simulation, named `<id>.<parameter>`. The analysis uses the unit of the timecourse, so with `mid` in ng/ml the concentrations are in ng/ml and the times in minutes, the time unit of the model:

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
res = simulator.run(MIDAZOLAM_SBML, doses, observables)
print(res["pk.cmax"].values.round(2), res.units["pk.cmax"])
print(res["auc_per_cmax"].values.round(1), res.units["auc_per_cmax"])
```

```text title="Output"
[21.17 31.84 64.19] nanogram / milliliter
[322.9 322.9 322.8] minute
```

A formula reads a PK parameter by its name: `auc_per_cmax` divides two of them and barely changes with the dose.

The peak of every curve is at (`pk.tmax`, `pk.cmax`):

```python
fig, ax = plt.subplots(figsize=(7, 3.5))
for k, dose in enumerate([5.0, 7.5, 15.0]):
    line = ax.plot(res["time"] / 60, res["mid"][k], label=f"{dose} mg")[0]
    tmax, cmax = res["pk.tmax"][k] / 60, res["pk.cmax"][k]
    ax.plot(tmax, cmax, "o", color=line.get_color())
ax.set(xlabel="time [hr]", ylabel="midazolam plasma [ng/ml]")
ax.legend()
```

![Output figure](images/outputs/observables-05.png)

`res.nca("pk")` returns the `NCAResult` of pkpdutils and `res.to_timecourses("mid")` its `Timecourses`, for further analysis with pkpdutils.

How the doses are found:

- The values the simulation assigns to the dose target are the doses, and their times are the dose times. A dose of a scan dimension and multiple dosing with `Change([0, 12], ...)` therefore need nothing more. A value set before the start is a dose at the start.
- A value of zero is no dose. A control point with a zero dose gets `NaN` for the parameters that need a dose (e.g. `pk.cl_f`), and the other points keep them.
- `dose=` may also be a quantity, a fixed dose at the start. Without a dose, the parameters that need one (clearance, volume) are left out.

The parameters are the ones pkpdutils derives for the route and dosing, e.g. `pk.cmax`, `pk.tmax`, `pk.auc_inf_obs`, `pk.thalf`, `pk.cl_f` and `pk.flags`, the flags of the analysis. `parameters=[...]` keeps a subset and `options=pkpdutils.NCAOptions(...)` changes the settings of the analysis. At a change after the start the timecourse has two rows, the states before and after it; the analysis sees the state after it. The analysis is only as exact as the time points of the simulation. The steps of the integrator can be coarse in a smooth elimination phase, so run a non-compartmental analysis on a simulation with `steps` or `times`.

## Custom functions

A `Custom` observable calls a function `function(time, values)` once per simulation. `time` holds the time points in the time unit of the model and `values` a dict with the timecourses of the `symbols`, each in its unit. A time appears twice at a change after the start or an event (the states before and after it). The function returns a float, or an array of the length of `time` with `kind=ObservableKind.TIMECOURSE` (`ObservableKind` is in `sbmlsim.simulation`). You give the unit of the result.

The function must be defined at the top level of a module, so that it can be sent to the worker processes of a scan; a lambda or a closure raises. `examples/observables.py` has `time_above`, the time during which `mid` (in ng/ml) is above 25 ng/ml:

```python
from examples.observables import time_above

res = simulator.run(
    MIDAZOLAM_SBML,
    doses,
    [mass, Custom("t_above", time_above, "min", symbols=["mid"])],
    keep=["t_above"],
)
print(res["t_above"].values)
```

```text title="Output"
[  0. 144. 324.]
```

## Keep only what you need

`keep` lists the observables in the result, all of them by default. The id of a `PK` observable keeps all of its parameters. Other observables are computed where a kept one needs them and then dropped, and an observable that no kept one needs is not computed at all:

```python
res = simulator.run(MIDAZOLAM_SBML, doses, observables, keep=["pk"])
print(len(res.variables), res.variables[:3], "time" in res.ds.dims)
```

```text title="Output"
44 ['pk.cmax', 'pk.tmax', 'pk.cmin'] False
```

A result whose kept observables are all values per simulation has no time dimension. This saves a lot of memory in a large scan: 100,000 points with 481 time points and two timecourses take 0.8 GB, the same points with ten values per simulation 8 MB.

`keep` may also name variables of the model, e.g. `keep=["pk.cmax", "[Cve_mid]"]`. A kept variable is a timecourse under its own name and in the unit of the model, so one run gives both the values per simulation and the timecourses for a figure.

## Errors

With `on_error="flag"`, a point that fails gets `NaN` for every observable and the variable `status` marks it, see [Parameter scans](scans.md); `res.nca` gives it the flag `NCAFlag.NO_DATA`. A point fails exactly when its own simulation or its own observables fail. With `on_error="raise"` (the default), the run raises for the first failing point in the order of the scan.
