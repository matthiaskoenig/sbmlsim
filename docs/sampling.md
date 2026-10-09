# Sampling and uncertainty

The sampler `sbmlsim.simulation.sampling` creates designs: dimensions of a scan whose values follow a design. Every design maps points of the unit cube through the inverse cumulative distribution function of a distribution per target, so any marginal works with every design, and it records how it was drawn (the method, the distributions, the options, the seed and the references), which the result keeps as its provenance.

## Distributions

```python
import numpy as np

from sbmlsim import Q
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Formula, Scan, Simulation, sampling
from sbmlsim.simulation.sampling import LogNormal, Normal, Truncated, Uniform
from sbmlsim.simulator import Simulator

simulator = Simulator()
model = simulator.load(REPRESSILATOR_SBML)
print(Uniform(1.0, 2.0).ppf(np.array([0.0, 0.5, 1.0])))
print(
    Truncated(Normal(Q(75, "kg"), Q(12, "kg")), lower=Q(40, "kg")).ppf(
        np.array([0.01, 0.5])
    )
)
```

| distribution | values |
| --- | --- |
| `Uniform(lower, upper)`, `Uniform(relative=r)` | uniform in the bounds, or in the reference times `[1 - r, 1 + r]` |
| `LogUniform(lower, upper)`, `LogUniform(factor=f)` | uniform in log10, or between the reference divided and multiplied by `f` |
| `Normal(mean, sd)`, `Normal(cv=c)` | normal, around the reference with `sd = c * |reference|` |
| `LogNormal(median, cv)`, `LogNormal(cv=c)` | lognormal, around the reference |
| `Truncated(distribution, lower, upper)` | the distribution restricted to the interval |
| `Empirical(values)` | the values with equal weights |
| `Fixed(value)` | one value for every point |

A distribution without a location is relative to the reference of its target: the value the model gives the target after the pre-initialization of the simulation, i.e. the changes of the model and of the simulation and the initial assignments; the design then needs the model, `model=`. Numbers are in the unit of the target in the model, quantities are converted by the run.

## Designs

```python
parameters = sampling.parameters_of(model)
simulation = Simulation(end=200, steps=200)

local = sampling.local(parameters, delta=0.1, model=model)
draws = sampling.lhs(
    {pid: LogNormal(cv=0.1) for pid in parameters}, 50, seed=1, model=model
)
print(local.labels[:3], len(draws), draws.design.method, draws.design.options["seed"])
```

| design | points |
| --- | --- |
| `local(targets, delta, model=...)` | the reference and every target alone at `1 + delta` and `1 - delta` times its reference |
| `random(distributions, n, seed=..., correlation=...)` | independent draws, or correlated ones (a Gaussian copula) |
| `lhs(distributions, n, seed=..., correlation=...)` | a Latin hypercube; with a correlation the rank reordering of Iman and Conover keeps its strata |
| `sobol(distributions, n, seed=...)`, `fast(...)`, `morris(...)` | the designs of SALib for the global sensitivity analyses |
| `fit_parameters(fisher, n, seed=...)` | the parameters of a fit from the normal of its Fisher covariance |
| `profile_parameters(identifiability, n, seed=...)` | every parameter of a fit from its profile likelihood, which follows asymmetric and open confidence intervals |
| `fit_repeats(result, size)` | the best parameter sets of the repeats of a fit |
| `population(function, covariates, n, seed=...)` | covariates drawn and mapped to the values of targets by a function of a module; the covariates are coordinates of the result |

A design is a dimension like any other, so it combines with the other dimensions of a scan, e.g. doses or conditions. `seed=None` draws a seed and records it, so a result can always be drawn again.

## Uncertainty

The uncertainty of a prediction is a scan over draws of the uncertain parameters and its summary over their dimension:

```python
from sbmlsim.sensitivity.uncertainty import plot_bands, plot_distribution

res = simulator.run(
    model,
    Scan(simulation, [draws]),
    [Formula("px", "PX"), Formula("px_max", "max(PX)")],
)
bands = res.summary("lhs", quantiles=[0.05, 0.5, 0.95])
figure = plot_bands(bands, "px")
figure.savefig("bands.png")
plot_distribution(res, "px_max", dim="lhs").savefig("px_max.png")
```

`plot_bands` draws the median and the band between the 5 % and the 95 % quantile of every label of the other dimensions, `plot_distribution` the distribution of a value per simulation as a histogram or a box. The draws of the parameters of a fit, `fit_parameters` and `profile_parameters`, give the uncertainty of the predictions of a fitted model.
