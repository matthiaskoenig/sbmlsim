# The analyses: sampling, sensitivity and uncertainty on the scan core (#249, sub-project 3)

The analyses of a model are scans: a design of parameter values is one dimension of a `Scan`, `Simulator.run` simulates it with observables, and the analysis is computed on the arrays of the `ScanResult`. One sampler module creates every design as a `Dimension`; the sensitivity analyses compute their indices from a result; the uncertainty analysis is a design, a run and `ScanResult.summary`. The analyses contain no simulation loop and no pool of their own, and they get the integrator settings of the `Simulator`, i.e. the tolerances per state.

This design is sub-project 3 of `2026-10-08-scan-core-design.md`, whose outline it follows; phases 1 and 2 of that design (the scan core and the observables) are merged into `develop` (`344e43ee`).

## The goal

- One sampler, `sbmlsim.simulation.sampling`: distributions with explicit or relative locations, and designs (`local`, `random`, `lhs`, `sobol`, `fast`, `morris`, `fit_parameters`, `profile_parameters`, `fit_repeats`, `population`) which return a `Dimension`. It replaces `ModelSensitivity`, the samplers of `sensitivity/` and the sampling of `fit/sampling.py`.
- Self-describing designs: every design dimension carries a serializable record (method, distributions, options, seed, references), which the scan writes into the provenance of the result, so an analysis needs only the result, also after `ScanResult.from_netcdf`.
- Sensitivity analyses as functions of a result: `local`, `sobol`, `fast` and `morris` give a `SensitivityResult` with the indices over `(parameter, *other dimensions, [time])`.
- The uncertainty analysis needs no own engine: a design of draws, a run, `summary`, and plots of bands and distributions.
- No API compatibility: `SensitivitySimulation`, the analysis classes of `sensitivity/`, `ModelSensitivity` and `SensitivityParameter` are replaced; callers, examples, tests and docs are migrated.

## The starting point

Measured on `develop` (`344e43ee`).

| | |
| --- | --- |
| `sensitivity/` | 2238 lines in 10 files: `SensitivitySimulation.simulate(r, changes)` written by the user on raw roadrunner (`resetAll`, `setValue`, `r.simulate`), its own `process_context().Pool` which pickles the simulation and the roadrunner instance into every task, static chunks, five analysis classes (local, sampling, Sobol, FAST, Morris) with `execute()`, `plot()`, a pickle cache and tables and figures in `results_path` |
| samples | uniform within `[lower_bound, upper_bound]` only (`SensitivityParameter`), no log or other distribution, no correlation |
| observables | written by hand inside `simulate`, e.g. the AUC with `np.trapezoid` and the maximum with `argmax` (`examples/sensitivity/sensitivity_example.py:50`) |
| conditions | `AnalysisGroup(changes)` per condition, each analysed separately |
| `ModelSensitivity` | difference and distribution scans (`simulation/sensitivity.py`), no seed (global numpy random state), normal draws which can go negative, the references read without the changes of the model |
| `fit/sampling.py` | the start values of a fit, uniform or log-uniform, random or LHS, tied to `FitParameter`, returns a `DataFrame` |
| fit uncertainty | `fit/fisher.py` (the covariance in the space of the `parameter_scale`) and `fit/identifiability.py` (profile likelihoods with asymmetric or open confidence intervals) compute what a fit knows about its parameters, but nothing turns it into simulations |

## Decisions

| | |
| --- | --- |
| compatibility | none; the old analysis classes, `SensitivitySimulation`, `SensitivityParameter`, `parameters_from_sbml` and `ModelSensitivity` are removed |
| design | a `Dimension` of values with a design record; a scan with a design is an ordinary scan, which other dimensions (doses, conditions, simulations, models) multiply |
| references | a distribution takes explicit numbers or quantities; one without a location is relative to the reference of its target, the value the model gives the target after the pre-initialization of the simulation (the changes of the model and of the simulation and the initial assignments) |
| marginals | every design maps points of the unit cube through the inverse CDF of each distribution, so any marginal works with every design |
| correlation | the Spearman rank correlation of the values, converted to the correlation of normal scores (`2 sin(π ρ / 6)`); a Gaussian copula for `random`, the rank reordering of Iman and Conover for `lhs` (it keeps the strata); refused for `sobol`, `fast` and `morris`, whose indices assume independence |
| seed | every random design takes `seed`; `None` draws a seed, which the record keeps, so a result is reproducible from its record alone |
| fit uncertainty | `fit_parameters` draws from the normal of the Fisher covariance (correlated, local), `profile_parameters` from the profile likelihood of every parameter (independent, follows asymmetric and open profiles), `fit_repeats` takes the parameter sets of the repeats of a fit |
| sensitivity | functions of a `ScanResult`; the indices per array element, i.e. per label of every other dimension and per time point of a timecourse on a common grid |
| result | `SensitivityResult`, an `xarray.Dataset` with one variable per observable and index, units, netCDF, `to_dataframe`, plots as functions; no `results_path`, no pickle cache (the `ScanResult` of the run is the cache) |
| uncertainty | a design of draws, `Simulator.run`, `ScanResult.summary` with quantiles; plots of bands and distributions |
| fit start values | `fit/sampling.py` keeps `SamplingType` and `create_samples`, implemented by the sampler with the same values for the same seed |

## The sampler

```python
from sbmlsim import Q
from sbmlsim.simulation import sampling
from sbmlsim.simulation.sampling import LogNormal, LogUniform, Normal, Truncated, Uniform

design = sampling.lhs(
    {
        "k1": LogNormal(cv=0.2),                         # median: the reference of k1
        "BW": Truncated(Normal(Q(75, "kg"), Q(12, "kg")), lower=Q(40, "kg")),
        "fumic": Uniform(0.2, 0.6),
    },
    n=500,
    seed=1,
    model=model,                                         # resolves the relative k1
    correlation=[[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]],
)
scan = Scan(simulation, [design, Dimension("dose", values={"PODOSE_mid": Q([5, 10], "mg")})])
```

`sbmlsim.simulation.sampling` is a package: distributions, references, designs, the designs of a fit, populations and the record.

### Distributions

Frozen and serializable (`to_dict`); the numbers are floats in the unit of the target in the model or quantities.

- `Uniform(lower, upper)` or `Uniform(relative=r)`: the reference times `[1 - r, 1 + r]`;
- `LogUniform(lower, upper)` or `LogUniform(factor=f)`: the reference divided and multiplied by `f`;
- `Normal(mean, sd)` or `Normal(cv=c)`: the mean is the reference, `sd = c * |reference|`;
- `LogNormal(median, cv)` or `LogNormal(cv=c)`: the median is the reference;
- `Truncated(distribution, lower=None, upper=None)`: the distribution restricted to the interval, by its inverse CDF on the restricted probabilities;
- `Empirical(values)`: the values with equal weights (the inverse CDF of their sorted values);
- `Fixed(value)`: one value for every point (a target which a design must keep, e.g. a start value of a fit without bounds).

Every distribution has `ppf(u, reference)`: the values at the probabilities `u` of the unit cube, vectorized; the probabilities 0 and 1 of an unbounded distribution are clipped into the open interval. A relative distribution without a reference raises with the target and the remedy (`model=`).

### References

`sampling.references(model, targets, simulation=None) -> dict[str, Quantity]` reads the value every target has after the pre-initialization: the simulation (a `Simulation(end=1)` by default) is compiled with the changes of the model as defaults (`Simulator.compile`), the model is initialized with the plan and the values are read in the unit of the target in the model. It initializes the model, which every simulation of the model does anyway. `sampling.parameters_of(model, *, species=False, exclude=None, exclude_zero=True) -> list[str]` lists the constant parameters (and the initial amounts of the species) which a local or global analysis of all parameters varies, as `ModelSensitivity` did. A design with a relative distribution takes `model=` and `simulation=` and resolves the references once, before it samples.

### Designs

| design | points |
| --- | --- |
| `local(targets, delta=0.1, *, model, simulation=None, id="local")` | the reference point and every target alone at `(1 + delta)` and `(1 - delta)` times its reference, `2 k + 1` points labelled `reference`, `<target>+`, `<target>-` |
| `random(distributions, n, *, seed=None, correlation=None, model=None, simulation=None, id="random")` | independent draws, `rng.random((n, d))`; with a correlation standard normals with the Cholesky factor of the correlation, mapped through the normal CDF (a Gaussian copula) |
| `lhs(distributions, n, *, seed=None, correlation=None, ...)` | a Latin hypercube, `qmc.LatinHypercube(d, rng=rng).random(n)`; with a correlation the columns are reordered to the ranks of correlated van der Waerden scores (Iman and Conover), which keeps one point per stratum |
| `sobol(distributions, n, *, seed=None, second_order=False, ...)` | the design of Saltelli for Sobol indices, `SALib.sample.sobol.sample` on the unit cube (scrambled), `n` a power of two, `n (d + 2)` points (`n (2 d + 2)` with second order) |
| `fast(distributions, n, *, m=4, seed=None, ...)` | the design of the extended FAST, `SALib.sample.fast_sampler.sample`, `n d` points |
| `morris(distributions, trajectories, *, levels=4, seed=None, ...)` | the trajectories of Morris, `SALib.sample.morris.sample`; the levels of the unit grid are mapped to the centres of `levels` strata, so an unbounded distribution has finite values |
| `fit_parameters(fisher, n, *, seed=None, targets=None, id="fit")` | the multivariate normal of the Fisher covariance around the fitted values in the space of the `parameter_scale`, transformed back into the units of the model; a rank deficient covariance warns once and uses the pseudo-inverse, as `FisherInformation.covariance` does |
| `profile_parameters(identifiability, n, *, seed=None, targets=None, id="profile")` | every fitted parameter from its profile likelihood: the likelihood ratio `exp(-(cost - cost_min))` (the cost is `0.5 Σ r²`) on the points of the profile, interpolated linearly in the space of the `parameter_scale`, its inverse CDF at uniform draws, transformed back; a side which stays below the threshold reaches the bound of the parameter, so a parameter which is not identifiable spreads over the plausible range; the parameters are independent, the profiles carry no joint information |
| `fit_repeats(result, size, *, targets=None, id="repeats")` | the `size` best parameter sets of the repeats of a fit (`OptimizationResult.parameter_sets`) as points |
| `population(function, covariates, n, *, seed=None, method="random", id="population")` | the covariates (distributions) drawn by `random` or `lhs`, `function(covariates: dict[str, np.ndarray]) -> dict[str, np.ndarray]` (a function of a module) maps them to the values of targets; the dimension carries the targets and the covariates as coordinates |

The designs of a fit write the values of the fitted parameters to their targets (`FitParameter.target_id`, the `pid` by default); `targets` maps a `pid` to another target, and two parameters of one target (the versions of a parameter) raise, since one scan sets a target once per point.

Every design returns `Dimension(id, values={target: values}, labels=..., design=Design(...))`, the values as quantities where the unit is known. A grid stays a plain `Dimension(values=...)`.

### The record

`Design(method, distributions, options, references)`, frozen, of JSON types: the method (`local`, `random`, ...), the distributions as dictionaries (`to_dict`), the options (`n`, `seed`, `delta`, `m`, `levels`, `trajectories`, `second_order`, `correlation`, the confidence level of a profile, the ids of the problem and of the parameter set of a fit) and the resolved references (`{target: {"value": ..., "unit": ...}}`). `Dimension` gains the keyword `design: Design | None = None`, `Dimension.to_dict` writes it and `Scan.to_dict` carries it into `attrs["scan"]` of the result. `Dimension` also gains `coordinates=`: arrays along the dimension which the result carries as coordinates and which are never set on a model, the covariates of a population. The unit cube of a SALib design is not stored: it follows from the method, the options, the seed and the number of targets, so an analysis creates it again and checks it against the values of the result.

## Sensitivity

```python
from sbmlsim import sensitivity

res = Simulator().run(model, Scan(simulation, [sampling.sobol({...}, n=1024, seed=1), doses]), observables)
s = sensitivity.sobol(res)                     # SensitivityResult
s["pk.auc_inf_obs.ST"]                         # (parameter, dose)
s.index("ST")                                  # (parameter, observable, dose) of the scalar observables
sensitivity.plot_heatmap(s, "ST")
```

`local(result, *, dim=None, observables=None)`, `sobol(result, *, dim=None, observables=None, conf_level=0.95, num_resamples=100)`, `fast(...)` and `morris(...)` read the record of the dimension of their method (`dim=` chooses one of several, none raises), the kept observables of the result (or `observables=`), and compute the indices of every array element: for every label of the other dimensions and, for a timecourse on a grid (`steps`, `times` or `time=`), for every time point; a ragged timecourse raises and names `time=`.

- `local`: central differences at the reference, `raw = (y(+) - y(-)) / (2 delta p_ref)` in the unit of the observable per unit of the parameter, and `normalized = (y(+) - y(-)) / (2 delta y_ref)`, the dimensionless `∂ ln y / ∂ ln p`;
- `sobol`: `S1`, `ST` and with second order `S2`, each with its `_conf`, `SALib.analyze.sobol.analyze` per element;
- `fast`: `S1`, `ST` with `_conf`, `SALib.analyze.fast.analyze`;
- `morris`: `mu`, `mu_star`, `sigma`, `mu_star_conf`, `SALib.analyze.morris.analyze` on the unit cube which the record recreates.

An element whose points contain a failed simulation (`NaN`, the `status` of a flagged run) has `NaN` indices; one warning counts them. An element which is constant has `NaN` indices of variance (Sobol, FAST) and zero effects (local, Morris).

`SensitivityResult` wraps an `xarray.Dataset`: one variable per observable and index, `<observable>.<index>` (`auc.ST`, `auc.ST_conf`, `[S2].normalized`), over `(parameter, *other dimensions, [time])`, with `attrs["units"]` and the method, its options and the provenance of the scan in `attrs`. `index(name, observables=None)` stacks the scalar observables into `(parameter, observable, *other dimensions)`; `sel`/`isel`, `to_dataframe`, `to_netcdf`/`from_netcdf`, and `classify(name)`, the classification of today (`sensitivity/classification.py`) of an index. The plots are functions which return a figure and save it only with a path: `plot_heatmap(result, index, ...)` (clustered, with a cutoff), `plot_indices(result, observable)` (S1 and ST bars with their intervals) and `plot_morris(result, observable)` (`mu_star` against `sigma`).

## Uncertainty

```python
draws = sampling.profile_parameters(identifiability, n=500, seed=1)
res = Simulator().run(model, Scan(simulation, [draws, doses]), observables, time=grid)
bands = res.summary("profile", quantiles=[0.05, 0.5, 0.95])
uncertainty.plot_bands(bands, "conc")                                   # median and band per dose
uncertainty.plot_distribution(res, "pk.auc_inf_obs", dim="profile")     # per dose
```

`sbmlsim.sensitivity.uncertainty` has `plot_bands(summary, key, *, lower="q0.05", center="q0.5", upper="q0.95", ax=None)` (a line and a band per label of the other dimensions) and `plot_distribution(result, key, *, dim, kind="hist", ax=None)` (a histogram or a box per label). A ragged timecourse is interpolated onto the union of its time points by `summary`, or onto `time=` in the run.

## The callers

- `simulation/sensitivity.py` (`ModelSensitivity`) is removed; `sampling.local` and `sampling.random` with relative distributions replace the difference and the distribution scans (`examples/model_sensitivity.py`, `examples/demo/demo.py`, `docs/scans.md`, `tests/test_sensitivity.py`).
- `fit/sampling.py`: `create_samples(parameters, size, sampling, seed, min_bound)` keeps its signature and its values for a seed: a `FitParameter` with finite bounds is a `LogUniform` or a `Uniform` (by its scale and the `SamplingType`, a non-positive lower bound of a logarithmic one replaced by `min_bound`), one with an infinite bound is `Fixed(start_value)`, `START` tiles the start values; `random` or `lhs` draw. `SamplingType` stays (the settings, the command line and the PEtab extension store it).
- `sensitivity/`: the analysis classes, `SensitivitySimulation`, `SensitivityOutput`, `AnalysisGroup`, `SensitivityParameter`, `ParameterType` and `parameters.py` are removed; `classification.py` stays; `plots.py` becomes the plot functions of `SensitivityResult`.
- Examples: `examples/sensitivity/sensitivity_example.py` (the simple chain: `Formula` observables of the AUC, the maximum and the time of the maximum, the three conditions as a dimension, all four analyses), `examples/model_sensitivity.py` (the repressilator: local indices and uncertainty bands), `examples/demo/demo.py`, `examples/fit_sampling.py`, `examples/README.md`.
- Docs: `docs/sensitivity.md` rewritten, an uncertainty section, `docs/scans.md` ("Sensitivity scans"), the API pages, `CLAUDE.md`.
- pkdb_models uses `ModelSensitivity` and `sensitivity/` and is stale against the engine already; its migration is the issue the core cleanup opened.

## Errors

When a design is created or a scan is compiled: a relative distribution without `model=`; a target the model has not; a correlation which is not a symmetric, positive definite matrix of ones on the diagonal, or one for `sobol`, `fast` or `morris`; `sobol` with an `n` which is no power of two; two fitted parameters of one target; a profile without a converged optimum; a covariance which is rank deficient warns once. When an analysis runs: a result without a dimension of its method or with several and no `dim=`; values of the result which do not match the unit cube the record recreates; a ragged timecourse.

## Phases

1. **Sampling and uncertainty**: the distributions, the references, every design, the correlations, the record in `Dimension`, the scan and the result, the start values of a fit on the sampler, `plot_bands` and `plot_distribution`, the removal of `ModelSensitivity`, examples and docs. Verification: the tests of the sampler and of the uncertainty, the start values of a fit unchanged for a seed.
2. **Sensitivity**: `local`, `sobol`, `fast`, `morris`, `SensitivityResult`, the plots and the classification, the removal of the old `sensitivity/` package, examples and docs. Verification: the Ishigami function, the power-law model, the chain model end to end.

Each phase has its own plan and pull request.

## Testing

- Distributions: `ppf` against scipy, relative locations from references, quantities and units, truncation, `Empirical`, `Fixed`, the clipped probabilities 0 and 1.
- References: after the changes of the model and of the simulation and after initial assignments, in the units of the model.
- Designs: shapes, labels and records; the same seed gives the same values, `seed=None` records a seed which reproduces them; one point per stratum of an LHS, also with a correlation; correlated draws reach the asked rank correlation; `sobol`, `fast` and `morris` with uniform marginals equal the samples of SALib; the errors of the list above.
- Designs of a fit: the covariance of the draws of `fit_parameters` matches the Fisher covariance in the space of the scale; the draws of `profile_parameters` on a synthetic profile follow `exp(-Δcost)` (a Kolmogorov-Smirnov test) and reach the bound on a flat side; `fit_repeats` takes the best sets; versions of a parameter raise.
- `population`: the function applied to the drawn covariates, the covariates as coordinates.
- The record survives `to_netcdf`/`from_netcdf` of a result.
- Start values of a fit: equal to the values of `fit/sampling.py` before the change for every `SamplingType` and a seed.
- Uncertainty: the quantile bands of a lognormal parameter of a linear model against its analytic quantiles; the plots draw a band per label.
- Sensitivity (phase 2): the Ishigami function against its analytic Sobol indices and against SALib; local `normalized` of a power-law model equal to its exponents; the chain model end to end with a dimension of conditions, a timecourse per time point and a failed point; netCDF of a `SensitivityResult`.

## Out of scope

- Sensitivities of the integrator (forward or adjoint sensitivities of roadrunner) and the derivatives of observables; the local analysis uses finite differences of the scan.
- Bayesian inference (MCMC); `profile_parameters` and `fit_parameters` are approximations of the uncertainty of a fit.
- A distributed backend; the scans run in the pool of `sbmlsim.parallel`.
- The observables and the plots over scan dimensions of the simulation experiments (sub-project 4).
