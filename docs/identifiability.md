# Identifiability

A fit always returns numbers. Whether your data actually determines those numbers is a separate question, and it decides whether you may interpret a fitted parameter, compare it between conditions or extrapolate with it. A parameter the data does not determine takes whatever value the optimizer happened to stop at.

`sbmlsim` has two analyses for this. Both start from a fitted parameter set:

| | [Profile likelihood](#profile-likelihood) | [Fisher information](#fisher-information) |
| --- | --- | --- |
| what it looks at | the cost along one parameter, with the others re-optimized | the curvature of the cost at the optimum |
| cost | one optimization per point, hundreds of simulations | one jacobian, `2k` simulations for `k` parameters |
| confidence intervals | can be asymmetric, unchanged by a transformation of the parameters | symmetric in the space the fit searches, exact only for a quadratic cost |
| finds | structural and practical non-identifiability | structural non-identifiability, sloppiness, correlated parameters |

The Fisher information is the fast local answer, the profile likelihood the slower and more reliable one. Where the two disagree, trust the profile.

## Quick start

The HCTZ example fits the model globally and then runs both analyses around the best parameter set:

```bash
python -m examples.hctz_fitting.fitting.identifiability --subset=PK --runs=2 --cores=4
```

It writes one report with the fit and a section **Identifiability** into `results/identifiability`. If you already have a fit, analyse its stored parameters without fitting again:

```bash
python -m examples.hctz_fitting.fitting.identifiability_report results/fit/<fit id>/parameters.json --cores=4
```

`<fit id>` is the directory a fit wrote its report into, e.g. `PK_20260908_144538__ea1ff`. The command reads the `optimization_result.json` next to the parameters, so the analyses run with the settings of the fit and the report shows how the fit was done. `--no-fisher` leaves out the Fisher information, and `--parameter` (repeatable) profiles only the named parameters. The tool is `identifiability_cli` of `sbmlsim.fit.cli`, which you can reuse for your own fit definitions like `fit_cli`.

In python, a `FitRun` returned by `run_fit` computes both analyses for its best parameter set and adds them to its report:

```py
from pathlib import Path

from sbmlsim.fit.cli import run_fit
from sbmlsim.fit.options import OptimizationAlgorithmType

runs = run_fit(
    definition,
    opid="PK",
    algorithm=OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION,
    size=2,
    n_cores=4,
)
run = runs["PK"]
identifiability = run.identifiability(n_cores=4)
fisher = run.fisher()
run.report(output_dir=Path("results"), identifiability=identifiability, fisher=fisher)
```

`definition` is the `FitDefinition` of [Parameter fitting](fitting.md). The analyses run in worker processes like a parallel fit, so a script needs its code under `if __name__ == "__main__":`.

## What you get

The report section **Identifiability** has a table of the parameters with their confidence intervals and classification, an overview of all profiles and one figure per parameter. With the Fisher information it also shows the standard errors and intervals, the eigenvalues and the correlation of the parameters. If the Fisher information is rank deficient, the report says that its errors cannot be read. Every value and figure in the report has a `?` that explains what to look for. The report directory also gets `identifiability.json`, `identifiability.tsv`, `fisher.json` and `fisher.tsv`.

Each parameter is classified from the shape of its profile (`Identifiability`):

| class | profile | meaning |
| --- | --- | --- |
| `IDENTIFIABLE` | crosses the threshold on both sides of the optimum | the confidence interval is finite |
| `NON_IDENTIFIABLE_LOWER`, `NON_IDENTIFIABLE_UPPER`, `NON_IDENTIFIABLE` | has a minimum, but stays below the threshold up to the lower bound, the upper bound or both | practically non-identifiable: the data does not determine the parameter towards small or large values |
| `STRUCTURAL` | flat over the scanned range | other parameters compensate for it, the data carries no information about it |

If a profile finds a lower cost than the fitted parameter set, the fit did not converge, and the analysis says so. The threshold is then taken relative to the lowest cost found.

## What to do about a non-identifiable parameter

You have four options: measure something else, fix the parameter to a literature value, reduce the model so that the parameter and the ones coupled to it become one, or report the parameter as undetermined and do not interpret it. The paths of the other parameters along a profile show which parameters are coupled to it, and so which reduction fits (Maiwald et al. 2016).

A poorly determined parameter is not by itself a reason to distrust a model. The question that often matters more is whether the *prediction* you care about is determined, even when some parameters are not (Simpson & Maclaren 2023).

## Profile likelihood

The profile likelihood (Raue et al. 2009) fixes one parameter at a series of values around the optimum and optimizes all other parameters again at every value. The best cost as a function of the fixed parameter is its profile, and it is compared with a threshold:

```
cost_threshold = cost_min + chi2.ppf(alpha, df) / 2
```

The cost of a fit is the cost of `scipy.optimize.least_squares`, `cost = 0.5 * Σ r²` with the weighted residuals `r`. When the residuals are divided by the errors of the data, `2 * cost` is the negative log-likelihood up to a constant, and this is the threshold of the likelihood ratio test. With the confidence level `alpha = 0.95` and `df = 1` (pointwise intervals of single parameters) it is `1.92` above the minimal cost; `df` equal to the number of parameters gives simultaneous intervals. Where the profile crosses the threshold are the bounds of the confidence interval. With another weighting of the residuals the threshold is a heuristic on the same scale.

```py
from pathlib import Path

from sbmlsim.fit.identifiability import (
    ProfileSettings,
    plot_profile,
    plot_profiles,
    profile_likelihood,
)

result = profile_likelihood(
    problem=problem,
    settings=settings,
    parameter_set=parameter_sets[0],
    profile_settings=ProfileSettings(alpha=0.95, max_points=30),
    n_cores=4,
)
print(result.summary_df())
result.to_json(Path("identifiability.json"))
fig = plot_profiles(result)
fig.savefig("profiles.png")
```

`problem`, `settings` and `parameter_sets` are the `OptimizationProblem`, the `FitSettings` and the fitted `ParameterSets` of [Parameter fitting](fitting.md).

The result is an `IdentifiabilityResult`: `summary_df()` is the table of the parameters with their intervals and classification, `report()` the same as text, and `to_json`/`from_json` store and load it. It holds a `ParameterProfile` per parameter with the values, the costs, the paths of all other parameters and the confidence interval. `plot_profiles` draws all profiles, `plot_profile(result, pid)` one profile with the paths of the other parameters along it. Both return a `matplotlib.figure.Figure` and save it when you pass `path=`. The figure is not attached to pyplot, so save it with `fig.savefig(...)` rather than calling `plt.show()`.

How the scans work:

- The scans run in the space the fit searches, i.e. the scale of every parameter (`FitParameter.scale` or `FitSettings.parameter_scale`), and the steps of `ProfileSettings` are in decades of the parameter.
- The step size adapts. A step that raises the cost by more than `max_cost_fraction` of the distance to the threshold is made smaller and repeated, a step that raises it only a little is made larger. This resolves the profile where it changes.
- Each point starts the optimization of the other parameters from the previous point, and their paths are stored.
- A scan stops when the profile crosses the threshold, at the bound of the parameter, or after `max_points` points.
- Each parameter needs two scans, one in each direction. They are independent and run in parallel with `n_cores` workers. An analysis of fewer scans than the jacobian of a re-optimization is wide, e.g. a profile of a single parameter with `pids`, runs its scans one after another instead and evaluates the points of the jacobian of every re-optimization on all workers (`split_cores` of `sbmlsim.fit.runner`), so a profile of one parameter uses every core as a fit does. A profile of sbmlsim-core evaluates the points of such a jacobian as one batch on the threads of one worker, see the `n_cores` of [Running the optimization](fitting.md#running-the-optimization).
- The analysis of a problem of sbmlsim-core walks its profiles in the engine itself (`engine_profiles`): the two sides of every profile run next to each other on the threads of one batch, the optimizations of their points and the evaluations of those run in rust, and the analysis reaches python once, as an answer. The walk is the one of this page, so the settings are the same and the profiles are the ones of the scans in python: around the optimum of a single start fit of the HCTZ example both paths walk the same points of every parameter (12, 12 and 15 on its iv data, 59, 68 and 32 on its PK data), bound them at the same values and classify every one of them the same way. The iv profiles take 0.15 s in the engine against 0.42 s of the scans on twenty cores; the PK profiles take 3.49 s against 2.94 s, the one place where the scans in python are ahead: the points are the same points, and the optimizer of a point takes about twice the evaluations of the objective in the engine, which is the trust region of basin accepting fewer of its trials. `n_cores` are the threads of the batch then, and the analysis has no workers, no progress of its scans and no profile which is reported before the others. What the engine has not keeps an analysis in python, and the gaps are logged so an analysis says why: the gaps of a fit (see the `engine_fit` of [Running the optimization](fitting.md#running-the-optimization)), a scan which holds the other parameters at the optimum (`reoptimize=False`) and an argument of the optimizer of a point which is no setting of the engine. `engine_profiles=True` requires the engine and names the gaps of an analysis it cannot walk, `engine_profiles=False` scans in python, which evaluates the points through the engine.
- `pids` restricts the analysis to some of the parameters.
- `ProfileSettings(reoptimize=False)` keeps the other parameters at the optimum. That is a plain scan of the cost and a lower bound of the profile: fast and enough to find non-identifiable parameters, but its intervals are too narrow for coupled parameters.

A long analysis says how far it is: every scan is a step of the activity of the analysis, with the count, the elapsed time, the estimate of the time which is left and the cores, the CPU and the memory of the run, and a scan which runs in this process reports its points as well, see [Progress](run.md#progress-and-provenance).

## Fisher information

The Fisher information is the local alternative to the profile: the curvature of the cost at the optimum, computed from one jacobian. For a fit that minimizes `cost = 0.5 * Σ r²`, the Gauss-Newton approximation of the Hessian is the Fisher information matrix, and its inverse, scaled by the variance of the residuals, is the covariance of the parameters:

```
FIM = J' J,   cov = σ² (J' J)⁻¹,   σ² = 2 cost / (n - k)
```

A fit of the `likelihood` or the `posterior` has its residuals in units of the noise of the data, so its covariance is `(J' J)⁻¹` without `σ²` and its intervals are the ones of the normal distribution; the Fisher information is the one of the data, the priors of a fit of the posterior are no part of it (the [Laplace approximation of the posterior](uncertainty.md#the-posterior-of-a-fit) adds them).

Here `J = ∂r/∂θ` is the jacobian in the space the optimizer searches (`FitSettings.parameter_scale`), `n` the number of data points and `k` the number of parameters. The jacobian comes from central differences with the relative step `step`, by default the cube root of `relative_tolerance` of the settings, `1e-2` for the default `1e-6`. The simulations are exact only to that tolerance, and a much smaller step divides this noise by a tiny number: on the HCTZ fit a step of `1e-6` gave standard errors up to 30 times too small.

```py
from sbmlsim.fit.fisher import fisher_information

fim = fisher_information(
    problem=problem, settings=settings, parameter_set=parameter_sets[0], n_cores=4
)
print(fim.summary_df)
print(fim.correlation)
print(fim.rank, fim.is_identifiable, fim.condition_number)
```

### Where the jacobian comes from

A problem of sbmlsim-core whose whole objective is a request of a fit, the problems the engine [fits itself](fitting.md#running-the-optimization), takes the forward sensitivities of its conditions as its jacobian: the derivatives of the residuals, integrated next to the states, one simulation per condition instead of two per parameter and condition. Every other problem is differentiated in python, by central differences of a step which balances the noise of the integrator against the truncation of the difference (`default_step`: `1e-2` of the parameter for the default tolerance `1e-6`, the cube root of the noise). `fisher_information` reports which of the two a matrix came from in `jacobian_of`, and the `fisher.json` of a record keeps it.

The sensitivities are not only cheaper, they are a different kind of answer: a difference quotient divides the noise of the integrator by its step, so its accuracy stops at the tolerance of the simulation however the step is chosen. At the default tolerance of `1e-6` the central differences of the iv fit of the HCTZ example are 1.5 % off the sensitivities and the single small entries of the jacobian of its PK fit are off by 100 %, where the sensitivities of that tolerance give the information of a tolerance of `1e-10` to a thousandth. The standard errors, the correlations and the rank of a sloppy model are read off the small entries of that matrix, which is why the information takes the sensitivities where the engine has them.

`engine_jacobian=False` keeps the central differences, `True` requires the sensitivities and names what keeps a problem from them (another engine, an objective of its own, a condition which carries no sensitivities: an event of the model, a change of the request or a steady state), and a `step=` which is given asks for the differences of that step. The finite differences of the engine itself are never taken: their step is the fixed `1e-3` of an optimizer and does not follow the tolerance of the integrator.

The jacobian of the differences is `2 k` evaluations of the residuals, two per parameter, which are independent of each other: `n_cores` evaluates them on the workers of a fit, the pool the jacobian of least squares uses, so a Fisher information of 35 parameters costs five rounds of simulations on 16 workers instead of seventy one after another. The step of a point is the same either way, so the jacobian is the one of a single process bitwise. The points of sbmlsim-core are one batch on the threads of one worker, as the ones of a fit are, and the jacobian of the sensitivities is one batch of all of its conditions. The commands pass their `--cores`, and the points report their [progress](run.md#progress-and-provenance).

What the result tells you:

- `summary_df` lists each parameter with its standard error, coefficient of variation and confidence interval `θ ± t · SE`. The interval is computed in the scaled space and transformed back, so on a logarithmic scale it is not symmetric around the value. The column `scale` appears when the parameters are searched in different spaces.
- `correlation` is the correlation of the parameters. A pair near `±1` is determined only together: the fit trades one against the other.
- `eigenvalues` say what the data constrains. A small eigenvalue is a combination of parameters the data does not determine. `rank` counts the directions the data does determine, and `is_identifiable` is true when that is all of them.
- `condition_number`, the ratio of the largest to the smallest eigenvalue, measures how sloppy the problem is.

A parameter that no data informs has a zero column in the jacobian and gives a zero eigenvalue, so the Fisher information finds structural non-identifiability without any scan. For example, intravenous data of HCTZ determines the renal excretion but carries no information about the absorption of an oral dose, so a fit of the absorption on that data alone has a rank deficient Fisher information.

The two analyses disagree where the cost is not quadratic around the optimum, which is common for non-linear models. A parameter that the Fisher information calls determined and the profile calls non-identifiable has a cost that is curved at the optimum and flat further away. The loss rate `delta` of the infected cells in the `Perelson_Science1996` example of the [benchmark collection](petab_benchmark.md#fitting-a-problem-of-the-collection) is such a parameter.

## Background

A parameter is identifiable when no other value of it could have produced the data. Two kinds of failure are usually distinguished (Raue et al. 2014, Wieland et al. 2021):

- **Structural non-identifiability** is a property of the model and what is observed, not of the data. The parameter enters the observables only in combination with others, so no amount of data of that kind determines it, only a product, ratio or sum. A concentration `k * S` observed alone can never separate `k` from `S`. This is decided on the equations, with differential algebra or Lie derivatives (Bellman & Åström 1970, Chiş et al. 2011, Villaverde et al. 2016). `sbmlsim` does not do this analysis, but a structural non-identifiability shows up in both analyses above as a perfectly flat direction.
- **Practical non-identifiability** is a property of the model *and the data at hand*. The parameter is determined in principle, but the data is too sparse, too noisy or measured at the wrong times, so its confidence interval extends to infinity in one or both directions. More or better data fixes this. A structural non-identifiability needs a different model or different measurements.

Between the two, most models in systems biology are **sloppy**: a few combinations of parameters are well determined and many are poorly determined, with sensitivities spread over orders of magnitude (Gutenkunst et al. 2007, Transtrum et al. 2015). A sloppy model still predicts well along the directions the data constrains. The profile likelihood gives reliable intervals where the cost is not quadratic, which is where the intervals of the Fisher information fail (Raue et al. 2009, Wieland et al. 2021).

The publications are listed under [References](references.md#parameter-fitting).
