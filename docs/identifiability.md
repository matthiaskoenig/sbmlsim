# Identifiability

The examples on this page continue the fit of the HCTZ model of [Parameter fitting](fitting.md): `op` is its `OptimizationProblem`, `settings` its `FitSettings`, `parameter_sets` the `ParameterSets` of the fit and `definition` its `FitDefinition`.

A fit always returns numbers. Whether the data determines those numbers is a separate question, and it is the one that decides whether a parameter may be interpreted, compared between conditions or extrapolated from. A parameter which the data does not determine takes whatever value the optimizer happened to stop at.

## What identifiability means

A parameter is **identifiable** when the data could not have been produced by a different value of it. Two things can go wrong, and they are usually distinguished (Raue et al. 2014, Wieland et al. 2021):

- **Structural non-identifiability** is a property of the model and its observables, not of the data. The parameter enters the observables only in combination with others, so no amount of data of that kind determines it: only a product, a ratio or a sum is determined. A model with a concentration `k * S` observed alone can never separate `k` from `S`. This is decided on the equations, with differential algebra or Lie derivatives (Bellman & Åström 1970, Chiş et al. 2011, Villaverde et al. 2016); `sbmlsim` does not analyse it, but a structural non-identifiability shows up in the analyses below as a perfectly flat direction.
- **Practical non-identifiability** is a property of the model *and the data at hand*. The parameter is determined in principle, but the data is too sparse, too noisy or measured in the wrong place to pin it down, so its confidence interval extends to infinity in one or both directions. More or better data fixes this; a different model is needed for the structural case.

Between the two extremes, most models of systems biology are **sloppy**: a few combinations of parameters are determined well and many are determined poorly, with the sensitivity spread over orders of magnitude (Gutenkunst et al. 2007, Transtrum et al. 2015). A sloppy model still predicts well along the directions the data constrains, which is why a poorly determined parameter is not by itself a reason to distrust a model - it is a reason not to interpret that parameter.

## The two analyses

`sbmlsim` implements the two analyses which work on a fitted model and its data:

| | [Profile likelihood](#profile-likelihood) | [Fisher information](#fisher-information) |
| --- | --- | --- |
| what it looks at | the cost along the parameter, re-optimizing the others | the curvature of the cost at the optimum |
| cost | a re-optimization per point, hundreds of simulations | one jacobian, `2k` simulations |
| intervals | asymmetric, invariant under a transformation of the parameters | symmetric in the scaled space, exact only for a quadratic cost |
| finds | structural and practical non-identifiability | structural non-identifiability, sloppiness, correlations |

The Fisher information is the cheap local answer and the profile likelihood the expensive exact one. They agree where the cost is a quadratic around the optimum and disagree where it is not, which is the normal case for a non-linear model: the profile is then the one to trust (Raue et al. 2009, Wieland et al. 2021).

Neither is a substitute for the other question worth asking, which is whether the *prediction* is determined even when the parameters are not (Simpson & Maclaren 2023).

## What to do about it

A parameter which comes back non-identifiable leaves four options: measure something else, fix the parameter to a literature value, reduce the model so that the parameter and the ones it is coupled to become one, or report it as undetermined and refrain from interpreting it. The paths of the other parameters along a profile say which parameters are coupled to it and therefore which reduction is the right one (Maiwald et al. 2016).

## Profile likelihood

A fit gives the parameters which describe the data best, the profile likelihood says how well the data determines every one of them. `sbmlsim.fit.identifiability` implements the method of Raue et al. 2009: a parameter is fixed at values around the optimum, all other parameters are optimized again for every value, and the resulting profile, the best cost as a function of the parameter, is compared with a threshold. The cost of a fit is the cost of `scipy.optimize.least_squares`, `cost = 0.5 * Σ r²` with the weighted residuals `r`. With residuals which are standardized by the errors of the data, `2 * cost` is the negative log-likelihood up to a constant, and the threshold of the likelihood ratio test on the cost is

```
cost_threshold = cost_min + chi2.ppf(alpha, df) / 2
```

with the confidence level `alpha` and `df = 1` for the pointwise confidence intervals of single parameters, i.e., `1.92` above the minimal cost at 95%; `df` equal to the number of parameters gives simultaneous intervals. The values at which the profile crosses the threshold are the bounds of the confidence interval of the parameter. These intervals are invariant under a transformation of the parameters and may be asymmetric, which is where the intervals of the [Fisher information](#fisher-information) fail for non-linear models (Wieland et al. 2021). With another weighting of the residuals the threshold is a heuristic on the same scale.

The shape of the profile classifies the parameter (`Identifiability`):

- `IDENTIFIABLE`: the profile crosses the threshold on both sides of the optimum, the confidence interval is finite,
- `NON_IDENTIFIABLE_LOWER`, `NON_IDENTIFIABLE_UPPER`, `NON_IDENTIFIABLE`: the profile has a minimum but stays below the threshold up to the lower bound, the upper bound or both bounds of the parameter, i.e., the parameter is practically non-identifiable, the data does not determine it towards small and/or large values,
- `STRUCTURAL`: the profile is flat over the scanned range, the parameter is compensated by the other parameters and the data carries no information about it.

The scans run in the space the fit searches, i.e. the scale of every parameter (`FitParameter.scale` or the `parameter_scale` of the settings), with adaptive steps: a step which raises the cost by more than `max_cost_fraction` of the distance to the threshold is reduced and repeated, a step which raises it by little is enlarged, so the profile is resolved where it changes. The other parameters start from the previous point of the profile and their paths are stored, so a parameter which is coupled to the scanned one is seen in its path (Maiwald et al. 2016). A scan stops when the profile crosses the threshold, at the bound of the parameter or after `max_points`. A scan which finds a lower cost than the parameter set reports that the fit did not converge, and the threshold is taken relative to the lowest cost of all profiles.

```py
from sbmlsim.fit.identifiability import ProfileSettings, profile_likelihood

result = profile_likelihood(
    problem=op,
    settings=settings,
    parameter_set=parameter_sets[0],
    profile_settings=ProfileSettings(alpha=0.95, max_points=30),
    n_cores=4,
)
print(result.summary_df())
result.to_json(Path("identifiability.json"))
```

`ProfileSettings` holds the confidence level, the degrees of freedom, the steps in decades of the parameter and `reoptimize`: without the re-optimization the other parameters stay at the optimum, which is a plain scan of the cost and a lower bound of the profile, fast and sufficient to find the non-identifiable parameters, but with intervals which are too narrow for coupled parameters. The scans are independent and run in the worker pool of the fit runner, two per parameter, so a parallel analysis needs the `if __name__ == "__main__":` guard like a parallel fit. `pids` restricts the analysis to some of the parameters.

The `IdentifiabilityResult` carries a `ParameterProfile` per parameter with the values, the costs, the paths of all parameters and the confidence interval, `summary_df()` is the table of the parameters with their intervals and classification, `report()` the text and `to_json`/`from_json` the storage. `plot_profiles` draws the overview of all profiles, `plot_profile` the profile of one parameter with the paths of the other parameters along it; the axis of a parameter is its scale, and the path of a parameter is `log10(parameter / optimum)` on a logarithmic scale and `parameter - optimum` on the linear one.

A report shows the analysis: `FitReport(..., identifiability=result)` adds the section **Identifiability** with the table, the overview and one figure per parameter, and writes `identifiability.json` and `identifiability.tsv`. `FitReport(..., fisher=fim)` adds the Fisher information to the same section, i.e. its table of errors and intervals, the eigenvalues and the correlation of the parameters, and writes `fisher.json` and `fisher.tsv`; a report of an information which is rank deficient says that its errors cannot be read. Every metric and every figure of a report carries a `?` which explains what the value is and what to look for in the figure. `FitRun.identifiability()` computes the profiles of the best parameter set of a finished fit, so a global optimization followed by the identifiability of its result is

```py
runs = run_fit(
    definition,
    algorithm=OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION,
    size=2,
    n_cores=4,
)
run = runs[opid]
identifiability = run.identifiability(n_cores=4)
fisher = run.fisher()
run.report(
    output_dir=Path("results"),
    identifiability=identifiability,
    fisher=fisher,
)
```

which is `examples/hctz_fitting/fitting/identifiability.py`. `identifiability_cli` is the command line tool for stored parameters, `examples/hctz_fitting/fitting/identifiability_report.py` on the HCTZ definitions; it computes both analyses, `--no-fisher` reports the profiles alone:

```bash
python -m examples.hctz_fitting.fitting.identifiability --subset=PK --runs=2 --cores=4
python -m examples.hctz_fitting.fitting.identifiability_report results/fit/PK/parameters.json --cores=4
```

## Fisher information

The profile likelihood follows the cost and costs a re-optimization per point. The Fisher information is the local alternative: the curvature of the cost at the optimum, from one jacobian. For a fit which minimizes `cost = 0.5 * Σ r²` with the weighted residuals `r`, the Gauss-Newton approximation of the Hessian is the Fisher information matrix, and its inverse, scaled by the variance of the residuals, is the covariance of the parameters:

```
FIM = J' J,   cov = σ² (J' J)⁻¹,   σ² = 2 cost / (n - k)
```

with the jacobian `J = ∂r/∂θ` in the space the optimizer searches, see `FitSettings.parameter_scale`:

```py
from sbmlsim.fit.fisher import fisher_information

fim = fisher_information(problem=op, settings=settings, parameter_set=parameter_sets[0])
print(fim.summary_df)
print(fim.correlation)
```

`summary_df` is the table of the parameters with their standard error, the coefficient of variation and the confidence interval `θ ± t · SE`, computed in the scaled space and transformed back, so on a logarithmic scale the interval is not symmetric around the value; the table has the column `scale` when the parameters are searched in different spaces. `correlation` is the correlation of the parameters: a pair at `±1` is a pair the data only determines together, i.e. the fit trades one against the other.

The eigenvalues say what the data constrains. A direction with a small eigenvalue is a combination of parameters the data does not determine, `rank` counts the ones it does, and `is_identifiable` is whether that is all of them. `condition_number`, the ratio of the largest to the smallest eigenvalue, is how sloppy the problem is.

A parameter which no data informs is exactly zero in the jacobian and gives a zero eigenvalue, i.e. the analysis finds a structural non-identifiability without scanning: the intravenous problem of the HCTZ example determines the renal excretion and carries no information about the absorption of an oral dose, so its Fisher information has rank 1 of 3.

The two analyses answer different questions and disagree where the cost is not a quadratic. The Fisher information is a local statement about the optimum, the profile likelihood follows the cost until it rises by the threshold, so the profile is the one to trust for a non-linear model and the Fisher information is the one to compute when a profile is too expensive. A parameter which the Fisher information calls determined and the profile calls non-identifiable has a cost which is curved at the optimum and flat away from it, which is what the profile is for; in the `Perelson_Science1996` example of the benchmark collection the loss rate of the infected cells is such a parameter.

The publications behind the methods are listed under [References](references.md#parameter-fitting).
