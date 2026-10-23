# Uncertainty

A prediction depends on inputs which are not known exactly: a dose and a body weight reported as mean ± SD, a parameter from the literature with a CV, a clearance with an interval. `sbmlsim.uncertainty` declares these values as uncertain where they are used, runs them through the simulations and answers questions about the predictions with their Monte Carlo error.

## Uncertain values

An `Uncertain` is a value with its distribution. It is given wherever a value is given, in a `Change`, in the pre-initialization changes of a `Simulation` or as the prior of a `FitParameter`:

```python
from sbmlsim import Q
from sbmlsim.uncertainty import Uncertain

dose = Uncertain(Q(5, "mg"), cv=0.05)  # log-normal, mean 5 mg, CV 5 %
weight = Uncertain(Q(75, "kg"), sd=Q(12, "kg"), low=Q(45, "kg"), kind="variability")
fraction = Uncertain.range(0.6, 0.9)  # uniform
clearance = Uncertain.interval(
    Q(12, "l/hr"), Q(30, "l/hr"), level=0.9
)  # a reported 90 % interval
auc = Uncertain.se(Q(40, "mg*hr/l"), Q(2, "mg*hr/l"), n=16)  # the SD is SE * sqrt(n)
print(dose.shape, dose.mean(), dose.sd(), weight.mean())
```

| given | distribution | nominal |
|---|---|---|
| `sd=` or `cv=` of a positive value | log-normal with that mean and SD | the value |
| `sd=`/`cv=` with `shape="normal"`, or a value which may be negative | normal | the value |
| `low=`/`high=` with them | the distribution above, truncated | the value |
| `Uncertain.range(low, high)` | uniform (`log=True`: log-uniform) | the (geometric) midpoint |
| `Uncertain.interval(low, high, level)` | log-normal (positive) or normal with these quantiles | the median |
| `Uncertain.se(mean, se, n)` | the SD `se * sqrt(n)` | the mean |
| `distribution=` | a distribution of `sbmlsim.simulation.sampling` | the value |

A positive quantity given with an SD or a CV, e.g. a dose, a body weight, a volume or a rate, is log-normal by default, whose median is `mean / sqrt(1 + cv²)` and whose logarithm has the standard deviation `sqrt(ln(1 + cv²))` (Limpert et al. 2001): a normal distribution of a large CV gives negative values. `shape="normal"` asks for the normal. `kind` is `uncertainty` (what is not known about a value) or `variability` (a value which differs between subjects). Two uses of one object, or two objects of one `name`, are one random variable.

A simulation simulates the nominal values, so declaring a value uncertain changes nothing until it is propagated:

```python
from sbmlsim.resources import MIDAZOLAM_SBML
from sbmlsim.simulation import PK, Change, Simulation
from sbmlsim.simulator import Simulator


def simulation(activity: float = 1.0) -> Simulation:
    return Simulation(
        end=24,
        time_unit="hr",
        steps=96,
        preinit_changes={"BW": weight, "LI__MIDOH_f": activity},
        changes=[Change(0, {"PODOSE_mid": dose})],
    )


control = simulation()
print([use.target for use in control.uncertain_values()])
nominal = Simulator().simulate(MIDAZOLAM_SBML, control)
```

## Covariances

A `Covariance` is the dependence between uncertain values: a covariance matrix (in the units of the targets, on the linear or the log scale), a correlation matrix or the coefficients of pairs. Every matrix is validated (symmetric, positive semi-definite, a unit diagonal of a correlation); `nearest=True` replaces a correlation which is not positive semi-definite by the nearest correlation matrix (Higham 2002).

```python
from sbmlsim.uncertainty import Covariance

correlation = Covariance.correlation({("BW", "LI__MIDOH_Vmax"): 0.4})
omega = Covariance(["CL", "V"], matrix=[[0.09, 0.03], [0.03, 0.04]], scale="log")
print(correlation.matrix, omega.correlation_matrix())
```

The draws are correlated through their normal scores (a Gaussian copula): for normal values this is the multivariate normal of the covariance, for log-normal values the multivariate normal of their logarithms, and a covariance of log-normal values on the linear scale is converted exactly. The variances of a covariance must be the variances of the marginals.

## Propagation

`propagate` runs what an `UncertaintyModel` declares uncertain through one simulation or several scenarios: the uncertain values of the simulations and the parameters with their distributions. The draws are one dimension of a scan, so the model is compiled once and the simulations run on the scan core (in the pool for many draws); they are quasi-random (scrambled Sobol points), correlated by the covariances. Several scenarios get the same draws (common random numbers), so a ratio between them is paired; an uncertain value of one target whose distribution differs between the scenarios, e.g. a dose of 100 mg in one and of 200 mg in another, is drawn at the same quantile of each. A parameter is drawn in every scenario, so a scenario must not set it: a draw would replace the value of the scenario, which raises; a scenario sets another target instead, as the inhibitor below sets the activity `LI__MIDOH_f` of the uncertain `LI__MIDOH_Vmax`. Without `n` the number of draws doubles until the intervals of the mean, the SD and the 5 % and 95 % quantiles of the outputs are narrower than `rel_tol` (0.1) times their SD, after the adaptive procedure of GUM Supplement 1 (JCGM 101:2008). A Latin hypercube (`design="lhs"`) has no continuation and needs `n`; Sobol and random points continue their sequence, so a result of more draws starts with the draws of fewer.

```python
from sbmlsim.uncertainty import UncertaintyModel, propagate

uq = UncertaintyModel(
    parameters={"LI__MIDOH_Vmax": Uncertain(0.005, cv=0.3)},
    covariance=[correlation],
)
res = propagate(
    MIDAZOLAM_SBML,
    {"control": simulation(), "inhibitor": simulation(activity=0.3)},
    uq,
    [PK("pk", "[Cve_mid]", dose="PODOSE_mid", route="oral")],
    outputs=["pk.auc_last", "pk.cmax"],
    n=512,
    seed=1234,
)
print(
    res.summary(["pk.auc_last"])[
        ["scenario", "mean", "mean_se", "sd", "q0.05", "q0.95"]
    ]
)
print(res.contributions("pk.auc_last", scenario="control"))
```

`summary` gives every output which is one value per draw with its Monte Carlo errors (a timecourse raises, it is reduced by an observable, e.g. `at`, `max` or a `PK` parameter): the standard errors of the mean and of the SD and the distribution-free intervals of the quantiles (Hahn & Meeker 1991). `contributions` ranks the uncertain values by their first-order indices, estimated from the same draws (Plischke et al. 2013); the index of a correlated value includes the effect of the values it is correlated with. `values(output, scenario)` gives the draws, `contrast(output, a, b)` the paired ratio or difference of two scenarios, and `to_netcdf`/`from_netcdf` store a result with its record (the seed, the design, the number of draws, the convergence history, the variables and the covariances).

## Probability statements

A question about a prediction is a probability over the draws: `P(H)`, the probability that an output satisfies a hypothesis H given everything that is uncertain, estimated by the fraction of the draws which satisfy H. `P(H)` is a property of the model and of what is uncertain, not of the number of draws: more draws can always be simulated, and they only estimate it more precisely. The statements are built for this, they are anytime valid: their interval is a confidence sequence (Robbins 1970; Lai 1976; Howard et al. 2021), which contains `P(H)` at every number of draws at once with probability 95 %, so a statement may be looked at after every batch and continued with more draws. A confidence interval of a fixed number of draws, e.g. the Wilson interval, misses `P(H)` far more often when it is looked at repeatedly.

A decision against a threshold `at_least=p0` is one of `accepted` (`P(H) >= p0` is shown), `rejected` (`P(H) < p0` is shown), `equivalent` (`P(H)` is shown to be within the `margin` of the threshold, two one-sided tests, Schuirmann 1987) and `undecided`, with the number of draws which would decide it (`n_needed`). More draws only make a statement more precise: an undecided one becomes decided, an equivalent one becomes the direction which is shown later, and a direction once shown is kept, so a decision is never reversed and is wrong with probability at most 5 %. The margin is a difference of probability which does not matter, by default a tenth of the distance of the threshold to 0 or 1 (0.01 for 0.9, 0.001 for 0.99); it makes a statement whose probability is the threshold decidable.

```python
from sbmlsim.uncertainty import Hypothesis

hypotheses = [
    Hypothesis(
        "pk.auc_last", contrast=("inhibitor", "control"), greater=1.25, at_least=0.9
    ),
    Hypothesis("pk.auc_last", contrast=("inhibitor", "control"), greater=3.0),
    Hypothesis("pk.cmax", scenario="control", less=Q(0.1, "mmol/l"), at_least=0.95),
]
report = res.test(hypotheses)
print(report.table()[["hypothesis", "estimate", "low", "high", "decision"]])
ratio = res.contrast("pk.auc_last", "inhibitor", "control")
print(res.probability(ratio, greater=2.0, at_least=0.5).decision)
```

A `TestReport` adjusts several decisions for their multiplicity, by Holm (1979, the family-wise error, by default) or by Benjamini and Yekutieli (2001, `adjust="by"`, the false discovery rate), on the anytime-valid p-values of the decisions; both are valid for any dependence between the statements, which share their draws. A decision whose adjusted p-value exceeds 5 % becomes `undecided` with the number of draws which decide it. `propagate(..., hypotheses=hypotheses)` draws until every hypothesis is decided after the adjustment (or `n_max`), which the anytime validity allows. The draws which failed (`NaN`) are excluded and counted, infinite values, e.g. a ratio over a zero, are values. The validity assumes independent draws, exact for `design="random"`; the counts of the quasi-random Sobol points fluctuate less, for which the statements are conservative in practice. A probability statement is about the predictive distribution, i.e. about what is uncertain, not a test of a study: the draws are not subjects.

`examples/uncertainty_midazolam.py` is the analysis of this page with the adaptive number of draws, which draws until its hypotheses are decided.

## References

- JCGM 101:2008. Evaluation of measurement data, Supplement 1 to the GUM: propagation of distributions using a Monte Carlo method.
- Limpert E, Stahel WA, Abbt M (2001). Log-normal distributions across the sciences: keys and clues. BioScience 51:341-352.
- Owen AB (1998). Scrambling Sobol' and Niederreiter-Xing points. J Complexity 14:466-489. Joe S, Kuo FY (2008). Constructing Sobol sequences with better two-dimensional projections. SIAM J Sci Comput 30:2635-2654.
- Nelsen RB (2006). An Introduction to Copulas, 2nd ed. Springer. Higham NJ (2002). Computing the nearest correlation matrix: a problem from finance. IMA J Numer Anal 22:329-343.
- Glasserman P (2003). Monte Carlo Methods in Financial Engineering. Springer.
- Hahn GJ, Meeker WQ (1991). Statistical Intervals: A Guide for Practitioners. Wiley.
- Plischke E, Borgonovo E, Smith CL (2013). Global sensitivity measures from given data. Eur J Oper Res 226:536-550.
- Robbins H (1970). Statistical methods related to the law of the iterated logarithm. Ann Math Stat 41:1397-1409. Lai TL (1976). On confidence sequences. Ann Stat 4:265-280. Howard SR, Ramdas A, McAuliffe J, Sekhon J (2021). Time-uniform, nonparametric, nonasymptotic confidence sequences. Ann Stat 49:1055-1080.
- Schuirmann DJ (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. J Pharmacokinet Biopharm 15:657-680.
- Gelman A, Carlin JB, Stern HS, Dunson DB, Vehtari A, Rubin DB (2013). Bayesian Data Analysis, 3rd ed. CRC Press.
- Holm S (1979). A simple sequentially rejective multiple test procedure. Scand J Stat 6:65-70. Benjamini Y, Yekutieli D (2001). The control of the false discovery rate in multiple testing under dependency. Ann Stat 29:1165-1188.
