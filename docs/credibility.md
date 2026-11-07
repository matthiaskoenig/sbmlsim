# Credibility

The fit metrics of a model pool its validation data: one GMFE and one percentage within 2-fold for doses between the fitted doses, held-out subjects and processes the model is known not to describe. A number like that answers no question. What the model is credible for is not written down anywhere. `sbmlsim.credibility` is how a model declares what it is used for, which evidence supports each use and when the use is accepted, and how the kind `credibility` of the [run layer](run.md) assesses the declaration against the adopted fits of the model. A domain, e.g. pharmacokinetics, adds its own evidence with extensions (`sbmlsim.credibility.extensions`).

## The frame

The frame is that of ASME V&V 40 (2018), which the FDA guidance on the credibility of computational modeling (2023) and ICH M15 on model-informed drug development (Step 4, 2026) took over, and which the EMA guideline on the reporting of PBPK modelling and simulation (2018) uses for the qualification of a platform. For PBPK models it is worked out by Kuemmel et al. (2020) and Musuamba et al. (2021), and Erdemir et al. (2020) give the rules of credible practice.

- The **question of interest** is what a decision needs to know, e.g. "Does the model predict the plasma concentration of the drug in renal impairment?".
- The **context of use** (COU) states the role of the model in answering the question: its quantities of interest and the domain it is claimed for.
- The **model risk** follows from the influence of the model on the decision (is it one argument among others or the only one) and the consequence of a wrong decision.
- The **evidence** is the data the model is compared with, each part with a role: calibration (the data the model was fitted to), validation (data held out of the fit) or limitation (data the model is known not to describe).
- The **assessment** decides per COU whether the validation evidence meets the acceptance, which part of the claimed domain the evidence covers, and which credibility factors of V&V 40 are assessed.

Credibility is per context of use. There is no score of a model.

## The ladder of contexts of use

A COU has one level, `CouLevel`. A model typically has two to five COUs.

| level | name | meaning | typical evidence |
| --- | --- | --- | --- |
| `C1` | describe | the model describes the data it integrates | calibration evidence |
| `C2` | interpolate | conditions inside the range of the data | held-out conditions, subjects, studies |
| `C3` | extrapolate with data | conditions for which some data exists | validation data of the condition |
| `C4` | extrapolate without data | conditions without data | the qualification of the platform across models, uncertainty |
| `C5` | clinical endpoint | downstream outcomes of the model | validation data of the endpoint |

`CouLevel.label` gives the name of a level (`describe`, `interpolate`, ...). C1 is the only level decided on calibration evidence, since its question is whether the model describes the data it was fitted to. Every other level is decided on validation evidence only.

## Model risk, the default acceptance and the minimum of points

`Influence` and `Consequence` of a COU are a `Level` (`LOW`, `MEDIUM` or `HIGH`); the model risk is the lookup `RISK_TABLE` (`model_risk`):

| influence \ consequence | low | medium | high |
| --- | --- | --- | --- |
| low | low | low | medium |
| medium | low | medium | high |
| high | medium | high | high |

Influence is low when the model is one argument among others, medium when the prediction of the model is the main argument, high when it stands alone (no data of the condition, C4). Consequence is low for an explanation, medium for a statement somebody may act on (a dose adjustment) and high only with a rationale.

A COU which declares no `acceptance` takes the default of its risk (`DEFAULT_ACCEPTANCE`, `default_acceptance(risk)`), all rules of which must be met on the decisive evidence:

| risk | rules |
| --- | --- |
| low | `within_2fold` >= 70 % |
| medium | `within_2fold` >= 80 % and GMFE <= 1.5 |
| high | `within_2fold` >= 90 %, `within_1.5fold` >= 70 % and GMFE <= 1.3 |

The thresholds are a starting point and not taken from a guideline, which leaves them to the COU. A COU overrides them with its own rules: `Acceptance(criterion, minimum)` (the percentage of the points a criterion of `sbmlsim.fit.evaluation`, e.g. `Fold(2.0)`, accepts is at least `minimum`), `MaxGmfe(maximum)` and `AfeRange(low, high)` (the bias).

A percentage of a few points says little, so a verdict also needs a minimum of evaluable decisive points. A COU which declares no `min_points` takes the default of its risk (`DEFAULT_MIN_POINTS`, `ContextOfUse.minimum_points`): 5 for a low, 10 for a medium and 20 for a high risk. A COU overrides it with `min_points` (at least 1), e.g. for an endpoint of which a study reports a few values only; the rationale of the COU says why.

## Declaring the contexts of use

A model package gets a module `credibility.py` with the object `CREDIBILITY`, a `Credibility`, and its [project](project.md) references it like the experiments and the fit definitions, `credibility=".credibility:CREDIBILITY"`; it is resolved when a command uses it, `Project.load_credibility()`. A declaration is plain python, a tree of frozen dataclasses. This is a declaration of a model with calibration data, held-out doses and data the model does not describe (the metadata classes and the filters are those of the project, the test package `tests/credibility_toy` has a complete one):

```py
from sbmlsim.credibility import (
    ContextOfUse,
    CouLevel,
    Credibility,
    Domain,
    Evidence,
    EvidenceRole,
    Level,
    QuantityOfInterest,
    Unclaimed,
)
from sbmlsim.fit.filters import metadata_filter, named_filter

QUANTITIES = (QuantityOfInterest("[A]", "the concentration of A"),)

CREDIBILITY = Credibility(
    contexts=(
        ContextOfUse(
            id="describe",
            title="Describe the decay",
            question="Does the model describe the intravenous decay of A?",
            level=CouLevel.C1,
            influence=Level.LOW,
            consequence=Level.LOW,
            quantities=QUANTITIES,
            domain=Domain(),
            evidence=(
                Evidence(
                    "intravenous data",
                    EvidenceRole.CALIBRATION,
                    metadata_filter(StudyMetaData, route={Route.IV}),
                ),
            ),
        ),
        ContextOfUse(
            id="dose",
            title="Decay after a dose",
            question="Does the model predict the decay after a dose of the data?",
            level=CouLevel.C2,
            influence=Level.MEDIUM,
            consequence=Level.MEDIUM,
            quantities=QUANTITIES,
            domain=Domain(
                metadata={"route": (Route.IV,), "health": (Health.HEALTHY,)},
                statement="the single doses of the data",
            ),
            evidence=(
                Evidence("doses", EvidenceRole.VALIDATION, named_filter(DOSE)),
                Evidence(
                    "other healthy data",
                    EvidenceRole.VALIDATION,
                    Unclaimed(
                        within=metadata_filter(
                            StudyMetaData, route={Route.IV}, health={Health.HEALTHY}
                        )
                    ),
                ),
            ),
        ),
    ),
    limitations=(
        Evidence(
            "scaled data",
            EvidenceRole.LIMITATION,
            named_filter(LIMITATION),
            "data at three times the decay which the model does not describe",
        ),
    ),
)
```

The project gives the declaration and its domain (`credibility_settings`, `CredibilitySettings`: the metadata class of the fit mappings, the fields a domain can claim, the evidence extensions and the battery hook):

```py
PROJECT = Project(
    ...,
    credibility=".credibility:CREDIBILITY",
    credibility_settings=CredibilitySettings(
        metadata_type=".metadata:StudyMetaData", domain_fields=("route", "health")
    ),
)
```

- `ContextOfUse`: `id` (a slug matching `^[a-z][a-z0-9_]*$`, unique in the model), `title`, `question` (one sentence ending with `?`), `level`, `influence`, `consequence`, `quantities`, `domain`, `evidence`, and optional `acceptance` (empty means the default of the risk), `limitations` (evidence of role `LIMITATION` of this COU, subtracted like the limitations of the model), `rationale` (why this level, influence and consequence), `extrapolation` (see the verdicts), `min_points` (`None` means the default of the risk) and the settings of the extensions (`extensions`). `ContextOfUse.risk`, `ContextOfUse.minimum_points` and `ContextOfUse.decisive_role` (calibration for C1, else validation) are derived.
- `QuantityOfInterest(id, description)`: an observable of the fit mappings or the observable of a value per simulation (`pk.cmax`). The observable column of the data holds the selection, `[A]` for a concentration, so a quantity is matched after the brackets are removed (`[A]` and `A` are the same quantity); a quantity which matches no observable of the fits is an error. A quantity restricts the evidence of its COU to its observables. `ratio` marks a ratio between arms: `ratio=True` declares it without a pairing, it is matched against no observable, restricts no evidence and is shown as "ratio, no pairing declared"; a `Pairing` pairs its arms and is assessed by the evidence extension which claims it. The absolute quantities decide the verdict, a ratio adds to it, so a COU has at least one quantity which is no ratio.
- `Domain`: `metadata`, the claimed values by field of the metadata of the fit mappings (the fields `CredibilitySettings.domain_fields` of the project names, every enum field of its metadata class by default), and `statement`, a sentence for what the metadata cannot say (dose range, duration). The coverage is assessed on the metadata; the statement is shown, marked "statement, not assessed". A COU above C1 claims at least one value of the metadata, without one its domain would be covered by any data.
- `Evidence(name, role, selection, description, fit=None)`: the selection is a filter of fit mappings (`named_filter` of the mapping keys by study id, `metadata_filter` of the metadata class, `sbmlsim.fit.filters`) or `Unclaimed`; `fit` restricts the evidence to one subset of the adopted fits, and a subset the adopted fits do not have is an error naming the subsets they have.
- `Credibility(contexts, limitations, notes)`: the limitations are evidence of the whole model. A limitation, of the model or of a COU, is data the model does not describe and is subtracted from the evidence of every COU.

## The checks of a declaration

A declaration is validated when it is built (`DeclarationError` lists every problem):

- the ids match `^[a-z][a-z0-9_]*$` and are unique;
- a question ends with `?`;
- a COU has at least one quantity of interest which is no ratio;
- a COU of level C2 to C5 claims at least one value of the metadata in its domain, a C1 COU may claim none;
- the evidence of `limitations`, of the model and of a COU, has the role `LIMITATION`;
- a C1 COU has calibration evidence, a C4 COU has an `extrapolation` text (it has no validation data of its condition by definition), every other COU has validation evidence or an `extrapolation` text;
- `Unclaimed` is validation evidence and at most once per COU;
- a declared `min_points` is at least 1;
- the setting of an extension (`ContextOfUse.extensions`) checks itself.

When the project loads it (`Project.load_credibility`), the declaration is checked against the credibility settings of the project as well: the claimed metadata values are members of the enums of the domain fields, a setting of `extensions` names an extension of the project and a pairing of a ratio quantity is claimed by one. The assessment checks the declaration against the data on top of that, and any failure is an error and not an empty result:

- a quantity which matches no observable;
- an evidence which selects no point, named with its COU, evidence and the fits searched;
- validation evidence on a mapping which is training data in any adopted fit. Circular evidence is a hard error and never a warning: a mapping fitted in one fit counts as fitted in every fit. The same holds for a limitation on a fitted mapping;
- kinds of mappings which differ from the recorded ones (see below).

`check_declaration(project)` (`sbmlsim.credibility.check_declaration`) returns every problem of a declaration and of the records as messages, from the fit definitions and the records, without a simulation; the tests of a model repository use it.

## The evidence

| role | what it selects | what the assessment does with it |
| --- | --- | --- |
| `CALIBRATION` | the training points of the selection | decides a C1 COU; for every other level it is reported and never decides |
| `VALIDATION` | held-out points | decides the acceptance of every level but C1; a training point is an error, an outlier is dropped and counted |
| `LIMITATION` | data the model does not describe | subtracted from every other evidence of every COU, reported apart, never in an acceptance |

A limitation stays a validation mapping for the fit (the kinds of the fit are not changed); for the credibility it is a limitation. The points a limitation removes from an evidence are counted per evidence and shown in the report; a calibration keeps only training points, so it counts only the removed points it would have kept.

`Unclaimed(within=...)` is for validation data which is computed instead of named, e.g. held-out subjects: it takes the validation points which no other evidence of the model selects (the selected points of the other evidence, of any COU) and no limitation, restricted to the mappings `within` accepts. It is allowed once per COU and only as validation evidence. Without `within` it would take every validation point nobody claims, so a model gives it the metadata of the data it means. Two `Unclaimed` of different COUs may share points. A mapping which is training data in any adopted fit counts as fitted in every fit, so `Unclaimed` and the unclaimed report exclude it.

### The kinds come from the records

The kind of a mapping (`training`, `validation`, `outlier`) is the one the record of an adopted fit stores (`metrics_mappings.tsv`), keyed by experiment and mapping, never by the mapping key alone (two studies can share a key). The assessment compares the recorded kinds with the kinds of the current selection of the fit definitions and fails on any difference, naming the mappings: a split changed after the adoption without a refit would let fitted data count as validation. The remedy is a refit and a new adoption, never an override. The data points are evaluated fresh on the current model and data, not read from the metric tables of the record.

## The verdicts

Per COU, on the union of its validation evidence (each point once; the calibration evidence for C1):

| verdict | meaning |
| --- | --- |
| `supported` | the acceptance is met on at least the minimum of evaluable points, every claimed value of the domain is covered by validation data (for C1 by calibration data) and every quantity of interest which is no ratio is covered |
| `supported_with_extrapolation` | the acceptance is met on at least the minimum of evaluable points, some claimed value or quantity is not covered, and the COU has an `extrapolation` text which says why the claim holds; a C4 COU, which extrapolates by definition, is never plainly `supported` but at best this |
| `not_supported` | an acceptance rule is not met |
| `insufficient_evidence` | no evidence that can be evaluated, fewer evaluable points than the minimum, or claimed values or quantities not covered and no `extrapolation` text |
| `not_verified` | a check of the verification failed: the decisive predictions change with tighter tolerances or another integration method, or the setup has a problem, so the comparison with the data is not to be trusted |

The verdict is decided in this order, the first step which applies decides:

1. no decisive point: `insufficient_evidence`;
2. a failed check of the verification: `not_verified` (see [Verification](#verification));
3. an acceptance rule not met: `not_supported` (a failure on few points is still a failure);
4. an acceptance rule not evaluable: `insufficient_evidence`;
5. fewer evaluable points than `minimum_points`: `insufficient_evidence`;
6. every claimed value and every quantity which is no ratio covered: `supported` (C4: `supported_with_extrapolation`);
7. an `extrapolation` text: `supported_with_extrapolation`;
8. else `insufficient_evidence`.

Coverage counts only the points the criteria can evaluate: a fold error needs a finite `DV` and `IPRED`, both greater than 0. The numbers are reported as `n_decisive` (the points of the deciding evidence), `n_evaluable` and `minimum_points`; a rule without an evaluable point has the value `NaN` and `met` of `None`, is not evaluable, and gives `insufficient_evidence`, never `supported`. A `not_supported` verdict is a finding to keep and explain, not a reason to loosen the acceptance. A `supported` C1 COU rests on the data the model was fitted to, so the console, the report and the release page show next to every verdict what decided it (decided on: calibration or validation).

Domain coverage is per claimed (field, value): `validated` when validation points with the value exist, `calibrated` when only calibration points exist (this counts as covered for a C1 COU), `uncovered` otherwise. The values of the points outside of the claimed domain are reported as well; evidence that does not fit the claim is a smell of the declaration.

Quantity coverage is per quantity of interest which is no ratio: `covered` when an evaluable decisive point of its observable lies inside the claimed domain, i.e. for every claimed field of the metadata the value of the point is one of the claimed values, `uncovered` otherwise. A quantity validated only outside the claim is uncovered, although every claimed value may be validated by the points of other quantities. The coverage is computed for every level, C4 included, and the coverage of combinations of values is not assessed (each claimed value on its own).

## Evidence extensions

A domain adds evidence with an `EvidenceExtension` (`sbmlsim.credibility.extensions`, whose module docstring is the contract), registered in `CredibilitySettings.extensions` of the project: it claims the pairings of ratio quantities, selects and prepares its data, assesses a context into a section of the record by the name of the extension (`ContextResult.section(name)`, `CredibilityAssessment.extensions`), adds rules to the acceptance of a context (they enter the verdict after the rules of the context and are rows of its acceptance table with the field `extension`), states a factor of V&V 40 which joins the one of the core (`combine_factor`), adds blocks and sections to the report, tables and columns of the scorecard, which only it judges, and pools its sections over models. Every hook is a no-op by default. The records read the sections of an extension and the fields of the records of pkpdsim 0.17 to 0.19 as such sections.

## Verification

A verdict compares the predictions of the adopted fits (`IPRED`) with the data. Whether these numbers are the solution of the model, and not of the settings of the integrator or of a broken setup, are three factors of V&V 40: the discretization error, the numerical solver error and the use error. The assessment verifies the decisive predictions of every COU with one check per factor (`sbmlsim.credibility.verification`, `verify`):

| check | factor | what is compared |
| --- | --- | --- |
| `tolerances` | discretization error | every adopted fit evaluated again at its reference parameters with the absolute and the relative tolerance of its settings divided by 100 (`TIGHTENING`; every kind and override of an `AbsoluteTolerance`) |
| `integrator` | numerical solver error | the same at the tightened tolerances with the BDF of CVODE of at most order 2 (`SECOND_METHOD_ORDER`) instead of order 5, a second integration method: another formula with its own local error, which stays stable on a stiff model |
| `setup` | use error | every build of the model is valid SBML with consistent units (the consistency checks of libsbml with the units, every issue of severity warning or above) and every decisive prediction is finite |

The difference of a prediction is relative, `|alt - ref| / max(|ref|, |alt|, floor)`, with the floor `1e-6` (`SCALE_FLOOR`) of the largest reference prediction of its mapping: a difference below a millionth of the peak of the curve is none, and a point at zero compares with the curve. A numerical check of a COU passes when the largest difference of its decisive predictions is at most `0.01` (`LIMIT`, more than an order below the narrowest default criterion, 1.5-fold and a GMFE of 1.3) and no acceptance rule changes its result with the alternative predictions; a prediction which is finite in the reference and not in the alternative is an infinite difference, and an evaluation of a fit which fails is a failed check with its message. A failed check makes the verdict `not_verified`; the three factors are `assessed` with the result as their statement.

The checks run with every assessment and evaluate every adopted fit twice more; `verification: false` of the kind skips them and the battery below for a quick look, the three factors are then `not_assessed` and the verdicts are those of the comparison alone. The non-stiff method of CVODE (Adams-Moulton) is no second method for stiff models, on which it takes too many steps, while the BDF of order 2 is A-stable. The verification does not run the experiments on a second simulator (the battery compares the variants with AMICI and COPASI), the explicit Runge-Kutta integrator of roadrunner (sbmlsim sets the settings of CVODE only, and an explicit method is no method for a stiff model) nor a check of the doses of the simulations against the interventions of the data (the link of a simulation to an intervention is not declared).

### Battery of the variants

The checks above ask whether the predictions a verdict is decided on are the solution of the model. The battery asks whether the model itself is sound, without any data (`sbmlsim.credibility.battery`, `run_battery`). Every variant of the project is simulated with roadrunner (`BatterySettings`: the horizon and the grid in the time of the model, `rtol 1e-10` and `atol 1e-16`) and checked against what every model must fulfil; a check is open (`Check`, `register_check`), the generic ones are:

| check | factor | what is checked |
| --- | --- | --- |
| `non_negativity` | numerical code verification | no species becomes negative in the case without a change and in the cases of the hook |
| `steady_state` | numerical code verification | every state but the sinks (`accumulating_states`) stays at its start; only when `BatterySettings.steady_state` declares that the model starts at its steady state (off by default, not applicable otherwise: a decay or an oscillator does not) |
| `cross_simulator` | numerical code verification | AMICI and COPASI (each one which is installed, through `Simulator(engine=...)`, roadrunner at `O0`) give the results of roadrunner; what an engine does not simulate as roadrunner (`UnsupportedModelError`) and a compilation of AMICI longer than `compile_timeout` (900 s) are not compared and named, and a check where nothing was compared is not applicable |
| `plausibility` | use error | the initial state and the adopted fits are plausible (finite values, fitted values not at their bounds) |
| `rebuild` | software quality assurance | the factory builds the models of the package again, `Project.create_models` and `changed_outputs` |

The settings of the battery are the ones of the hook with the ones of the project over them (`CredibilitySettings.battery_settings`, e.g. `{"engines": ["copasi"], "steady_state": true}`). A project adds the simulations and checks of its domain with a `BatteryHook` (`CredibilitySettings.battery`): it gives what it reads of a variant, the SBML to simulate, the selections, `BatteryCase`s (changes at times), checks of its own (`contribute`), sections of the record and the HTML and tables of the report, e.g. the doses of every route and the mass balance of a PBPK model.

## Cross-validation

A fit describes its training data; how well it predicts a study it has not seen is a question of the form of the model. `cross_validate` (`sbmlsim.credibility.cross_validation`) refits the problem of an adopted fit once per training study without that study (its training mappings become validation mappings, `held_out`), by least squares from the adopted parameters with the adopted fits of the other problems fixed, and predicts the held-out study with the refitted parameters. `write_cross_validation(result, out)` stores the result in `fits/<fit id>/cross_validation.json` of an out, which is committed next to the record of the fit, where the assessment reads it (`read_cross_validations`): per study its points, the GMFE and the share within 2-fold of the held-out predictions and of the predictions of the adopted fit for the same points, and the cost of the refit, and the GMFE of all held-out points together. The input `cross_validation` of the kind `credibility` takes such an out instead of the cross-validations of the records.

The factor model form of a COU is `assessed` when an adopted fit of its decisive evidence is cross-validated, with the statement of the cross-validation; the record has them (`CredibilityAssessment.cross_validations`), `cross_validation.tsv` one row per fit and study, the report a section Cross-validation. A held-out GMFE close to the fitted one is a model form which carries across studies, a much larger one a model whose parameters are fitted to every study. The cross-validation fits the problem once per training study, so it runs on demand and is stored with the fit.

## Parameter uncertainty

The verdict is the acceptance at the adopted parameters, the best estimate. Parameters which the data fit about as well may predict differently, so `uncertainty: true` of the kind propagates the uncertainty of the adopted fits to the decisive predictions (`sbmlsim.credibility.uncertainty`, `propagate_fits`, `PropagationSettings`).

Every adopted fit is drawn from its record: the marginal of every fitted parameter is the distribution the releases draw (`ParameterUncertainty.distribution`: the profile likelihood interval, else the standard error of the Fisher information, truncated to the bounds; uniform in the bounds without a closed side of the profile, see [Uncertainty](uncertainty.md#the-uncertainty-of-the-fitted-parameters)), and the dependence is the correlation of the Fisher information of the fit (`fisher.json`) in the space of the fit, as the correlation of the normal scores of a Gaussian copula (`sbmlsim.uncertainty.draws.draw`, scrambled Sobol points, `draws` 128 and `seed` 1234): for a normal or log-normal marginal this is the multivariate normal of the Fisher covariance in the scale of the fit, with the widths of the profile. A fit without a Fisher information or with one which is not of full rank is drawn independently, and the record says why. The fits are drawn independently of each other; the parameters a fit fixes from another adopted fit stay at their adopted values, so the uncertainty a later fit inherits is not propagated. Every fit is evaluated at each of its draws with the report of the fresh evaluation (`Propagation`); a draw whose evaluation fails is counted.

Per COU (`ContextResult.uncertainty`) the record has the draws of the fits of its decisive points (parameters, correlated or not and why, failed draws), the 90 % bands of the decisive predictions (the quantiles 0.05 and 0.95 over the draws), their coverage of the evaluable decisive observations, per quantity of interest which is no ratio the median fold width of the bands (`band_fold`, upper over lower, the precision of its predictions) with its coverage, and per acceptance rule the probability that it is met over the draws with the anytime-valid confidence sequence and the decision of the statement `P(met) >= 0.9` of `sbmlsim.uncertainty.testing` at the level 0.95 for the rules of the COU together (Bonferroni): `accepted` (the acceptance is robust to the parameter uncertainty), `rejected` (more than a tenth of the plausible parameters fail it), `equivalent` (within the margin of 0.05 of the threshold), `undecided` with the draws it would need, or `not_evaluable`. A draw on which a decisive prediction is not finite while the reference is finite is a failed draw of every rule. A band of the parameter uncertainty holds the prediction of the mean, not the observations: the variability between the subjects and the error of the measurements lie outside of it, so its coverage is reported and decides nothing.

The draws also give the sensitivity of the quantities of interest, without further simulations: per quantity the mean log ratio of its predictions to the adopted predictions on every draw, per rule its value on every draw, and per fitted parameter its first-order index, the share of the variance of the target over the draws which the parameter explains alone (`sensitivity_indices`, estimated from the given draws by binning, `sbmlsim.uncertainty.statistics.first_order_given_data`, Plischke et al. 2013). Unlike the Sobol designs of a [sensitivity analysis](sensitivity.md), which vary the parameters in boxes of a relative width, the indices are those of the uncertainty of the fits, so they name the parameters whose uncertainty the predictions of a COU depend on. The index of a parameter includes the effect of the parameters it is correlated with, so the indices of correlated parameters do not add up; with few draws an index has a bias of up to about `sqrt(n)/2/n`. The report shows the leading parameters of every target (up to three of at least 0.05), `uncertainty_sensitivity.tsv` every index (`context`, `target`, `kind`, `fit`, `parameter`, `index`).

The verdict does not change. The factor model inputs is `assessed` with the draws, the largest band fold, the coverage, the parameter carrying most of the variance of every quantity and the decision of every rule; without `uncertainty` it stays `not_assessed`. The scorecard has the columns `uncertainty_robust` (every rule accepted: true, one rejected: false, else empty) and `band_fold` (the largest band fold), and `compare` reports the regression `uncertainty_robust` of a COU which was robust. `uncertainty.tsv` has one row per COU and rule (`context`, `rule`, `limit`, `estimate`, `low`, `high`, `threshold`, `decision`, `n`, `n_failed`, `n_needed`), `uncertainty_quantities.tsv` one per COU and quantity (`context`, `quantity`, `n_points`, `band_fold`, `coverage`).

## Variability and the predictive check

The adopted fits predict every curve with the typical subject of its study group, while the observations scatter between the subjects of a study and between studies. A model declares the variability of its subjects (`Credibility(variability=Variability({...}))`, a `sbmlsim.fit.variability.Variability`: a relative distribution per target, e.g. `LogNormal(cv=0.3)` of a clearance, a factor with the median 1, correlated by `Covariance`s, e.g. the omega matrix of the logarithms, with `between=Variability({...})` the variability between the studies, the factors which all subjects of a study share), and `variability: true` of the kind (`subjects` 64, `studies` 16) checks the decisive points with it (`sbmlsim.credibility.variability`, `predictive_checks`, `VariabilitySettings`): every adopted fit is evaluated for the subjects, each subject a factor per target times the value its simulation gives the target (the value the experiment sets for the study group, of a fitted parameter or of the model), so the subjects of every study group vary around its inputs; with a variability between the studies the subjects are `studies` studies of `subjects` subjects each. With `uncertainty` the subjects are nested in the draws of the parameters of the fit, without it they vary around the adopted parameters. A data point of a group of `N` subjects (its count) is compared with the means of virtual groups of `N` subjects drawn from the subjects of a study (the geometric mean for data with a geometric standard deviation).

Per COU (`ContextResult.variability`) the record has the subjects, the studies, the parameter vectors, the declared targets and distributions, the evaluable decisive points with a band and how many are means of groups, the coverage (the share inside their band) with its Wilson interval at the level 0.95, the expected coverage (90 %, the probability between the quantiles) and the agreement: `consistent` (the expected coverage lies in the interval), `narrow` (the bands hold fewer observations than expected: the declared variability with the parameter uncertainty does not explain the scatter, or the typical predictions are biased) or `wide`, and the coverage per quantity of interest. The verdict does not change; the factor equivalency of inputs states the predictive check. The band of a mean of `N` subjects narrows with `sqrt(N)`, so the means of the studies test the typical predictions hard: their scatter between studies (other populations, assays and protocols) is no variability of the subjects, which the variability between the studies declares. `variability.tsv`, `variability_quantities.tsv` and `vpc.tsv` hold the tables.

## Locked splits

A validation counts only if its data was set apart before the predictions were seen. The kinds of the mappings (training, validation) are fixed by the records of the adopted fits, a changed kind is refused; which validation data answers which question is fixed by the declaration, which can be changed after a verdict was seen. The lock fixes it as well (`sbmlsim.credibility.lock`): the kind with `lock: true` writes `credibility.lock.json` into its out, the mappings (`fit/experiment/mapping`) of every evidence and every limitation by `context/name` (a limitation of the model `/name`), which is reviewed and committed into the root of the project with the declaration. Afterwards:

- an assessment compares the evidence with the lock before it evaluates anything and refuses a difference (`LockMismatchError`), naming every evidence which is new, gone or selects other mappings with the added (`+`) and removed (`-`) mappings (`lock_differences`); `check_declaration` reports the same;
- a changed split is renewed explicitly with `lock: true`, so the change is a reviewed diff of the lock in the history of the repository;
- the record says whether the evidence matched the lock (`Provenance.lock`: `matched`, `written` when the lock was written with the assessment, `None` without a lock), the report shows it in the provenance.

A model without a lock is assessed as before.

## Across models: scorecard

Every model is assessed on its own. The scorecard reads the records of many models (`credibility.json`) into one table, one row per model and COU with the level, the risk, the verdict, the points, whether it is verified, the failed checks and the value of every acceptance rule, with the columns of the extensions (`sbmlsim.credibility.scorecard`, `load_records`, `scorecard`). Stored as a TSV it is the baseline of a repository of models: a change of the shared library, of sbmlsim or of the data which makes a COU worse shows across all models at once (`compare`): a COU of the baseline which is missing, a worse verdict (supported, supported with extrapolation, insufficient evidence, not verified, not supported), a COU which is no longer verified, a regression an extension judges, and a rule whose value got worse by more than the tolerance (5 points of a percentage, 5 % of a GMFE, an AFE 5 % further from 1; a value which is no longer evaluable). A new model, COU or rule is no regression. An extension pools its sections over the models (`pool`), e.g. the qualification of a scaling across drugs.

```bash
python -m sbmlsim.run verify scorecard results/ --out scorecard.tsv
python -m sbmlsim.run verify scorecard results/ --baseline scorecard.tsv
```

The command prints the verdicts, the pooled qualification and the regressions against `--baseline`; it exits with 1 for a regression and 2 for a record or baseline it cannot read, see [Verify](run.md#verify).

## Running and reading the report

The kind `credibility` assesses the project into `out/credibility/` (`python -m sbmlsim.run credibility`, the task type `sbmlsim.credibility`):

| field | default | meaning |
| --- | --- | --- |
| `verification` | `true` | verify the decisive predictions |
| `battery` | `true` | run the battery of the checks of the variants with the verification |
| `cross_simulator` | `true` | run the check of the second engines of the battery, as the settings of the battery say; `false` switches it off alone |
| `lock` | `false` | write the lock of the evidence into the out, `credibility.lock.json`, instead of comparing the evidence with the lock of the project; the lock of the project is never written |
| `uncertainty` | `false` | propagate the parameter uncertainty of the adopted fits, `draws` (128) and `seed` (1234) of the propagation |
| `variability` | `false` | check the decisive points with the declared variability of the subjects, `subjects` (64) and `studies` (16) |
| `source_commit` | none | the commit of the source repository of a release, recorded in the provenance |

Its input `cross_validation` is an out with `fits/<fit id>/cross_validation.json`, the cross-validations of the adopted fits instead of the ones of their records, and its input `record` is an earlier `credibility.json` (or an out of the kind), whose outputs are written again without an assessment (an option of an assessment with it is refused, also `verification`, `battery` or `cross_simulator` switched off; a `cross_validation` input with none of the adopted fits is refused as well):

```bash
python -m sbmlsim.run credibility credibility.yaml --out results/credibility
python -m sbmlsim.run credibility credibility.yaml --out results/report --input record=results/credibility
```

The console shows a table of the COUs, their verdicts, what decided them and the evaluable points as `evaluable/minimum`, and below it how many COUs are verified. `index.html` of the report (one page, no network) has:

- the **overview**: the COUs with level, risk, verdict (in words and in a colour), the role of the evidence which decided it and the decisive and evaluable points with the minimum, and the limitations of the model;
- a **section per COU**: question, rationale, the evaluable points with their minimum, quantities with their coverage and points, the claimed domain with its coverage and the statement (not assessed), the evidence table (role, points, mappings, studies, GMFE, AFE, the percentages of the criteria, the points removed by limitations and outliers), the acceptance table (value, limit, met), the checks of the verification (points, largest difference and its point, whether the acceptance changed, the result and the problems), the coverage of the predictive check, the table of the credibility factors, the extrapolation text and the blocks of the extensions;
- the **limitations**: their metrics, explicitly outside of any acceptance;
- the **unclaimed validation data**: the validation points per fit which no evidence and no limitation selects (held-out data which answers no declared question);
- the **verification**: the settings of the integrator of every adopted fit and of its alternatives, the issues of the builds, the battery with the statement of every check and the sections of the hook;
- the **provenance**: the versions of sbmlsim, sbmlutils, libroadrunner and the distributions of the extensions, the commit of the model repository and whether the folder of the model had changes, the adopted fits and the date. A release passes the commit of its source repository (`source_commit`), the assessment then records it and leaves the dirty state unknown.

The factors of V&V 40 are listed per COU with a status: `assessed` (output comparison, test samples, test conditions and the relevance of the validation activities, from the metrics, the evidence and the coverage; the discretization error, the numerical solver error and the use error from the verification, the numerical code verification and the software quality assurance from the battery after their platform statement), `platform` (a statement which holds for the whole stack, e.g. the code verification by the SBML Test Suite and the PEtab suites of sbmlsim) or `not_assessed`, so a reader sees what is missing.

The record `credibility.json` (`CredibilityAssessment.to_json`, read back with `from_json`) holds everything, and the tables `contexts.tsv`, `evidence.tsv`, `domain.tsv` and `quantities.tsv` are one row per COU, per COU and evidence, per COU, field and value and per COU and quantity, `verification.tsv` one row per COU and check (the applicable checks of the battery included), `battery.tsv` one row per check of the battery (`check`, `factor`, `status`, `cases`, `largest`, `at`, `statement`, `problems`), the tables of the uncertainty and the variability named above and the tables of the extensions (`write_outputs`). The assessment is reproducible from the record alone: the input `record` of the kind writes the outputs again with the current report.

## References

- ASME V&V 40-2018, Assessing Credibility of Computational Modeling through Verification and Validation: Application to Medical Devices.
- FDA, Assessing the Credibility of Computational Modeling and Simulation in Medical Device Submissions (2023).
- ICH M15, General Principles for Model-Informed Drug Development (Step 4, 2026).
- EMA, Guideline on the reporting of physiologically based pharmacokinetic (PBPK) modelling and simulation (2018).
- Kuemmel C et al. (2020), Consideration of a credibility assessment framework in model-informed drug development: potential application to physiologically-based pharmacokinetic modeling and simulation, CPT Pharmacometrics Syst Pharmacol 9:21-28.
- Musuamba FT et al. (2021), Scientific and regulatory evaluation of mechanistic in silico drug and disease models in drug development: building model credibility, CPT Pharmacometrics Syst Pharmacol 10:804-825.
- Erdemir A et al. (2020), Credible practice of modeling and simulation in healthcare: ten rules from a multidisciplinary perspective, J Transl Med 18:369.
- Plischke E, Borgonovo E, Smith CL (2013). Global sensitivity measures from given data. Eur J Oper Res 226:536-550.
