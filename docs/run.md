# Analyses

`python -m sbmlsim.run` runs the analyses of a model package from the command line: its simulation experiments, fits, profile likelihoods, scans, sensitivity and uncertainty analyses, the build of its models and the actions of its own. It runs an analysis of a [project](project.md) from a spec, a small YAML or JSON file of its settings, into an output directory. The same analysis runs whole on one machine or in units on several machines, which merge to the result of the whole run, so a fit of 64 repeats or a scan of 10000 points can be spread over a cluster with [simrun](#simrun) and give the same numbers as on a laptop.

## Quick start

The HCTZ example of the [fitting](fitting.md) has a project, `examples.hctz_fitting.project:PROJECT`. Run from the root of the repository, which `python -m` puts on the path, list its analyses:

```bash
python -m sbmlsim.run list examples.hctz_fitting.project:PROJECT
```

```text title="Output"
kind             name   description
simulate         all    the simulation experiments of the project
fit              PK     fit of the problem 'PK'
identifiability  PK     profile likelihoods of the fit of the problem 'PK'
fisher           PK     Fisher information of the fit of the problem 'PK'
report           PK     report of the fit of the problem 'PK'
petab            PK     PEtab v2 problem of the fit problem 'PK'
fit              IV     fit of the problem 'IV'
...
scan             dose   oral doses of 12.5, 25 and 50 mg over one day
scan             sobol  Sobol design of the dissolution and the renal excretion of a 25 mg oral dose
sensitivity      sobol  sobol indices of the scan 'sobol'
uncertainty      cl     the AUC of a 25 mg oral dose under a renal excretion uncertain by 30 %
```

A spec names the project and the analysis:

```yaml title="fit.yaml"
project: examples.hctz_fitting.project:PROJECT
subset: PK
repeats: 8
seed: 1
```

and the run fits the problem, then profiles the parameters of the fit and writes the report of both:

```bash
python -m sbmlsim.run fit fit.yaml --out results/fit
python -m sbmlsim.run identifiability profiles.yaml --out results/profiles --input fit=results/fit
python -m sbmlsim.run report report.yaml --out results/report --input fit=results/profiles
```

where `profiles.yaml` and `report.yaml` hold the same `project` and `subset`. The fit uses every CPU it may use, see [CPU quota](#cpu-quota-and-threads), `--workers 4` limits it.

## The command

```bash
python -m sbmlsim.run <kind> <spec> --out DIR [--workers N] [--resume] [--provenance FILE] [--input PORT=PATH ...] [--summary FILE]
python -m sbmlsim.run plan <kind> <spec> [-o FILE]
python -m sbmlsim.run unit <kind> <spec> <index> --out DIR [--input PORT=PATH ...]
python -m sbmlsim.run merge <kind> <spec> --out DIR [--input PORT=PATH ...] [--summary FILE]
python -m sbmlsim.run list <project> [--json] [--experiments]
python -m sbmlsim.run verify <subcommand> ...
```

The first form runs an analysis whole (its command is `run`, which may be left out). `plan`, `unit` and `merge` run it in [units](#units-and-merges), `list` lists the analyses of a project as a table or, with `--json`, as JSON; with `--experiments` it also lists the groups of the simulation experiments and every experiment (kind `simulate`, which runs them by these names), which imports the experiments. `verify` is a family of commands that compare the results of two checkouts, see [Verify](#verify). `--profile`, anywhere in the command line, runs a command which writes into `--out` under the sampling profiler of python 3.15, see [Profiling](#profiling).

| argument | meaning |
| --- | --- |
| `<kind>` | the kind of analysis, see [Kinds](#kinds), or `module:attribute` of a kind of another package |
| `<spec>` | the spec, a JSON file (`.json`) or a YAML file; a relative path of a path field of the spec, e.g. the input `fit`, is relative to the file, a path inside another value, e.g. in `optimizer`, is taken as it is |
| `--out DIR` | the output directory; a whole run needs it empty or not existing (unless `--resume`), while the `unit` and `merge` commands of one plan share it, see [Units](#units-and-merges); nothing is written outside of it |
| `--workers N` | the number of processes, every CPU of the quota for a kind which uses several and 1 for the others without |
| `--resume` | continue the run of the same spec in `--out`, see [Checkpoints](#checkpoints-and-resume) |
| `--provenance FILE` | the provenance of the sources the run runs on, see [Provenance](#progress-and-provenance) |
| `--input PORT=PATH` | the path of an input of the spec, the output of another run, e.g. `--input fit=results/fit`; repeated per input, also as `--input=PORT=PATH` |
| `--summary FILE` | write the summary of a successful run also into this file |

`plan` writes the planned spec as JSON to stdout or to `-o FILE`; the messages of every command go to stderr. `unit` and `merge` take no `--workers`, `--resume` or `--provenance`: a unit runs serially in its process, a unit or a merge into an output with files continues it, and the provenance of their sources is the file of the environment variable `SIMRUN_PROVENANCE`, if set.

## Kinds

| kind | runs | units | input |
| --- | --- | --- | --- |
| `simulate` | the simulation experiments of the project with their report | whole | |
| `fit` | the repeats of the optimization of a fit problem | blocks of repeats | |
| `identifiability` | the profile likelihoods of the parameters of a fit | one per parameter | `fit` |
| `fisher` | the Fisher information of a fit | whole | `fit` |
| `report` | the report of a fit, with its profiles and Fisher information when the input has them | whole | `fit` |
| `petab` | the fit problem as a PEtab v2 problem, starting from the best parameters of a fit if one is given | whole | `fit` |
| `adopt` | the adoption of a finished fit: its record and the adopted fits of the project with it | whole | `fit` |
| `scan` | a scan of the project | blocks of points | |
| `sensitivity` | a scan with sensitivity indices, and the indices, or a sensitivity analysis of the project over the groups of its simulations | blocks of points, or one per analysis | |
| `uncertainty` | the propagation of an uncertainty analysis | blocks of draws | |
| `factory` | the models built by the factory of the project, compared with the ones of the package | whole | |
| `action` | an action of the project, an analysis of its own | whole | |
| `credibility` | the assessment of the contexts of use of the project against its adopted fits, its record, tables and report | whole | `cross_validation`, `record` |
| `release` | the build of a release of the project as a repository of its own, which `python -m sbmlsim.release publish` publishes, see [Release](release.md) | whole | |

A spec forbids unknown fields. The fit kinds name a fit problem, the other kinds an analysis of the project:

| field | kinds | meaning |
| --- | --- | --- |
| `project` | all | the project, `module:attribute` |
| `subset` | `fit`, `identifiability`, `fisher`, `report`, `petab` | the fit problem, a key of the fits of the project |
| `analysis` | `scan`, `sensitivity`, `uncertainty` | the scan, the sensitivity analysis or the uncertainty analysis, a key of the project |
| `fit` | `identifiability`, `fisher`, `report`, `petab`, `adopt` | the records of a fit, the input `fit` |
| `name` | `action` | the action, a key of the actions of the project |
| `arguments` | `action` | the arguments of the action, JSON values by the names of the keywords of its function, `{}` without; the plan refuses arguments the function does not take. They are passed as they are: a relative path among them is relative to the current directory of the run, not to the spec file |
| `version` | `release` | the version of the release, `X.Y.Z` |
| `repository` | `release` | a checkout of the release repository, cloned instead of the repository on GitHub; only read |
| `framework_path` | `release` | a local checkout of the framework of the release (`ReleaseSettings.framework`) instead of the released one; such a build is not published |
| `units` | the split kinds | the number of units, set by the plan |

### fit

| field | default | meaning |
| --- | --- | --- |
| `opid` | the subset, with `adopt` the id of a fit | the id of the fit, the directory of its records in the output; an adopted fit has the id of a fit of its subset, `<subset>_<date>_<time>__<hash>` |
| `repeats` | 4 | the number of optimization runs |
| `algorithm` | `LSQ` | least squares `LSQ` or differential evolution `DE` |
| `seed` | drawn by the plan | the seed of the start values and of the runs |
| `studies` | all | the studies of the problem to fit |
| `timeout` | none | the seconds a single run may take |
| `optimizer` | `{}` | further arguments of the optimizer, e.g. `{"max_nfev": 50}`, and of the runner, e.g. the sampling of the start values `sampling` by its name (`LOGUNIFORM`, `UNIFORM`, `LOGUNIFORM_LHS`, `UNIFORM_LHS`, `START`); a setting of the spec, e.g. `seed`, is refused here |
| `start` | none | the point least squares starts from instead of samples of the bounds: the values by parameter id (`{"k": 0.3}`), a JSON file of them, the directory of the records of a fit or the id of a fit, adopted in the project (`fits_dir`) before a directory of that name; needs `LSQ` and one repeat. The plan resolves it into the values, which the planned spec keeps, so every unit and a resume start from the same point, and keeps the given start in `start_from`, which the provenance of the fit records; a relative path of a spec file is relative to the file, a fit id stays an id |
| `record_adopted` | `true` | record the adopted fits of the other subsets, whose values the models of the project carry, and these values in the provenance of the fit and name them in its report; nothing is fixed by it, the models carry the values |
| `fixed_fits`, `fixed_values` | set by the plan | the adopted fits of the other subsets and the values of their parameters which the fit does not fit, read once by the plan with `record_adopted`, so every unit records the same |
| `identifiability` | `false` | compute the profile likelihoods of the parameters of the best run after the fit, `<opid>/identifiability.json` |
| `profile` | the defaults | the settings of these profiles, the ones of the kind `identifiability` (`ProfileSpec`), e.g. `{"max_points": 10}`, with `pids` the parameters to profile (every parameter which is no element of a network without) |
| `fisher` | `true` | compute the Fisher information with the profiles, `<opid>/fisher.json` |
| `report` | `false`, with `adopt` `true` | write the report of the fit into `<opid>`, with the title of the project and the fixed adopted fits |
| `mapping_figures` | `true` | draw the figure of every fit mapping in the report |
| `adopt` | `false` | adopt the fit, which needs `identifiability`: its record into `fits/<opid>` and `adopted.json`, the adopted fits of the project with this fit for its subset |

The plan refuses a fit whose upper bounds let a remainder of a model of the project become negative, an initial assignment `f_c = 1 - f_a - f_b` whose fitted fractions and the values of the others sum above 1 (`sbmlsim.fit.project_fits.remainder_problems`), and checks the start before a repeat runs.

### identifiability

| field | default | meaning |
| --- | --- | --- |
| `pids` | set by the plan | the parameters to profile, every parameter which is no element of a network without |
| `alpha` | 0.95 | the confidence level of the intervals |
| `degrees_of_freedom` | 1 | the degrees of freedom of the threshold |
| `initial_step`, `min_step`, `max_step` | 0.1, 0.01, 1.0 | the steps of a scan in decades |
| `max_points` | 50 | the largest number of points of a scan in one direction |
| `reoptimize` | `true` | optimize the other parameters at every point |
| `optimizer` | `{}` | arguments of the least squares optimizer of the other parameters |
| `fisher` | `true` | compute the Fisher information as well |

The settings are the ones of the [profile likelihood](identifiability.md#profile-likelihood). A profile is computed on the problem of the studies the fit was restricted to and with the settings the fit ran with, both read from its records; the same holds for `fisher`, `report` and `petab`.

### fisher, report and petab

| kind | field | default | meaning |
| --- | --- | --- | --- |
| `fisher` | `step` | of `fisher_information` | the relative step of the finite differences |
| `report` | `mapping_figures` | `true` | draw the figure of every fit mapping |
| `report` | `group_by` | the groupings of the problem | the groupings of the report, the columns of one grouping per list |
| `report` | `adopted` | `false` | report the adopted fit of the subset from its record in the project, of every adopted fit for the subset `all` (no fit subset of a project may be named `all`), into `<fit id>/`, without the input `fit` |
| `adopt` | `force` | `false` | adopt a fit which the project has adopted already, replacing its record |

`adopt` adopts the fit of its input `fit`, the out of a fit or a records directory: the id of its result, which has the form of the id of a fit of a subset of the project, names the subset; it needs the profiles or the Fisher information of the fit and writes its record into `fits/<fit id>` and `adopted.json` with the fit adopted for its subset, as `fit` with `adopt` does.
| `petab` | `required_extension` | `true` | mark the extension `sbmlsim` as required |

### simulate, scan, sensitivity and uncertainty

| kind | field | default | meaning |
| --- | --- | --- | --- |
| `simulate` | `experiments` | every experiment | the names of the groups and of the classes of the experiments to run, every experiment once; the plan sets the names of the classes |
| `simulate` | `figure_formats` | the formats of the project | the formats of the figures, e.g. `["svg", "html"]`, which the plan writes in lower case |
| `simulate` | `uncertainty` | none | run the experiments over draws of the fitted parameters of the default variant of the model, the curves of their simulations as bands (see [Simulation experiments](experiments.md#over-the-draws-of-the-fitted-parameters)): `draws` (100), `seed` (drawn by the plan without) and the `quantiles` of the bands (`[0.05, 0.95]`) |
| `scan`, `sensitivity` | `scan_hash` | set by the plan | the hash of the scan the plan built; none for a sensitivity analysis over groups |
| `sensitivity` | `analyses` | the default analyses of the runner | the analyses of a sensitivity analysis over groups (`Project.sensitivities`), e.g. `[local, sampling]`, which the plan sets for a single runner |
| `sensitivity` | `subfolders` | every subfolder | the subfolders of a sensitivity analysis of several runners, which the plan sets |
| `sensitivity` | `run_plan` | set by the plan | what every unit of a sensitivity analysis over groups computes: the hash of the table of the parameters and every analysis with its groups and resolved settings, e.g. the number of samples of `SBMLSIM_SENSITIVITY_N`; a unit or a merge which resolves to another plan fails with a `PlanMismatch` |
| `uncertainty` | `seed` | the seed of the analysis, else drawn by the plan | the seed of the draws |

The scan, its observables and its design are defined in the project, so the spec of a scan, a sensitivity or an uncertainty analysis only names it, e.g. the Sobol indices of the scan `sobol` and the propagation `cl` with a seed of its own:

```yaml title="sobol.yaml"
project: examples.hctz_fitting.project:PROJECT
analysis: sobol
```

```yaml title="cl.yaml"
project: examples.hctz_fitting.project:PROJECT
analysis: cl
seed: 7
```

```bash
python -m sbmlsim.run sensitivity sobol.yaml --out results/sobol
python -m sbmlsim.run uncertainty cl.yaml --out results/cl
```

A sensitivity analysis of the project over the groups of its simulations (`Project.sensitivities`, see [Sensitivity analysis](sensitivity.md#sensitivity-analyses-of-a-project)) is run by the same kind; the spec names it and may choose its analyses:

```yaml title="decay.yaml"
project: my_model:PROJECT
analysis: decay
analyses: [local, sampling]
```

```bash
python -m sbmlsim.run sensitivity decay.yaml --out results/sensitivity
```

Every analysis is a unit, which writes its folder of the output, and the merge writes the table of the parameters; a resumed run keeps the analyses whose folder was written by the same plan. The plan fixes the numbers of samples: a unit run with another `SBMLSIM_SENSITIVITY_N` than its plan fails.

## The output

Every run writes the same files describing it into its output, besides its results:

| file | content |
| --- | --- |
| `run.json` | the record of the run: the kind, the hash of the spec, the versions of the packages and of the project, python, the host, the provenance of the sources or else the git revision of the project, the variables of the task which simrun sets in a pod (`SIMRUN_TASK_DIR`, `SIMRUN_TASK_TYPE`, `SIMRUN_INPUTS`, `SIMRUN_ATTEMPT`, `SIMRUN_ROLE`, `SIMRUN_WORKER`, `SIMRUN_NODE`, `SIMRUN_PROVENANCE`, `SIMRUN_MEMORY_*`, ...; never a credential such as `SIMRUN_TOKEN`), the workers, the times and the status (`running`, `done`, or `failed` with its error) |
| `spec.json` | the planned spec, which a resume runs as it is |
| `summary.json` | the summary of the run, e.g. the best cost of a fit or the failed points of a scan |
| `progress.jsonl` | the events of the run, see [Progress](#progress-and-provenance); the events of a unit go to `progress/unit-<index>.jsonl` |

The results of the kinds:

| kind | results |
| --- | --- |
| `simulate` | `index.html` and a directory per experiment |
| `fit` | `<opid>/runs/<opid>_run_<k>.json` (the repeats), `<opid>/optimization_result.json`, `<opid>/parameters.json` (the best run), `<opid>/fits.tsv`, with `identifiability` also `<opid>/identifiability.json` and `<opid>/fisher.json`, with `report` the report in `<opid>`, with `adopt` also `fits/<opid>` (the record of the adopted fit) and `adopted.json` |
| `identifiability` | `profiles/<pid>.json` (the profiles), `identifiability.json`, `fisher.json` and a copy of the records of the fit, `optimization_result.json` and `parameters.json` |
| `fisher` | `fisher.json` and a copy of the records of the fit |
| `report` | `report/`, with `adopted` a directory per adopted fit, `<fit id>/` |
| `adopt` | `fits/<fit id>` (the record of the fit) and `adopted.json` |
| `petab` | `petab/problem.yaml` and its tables |
| `scan` | `blocks/points-<start>-<stop>.nc` (the points of the units), `result.nc` (the `ScanResult`) |
| `sensitivity` | as `scan`, and `sensitivity.nc` (the `SensitivityResult`); of a sensitivity analysis over groups the files of its runner (see [Sensitivity analysis](sensitivity.md#sensitivity-analyses-of-a-project)): `parameters.tsv`, `parameters.tex` and a folder per analysis, `<analysis>/` or `<subfolder>/<analysis>/`, each written atomically by its unit with the hash of its plan in `.sensitivity-plan`, so a folder of another plan, e.g. of a smoke run, is computed again |
| `uncertainty` | `checkpoint/<scenario>/draws-<start>-<stop>.nc` (the draws of the units), `uncertainty/` (the `UncertaintyResult`) |
| `factory` | the files of the factory with the layout of the package, e.g. `models/toy.xml`; the files of the package are never written. The summary has the number of the compared outputs (`outputs`): the `factory_outputs` of the project (`declared` true), else, with a warning, the variants and every file the factory wrote which the package has (`declared` false); a rebuild with nothing to compare fails. `changed` names the compared outputs which differ from the ones of the package, an SBML file compared without the dates of its model history and an antimony file (`.ant`) without its `model created`/`model modified` lines |
| `action` | what the action writes; the summary is the mapping the action returns, followed by its `action` |
| `credibility` | `credibility/` (the record `credibility.json`, the tables and the report `index.html`, see [Credibility](credibility.md#running-and-reading-the-report)), with `lock` also `credibility.lock.json`; the summary has the verdict of every context of use (`verdicts`), the verified contexts (`verified`) and the state of the lock (`lock`) |
| `release` | `repository/`, a clone of the release repository with the build of the release, not committed; the summary has the version, the source commit, the framework and its version, the number of files of the manifest, the steps which ran, the outputs of the factory which differ from the source package (`changed`, part of the release), the skipped steps with their reason (`skipped`) and whether the build can be published (`publishable`) |

The input `fit` of a kind is the output of a `fit` run, whose records are in its directory `<opid>`, or a directory of records itself, e.g. the output of an `identifiability` or `fisher` run, which copy the records of their fit, so a report shows the fit and its analyses from one directory. Every file is written atomically: a reader sees the old or the new file and never a partial one. A run never writes into the project: even `adopt` writes `adopted.json` and the record `fits/<opid>` into the output, which you copy into the directory of the fit records of the project (`fits_dir`) and commit to adopt the fit. A report of a project has its context: the title of the project and the adopted fits the fit kept fixed, as its records name them.

The results are the files of the classes of `sbmlsim`, which read them back: `OptimizationResult.from_json` reads `optimization_result.json` of a fit, `IdentifiabilityResult.from_json` reads `identifiability.json`, `ScanResult.from_netcdf` reads `result.nc` of a scan, `SensitivityResult.from_netcdf` reads `sensitivity.nc` and `UncertaintyResult.from_netcdf` reads the directory `uncertainty/`. The fitted parameters of `parameters.json` are the input of a report or of the [identifiability](identifiability.md) tools.

## Units and merges

A run of a split kind computes units, each of its own files, and merges them. `python -m sbmlsim.run <kind>` does all of it in one process and its pool. The commands `plan`, `unit` and `merge` do it in separate processes, which may run at the same time, on one machine or on several machines which share the output directory. First write the scan of three doses as a spec:

```yaml title="dose.yaml"
project: examples.hctz_fitting.project:PROJECT
analysis: dose
```

`plan` completes the spec into what every unit computes alike and sets the number of units, `units`:

```bash
python -m sbmlsim.run plan scan dose.yaml -o dose-planned.json
```

```json title="dose-planned.json"
{
  "units": 3,
  "project": "examples.hctz_fitting.project:PROJECT",
  "analysis": "dose",
  "scan_hash": "cdbdb9ebc0c25da8b9a67857cd198d315c47ffb183a6330148b3ed4b4f3ce080"
}
```

The units are numbered from 0 to `units - 1`. Compute each one into the shared output, in any order, here one after another (`&` and `wait`, or one machine per unit, run them in parallel), then merge them:

```bash
for i in 0 1 2; do python -m sbmlsim.run unit scan dose-planned.json $i --out results/dose; done
python -m sbmlsim.run merge scan dose-planned.json --out results/dose
```

The merge writes `result.nc`, the same result as a whole run, `python -m sbmlsim.run scan dose.yaml --out results/dose-whole`. A unit which is missing makes the merge exit with 3 and name it, e.g. `1 of 3 units are missing: 1`, so compute that unit and merge again. A fit splits the same way, and so does a profile likelihood, whose `unit` and `merge` need the records of the fit as their input:

```bash
python -m sbmlsim.run plan fit fit.yaml -o fit-planned.json
for i in $(seq 0 7); do python -m sbmlsim.run unit fit fit-planned.json $i --out results/fit; done
python -m sbmlsim.run merge fit fit-planned.json --out results/fit

python -m sbmlsim.run plan identifiability profiles.yaml -o profiles-planned.json
python -m sbmlsim.run unit identifiability profiles-planned.json 0 --out results/profiles --input fit=results/fit
python -m sbmlsim.run merge identifiability profiles-planned.json --out results/profiles --input fit=results/fit
```

with one `unit` per parameter of `pids` in `profiles-planned.json`. The rules:

- A fit, a scan and a propagation of a given number of draws are cut into at most 64 units of consecutive items, `max(1, ceil(n / 64))` repeats, points or draws each; a profile likelihood has one unit per parameter, and the adaptive number of draws of a propagation is one unit, whose rounds are its checkpoints.
- The plan of a fit and of a propagation draws the seed when the spec has none, so every unit draws the same start values or the same points. The plan of a scan records the hash of the identity of the scan, which every unit and the merge compare with the scan they build: its record without the numbers a design draws and the references it reads (`scan_identity`: the method, the distributions and the options of a design with its seed and its number of points, the labels, the targets, and the whole record of the simulation and of a dimension of given values), so a unit on another platform than the plan, whose numerics differ in the last bits, builds the same scan. The plan builds the scan twice and compares the whole records; a scan which builds other points every time, e.g. a design without a seed, is refused.
- A whole run computes the same units on the workers of the run: the consecutive blocks of a scan or of a propagation of a given number of draws which have no file run in spans of as many blocks as the run has workers, a span one run of the simulator, so it is parallel whatever the size of a block, and the result of a span is cut into the files of its blocks, each the file its unit writes (a block whose messages of failed points the span does not keep, more than ten failed points before them, runs alone).
- The first unit writes `spec.json` into the output, and a unit or a merge of another plan is refused, so the units of two plans never mix. A unit writes no `run.json`, the merge does.
- A unit computed again, e.g. after its machine was lost, writes the same complete files again, and a unit whose files exist is not computed again; the merge names the units it misses and exits with 3.
- The merged result equals the result of the whole run bitwise with roadrunner, apart from the times and durations of the records: the blocks of a scan combine to the `ScanResult` of the whole scan (`ScanResult.combine`), the repeats of a fit to its `OptimizationResult`, the profiles to the `IdentifiabilityResult` (`IdentifiabilityResult.combine`) and the blocks of draws to the `UncertaintyResult`, every block taking its rows of the points of the design of all the draws, so a Latin hypercube stays one hypercube.

## Checkpoints and resume

The units are the checkpoints of a run: a repeat of a fit is written once it finished, a profile once both of its scans finished, a block of points or draws once it ran, in a whole run once its span of about `workers` blocks ran. A run which was interrupted, by Ctrl-C, a lost machine or a time limit, continues with `--resume` into the same output, which computes only what is missing:

```bash
python -m sbmlsim.run fit fit.yaml --out results/fit --resume
```

A resume runs the planned spec of `spec.json`, so a drawn seed stays, and is accepted only for the spec file the run was started with (and not for `spec.json`, the planned spec); its inputs are left out of the comparison, so they may be staged at another path. A run into an output which is not empty without `--resume` is refused, and so is a resume with another spec, as both would mix the results of two runs. A resume of a finished run runs it again, keeps every result and merges again; `--resume` into an empty output starts a run, and `progress.jsonl` collects the events of every attempt.

## Progress and provenance

`progress.jsonl` has one JSON line per step of the analysis, which a program outside of the run can follow, e.g. `tail -f`:

```json
{"time": "2026-11-06T13:03:03.374+00:00", "event": "scan.chunk", "points": 1, "done": 1, "total": 1}
{"time": "2026-11-06T13:03:03.490+00:00", "event": "scan.block", "analysis": "dose", "start": 0, "stop": 1, "points": 3}
```

| event | fields |
| --- | --- |
| `fit.repeat` | `opid`, `run`: a repeat of a fit finished and is stored, its file written right after the event |
| `identifiability.profile` | `opid`, `pid`: the profile of a parameter finished |
| `scan.chunk` | `points`, `done`, `total`: a chunk of points of a scan finished |
| `scan.block` | `analysis`, `start`, `stop`, `points`: the block of a unit of a scan finished |
| `sensitivity.analysis` | `analysis`, `folder`: an analysis of a sensitivity analysis over groups finished, its folder written |
| `propagation.block` | `scenario`, `start`, `stop`: a block of draws ran or was read |
| `experiment` | `sid`, `phase`, `activity`, `completed`, `total`: a phase of a simulation experiment |

The provenance of a run is the git revision of the root of the project, recorded in `run.json`. A run on a snapshot of the sources, e.g. on a cluster, has no checkout: `--provenance FILE`, or the file the environment variable `SIMRUN_PROVENANCE` names, is the JSON list of its sources (name, branch, commit, uncommitted changes), which `run.json`, the records of a fit (`FitProvenance.sources`) and the methods of its report record instead of the git revision.

## Exit codes

| code | meaning |
| --- | --- |
| 0 | done |
| 2 | a bad command line |
| 3 | the analysis failed, its traceback is logged; also a merge with missing units |
| 4 | a configuration error, its message on stderr: an invalid spec or input, an unknown kind, project or analysis, a planned spec which does not match what it resolves to, an output which is not empty without `--resume` or a resume of another spec |

An output which a run may not run into raises `sbmlsim.run.OutInUse`, a `ConfigurationError` with the output (`out`) and why (`reason`: `not_a_directory`, `not_empty` without `--resume`, `no_record` and `unreadable` for a resume without a readable `run.json`, `other_run` for the record of another kind or spec). `sbmlsim.run.check_out(kind, spec, out, resume=...)` makes these checks and the validation of the spec without running and without writing anything, so a wrapper refuses before it starts:

```py
from sbmlsim import run

try:
    run.check_out("release", {"project": "my_model:PROJECT", "version": "0.8.0"}, out)
except run.OutInUse as err:
    print(err.reason, err.out)
```

## Verify

`python -m sbmlsim.run verify <subcommand>` checks that a change of a model changes only what it is meant to change: a refactoring, an update of sbmlsim or a refresh of the study data should leave the simulations of the model as they were. It is a family of local commands like `plan` and `list`, not a kind: it compares the results of checkouts on this machine, writes only where it is told to (`--out`) and has no units, no merge and no simrun task type. The result of a command goes to stdout, the progress and the problems are logged on stderr. The exit code is 0 when nothing differs, 1 when a difference is found and 2 for a command that cannot be run as it was asked (an unknown project or experiment, a bad option, a missing directory).

```bash
git worktree add --detach ../base develop
# the experiments simulate the models of the project, which are generated:
# create them with the factory of each checkout right before its dump
python -m sbmlsim.run verify dump --root ../base --project my_models.hctz:PROJECT --out base.pkl --atol 1e-16 --rtol 1e-12 --grid 1000
python -m sbmlsim.run verify dump --root . --project my_models.hctz:PROJECT --out new.pkl --order base.pkl --atol 1e-16 --rtol 1e-12 --grid 1000
python -m sbmlsim.run verify compare base.pkl new.pkl
```

| subcommand | what it does |
| --- | --- |
| `dump --project REF --out FILE` | runs the experiments of the project (or the named ones, `--all` adds those defined below its package but not in its experiments) serially and stores the results, datasets, fit mappings, figures and the variant of every task in a pickle; `--atol`/`--rtol` take a number or `model` for the tolerance of the project, `--grid N` outputs every simulation at `N + 1` equidistant times and at its changes instead of the steps of the integrator, `--order FILE` runs in the order of another dump, `--skip A,B` leaves experiments out, `--reduced-selections` simulates only the selections the experiments need |
| `compare OLD NEW` | compares two dumps: the simulations on the matched rows of their output times (a row at a time which only one dump has is compared with the other dump interpolated at its time, relative to the scale of the array, an array whose interpolation cannot resolve the `--threshold` is reported as unresolved, not as equal), everything else exactly; `--renames FILE` applies the expected renames and differences of a change to the old dump first |
| `ranges DUMP` | lists the fit mappings whose data lie outside of the simulated range |
| `fits --project REF --out DIR` | starts a minimal fit of every fit definition |
| `definitions --project REF --out FILE` | writes the fit definitions as JSON (settings, parameters, mappings), to compare two checkouts with `diff`; the checkout is `<root>` |
| `settings --project REF --out FILE` | writes the settings a run of the experiments hands to sbmlsim as JSON, without simulating |
| `png OLD NEW` | compares the PNG figures of two directories pixel by pixel; `--sids` pairs an old figure with the figure of the SId of its file name |
| `sbml OLD NEW` | compares SBML models (files or directories) by their elements, ignoring the order and the annotations |
| `scorecard RECORDS...` | reads the credibility records of many models into one table, compares it with `--baseline` and pools a qualification with the evidence extension named by `--pool` (`--field`); the extensions are those of `--project` and `--extensions MODULE:ATTRIBUTE` |

`--project` is the reference `module:attribute` of the project and `--root` the directory put first on the python path, e.g. a checkout (the project must be imported from below it). The variant of the model a task runs on is stored with every task: the variant `SimulationExperiment.variant_of(simulation)` names, which a framework that builds a model per group of subjects implements, else the variant of the project whose file is the source of the model of the task; a study that moves to another variant is a difference of its own. The renames file has the parts `metadata` (`fields`, `members` and `added`, the renamed metadata of the fit mappings), `strings` and `ids` (renamed texts and ids), `sids` (`true` when the ids became SIds), `figure_fields` (fields of a figure left out of both dumps) and `expected` (patterns of the keys per part of the dump, `results`, `datasets`, `mappings`, `figures`, `ranges` and `builds`, whose differences are expected: they are printed and not counted, and an expected difference that does not occur is counted).

## CPU quota and threads

The number of processes of a run is the number of CPUs the process may use, bounded by the CPU quota of its cgroup, so a run in a container with a quota of 4 CPUs on a host of 64 starts 4 workers and not 64. With cgroup v2 the quota is the smallest one of the cgroup of the process and its parents (`cpu.max`), with cgroup v1 the one quota of the cgroup as the container sees it (`cpu.cfs_quota_us` over `cpu.cfs_period_us`), rounded down and at least 1; the pools of `sbmlsim` (`Simulator`, a fit, the experiments) bound their default size the same way.

A run uses its CPUs through the processes of its pools, so the numerical libraries run with one thread each: the command sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 1 unless they are set, starting itself once more with them, and draws its figures with the matplotlib backend `Agg` (`MPLBACKEND` unless set).

## Profiling

`--profile` runs a command which writes into `--out` (`run`, `unit`, `merge`) under the statistical sampling profiler of python 3.15 (`profiling.sampling`, PEP 799), which samples the stack of the command from a process of its own, so the command runs at nearly its normal speed:

```bash
python -m sbmlsim.run simulate simulate.yaml --out results/simulate --profile
```

After the command the profile is written into `profiling/<command>_<kind>_<time>/` of the output (of a unit `profiling/unit_<kind>_<index>_<time>/`, so the units which compute into one output at once do not share a folder): `flamegraph.html` (the interactive flame graph), `heatmap/` (the source lines with their samples), `profile.pstats` (for `pstats` and its viewers), `summary.txt` (the functions with the most samples) and `profile.bin` (the samples, against which a later profile is compared, `python -m profiling.sampling replay --diff-flamegraph profile.bin`). The command exits with its own code, which it records for the profiler (the profiler exits with 0 whatever the command did), 1 if it recorded none; a failure of the profiler or of the conversion of the profile is logged and does not change the code. With an older python `--profile` is a bad command line (exit code 2) and the command does not run.

`sbmlsim.run.profiling.run_profiled(arguments, folder, module=...)` profiles any entry point which runs as `python -m <module>`, e.g. the command line of a framework of its own, which calls `record_exit_code(code)` before it exits. The profiler runs with one change: python 3.15.2 reads now and then a name of a frame while it changes, which broke the writer of the profile and ended the command; such a sample is left out and their number reported on stderr.

## Projects

A project, `sbmlsim.project.Project`, describes a model package once: its model variants, factory, experiments, fits, scans, uncertainty and sensitivity analyses and the settings of its simulator, figures, checks and release. It is an attribute of a module of the package, which the run layer imports, and every spec names it as `project: module:attribute`. The fields, the references and the project of the HCTZ example are described on the page [Projects](project.md).

## Credibility

The kind `credibility` assesses the contexts of use of a model declared in `Project.credibility` against its adopted fits into `out/credibility/`, with the options `verification`, `battery`, `cross_simulator`, `lock`, `uncertainty`, `variability` and `source_commit` and the inputs `cross_validation` and `record`; the declaration, the evidence, the verdicts and the options are described on the page [Credibility](credibility.md#running-and-reading-the-report).

```bash
python -m sbmlsim.run credibility credibility.yaml --out results/credibility
python -m sbmlsim.run credibility credibility.yaml --out results/report --input record=results/credibility
```

## Release

The kind `release` builds the release of a project as a repository of its own into `out/repository`; it commits nothing, and `python -m sbmlsim.release publish` publishes the build. The build, the release metadata (`Project.release`), the review and the publishing are described on the page [Release](release.md).

```yaml title="release.yaml"
project: my_model:PROJECT
version: 0.8.0
```

```bash
python -m sbmlsim.run release release.yaml --out results/release
```

## Checks of a project

`sbmlsim.project_checks` has the checks of a model package as functions, which its tests call (`check_model`, `check_registry`, `initialize_experiments`, `run_all_experiments`, `check_battery`, `start_fits`), see [Projects](project.md#checks-of-a-project).

## simrun

[simrun](https://github.com/matthiaskoenig/simrun) runs the tasks of a workflow on a cluster or, with `--local`, on this machine; [Workflows with simrun](simrun.md) runs the analyses of the tutorial as a workflow and shows how to convert an analysis. `sbmlsim` registers its kinds but `action` as the task types `sbmlsim.<kind>` of simrun, `sbmlsim.fit`, `sbmlsim.identifiability`, `sbmlsim.scan`, `sbmlsim.adopt`, `sbmlsim.factory`, `sbmlsim.credibility`, `sbmlsim.release` and so on (`verify` is a command and no kind, so it has no task type until a workflow needs one); the extra `simrun` installs simrun, which needs python 3.14:

```bash
pip install "sbmlsim[simrun]"
```

The settings of a task are the spec of its kind. A split kind (`fit`, `identifiability`, `scan`, `sensitivity`, `uncertainty`) is split by simrun: its workers compute the units and its merge merges them. The other kinds run as the command `python -m sbmlsim.run <kind> {settings} --out {out}/out --summary {out}/summary.json --resume --input={input}`, whose `--resume` lets a retry after an eviction continue the run it finds. A task writes the output of its run into its output port `out`, which a task downstream reads as the input `<task>.out`, and its summary into `summary.json` of the task.

A workflow which fits the HCTZ problem in units, profiles its parameters and writes the report of both:

```yaml title="hctz-fit.yaml"
name: hctz-fit
tasks:
  - id: fit
    type: sbmlsim.fit
    settings:
      project: examples.hctz_fitting.project:PROJECT
      subset: PK
      repeats: 64
  - id: identifiability
    type: sbmlsim.identifiability
    settings:
      project: examples.hctz_fitting.project:PROJECT
      subset: PK
    inputs:
      fit: fit.out
  - id: report
    type: sbmlsim.report
    settings:
      project: examples.hctz_fitting.project:PROJECT
      subset: PK
    inputs:
      fit: identifiability.out
```

```bash
simrun submit hctz-fit.yaml --source sbmlsim=~/git/sbmlsim
```

The HCTZ project lives in the repository of `sbmlsim`, which is the only source here. A project of a model package outside of `sbmlsim` adds the source of the model first and then the one of `sbmlsim`, as simrun orders them, e.g. `simrun submit fit.yaml --source model=~/git/my_model --source sbmlsim=~/git/sbmlsim`.

The fit is 64 units of one repeat, the profile likelihood one unit per parameter, and the report reads the fit and its profiles from the output of the identifiability analysis.

- **The plan runs in the client.** `simrun submit` plans every task whose type is installed where it runs, before it takes the snapshots, and submits the planned settings: the seed of the fit, the units, the parameters of the profiles. The client therefore needs `sbmlsim[simrun]` and the model package, so that it imports the project; a task submitted without its plan has no `units` and fails in the service.
- **The project is imported in the pods.** The pods install the snapshots editable, which installs the package of a source and not the rest of its repository: a model package is found, while the HCTZ example in `examples/` needs the root of the repository on `PYTHONPATH`.
- **Provenance.** simrun writes the provenance of the snapshots of a run and names it in `SIMRUN_PROVENANCE`, which `run.json` and the records of a fit record, see [Provenance](#progress-and-provenance).
- **Development.** The development environment is python 3.14 (`.python-version`), so `uv sync --extra dev` installs simrun and `uv run ty check` checks `sbmlsim.simrun` against it; the tests of the plugin skip without simrun. The tox environments, which resolve from `pyproject.toml`, do not install it.
