# Analyses

`python -m sbmlsim.run` runs the analyses of a model package from the command line: its simulation experiments, fits, profile likelihoods, scans, sensitivity and uncertainty analyses, the build of its models and the actions of its own. It runs an analysis of a [project](#projects) from a spec, a small YAML or JSON file of its settings, into an output directory. The same analysis runs whole on one machine or in units on several machines, which merge to the result of the whole run, so a fit of 64 repeats or a scan of 10000 points can be spread over a cluster with [simrun](#simrun) and give the same numbers as on a laptop.

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
```

The first form runs an analysis whole (its command is `run`, which may be left out). `plan`, `unit` and `merge` run it in [units](#units-and-merges), `list` lists the analyses of a project as a table or, with `--json`, as JSON; with `--experiments` it also lists the groups of the simulation experiments and every experiment (kind `simulate`, which runs them by these names), which imports the experiments. `--profile`, anywhere in the command line, runs a command which writes into `--out` under the sampling profiler of python 3.15, see [Profiling](#profiling).

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
| `sensitivity` | a scan with sensitivity indices, and the indices | blocks of points | |
| `uncertainty` | the propagation of an uncertainty analysis | blocks of draws | |
| `factory` | the models built by the factory of the project, compared with the ones of the package | whole | |
| `action` | an action of the project, an analysis of its own | whole | |

A spec forbids unknown fields. The fit kinds name a fit problem, the other kinds an analysis of the project:

| field | kinds | meaning |
| --- | --- | --- |
| `project` | all | the project, `module:attribute` |
| `subset` | `fit`, `identifiability`, `fisher`, `report`, `petab` | the fit problem, a key of the fits of the project |
| `analysis` | `scan`, `sensitivity`, `uncertainty` | the scan or the uncertainty analysis, a key of the project |
| `fit` | `identifiability`, `fisher`, `report`, `petab`, `adopt` | the records of a fit, the input `fit` |
| `name` | `action` | the action, a key of the actions of the project |
| `arguments` | `action` | the arguments of the action, JSON values by the names of the keywords of its function, `{}` without; the plan refuses arguments the function does not take. They are passed as they are: a relative path among them is relative to the current directory of the run, not to the spec file |
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
| `scan`, `sensitivity` | `scan_hash` | set by the plan | the hash of the scan the plan built |
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
| `sensitivity` | as `scan`, and `sensitivity.nc` (the `SensitivityResult`) |
| `uncertainty` | `checkpoint/<scenario>/draws-<start>-<stop>.nc` (the draws of the units), `uncertainty/` (the `UncertaintyResult`) |
| `factory` | the files of the factory with the layout of the package, e.g. `models/toy.xml`; the files of the package are never written. The summary has the number of the compared outputs (`outputs`): the `factory_outputs` of the project (`declared` true), else, with a warning, the variants and every file the factory wrote which the package has (`declared` false); a rebuild with nothing to compare fails. `changed` names the compared outputs which differ from the ones of the package, an SBML file compared without the dates of its model history |
| `action` | what the action writes; the summary is the mapping the action returns, followed by its `action` |

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

A project, `sbmlsim.project.Project`, describes a model package once: its models, its analyses and their settings. It is an attribute of a module of the package, which the run layer imports. Nothing of it runs when it is created: the experiments, the scans, the observables and the uncertainty models are functions which build their objects when an analysis runs, so every unit of a run builds them again, and each must build the same object every time, e.g. a design with a fixed seed.

The experiments, the fits, the factory and the credibility declaration are objects or references `"module:attribute"` to them, resolved on their first use and kept. The experiments of a package import its project for its paths, so the module of the project cannot import them, and with references importing the project imports none of them, which keeps a package of many models cheap to import. A reference which starts with a dot is relative to `package`:

```py
from pathlib import Path

from sbmlsim.project import FigureOutputs, Project, SimulatorSettings

PROJECT = Project(
    name="toy",
    root=Path(__file__).parent,
    package=__name__,
    variants={"default": Path("models/toy.xml"), "slow": Path("models/toy_slow.xml")},
    factory=".factory:create_toy_models",
    factory_outputs=[Path("models/toy.xml"), Path("models/toy_slow.xml")],
    experiments=".experiments:EXPERIMENTS",
    fits=".fitting:FIT_DEFINITIONS",
    fits_dir=Path("fits"),
    simulator=SimulatorSettings(relative_tolerance=1e-10, jit_optimization="O0"),
    figures=FigureOutputs(formats=("svg", "png"), savefig_dpi=150),
)
```

A relative path of a project (`variants`, `factory_dir`, `base_path`, `data_path`, `fits_dir`, `adopted`, the model of a scan) is relative to its `root` and stored absolute, so `project.replace(root=...)`, which validates a copy, does not move the paths it has; `replace` and not pydantic's `model_copy`, which validates nothing, changes a field. The module `sbmlsim.project` imports neither the experiments, the fits nor roadrunner, so a model package which defines its project imports fast. The outputs of a run go where the run is told: `PROJECT.run_experiments(["studies"], "results")` writes into `results` of the current directory.

| field | meaning |
| --- | --- |
| `name`, `title`, `description` | the name of the project, its title in the reports (the name without) and what it is |
| `root` | the directory of the model package, which relative paths are relative to |
| `package` | the dotted name of the model package, which relative references are relative to |
| `variants` | the SBML files of the model by the name of its variant, the first the default: `model_path` (the default variant, a property), `variant_path(variant)` and `load_roadrunner(variant)`, which compiles at the level of the simulator settings |
| `factory`, `factory_dir`, `factory_outputs` | the function which builds the models, called with the directory it writes into, the directory it writes into in the package (relative to `root`, the directory of the first variant without) and the files it writes, relative to `root`. `create_models()` builds in place, `create_models(out)` rebuilds into `out` with the layout of the package (into `out / factory_dir.relative_to(root)`), where `factory_paths(out)` are the outputs to compare with `factory_paths()` and `changed_outputs(out)` the ones which differ, the kind `factory` |
| `experiments` | the simulation experiment classes, the kind `simulate`: a sequence of classes, groups of them by name (the group `all` has every experiment, the groups together without it) or a function returning either; `load_experiments()` gives the groups, `experiment_classes(names)` the classes of names of groups or classes, `run_experiments(experiments, out)` runs them with the settings of the project (`n_workers` the processes of the run, `scan_workers` the ones of the scans of the simulator, `n_workers` without; `n_workers=None, scan_workers=1` runs the experiments in a pool and the scans of each serially) and raises an `ExperimentRunError` after the report when one failed, with `uncertainty` over draws of the fitted parameters and with `collect_figures` the static figures copied into `_figures` |
| `base_path`, `data_path` | the base path and the datasets of the experiments |
| `fits` | the fit problems by their subset, the `FitDefinition`s of [Parameter fitting](fitting.md); `load_fit_definitions()` gives them at the optimization level of the simulator settings |
| `fits_dir` | the directory of the records of the adopted fits, one directory per fit id, `adopted_record(fit_id)`; `fit_records()` reads them, the fitted values with their uncertainty which a factory gives its parameters, `fit_records().parameter("k", unit=U.per_min)` (`sbmlsim.fit.records.AdoptedRecords`) |
| `adopted` | the file of the adopted fits, `{"fits": {"<subset>": "<fit id>"}}`, `adopted.json` in `fits_dir` without |
| `credibility` | the declaration of the contexts of use of the model, `load_credibility()` |
| `scans` | the scans by their name, `ScanAnalysis`: the model, a function building the `Scan`, the observables, `time`, `keep`, `on_error` and, for a sensitivity analysis, `indices` (`local`, `sobol`, `fast` or `morris`) with `index_options` |
| `uncertainties` | the uncertainty analyses by their name, `UncertaintyAnalysis`: the model, functions building the simulation or scenarios, the `UncertaintyModel` and the observables, and the arguments of [`propagate`](uncertainty.md) (`outputs`, `n` or the adaptive `rel_tol`, `n_start` and `n_max`, `design`, `seed`, `keep`, `time`, `on_error`) |
| `actions` | the actions of the project by their name, analyses of its own which the kind `action` runs, `ProjectAction`: the function or a reference to it and a `description`; a function or a reference alone is an action without a description. The function is called as `function(project, out, *, workers, **arguments)` (`ActionFunction`) and returns its summary, a mapping of JSON values, or `None` |
| `simulator` | the settings of the simulator of the analyses, `SimulatorSettings`: the engine, `absolute_tolerance`, `relative_tolerance`, `jit_optimization` (the level LLJIT compiles the models of the experiments, fits and tests with) and the other settings of the integrator, `integrator`; the workers belong to a run |
| `figure_settings` | the rendering of the figures of the experiments, a `FigureSettings` of the plots |
| `figures` | the files of the figures, `FigureOutputs`: the `formats`, the resolution of the static images `savefig_dpi` and other parameters of matplotlib, `rc` |
| `checks` | the known exceptions of the [checks](#checks-of-a-project) of the model, `CheckSettings`, each with its reason, and the module of its experiments, `experiment_modules` (`experiments` below the package) |
| `release` | the release metadata, `sbmlsim.release.settings.ReleaseSettings`, `None` for a project which is not released |

The project of the HCTZ example, [`examples/hctz_fitting/project.py`](https://github.com/matthiaskoenig/sbmlsim/blob/develop/examples/hctz_fitting/project.py), is the template of the project of a model package: copy it into a module of your package and replace the experiments, the fits and the analyses with your own. It has the three simulation experiments of the studies, the fit problems `PK` (all pharmacokinetics data) and `IV` (the intravenous data of Beermann 1976), the scans `dose` (three oral doses on a grid of times) and `sobol` (a Sobol design of the dissolution and the renal excretion with its Sobol indices) and the uncertainty analysis `cl` (the AUC under an uncertain renal excretion, 32 draws). A project must be importable wherever the run layer runs: a model package installs it with its package, while the HCTZ example lives in `examples/` of the repository, which is not installed, and needs the root of the repository on the path, `python -m` from the root or `PYTHONPATH`.

An action is an analysis of the package which none of the kinds is, e.g. a table of the studies of its data:

```py
def write_rates(
    project: Project, out: Path, /, *, workers: int, factor: float = 1.0
) -> dict[str, int]:
    ...  # writes rates.tsv into out
    return {"rows": len(project.variants)}


PROJECT = Project(
    ...,
    actions={
        "rates": ProjectAction(
            function=".actions:write_rates", description="the table of the rates"
        )
    },
)
```

```yaml title="rates.yaml"
project: my_model:PROJECT
name: rates
arguments:
  factor: 2
```

```bash
python -m sbmlsim.run action rates.yaml --out results/rates
python -m sbmlsim.run factory factory.yaml --out results/models
```

where `factory.yaml` names the `project` alone; its summary lists the outputs of the factory which differ from the committed ones.

## Checks of a project

`sbmlsim.project_checks` has the checks of a model package as functions, which its tests call, so a package keeps them wherever it lives; each raises an `AssertionError` naming every problem:

```py
from sbmlsim import project_checks


def test_the_project(tmp_path: Path) -> None:
    project_checks.check_model(PROJECT)
    project_checks.check_registry(PROJECT)
    project_checks.initialize_experiments(PROJECT)
    project_checks.run_all_experiments(PROJECT, tmp_path / "experiments")
    project_checks.start_fits(PROJECT, tmp_path / "fits")
```

| function | checks |
| --- | --- |
| `check_model` | the project belongs to its package (the package imports and has the project, its directory is `root`, the name of the project is the last component of the package or its end `_<name>`), the variants, the factory directory, the fit records and the adopted fits lie below `root`, the variants and the data exist, every reference resolves, every action takes the arguments of its call (`function(project, out, *, workers)`), the experiments load, the fits of a reference are not empty, and every variant has the values and the distributions of the adopted fits |
| `check_registry` | every module of the experiments imports (`experiment_modules`), every experiment it defines is in the group `all` once, and the known exceptions still hold |
| `initialize_experiments` | every experiment initializes with its data, figures and fit mappings |
| `run_all_experiments` | every experiment runs and writes its report, together into `all` or with `CheckSettings.one_at_a_time` one at a time |
| `check_battery` | a battery of checks of the models, which a framework runs (`run(project)`), against the known failures of `CheckSettings.battery` |
| `start_fits` | every fit problem starts, one repeat of a few evaluations |

The known exceptions are the `checks` of the project, `CheckSettings`: `not_registered` (experiments defined but not in `all` on purpose), `not_importable` (modules of the experiments which do not import in the test environment), `one_at_a_time` (why the experiments run one at a time) and `battery` (the checks of a battery which fail), each with its reason; an exception which no longer holds is a problem as well.

## simrun

[simrun](https://github.com/matthiaskoenig/simrun) runs the tasks of a workflow on a cluster. `sbmlsim` registers its kinds but `action` as the task types `sbmlsim.<kind>` of simrun, `sbmlsim.fit`, `sbmlsim.identifiability`, `sbmlsim.scan`, `sbmlsim.adopt`, `sbmlsim.factory` and so on; the extra `simrun` installs simrun, which needs python 3.14:

```bash
pip install "sbmlsim[simrun]"
```

Until simrun is released on PyPI, install it from its repository first, `pip install git+https://github.com/matthiaskoenig/simrun.git`.

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
- **Development.** The development environment is python 3.14 (`.python-version`), so `uv sync --extra dev` installs simrun and `uv run ty check` checks `sbmlsim.simrun` against it; the tests of the plugin skip without simrun. Until simrun is released on PyPI the lock takes it from git, and the tox environments, which resolve from `pyproject.toml`, do not install it.
