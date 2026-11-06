# Analyses

`python -m sbmlsim.run` runs the analyses of a model package from the command line: its simulation experiments, fits, profile likelihoods, scans, sensitivity and uncertainty analyses. It runs an analysis of a [project](#projects) from a spec, a small YAML or JSON file of its settings, into an output directory. The same analysis runs whole on one machine or in units on several machines, which merge to the result of the whole run, so a fit of 64 repeats or a scan of 10000 points can be spread over a cluster with [simrun](#simrun) and give the same numbers as on a laptop.

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
python -m sbmlsim.run list <project> [--json]
```

The first form runs an analysis whole (its command is `run`, which may be left out). `plan`, `unit` and `merge` run it in [units](#units-and-merges), `list` lists the analyses of a project as a table or, with `--json`, as JSON.

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
| `scan` | a scan of the project | blocks of points | |
| `sensitivity` | a scan with sensitivity indices, and the indices | blocks of points | |
| `uncertainty` | the propagation of an uncertainty analysis | blocks of draws | |

A spec forbids unknown fields. The fit kinds name a fit problem, the other kinds an analysis of the project:

| field | kinds | meaning |
| --- | --- | --- |
| `project` | all | the project, `module:attribute` |
| `subset` | `fit`, `identifiability`, `fisher`, `report`, `petab` | the fit problem, a key of the fits of the project |
| `analysis` | `scan`, `sensitivity`, `uncertainty` | the scan or the uncertainty analysis, a key of the project |
| `fit` | `identifiability`, `fisher`, `report`, `petab` | the records of a fit, the input `fit` |
| `units` | the split kinds | the number of units, set by the plan |

### fit

| field | default | meaning |
| --- | --- | --- |
| `opid` | the subset | the id of the fit, the directory of its records in the output |
| `repeats` | 4 | the number of optimization runs |
| `algorithm` | `LSQ` | least squares `LSQ` or differential evolution `DE` |
| `seed` | drawn by the plan | the seed of the start values and of the runs |
| `studies` | all | the studies of the problem to fit |
| `timeout` | none | the seconds a single run may take |
| `optimizer` | `{}` | further arguments of the optimizer, e.g. `{"max_nfev": 50}`; a setting of the spec, e.g. `seed`, is refused here |
| `adopt` | `false` | write `adopted.json`, the adopted fits of the project with this fit for its subset |

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
| `petab` | `required_extension` | `true` | mark the extension `sbmlsim` as required |

### simulate, scan, sensitivity and uncertainty

| kind | field | default | meaning |
| --- | --- | --- | --- |
| `simulate` | `experiments` | every experiment | the names of the classes of the experiments to run |
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
| `run.json` | the record of the run: the kind, the hash of the spec, the versions of the packages and of the project, python, the host, the provenance of the sources or else the git revision of the project, the variables `SIMRUN_*` of the environment, the workers, the times and the status (`running`, `done`, or `failed` with its error) |
| `spec.json` | the planned spec, which a resume runs as it is |
| `summary.json` | the summary of the run, e.g. the best cost of a fit or the failed points of a scan |
| `progress.jsonl` | the events of the run, see [Progress](#progress-and-provenance); the events of a unit go to `progress/unit-<index>.jsonl` |

The results of the kinds:

| kind | results |
| --- | --- |
| `simulate` | `index.html` and a directory per experiment |
| `fit` | `<opid>/runs/<opid>_run_<k>.json` (the repeats), `<opid>/optimization_result.json`, `<opid>/parameters.json` (the best run), `<opid>/fits.tsv`, with `adopt` also `adopted.json` |
| `identifiability` | `profiles/<pid>.json` (the profiles), `identifiability.json`, `fisher.json` and a copy of the records of the fit, `optimization_result.json` and `parameters.json` |
| `fisher` | `fisher.json` and a copy of the records of the fit |
| `report` | `report/` |
| `petab` | `petab/problem.yaml` and its tables |
| `scan` | `blocks/points-<start>-<stop>.nc` (the points of the units), `result.nc` (the `ScanResult`) |
| `sensitivity` | as `scan`, and `sensitivity.nc` (the `SensitivityResult`) |
| `uncertainty` | `checkpoint/<scenario>/draws-<start>-<stop>.nc` (the draws of the units), `uncertainty/` (the `UncertaintyResult`) |

The input `fit` of a kind is the output of a `fit` run, whose records are in its directory `<opid>`, or a directory of records itself, e.g. the output of an `identifiability` or `fisher` run, which copy the records of their fit, so a report shows the fit and its analyses from one directory. Every file is written atomically: a reader sees the old or the new file and never a partial one. A run never writes into the project: even `adopt` writes `adopted.json` into the output, which you copy into the project to adopt the fit.

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
- The plan of a fit and of a propagation draws the seed when the spec has none, so every unit draws the same start values or the same points. The plan of a scan records the hash of the scan, which every unit and the merge compare with the scan they build; a scan which builds other points every time, e.g. a design without a seed, is refused by the plan.
- The first unit writes `spec.json` into the output, and a unit or a merge of another plan is refused, so the units of two plans never mix. A unit writes no `run.json`, the merge does.
- A unit computed again, e.g. after its machine was lost, writes the same complete files again, and a unit whose files exist is not computed again; the merge names the units it misses and exits with 3.
- The merged result equals the result of the whole run bitwise with roadrunner, apart from the times and durations of the records: the blocks of a scan combine to the `ScanResult` of the whole scan (`ScanResult.combine`), the repeats of a fit to its `OptimizationResult`, the profiles to the `IdentifiabilityResult` (`IdentifiabilityResult.combine`) and the blocks of draws to the `UncertaintyResult`, every block taking its rows of the points of the design of all the draws, so a Latin hypercube stays one hypercube.

## Checkpoints and resume

The units are the checkpoints of a run: a repeat of a fit is written once it finished, a profile once both of its scans finished, a block of points or draws once it ran. A run which was interrupted, by Ctrl-C, a lost machine or a time limit, continues with `--resume` into the same output, which computes only what is missing:

```bash
python -m sbmlsim.run fit fit.yaml --out results/fit --resume
```

A resume runs the planned spec of `spec.json`, so a drawn seed stays, and is accepted only for the spec the run was started with; its inputs are left out of the comparison, so they may be staged at another path. A run into an output which is not empty without `--resume` is refused, and so is a resume with another spec, as both would mix the results of two runs. A resume of a finished run runs it again, keeps every result and merges again; `--resume` into an empty output starts a run, and `progress.jsonl` collects the events of every attempt.

## Progress and provenance

`progress.jsonl` has one JSON line per step of the analysis, which a program outside of the run can follow, e.g. `tail -f`:

```json
{"time": "2026-11-06T13:03:03.374+00:00", "event": "scan.chunk", "points": 1, "done": 1, "total": 1}
{"time": "2026-11-06T13:03:03.490+00:00", "event": "scan.block", "analysis": "dose", "start": 0, "stop": 1, "points": 3}
```

| event | fields |
| --- | --- |
| `fit.repeat` | `opid`, `run`: a repeat of a fit finished and is stored, its file written right after the event |
| `profile` | `opid`, `pid`: the profile of a parameter finished |
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

## Projects

A project, `sbmlsim.project.Project`, names the analyses of a model package: it is an attribute of a module of the package, which the run layer imports. Nothing of it runs when it is created: the experiments, the scans, the observables and the uncertainty models are functions which build their objects when an analysis runs, so every unit of a run builds them again, and each must build the same object every time, e.g. a design with a fixed seed.

| field | meaning |
| --- | --- |
| `name`, `description` | the name of the project and what it is |
| `root` | the directory of the model package; a relative path of a model of an analysis is relative to it |
| `experiments` | a function returning the simulation experiment classes, the kind `simulate` |
| `base_path`, `data_path` | the base path and the datasets of the experiments |
| `fits` | the fit problems by their subset, the `FitDefinition`s of [Parameter fitting](fitting.md) |
| `adopted` | the file of the adopted fits, `{"fits": {"<subset>": "<fit id>"}}` |
| `scans` | the scans by their name, `ScanAnalysis`: the model, a function building the `Scan`, the observables, `time`, `keep`, `on_error` and, for a sensitivity analysis, `indices` (`local`, `sobol`, `fast` or `morris`) with `index_options` |
| `uncertainties` | the uncertainty analyses by their name, `UncertaintyAnalysis`: the model, functions building the simulation or scenarios, the `UncertaintyModel` and the observables, and the arguments of [`propagate`](uncertainty.md) (`outputs`, `n` or the adaptive `rel_tol`, `n_start` and `n_max`, `design`, `seed`, `keep`, `time`, `on_error`) |
| `simulator` | the engine and the integrator settings of the analyses |
| `figure_settings` | the settings of the figures of the experiments |

The project of the HCTZ example, [`examples/hctz_fitting/project.py`](https://github.com/matthiaskoenig/sbmlsim/blob/develop/examples/hctz_fitting/project.py), is the template of the project of a model package: copy it into a module of your package and replace the experiments, the fits and the analyses with your own. It has the three simulation experiments of the studies, the fit problems `PK` (all pharmacokinetics data) and `IV` (the intravenous data of Beermann 1976), the scans `dose` (three oral doses on a grid of times) and `sobol` (a Sobol design of the dissolution and the renal excretion with its Sobol indices) and the uncertainty analysis `cl` (the AUC under an uncertain renal excretion, 32 draws). A project must be importable wherever the run layer runs: a model package installs it with its package, while the HCTZ example lives in `examples/` of the repository, which is not installed, and needs the root of the repository on the path, `python -m` from the root or `PYTHONPATH`.

## simrun

[simrun](https://github.com/matthiaskoenig/simrun) runs the tasks of a workflow on a cluster. `sbmlsim` registers its kinds as the task types `sbmlsim.<kind>` of simrun, `sbmlsim.fit`, `sbmlsim.identifiability`, `sbmlsim.scan` and so on; the extra `simrun` installs simrun, which needs python 3.14:

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
