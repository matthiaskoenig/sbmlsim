# Projects

A project describes a model package once, so that the same description serves the [run layer](run.md) (`python -m sbmlsim.run`, the task types of [simrun](run.md#simrun)), the [release](release.md), the [credibility](credibility.md) assessment, the checks of the tests and the tools which compare checkouts. The model package defines it; sbmlsim and the frameworks on top of it read it.

A project, `sbmlsim.project.Project`, holds the models of a package, its analyses and their settings. It is an attribute of a module of the package, which the run layer imports. Nothing of it runs when it is created: the experiments, the scans, the observables and the uncertainty models are functions which build their objects when an analysis runs, so every unit of a run builds them again, and each must build the same object every time, e.g. a design with a fixed seed.

The experiments, the fits, the factory, the credibility declaration and the sensitivity analyses are objects or references `"module:attribute"` to them, resolved on their first use and kept. The experiments of a package import its project for its paths, so the module of the project cannot import them, and with references importing the project imports none of them, which keeps a package of many models cheap to import. A reference which starts with a dot is relative to `package`:

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
| `credibility` | the declaration of the contexts of use of the model, `load_credibility()`, which checks its domains against the metadata of the project, see [Credibility](credibility.md) |
| `credibility_settings` | the domain of the credibility, `CredibilitySettings`: the metadata of the fit mappings (`metadata_type`, a class or a reference, `MappingMetaData` without), the fields of it a domain can claim (`domain_fields`, every enum field without) the evidence extensions (`extensions`, objects or references), the battery hook (`battery`, an object or a reference) and the settings of the battery over the ones of the hook (`battery_settings`); `domain_fields()`, `evidence_extensions()` and `battery_hook()` resolve them |
| `scans` | the scans by their name, `ScanAnalysis`: the model, a function building the `Scan`, the observables, `time`, `keep`, `on_error` and, for a sensitivity analysis, `indices` (`local`, `sobol`, `fast` or `morris`) with `index_options` |
| `uncertainties` | the uncertainty analyses by their name, `UncertaintyAnalysis`: the model, functions building the simulation or scenarios, the `UncertaintyModel` and the observables, and the arguments of [`propagate`](uncertainty.md) (`outputs`, `n` or the adaptive `rel_tol`, `n_start` and `n_max`, `design`, `seed`, `keep`, `time`, `on_error`) |
| `sensitivities` | the sensitivity analyses of the model over the groups of its simulations by their name, the kind `sensitivity`: a `SensitivityRunner` of `sbmlsim.sensitivity.runner`, the runners of the subfolders of a model with several simulations by subfolder, or a reference `"module:attribute"` to either, resolved when it runs; `sensitivity(name)` gives it bound to the project (a runner without a project of its own runs the variants and the simulator settings of the project which lists it). A name is no name of a scan with indices |
| `actions` | the actions of the project by their name, analyses of its own which the kind `action` runs, `ProjectAction`: the function or a reference to it and a `description`; a function or a reference alone is an action without a description. The function is called as `function(project, out, *, workers, **arguments)` (`ActionFunction`) and returns its summary, a mapping of JSON values, or `None` |
| `simulator` | the settings of the simulator of the analyses, `SimulatorSettings`: the engine, `absolute_tolerance`, `relative_tolerance`, `jit_optimization` (the level LLJIT compiles the models of the experiments, fits and tests with) and the other settings of the integrator, `integrator`; the workers belong to a run |
| `figure_settings` | the rendering of the figures of the experiments, a `FigureSettings` of the plots |
| `figures` | the files of the figures, `FigureOutputs`: the `formats`, the resolution of the static images `savefig_dpi` and other parameters of matplotlib, `rc` |
| `checks` | the known exceptions of the [checks](#checks-of-a-project) of the model, `CheckSettings`, each with its reason, and the module of its experiments, `experiment_modules` (`experiments` below the package) |
| `release` | the release metadata, `sbmlsim.release.settings.ReleaseSettings`, `None` for a project which is not released; besides the metadata of the citation it names the framework of the release (`framework`, `sbmlsim` by default), the core libraries whose release notes the notes of a release summarize (`core_libraries`, `sbmlsim` and `sbmlutils` by default), the COMBINE archive the factory writes (`archive`) and the release hook of a framework (`hook`, an object or a reference), see [Release](release.md) |

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

## The analyses of a project

`python -m sbmlsim.run list my_model:PROJECT` lists the analyses a project defines (`--experiments` also the groups and classes of the experiments, which imports them) and `python -m sbmlsim.run <kind> spec.yaml --out DIR` runs one of them, see [Analyses](run.md). The kinds which take a project name it by `project: module:attribute` in their spec, the kinds `scan`, `sensitivity` and `uncertainty` the analysis by `analysis`, `fit` and its relatives a fit problem by `subset`, `simulate` the experiments by the names of their groups or classes. `Project.run_experiments(experiments, out)` is the same run from python, with the simulator, the figure settings, the formats and the rc parameters of the project.

## Variants

`variants` are the SBML files of the model by the name of a variant, the first the default: a model which is built for several groups of subjects has one per group. The variants share their ids, so the changes of an experiment and the parameters of a fit apply to all of them. An experiment names the variant of a task with `SimulationExperiment.variant_of(simulation)`, the draws of the fitted parameters are those of the default variant and run on every variant (`check_fit_parameters` checks that they have the same fitted values and distributions), a sensitivity group names its variant (`SensitivityGroup(variant=...)`, a global design is the same relative design around the values of that variant) and the battery of the credibility simulates every variant.

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
