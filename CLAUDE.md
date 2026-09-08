# CLAUDE.md

This file provides guidance when working with code in this repository.

## Project

`sbmlsim` is a python library for the simulation of models in the Systems Biology Markup Language (SBML), built on libroadrunner: timecourse simulations with changes, parameter scans, simulation experiments with datasets, figures and reports, parameter fitting, sensitivity analysis and the execution of SED-ML from COMBINE archives. Pure library, no CLI entry points. Requires python >= 3.13, packaged with hatchling (version is read from `src/sbmlsim/__init__.py`). Runtime dependencies are `libroadrunner` (simulation), `sbmlutils`, `python-libsbml`, `python-libsedml`, `python-libnuml` and `pymetadata` (SBML, SED-ML, NuML, COMBINE archives), `numpy`, `pandas`, `xarray`, `scipy`, `sympy`, `pint`, `pydantic`, `dill`, `xmltodict` (data, units, numerics), `petab`, `SALib` (fitting and sensitivity analysis) and `matplotlib`, `seaborn`, `jinja2`, `rich` (plots and reports). `amici`, `basico`/COPASI, `h5py`, `pypesto` and `juliacall` are not dependencies; the comparison scripts and examples which use them import them optionally.

## Commands

```bash
# environment (uv based)
uv sync --extra dev
uv run pre-commit install

# tests
pytest                                             # all tests (about 1 min)
pytest tests/simulation/test_simulation.py         # single file
pytest tests/simulation/test_scan.py::test_scan1d  # single test
pytest -rs                                         # list the skipped tests
tox r -e py3.14                                    # single tox env (py3.13, py3.14 available)
tox run-parallel                                   # full matrix + ty

# lint / format / types
ruff check
ruff format
tox -e ty                       # ty type check (config in [tool.ty] in pyproject.toml)
uvx ty check                    # same check, straight from the working tree

# examples, they are modules of the `examples` package and not part of sbmlsim
python -m examples.timecourse
python -m examples.sensitivity.sensitivity_example
```

Release steps are in `docs/development.md` (there is no separate `RELEASE.md`); version bumps go through `uvx bump-my-version bump [major|minor|patch]` (updates `src/sbmlsim/__init__.py` and `CITATION.cff`), and pushing the tag triggers the PyPI release workflow.

Documentation is [Zensical](https://zensical.org/): markdown sources in `docs/`, configured in `zensical.toml`, built into the gitignored `site/` (`uv run zensical build --clean`, `uv run zensical serve` for the preview). The API reference is rendered from the docstrings by mkdocstrings; a page in `docs/api/` is just `::: sbmlsim.<module>`, so nothing is generated into the repository. `scripts/llms_txt.py` runs after the build and writes the agent facing files (`llms.txt`, `llms-full.txt` and the markdown of every page) into `site/`. The `documentation` workflow runs both and publishes the site from `develop`.

## Architecture

**`simulation/`, `simulator/`, `result/` — the simulation core.** `Timecourse(start, end, steps, changes)` is one period of a simulation with its changes, `TimecourseSim` concatenates timecourses (dosing protocols, pre-simulations with `discard=True`), `ScanSim(simulation, dimensions)` runs a `TimecourseSim` over the `Dimension` objects of `simulation/range.py`, each a set of changes with vectors of values. `simulation/sensitivity.py` builds the scans of `ModelSensitivity` (all parameters by relative difference or from distributions), `simulation/algorithm.py` and `kisaos.py` describe integrators by their KISAO terms. `SimulatorSerial` in `simulator/simulation_serial.py` loads a model and runs `run_timecourse`/`run_scan`; every timecourse simulation is a `ScanSim` internally. Results are `XResult` (`result/xresult.py`), an `xarray.Dataset` with a `_time` dimension plus one dimension per scan dimension, and the `UnitsInformation` of its variables; `dim_mean` etc. reduce over the scan dimensions and return quantities.

**`model/` — models.** `AbstractModel` is the description (source, language, changes, selections) without loading; `model_resources.Source` resolves the source (path relative to `base_path`, URL, BioModels URN). `RoadrunnerSBMLModel` owns the `roadrunner.RoadRunner` instance `r` (`None` until loaded), reads the units into `uinfo`, applies the changes and the integrator `settings`. `ModelChange.clamp_species` is the structural change applied through `Timecourse.model_manipulations`.

**`units.py` — units.** `UnitsInformation` maps identifiers to unit strings and holds the pint `UnitRegistry`; `from_sbml` reads the units of a model, `normalize_changes` converts quantities to model units, and `Quantity` (an alias of pint's `PlainQuantity`, the base of every quantity a registry creates) is the type used in all annotations. Every experiment shares one registry.

**`experiment/`, `data.py`, `task/`, `plot/`, `report/` — simulation experiments.** `SimulationExperiment` (`experiment/experiment.py`) is subclassed with `models()`, `datasets()`, `simulations()`, `tasks()`, `data()`, `figures()`, `reports()` and `fit_mappings()`, each returning a dict keyed by id; `initialize()` collects them and `run(simulator)` runs the tasks, evaluates the data and renders the figures. `ExperimentRunner` (`experiment/runner.py`) resolves the abstract models into roadrunner models, runs several experiments and writes results, figures and the JSON serialization. `Data` (`data.py`) is a promise for data of a task (`Data("[X]", task=...)`), a dataset column or a function of other data; `DataSet` is a `pandas.DataFrame` with `uinfo`. `Task` pairs a model and a simulation. `plot/plotting.py` is the backend independent figure model (`Figure`, `Plot`, `Axis`, `Curve`, `ShadedArea`, `Style` with `Line`, `Marker`, `Fill`), which is also what SED-ML plots map onto; `plot/serialization_matplotlib.py` renders it. `plot/plotting_deprecated_matplotlib.py` is legacy code kept for the glucose example. `report/experiment_report.py` renders HTML/markdown reports from jinja2 templates in `resources/templates/`.

**`fit/` — parameter fitting.** `FitExperiment` names an experiment class and its mappings, `FitMapping` pairs a reference `FitData` (dataset) with an observable `FitData` (task), `FitParameter` carries bounds and unit. `OptimizationProblem` (`fit/optimization.py`) is picklable; `initialize` creates the runner, resolves the data and weights (`fit/options.py` enumerates residual, loss, weighting types), `residuals`/`cost_least_square` are the objective, `optimize` runs scipy least squares or differential evolution. `fit/runner.py` samples start points (`fit/sampling.py`) and runs in parallel, `fit/result.py` collects the fits into an `OptimizationResult`, `fit/analysis.py` writes the plots and the HTML report of a fit, `fit/petab_omex.py` packages PEtab problems as archives. The fit tests are skipped (`no fit support`) while this part is reworked.

**`sensitivity/` — sensitivity analysis.** `SensitivitySimulation` (`sensitivity/analysis.py`) is subclassed with `simulate(r, changes) -> outputs`; `SensitivityAnalysis` is the base of `LocalSensitivityAnalysis` and the SALib based `SamplingSensitivityAnalysis`, `SobolSensitivityAnalysis`, `FASTSensitivityAnalysis` and `MorrisSensitivityAnalysis`: `execute()` creates samples, simulates them in a multiprocessing pool (which is why the simulation class must live in an importable module) and computes `sensitivity[group][key]` as `xarray.DataArray`; `plot()` writes figures into `results_path`. `SensitivityParameter`, `SensitivityOutput` and `AnalysisGroup` (`sensitivity/parameters.py`, `analysis.py`) describe the problem.

**`combine/` — SED-ML and COMBINE archives.** `combine/sedml/io.py` reads a SED-ML file, string or archive (`SEDMLReader`, extracting archives with pymetadata), `combine/sedml/parser.py` holds `SEDMLParser`, which translates the document into a `SimulationExperiment` subclass (`parser.exp_class`), and `SEDMLSerializer`, which writes an experiment as SED-ML/OMEX; `combine/sedml/runner.py::execute_sedml` does read, parse and run. `combine/sedml/task.py` builds the task tree of repeated tasks, `data.py`/`numl.py` read data descriptions, `combine/mathml.py` evaluates MathML through sympy, `combine/datagenerator.py` post-processes results. `combine/sedml/parser.py` is 1900 lines and the module most sensitive to libsedml types: libsedml objects are narrowed with `cast(libsedml.SedUniformTimeCourse, sed_sim)` after the type code check.

**`comparison/` — simulator comparison.** Scripts comparing roadrunner with AMICI and COPASI on PBPK models plus `diff.py` for the numerical comparison of results. `comparison/amicitesting/` and `comparison/results/` are generated AMICI model code and compiled extensions; they are excluded from the wheel (`[tool.hatch.build]`), from ruff and from ty.

`console.py` (rich console, for scripts and examples) and `log.py` provide the shared output/logging. Modules get their logger from the standard library with `logging.getLogger(__name__)`. The package never configures logging: `log.enable_rich_logging()` is the opt-in for scripts. Library code logs, it does not print, and log calls use lazy `%s` formatting rather than f-strings (enforced by ruff `G`).

## Conventions

- Type checking is done with [ty](https://docs.astral.sh/ty/) (mypy was removed in 0.6.0). `[tool.ty.terminal] error-on-warning = true` means warnings fail the check, so the tree must stay at zero diagnostics. Suppress a diagnostic with a rule-specific `# ty: ignore[rule-name]`, never a blanket `# type: ignore`. ty also runs as a pre-commit hook (`--extra dev`).
- libsbml, libsedml, libnuml and roadrunner have no type stubs and build their objects through SWIG, so annotate their objects explicitly and use the getters (`getId()`) rather than the attributes SWIG synthesizes. `RoadrunnerSBMLModel.r` and `SimulatorSerial.r` are `None` until a model is loaded; use `r_loaded`/`model_loaded` or check for `None` before using them.
- Every module, class and function of the package carries full type annotations and a docstring (ruff `D`, google convention); `examples/` and `tests/` are exempt from the docstring rules. Newer docstrings are google style, older ones still use the `:param:` form.
- `examples/` at the top level holds the runnable examples, they are not part of the package and are run as modules (`python -m examples.timecourse`). An example writes into the current working directory and never opens a window; the `conftest.py` at the root selects the `Agg` backend for the test session and puts the repository on `sys.path`. `tests/examples/test_example_scripts.py` runs the working examples; the experiments with post processing functions (`examples/demo`, `examples/repressilator`, `examples/midazolam`, `examples/covid`) currently fail and are listed in `examples/README.md`.
- The packaged models live in `src/sbmlsim/resources/models/` and are named in `resources/__init__.py` (`REPRESSILATOR_SBML`, `DEMO_SBML`, `MIDAZOLAM_SBML`); test fixtures live in `tests/data/`.
- An optional dependency which is never installed (amici, basico, pypesto, juliacall, h5py) is imported with `# ty: ignore[unresolved-import]`.
- Library code does not call `plt.show()`; figures are returned or saved, `show_figures` defaults to `False`.
- Markdown carries no hard line wraps: a paragraph, a list item or a table row is a single line and the wrapping is left to the editor. Code fences, headings and the rows of badges keep their line structure.
- Release notes go in `release-notes/` as part of a release commit.
