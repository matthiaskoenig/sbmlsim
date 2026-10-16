"""Running fits and their reports from the command line.

A fit problem is defined by the fit experiments which enter it and the
parameters which are adjusted, see `FitDefinition`. Everything else, i.e.
creating the optimization problems for a strategy, running the optimizations
and reporting them, is the same for every problem and lives here, so that a model
only has to define its fits:

```python
FIT_DEFINITIONS = {
    "PK": FitDefinition(
        mapping_collections=f_collections_pk,
        parameters=parameters_pk,
        base_path=MODEL_PATH,
        data_path=DATA_PATH,
    ),
}

if __name__ == "__main__":
    fit_cli(FIT_DEFINITIONS, prog="fit_mymodel")
```

`fit_cli` runs a fit and reports it, `report_cli` reports parameters which were
stored earlier without optimizing again, with the records of the fit next to
them (`load_fit_records`). The report of a stored fit is `report_from_records`,
which `report_cli` and `identifiability_cli` use and a caller uses for a report
of its own, e.g. with the context of its model.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sbmlsim import log
from sbmlsim.fit import display
from sbmlsim.fit.derived import DerivedChanges, hook_summaries
from sbmlsim.fit.fisher import FisherInformation, fisher_information
from sbmlsim.fit.identifiability import (
    IdentifiabilityResult,
    ProfileSettings,
    profile_likelihood,
)
from sbmlsim.fit.methods import ReportContext
from sbmlsim.fit.objects import FitMappingCollection, FitParameter, MappingKind
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import (
    FitSettings,
    OptimizationAlgorithmType,
    OptimizationStrategy,
)
from sbmlsim.fit.parameters import ParameterSet, ParameterSets
from sbmlsim.fit.provenance import create_provenance
from sbmlsim.fit.report import FitReport
from sbmlsim.fit.result import OptimizationResult, fit_id
from sbmlsim.fit.runner import run_optimization
from sbmlsim.fit.sampling import SamplingType
from sbmlsim.utils import paths_text

logger = logging.getLogger(__name__)

#: short names of the optimization algorithms on the command line
ALGORITHMS: dict[str, OptimizationAlgorithmType] = {
    "LSQ": OptimizationAlgorithmType.LEAST_SQUARE,
    "DE": OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION,
}

#: what the strategies do, for the output of the tools
STRATEGY_INFO: dict[OptimizationStrategy, str] = {
    OptimizationStrategy.ALL: "one parameter set for all experiments",
    OptimizationStrategy.SINGLE: "one parameter set per experiment",
}

#: default arguments of the optimizers
ALGORITHM_KWARGS: dict[OptimizationAlgorithmType, dict[str, Any]] = {
    OptimizationAlgorithmType.LEAST_SQUARE: {
        "sampling": SamplingType.LOGUNIFORM_LHS,
        "diff_step": 0.05,
    },
    OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION: {},
}


@dataclass
class FitDefinition:
    """Definition of a fit problem.

    This is what a model provides: which fit mapping collections enter the fit,
    which parameters are adjusted and where the experiments and their data are.

    Attributes:
        mapping_collections: callable which creates the fit mapping collections
            by study id, e.g., a function of `sbmlsim.fit.helpers`. It is called
            when the fit runs, instantiating the experiments loads the models
            and the data.
        parameters: parameters which are adjusted in the fit.
        base_path: base path of the simulation experiments.
        data_path: path or paths of the datasets of the simulation experiments.
        settings: settings of the fit.
        hybridizations: the derived changes of the problem, see
            `sbmlsim.fit.derived`, e.g. the hybridizations of its neural
            networks.
        description: description of the fit problem for the reports of a fit.
    """

    mapping_collections: Callable[[], dict[str, list[FitMappingCollection]]]
    parameters: list[FitParameter]
    base_path: Path
    data_path: Path | Sequence[Path]
    settings: FitSettings = field(default_factory=FitSettings)
    hybridizations: Sequence[DerivedChanges] = ()
    description: str = ""

    def collections(
        self, study_ids: Sequence[str] | None = None
    ) -> list[FitMappingCollection]:
        """Create the fit mapping collections of the definition.

        Args:
            study_ids: studies to use, all studies by default.

        Returns:
            The collections of the selected studies.

        Raises:
            KeyError: if a study id is not part of the definition.
        """
        collections_by_study = self.mapping_collections()
        if study_ids:
            missing = [sid for sid in study_ids if sid not in collections_by_study]
            if missing:
                raise KeyError(
                    f"Unknown studies '{missing}', the definition has "
                    f"'{sorted(collections_by_study)}'."
                )
            selected = [collections_by_study[sid] for sid in study_ids]
        else:
            selected = list(collections_by_study.values())

        # every study contributes a list of collections
        return list(itertools.chain(*selected))

    def problem(
        self, opid: str, mapping_collections: list[FitMappingCollection] | None = None
    ) -> OptimizationProblem:
        """Create the optimization problem of the definition.

        Args:
            opid: id of the optimization problem.
            mapping_collections: experiments of the problem, all experiments of the
                definition by default.

        Returns:
            The uninitialized optimization problem.
        """
        return OptimizationProblem(
            opid=opid,
            mapping_collections=(
                self.collections()
                if mapping_collections is None
                else mapping_collections
            ),
            fit_parameters=self.parameters,
            base_path=self.base_path,
            data_path=self.data_path,
            hybridizations=self.hybridizations,
        )


@dataclass
class FitRun:
    """A finished fit: the problem which was optimized and its result."""

    problem: OptimizationProblem
    result: OptimizationResult

    def report(self, output_dir: Path, name: str | None = None, **kwargs: Any) -> Path:
        """Create the report of the fit.

        Args:
            output_dir: base directory of the reports.
            name: name of the report, the id of the problem by default.
            kwargs: additional arguments of `FitReport.from_optimization_result`.

        Returns:
            Path of the directory the report was written to.
        """
        report = FitReport.from_optimization_result(
            problem=self.problem, opt_result=self.result, **kwargs
        )
        return report.create(
            output_dir=output_dir, name=name if name else self.problem.opid
        )

    def fisher(self, **kwargs: Any) -> FisherInformation:
        """Get the Fisher information of the parameters of the best run.

        The local analysis next to `identifiability`, from one jacobian
        instead of a scan per parameter.

        Args:
            kwargs: additional arguments of `fisher_information`.

        Returns:
            The information with the errors and the correlations of the
            parameters.
        """
        return fisher_information(
            problem=self.problem,
            settings=self.result.settings_stored,
            parameter_set=self.result.parameter_set(),
            **kwargs,
        )

    def identifiability(
        self,
        profile_settings: ProfileSettings | None = None,
        n_cores: int = 1,
        **kwargs: Any,
    ) -> IdentifiabilityResult:
        """Analyse the identifiability of the parameters of the best run.

        Args:
            profile_settings: settings of the profile likelihood.
            n_cores: number of worker processes of the scans.
            kwargs: additional arguments of `profile_likelihood`.

        Returns:
            The profiles of the parameters around the best parameter set.
        """
        return profile_likelihood(
            problem=self.problem,
            settings=self.result.settings_stored,
            parameter_set=self.result.parameter_set(),
            profile_settings=profile_settings,
            n_cores=n_cores,
            **kwargs,
        )


def run_fit(
    definition: FitDefinition,
    opid: str | None = None,
    strategy: OptimizationStrategy = OptimizationStrategy.ALL,
    algorithm: OptimizationAlgorithmType = OptimizationAlgorithmType.LEAST_SQUARE,
    size: int = 4,
    n_cores: int = 1,
    seed: int | None = None,
    study_ids: Sequence[str] | None = None,
    timeout: float | None = None,
    output_dir: Path | None = None,
    arguments: Mapping[str, Any] | None = None,
    fixed_values: Mapping[str, float] | None = None,
    **kwargs: Any,
) -> dict[str, FitRun]:
    """Run the fit of a definition.

    Args:
        definition: definition of the fit problem.
        opid: id of the fit, `fit_id()` by default. It is the id of the
            optimization problem, of its result and of the directory of its
            report, so everything a fit produces carries the same key.
        strategy: fit all experiments together or every experiment on its own.
        algorithm: optimization algorithm.
        size: number of optimization runs per problem.
        n_cores: number of workers.
        seed: seed of the optimizations.
        study_ids: experiments to fit, all experiments by default.
        timeout: seconds a single optimization may run.
        output_dir: directory of the results. The single runs are written into
            `<output_dir>/<opid>/runs` while the fit runs, so an interrupted
            fit leaves the runs which finished.
        arguments: further arguments to record in the provenance of the fit,
            e.g. the command line of the caller; they come after the arguments
            of `run_fit` and replace one of the same name.
        fixed_values: values of the model which the caller keeps fixed, e.g.
            the fitted values of an earlier fit; recorded in the provenance,
            not applied.
        kwargs: additional arguments of the optimizer, they replace the
            defaults of `ALGORITHM_KWARGS`.

    Returns:
        The finished fits by optimization id.
    """
    if opid is None:
        opid = fit_id()
    mapping_collections = definition.collections(study_ids=study_ids)
    optimizer_kwargs = {**ALGORITHM_KWARGS.get(algorithm, {}), **kwargs}

    if strategy == OptimizationStrategy.SINGLE:
        # one problem per experiment, i.e., individual parameters
        problems = [
            definition.problem(
                opid=f"{collection.experiment_class.__name__}_{opid}",
                mapping_collections=[collection],
            )
            # a collection which a fit does not fit is not a problem of its
            # own: the validation data is evaluated with the training data and
            # the outliers and the excluded data are not used at all
            for collection in mapping_collections
            if collection.kind is MappingKind.TRAINING
        ]
    else:
        # one problem for all experiments
        problems = [
            definition.problem(opid=opid, mapping_collections=mapping_collections)
        ]

    runs: dict[str, FitRun] = {}
    for problem in problems:
        # the files are read once the problem is initialized
        problem.initialize(definition.settings)
        provenance = create_provenance(
            arguments={
                "size": size,
                "algorithm": algorithm.name,
                "strategy": strategy.name,
                "seed": seed,
                "study_ids": list(study_ids) if study_ids else None,
                "timeout": timeout,
                "optimizer": {k: str(v) for k, v in optimizer_kwargs.items()},
                **dict(arguments or {}),
            },
            seed=seed,
            run_seeds=OptimizationProblem.run_seeds(size, algorithm, seed),
            base_path=Path(definition.base_path),
            files=problem.files(),
            start_values={p.pid: p.start_value for p in problem.parameters},
            fixed_values=fixed_values,
        )
        result = run_optimization(
            problem=problem,
            settings=definition.settings,
            size=size,
            n_cores=n_cores,
            seed=seed,
            algorithm=algorithm,
            timeout=timeout,
            runs_dir=(Path(output_dir) / problem.opid / "runs" if output_dir else None),
            provenance=provenance,
            **optimizer_kwargs,
        )
        runs[problem.opid] = FitRun(problem=problem, result=result)

    return runs


def load_parameter_sets(paths: Sequence[Path]) -> ParameterSets:
    """Load the parameter sets of the given JSON files.

    The sets of all files are combined, a set which occurs in more than one
    file is prefixed with the name of its directory to keep the ids unique.

    Args:
        paths: JSON files written by `ParameterSets.to_json`.

    Returns:
        All parameter sets of the files.
    """
    sets: list[ParameterSet] = []
    sids: set[str] = set()
    for path in paths:
        for pset in ParameterSets.from_json(path):
            if pset.sid in sids:
                pset.sid = f"{path.parent.name}_{pset.sid}"
            sids.add(pset.sid)
            sets.append(pset)

    return ParameterSets(sets)


@dataclass(frozen=True)
class FitRecords:
    """The records of a fit which a report of its parameters is built from.

    Attributes:
        opt_result: the result of the optimization, `optimization_result.json`,
            with the settings, the runs and the provenance of the fit.
        identifiability: the profiles of the parameters, `identifiability.json`.
        fisher: the Fisher information of the parameters, `fisher.json`.
    """

    opt_result: OptimizationResult | None = None
    identifiability: IdentifiabilityResult | None = None
    fisher: FisherInformation | None = None

    @property
    def names(self) -> list[str]:
        """Names of the files the records were read from."""
        return [
            name
            for name, record in [
                ("optimization_result.json", self.opt_result),
                ("identifiability.json", self.identifiability),
                ("fisher.json", self.fisher),
            ]
            if record is not None
        ]


def load_fit_records(directory: Path) -> FitRecords:
    """Load the records of a fit from the folder of its report.

    A report writes them next to its `parameters.json`; a file which is not
    there is a record the fit does not have, e.g. a fit without an
    identifiability analysis.

    Args:
        directory: the folder of the report of the fit.

    Returns:
        The records which are in the folder.
    """
    opt_path = directory / "optimization_result.json"
    identifiability_path = directory / "identifiability.json"
    fisher_path = directory / "fisher.json"
    return FitRecords(
        opt_result=(
            OptimizationResult.from_json(opt_path) if opt_path.exists() else None
        ),
        identifiability=(
            IdentifiabilityResult.from_json(identifiability_path)
            if identifiability_path.exists()
            else None
        ),
        fisher=(
            FisherInformation.from_dict(
                json.loads(fisher_path.read_text(encoding="utf-8"))
            )
            if fisher_path.exists()
            else None
        ),
    )


def _settings_of(definition: FitDefinition, records: FitRecords) -> FitSettings:
    """Get the settings a report of stored parameters is computed with.

    The settings the fit ran with, stored in its record, describe the fit; the
    definition may have changed since. Without a record with settings they
    are the settings of the definition. A difference is logged as a warning.

    Args:
        definition: the definition of the fit.
        records: the records of the fit.

    Returns:
        The settings of the record if it has them, else of the definition.
    """
    opt_result = records.opt_result
    if opt_result is None or opt_result.settings is None:
        return definition.settings
    stored = opt_result.settings
    if stored != definition.settings:
        current = definition.settings.to_dict()
        differ = sorted(
            key for key, value in stored.to_dict().items() if current.get(key) != value
        )
        logger.warning(
            "The settings of the record of the fit '%s' differ from the settings "
            "of its definition in %s; the report uses the settings of the "
            "record, which the fit ran with.",
            opt_result.sid,
            ", ".join(differ),
        )
    return stored


def report_from_records(
    definition: FitDefinition,
    directory: Path,
    *,
    opid: str,
    parameters: Sequence[Path] | ParameterSets | None = None,
    context: ReportContext | None = None,
    mapping_figures: bool = True,
    show_titles: bool = False,
) -> FitReport:
    """Build the report of a stored fit from the records of its folder.

    No fit is run. The records of the folder (`load_fit_records`), i.e. the
    result of the optimization, the profiles and the Fisher information,
    give the report its methods, its provenance, its runs and its
    identifiability; a record which is not there is left out. The settings of
    the report are the settings the fit ran with, stored in its result; when
    they differ from the settings of the definition, which may have changed
    since, a warning names the settings which differ. Without a result with
    settings the report has the settings of the definition.

    `report_cli` and `identifiability_cli` build their reports with it, a
    caller builds the report of a fit of its own with it, e.g. with the
    context of its model. The report is written by `FitReport.create`.

    Args:
        definition: the definition of the fit, its problem is reported.
        directory: the folder of the records of the fit, e.g. the folder of
            its report.
        opid: the id of the problem of the report, e.g. the subset of the fit.
            The report names the fit by the id of its result if it has one.
        parameters: the parameter sets to report, or the JSON files of them,
            `parameters.json` of the folder by default. Several sets are
            combined into a single report, which compares them.
        context: what the report says about the fit beyond the fit, e.g. its
            title; an empty context by default.
        mapping_figures: draw the two figures of every fit mapping.
        show_titles: add titles to the figures.

    Returns:
        The report, initialized and ready for `FitReport.create`.
    """
    records = load_fit_records(directory)
    if parameters is None:
        parameter_sets = load_parameter_sets([directory / "parameters.json"])
    elif isinstance(parameters, ParameterSets):
        parameter_sets = parameters
    else:
        parameter_sets = load_parameter_sets(parameters)
    # the problem is a definition, `FitReport` initializes a copy of its own
    # with the settings, so later fits and edits cannot change the report
    return FitReport(
        problem=definition.problem(opid=opid),
        settings=_settings_of(definition, records),
        parameter_sets=parameter_sets,
        opt_result=records.opt_result,
        identifiability=records.identifiability,
        fisher=records.fisher,
        show_titles=show_titles,
        mapping_figures=mapping_figures,
        context=context,
    )


def _display_records(
    report: FitReport,
    subset: str,
    parameters: Sequence[Path],
    output_dir: Path,
    *,
    section: str = "Report",
    icon: str = display.ICON_REPORT,
) -> None:
    """Show the report of a stored fit before it is created.

    Args:
        report: the report, see `report_from_records`.
        subset: the fit problem.
        parameters: the files of the parameter sets.
        output_dir: the directory of the report.
        section: the title of the section.
        icon: the icon of the section.
    """
    problem = report.problem
    names = FitRecords(
        opt_result=report.opt_result,
        identifiability=report.identifiability,
        fisher=report.fisher,
    ).names
    display.section(section, icon=icon)
    display.key_values(
        {
            "problem": subset,
            "parameters": ", ".join(str(p) for p in parameters),
            "sets": ", ".join(pset.sid for pset in report.parameter_sets),
            "records": ", ".join(names) if names else "none",
            "output": output_dir,
        }
    )
    display.print_parameters(
        problem.parameters,
        coverage=(
            problem.parameter_mapping.coverage()
            if problem.parameter_mapping is not None
            else None
        ),
        hooks=hook_summaries(problem.hybridizations),
    )
    display.print_settings(report.settings)


def _definition(definitions: dict[str, FitDefinition], key: str) -> FitDefinition:
    """Get the definition for a key."""
    return definitions[key]


def _add_common_arguments(
    parser: argparse.ArgumentParser, definitions: dict[str, FitDefinition]
) -> None:
    """Add the arguments which the fit and the report share."""
    keys = list(definitions)
    parser.add_argument(
        "-x",
        "--subset",
        choices=keys,
        default=keys[0],
        help="fit problem to run",
    )
    parser.add_argument(
        "-n", "--name", default=None, help="name of the report, the fit id by default"
    )
    parser.add_argument(
        "--no-mapping-figures",
        action="store_true",
        help="leave the two figures of every fit mapping out of the report. "
        "They are almost the whole cost of it, so a report which is read for "
        "its tables and its overview figures is created much faster",
    )


def fit_cli(
    definitions: dict[str, FitDefinition],
    args: Sequence[str] | None = None,
    prog: str | None = None,
    description: str = "Parameter fitting.",
) -> dict[str, FitRun]:
    """Run a fit of one of the definitions from the command line and report it.

    Args:
        definitions: fit problems by name, the name is the `--subset` argument.
        args: command line arguments, `sys.argv` by default.
        prog: name of the program in the help, the script by default.
        description: description of the program in the help.

    Returns:
        The finished fits by optimization id.

    Raises:
        ValueError: if no definitions are given.
    """
    if not definitions:
        raise ValueError("At least one FitDefinition is required.")

    parser = argparse.ArgumentParser(prog=prog, description=description)
    _add_common_arguments(parser, definitions)
    parser.add_argument(
        "-c", "--cores", type=int, default=1, help="number of cores for the fitting"
    )
    parser.add_argument(
        "-r", "--runs", type=int, default=4, help="number of optimization runs"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=1234, help="seed of the optimization"
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=list(ALGORITHMS),
        default="LSQ",
        help="optimization algorithm",
    )
    parser.add_argument(
        "-t",
        "--strategy",
        type=OptimizationStrategy,
        choices=list(OptimizationStrategy),
        default=OptimizationStrategy.ALL,
        help="fit the experiments together or every experiment on its own",
    )
    parser.add_argument(
        "-e",
        "--experiments",
        nargs="+",
        default=None,
        help="experiments to fit, all experiments of the problem by default",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="seconds a single optimization may run, no limit by default",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("results") / "fit",
        help="directory for the results of the fit",
    )
    options = parser.parse_args(args)
    log.enable_rich_logging()

    definition = _definition(definitions, options.subset)
    algorithm = ALGORITHMS[options.method]
    # the id of the fit, created before it starts, so that the problem, its
    # result and the directory of its report carry the same key
    opid = fit_id(options.subset)

    display.section("Fit", icon=display.ICON_FIT)
    display.key_values(
        {
            "fit": opid,
            "problem": options.subset,
            "strategy": f"{options.strategy.value} ({STRATEGY_INFO[options.strategy]})",
            "algorithm": f"{options.method} ({algorithm.name})",
            "experiments": options.experiments or "all of the problem",
            "runs": f"{options.runs} on {options.cores} core(s)",
            "seed": options.seed,
            "timeout": (f"{options.timeout} s per run" if options.timeout else "none"),
            "base path": paths_text(definition.base_path),
            "data path": paths_text(definition.data_path),
            "output": options.output_dir,
        }
    )
    # the parameters are printed per problem, with their coverage, once
    # `run_fit` initializes each one in `run_optimization` -- printing them
    # again here from `definition.parameters` alone would only repeat the
    # same block without the coverage, once per problem for `SINGLE`
    display.print_settings(definition.settings)

    runs = run_fit(
        definition=definition,
        opid=opid,
        strategy=options.strategy,
        algorithm=algorithm,
        size=options.runs,
        n_cores=options.cores,
        seed=options.seed,
        study_ids=options.experiments,
        timeout=options.timeout,
        output_dir=options.output_dir,
    )

    # the fit only optimizes, the report is created from its parameters
    for run in runs.values():
        run.report(
            output_dir=options.output_dir,
            name=options.name,
            show_titles=False,
            mapping_figures=not options.no_mapping_figures,
        )

    return runs


def report_cli(
    definitions: dict[str, FitDefinition],
    args: Sequence[str] | None = None,
    prog: str | None = None,
    description: str = "Report of a fit for stored parameters.",
    context: (
        ReportContext | Callable[[str, FitDefinition], ReportContext] | None
    ) = None,
) -> Path:
    """Report stored parameters from the command line, without optimizing.

    The parameters come from the `parameters.json` a fit wrote. Several files
    are combined into a single report, which compares their parameter sets.
    The records of the fit next to the first file (`load_fit_records`) give the
    report the methods, the provenance, the runs and the identifiability of
    the fit, see `report_from_records`.

    Args:
        definitions: fit problems by name, the name is the `--subset` argument.
        args: command line arguments, `sys.argv` by default.
        prog: name of the program in the help, the script by default.
        description: description of the program in the help.
        context: what the report says about the fit beyond the fit, e.g. the
            title of the model of a caller: a `ReportContext`, or a callable
            which gets the subset and its definition and returns it. An empty
            context by default.

    Returns:
        Path of the directory the report was written to.

    Raises:
        ValueError: if no definitions are given.
    """
    if not definitions:
        raise ValueError("At least one FitDefinition is required.")

    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "parameters",
        type=Path,
        nargs="+",
        help="JSON files with the parameter sets to report",
    )
    _add_common_arguments(parser, definitions)
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("results") / "report",
        help="directory for the report",
    )
    options = parser.parse_args(args)
    log.enable_rich_logging()

    definition = _definition(definitions, options.subset)
    # no fit is run here, the parameter sets come from the given files; the
    # parameters and their coverage are shown before the report is created
    report = report_from_records(
        definition,
        options.parameters[0].parent,
        opid=options.subset,
        parameters=options.parameters,
        context=(
            context
            if context is None or isinstance(context, ReportContext)
            else context(options.subset, definition)
        ),
        mapping_figures=not options.no_mapping_figures,
    )
    _display_records(report, options.subset, options.parameters, options.output_dir)
    return report.create(
        output_dir=options.output_dir,
        name=options.name if options.name else "report",
    )


def identifiability_cli(
    definitions: dict[str, FitDefinition],
    args: Sequence[str] | None = None,
    prog: str | None = None,
    description: str = "Identifiability of stored parameters by profile likelihood.",
) -> Path:
    """Analyse the identifiability of stored parameters from the command line.

    The parameters come from the `parameters.json` a fit wrote; the profiles
    and the Fisher information are computed around its first parameter set,
    and the report of the parameters with the identifiability section is
    written. `--no-fisher` reports the profiles alone. The result of the
    optimization next to the file (`load_fit_records`) gives the report the
    methods, the optimizer and the provenance of the fit, and its settings
    are the settings of the analyses.

    Args:
        definitions: fit problems by name, the name is the `--subset` argument.
        args: command line arguments, `sys.argv` by default.
        prog: name of the program in the help, the script by default.
        description: description of the program in the help.

    Returns:
        Path of the directory the report was written to.

    Raises:
        ValueError: if no definitions are given.
    """
    if not definitions:
        raise ValueError("At least one FitDefinition is required.")

    defaults = ProfileSettings()
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "parameters",
        type=Path,
        help="JSON file with the parameter set to analyse, the first set is used",
    )
    _add_common_arguments(parser, definitions)
    parser.add_argument(
        "-c", "--cores", type=int, default=1, help="number of cores for the scans"
    )
    parser.add_argument(
        "-p",
        "--parameter",
        action="append",
        default=None,
        help=(
            "parameter to profile, all parameters which are no elements of a "
            "network by default; repeatable"
        ),
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=defaults.alpha,
        help="confidence level of the intervals",
    )
    parser.add_argument(
        "--df",
        type=int,
        default=defaults.degrees_of_freedom,
        help="degrees of freedom of the threshold, 1 for pointwise intervals",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=defaults.max_points,
        help="largest number of points of a profile in one direction",
    )
    parser.add_argument(
        "--initial-step",
        type=float,
        default=defaults.initial_step,
        help="first step of a scan in decades of the parameter",
    )
    parser.add_argument(
        "--min-step",
        type=float,
        default=defaults.min_step,
        help="smallest step of a scan in decades",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=defaults.max_step,
        help="largest step of a scan in decades",
    )
    parser.add_argument(
        "--no-reoptimize",
        action="store_true",
        help="keep the other parameters at the optimum, i.e., scan the cost "
        "instead of computing the profile likelihood",
    )
    parser.add_argument(
        "--no-fisher",
        action="store_true",
        help="skip the Fisher information, i.e., report the profiles alone",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("results") / "identifiability",
        help="directory for the report",
    )
    options = parser.parse_args(args)
    log.enable_rich_logging()

    definition = _definition(definitions, options.subset)
    parameter_sets = load_parameter_sets([options.parameters])
    parameter_set = parameter_sets[0]
    profile_settings = ProfileSettings(
        alpha=options.alpha,
        degrees_of_freedom=options.df,
        initial_step=options.initial_step,
        min_step=options.min_step,
        max_step=options.max_step,
        max_points=options.max_points,
        reoptimize=not options.no_reoptimize,
    )

    # the records of the fit next to the parameters give the report the
    # methods, the optimizer and the provenance of the fit, and the settings
    # the fit ran with, which the analyses use as well; the analyses run on
    # the initialized problem of the report, so the parameters and their
    # coverage are shown before the scans start, and `profile_likelihood`
    # initializing the same problem with the same settings again is a no-op
    report = report_from_records(
        definition,
        options.parameters.parent,
        opid=options.subset,
        parameters=ParameterSets([parameter_set]),
        mapping_figures=not options.no_mapping_figures,
    )
    problem, settings = report.problem, report.settings
    _display_records(
        report,
        options.subset,
        [options.parameters],
        options.output_dir,
        section="Identifiability",
        icon=display.ICON_IDENTIFIABILITY,
    )

    result = profile_likelihood(
        problem=problem,
        settings=settings,
        parameter_set=parameter_set,
        profile_settings=profile_settings,
        pids=options.parameter,
        n_cores=options.cores,
    )

    # the local analysis next to the profiles, one jacobian instead of a scan
    # per parameter, so it is cheap enough to come with them
    fisher: FisherInformation | None = None
    if not options.no_fisher:
        fisher = fisher_information(
            problem=problem,
            settings=settings,
            parameter_set=parameter_set,
            alpha=options.alpha,
        )
        display.section("Fisher information", icon=display.ICON_IDENTIFIABILITY)
        display.key_values(
            {
                "rank": f"{fisher.rank} of {fisher.k}",
                "condition number": f"{fisher.condition_number:.4g}",
            }
        )

    # the analyses of this run replace those of the record
    report.identifiability = result
    report.fisher = fisher
    return report.create(
        output_dir=options.output_dir,
        name=options.name if options.name else "identifiability",
    )
