"""Running fits and their reports from the command line.

A fit problem is defined by the fit experiments which enter it and the
parameters which are adjusted, see `FitDefinition`. Everything else — creating
the optimization problems for a strategy, running the optimizations and
reporting them — is the same for every problem and lives here, so that a model
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
stored earlier without optimizing again.
"""

from __future__ import annotations

import argparse
import itertools
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sbmlsim import log
from sbmlsim.fit import display
from sbmlsim.fit.identifiability import (
    IdentifiabilityResult,
    ProfileSettings,
    profile_likelihood,
)
from sbmlsim.fit.objects import FitMappingCollection, FitParameter, MappingKind
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import (
    FitSettings,
    OptimizationAlgorithmType,
    OptimizationStrategy,
)
from sbmlsim.fit.parameters import ParameterSet, ParameterSets
from sbmlsim.fit.report import FitReport
from sbmlsim.fit.result import OptimizationResult, fit_id
from sbmlsim.fit.runner import run_optimization
from sbmlsim.fit.sampling import SamplingType

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
        data_path: path of the datasets of the simulation experiments.
        settings: settings of the fit.
    """

    mapping_collections: Callable[[], dict[str, list[FitMappingCollection]]]
    parameters: list[FitParameter]
    base_path: Path
    data_path: Path
    settings: FitSettings = field(default_factory=FitSettings)

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
        result = run_optimization(
            problem=problem,
            settings=definition.settings,
            size=size,
            n_cores=n_cores,
            seed=seed,
            algorithm=algorithm,
            timeout=timeout,
            runs_dir=(Path(output_dir) / problem.opid / "runs" if output_dir else None),
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
            "base path": definition.base_path,
            "data path": definition.data_path,
            "output": options.output_dir,
        }
    )
    display.print_parameters(definition.parameters)
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
        run.report(output_dir=options.output_dir, name=options.name, show_titles=False)

    return runs


def report_cli(
    definitions: dict[str, FitDefinition],
    args: Sequence[str] | None = None,
    prog: str | None = None,
    description: str = "Report of a fit for stored parameters.",
) -> Path:
    """Report stored parameters from the command line, without optimizing.

    The parameters come from the `parameters.json` a fit wrote. Several files
    are combined into a single report, which compares their parameter sets.

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
    parameter_sets = load_parameter_sets(options.parameters)

    display.section("Report", icon=display.ICON_REPORT)
    display.key_values(
        {
            "problem": options.subset,
            "parameters": ", ".join(str(p) for p in options.parameters),
            "sets": ", ".join(pset.sid for pset in parameter_sets),
            "output": options.output_dir,
        }
    )
    display.print_settings(definition.settings)

    # only the definition of the problem is needed, no fit is run here
    report = FitReport(
        problem=definition.problem(opid=options.subset),
        settings=definition.settings,
        parameter_sets=parameter_sets,
        show_titles=False,
    )
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
    are computed around its first parameter set, and the report of the
    parameters with the identifiability section is written.

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
        help="parameter to profile, all parameters by default; repeatable",
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

    display.section("Identifiability", icon=display.ICON_IDENTIFIABILITY)
    display.key_values(
        {
            "problem": options.subset,
            "parameters": str(options.parameters),
            "set": parameter_set.sid,
            "output": options.output_dir,
        }
    )
    display.print_settings(definition.settings)

    problem = definition.problem(opid=options.subset)
    result = profile_likelihood(
        problem=problem,
        settings=definition.settings,
        parameter_set=parameter_set,
        profile_settings=profile_settings,
        pids=options.parameter,
        n_cores=options.cores,
    )
    report = FitReport(
        problem=problem,
        settings=definition.settings,
        parameter_sets=ParameterSets([parameter_set]),
        identifiability=result,
        show_titles=False,
    )
    return report.create(
        output_dir=options.output_dir,
        name=options.name if options.name else "identifiability",
    )
