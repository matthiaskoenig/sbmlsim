"""The kinds of a fit: the fit, its profiles, Fisher information, report, PEtab.

Every spec names a fit problem of a project, `project` (`module:attribute` of
a `sbmlsim.project.Project`) and `subset` (a key of `Project.fits`). The
kinds which start from a finished fit read its records from the input `fit`
(`--input fit=DIR`): the out of a `fit` run, whose records are in its
directory `<opid>`, or a records directory itself, i.e. a directory with the
`parameters.json` of the fit and the `optimization_result.json` of its
repeats next to it, e.g. the out of an `identifiability` or a `fisher` run,
which copy the records of their fit, so a report reads the analyses and the
fit from one directory.

- `fit` (`FitSpec`) optimizes the problem, one unit per block of repeats,
  `unit_blocks(repeats)`: `<opid>/runs/<opid>_run_<k>.json` are the repeats
  (the units and the checkpoints of a resume), the merge writes
  `<opid>/optimization_result.json`, `<opid>/parameters.json` (the best run)
  and `<opid>/fits.tsv`, with `identifiability` the profiles and the Fisher
  information of the best run, `<opid>/identifiability.json` and
  `<opid>/fisher.json`, with `report` the report of the fit into `<opid>`,
  and with `adopt` the record of the fit, `fits/<opid>`, and `adopted.json`,
  the adopted fits of the project with the fit adopted for its subset (the
  files of the project are never written). Least squares starts from
  `start`, which the plan resolves into a point, instead of samples of the
  bounds, and with `record_adopted` the fit records the adopted fits of the
  other subsets, whose values its models carry, see
  `sbmlsim.fit.project_fits`; the plan refuses a fit whose bounds let a
  remainder `1 - a - b` of a model of the project become negative.
- `identifiability` (`IdentifiabilitySpec`) computes the profile likelihoods,
  one unit per parameter, `profiles/<pid>.json`, which the merge combines
  into `identifiability.json`, with `fisher` also `fisher.json`.
- `fisher` (`FisherSpec`) computes the Fisher information, `fisher.json`.
- `report` (`ReportSpec`) writes the report of the records, `report/`, with
  `adopted` the report of the adopted fit of the subset, of every adopted fit
  for the subset `all`, from its record in the project into `<fit id>/`; a
  report has the context of the project (`sbmlsim.fit.cli.records_context`).
- `petab` (`PetabSpec`) writes the problem as PEtab v2, `petab/problem.yaml`,
  which starts from the best parameters of a fit when one is given.
- `adopt` (`AdoptSpec`) adopts a finished fit of the input `fit`: its record
  into `fits/<fit id>` and `adopted.json`, as `fit` with `adopt` does.

Every record file is written atomically, so a unit computed again leaves
complete files. The modules of the fit are imported when a kind runs, not
when this module is imported.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, override

import numpy as np
from pydantic import Field, model_validator

from sbmlsim.atomic import atomic_path, write_text_atomic
from sbmlsim.cgroup import available_cpus
from sbmlsim.run.kinds import (
    ConfigurationError,
    Kind,
    PlanMismatch,
    Spec,
    SplitSpec,
    load_project_ref,
    planned,
    register,
)
from sbmlsim.run.units import MissingUnits, unit_blocks

if TYPE_CHECKING:
    from sbmlsim.fit.cli import FitDefinition, FitRecords
    from sbmlsim.fit.identifiability import ProfileSettings
    from sbmlsim.fit.optimization import OptimizationProblem
    from sbmlsim.fit.options import FitSettings
    from sbmlsim.fit.parameters import ParameterSet
    from sbmlsim.fit.sampling import SamplingType
    from sbmlsim.run.context import RunContext

__all__ = [
    "ADOPT",
    "FISHER",
    "FIT",
    "IDENTIFIABILITY",
    "PETAB",
    "REPORT",
    "AdoptSpec",
    "FisherSpec",
    "FitProblemSpec",
    "FitSpec",
    "IdentifiabilitySpec",
    "PetabSpec",
    "ProfileSpec",
    "ReportSpec",
]

logger = logging.getLogger(__name__)

#: the result of the optimization of a fit, its repeats, settings and provenance
RESULT_FILE = "optimization_result.json"
#: the best parameter set of a fit
PARAMETERS_FILE = "parameters.json"
#: the repeats of a fit as a table
FITS_FILE = "fits.tsv"
#: the profiles of a fit
IDENTIFIABILITY_FILE = "identifiability.json"
#: the Fisher information of a fit
FISHER_FILE = "fisher.json"
#: the records of a fit which an analysis of it copies into its out
FIT_RECORDS: tuple[str, ...] = (RESULT_FILE, PARAMETERS_FILE)
#: the directory of the profiles of the units of an identifiability analysis
PROFILES_DIR = "profiles"
#: the name of the report in the out of a report
REPORT_NAME = "report"
#: the directory of the PEtab problem in the out of a petab run
PETAB_DIR = "petab"
#: the directory of the records of the adopted fits in the out of a fit
FITS_DIR: Final = "fits"


# ----------------------------------------------------------------------------
# the specs
# ----------------------------------------------------------------------------
class FitProblemSpec(Spec):
    """The fit problem of a project which a spec of a fit kind names.

    Attributes:
        project: the project, `module:attribute`.
        subset: the fit problem, a key of the fits of the project.
    """

    project: str
    subset: str


class ProfileSpec(Spec):
    """The settings of profile likelihoods, of an analysis or after a fit.

    Every setting of `sbmlsim.fit.identifiability.ProfileSettings` with its
    default, `optimizer` its `optimizer_kwargs`.

    Attributes:
        alpha: the confidence level of the intervals.
        degrees_of_freedom: the degrees of freedom of the threshold.
        initial_step: the first step of a scan in decades.
        min_step: the smallest step in decades.
        max_step: the largest step in decades.
        step_factor: the factor a step is reduced or enlarged by.
        max_cost_fraction: the largest increase of the cost a single step may
            cause, as a fraction of the distance between the minimal cost and
            the threshold.
        max_points: the largest number of points of a scan in one direction.
        flatness: a profile which rises by less than this fraction of the
            distance between the minimal cost and the threshold, on both
            sides, is flat.
        reoptimize: optimize the other parameters at every point.
        optimizer: arguments of the least squares optimizer of the other
            parameters, `ProfileSettings.optimizer_kwargs`.
        pids: the parameters to profile, distinct parameters of the problem;
            the ones `profile_likelihood` profiles by default without, every
            parameter which is no element of a network (the plan of the kind
            `identifiability` sets them).
    """

    alpha: float = 0.95
    degrees_of_freedom: int = 1
    initial_step: float = 0.1
    min_step: float = 0.01
    max_step: float = 1.0
    step_factor: float = 2.0
    max_cost_fraction: float = 0.2
    max_points: int = 50
    flatness: float = 0.05
    reoptimize: bool = True
    optimizer: dict[str, Any] = Field(default_factory=dict)
    pids: list[str] | None = None


class FitSpec(FitProblemSpec, SplitSpec):
    """The spec of a fit: the repeats of the optimization of a problem.

    Attributes:
        opid: the id of the fit, the directory of its records; the plan sets
            the subset without, with `adopt` the id of a fit.
        repeats: the number of optimization runs.
        algorithm: the optimizer, least squares `LSQ` or differential
            evolution `DE`.
        seed: the seed of the start values and the runs; the plan draws one
            without.
        studies: the studies of the problem to fit, all without.
        timeout: the seconds a single run may take, no limit without.
        optimizer: further arguments of the optimizer, see `run_fit`.
        start: the point least squares starts from instead of samples of the
            bounds: the values by parameter id, a JSON file of them, the
            directory of the records of a fit or the id of an adopted fit of
            the project, see `sbmlsim.fit.project_fits.start_point`; needs
            `LSQ` and one repeat. The plan resolves it into the values, so
            every unit and a resume start from the same point, and keeps the
            given start in `start_from`; a relative path of a spec file is
            relative to the file, a fit id stays a name.
        start_from: the start as it was given, set by the plan, which the
            provenance of the fit records.
        record_adopted: record the adopted fits of the other subsets of the
            project, whose values the models of the project carry, and these
            values in the provenance of the fit; its report names them.
        fixed_fits: the adopted fits of the other subsets, set by the plan with
            `record_adopted`, so every unit records the same ones.
        fixed_values: the values of their parameters which the fit does not
            fit, set by the plan with them.
        identifiability: compute the profile likelihoods of the parameters of
            the best run after the fit, `<opid>/identifiability.json`.
        profile: the settings of these profiles.
        fisher: compute the Fisher information with the profiles,
            `<opid>/fisher.json`.
        report: write the report of the fit into `<opid>`, with the context
            of the project; true by default with `adopt`, so that the record
            of an adopted fit has its report.
        mapping_figures: draw the figure of every fit mapping in the report.
        adopt: adopt the fit, which needs its `identifiability`: copy its
            record into `fits/<opid>` and write `adopted.json`, the adopted
            fits of the project with this fit for its subset.
    """

    path_strings: ClassVar[tuple[str, ...]] = ("start",)

    opid: str | None = None
    repeats: int = Field(default=4, ge=1)
    algorithm: Literal["LSQ", "DE"] = "LSQ"
    seed: int | None = None
    studies: list[str] | None = None
    timeout: float | None = None
    optimizer: dict[str, Any] = Field(default_factory=dict)
    start: dict[str, float] | str | None = None
    start_from: str | None = planned()
    record_adopted: bool = True
    fixed_fits: list[str] | None = planned()
    fixed_values: dict[str, float] | None = planned()
    identifiability: bool = False
    profile: ProfileSpec = Field(default_factory=ProfileSpec)
    fisher: bool = True
    report: bool = False
    mapping_figures: bool = True
    adopt: bool = False

    @override
    @classmethod
    def is_name(cls, field: str, value: str) -> bool:
        """Check whether a start is the id of a fit, which is no path.

        Args:
            field: the name of the field, `start`.
            value: its string.

        Returns:
            Whether the string is one name of the form of the id of a fit,
            `<subset>_<date>_<time>__<hash>`, which the plan finds in the
            fits of the project.
        """
        from sbmlsim.fit.result import FIT_ID

        return (
            field == "start"
            and len(Path(value).parts) == 1
            and FIT_ID.match(value) is not None
        )

    @model_validator(mode="before")
    @classmethod
    def _report_of_an_adopted_fit(cls, data: Any) -> Any:
        """Report an adopted fit unless `report` is given."""
        if isinstance(data, dict) and data.get("adopt") and "report" not in data:
            return {**data, "report": True}
        return data


class AdoptSpec(Spec):
    """The spec of the adoption of a finished fit.

    Attributes:
        project: the project, `module:attribute`.
        fit: the records of the fit, the input `fit`: a records directory or
            the out of a fit run; the id of the fit, the id of its result,
            names its subset, `<subset>_<date>_<time>__<hash>`.
        force: adopt the fit although the project has adopted it already,
            i.e. replace its record.
    """

    project: str
    fit: Path | None = None
    force: bool = False


class IdentifiabilitySpec(FitProblemSpec, SplitSpec, ProfileSpec):
    """The spec of the profile likelihoods of the parameters of a fit.

    The settings of the profiles are the ones of `ProfileSpec`.

    Attributes:
        fit: the records of the fit, the input `fit`.
        fisher: compute the Fisher information as well, `fisher.json`.
    """

    fit: Path | None = None
    fisher: bool = True


class FisherSpec(FitProblemSpec):
    """The spec of the Fisher information of the parameters of a fit.

    Attributes:
        fit: the records of the fit, the input `fit`.
        step: the relative step of the finite differences, the default of
            `fisher_information` without.
    """

    fit: Path | None = None
    step: float | None = None


class ReportSpec(FitProblemSpec):
    """The spec of the report of the records of a fit.

    Attributes:
        subset: the fit problem; with `adopted` the subset whose adopted fit
            is reported, `all` (`sbmlsim.project.ALL_SUBSETS`) for every
            adopted fit.
        fit: the records, the input `fit`, e.g. the out of an
            identifiability analysis, whose profiles the report shows.
        adopted: report the adopted fits of the project from their records
            (`Project.adopted_record`) instead of the input `fit`, each into
            the directory of its fit id.
        mapping_figures: draw the figure of every fit mapping.
        group_by: the groupings of the report, the columns of one grouping
            per list; the groupings of the fit problem without.
    """

    fit: Path | None = None
    adopted: bool = False
    mapping_figures: bool = True
    group_by: list[list[str]] | None = None

    @model_validator(mode="after")
    def _check_adopted(self) -> ReportSpec:
        """Check that a report of the adopted fits has no input `fit`.

        Raises:
            ValueError: for a report of the adopted fits with an input `fit`.
        """
        if self.adopted and self.fit is not None:
            raise ValueError(
                "The report of the adopted fits reads their records in the "
                "project, not the input 'fit'."
            )
        return self


class PetabSpec(FitProblemSpec):
    """The spec of the PEtab v2 problem of a fit problem.

    Attributes:
        fit: the records of a fit, the problem starts from its best
            parameters; the start values of the problem without.
        required_extension: mark the extension `sbmlsim` as required.
    """

    fit: Path | None = None
    required_extension: bool = True


# ----------------------------------------------------------------------------
# what the kinds share
# ----------------------------------------------------------------------------


def _definition(spec: FitProblemSpec, subset: str | None = None) -> FitDefinition:
    """Get the fit problem a spec names, or another subset of its project.

    Raises:
        ConfigurationError: for a project which is not loaded or a subset it
            does not have.
    """
    subset = spec.subset if subset is None else subset
    try:
        return load_project_ref(spec.project).fit(subset)
    except (ValueError, TypeError, KeyError) as err:
        message = err.args[0] if isinstance(err, KeyError) and err.args else err
        raise ConfigurationError(
            f"The fit problem '{subset}' of the project '{spec.project}' "
            f"is not resolved: {message}"
        ) from err
    except ConfigurationError as err:
        raise ConfigurationError(
            f"The fit problem '{subset}' of the project '{spec.project}' "
            f"is not resolved: {err}"
        ) from err


def _records_dir(fit: Path | None) -> Path:
    """Get the directory of the records of a fit from the input `fit`.

    Args:
        fit: a records directory, with `parameters.json`, or the out of a fit
            run, which has one in its directory `<opid>`.

    Raises:
        ConfigurationError: without the input or without records in it.
    """
    if fit is None:
        raise ConfigurationError(
            "The kind needs the records of a fit, the input 'fit' (--input fit=DIR)."
        )
    if (fit / PARAMETERS_FILE).exists():
        return fit
    candidates = (
        sorted(d for d in fit.iterdir() if (d / PARAMETERS_FILE).exists())
        if fit.is_dir()
        else []
    )
    if len(candidates) != 1:
        raise ConfigurationError(
            f"The input fit '{fit}' is no directory of the records of a fit: it "
            f"has no {PARAMETERS_FILE} and not exactly one directory with one, "
            f"it has {[d.name for d in candidates]}."
        )
    return candidates[0]


def _fit_opid(records: FitRecords, subset: str) -> str:
    """Get the id of a fit from its records, its subset without."""
    opt_result = records.opt_result
    if opt_result is not None and opt_result.opid:
        return opt_result.opid
    return subset


def _stored_fit(
    spec: FitProblemSpec, directory: Path
) -> tuple[OptimizationProblem, FitSettings, ParameterSet]:
    """Get the problem, the settings and the best parameter set of a stored fit.

    The problem is the one of the studies the fit was restricted to and the
    settings are the ones the fit ran with, both stored in its records, see
    `sbmlsim.fit.cli.problem_of_records` and `sbmlsim.fit.cli.settings_of`.

    Args:
        spec: the spec, it names the problem.
        directory: the records of the fit.

    Returns:
        The uninitialized problem, the settings and the first parameter set of
        `parameters.json`.

    Raises:
        ConfigurationError: if a study of the fit is no study of the problem.
    """
    from sbmlsim.fit.cli import (
        load_fit_records,
        load_parameter_sets,
        problem_of_records,
        settings_of,
    )

    definition = _definition(spec)
    records = load_fit_records(directory)
    try:
        problem = problem_of_records(
            definition, records, opid=_fit_opid(records, spec.subset)
        )
    except KeyError as err:
        raise ConfigurationError(
            f"The fit in '{directory}' is not a fit of the problem "
            f"'{spec.subset}': {err.args[0] if err.args else err}"
        ) from err
    settings = settings_of(definition, records)
    parameter_set = load_parameter_sets([directory / PARAMETERS_FILE])[0]
    return problem, settings, parameter_set


def _copy_records(directory: Path, ctx: RunContext) -> None:
    """Copy the records of a fit into the out, atomically, the ones it has."""
    for name in FIT_RECORDS:
        source, target = directory / name, ctx.out / name
        if not source.exists() or (target.exists() and source.samefile(target)):
            continue
        with atomic_path(target) as temporary:
            shutil.copyfile(source, temporary)


def _write_fisher(
    directory: Path,
    problem: OptimizationProblem,
    settings: FitSettings,
    parameter_set: ParameterSet,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute the Fisher information, write `fisher.json` and summarize it."""
    from sbmlsim.fit.fisher import fisher_information

    fisher = fisher_information(
        problem=problem, settings=settings, parameter_set=parameter_set, **kwargs
    )
    write_text_atomic(
        directory / FISHER_FILE, json.dumps(fisher.to_dict(), indent=2) + "\n"
    )
    return {"rank": fisher.rank, "condition": fisher.condition_number}


# ----------------------------------------------------------------------------
# fit
# ----------------------------------------------------------------------------
def _check_opid(opid: str) -> None:
    """Check that the id of a fit is one plain path component.

    The records of a fit are written into `out/<opid>`, so an id which is
    empty, `.` or `..`, has a separator or is absolute would write them
    elsewhere.

    Raises:
        ConfigurationError: for such an id.
    """
    if (
        opid in ("", ".", "..")
        or "/" in opid
        or "\\" in opid
        or PurePosixPath(opid).is_absolute()
        or PureWindowsPath(opid).drive
    ):
        raise ConfigurationError(
            f"The id of a fit is the name of a directory in the output, without "
            f"a separator, not '{opid}'."
        )


#: the arguments of `run_optimization` a fit takes in `FitSpec.optimizer`
_RUNNER_ARGUMENTS: frozenset[str] = frozenset({"sampling", "serial", "show_progress"})


def _sampling(value: Any) -> SamplingType:
    """Get the sampling of the start values of a fit from a spec.

    Args:
        value: a `SamplingType`, its name (any case) or its value, as a YAML
            or JSON spec gives it.

    Raises:
        ConfigurationError: for a value which is no sampling.
    """
    from sbmlsim.fit.sampling import SamplingType

    if isinstance(value, SamplingType):
        return value
    if isinstance(value, str) and value.upper() in SamplingType.__members__:
        return SamplingType[value.upper()]
    if isinstance(value, int) and not isinstance(value, bool):
        try:
            return SamplingType(value)
        except ValueError:
            pass
    raise ConfigurationError(
        f"The sampling {value!r} of the optimizer is none of "
        f"{list(SamplingType.__members__)}."
    )


def _optimizer_arguments(spec: FitSpec) -> dict[str, Any]:
    """Get the arguments of the optimizer of a fit, its `sampling` a `SamplingType`.

    Raises:
        ConfigurationError: for a sampling which is no `SamplingType`.
    """
    arguments = dict(spec.optimizer)
    if "sampling" in arguments:
        arguments["sampling"] = _sampling(arguments["sampling"])
    return arguments


def _check_fit_arguments(spec: FitSpec, definition: FitDefinition) -> None:
    """Check the studies and the arguments of the optimizer of a fit.

    Raises:
        ConfigurationError: for a study which is no study of the problem, an
            argument of the optimizer which is a setting of the spec, and one
            which the optimizer does not accept.
    """
    import inspect

    from sbmlsim.fit.cli import ALGORITHM_KWARGS, ALGORITHMS, run_fit
    from sbmlsim.fit.optimization import OptimizationProblem
    from sbmlsim.fit.runner import run_optimization

    if spec.studies:
        try:
            definition.collections(study_ids=spec.studies)
        except KeyError as err:
            raise ConfigurationError(
                f"The studies of the fit are invalid: "
                f"{err.args[0] if err.args else err}"
            ) from err
    # the arguments of the runner which `run_fit` does not set, e.g. the
    # sampling of the start values, are arguments of a fit as well
    runner_arguments = set(inspect.signature(run_optimization).parameters)
    fixed = sorted(
        set(spec.optimizer)
        & (set(inspect.signature(run_fit).parameters) | runner_arguments)
        - {"kwargs", *_RUNNER_ARGUMENTS}
    )
    if fixed:
        raise ConfigurationError(
            f"The arguments {fixed} of the optimizer are settings of the fit, "
            "set them in the spec."
        )
    algorithm = ALGORITHMS[spec.algorithm]
    arguments = {
        name: value
        for name, value in {**ALGORITHM_KWARGS[algorithm], **spec.optimizer}.items()
        if name not in runner_arguments
    }
    try:
        OptimizationProblem.check_optimizer_arguments(algorithm, arguments)
    except (TypeError, ValueError) as err:
        raise ConfigurationError(
            f"The arguments of the optimizer are invalid: {err}"
        ) from err


def _start_point(spec: FitSpec, definition: FitDefinition) -> dict[str, float]:
    """Get the point a fit starts from, see `start_point`.

    Raises:
        ConfigurationError: for a start which is no point, file or fit, or
            whose values are no start values of the parameters of the fit.
    """
    from sbmlsim.fit.project_fits import start_point

    assert spec.start is not None
    try:
        return start_point(load_project_ref(spec.project), definition, spec.start)
    except (ValueError, OSError) as err:
        raise ConfigurationError(f"The start of the fit is invalid: {err}") from err


def _fixed(
    spec: FitSpec, definition: FitDefinition
) -> tuple[list[str], dict[str, float]]:
    """Get the adopted fits a fit keeps fixed and their values, see `fixed_values`.

    The ones of a planned spec, else read from the project.

    Raises:
        ConfigurationError: if the records of the adopted fits are not read.
    """
    from sbmlsim.fit.project_fits import fixed_values

    if not spec.record_adopted:
        return [], {}
    if spec.fixed_fits is not None:
        return list(spec.fixed_fits), dict(spec.fixed_values or {})
    try:
        return fixed_values(load_project_ref(spec.project), spec.subset, definition)
    except (ValueError, OSError) as err:
        raise ConfigurationError(
            f"The adopted fits of '{spec.project}' are not read: {err}"
        ) from err


def _check_start(spec: FitSpec, definition: FitDefinition) -> None:
    """Check that a fit can start from its start.

    Raises:
        ConfigurationError: for a start of differential evolution, of several
            repeats, with a sampling of the optimizer or which is invalid.
    """
    if spec.start is None:
        return
    if spec.algorithm != "LSQ":
        raise ConfigurationError(
            "The start is the start of least squares; differential evolution "
            "draws its population from the bounds."
        )
    if spec.repeats != 1:
        raise ConfigurationError(
            "The start starts every repeat from the point, the repeats would be "
            "identical: give one repeat."
        )
    if "sampling" in spec.optimizer:
        raise ConfigurationError(
            "A fit with a start samples no start values, it has no 'sampling'."
        )


def _check_remainders(
    spec: FitSpec, definition: FitDefinition, fixed: dict[str, float]
) -> None:
    """Check that the bounds of a fit let no remainder of a model become negative.

    Every variant of the project is checked, see `remainder_problems`; a
    project without model variants has no model to check, which is logged.

    Raises:
        ConfigurationError: naming the remainders which can become negative.
    """
    from sbmlsim.fit.project_fits import remainder_problems

    project = load_project_ref(spec.project)
    if not project.variants:
        logger.info(
            "'%s': the project '%s' has no model variants, the remainders of "
            "its models are not checked.",
            spec.subset,
            project.name,
        )
        return
    problems = list(
        dict.fromkeys(
            problem
            for variant in project.variants
            for problem in remainder_problems(
                project.variant_path(variant), definition.parameters, fixed
            )
        )
    )
    if problems:
        raise ConfigurationError("; ".join(problems))


def _plan_opid(spec: FitSpec) -> str:
    """Get the id of a fit: the subset, or the id of a fit when it is adopted.

    Raises:
        ConfigurationError: for an id which is no name of a directory, or an
            adopted fit whose id is not the id of a fit of its subset.
    """
    from sbmlsim.fit.result import fit_id, fit_subset

    if spec.opid is None:
        opid = fit_id(spec.subset) if spec.adopt else spec.subset
    else:
        opid = spec.opid
    _check_opid(opid)
    if spec.adopt:
        try:
            subset = fit_subset(opid)
        except ValueError as err:
            raise ConfigurationError(
                f"An adopted fit has the id of a fit: {err}"
            ) from err
        if subset != spec.subset:
            raise ConfigurationError(
                f"The adopted fit '{opid}' is a fit of '{subset}', not of the "
                f"problem '{spec.subset}'."
            )
    return opid


def _plan_fit(spec: FitSpec) -> FitSpec:
    """Plan a fit: its id, its seed and its units.

    Raises:
        ConfigurationError: for an unknown problem or study, an id which is
            no name of a directory, invalid arguments of the optimizer, an
            invalid start or profile settings, an adopted fit without its
            identifiability and a remainder which can become negative.
    """
    definition = _definition(spec)
    opid = _plan_opid(spec)
    _check_fit_arguments(spec, definition)
    _check_start(spec, definition)
    start = None if spec.start is None else _start_point(spec, definition)
    start_from = (
        spec.start_from
        if spec.start_from is not None or not isinstance(spec.start, str)
        else spec.start
    )
    if spec.identifiability:
        _profile_settings(spec.profile)
        if spec.profile.pids is not None:
            _check_pids(spec.profile.pids, definition, spec.subset)
    elif spec.adopt:
        raise ConfigurationError(
            "An adopted fit carries the uncertainty of its parameters: adopt a "
            "fit with its 'identifiability'."
        )
    fixed_fits, fixed = _fixed(spec, definition)
    _check_remainders(spec, definition, fixed)
    seed = (
        spec.seed
        if spec.seed is not None
        else int(np.random.default_rng().integers(2**31))
    )
    optimizer = _optimizer_arguments(spec)
    if "sampling" in optimizer:
        # the name, which the JSON of the planned spec keeps
        optimizer["sampling"] = optimizer["sampling"].name
    return spec.model_copy(
        update={
            "opid": opid,
            "seed": seed,
            "units": len(unit_blocks(spec.repeats)),
            "optimizer": optimizer,
            "start": start,
            "start_from": start_from,
            "fixed_fits": fixed_fits if spec.record_adopted else None,
            "fixed_values": fixed if spec.record_adopted else None,
        }
    )


def _planned_fit(spec: FitSpec) -> tuple[str, list[range]]:
    """Get the id and the blocks of repeats of a planned fit.

    Raises:
        ConfigurationError: for a spec which is not planned.
        PlanMismatch: for units which are not the blocks of the repeats.
    """
    if spec.opid is None or spec.seed is None or spec.units is None:
        raise ConfigurationError("The spec of the fit is not planned: plan it first.")
    _check_opid(spec.opid)
    blocks = unit_blocks(spec.repeats)
    if spec.units != len(blocks):
        raise PlanMismatch(
            f"The fit has {spec.units} units, its {spec.repeats} repeats are "
            f"{len(blocks)} units; plan the spec again."
        )
    return spec.opid, blocks


def _runs_dir(ctx: RunContext, opid: str) -> Path:
    """Get the directory of the repeats of a fit."""
    return ctx.out / opid / "runs"


def _run_file(ctx: RunContext, opid: str, k: int) -> Path:
    """Get the file of the repeat `k` of a fit, see `run_optimization`."""
    return _runs_dir(ctx, opid) / f"{opid}_run_{k}.json"


def _run_repeats(
    spec: FitSpec, opid: str, ctx: RunContext, indices: list[int], n_cores: int
) -> None:
    """Run repeats of a fit, which write their files into the out.

    Raises:
        KeyboardInterrupt: if the fit was interrupted (Ctrl-C), also after
            some of the repeats finished, which are kept.
        RuntimeError: naming the repeats which were lost otherwise, e.g. by a
            worker which died.
    """
    from sbmlsim.fit.cli import ALGORITHMS, run_fit
    from sbmlsim.fit.project_fits import FIXED_FITS, with_start
    from sbmlsim.fit.sampling import SamplingType

    logger.info("'%s': the repeats %s of %d run.", opid, indices, spec.repeats)
    definition = _definition(spec)
    optimizer = _optimizer_arguments(spec)
    arguments: dict[str, Any] = {}
    if spec.start is not None:
        if not isinstance(spec.start, dict):
            raise ConfigurationError(
                "The start of the fit is not planned: plan the spec first."
            )
        # the point the plan resolved, so every unit and a resume start alike
        definition = with_start(definition, spec.start)
        optimizer["sampling"] = SamplingType.START
        arguments["start"] = (
            spec.start_from if spec.start_from is not None else spec.start
        )
    fixed_fits, fixed = _fixed(spec, definition)
    if spec.record_adopted:
        arguments[FIXED_FITS] = fixed_fits
    runs = run_fit(
        definition,
        opid=opid,
        algorithm=ALGORITHMS[spec.algorithm],
        size=spec.repeats,
        n_cores=n_cores,
        seed=spec.seed,
        study_ids=spec.studies,
        timeout=spec.timeout,
        output_dir=ctx.out,
        arguments=arguments,
        fixed_values=fixed,
        indices=indices,
        sources=ctx.provenance,
        **optimizer,
    )
    if any(fit_run.result.interrupted for fit_run in runs.values()):
        raise KeyboardInterrupt
    lost = [k for k in indices if not _run_file(ctx, opid, k).exists()]
    if lost:
        raise RuntimeError(
            f"'{opid}': the repeats {lost} were lost, see the log; the run with "
            "--resume, or the unit again, runs them again and keeps the others."
        )


def _compute_fit_unit(spec: FitSpec, index: int, ctx: RunContext) -> None:
    """Run the block of repeats of a unit serially, the ones without a file."""
    opid, blocks = _planned_fit(spec)
    missing = [k for k in blocks[index] if not _run_file(ctx, opid, k).exists()]
    if missing:
        _run_repeats(spec, opid, ctx, missing, n_cores=1)


def _merge_fit(spec: FitSpec, ctx: RunContext) -> dict[str, Any]:
    """Merge the repeats of a fit into its records.

    Raises:
        MissingUnits: naming the units with repeats without a file.
    """
    from sbmlsim.fit.result import OptimizationResult

    opid, blocks = _planned_fit(spec)
    missing = [
        index
        for index, block in enumerate(blocks)
        if not all(_run_file(ctx, opid, k).exists() for k in block)
    ]
    if missing:
        raise MissingUnits(missing, len(blocks))
    result = OptimizationResult.from_directory(_runs_dir(ctx, opid), sid=opid)
    records = ctx.out / opid
    write_text_atomic(records / RESULT_FILE, str(result.to_json()))
    write_text_atomic(records / PARAMETERS_FILE, str(result.parameter_sets().to_json()))
    with atomic_path(records / FITS_FILE) as temporary:
        result.to_tsv(temporary)
    df = result.df_fits
    summary: dict[str, Any] = {
        "opid": opid,
        "repeats": spec.repeats,
        "converged": int(df.success.astype(bool).sum()),
        "best_cost": float(df.cost.iloc[0]),
    }
    if spec.identifiability:
        summary.update(_fit_identifiability(spec, records, ctx))
    if spec.report:
        _fit_report(spec, opid, records, ctx)
    if spec.adopt:
        _adopt(spec, spec.subset, opid, records, ctx, force=True)
        summary["adopted"] = f"{FITS_DIR}/{opid}"
    return summary


def _fit_identifiability(
    spec: FitSpec, records: Path, ctx: RunContext
) -> dict[str, Any]:
    """Profile the parameters of the best run of a fit, with its Fisher information.

    The records of the fit get `identifiability.json` and, with `fisher`,
    `fisher.json`; a resumed run keeps the ones which exist.
    """
    from sbmlsim.fit.identifiability import IdentifiabilityResult, profile_likelihood

    problem, settings, parameter_set = _stored_fit(spec, records)
    path = records / IDENTIFIABILITY_FILE
    if ctx.resume and path.exists():
        result = IdentifiabilityResult.from_json(path)
    else:
        result = profile_likelihood(
            problem=problem,
            settings=settings,
            parameter_set=parameter_set,
            profile_settings=_profile_settings(spec.profile),
            pids=spec.profile.pids,
            n_cores=ctx.workers or available_cpus(),
        )
        write_text_atomic(path, str(result.to_json()) + "\n")
    summary: dict[str, Any] = {
        "n_identifiable": result.n_identifiable,
        "n_parameters": len(result.profiles),
    }
    if spec.fisher and not (ctx.resume and (records / FISHER_FILE).exists()):
        _write_fisher(
            records, problem, settings, parameter_set, alpha=result.settings.alpha
        )
    return summary


def _fit_report(spec: FitSpec, opid: str, records: Path, ctx: RunContext) -> None:
    """Write the report of a fit into its records, with the context of the project."""
    from sbmlsim.fit.cli import load_fit_records, records_context, report_from_records

    definition = _definition(spec)
    loaded = load_fit_records(records)
    report = report_from_records(
        definition,
        records,
        opid=opid,
        records=loaded,
        context=records_context(load_project_ref(spec.project))(
            spec.subset, definition, loaded
        ),
        mapping_figures=spec.mapping_figures,
    )
    report.create(output_dir=ctx.out, name=opid)


def _adopt(
    spec: FitProblemSpec | AdoptSpec,
    subset: str,
    opid: str,
    records: Path,
    ctx: RunContext,
    *,
    force: bool,
) -> None:
    """Adopt a fit: its record into `fits/<opid>` and `adopted.json` into the out.

    Args:
        spec: the spec, it names the project.
        subset: the subset of the fit.
        opid: the id of the fit.
        records: the records of the fit.
        ctx: the context of the run.
        force: adopt a fit which the project has adopted already.

    Raises:
        ConfigurationError: if the adopted fits of the project are not read,
            the project has adopted the fit already and `force` is not set, or
            the fit has no record of the uncertainty of its parameters.
    """
    from sbmlsim.fit.records import adopt_fit
    from sbmlsim.project import ADOPTED_FILE, write_adopted

    project = load_project_ref(spec.project)
    try:
        adopted = project.adopted_fits()
    except ValueError as err:
        raise ConfigurationError(
            f"The adopted fits of '{spec.project}' are not read: {err}"
        ) from err
    if (
        not force
        and project.fits_dir is not None
        and (project.fits_dir / opid).exists()
    ):
        raise ConfigurationError(
            f"The fit '{opid}' is adopted already: {project.fits_dir / opid}; "
            "'force' adopts it again."
        )
    try:
        # the out of a resumed run has the record already
        adopt_fit(records, ctx.out / FITS_DIR, force=True, fit_id=opid)
    except FileNotFoundError as err:
        raise ConfigurationError(f"The fit '{opid}' is not adopted: {err}") from err
    write_adopted(ctx.out / ADOPTED_FILE, {**adopted, subset: opid})


def _run_fit(spec: FitSpec, ctx: RunContext) -> dict[str, Any]:
    """Run the repeats of a fit, with `resume` the ones without a file, and merge."""
    opid, _ = _planned_fit(spec)
    missing = [
        k
        for k in range(spec.repeats)
        if not (ctx.resume and _run_file(ctx, opid, k).exists())
    ]
    if missing:
        _run_repeats(spec, opid, ctx, missing, n_cores=ctx.workers or available_cpus())
    return _merge_fit(spec, ctx)


# ----------------------------------------------------------------------------
# identifiability
# ----------------------------------------------------------------------------
def _profile_settings(spec: ProfileSpec) -> ProfileSettings:
    """Get the settings of the profiles of a spec.

    Raises:
        ConfigurationError: for settings out of their range.
    """
    from sbmlsim.fit.identifiability import ProfileSettings

    try:
        return ProfileSettings(
            alpha=spec.alpha,
            degrees_of_freedom=spec.degrees_of_freedom,
            initial_step=spec.initial_step,
            min_step=spec.min_step,
            max_step=spec.max_step,
            step_factor=spec.step_factor,
            max_cost_fraction=spec.max_cost_fraction,
            max_points=spec.max_points,
            flatness=spec.flatness,
            reoptimize=spec.reoptimize,
            optimizer_kwargs=spec.optimizer,
        )
    except ValueError as err:
        raise ConfigurationError(f"The profile settings are invalid: {err}") from err


def _plan_identifiability(spec: IdentifiabilitySpec) -> IdentifiabilitySpec:
    """Plan the profiles: the parameters, one unit each."""
    from sbmlsim.fit.derived import group_parameters, hook_summaries

    definition = _definition(spec)
    _profile_settings(spec)
    pids = spec.pids
    if pids is None:
        single, _ = group_parameters(
            definition.parameters, hook_summaries(definition.hybridizations)
        )
        pids = [p.pid for p in single]
    _check_pids(pids, definition, spec.subset)
    return spec.model_copy(update={"pids": list(pids), "units": len(pids)})


def _check_pids(pids: list[str], definition: FitDefinition, subset: str) -> None:
    """Check the parameters to profile.

    Raises:
        ConfigurationError: for no parameters or parameters which are not
            distinct parameters of the problem.
    """
    known = [p.pid for p in definition.parameters]
    unknown = [pid for pid in pids if pid not in known]
    if unknown or not pids or len(set(pids)) != len(pids):
        raise ConfigurationError(
            f"The parameters to profile {pids} are not distinct parameters of "
            f"the problem '{subset}', which has {known}."
        )


def _planned_pids(spec: IdentifiabilitySpec) -> list[str]:
    """Get the parameters of planned profiles.

    Raises:
        ConfigurationError: for a spec which is not planned.
        PlanMismatch: for units which are not the parameters.
    """
    if spec.pids is None or spec.units is None:
        raise ConfigurationError(
            "The spec of the profiles is not planned: plan it first."
        )
    if spec.units != len(spec.pids):
        raise PlanMismatch(
            f"The profiles have {spec.units} units, their {len(spec.pids)} "
            "parameters are one unit each; plan the spec again."
        )
    return list(spec.pids)


def _profile_file(ctx: RunContext, pid: str) -> Path:
    """Get the file of the profile of a parameter."""
    return ctx.out / PROFILES_DIR / f"{pid}.json"


def _profile(
    spec: IdentifiabilitySpec, ctx: RunContext, pids: list[str], n_cores: int
) -> None:
    """Profile parameters, every profile written to its file once it finished."""
    from sbmlsim.fit.identifiability import IdentifiabilityResult, profile_likelihood

    problem, settings, parameter_set = _stored_fit(spec, _records_dir(spec.fit))
    (ctx.out / PROFILES_DIR).mkdir(parents=True, exist_ok=True)

    def write(part: IdentifiabilityResult) -> None:
        for pid in part.profiles:
            write_text_atomic(_profile_file(ctx, pid), str(part.to_json()) + "\n")

    profile_likelihood(
        problem=problem,
        settings=settings,
        parameter_set=parameter_set,
        profile_settings=_profile_settings(spec),
        pids=pids,
        n_cores=n_cores,
        on_profile=write,
    )


def _compute_identifiability_unit(
    spec: IdentifiabilitySpec, index: int, ctx: RunContext
) -> None:
    """Profile the parameter of a unit, unless its file exists."""
    pid = _planned_pids(spec)[index]
    if not _profile_file(ctx, pid).exists():
        _profile(spec, ctx, [pid], n_cores=1)


def _merge_identifiability(
    spec: IdentifiabilitySpec, ctx: RunContext
) -> dict[str, Any]:
    """Combine the profiles, copy the records of the fit, add its Fisher information.

    Raises:
        MissingUnits: naming the units whose profile has no file.
    """
    from sbmlsim.fit.identifiability import IdentifiabilityResult

    pids = _planned_pids(spec)
    missing = [k for k, pid in enumerate(pids) if not _profile_file(ctx, pid).exists()]
    if missing:
        raise MissingUnits(missing, len(pids))
    result = IdentifiabilityResult.combine(
        [IdentifiabilityResult.from_json(_profile_file(ctx, pid)) for pid in pids]
    )
    write_text_atomic(ctx.out / IDENTIFIABILITY_FILE, str(result.to_json()) + "\n")
    directory = _records_dir(spec.fit)
    _copy_records(directory, ctx)
    if spec.fisher:
        problem, settings, parameter_set = _stored_fit(spec, directory)
        _write_fisher(ctx.out, problem, settings, parameter_set, alpha=spec.alpha)
    return {"n_identifiable": result.n_identifiable, "n_parameters": len(pids)}


def _run_identifiability(spec: IdentifiabilitySpec, ctx: RunContext) -> dict[str, Any]:
    """Profile the parameters, with `resume` the ones without a file, and merge."""
    pids = _planned_pids(spec)
    missing = [
        pid for pid in pids if not (ctx.resume and _profile_file(ctx, pid).exists())
    ]
    if missing:
        _profile(spec, ctx, missing, n_cores=ctx.workers or available_cpus())
    return _merge_identifiability(spec, ctx)


# ----------------------------------------------------------------------------
# fisher, report, petab
# ----------------------------------------------------------------------------
def _run_fisher(spec: FisherSpec, ctx: RunContext) -> dict[str, Any]:
    """Compute the Fisher information and copy the records of the fit."""
    directory = _records_dir(spec.fit)
    problem, settings, parameter_set = _stored_fit(spec, directory)
    summary = _write_fisher(ctx.out, problem, settings, parameter_set, step=spec.step)
    _copy_records(directory, ctx)
    return summary


def _run_report(spec: ReportSpec, ctx: RunContext) -> dict[str, Any]:
    """Write the report of the records of a fit into `report/`.

    With `adopted` the report of every adopted fit of the project, of the
    subset if the spec names one, into the directory of its subset.

    Raises:
        ConfigurationError: for a project without adopted fits or an adopted
            fit without its record.
    """
    from sbmlsim.fit.cli import load_fit_records, records_context, report_from_records
    from sbmlsim.fit.evaluation import Grouping
    from sbmlsim.project import ALL_SUBSETS

    project = load_project_ref(spec.project)
    context = records_context(project)
    groupings = (
        None
        if spec.group_by is None
        else [Grouping.by(*columns) for columns in spec.group_by]
    )
    if not spec.adopted:
        subset = spec.subset
        directory = _records_dir(spec.fit)
        definition = _definition(spec)
        records = load_fit_records(directory)
        report = report_from_records(
            definition,
            directory,
            opid=_fit_opid(records, subset),
            records=records,
            context=context(subset, definition, records),
            mapping_figures=spec.mapping_figures,
            groupings=groupings,
        )
        path = report.create(output_dir=ctx.out, name=REPORT_NAME)
        return {"report": path.relative_to(ctx.out).as_posix()}

    try:
        adopted = project.adopted_fits()
    except ValueError as err:
        raise ConfigurationError(
            f"The adopted fits of '{spec.project}' are not read: {err}"
        ) from err
    if spec.subset != ALL_SUBSETS:
        adopted = {k: v for k, v in adopted.items() if k == spec.subset}
    if not adopted:
        raise ConfigurationError(
            f"The project '{spec.project}' has no adopted fit"
            + ("." if spec.subset == ALL_SUBSETS else f" of '{spec.subset}'.")
        )
    reports: dict[str, str] = {}
    for subset, fit_id in adopted.items():
        try:
            directory = project.adopted_record(fit_id)
        except (ValueError, FileNotFoundError) as err:
            raise ConfigurationError(
                f"The adopted fit '{fit_id}' is not reported: {err}"
            ) from err
        definition = _definition(spec, subset)
        records = load_fit_records(directory)
        report = report_from_records(
            definition,
            directory,
            opid=subset,
            records=records,
            context=context(subset, definition, records),
            mapping_figures=spec.mapping_figures,
            groupings=groupings,
        )
        path = report.create(output_dir=ctx.out, name=fit_id)
        reports[subset] = path.relative_to(ctx.out).as_posix()
    return {"reports": reports}


def _run_adopt(spec: AdoptSpec, ctx: RunContext) -> dict[str, Any]:
    """Adopt a finished fit: its record into `fits/<fit id>` and `adopted.json`.

    The id of the fit is the id of its result, else the name of its records
    directory, and has the form of the id of a fit of a subset of the project.

    Raises:
        ConfigurationError: for an input which is no records of a fit, an id
            which is no id of a fit of a subset of the project, a fit the
            project has adopted already without `force`, and a fit without
            the uncertainty of its parameters.
    """
    from sbmlsim.fit.cli import load_fit_records
    from sbmlsim.fit.result import fit_subset

    records = _records_dir(spec.fit)
    opid = _fit_opid(load_fit_records(records), records.name)
    try:
        subset = fit_subset(opid)
    except ValueError as err:
        raise ConfigurationError(f"The fit is not adopted: {err}") from err
    try:
        load_project_ref(spec.project).fit(subset)
    except (KeyError, TypeError, ValueError) as err:
        raise ConfigurationError(
            f"The fit '{opid}' is no fit of a subset of '{spec.project}': "
            f"{err.args[0] if err.args else err}"
        ) from err
    _check_opid(opid)
    _adopt(spec, subset, opid, records, ctx, force=spec.force)
    return {"subset": subset, "opid": opid, "adopted": f"{FITS_DIR}/{opid}"}


def _run_petab(spec: PetabSpec, ctx: RunContext) -> dict[str, Any]:
    """Write the problem as PEtab v2, from the best parameters of a given fit."""
    from sbmlsim.fit.petab_v2.export import to_petab

    if spec.fit is None:
        definition = _definition(spec)
        problem = definition.problem(opid=spec.subset)
        settings, parameter_set = definition.settings, None
    else:
        problem, settings, parameter_set = _stored_fit(spec, _records_dir(spec.fit))
    path = to_petab(
        problem,
        ctx.out / PETAB_DIR,
        settings=settings,
        parameter_set=parameter_set,
        required_extension=spec.required_extension,
    )
    return {"problem": Path(path).resolve().relative_to(ctx.out.resolve()).as_posix()}


#: the fit of a problem, split into blocks of repeats
FIT: Kind[FitSpec] = Kind(
    name="fit",
    spec=FitSpec,
    run=_run_fit,
    plan=_plan_fit,
    compute_unit=_compute_fit_unit,
    merge=_merge_fit,
    cost_setting="subset",
    description="Fit of a project, split into its repeats",
)

#: the profile likelihoods of a fit, one unit per parameter
IDENTIFIABILITY: Kind[IdentifiabilitySpec] = Kind(
    name="identifiability",
    spec=IdentifiabilitySpec,
    run=_run_identifiability,
    plan=_plan_identifiability,
    compute_unit=_compute_identifiability_unit,
    merge=_merge_identifiability,
    inputs=("fit",),
    cost_setting="subset",
    description="Profile likelihoods of a fit, a parameter per unit",
)

#: the Fisher information of a fit, one jacobian in one process
FISHER: Kind[FisherSpec] = Kind(
    name="fisher",
    spec=FisherSpec,
    run=_run_fisher,
    inputs=("fit",),
    uses_cores=False,
    description="Fisher information of a fit",
)

#: the report of the records of a fit
REPORT: Kind[ReportSpec] = Kind(
    name="report",
    spec=ReportSpec,
    run=_run_report,
    inputs=("fit",),
    uses_cores=False,
    description="HTML report of a fit with its profiles and metrics",
)

#: the PEtab v2 problem of a fit problem
PETAB: Kind[PetabSpec] = Kind(
    name="petab",
    spec=PetabSpec,
    run=_run_petab,
    inputs=("fit",),
    uses_cores=False,
    description="PEtab v2 problem of a fit",
)

#: the adoption of a finished fit
ADOPT: Kind[AdoptSpec] = Kind(
    name="adopt",
    spec=AdoptSpec,
    run=_run_adopt,
    inputs=("fit",),
    uses_cores=False,
    description="Adopt a fit as a record of the project",
)

for _kind in (FIT, IDENTIFIABILITY, FISHER, REPORT, PETAB, ADOPT):
    register(_kind)
