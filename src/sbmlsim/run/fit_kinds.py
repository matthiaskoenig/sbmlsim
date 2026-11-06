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
  and `<opid>/fits.tsv`, with `adopt` also `adopted.json`, the adopted fits of
  the project with the fit adopted for its subset (the file of the project is
  never written).
- `identifiability` (`IdentifiabilitySpec`) computes the profile likelihoods,
  one unit per parameter, `profiles/<pid>.json`, which the merge combines
  into `identifiability.json`, with `fisher` also `fisher.json`.
- `fisher` (`FisherSpec`) computes the Fisher information, `fisher.json`.
- `report` (`ReportSpec`) writes the report of the records, `report/`.
- `petab` (`PetabSpec`) writes the problem as PEtab v2, `petab/problem.yaml`,
  which starts from the best parameters of a fit when one is given.

Every record file is written atomically, so a unit computed again leaves
complete files. The modules of the fit are imported when a kind runs, not
when this module is imported.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import Field

from sbmlsim.atomic import atomic_path, write_text_atomic
from sbmlsim.cgroup import available_cpus
from sbmlsim.run.kinds import (
    ConfigurationError,
    Kind,
    PlanMismatch,
    Spec,
    SplitSpec,
    register,
)
from sbmlsim.run.units import MissingUnits, unit_blocks

if TYPE_CHECKING:
    from sbmlsim.fit.cli import FitDefinition, FitRecords
    from sbmlsim.fit.identifiability import ProfileSettings
    from sbmlsim.fit.optimization import OptimizationProblem
    from sbmlsim.fit.options import FitSettings
    from sbmlsim.fit.parameters import ParameterSet
    from sbmlsim.run.context import RunContext

__all__ = [
    "FISHER",
    "FIT",
    "IDENTIFIABILITY",
    "PETAB",
    "REPORT",
    "FisherSpec",
    "FitProblemSpec",
    "FitSpec",
    "IdentifiabilitySpec",
    "PetabSpec",
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


class FitSpec(FitProblemSpec, SplitSpec):
    """The spec of a fit: the repeats of the optimization of a problem.

    Attributes:
        opid: the id of the fit, the directory of its records; the plan sets
            the subset without.
        repeats: the number of optimization runs.
        algorithm: the optimizer, least squares `LSQ` or differential
            evolution `DE`.
        seed: the seed of the start values and the runs; the plan draws one
            without.
        studies: the studies of the problem to fit, all without.
        timeout: the seconds a single run may take, no limit without.
        optimizer: further arguments of the optimizer, see `run_fit`.
        adopt: write `adopted.json`, the adopted fits of the project with this
            fit for its subset.
    """

    opid: str | None = None
    repeats: int = Field(default=4, ge=1)
    algorithm: Literal["LSQ", "DE"] = "LSQ"
    seed: int | None = None
    studies: list[str] | None = None
    timeout: float | None = None
    optimizer: dict[str, Any] = Field(default_factory=dict)
    adopt: bool = False


class IdentifiabilitySpec(FitProblemSpec, SplitSpec):
    """The spec of the profile likelihoods of the parameters of a fit.

    The settings are the ones of `sbmlsim.fit.identifiability.ProfileSettings`
    with its defaults.

    Attributes:
        fit: the records of the fit, the input `fit`.
        pids: the parameters to profile; the plan sets the ones
            `profile_likelihood` profiles by default, every parameter which is
            no element of a network.
        alpha: the confidence level of the intervals.
        degrees_of_freedom: the degrees of freedom of the threshold.
        initial_step: the first step of a scan in decades.
        min_step: the smallest step in decades.
        max_step: the largest step in decades.
        max_points: the largest number of points of a scan in one direction.
        reoptimize: optimize the other parameters at every point.
        optimizer: arguments of the least squares optimizer of the other
            parameters, `ProfileSettings.optimizer_kwargs`.
        fisher: compute the Fisher information as well, `fisher.json`.
    """

    fit: Path | None = None
    pids: list[str] | None = None
    alpha: float = 0.95
    degrees_of_freedom: int = 1
    initial_step: float = 0.1
    min_step: float = 0.01
    max_step: float = 1.0
    max_points: int = 50
    reoptimize: bool = True
    optimizer: dict[str, Any] = Field(default_factory=dict)
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
        fit: the records, the input `fit`, e.g. the out of an
            identifiability analysis, whose profiles the report shows.
        mapping_figures: draw the figure of every fit mapping.
        group_by: the groupings of the report, the columns of one grouping
            per list; the groupings of the fit problem without.
    """

    fit: Path | None = None
    mapping_figures: bool = True
    group_by: list[list[str]] | None = None


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
def _definition(spec: FitProblemSpec) -> FitDefinition:
    """Get the fit problem a spec names.

    Raises:
        ConfigurationError: for a project which is not loaded or a subset it
            does not have.
    """
    from sbmlsim.project import load_project

    try:
        return load_project(spec.project).fit(spec.subset)
    except (ImportError, ValueError, KeyError) as err:
        message = err.args[0] if isinstance(err, KeyError) and err.args else err
        raise ConfigurationError(
            f"The fit problem '{spec.subset}' of the project '{spec.project}' "
            f"is not resolved: {message}"
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
    ctx: RunContext,
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
        ctx.out / FISHER_FILE, json.dumps(fisher.to_dict(), indent=2) + "\n"
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


def _plan_fit(spec: FitSpec) -> FitSpec:
    """Plan a fit: its id, its seed and its units.

    Raises:
        ConfigurationError: for an unknown problem or study, an id which is
            no name of a directory and invalid arguments of the optimizer.
    """
    opid = spec.subset if spec.opid is None else spec.opid
    _check_opid(opid)
    _check_fit_arguments(spec, _definition(spec))
    seed = (
        spec.seed
        if spec.seed is not None
        else int(np.random.default_rng().integers(2**31))
    )
    return spec.model_copy(
        update={
            "opid": opid,
            "seed": seed,
            "units": len(unit_blocks(spec.repeats)),
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

    logger.info("'%s': the repeats %s of %d run.", opid, indices, spec.repeats)
    runs = run_fit(
        _definition(spec),
        opid=opid,
        algorithm=ALGORITHMS[spec.algorithm],
        size=spec.repeats,
        n_cores=n_cores,
        seed=spec.seed,
        study_ids=spec.studies,
        timeout=spec.timeout,
        output_dir=ctx.out,
        indices=indices,
        **spec.optimizer,
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
    if spec.adopt:
        from sbmlsim.project import ADOPTED_FILE, load_project, write_adopted

        try:
            adopted = load_project(spec.project).adopted_fits()
        except ValueError as err:
            raise ConfigurationError(
                f"The adopted fits of '{spec.project}' are not read: {err}"
            ) from err
        write_adopted(ctx.out / ADOPTED_FILE, {**adopted, spec.subset: opid})
    df = result.df_fits
    return {
        "opid": opid,
        "repeats": spec.repeats,
        "converged": int(df.success.astype(bool).sum()),
        "best_cost": float(df.cost.iloc[0]),
    }


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
def _profile_settings(spec: IdentifiabilitySpec) -> ProfileSettings:
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
            max_points=spec.max_points,
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
    known = [p.pid for p in definition.parameters]
    unknown = [pid for pid in pids if pid not in known]
    if unknown or not pids or len(set(pids)) != len(pids):
        raise ConfigurationError(
            f"The parameters to profile {pids} are not distinct parameters of "
            f"the problem '{spec.subset}', which has {known}."
        )
    return spec.model_copy(update={"pids": list(pids), "units": len(pids)})


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
        _write_fisher(ctx, problem, settings, parameter_set, alpha=spec.alpha)
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
    summary = _write_fisher(ctx, problem, settings, parameter_set, step=spec.step)
    _copy_records(directory, ctx)
    return summary


def _run_report(spec: ReportSpec, ctx: RunContext) -> dict[str, Any]:
    """Write the report of the records of a fit into `report/`."""
    from sbmlsim.fit.cli import load_fit_records, report_from_records
    from sbmlsim.fit.evaluation import Grouping

    directory = _records_dir(spec.fit)
    definition = _definition(spec)
    records = load_fit_records(directory)
    report = report_from_records(
        definition,
        directory,
        opid=_fit_opid(records, spec.subset),
        records=records,
        mapping_figures=spec.mapping_figures,
        groupings=(
            None
            if spec.group_by is None
            else [Grouping.by(*columns) for columns in spec.group_by]
        ),
    )
    path = report.create(output_dir=ctx.out, name=REPORT_NAME)
    return {"report": path.relative_to(ctx.out).as_posix()}


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
)

#: the Fisher information of a fit, one jacobian in one process
FISHER: Kind[FisherSpec] = Kind(
    name="fisher",
    spec=FisherSpec,
    run=_run_fisher,
    inputs=("fit",),
    uses_cores=False,
)

#: the report of the records of a fit
REPORT: Kind[ReportSpec] = Kind(
    name="report", spec=ReportSpec, run=_run_report, inputs=("fit",), uses_cores=False
)

#: the PEtab v2 problem of a fit problem
PETAB: Kind[PetabSpec] = Kind(
    name="petab", spec=PetabSpec, run=_run_petab, inputs=("fit",), uses_cores=False
)

for _kind in (FIT, IDENTIFIABILITY, FISHER, REPORT, PETAB):
    register(_kind)
