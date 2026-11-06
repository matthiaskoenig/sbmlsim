"""The kinds of a project's scans, sensitivity and uncertainty analyses and experiments.

Every spec names a project, `project` (`module:attribute` of a
`sbmlsim.project.Project`), and an analysis of it, `analysis` (a key of
`Project.scans`, `Project.sensitivities` or `Project.uncertainties`); the model of an analysis which
is a relative `Path` is relative to `Project.root`.

- `scan` (`ScanSpec`) runs a scan of the project, one unit per block of
  consecutive points, `unit_blocks(scan.size)`: a unit writes the result of
  its points, `blocks/points-<start>-<stop>.nc`, and the merge combines them
  into `result.nc` (`ScanResult.combine`), the result of a run of the whole
  scan, bitwise; a whole run runs the consecutive blocks in spans of a block
  per worker, cut into the files of the blocks. The plan records the hash of
  the identity of the scan, `scan_hash`, which every unit and the merge
  compare with the scan they build, and refuses a scan whose records of two
  builds differ, a design which draws other points every time, one without
  a seed.
- `sensitivity` (`SensitivitySpec`) is a scan whose analysis has
  sensitivity indices: the merge also computes the indices of the result,
  `sensitivity.nc` (`SensitivityResult`). A spec which names a sensitivity
  analysis of the project over the groups of its simulations
  (`Project.sensitivities`, `sbmlsim.sensitivity.runner`) runs its analyses
  into the out, one unit per analysis (and subfolder), each written into its
  folder atomically, `<analysis>` or `<subfolder>/<analysis>`; the merge
  writes the table of the parameters, `parameters.tsv` and `.tex`. The plan
  checks the run (`prepare_run`) and sets the analyses, or the subfolders,
  which run.
- `uncertainty` (`UncertaintySpec`) propagates the uncertainty of an
  analysis (`sbmlsim.uncertainty.Propagation`) with the seed of the plan:
  a given number of draws `n` in one unit per block, `unit_blocks(n)`, the
  checkpoints of the propagation in `checkpoint/`, the consecutive blocks of
  a whole run in spans of a block per worker; the adaptive number of
  draws is one unit, whose rounds are its checkpoints. The merge writes the
  result into `uncertainty/` (`UncertaintyResult.to_netcdf`).
- `simulate` (`SimulateSpec`) runs simulation experiments of the project with
  their report, `index.html` and a directory per experiment; the phases of
  the experiments are the events `experiment` of the run. With `uncertainty`
  (`DrawsSpec`) the experiments run over draws of the fitted parameters of
  the model of the project, with the seed of the plan, and the curves of their
  simulations become bands, see `sbmlsim.experiment.draws`.

Every result file is written atomically, so a unit computed again leaves
complete files. The modules of the analyses are imported when a kind runs, not
when this module is imported.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from sbmlsim import events
from sbmlsim.atomic import atomic_directory, atomic_path, write_text_atomic
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
    from sbmlsim.experiment import SimulationExperiment
    from sbmlsim.experiment.draws import UncertaintySettings
    from sbmlsim.experiment.progress import Event
    from sbmlsim.model.model_sbml import SBMLModel
    from sbmlsim.project import Project, ScanAnalysis, UncertaintyAnalysis
    from sbmlsim.result import ScanResult
    from sbmlsim.run.context import RunContext
    from sbmlsim.sensitivity.runner import PreparedRun, SensitivityUnit
    from sbmlsim.simulation import Scan
    from sbmlsim.uncertainty import Propagation, UncertaintyResult

__all__ = [
    "SCAN",
    "SENSITIVITY",
    "SIMULATE",
    "UNCERTAINTY",
    "DrawsSpec",
    "ScanSpec",
    "SensitivitySpec",
    "SimulateSpec",
    "UncertaintySpec",
]

logger = logging.getLogger(__name__)

#: the directory of the results of the units of a scan
BLOCKS_DIR = "blocks"
#: the result of a scan
RESULT_FILE = "result.nc"
#: the sensitivity indices of a scan
SENSITIVITY_FILE = "sensitivity.nc"
#: the directory of the checkpoints of a propagation
CHECKPOINT_DIR = "checkpoint"
#: the file of the adaptive propagation which finished its rounds
ADAPTIVE_FILE = "adaptive.json"
#: the directory of the result of a propagation
UNCERTAINTY_DIR = "uncertainty"


# ----------------------------------------------------------------------------
# the specs
# ----------------------------------------------------------------------------
class ScanSpec(SplitSpec):
    """The spec of a scan of a project.

    Attributes:
        project: the project, `module:attribute`.
        analysis: the scan, a key of the scans of the project.
        scan_hash: the hash of the scan, set by the plan, see `scan_hash`.
    """

    project: str
    analysis: str
    scan_hash: str | None = None


class SensitivitySpec(ScanSpec):
    """The spec of a sensitivity analysis of a project.

    The analysis is a scan with `indices` (`sbmlsim.project.ScanAnalysis`) or
    a sensitivity analysis over the groups of the simulations of the project
    (`Project.sensitivities`).

    Attributes:
        analyses: the names of the analyses of a runner of
            `Project.sensitivities`, its default analyses without; the plan
            sets them for a single runner.
        subfolders: the subfolders of the runners by subfolder which run,
            every one without; the plan sets them.
    """

    analyses: list[str] | None = None
    subfolders: list[str] | None = None


class UncertaintySpec(SplitSpec):
    """The spec of an uncertainty analysis of a project.

    Attributes:
        project: the project, `module:attribute`.
        analysis: the analysis, a key of the uncertainties of the project.
        seed: the seed of the draws; the plan sets the seed of the analysis
            without, else draws one.
    """

    project: str
    analysis: str
    seed: int | None = None


class DrawsSpec(BaseModel):
    """The draws of the fitted parameters of a run of experiments.

    The fields of `sbmlsim.experiment.draws.UncertaintySettings`.

    Attributes:
        draws: the number of draws, at least two.
        seed: the seed of the draws, drawn by the plan without.
        quantiles: the lower and the upper quantile of the bands.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    draws: int = 100
    seed: int | None = None
    quantiles: tuple[float, float] = (0.05, 0.95)

    @model_validator(mode="after")
    def _check(self) -> Self:
        """Check the draws and the quantiles as the settings do.

        `sbmlsim.distrib.check_band`, so a spec is checked without importing
        the experiments.
        """
        from sbmlsim.distrib import check_band

        check_band(self.draws, self.quantiles)
        return self

    def settings(self, seed: int | None = None) -> UncertaintySettings:
        """Get the settings of the run.

        Args:
            seed: the seed, the one of the spec without.

        Raises:
            ValueError: for invalid settings or a spec without a seed.
        """
        from sbmlsim.experiment.draws import UncertaintySettings

        seed = self.seed if seed is None else seed
        if seed is None:
            raise ValueError("The draws have no seed: plan the spec first.")
        return UncertaintySettings(
            draws=self.draws, seed=seed, quantiles=self.quantiles
        )


class SimulateSpec(Spec):
    """The spec of a run of simulation experiments of a project.

    The experiments run with the settings of the simulator and of the figures
    of the project, see `sbmlsim.project.Project.run_experiments`.

    Attributes:
        project: the project, `module:attribute`.
        experiments: the names of the groups and of the classes of the
            experiments to run; the plan sets the names of the classes, every
            experiment of the project without.
        uncertainty: run the experiments over draws of the fitted parameters
            of the default variant of the model, their curves as bands; once
            without.
        figure_formats: the formats of the figures, e.g. `["svg", "html"]`,
            the ones of the project (`FigureOutputs.formats`) without.
    """

    project: str
    experiments: list[str] | None = None
    uncertainty: DrawsSpec | None = None
    figure_formats: list[str] | None = None


# ----------------------------------------------------------------------------
# what the kinds share
# ----------------------------------------------------------------------------
def _project(ref: str) -> Project:
    """Load a project.

    Raises:
        ConfigurationError: for a project which is not loaded.
    """
    from sbmlsim.project import load_project

    try:
        return load_project(ref)
    except (ImportError, ValueError) as err:
        raise ConfigurationError(f"The project '{ref}' is not loaded: {err}") from err


def _model(project: Project, model: Path | str) -> Path | str:
    """Get the model of an analysis, a relative path relative to the project.

    A string is a source as it is, e.g. inline SBML or a path of a resource.
    """
    if isinstance(model, Path) and not model.is_absolute():
        return project.root / model
    return model


def _loaded_model(project: Project, model: Path | str) -> SBMLModel:
    """Load the model of an analysis in the engine and at the level of the project.

    The model is compiled at `SimulatorSettings.jit_optimization` of the
    project; the simulator uses a model of its engine as it is, and the
    workers of its pool load it at the level of this model (`ModelSpec`).
    """
    from sbmlsim.model import model_sbml

    settings = project.simulator
    return model_sbml.load_model(
        _model(project, model),
        settings.engine,
        jit_optimization=settings.jit_optimization,
    )


def _times(time: Any) -> list[float] | None:
    """Get the grid of times of an analysis as a list, `None` without one."""
    return None if time is None else [float(t) for t in time]


def _workers(ctx: RunContext) -> int:
    """Get the processes of a run, every CPU of the quota if not resolved."""
    return ctx.workers or available_cpus()


# ----------------------------------------------------------------------------
# scan and sensitivity
# ----------------------------------------------------------------------------
def _check_indices(spec: ScanSpec, analysis: ScanAnalysis) -> None:
    """Check that the scan of a sensitivity analysis has valid indices.

    Raises:
        ConfigurationError: for a scan without indices or with options which
            its function of the indices does not take.
    """
    import sbmlsim.sensitivity

    if analysis.indices is None:
        raise ConfigurationError(
            f"The scan '{spec.analysis}' has no sensitivity indices: set "
            "`indices` of its ScanAnalysis, or run it as a scan."
        )
    function = getattr(sbmlsim.sensitivity, analysis.indices)
    try:
        inspect.signature(function).bind(None, **analysis.index_options)
    except TypeError as err:
        raise ConfigurationError(
            f"The options {analysis.index_options} of the {analysis.indices} "
            f"indices of the scan '{spec.analysis}' are invalid: {err}"
        ) from err


def _scan_analysis(spec: ScanSpec) -> tuple[Project, ScanAnalysis]:
    """Get the project and the scan a spec names, with indices for a sensitivity.

    Raises:
        ConfigurationError: for an unknown project or scan, and a sensitivity
            analysis of a scan without valid indices.
    """
    project = _project(spec.project)
    try:
        analysis = project.scan(spec.analysis)
    except KeyError as err:
        message = err.args[0] if err.args else str(err)
        if isinstance(spec, SensitivitySpec):
            message += f" Its sensitivity analyses over groups are {sorted(project.sensitivities)}."
        raise ConfigurationError(message) from err
    if isinstance(spec, SensitivitySpec):
        _check_indices(spec, analysis)
    return project, analysis


def _build_scan(spec: ScanSpec, analysis: ScanAnalysis) -> Scan:
    """Build the scan of an analysis.

    Raises:
        ConfigurationError: if building it fails.
    """
    from sbmlsim.simulation import Scan

    try:
        return Scan.of(analysis.scan())
    except (ValueError, TypeError, KeyError) as err:
        raise ConfigurationError(
            f"The scan '{spec.analysis}' of '{spec.project}' is not built: {err}"
        ) from err


def _canonical(value: Any) -> Any:
    """Convert a numpy value of a record to JSON, the `default` of the hash.

    Raises:
        TypeError: for any other value, which has no canonical JSON.
    """
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"A '{type(value).__name__}' has no canonical JSON.")


def scan_identity(scan: Scan) -> dict[str, Any]:
    """Get the identity of a scan: its record without what a design computes.

    A dimension of a design (`Dimension.design`, a design of
    `sbmlsim.simulation.sampling`) is identified by its definition: its id,
    kind, labels, time `at`, simulations and models, the targets of its values
    and the names of its coordinates, and the record of its design without the
    references (`Design.references`: the method, the distributions and the
    options, its seed and number of points among them). The numbers it draws,
    its values and coordinates (e.g. the covariates of a population), and the
    references, which the design reads from the model, are left out: the
    plan and the units of a split run may run on other platforms, whose
    numerics may differ in the last bits. Every other dimension, of values,
    simulations or models given by the user, and the simulation are
    identified by their whole record.

    Args:
        scan: the scan.

    Returns:
        The identity, JSON types.
    """
    record = scan.to_dict()
    for dimension, entry in zip(scan.dimensions, record["dimensions"], strict=True):
        if dimension.design is None:
            continue
        entry["values"] = sorted(entry["values"])
        entry["coordinates"] = sorted(entry["coordinates"])
        entry["design"] = {
            k: v for k, v in entry["design"].items() if k != "references"
        }
    return record


def _digest(record: Any) -> str:
    """Get the sha256 of the canonical JSON of a record."""
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), default=_canonical)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scan_hash(scan: Scan) -> str:
    """Get the hash of a scan: the sha256 of the canonical JSON of its identity.

    The identity (`scan_identity`) is the record of the scan without the
    numbers its designs draw and the references they read, so a unit on
    another platform than the plan builds a scan of the same hash, while a
    scan of another simulation, other given values or another design, seed or
    number of points has another hash. That two builds draw the same numbers,
    e.g. a design with a seed, the plan checks on the whole records.

    Args:
        scan: the scan.

    Returns:
        The hexadecimal digest.
    """
    return _digest(scan_identity(scan))


def _plan_scan[S: ScanSpec](spec: S) -> S:
    """Plan a scan: its hash and its units.

    The scan is built twice, so a scan which differs between builds is found.

    Raises:
        ConfigurationError: for an unknown project or scan, a sensitivity
            analysis without valid indices, a scan which is not built and one
            which differs between two builds, e.g. a design without a seed.
    """
    _, analysis = _scan_analysis(spec)
    scan = _build_scan(spec, analysis)
    digest = scan_hash(scan)
    # the whole records: a design without a seed draws other numbers
    if _digest(_build_scan(spec, analysis).to_dict()) != _digest(scan.to_dict()):
        raise ConfigurationError(
            f"The scan '{spec.analysis}' differs between two builds: a design "
            "without a seed draws other points each time, give it a seed."
        )
    return spec.model_copy(
        update={"scan_hash": digest, "units": len(unit_blocks(scan.size))}
    )


@dataclass(frozen=True)
class _PlannedScan:
    """A planned scan as it resolves.

    Attributes:
        spec: the planned spec.
        project: the project.
        analysis: the scan of the project.
        scan: the scan it builds, whose hash is the one of the plan.
        blocks: the points of every unit.
    """

    spec: ScanSpec
    project: Project
    analysis: ScanAnalysis
    scan: Scan
    blocks: list[range]


def _planned_scan(spec: ScanSpec) -> _PlannedScan:
    """Resolve a planned scan: its project, its analysis, its scan and its blocks.

    Raises:
        ConfigurationError: for a spec which is not planned.
        PlanMismatch: for a scan of another hash or units which are not its
            blocks.
    """
    if spec.scan_hash is None or spec.units is None:
        raise ConfigurationError("The spec of the scan is not planned: plan it first.")
    project, analysis = _scan_analysis(spec)
    scan = _build_scan(spec, analysis)
    if scan_hash(scan) != spec.scan_hash:
        raise PlanMismatch(
            f"The scan '{spec.analysis}' is not the scan of the plan, its hash "
            "differs; plan the spec again."
        )
    blocks = unit_blocks(scan.size)
    if spec.units != len(blocks):
        raise PlanMismatch(
            f"The scan has {spec.units} units, its {scan.size} points are "
            f"{len(blocks)} units; plan the spec again."
        )
    return _PlannedScan(spec, project, analysis, scan, blocks)


def _block_file(ctx: RunContext, block: range) -> Path:
    """Get the file of the result of a block of points."""
    return ctx.out / BLOCKS_DIR / f"points-{block.start}-{block.stop}.nc"


def _spans(indices: list[int], limit: int) -> list[list[int]]:
    """Group indices of blocks into runs of consecutive indices, in their order.

    Args:
        indices: the indices, in any order and with repeats.
        limit: the most indices of a run, at least 1.
    """
    spans: list[list[int]] = []
    for index in sorted(set(indices)):
        if spans and spans[-1][-1] == index - 1 and len(spans[-1]) < max(1, limit):
            spans[-1].append(index)
        else:
            spans.append([index])
    return spans


def _run_blocks(
    planned: _PlannedScan, ctx: RunContext, indices: list[int], workers: int
) -> None:
    """Run blocks of points of a scan, each written to its file.

    The consecutive blocks run in spans of as many blocks as there are
    workers, a span of points on the workers (`Simulator.run` with `points`),
    so a whole run is parallel whatever the size of its blocks and keeps the
    blocks of a span once it ran; the result of a span is cut into the
    results of its blocks (`ScanResult.part`), each bitwise the file a unit
    writes; a
    block whose messages of failed points the span does not keep
    (`LostErrors`, more than `MAX_ERRORS` failed points before them) runs
    alone. The events `scan.chunk` count the points of a span, the event
    `scan.block` follows the file of every block.

    Args:
        planned: the planned scan.
        ctx: the context.
        indices: the indices of the blocks (units) to run.
        workers: the processes of the simulator.
    """
    from sbmlsim.result.scan import LostErrors

    if not indices:
        return
    analysis, scan = planned.analysis, planned.scan
    simulator = planned.project.simulator.simulator(workers)
    model = simulator.load(_loaded_model(planned.project, analysis.model))
    observables = (
        list(analysis.observables()) if analysis.observables is not None else None
    )

    def simulate(points: range) -> ScanResult:
        """Run points of the scan."""
        return simulator.run(
            model,
            scan,
            observables,
            time=_times(analysis.time),
            keep=list(analysis.keep) if analysis.keep is not None else None,
            on_error=analysis.on_error,
            progress=False,
            points=points,
        )

    (ctx.out / BLOCKS_DIR).mkdir(parents=True, exist_ok=True)
    for span in _spans(indices, limit=workers):
        blocks = [planned.blocks[index] for index in span]
        result = simulate(range(blocks[0].start, blocks[-1].stop))
        for block in blocks:
            part = result
            if len(blocks) > 1:
                try:
                    part = result.part(block)
                except LostErrors as err:
                    logger.info("%s", err)
                    part = simulate(block)
            with atomic_path(_block_file(ctx, block)) as temporary:
                part.to_netcdf(temporary)
            events.emit(
                "scan.block",
                analysis=planned.spec.analysis,
                start=block.start,
                stop=block.stop,
                points=scan.size,
            )


def _compute_scan_unit(spec: ScanSpec, index: int, ctx: RunContext) -> None:
    """Run the block of points of a unit serially, unless its file exists."""
    planned = _planned_scan(spec)
    if not _block_file(ctx, planned.blocks[index]).exists():
        _run_blocks(planned, ctx, [index], workers=1)


def _merged(
    planned: _PlannedScan, ctx: RunContext
) -> tuple[ScanResult, dict[str, Any]]:
    """Combine the blocks of a scan into `result.nc`.

    Returns:
        The result and the summary: the points and the points which failed.

    Raises:
        MissingUnits: naming the units whose block has no file.
    """
    from sbmlsim.result import ScanResult
    from sbmlsim.result.scan import STATUS

    blocks = planned.blocks
    missing = [
        k for k, block in enumerate(blocks) if not _block_file(ctx, block).exists()
    ]
    if missing:
        raise MissingUnits(missing, len(blocks))
    result = ScanResult.combine(
        [ScanResult.from_netcdf(_block_file(ctx, block)) for block in blocks]
    )
    with atomic_path(ctx.out / RESULT_FILE) as temporary:
        result.to_netcdf(temporary)
    failed = (
        int(np.count_nonzero(result.ds[STATUS].values)) if STATUS in result.ds else 0
    )
    return result, {"points": planned.scan.size, "failed": failed}


def _with_indices(
    planned: _PlannedScan, result: ScanResult, ctx: RunContext
) -> dict[str, Any]:
    """Compute the sensitivity indices of a scan into `sensitivity.nc`.

    Returns:
        The part of the summary: the function of the indices.
    """
    import sbmlsim.sensitivity

    analysis = planned.analysis
    indices = str(analysis.indices)
    sensitivity = getattr(sbmlsim.sensitivity, indices)(
        result, **analysis.index_options
    )
    with atomic_path(ctx.out / SENSITIVITY_FILE) as temporary:
        sensitivity.to_netcdf(temporary)
    return {"indices": indices}


def _merge_planned(planned: _PlannedScan, ctx: RunContext) -> dict[str, Any]:
    """Combine the blocks of a scan, with the indices of a sensitivity analysis."""
    result, summary = _merged(planned, ctx)
    if isinstance(planned.spec, SensitivitySpec):
        summary |= _with_indices(planned, result, ctx)
    return summary


def _merge_scan(spec: ScanSpec, ctx: RunContext) -> dict[str, Any]:
    """Merge the blocks of a scan, see `_merge_planned`, and summarize them."""
    return _merge_planned(_planned_scan(spec), ctx)


def _run_scan(spec: ScanSpec, ctx: RunContext) -> dict[str, Any]:
    """Run the blocks of a scan, with `resume` the ones without a file, and merge."""
    planned = _planned_scan(spec)
    missing = [
        k
        for k, block in enumerate(planned.blocks)
        if not (ctx.resume and _block_file(ctx, block).exists())
    ]
    _run_blocks(planned, ctx, missing, workers=_workers(ctx))
    return _merge_planned(planned, ctx)


# ----------------------------------------------------------------------------
# sensitivity analyses over the groups of the simulations
# ----------------------------------------------------------------------------
def _message(err: BaseException) -> str:
    """Get the message of an error with its notes."""
    return " ".join([str(err), *getattr(err, "__notes__", [])])


def _project_sensitivity(spec: SensitivitySpec) -> Any | None:
    """Get the runners of a project sensitivity a spec names, `None` for a scan.

    Raises:
        ConfigurationError: for an unknown project, a sensitivity which is not
            resolved, and analyses or subfolders of a scan.
    """
    project = _project(spec.project)
    if spec.analysis not in project.sensitivities:
        if spec.analyses is not None or spec.subfolders is not None:
            raise ConfigurationError(
                f"'{spec.analysis}' of '{spec.project}' is no sensitivity analysis "
                "of the project over groups: a scan has neither analyses nor "
                "subfolders."
            )
        return None
    try:
        return project.sensitivity(spec.analysis)
    except (ImportError, AttributeError, TypeError, ValueError) as err:
        raise ConfigurationError(
            f"The sensitivity analysis '{spec.analysis}' of '{spec.project}' is not "
            f"resolved: {err}"
        ) from err


def _prepared(spec: SensitivitySpec, target: Any) -> PreparedRun:
    """Check the run of the runners of a spec, see `prepare_run`.

    Raises:
        ConfigurationError: for a run which cannot run, naming why.
    """
    from sbmlsim.sensitivity.runner import prepare_run

    if spec.scan_hash is not None:
        raise ConfigurationError(
            f"The sensitivity analysis '{spec.analysis}' is no scan, it has no "
            "scan_hash."
        )
    try:
        return prepare_run(target, spec.analyses, spec.subfolders)
    except (TypeError, ValueError) as err:
        raise ConfigurationError(
            f"The sensitivity analysis '{spec.analysis}' of '{spec.project}' "
            f"cannot run: {_message(err)}"
        ) from err


def _plan_runner(spec: SensitivitySpec, target: Any) -> SensitivitySpec:
    """Plan a run of the runners of a project sensitivity: what runs and its units."""
    prepared = _prepared(spec, target)
    update: dict[str, Any] = {"units": len(prepared.units)}
    if None in prepared.runners:
        update["analyses"] = list(prepared.plans[None][0])
    else:
        update["subfolders"] = [str(key) for key in prepared.plans]
    return spec.model_copy(update=update)


def _planned_runner(spec: SensitivitySpec, target: Any) -> PreparedRun:
    """Resolve a planned run of the runners of a project sensitivity.

    Raises:
        ConfigurationError: for a spec which is not planned.
        PlanMismatch: for other units than the ones of the plan.
    """
    if spec.units is None:
        raise ConfigurationError(
            "The spec of the sensitivity analysis is not planned: plan it first."
        )
    prepared = _prepared(spec, target)
    if len(prepared.units) != spec.units:
        raise PlanMismatch(
            f"The sensitivity analysis has {spec.units} units, its analyses are "
            f"{len(prepared.units)}; plan the spec again."
        )
    return prepared


def _run_analysis(
    spec: SensitivitySpec,
    prepared: PreparedRun,
    unit: SensitivityUnit,
    ctx: RunContext,
    workers: int,
) -> None:
    """Run an analysis of a project sensitivity into its folder of the out, atomically."""
    folder = ctx.out / unit.folder
    folder.parent.mkdir(parents=True, exist_ok=True)
    with atomic_directory(folder) as temporary:
        prepared.run_unit(unit, temporary, prepared.simulator(workers))
    events.emit(
        "sensitivity.analysis",
        analysis=spec.analysis,
        folder=unit.folder.as_posix(),
    )


def _merge_runner(
    spec: SensitivitySpec, prepared: PreparedRun, ctx: RunContext
) -> dict[str, Any]:
    """Write the table of the parameters once every analysis has its folder.

    Raises:
        MissingUnits: naming the units whose analysis has no folder.
    """
    units = prepared.units
    missing = [
        k for k, unit in enumerate(units) if not (ctx.out / unit.folder).is_dir()
    ]
    if missing:
        raise MissingUnits(missing, len(units))
    table = prepared.write_table(ctx.out)
    summary: dict[str, Any] = {
        "analyses": [unit.folder.as_posix() for unit in units],
        "parameters": len(table),
    }
    return summary


def _plan_sensitivity(spec: SensitivitySpec) -> SensitivitySpec:
    """Plan a sensitivity analysis: a scan with indices or a project sensitivity."""
    target = _project_sensitivity(spec)
    if target is None:
        return _plan_scan(spec)
    return _plan_runner(spec, target)


def _compute_sensitivity_unit(
    spec: SensitivitySpec, index: int, ctx: RunContext
) -> None:
    """Compute a unit of a sensitivity analysis, unless its result exists."""
    target = _project_sensitivity(spec)
    if target is None:
        _compute_scan_unit(spec, index, ctx)
        return
    prepared = _planned_runner(spec, target)
    unit = prepared.units[index]
    if not (ctx.out / unit.folder).is_dir():
        _run_analysis(spec, prepared, unit, ctx, workers=1)


def _merge_sensitivity(spec: SensitivitySpec, ctx: RunContext) -> dict[str, Any]:
    """Merge the units of a sensitivity analysis."""
    target = _project_sensitivity(spec)
    if target is None:
        return _merge_scan(spec, ctx)
    return _merge_runner(spec, _planned_runner(spec, target), ctx)


def _run_sensitivity(spec: SensitivitySpec, ctx: RunContext) -> dict[str, Any]:
    """Run a sensitivity analysis whole, with `resume` the analyses without a folder."""
    target = _project_sensitivity(spec)
    if target is None:
        return _run_scan(spec, ctx)
    prepared = _planned_runner(spec, target)
    workers = _workers(ctx)
    for unit in prepared.units:
        if not (ctx.resume and (ctx.out / unit.folder).is_dir()):
            _run_analysis(spec, prepared, unit, ctx, workers)
    return _merge_runner(spec, prepared, ctx)


# ----------------------------------------------------------------------------
# uncertainty
# ----------------------------------------------------------------------------
def _uncertainty_analysis(spec: UncertaintySpec) -> tuple[Project, UncertaintyAnalysis]:
    """Get the project and the uncertainty analysis a spec names.

    Raises:
        ConfigurationError: for an unknown project or analysis.
    """
    project = _project(spec.project)
    try:
        return project, project.uncertainty(spec.analysis)
    except KeyError as err:
        raise ConfigurationError(err.args[0] if err.args else str(err)) from err


def _draw_blocks(analysis: UncertaintyAnalysis) -> list[range] | None:
    """Get the blocks of the draws of an analysis, `None` for the adaptive number."""
    return None if analysis.n is None else unit_blocks(max(analysis.n, 0))


def _propagation(
    spec: UncertaintySpec,
    project: Project,
    analysis: UncertaintyAnalysis,
    workers: int,
    checkpoint: Path | None,
) -> Propagation:
    """Create the propagation of an analysis with the seed of the spec.

    With a given number of draws a block of the propagation is a unit,
    `unit_blocks(n)`.

    Raises:
        ValueError: for an invalid analysis, see `Propagation`.
    """
    from sbmlsim.uncertainty import Propagation

    blocks = _draw_blocks(analysis)
    return Propagation(
        _loaded_model(project, analysis.model),
        analysis.simulation(),
        analysis.uq(),
        list(analysis.observables()) if analysis.observables is not None else (),
        outputs=list(analysis.outputs) if analysis.outputs is not None else None,
        n=analysis.n,
        rel_tol=analysis.rel_tol,
        n_start=analysis.n_start,
        n_max=analysis.n_max,
        design=analysis.design,
        seed=spec.seed,
        simulator=project.simulator.simulator(workers),
        keep=list(analysis.keep) if analysis.keep is not None else None,
        time=_times(analysis.time),
        on_error=analysis.on_error,
        checkpoint=checkpoint,
        block_size=len(blocks[0]) if blocks else None,
    )


def _plan_uncertainty(spec: UncertaintySpec) -> UncertaintySpec:
    """Plan an uncertainty analysis: its seed and its units.

    The seed is the one of the spec, else of the analysis, else drawn.

    Raises:
        ConfigurationError: for an unknown project or analysis and an analysis
            which is no valid propagation.
    """
    project, analysis = _uncertainty_analysis(spec)
    seed = spec.seed if spec.seed is not None else analysis.seed
    if seed is None:
        seed = int(np.random.default_rng().integers(2**31))
    planned = spec.model_copy(update={"seed": seed})
    try:
        _propagation(planned, project, analysis, workers=1, checkpoint=None)
    except (ValueError, TypeError, KeyError) as err:
        raise ConfigurationError(
            f"The uncertainty analysis '{spec.analysis}' is invalid: {err}"
        ) from err
    blocks = _draw_blocks(analysis)
    return planned.model_copy(update={"units": 1 if blocks is None else len(blocks)})


def _planned_uncertainty(
    spec: UncertaintySpec, ctx: RunContext, workers: int
) -> tuple[Propagation, list[range] | None]:
    """Get the propagation of a planned spec and its blocks, `None` if adaptive.

    Raises:
        ConfigurationError: for a spec which is not planned.
        PlanMismatch: for units which are not the blocks of the draws.
    """
    if spec.seed is None or spec.units is None:
        raise ConfigurationError(
            "The spec of the uncertainty analysis is not planned: plan it first."
        )
    project, analysis = _uncertainty_analysis(spec)
    blocks = _draw_blocks(analysis)
    n_units = 1 if blocks is None else len(blocks)
    if spec.units != n_units:
        raise PlanMismatch(
            f"The uncertainty analysis has {spec.units} units, its draws are "
            f"{n_units} units; plan the spec again."
        )
    propagation = _propagation(
        spec, project, analysis, workers, checkpoint=ctx.out / CHECKPOINT_DIR
    )
    return propagation, blocks


@contextmanager
def _checkpoints_of_the_plan() -> Generator[None]:
    """Raise a checkpoint of another propagation in the out as `PlanMismatch`.

    The out holds the checkpoints of a plan whose analysis resolves to
    another propagation now, e.g. after its model changed.

    Raises:
        PlanMismatch: for a checkpoint of another propagation.
    """
    from sbmlsim.uncertainty import CheckpointMismatch

    try:
        yield
    except CheckpointMismatch as err:
        raise PlanMismatch(
            f"{err} The output holds the checkpoints of a propagation which the "
            "analysis does not resolve to anymore; run it into another output."
        ) from err


def _adaptive_file(ctx: RunContext) -> Path:
    """Get the file of the adaptive propagation which finished its rounds."""
    return ctx.out / CHECKPOINT_DIR / ADAPTIVE_FILE


def _compute(
    propagation: Propagation,
    blocks: list[range] | None,
    ctx: RunContext,
    units: list[int],
) -> None:
    """Run units of a propagation, which checkpoints them.

    A unit of a given number of draws is a block of the propagation, and the
    consecutive blocks of a whole run run in spans of a block per worker
    (`Propagation.run_blocks`); the unit of the adaptive number of draws runs
    its rounds and writes `ADAPTIVE_FILE` once they finished.

    Raises:
        PlanMismatch: for a checkpoint of another propagation in the out.
    """
    with _checkpoints_of_the_plan():
        if blocks is not None:
            propagation.run_blocks([blocks[index] for index in units])
            return
        if units:
            result = propagation.run()
            write_text_atomic(
                _adaptive_file(ctx), json.dumps({"n": result.record["n"]}) + "\n"
            )


def _compute_uncertainty_unit(
    spec: UncertaintySpec, index: int, ctx: RunContext
) -> None:
    """Run the draws of a unit serially, unless its checkpoint exists."""
    propagation, blocks = _planned_uncertainty(spec, ctx, workers=1)
    _compute(propagation, blocks, ctx, [index])


def _write_uncertainty(
    propagation: Propagation, blocks: list[range] | None, ctx: RunContext
) -> dict[str, Any]:
    """Merge the checkpoints of a propagation into `uncertainty/` and summarize it.

    Raises:
        MissingUnits: naming the units without their checkpoint.
        PlanMismatch: for a checkpoint of another propagation in the out.
    """
    if blocks is not None:
        missing = [
            k for k, block in enumerate(blocks) if not propagation.checkpointed(block)
        ]
        if missing:
            raise MissingUnits(missing, len(blocks))
    elif not _adaptive_file(ctx).exists():
        raise MissingUnits([0], 1)
    with _checkpoints_of_the_plan():
        result = propagation.run()
    _replace_directory(result, ctx.out / UNCERTAINTY_DIR)
    history = result.record.get("history") or []
    return {
        "n": int(result.record["n"]),
        "converged": bool(history[-1]["converged"]) if history else True,
    }


def _replace_directory(result: UncertaintyResult, target: Path) -> None:
    """Write the result of a propagation into a directory which replaces the target.

    The result is written into a temporary directory next to the target,
    which replaces it once it is complete, so the target is a complete
    result, if any, see `sbmlsim.atomic.atomic_directory`.
    """
    with atomic_directory(target) as temporary:
        result.to_netcdf(temporary)


def _merge_uncertainty(spec: UncertaintySpec, ctx: RunContext) -> dict[str, Any]:
    """Merge the checkpoints of the units into the result of the propagation."""
    propagation, blocks = _planned_uncertainty(spec, ctx, workers=_workers(ctx))
    return _write_uncertainty(propagation, blocks, ctx)


def _run_uncertainty(spec: UncertaintySpec, ctx: RunContext) -> dict[str, Any]:
    """Run every unit of a propagation and merge them.

    A resume reads the checkpoints which exist instead of simulating them.
    """
    propagation, blocks = _planned_uncertainty(spec, ctx, workers=_workers(ctx))
    _compute(propagation, blocks, ctx, list(range(spec.units or 0)))
    return _write_uncertainty(propagation, blocks, ctx)


# ----------------------------------------------------------------------------
# simulate
# ----------------------------------------------------------------------------
def _experiments(
    spec: SimulateSpec,
) -> tuple[Project, list[type[SimulationExperiment]]]:
    """Get the project and the experiments a spec names, in its order.

    A name is a group of experiments or an experiment class, see
    `Project.experiment_classes`; every experiment runs once.

    Raises:
        ConfigurationError: for a project without experiments, no names and
            a name which is neither a group nor an experiment.
    """
    project = _project(spec.project)
    if project.experiments is None:
        raise ConfigurationError(
            f"The project '{spec.project}' has no simulation experiments."
        )
    if spec.experiments is not None and not spec.experiments:
        raise ConfigurationError(
            f"The run of experiments of '{spec.project}' names no experiments."
        )
    try:
        classes = project.experiment_classes(spec.experiments)
    except KeyError as err:
        raise ConfigurationError(str(err.args[0])) from err
    return project, classes


def _plan_simulate(spec: SimulateSpec) -> SimulateSpec:
    """Plan a run of experiments: the names of the experiments, the seed of the draws.

    The figure formats are written in lower case.

    Raises:
        ConfigurationError: for unknown experiments, empty figure formats
            and, with `uncertainty`, a
            default variant of the model without a parameter with a
            distribution.
    """
    project, experiments = _experiments(spec)
    update: dict[str, Any] = {"experiments": [cls.__name__ for cls in experiments]}
    if spec.figure_formats is not None:
        from sbmlsim.project import normalize_formats

        formats = list(
            normalize_formats([f.strip() for f in spec.figure_formats if f.strip()])
        )
        if not formats:
            raise ConfigurationError(
                "The figure formats of a run of experiments are not empty, "
                "the formats of the project without."
            )
        update["figure_formats"] = formats
    draws = spec.uncertainty
    if draws is not None:
        from sbmlsim.simulation.sampling import parameter_distributions

        try:
            path = project.model_path
            if not path.exists():
                raise ValueError(f"The model '{path}' does not exist.")
            if not parameter_distributions(path):
                raise ValueError(
                    f"No parameter of '{path}' has a distribution: the uncertainty "
                    "of the fitted parameters comes from the adopted fits of the "
                    "model."
                )
        except ValueError as err:
            raise ConfigurationError(
                f"The experiments of '{spec.project}' cannot run over draws: {err}"
            ) from err
        if draws.seed is None:
            seed = int(np.random.default_rng().integers(2**31))
            update["uncertainty"] = draws.model_copy(update={"seed": seed})
    return spec.model_copy(update=update)


def _experiment_event(event: Event) -> None:
    """Record the event of the phase of an experiment, see `sbmlsim.events`."""
    events.emit(
        "experiment",
        sid=event.sid,
        phase=str(event.phase),
        activity=event.activity,
        completed=event.completed,
        total=event.total,
    )


def _run_simulate(spec: SimulateSpec, ctx: RunContext) -> dict[str, Any]:
    """Run the experiments and write their report into the out.

    The experiments run in a pool of the workers of the run, at most one per
    experiment, each serially in its worker, which builds a serial simulator
    of its own. The experiments which run in this process, a single one or
    the ones `run_experiments` runs serially (see
    `ExperimentRunner.run_experiments`), run their scans on the workers of
    the run with the simulator of the run.

    With `uncertainty` the experiments run over the draws of the fitted
    parameters with the seed of the plan.

    Raises:
        ConfigurationError: for draws without the seed of the plan.
        ExperimentRunError: if an experiment or a figure failed, after every
            experiment ran and the report was written.
    """
    from sbmlsim.experiment.progress import reporting
    from sbmlsim.project import Project

    project, experiments = _experiments(spec)
    settings: UncertaintySettings | None = None
    if spec.uncertainty is not None:
        if spec.uncertainty.seed is None:
            raise ConfigurationError(
                "The draws of the run of experiments have no seed: plan it first."
            )
        settings = spec.uncertainty.settings()
    with reporting(_experiment_event, shown=False):
        # the base implementation, which is the contract of the run layer: a
        # subclass of `Project` may override `run_experiments` with another
        # signature and result, e.g. a model package of its own
        results = Project.run_experiments(
            project,
            experiments,
            ctx.out,
            n_workers=_workers(ctx),
            progress=False,
            uncertainty=settings,
            figure_formats=spec.figure_formats,
        )
    summary: dict[str, Any] = {
        "experiments": len(results),
        "failed": sum(1 for result in results if result.failed),
    }
    if settings is not None:
        summary["draws"] = settings.draws
        summary["seed"] = settings.seed
    return summary


#: a scan of a project, split into blocks of points
SCAN: Kind[ScanSpec] = Kind(
    name="scan",
    spec=ScanSpec,
    run=_run_scan,
    plan=_plan_scan,
    compute_unit=_compute_scan_unit,
    merge=_merge_scan,
    cost_setting="analysis",
)

#: the sensitivity indices of a scan of a project, split as the scan, or a
#: sensitivity analysis of the project over groups, split into its analyses
SENSITIVITY: Kind[SensitivitySpec] = Kind(
    name="sensitivity",
    spec=SensitivitySpec,
    run=_run_sensitivity,
    plan=_plan_sensitivity,
    compute_unit=_compute_sensitivity_unit,
    merge=_merge_sensitivity,
    cost_setting="analysis",
)

#: an uncertainty analysis of a project, split into blocks of draws
UNCERTAINTY: Kind[UncertaintySpec] = Kind(
    name="uncertainty",
    spec=UncertaintySpec,
    run=_run_uncertainty,
    plan=_plan_uncertainty,
    compute_unit=_compute_uncertainty_unit,
    merge=_merge_uncertainty,
    cost_setting="analysis",
)

#: the simulation experiments of a project and their report
SIMULATE: Kind[SimulateSpec] = Kind(
    name="simulate", spec=SimulateSpec, run=_run_simulate, plan=_plan_simulate
)

for _kind in (SCAN, SENSITIVITY, UNCERTAINTY, SIMULATE):
    register(_kind)
