"""The kinds of the run layer as task types of simrun.

simrun runs the tasks of a workflow on a cluster; this module registers the
ten kinds of `python -m sbmlsim.run` (`sbmlsim.run`) as its task types
`sbmlsim.<kind>`, entry points of the group `simrun.task_types` in the
`pyproject.toml` of sbmlsim, see the plugins of simrun. The settings of a task
are the spec of its kind, and `plan` is the plan of the kind, which simrun's
client calls before it submits a task.

A splittable kind (`fit`, `identifiability`, `scan`, `sensitivity`,
`uncertainty`) is split: simrun's workers compute its units with
`sbmlsim.run.compute_unit` and its merge merges them with `sbmlsim.run.merge`.
Every other kind (`simulate`, `fisher`, `report`, `petab`, `adopt`) runs as the
command
(`simrun-exec` runs it, `{settings}`, `{out}` and `{input}` filled by simrun)

    python -m sbmlsim.run <kind> {settings} --out {out}/out
        --summary {out}/summary.json --resume --input={input}

in which `--resume` lets a retry after an eviction continue the run it finds
and starts a run in an empty output.

A task writes the output of its run into its output port `out`, the directory
`out/out` of the task (`OUTPUT`), which a task downstream reads as the input
`<task>.out`, e.g. the fit of an identifiability analysis. The run, the command
of an unsplit task or the merge of a split one, also writes its summary to
`out/summary.json`, the summary of the task which `simrun compare` reads.

The resource hints are the maximum resident memory of a run on one worker
(`/usr/bin/time -v`, python 3.14 on linux, the HCTZ project
`examples.hctz_fitting.project`, `--workers 1`) times 1.5, rounded up to
0.5 GiB: the fit of the subset `PK` with 4 repeats 0.36 GiB, the scan `dose`
0.37 GiB, the sensitivity `sobol` 0.38 GiB, the uncertainty `cl` 0.38 GiB, the
simulation of the three experiments 0.38 GiB and the report of the fit with
its figures 0.48 GiB, so 1 GiB per worker for every kind. The base memory is
the process of simrun's worker besides the processes of its units, 0.10 GiB
measured with this module imported, 0.5 GiB.

This module imports simrun; `import sbmlsim` and `sbmlsim.run` do not import
it, only simrun's loading of the entry points and the tests do.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from simrun.plugins import ResourceHint, TaskPaths, TaskType

from sbmlsim import run
from sbmlsim.run.context import SUMMARY_FILE
from sbmlsim.run.fit_kinds import (
    AdoptSpec,
    FisherSpec,
    FitSpec,
    IdentifiabilitySpec,
    PetabSpec,
    ReportSpec,
)
from sbmlsim.run.kinds import Spec, get_kind
from sbmlsim.run.scan_kinds import (
    ScanSpec,
    SensitivitySpec,
    SimulateSpec,
    UncertaintySpec,
)

__all__ = [
    "ADOPT",
    "FISHER",
    "FIT",
    "IDENTIFIABILITY",
    "OUTPUT",
    "PETAB",
    "REPORT",
    "SCAN",
    "SENSITIVITY",
    "SIMULATE",
    "TASK_TYPES",
    "UNCERTAINTY",
]

#: the module of the functions the task types reference
MODULE = "sbmlsim.simrun"

#: the output port of a task, the directory of the output of its run in `out`
OUTPUT = "out"

#: the resources of a task: 1.5 times the measured memory of one worker
HINT = ResourceHint(memory_base_gb=0.5, memory_per_worker_gb=1.0)


def _run_out(paths: TaskPaths) -> Path:
    """Get the output directory of the run of a task, its output port."""
    return paths.out / OUTPUT


def _plan[S: Spec](kind: str, settings: S) -> S:
    """Plan the settings of a task with the plan of its kind."""
    # `sbmlsim.run.plan` checks that the plan returns the spec class of the kind
    return cast(S, run.plan(kind, settings))


def _compute_unit(kind: str, settings: Spec, index: int, paths: TaskPaths) -> None:
    """Compute one unit of a task into its output port."""
    run.compute_unit(kind, settings, index, _run_out(paths), inputs=paths.inputs)


def _merge(kind: str, settings: Spec, paths: TaskPaths) -> None:
    """Merge the units of a task, its summary also in `out/summary.json`."""
    run.merge(
        kind,
        settings,
        _run_out(paths),
        inputs=paths.inputs,
        summary=paths.out / SUMMARY_FILE,
    )


def simulate_plan(settings: SimulateSpec) -> SimulateSpec:
    """Plan a simulation of the experiments of a project, see `sbmlsim.run.plan`."""
    return _plan("simulate", settings)


def fit_plan(settings: FitSpec) -> FitSpec:
    """Plan a fit: its id, its seed and its units, see `sbmlsim.run.plan`."""
    return _plan("fit", settings)


def fit_count_units(settings: FitSpec) -> int:
    """Count the units of a planned fit, see `sbmlsim.run.count_units`."""
    return run.count_units("fit", settings)


def fit_compute_unit(settings: FitSpec, index: int, paths: TaskPaths) -> None:
    """Compute one unit of a fit, see `sbmlsim.run.compute_unit`."""
    _compute_unit("fit", settings, index, paths)


def fit_merge(settings: FitSpec, paths: TaskPaths) -> None:
    """Merge the units of a fit, see `sbmlsim.run.merge`."""
    _merge("fit", settings, paths)


def identifiability_plan(settings: IdentifiabilitySpec) -> IdentifiabilitySpec:
    """Plan a profile likelihood: its parameters and units."""
    return _plan("identifiability", settings)


def identifiability_count_units(settings: IdentifiabilitySpec) -> int:
    """Count the units of a planned profile likelihood, one per parameter."""
    return run.count_units("identifiability", settings)


def identifiability_compute_unit(
    settings: IdentifiabilitySpec, index: int, paths: TaskPaths
) -> None:
    """Compute the profile of one parameter."""
    _compute_unit("identifiability", settings, index, paths)


def identifiability_merge(settings: IdentifiabilitySpec, paths: TaskPaths) -> None:
    """Merge the profiles of a profile likelihood."""
    _merge("identifiability", settings, paths)


def fisher_plan(settings: FisherSpec) -> FisherSpec:
    """Plan the Fisher information of a fit, see `sbmlsim.run.plan`."""
    return _plan("fisher", settings)


def report_plan(settings: ReportSpec) -> ReportSpec:
    """Plan the report of a fit, see `sbmlsim.run.plan`."""
    return _plan("report", settings)


def adopt_plan(settings: AdoptSpec) -> AdoptSpec:
    """Plan the adoption of a fit, see `sbmlsim.run.plan`."""
    return _plan("adopt", settings)


def petab_plan(settings: PetabSpec) -> PetabSpec:
    """Plan the PEtab problem of a fit, see `sbmlsim.run.plan`."""
    return _plan("petab", settings)


def scan_plan(settings: ScanSpec) -> ScanSpec:
    """Plan a scan: the hash of its scan and its units."""
    return _plan("scan", settings)


def scan_count_units(settings: ScanSpec) -> int:
    """Count the units of a planned scan, see `sbmlsim.run.count_units`."""
    return run.count_units("scan", settings)


def scan_compute_unit(settings: ScanSpec, index: int, paths: TaskPaths) -> None:
    """Compute one block of the points of a scan."""
    _compute_unit("scan", settings, index, paths)


def scan_merge(settings: ScanSpec, paths: TaskPaths) -> None:
    """Merge the blocks of a scan into its result."""
    _merge("scan", settings, paths)


def sensitivity_plan(settings: SensitivitySpec) -> SensitivitySpec:
    """Plan a sensitivity analysis: the hash of its scan and its units."""
    return _plan("sensitivity", settings)


def sensitivity_count_units(settings: SensitivitySpec) -> int:
    """Count the units of a planned sensitivity analysis."""
    return run.count_units("sensitivity", settings)


def sensitivity_compute_unit(
    settings: SensitivitySpec, index: int, paths: TaskPaths
) -> None:
    """Compute one block of the points of the scan of a sensitivity analysis."""
    _compute_unit("sensitivity", settings, index, paths)


def sensitivity_merge(settings: SensitivitySpec, paths: TaskPaths) -> None:
    """Merge the blocks of a sensitivity analysis and compute its indices."""
    _merge("sensitivity", settings, paths)


def uncertainty_plan(settings: UncertaintySpec) -> UncertaintySpec:
    """Plan an uncertainty propagation: its seed and its units."""
    return _plan("uncertainty", settings)


def uncertainty_count_units(settings: UncertaintySpec) -> int:
    """Count the units of a planned uncertainty propagation."""
    return run.count_units("uncertainty", settings)


def uncertainty_compute_unit(
    settings: UncertaintySpec, index: int, paths: TaskPaths
) -> None:
    """Compute one block of the draws of an uncertainty propagation."""
    _compute_unit("uncertainty", settings, index, paths)


def uncertainty_merge(settings: UncertaintySpec, paths: TaskPaths) -> None:
    """Merge the blocks of an uncertainty propagation into its result."""
    _merge("uncertainty", settings, paths)


def _task_type(kind_name: str) -> TaskType:
    """Get the task type `sbmlsim.<kind>` of a kind of the run layer.

    A splittable kind references its functions in this module, any other kind
    runs as the command of the run layer.
    """
    kind = get_kind(kind_name)
    ref = f"{MODULE}:{kind_name}"
    if kind.splittable:
        split = {
            "count_units": f"{ref}_count_units",
            "compute_unit": f"{ref}_compute_unit",
            "merge": f"{ref}_merge",
        }
        command = None
    else:
        split = {}
        command = (
            f"python -m sbmlsim.run {kind_name} {{settings}} "
            f"--out {{out}}/{OUTPUT} --summary {{out}}/{SUMMARY_FILE} "
            "--resume --input={input}"
        )
    return TaskType(
        name=f"sbmlsim.{kind_name}",
        settings=kind.spec,
        inputs=kind.inputs,
        outputs=(OUTPUT,),
        hint=HINT,
        checkpoints=True,
        command=command,
        uses_cores=kind.uses_cores,
        plan=f"{ref}_plan",
        cost_setting=kind.cost_setting,
        **split,
    )


SIMULATE = _task_type("simulate")
FIT = _task_type("fit")
IDENTIFIABILITY = _task_type("identifiability")
FISHER = _task_type("fisher")
REPORT = _task_type("report")
PETAB = _task_type("petab")
SCAN = _task_type("scan")
SENSITIVITY = _task_type("sensitivity")
UNCERTAINTY = _task_type("uncertainty")
ADOPT = _task_type("adopt")

#: the task types by the name of their kind
TASK_TYPES: dict[str, TaskType] = {
    task_type.name.removeprefix("sbmlsim."): task_type
    for task_type in (
        SIMULATE,
        FIT,
        IDENTIFIABILITY,
        FISHER,
        REPORT,
        PETAB,
        SCAN,
        SENSITIVITY,
        UNCERTAINTY,
        ADOPT,
    )
}
