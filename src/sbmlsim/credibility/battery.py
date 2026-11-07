"""The battery of the checks of a model: its variants, not its predictions.

The verification of the predictions (`sbmlsim.credibility.verification`) asks
whether the numbers a context of use is decided on are the solution of the
model. The battery asks whether the model itself is sound, independent of any
data: every variant of the project (`sbmlsim.project.Project.variants`) is
simulated with roadrunner from its initial state, without a change and with
the cases of the battery hook of the project, and the results are checked
against what every model must fulfil (`BATTERY_CHECKS`):

- `non_negativity`: no species becomes negative beyond the floor of the
  integrator in any simulation.
- `steady_state`: without a change every state but the accumulating ones
  (`accumulating_states`, the sinks which nothing drains) stays at its start
  (`BatterySettings.steady_limit`): the model starts at its steady state. A
  model which is not meant to, e.g. a decay or an oscillator, does not; the
  check runs only when the settings say the model starts at its steady state
  (`BatterySettings.steady_state`, off by default) and is not applicable
  otherwise.
- `cross_simulator`: a second engine of sbmlsim (`BatterySettings.engines`,
  AMICI and COPASI, each one which is installed) gives the results of
  roadrunner without a change and with every case of the hook which it
  repeats (`BatteryCase.cross_simulator`), within `LIMIT`; both run through
  `sbmlsim.simulator.Simulator` on the same simulation, roadrunner compiled
  at `O0`. What an engine does not simulate as roadrunner
  (`sbmlsim.engine.UnsupportedModelError`, e.g. an algebraic rule in COPASI)
  is no failure: the statement names it as not compared, and a check which
  compared nothing is not applicable. AMICI compiles a variant in a worker of
  a pool of its own (`sbmlsim.parallel.start_pool`) which is stopped after
  `BatterySettings.compile_timeout`, so a compilation which takes too long
  is not compared either.
- `plausibility`: the initial state has finite values and positive
  compartments, and no value of an adopted fit lies at a bound of its fit
  (`BatterySettings.bound_margin`).
- `rebuild`: the factory of the project builds the models again
  (`sbmlsim.project.Project.create_models` into a temporary directory),
  equal to the package but for the dates of the model history
  (`sbmlsim.project.Project.changed_outputs`).

A domain adds its simulations and checks with a `BatteryHook`, which the
project registers (`sbmlsim.project.CredibilitySettings.battery`): what it
reads from a variant (`BatteryHook.read`), the SBML the battery simulates
instead of the file (`BatteryHook.sbml`, e.g. with parameters which integrate
the flows of the reactions), further selections, the cases of its changes, e.g.
a dose of every route (`BatteryCase`), and further checks on the simulations
of every variant (`BatteryHook.contribute`, `BatteryRun`), which it registers
(`sbmlsim.credibility.verification.register_check`), with sections of plain
data in the record of the battery (`Battery.sections`) which the report shows
(`BatteryHook.report_html`, `BatteryHook.report_tables`). The settings are
the ones of the hook (`BatteryHook.settings`), else the defaults, with the
settings of the project over them
(`sbmlsim.project.CredibilitySettings.battery_settings`, see
`battery_settings_of`), so a project without a hook sets its engines, its
horizon or its tolerances as well. A check without what it needs is not
applicable and assesses nothing. A hook also says which generic check does
not apply to which variant (`BatteryHook.not_applicable`, e.g. no steady
state for a model of another kind than the others): the check runs on the
other variants and its statement names the variants left out with their
reason, and when it leaves out every variant the check is not applicable and
its statement gives the reasons per variant. The tables of the hook are
written whether or not the battery ran (`BatteryHook.report_tables`).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Self

import numpy as np

from sbmlsim.credibility.verification import (
    BATTERY_CHECKS,
    CROSS_SIMULATOR,
    LIMIT,
    LISTED_ISSUES,
    NON_NEGATIVITY,
    PLAUSIBILITY,
    REBUILD,
    SCALE_FLOOR,
    STEADY_STATE,
    Check,
    check_of,
)
from sbmlsim.engine import UnsupportedModelError
from sbmlsim.model.model_roadrunner import reset_all

if TYPE_CHECKING:
    import libsbml
    import pandas as pd
    import roadrunner

    from sbmlsim.project import Project
    from sbmlsim.simulation import Simulation

__all__ = [
    "ABSOLUTE_TOLERANCE",
    "BOUND_MARGIN",
    "COMPILE_TIMEOUT",
    "HORIZON",
    "RELATIVE_TOLERANCE",
    "STEADY_LIMIT",
    "STEPS",
    "Battery",
    "BatteryBuild",
    "BatteryCase",
    "BatteryCollector",
    "BatteryContribution",
    "BatteryHook",
    "BatteryResult",
    "BatteryRun",
    "BatterySettings",
    "BatteryStatus",
    "CompileTimeout",
    "accumulating_states",
    "battery_settings",
    "battery_settings_of",
    "math_names",
    "not_applicable_result",
    "read_build",
    "run_battery",
    "with_deadline",
]

logger = logging.getLogger(__name__)

#: the end of every simulation by default, in the time of the model
HORIZON: Final = 100.0
#: the intervals of the grid of the results by default
STEPS: Final = 100
#: the relative tolerance of the simulations of the battery
RELATIVE_TOLERANCE: Final = 1e-10
#: the absolute tolerance of the simulations of the battery, also the floor of
#: a value which is zero
ABSOLUTE_TOLERANCE: Final = 1e-16
#: the largest change of a state without a change relative to its start
STEADY_LIMIT: Final = 1e-6
#: the share of the range of a bound (in the scale of the fit) within which a
#: fitted value lies at the bound
BOUND_MARGIN: Final = 1e-3
#: the longest compilation of a variant by AMICI (s), see `BatterySettings`
COMPILE_TIMEOUT: Final = 900.0

#: the second engines of the check `cross_simulator`, by their name
_ENGINE_LABELS: Final = {"amici": "AMICI", "copasi": "COPASI"}


@unique
class BatteryStatus(StrEnum):
    """The result of a check of the battery."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, kw_only=True)
class BatteryResult:
    """The result of a check of the battery over the variants of a model.

    Attributes:
        check: the check.
        status: whether it passed, failed or was not applicable.
        cases: what was checked, e.g. the species of the non-negativity over
            the variants.
        largest: the largest relative deviation of the check, `NaN` for a
            check without one.
        at: where the largest deviation is, e.g. `default: S1`.
        problems: what failed, a line each.
        statement: what was checked in words, the reason for a check which
            is not applicable.
    """

    check: Check
    status: BatteryStatus
    cases: int
    largest: float
    at: str
    problems: tuple[str, ...]
    statement: str

    @property
    def passed(self) -> bool | None:
        """Get whether the check passed, `None` if it was not applicable."""
        if self.status is BatteryStatus.NOT_APPLICABLE:
            return None
        return self.status is BatteryStatus.PASSED

    def to_dict(self) -> dict[str, object]:
        """Get the result as plain data, `NaN` as `None`.

        The record names the label and the factor of the check, so a check of
        a hook is read without its module.
        """
        return {
            "check": self.check.id,
            "label": self.check.label,
            "factor": self.check.factor,
            "status": self.status.value,
            "cases": self.cases,
            "largest": None if math.isnan(self.largest) else self.largest,
            "at": self.at,
            "problems": list(self.problems),
            "statement": self.statement,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Get the result from plain data, `None` as `NaN`.

        A record of pkpdsim 0.19 has neither the label nor the factor of its
        checks, see `sbmlsim.credibility.verification.check_of`.
        """
        largest = data["largest"]
        return cls(
            check=check_of(data["check"], data.get("factor"), data.get("label")),
            status=BatteryStatus(data["status"]),
            cases=int(data["cases"]),
            largest=math.nan if largest is None else float(largest),
            at=data["at"],
            problems=tuple(data["problems"]),
            statement=data["statement"],
        )


@dataclass(frozen=True, kw_only=True)
class Battery:
    """The results of the battery of a model.

    Attributes:
        results: the result of every check, the generic ones and the ones of
            the hook, in the order of the report.
        sections: the sections of the hook, plain data by their name, e.g.
            the fates of the doses of a PBPK model; a record of pkpdsim 0.19
            has its fates as the section `fates`.
        settings: the settings of the simulations in words.
    """

    results: tuple[BatteryResult, ...]
    sections: Mapping[str, Any]
    settings: str

    def result(self, check: Check | str) -> BatteryResult:
        """Get the result of a check.

        Args:
            check: the check or its id.

        Returns:
            The result.

        Raises:
            KeyError: if the battery has no result of the check.
        """
        check_id = str(check)
        for result in self.results:
            if result.check.id == check_id:
                return result
        raise KeyError(f"The battery has no result of the check '{check_id}'.")

    @property
    def failed(self) -> list[BatteryResult]:
        """Get the checks which failed."""
        return [r for r in self.results if r.status is BatteryStatus.FAILED]

    def to_dict(self) -> dict[str, object]:
        """Get the battery as plain data."""
        return {
            "results": [r.to_dict() for r in self.results],
            "sections": dict(self.sections),
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Get the battery from plain data.

        A record of pkpdsim 0.19 has its sections as fields of their own,
        `fates`, which are read as the sections.
        """
        sections = (
            dict(data["sections"])
            if "sections" in data
            else {k: v for k, v in data.items() if k not in ("results", "settings")}
        )
        return cls(
            results=tuple(BatteryResult.from_dict(r) for r in data["results"]),
            sections=sections,
            settings=data["settings"],
        )


@dataclass(frozen=True, kw_only=True)
class BatterySettings:
    """The settings of the simulations of the battery.

    Attributes:
        horizon: the end of every simulation in the time of the model.
        steps: the intervals of the grid of the results.
        relative_tolerance: the relative tolerance of every simulation.
        absolute_tolerance: the absolute tolerance, also the floor of a value
            which is zero.
        steady_limit: the largest change of a state without a change relative
            to its start.
        bound_margin: the share of the range of a bound (in the scale of the
            fit) within which a fitted value lies at the bound.
        steady_state: whether the model starts at its steady state, so the
            check `steady_state` runs; it is not applicable without, e.g. for
            a decay or an oscillator.
        cross_simulator: run the check `cross_simulator`.
        engines: the second engines of the check `cross_simulator`, each one
            which is installed is compared with roadrunner.
        compile_timeout: the longest compilation of a variant by AMICI (s);
            a variant which takes longer is not compared with AMICI.
    """

    horizon: float = HORIZON
    steps: int = STEPS
    relative_tolerance: float = RELATIVE_TOLERANCE
    absolute_tolerance: float = ABSOLUTE_TOLERANCE
    steady_limit: float = STEADY_LIMIT
    bound_margin: float = BOUND_MARGIN
    steady_state: bool = False
    cross_simulator: bool = True
    engines: tuple[str, ...] = ("amici", "copasi")
    compile_timeout: float = COMPILE_TIMEOUT

    def __post_init__(self) -> None:
        """Check the settings, the engines as a tuple.

        Raises:
            ValueError: for a horizon, a tolerance, a limit or a timeout which
                is not positive, fewer than one step and an unknown engine.
        """
        object.__setattr__(self, "engines", tuple(self.engines))
        for name in (
            "horizon",
            "relative_tolerance",
            "absolute_tolerance",
            "steady_limit",
            "bound_margin",
            "compile_timeout",
        ):
            if not getattr(self, name) > 0:
                raise ValueError(
                    f"The {name} of the battery is positive, not {getattr(self, name)}."
                )
        if self.steps < 1:
            raise ValueError(f"The battery needs a step, not {self.steps}.")
        unknown = [e for e in self.engines if e not in _ENGINE_LABELS]
        if unknown:
            raise ValueError(
                f"The second engines of the battery are of {sorted(_ENGINE_LABELS)}, "
                f"not {unknown}."
            )

    @property
    def times(self) -> np.ndarray:
        """Get the times of the grid of the results."""
        return np.linspace(0.0, self.horizon, self.steps + 1)


def battery_settings_of(project: Project, hook: BatteryHook | None) -> BatterySettings:
    """Get the settings of the battery of a project.

    The settings of the hook, the defaults without one, with the settings of
    the project over them (`sbmlsim.project.CredibilitySettings.battery_settings`).

    Args:
        project: the project.
        hook: its battery hook.

    Returns:
        The settings.

    Raises:
        ValueError: for a setting of the project which the battery does not
            have or which is not valid.
    """
    settings = BatterySettings() if hook is None else hook.settings()
    overrides = dict(project.credibility_settings.battery_settings)
    known = {f.name for f in dataclasses.fields(BatterySettings)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(
            f"The battery settings of the project '{project.name}' have "
            f"{unknown}, which are no settings of the battery: {sorted(known)}."
        )
    return dataclasses.replace(settings, **overrides)


@dataclass(frozen=True, kw_only=True)
class BatteryCase:
    """A simulation of the battery with changes, e.g. a dose.

    Attributes:
        name: the name of the case, unique per variant, e.g. `IVDOSE_d`.
        changes: the changes by their time in the time of the model, the
            value of every id at that time; times on the grid of the battery.
        added: the ids whose change is added to their value instead of setting
            it, e.g. the state of a dose.
        cross_simulator: whether the check `cross_simulator` repeats the case
            with the second engines.
    """

    name: str
    changes: Mapping[float, Mapping[str, float]]
    added: frozenset[str] = frozenset()
    cross_simulator: bool = True


@dataclass(frozen=True, kw_only=True)
class BatteryBuild:
    """What the battery reads from a variant of a model.

    Attributes:
        variant: the name of the variant, e.g. `default`.
        path: the SBML file.
        sbml: the SBML the battery simulates, the file or the one of the hook.
        species: the species which are states (not constant, no boundary
            condition, no assignment rule), selected as their amounts.
        states: the parameters and compartments with a rate rule.
        accumulating: the states which nothing drains, see
            `accumulating_states`.
        selections: the further selections of the hook.
        cases: the cases of the hook.
        data: what the hook read from the variant (`BatteryHook.read`).
    """

    variant: str
    path: Path
    sbml: str
    species: tuple[str, ...]
    states: tuple[str, ...]
    accumulating: frozenset[str]
    selections: tuple[str, ...] = ()
    cases: tuple[BatteryCase, ...] = ()
    data: object = None

    @property
    def columns(self) -> list[str]:
        """Get the columns of the results: the states, the species first, and the selections of the hook."""
        return list(dict.fromkeys([*self.species, *self.states, *self.selections]))


def math_names(ast: libsbml.ASTNode | None) -> set[str]:
    """Get the names a formula of a model reads.

    Args:
        ast: the formula, `None` for none.

    Returns:
        The names of the formula (`AST_NAME`), not the time.
    """
    import libsbml

    found: set[str] = set()
    stack = [] if ast is None else [ast]
    while stack:
        node = stack.pop()
        if node.getType() == libsbml.AST_NAME:
            found.add(node.getName())
        stack.extend(node.getChild(k) for k in range(node.getNumChildren()))
    return found


def accumulating_states(model: libsbml.Model) -> tuple[str, ...]:
    """Get the states of a parsed model which nothing drains, sorted.

    A state of a rate rule or a species a reaction produces and none consumes
    accumulates when no rate (of a rate rule or a reaction) reads it, also
    through the assignment rules and the compartments of the species read: a
    sink, e.g. the amount in the urine, which grows from its start without a
    change.

    Args:
        model: the model.

    Returns:
        The ids of the accumulating states.
    """
    assigned: dict[str, set[str]] = {}
    rated: dict[str, set[str]] = {}
    for rule in model.getListOfRules():
        if rule.isAssignment():
            assigned[rule.getVariable()] = math_names(rule.getMath())
        elif rule.isRate():
            rated[rule.getVariable()] = math_names(rule.getMath())
    read = set().union(*rated.values())
    consumed: set[str] = set()
    changed: set[str] = set()
    for reaction in model.getListOfReactions():
        law = reaction.getKineticLaw()
        if law is not None:
            local = {p.getId() for p in law.getListOfLocalParameters()}
            read |= math_names(law.getMath()) - local
        consumed |= {r.getSpecies() for r in reaction.getListOfReactants()}
        changed |= {r.getSpecies() for r in reaction.getListOfProducts()}
    # what the rates read through the assignment rules, and the compartments of
    # the species they read
    compartments = {s.getId(): s.getCompartment() for s in model.getListOfSpecies()}
    stack = list(read)
    while stack:
        sid = stack.pop()
        for further in (assigned.get(sid, set()) | {compartments.get(sid, "")}) - read:
            if further:
                read.add(further)
                stack.append(further)
    states = set(rated) | (changed - consumed)
    return tuple(sorted(states - read - consumed))


def read_build(
    variant: str, path: Path, hook: BatteryHook | None = None
) -> BatteryBuild:
    """Read a variant of a model for the battery.

    Args:
        variant: the name of the variant.
        path: its SBML file.
        hook: the battery hook of the project, which reads the variant first
            and may give the SBML, further selections and cases.

    Returns:
        The variant with the SBML the battery simulates.
    """
    import libsbml

    data = None if hook is None else hook.read(variant, path)
    sbml = None if hook is None else hook.sbml(data)
    if sbml is None:
        sbml = path.read_text(encoding="utf-8")
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml)
    model: libsbml.Model = doc.getModel()
    assigned = {r.getVariable() for r in model.getListOfRules() if r.isAssignment()}
    species = tuple(
        sp.getId()
        for sp in model.getListOfSpecies()
        if not sp.getConstant()
        and not sp.getBoundaryCondition()
        and sp.getId() not in assigned
    )
    sized = {p.getId() for p in model.getListOfParameters()} | {
        c.getId() for c in model.getListOfCompartments()
    }
    states = tuple(
        rule.getVariable()
        for rule in model.getListOfRules()
        if rule.isRate() and rule.getVariable() in sized
    )
    return BatteryBuild(
        variant=variant,
        path=path,
        sbml=sbml,
        species=species,
        states=states,
        accumulating=frozenset(accumulating_states(model)),
        selections=() if hook is None else tuple(hook.selections(data)),
        cases=() if hook is None else tuple(hook.cases(data)),
        data=data,
    )


class BatteryRun:
    """The simulations of a variant with roadrunner on the grid of the battery.

    Attributes:
        build: the variant.
        settings: the settings of the battery.
        r: the instance of roadrunner, compiled at `O0`.
        columns: the selections of the results without the time, see
            `BatteryBuild.columns`.
        index: the column of every selection.
        values: the value of every global parameter in the initial state.
        zero: the simulation without a change.
        cases: the simulation of every case of the hook, by its name.
    """

    def __init__(self, build: BatteryBuild, settings: BatterySettings) -> None:
        """Load a variant at `O0` with the tolerances of the battery.

        Args:
            build: the variant.
            settings: the settings of the battery.
        """
        from sbmlsim.model.model_roadrunner import load_roadrunner

        start = time.perf_counter()
        self.build = build
        self.settings = settings
        self.r: roadrunner.RoadRunner = load_roadrunner(build.sbml, "O0")
        self.r.integrator.relative_tolerance = settings.relative_tolerance
        self.r.integrator.absolute_tolerance = settings.absolute_tolerance
        self.columns = build.columns
        self.r.timeCourseSelections = ["time", *self.columns]
        self.index = {sid: k for k, sid in enumerate(self.columns)}
        reset_all(self.r)
        self.values = {
            pid: float(self.r[pid]) for pid in self.r.model.getGlobalParameterIds()
        }
        logger.info(
            "Load the %s variant '%s' in %.1f s",
            build.variant,
            build.path.name,
            time.perf_counter() - start,
        )
        self.zero = self.run(BatteryCase(name="without a change", changes={}))
        self.cases = {case.name: self.run(case) for case in build.cases}
        logger.info(
            "Simulate the %s variant '%s' in %.1f s",
            build.variant,
            build.path.name,
            time.perf_counter() - start,
        )

    def run(self, case: BatteryCase) -> np.ndarray:
        """Simulate from the initial state with the changes of a case.

        An id of `BatteryCase.added` is increased by its change, any other id
        set to it. The row of a change time is the state before the change,
        except at 0.

        Args:
            case: the case.

        Returns:
            The results on the grid, `steps + 1` rows of the columns.

        Raises:
            ValueError: for a change at a time which is no time of the grid
                before the horizon.
        """
        horizon = self.settings.horizon
        step = horizon / self.settings.steps
        for at in case.changes:
            k = at / step
            if not (0.0 <= at < horizon) or not math.isclose(
                k, round(k), rel_tol=0.0, abs_tol=1e-9 * max(1.0, abs(k))
            ):
                raise ValueError(
                    f"The case '{case.name}' changes the model at {at:g}, which "
                    f"is no time of the grid of the battery before its horizon "
                    f"{horizon:g}, a multiple of {step:g}."
                )
        r = self.r
        reset_all(r)
        times = sorted({0.0, *case.changes})
        bounds = [*times, horizon]
        rows: list[np.ndarray] = []
        for k, start in enumerate(times):
            for sid, value in case.changes.get(start, {}).items():
                r[sid] = float(r[sid]) + value if sid in case.added else value
            end = bounds[k + 1]
            points = round((end - start) / step) + 1
            result = np.asarray(r.simulate(start, end, points))[:, 1:]
            rows.append(result if k == 0 else result[1:])
        return np.vstack(rows)

    def column(self, result: np.ndarray, sid: str) -> np.ndarray:
        """Get the values of a selection over the grid.

        Args:
            result: a simulation of the variant.
            sid: the selection.

        Returns:
            Its values.
        """
        return result[:, self.index[sid]]

    def results(self) -> list[tuple[str, np.ndarray]]:
        """Get every simulation by its name, the one without a change first."""
        return [("without a change", self.zero), *self.cases.items()]

    def scales(self, ids: Sequence[str]) -> np.ndarray:
        """Get the largest absolute value of ids over the simulations.

        Args:
            ids: the selections.

        Returns:
            The largest absolute value of every id.
        """
        columns = [self.index[sid] for sid in ids]
        return np.max(
            np.abs(np.vstack([result[:, columns] for _, result in self.results()])),
            axis=0,
            initial=0.0,
        )


@dataclass
class BatteryCollector:
    """The cases and problems of a check over the variants.

    Attributes:
        cases: what was checked.
        largest: the largest relative deviation, `NaN` without one.
        at: where the largest deviation is.
        problems: what failed, a line each.
    """

    cases: int = 0
    largest: float = math.nan
    at: str = ""
    problems: list[str] = dataclasses.field(default_factory=list)

    def deviation(self, value: float, at: str) -> None:
        """Record a deviation, the largest one is kept, where only above zero.

        Args:
            value: the relative deviation.
            at: where it is.
        """
        if math.isnan(self.largest) or value > self.largest:
            self.largest, self.at = value, at if value > 0 else ""

    def result(self, check: Check, statement: str) -> BatteryResult:
        """Get the result of the check, failed with a problem.

        Args:
            check: the check.
            statement: what it checks in words.

        Returns:
            The result, with the first `LISTED_ISSUES` problems.
        """
        problems = self.problems[:LISTED_ISSUES]
        if len(self.problems) > LISTED_ISSUES:
            problems.append(f"and {len(self.problems) - LISTED_ISSUES} more")
        return BatteryResult(
            check=check,
            status=BatteryStatus.FAILED if self.problems else BatteryStatus.PASSED,
            cases=self.cases,
            largest=self.largest,
            at=self.at,
            problems=tuple(problems),
            statement=statement,
        )


def not_applicable_result(check: Check, reason: str) -> BatteryResult:
    """Get the result of a check which is not applicable.

    Args:
        check: the check.
        reason: why it is not applicable.

    Returns:
        The result.
    """
    return BatteryResult(
        check=check,
        status=BatteryStatus.NOT_APPLICABLE,
        cases=0,
        largest=math.nan,
        at="",
        problems=(),
        statement=reason,
    )


@dataclass(frozen=True, kw_only=True)
class BatteryContribution:
    """What a battery hook adds to the battery.

    Attributes:
        results: the results of its checks, of `BatteryHook.checks`.
        sections: its sections of plain data by their name, e.g. `fates`.
    """

    results: tuple[BatteryResult, ...] = ()
    sections: Mapping[str, Any] = dataclasses.field(default_factory=dict)


class BatteryHook:
    """The simulations and checks a project adds to the battery, see the module.

    A subclass names itself and overrides what it adds; every method adds
    nothing by default. A hook registers its checks
    (`sbmlsim.credibility.verification.register_check`) when its module is
    imported. `BatteryHook.not_applicable` leaves a variant out of a generic check.

    Attributes:
        name: the name of the hook, e.g. `pk`.
        checks: the checks the hook adds, whose results `contribute` gives.
        order: the ids of every check of the battery in the order of the
            report, the generic ones (`BATTERY_CHECKS`) among them; empty for
            the generic checks followed by the checks of the hook.
    """

    name: ClassVar[str] = ""
    checks: ClassVar[tuple[Check, ...]] = ()
    order: ClassVar[tuple[str, ...]] = ()

    def settings(self) -> BatterySettings:
        """Get the settings of the simulations of the battery.

        Returns:
            The settings, the defaults by default.
        """
        return BatterySettings()

    def read(self, variant: str, path: Path) -> object:
        """Read what the hook needs from a variant.

        Args:
            variant: the name of the variant.
            path: its SBML file.

        Returns:
            What `sbml`, `selections`, `cases` and `contribute` get as
            `BatteryBuild.data`, `None` by default.
        """
        return None

    def sbml(self, data: object) -> str | None:
        """Get the SBML the battery simulates instead of the file of the variant.

        Args:
            data: what `read` returned.

        Returns:
            The SBML, `None` for the file.
        """
        return None

    def selections(self, data: object) -> Sequence[str]:
        """Get the further selections of the simulations of a variant.

        Args:
            data: what `read` returned.

        Returns:
            The selections, e.g. parameters which integrate the flows.
        """
        return ()

    def cases(self, data: object) -> Sequence[BatteryCase]:
        """Get the simulations with changes of a variant, e.g. its doses.

        Args:
            data: what `read` returned.

        Returns:
            The cases.
        """
        return ()

    def contribute(self, runs: Sequence[BatteryRun]) -> BatteryContribution:
        """Check the simulations of the variants with the checks of the hook.

        Args:
            runs: the simulations of every variant.

        Returns:
            The results of the checks and the sections of the hook.
        """
        return BatteryContribution()

    def not_applicable(self, check: str, data: object) -> str | None:
        """Get why a generic check does not apply to a variant, None when it does.

        Args:
            check: the id of a generic check of the battery (`BATTERY_CHECKS`:
                `non_negativity`, `steady_state`, `cross_simulator`,
                `plausibility`, `rebuild`).
            data: what `read` returned for the variant.

        Returns:
            The reason the variant is left out of the check, `None` (by
            default) when the check applies to it.
        """
        return None

    def describe(self) -> str:
        """Get the settings of the cases of the hook in words.

        Returns:
            A sentence the settings of the battery end with, e.g. the doses;
            empty by default.
        """
        return ""

    def report_html(self, sections: Mapping[str, Any]) -> str:
        """Get the HTML of the sections of the hook in the report.

        Args:
            sections: the sections of the hook in the record of the battery.

        Returns:
            The HTML after the table of the battery, e.g. the fates of the
            doses; empty for none.
        """
        return ""

    def report_tables(self, sections: Mapping[str, Any]) -> Mapping[str, pd.DataFrame]:
        """Get the tables the report writes for the sections of the hook.

        Args:
            sections: the sections of the hook in the record of the battery.

        Returns:
            The tables by the name of their file without the suffix, e.g.
            `fates`; empty for none. The report calls this whenever the
            project has a hook, with empty sections when the battery did not
            run, and the hook must then return its tables empty, with their
            columns, so the files of a report depend on the hook only.
        """
        return {}


def battery_settings(settings: BatterySettings, hook: BatteryHook | None) -> str:
    """Get the settings of the simulations of the battery in words.

    Args:
        settings: the settings.
        hook: the battery hook, whose description follows.

    Returns:
        The settings in words.
    """
    text = (
        f"roadrunner CVODE BDF, rtol {settings.relative_tolerance:g}, atol "
        f"{settings.absolute_tolerance:g}; every simulation from 0 to "
        f"{settings.horizon:g} on a grid of {settings.steps} intervals."
    )
    if settings.steady_state:
        text += " The model starts at its steady state, which is checked."
    if settings.cross_simulator:
        text += (
            f" Second simulators: {', '.join(settings.engines)}, a compilation "
            f"longer than {settings.compile_timeout:g} s is not compared."
        )
    described = "" if hook is None else hook.describe()
    return f"{text} {described}" if described else text


def _non_negativity(
    runs: Sequence[BatteryRun], collector: BatteryCollector, floor: float
) -> None:
    """Check that no species becomes negative beyond the floor."""
    for run in runs:
        species = list(run.build.species)
        if not species:
            continue
        columns = [run.index[sid] for sid in species]
        scale = run.scales(species)
        limit = np.maximum(SCALE_FLOOR * scale, floor)
        collector.cases += len(species)
        for name, result in run.results():
            low = result[:, columns].min(axis=0)
            relative = np.maximum(-low, 0.0) / np.maximum(scale, floor)
            k = int(np.argmax(relative))
            collector.deviation(
                float(relative[k]), f"{run.build.variant} {name}: {species[k]}"
            )
            for k in np.flatnonzero(low < -limit):
                collector.problems.append(
                    f"{run.build.variant} {name}: '{species[k]}' reaches "
                    f"{low[k]:.3g}, below the floor {-limit[k]:.3g}"
                )


def _steady_state(
    runs: Sequence[BatteryRun], collector: BatteryCollector, settings: BatterySettings
) -> None:
    """Check that every state but the sinks stays at its start without a change.

    A state may change by the steady limit of its start plus the floor of its
    largest value over the simulations of its variant, so a state which
    starts at zero may move by the noise of the integrator.
    """
    floor = settings.absolute_tolerance
    for run in runs:
        build = run.build
        ids = [
            sid
            for sid in dict.fromkeys((*build.species, *build.states))
            if sid not in build.accumulating
        ]
        if not ids:
            continue
        values = run.zero[:, [run.index[sid] for sid in ids]]
        start = values[0]
        change = np.max(np.abs(values - start), axis=0)
        allowed = settings.steady_limit * np.abs(start) + np.maximum(
            SCALE_FLOOR * run.scales(ids), floor
        )
        relative = np.where(
            start != 0.0, change / np.maximum(np.abs(start), floor), 0.0
        )
        collector.cases += len(ids)
        k = int(np.argmax(relative))
        collector.deviation(float(relative[k]), f"{build.variant}: {ids[k]}")
        for k in np.flatnonzero(change > allowed):
            collector.problems.append(
                f"{build.variant}: '{ids[k]}' changes by {change[k]:.3g} from "
                f"{start[k]:.3g} without a change"
            )


def _plausibility(
    project: Project,
    runs: Sequence[BatteryRun],
    collector: BatteryCollector,
    margin: float,
) -> None:
    """Check the values of the initial state and of the adopted fits."""
    for run in runs:
        r = run.r
        reset_all(r)
        name = run.build.variant
        for cid in r.model.getCompartmentIds():
            size = float(r[cid])
            collector.cases += 1
            if not math.isfinite(size) or size <= 0.0:
                collector.problems.append(
                    f"{name}: the compartment '{cid}' has the size {size:g}"
                )
        for pid in r.model.getGlobalParameterIds():
            value = float(r[pid])
            collector.cases += 1
            if not math.isfinite(value):
                collector.problems.append(f"{name}: the parameter '{pid}' is {value}")
        for sid in r.model.getFloatingSpeciesIds():
            value = float(r[sid])
            collector.cases += 1
            if not math.isfinite(value) or value < 0.0:
                collector.problems.append(
                    f"{name}: the species '{sid}' starts at {value:g}"
                )
    try:
        if not project.adopted_fits():
            return
        records = project.fit_records()
    except (FileNotFoundError, ValueError) as err:
        collector.problems.append(str(err))
        return
    for pid, parameter in records.parameters.items():
        collector.cases += 1
        scale = parameter.scale
        value = float(scale.to_scale(parameter.value))
        lower, upper = parameter.lower_bound, parameter.upper_bound
        if not (math.isfinite(lower) and math.isfinite(upper)):
            continue
        span = abs(float(scale.to_scale(upper)) - float(scale.to_scale(lower)))
        for side, bound in (("lower", lower), ("upper", upper)):
            distance = (
                abs(value - float(scale.to_scale(bound))) / span if span > 0 else 0.0
            )
            if distance < margin:
                collector.problems.append(
                    f"the value {parameter.value:g} of '{pid}' of the fit "
                    f"'{parameter.fit_id}' lies at its {side} bound {bound:g}"
                )


def _rebuild(project: Project) -> BatteryResult:
    """Check that the factory builds the models of the project again."""
    statement = (
        "The factory builds the models of the project again, equal but for "
        "the dates of the model history."
    )
    if project.factory is None or project.factory_dir is None:
        return not_applicable_result(REBUILD, "The project has no factory.")
    collector = BatteryCollector()
    with tempfile.TemporaryDirectory(prefix="sbmlsim-rebuild-") as tmp:
        out = Path(tmp)
        start = time.perf_counter()
        try:
            project.create_models(out)
        except Exception as err:  # a failure of a factory of a model is the result
            collector.problems.append(
                f"the factory failed: {type(err).__name__}: {err}"
            )
            return collector.result(REBUILD, statement)
        logger.info("Rebuild '%s' in %.1f s", project.name, time.perf_counter() - start)
        compared = project.compared_outputs(out)
        collector.cases = len(compared)
        if not compared:
            collector.problems.append(
                "the rebuild has nothing to compare: no factory_outputs, no "
                "variants and no file of the package"
            )
        for path in project.changed_outputs(out):
            collector.problems.append(f"'{path.as_posix()}' differs from its rebuild")
    return collector.result(REBUILD, statement)


def _simulation(case: BatteryCase, settings: BatterySettings) -> Simulation:
    """Get the simulation of sbmlsim of a case of the battery."""
    from sbmlsim.simulation import Change, Simulation

    changes = [
        Change(
            at,
            {
                sid: (f"{sid} + {value!r}" if sid in case.added else value)
                for sid, value in values.items()
            },
        )
        for at, values in sorted(case.changes.items())
    ]
    return Simulation(
        start=0.0, end=settings.horizon, changes=changes, times=list(settings.times)
    )


class CompileTimeout(Exception):
    """A compilation did not end within its deadline, see `with_deadline`."""


def with_deadline(
    function: Callable[..., Any], args: Sequence[Any], timeout: float
) -> Any:  # what the function returns
    """Call a function in a worker of a pool of its own, stopped after a deadline.

    In a worker of a pool, which starts no pool, the function is called in
    the process, without a deadline.

    Args:
        function: a module level function.
        args: its arguments, picklable.
        timeout: the deadline (s).

    Returns:
        What the function returns; an exception of the function is raised.

    Raises:
        CompileTimeout: if the function did not return within the deadline;
            the worker is stopped.
    """
    from concurrent.futures import TimeoutError as FutureTimeout

    from sbmlsim import parallel

    if parallel.in_worker():
        logger.warning(
            "A worker of a pool calls %s without a deadline.",
            getattr(function, "__qualname__", function),
        )
        return function(*args)
    pool = parallel.start_pool(1)
    try:
        return pool.submit(function, *args).result(timeout=timeout)
    except FutureTimeout as err:
        raise CompileTimeout(f"took longer than {timeout:g} s") from err
    finally:
        parallel.stop(pool)


def _compile_amici(sbml: str, selections: list[str], cache: str | None) -> None:
    """Compile a variant with AMICI into the cache of compiled models of AMICI."""
    from sbmlsim.engine import Engine, using_amici_cache
    from sbmlsim.model import AbstractModel
    from sbmlsim.model.model_sbml import load_model

    with using_amici_cache(cache):
        load_model(AbstractModel(sbml, selections=selections), Engine.AMICI)


def _engine_model(
    engine: str, build: BatteryBuild, settings: BatterySettings
) -> tuple[Any, Any]:
    """Load a variant in an engine with the tolerances of the battery.

    roadrunner compiles the variant at `O0`; AMICI compiles it in a worker
    with the deadline of the settings first, see `with_deadline`.

    Returns:
        The simulator and the model.

    Raises:
        UnsupportedModelError: if the engine does not simulate the variant as
            roadrunner.
        CompileTimeout: if AMICI did not compile the variant in time.
    """
    from sbmlsim.engine import Engine, amici_cache, using_amici_cache
    from sbmlsim.model import AbstractModel
    from sbmlsim.model.model_sbml import load_model
    from sbmlsim.simulator import Simulator

    simulator = Simulator(
        n_workers=1,
        engine=engine,
        relative_tolerance=settings.relative_tolerance,
        absolute_tolerance=settings.absolute_tolerance,
    )
    selections = ["time", *build.species, *build.states]
    description = AbstractModel(build.sbml, selections=selections)
    if engine == Engine.ROADRUNNER:
        return simulator, simulator.load(
            load_model(description, Engine.ROADRUNNER, jit_optimization="O0")
        )
    if engine == Engine.AMICI:
        cache = amici_cache(Engine.AMICI, ())
        with_deadline(
            _compile_amici, (build.sbml, selections, cache), settings.compile_timeout
        )
        with using_amici_cache(cache):
            return simulator, simulator.load(description)
    return simulator, simulator.load(description)


def _values(
    simulator: Any,  # a Simulator, imported lazily
    model: Any,  # its model
    build: BatteryBuild,
    case: BatteryCase,
    settings: BatterySettings,
) -> np.ndarray:
    """Simulate a case with an engine, the amounts of the species and the states."""
    result = simulator.simulate(model, _simulation(case, settings))
    return np.column_stack([result[c] for c in (*build.species, *build.states)])


def _installed(engine: str) -> bool:
    """Check whether a second engine is installed."""
    from sbmlsim.engine import import_amici, import_copasi

    try:
        (import_amici if engine == "amici" else import_copasi)()
    except ImportError:
        return False
    return True


def _cross_simulator(
    runs: Sequence[BatteryRun], settings: BatterySettings
) -> BatteryResult:
    """Compare the results of roadrunner with the results of the second engines.

    A variant or a case an engine does not simulate as roadrunner, or a
    variant AMICI does not compile in time, is not compared and named in the
    statement; an error of an engine on anything else is a problem.
    """
    engines = [e for e in settings.engines if _installed(e)]
    if not engines:
        names = [_ENGINE_LABELS[e] for e in settings.engines]
        missing = (
            f"{names[0]} is not installed"
            if len(names) == 1
            else f"Neither {' nor '.join(names)} is installed"
        )
        return not_applicable_result(
            CROSS_SIMULATOR,
            f"{missing}, install sbmlsim with the extra "
            + " or ".join(f"'{e}'" for e in settings.engines)
            + ".",
        )
    labels = " and ".join(_ENGINE_LABELS[e] for e in engines)
    statement = (
        f"{labels} {'gives' if len(engines) == 1 else 'give'} the results of "
        "roadrunner without a change and with every case of the battery hook "
        f"it repeats, within {LIMIT:g}."
    )
    collector = BatteryCollector()
    skipped: list[str] = []
    floor = settings.absolute_tolerance
    for run in runs:
        build = run.build
        columns = [*build.species, *build.states]
        if not columns:
            continue
        cases = [
            BatteryCase(name="without a change", changes={}),
            *(c for c in build.cases if c.cross_simulator),
        ]
        try:
            reference, model = _engine_model("roadrunner", build, settings)
            expected = [_values(reference, model, build, c, settings) for c in cases]
        except (RuntimeError, ValueError) as err:
            collector.problems.append(f"{build.variant}: roadrunner failed: {err}")
            continue
        for engine in engines:
            label = _ENGINE_LABELS[engine]
            try:
                simulator, model = _engine_model(engine, build, settings)
            except UnsupportedModelError as err:
                skipped.append(f"{build.variant}: not compared with {label}: {err}")
                continue
            except CompileTimeout as err:
                skipped.append(
                    f"{build.variant}: not compared with {label}: compiling {err}."
                )
                continue
            except Exception as err:  # an error of an engine on a variant is the result
                collector.problems.append(
                    f"{build.variant}: {label} cannot simulate the variant: "
                    f"{type(err).__name__}: {err}"
                )
                continue
            for case, want in zip(cases, expected, strict=True):
                try:
                    actual = _values(simulator, model, build, case, settings)
                except UnsupportedModelError as err:
                    skipped.append(
                        f"{build.variant} {case.name}: not compared with {label}: {err}"
                    )
                    continue
                except (
                    Exception
                ) as err:  # an error of an engine on a case is the result
                    collector.problems.append(
                        f"{build.variant} {case.name}: {label} failed: "
                        f"{type(err).__name__}: {err}"
                    )
                    continue
                collector.cases += 1
                scale = np.max(np.abs(want), axis=0)
                limit = np.maximum(SCALE_FLOOR * scale, floor)
                with np.errstate(invalid="ignore", divide="ignore"):
                    difference = np.abs(actual - want) / np.maximum(
                        np.maximum(np.abs(want), np.abs(actual)), limit
                    )
                difference = np.where(actual == want, 0.0, difference)
                largest = np.nanmax(
                    np.where(np.isfinite(difference), difference, np.inf), axis=0
                )
                k = int(np.argmax(largest))
                at = f"{build.variant} {case.name} {label}: {columns[k]}"
                collector.deviation(float(largest[k]), at)
                if largest[k] > LIMIT:
                    collector.problems.append(
                        f"{build.variant} {case.name}: '{columns[k]}' differs "
                        f"between {label} and roadrunner by {largest[k]:.2g}"
                    )
    if skipped:
        statement = f"{statement} " + " ".join(
            s if s.endswith(".") else f"{s}." for s in skipped
        )
        if not collector.cases and not collector.problems:
            return not_applicable_result(
                CROSS_SIMULATOR,
                "No case was compared. "
                + " ".join(s if s.endswith(".") else f"{s}." for s in skipped),
            )
    return collector.result(CROSS_SIMULATOR, statement)


_STATEMENTS: Final = {
    NON_NEGATIVITY.id: (
        "No species becomes negative beyond {floor:g} of its largest value."
    ),
    STEADY_STATE.id: (
        "Without a change every state but the sinks stays at its start, within "
        "{steady:g} of it."
    ),
    PLAUSIBILITY.id: (
        "The initial state has finite values, positive compartments and "
        "species which are not negative, and no fitted value lies within "
        "{margin:g} of the range of a bound."
    ),
}


def _statement(check: Check, settings: BatterySettings) -> str:
    """Get what a generic check checks, in words."""
    return _STATEMENTS[check.id].format(
        floor=SCALE_FLOOR, steady=settings.steady_limit, margin=settings.bound_margin
    )


def _ordered(
    results: Mapping[str, BatteryResult], hook: BatteryHook | None
) -> tuple[BatteryResult, ...]:
    """Order the results of the battery, see `BatteryHook.order`.

    Raises:
        ValueError: if the order of the hook is not every check of the battery
            once.
    """
    if hook is not None and hook.order:
        order = list(hook.order)
    else:
        own = () if hook is None else hook.checks
        order = [*(c.id for c in BATTERY_CHECKS), *(c.id for c in own)]
    if sorted(order) != sorted(results):
        raise ValueError(
            f"The order of the battery {order} is not its checks {sorted(results)}."
        )
    return tuple(results[check] for check in order)


def _with_left_out(
    check: Check, result: BatteryResult, left: Mapping[str, str], variants: int
) -> BatteryResult:
    """Name the variants a hook left out of a check, see `BatteryHook.not_applicable`.

    A check which left out every variant is not applicable and gives the
    reasons; a check which ran on the others keeps its status and names them
    in its statement.
    """
    if not left:
        return result
    text = " ".join(f"{v}: {reason.rstrip('.')}." for v, reason in left.items())
    if len(left) == variants:
        return not_applicable_result(check, f"Not applicable to any variant. {text}")
    return dataclasses.replace(result, statement=f"{result.statement} Left out: {text}")


def run_battery(
    project: Project,
    *,
    rebuild: bool = True,
    cross_simulator: bool = True,
    hook: BatteryHook | None = None,
) -> Battery:
    """Run the battery of the checks of a project on its variants.

    Args:
        project: the project, every variant of `Project.variants`.
        rebuild: run the check `rebuild`, which runs the factory.
        cross_simulator: run the check `cross_simulator` with the second
            engines which are installed, as the settings say; `False`
            switches it off whatever the settings.
        hook: the battery hook, the one of the project
            (`sbmlsim.project.Project.battery_hook`) without.

    Returns:
        The results of the checks and the sections of the hook.

    Raises:
        ValueError: if the hook gives a result of a check it does not add, or
            misses one, and for invalid settings of the project.
    """
    hook = project.battery_hook() if hook is None else hook
    settings = battery_settings_of(project, hook)
    builds = [read_build(name, path, hook) for name, path in project.variants.items()]
    runs = [BatteryRun(build, settings) for build in builds]
    results: dict[str, BatteryResult] = {}
    left: dict[str, dict[str, str]] = {}

    def applying(check: Check) -> list[BatteryRun]:
        """Get the runs of the variants the hook does not leave out of a check."""
        reasons = {}
        if hook is not None:
            for run in runs:
                reason = hook.not_applicable(check.id, run.build.data)
                if reason is not None:
                    reasons[run.build.variant] = reason
        left[check.id] = reasons
        return [r for r in runs if r.build.variant not in reasons]

    def named(check: Check, result: BatteryResult) -> BatteryResult:
        """Name the variants left out of a check in its statement."""
        return _with_left_out(check, result, left[check.id], len(runs))

    checks = [
        (
            NON_NEGATIVITY,
            lambda r, c: _non_negativity(r, c, settings.absolute_tolerance),
            "The model has no species.",
        ),
    ]
    if settings.steady_state:
        checks.append(
            (
                STEADY_STATE,
                lambda r, c: _steady_state(r, c, settings),
                "The model has no state which does not accumulate.",
            )
        )
    else:
        results[STEADY_STATE.id] = not_applicable_result(
            STEADY_STATE,
            "The model is not declared to start at its steady state "
            "(BatterySettings.steady_state), e.g. a decay or an oscillator.",
        )
    for check, function, nothing in checks:
        collector = BatteryCollector()
        function(applying(check), collector)
        results[check.id] = named(
            check,
            collector.result(check, _statement(check, settings))
            if collector.cases
            else not_applicable_result(check, nothing),
        )
    results[CROSS_SIMULATOR.id] = (
        named(CROSS_SIMULATOR, _cross_simulator(applying(CROSS_SIMULATOR), settings))
        if cross_simulator and settings.cross_simulator
        else not_applicable_result(CROSS_SIMULATOR, "Not run.")
    )
    collector = BatteryCollector()
    _plausibility(project, applying(PLAUSIBILITY), collector, settings.bound_margin)
    results[PLAUSIBILITY.id] = named(
        PLAUSIBILITY, collector.result(PLAUSIBILITY, _statement(PLAUSIBILITY, settings))
    )
    rebuilt = applying(REBUILD)
    results[REBUILD.id] = named(
        REBUILD,
        _rebuild(project)
        if rebuild and (rebuilt or not runs)
        else not_applicable_result(REBUILD, "Not run."),
    )
    sections: Mapping[str, Any] = {}
    if hook is not None:
        contribution = hook.contribute(runs)
        given = [r.check.id for r in contribution.results]
        expected = [c.id for c in hook.checks]
        if sorted(given) != sorted(expected):
            raise ValueError(
                f"The battery hook '{hook.name}' gives the results of {given}, not "
                f"of its checks {expected}."
            )
        results.update({r.check.id: r for r in contribution.results})
        sections = dict(contribution.sections)
    return Battery(
        results=_ordered(results, hook),
        sections=sections,
        settings=battery_settings(settings, hook),
    )
