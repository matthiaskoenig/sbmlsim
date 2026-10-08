# Robust integrator tolerances Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CVODE gets one absolute tolerance per state, set by sbmlsim from the kind of the state, and a model which does not read the time is integrated in local time, so that late restarts no longer warn "t + h = t".

**Architecture:** A pure module `sbmlsim/model/tolerances.py` turns a tolerance per kind (`AbsoluteTolerance`) into a tolerance per state from the `ModelSymbols` and the initial volumes; `RoadrunnerSBMLModel.set_integrator_settings` (now a method of the model) sets it by id with `setIndividualTolerance`. `ModelSymbols.time_dependent` decides whether the executor integrates every segment of a plan from the time 0.

**Tech Stack:** python 3.13+, libroadrunner 2.10 (CVODE), python-libsbml, numpy, pandas, pytest (xdist), ruff, ty.

**Spec:** `docs/superpowers/specs/2026-10-08-integrator-tolerances-design.md`

## Global Constraints

- Tolerances are plain numbers in the units of the model of the state; no pint quantities.
- Kinds: `AMOUNT` (species with `hasOnlySubstanceUnits=true`), `CONCENTRATION` (other species, tolerance times the reference volume of the compartment), `OTHER` (every other state, i.e. the targets of rate rules).
- Reference volume: the initial volume, raised to `1e-6` times the largest finite positive initial volume when it is smaller, not finite or not positive.
- Defaults: `1e-10` for every kind in `SimulatorSerial`, `1e-6` in `FitSettings`; `relative_tolerance` stays a scalar.
- A float tolerance means the same number for every kind; stored settings with a float read back.
- `ModelSymbols.time_dependent` is true if a rule, kinetic law, initial assignment, function definition or event (trigger, delay, priority, assignment) reads the csymbol `time` or `delay`; such a model keeps the absolute time.
- Every module and function of the package has type annotations and a google style docstring; `ruff check`, `ruff format` and `uv run ty check` stay at zero diagnostics; markdown has no hard line wraps; never the em dash character.
- Commits end without any attribution line.

## Review Focus

- An override for an id which is not a state (e.g. a parameter without a rate rule) raises a `ValueError` which names the states; covered in Task 1.
- A model whose compartments have no finite positive volume (all `NaN` or 0) gets the reference volume `1e-6` and no crash; covered in Task 1.
- An event whose trigger becomes true by a change fires in local time as in absolute time; covered in Task 4.
- A formula of a change which reads `time` is evaluated with the absolute time of the change in local time; covered in Task 4.
- Stored settings of 0.8.4 (a float `absolute_tolerance`, also inside the `sbmlsim` block of a PEtab problem) read back; covered in Task 3.

---

## File Structure

- Create `src/sbmlsim/model/tolerances.py`: `StateKind`, `AbsoluteTolerance`, `StateTolerance`, `state_kinds`, `state_tolerances`. Pure, no roadrunner.
- Modify `src/sbmlsim/model/symbols.py`: `ModelSymbols.time_dependent` and `_reads_time`.
- Modify `src/sbmlsim/model/model_roadrunner.py`: `set_integrator_settings` becomes a method, `state_ids`, `tolerances`, the default tolerance at the load; `_tolerance_volume_factor` and `set_default_settings` are removed.
- Modify `src/sbmlsim/simulator/simulation_serial.py`: the simulator calls the method of the model.
- Modify `src/sbmlsim/simulator/executor.py`: local time.
- Modify `src/sbmlsim/fit/options.py`, `fit/optimization.py`, `fit/display.py`, `fit/report.py`, `resources/templates/fit_report.html`: `FitSettings.absolute_tolerance` is an `AbsoluteTolerance`, the report shows the tolerances of the states.
- Tests: create `tests/model/test_tolerances.py`, `tests/model/test_time_dependent.py`, `tests/simulator/test_local_time.py`; modify `tests/simulator/models.py` (the tolerance probe), `tests/simulator/test_simulator_serial.py`, `tests/fit/test_fit.py`, `tests/fit/test_display.py`, `tests/fit/test_report.py`.
- Docs: `docs/simulation.md`, `docs/models.md`, `docs/fitting.md`, `CLAUDE.md`, the spec (amendments).
- pkdb_models (`/home/mkoenig/git/pkdb_models`, branch `sbmlsim-0.8.4`): `pkdb_models/models/hctz/fitting/fitting.py`, `pkdb_models/models/hctz/helpers.py`.

Environment: `cd /home/mkoenig/git/sbmlsim-fix-084`, run tools from `.venv/bin/` (`.venv/bin/pytest`, `.venv/bin/ruff`, `.venv/bin/ty check`); the environment was synced with `UV_NO_SOURCES=1 uv sync --extra dev`.

---

### Task 1: The tolerances per state

**Files:**
- Create: `src/sbmlsim/model/tolerances.py`
- Modify: `tests/simulator/models.py` (add `TOLERANCE_PROBE`)
- Test: `tests/model/test_tolerances.py`

**Interfaces:**
- Consumes: `sbmlsim.model.symbols.ModelSymbols` (`species`, `only_substance`, `species_compartment`).
- Produces:
  - `class StateKind(StrEnum)`: `AMOUNT = "amount"`, `CONCENTRATION = "concentration"`, `OTHER = "other"`.
  - `@dataclass(frozen=True) class AbsoluteTolerance(amount: float = 1e-10, concentration: float = 1e-10, other: float = 1e-10, ids: Mapping[str, float] | tuple[tuple[str, float], ...] = ())` with `of(value: float | AbsoluteTolerance) -> AbsoluteTolerance` (classmethod), `overrides -> dict[str, float]` (property), `of_kind(kind: StateKind) -> float`, `to_dict() -> dict[str, Any]`, `from_dict(d: float | Mapping[str, Any]) -> AbsoluteTolerance` (classmethod), `__str__`.
  - `@dataclass(frozen=True) class StateTolerance(sid: str, kind: StateKind, compartment: str | None, volume: float | None, volume_raised: bool, absolute_tolerance: float)`.
  - `VOLUME_FLOOR: float = 1e-6`.
  - `state_kinds(states: Iterable[str], symbols: ModelSymbols) -> dict[str, StateKind]`.
  - `state_tolerances(states: Sequence[str], symbols: ModelSymbols, initial_volumes: Mapping[str, float], tolerance: AbsoluteTolerance) -> list[StateTolerance]`.

- [ ] **Step 1: Add the tolerance probe model**

Append to `tests/simulator/models.py`:

```python
#: states of every kind: concentration species in a normal and in a degenerate
#: compartment, an amount species, a parameter with a rate rule, a species
#: with an assignment rule and a boundary species, which are no states
TOLERANCE_PROBE = """
model tolerances
  compartment C = 2; compartment U = 1e-12;
  species A in C; species S in U; substanceOnly species X in C;
  species Y in C; $Bnd in C;
  A = 1; S = 0; X = 3; Bnd = 1; D = 5
  Y := 2*A
  D' = -0.1*D
  J1: A -> X; 0.5*A
  J2: A -> S; 0.1*A
end
"""
```

- [ ] **Step 2: Write the failing tests**

Create `tests/model/test_tolerances.py`:

```python
"""The absolute tolerances of the integrator, one per state of a model."""

import math

import pytest

from sbmlsim.model.symbols import ModelSymbols
from sbmlsim.model.tolerances import (
    VOLUME_FLOOR,
    AbsoluteTolerance,
    StateKind,
    state_kinds,
    state_tolerances,
)
from tests.simulator.models import TOLERANCE_PROBE, sbml

#: the states of the probe as roadrunner integrates them
STATES = ["A", "S", "X", "D"]
VOLUMES = {"C": 2.0, "U": 1e-12}


@pytest.fixture(scope="module")
def symbols() -> ModelSymbols:
    """Get the symbols of the tolerance probe."""
    return ModelSymbols.from_sbml(sbml(TOLERANCE_PROBE))


def test_kinds(symbols: ModelSymbols) -> None:
    """Amount species, concentration species and other states are told apart."""
    assert state_kinds(STATES, symbols) == {
        "A": StateKind.CONCENTRATION,
        "S": StateKind.CONCENTRATION,
        "X": StateKind.AMOUNT,
        "D": StateKind.OTHER,
    }


def test_tolerance_per_state(symbols: ModelSymbols) -> None:
    """A concentration species gets its tolerance times the reference volume."""
    tolerance = AbsoluteTolerance(amount=1e-9, concentration=1e-8, other=1e-7)
    by_id = {t.sid: t for t in state_tolerances(STATES, symbols, VOLUMES, tolerance)}
    assert by_id["A"].absolute_tolerance == pytest.approx(1e-8 * 2.0)
    assert by_id["A"].compartment == "C"
    assert by_id["X"].absolute_tolerance == pytest.approx(1e-9)
    assert by_id["X"].volume is None
    assert by_id["D"].absolute_tolerance == pytest.approx(1e-7)


def test_degenerate_volume_is_raised(symbols: ModelSymbols) -> None:
    """A tiny compartment does not collapse the tolerance of its species."""
    by_id = {
        t.sid: t
        for t in state_tolerances(STATES, symbols, VOLUMES, AbsoluteTolerance())
    }
    assert by_id["S"].volume_raised
    assert by_id["S"].volume == pytest.approx(VOLUME_FLOOR * 2.0)
    assert by_id["S"].absolute_tolerance == pytest.approx(1e-10 * VOLUME_FLOOR * 2.0)
    assert not by_id["A"].volume_raised


def test_no_finite_volume(symbols: ModelSymbols) -> None:
    """Without a finite positive volume the reference volume is the floor of 1."""
    volumes = {"C": math.nan, "U": 0.0}
    by_id = {
        t.sid: t
        for t in state_tolerances(STATES, symbols, volumes, AbsoluteTolerance())
    }
    assert by_id["A"].volume == pytest.approx(VOLUME_FLOOR)
    assert by_id["A"].volume_raised


def test_override_by_id(symbols: ModelSymbols) -> None:
    """An override is the tolerance of the state, not multiplied by a volume."""
    tolerance = AbsoluteTolerance(ids={"A": 1e-14})
    by_id = {t.sid: t for t in state_tolerances(STATES, symbols, VOLUMES, tolerance)}
    assert by_id["A"].absolute_tolerance == pytest.approx(1e-14)


def test_override_of_no_state(symbols: ModelSymbols) -> None:
    """An override of an id which is not a state names the states."""
    with pytest.raises(ValueError, match="not a state.*'A'"):
        state_tolerances(STATES, symbols, VOLUMES, AbsoluteTolerance(ids={"Y": 1e-9}))


@pytest.mark.parametrize("value", [0.0, -1e-9, math.nan, math.inf])
def test_invalid_tolerance(value: float) -> None:
    """A tolerance is finite and positive."""
    with pytest.raises(ValueError, match="finite and positive"):
        AbsoluteTolerance(amount=value)
    with pytest.raises(ValueError, match="finite and positive"):
        AbsoluteTolerance(ids={"A": value})


def test_of_a_float() -> None:
    """A float is the same tolerance for every kind."""
    tolerance = AbsoluteTolerance.of(1e-6)
    assert tolerance == AbsoluteTolerance(amount=1e-6, concentration=1e-6, other=1e-6)
    assert AbsoluteTolerance.of(tolerance) is tolerance


def test_round_trip() -> None:
    """A tolerance is a dictionary and back, a float of stored settings reads."""
    tolerance = AbsoluteTolerance(amount=1e-9, ids={"b": 1e-12, "a": 1e-13})
    assert AbsoluteTolerance.from_dict(tolerance.to_dict()) == tolerance
    assert AbsoluteTolerance.from_dict(1e-6) == AbsoluteTolerance.of(1e-6)
    assert tolerance.overrides == {"a": 1e-13, "b": 1e-12}
    # the overrides are sorted, so equal tolerances are equal and hash alike
    assert hash(tolerance) == hash(
        AbsoluteTolerance(amount=1e-9, ids={"a": 1e-13, "b": 1e-12})
    )
    assert "2 by id" in str(tolerance)
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `.venv/bin/pytest -n 0 -q tests/model/test_tolerances.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbmlsim.model.tolerances'`

- [ ] **Step 4: Implement the module**

Create `src/sbmlsim/model/tolerances.py`:

```python
"""The absolute tolerances of the integrator, one per state of a model.

CVODE weighs the error of a state `x` with `1 / (rtol * |x| + atol)`. sbmlsim
sets `atol` of every state by its kind instead of handing roadrunner one value,
which roadrunner scales by the initial value or by the volume of a state:

| kind | states | absolute tolerance |
| --- | --- | --- |
| `AMOUNT` | species with `hasOnlySubstanceUnits=true` | `amount` |
| `CONCENTRATION` | other species, integrated as amounts | `concentration * V` |
| `OTHER` | every other state, i.e. the targets of rate rules | `other` |

The numbers are in the units of the model of the state, so they need no units.
`V` is the reference volume of the compartment: its initial volume, raised to
`VOLUME_FLOOR` times the largest finite positive initial volume of the model
when it is smaller, not finite or not positive, so that a degenerate
compartment does not collapse the tolerance of its species.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from sbmlsim.model.symbols import ModelSymbols

#: the smallest reference volume relative to the largest initial volume
VOLUME_FLOOR = 1e-6


class StateKind(StrEnum):
    """The kind of a state, which decides its absolute tolerance."""

    AMOUNT = "amount"
    CONCENTRATION = "concentration"
    OTHER = "other"


def _checked(name: str, value: float) -> float:
    """Get a tolerance as a float.

    Raises:
        ValueError: if the tolerance is not finite or not positive.
    """
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"The absolute tolerance of '{name}' must be finite and positive, "
            f"but is {value}."
        )
    return value


@dataclass(frozen=True)
class AbsoluteTolerance:
    """The absolute tolerance per kind of state, with overrides per id.

    Attributes:
        amount: tolerance of a species with `hasOnlySubstanceUnits=true`.
        concentration: tolerance of the concentration of a species, which is
            multiplied by the reference volume of its compartment.
        other: tolerance of every other state, i.e. the target of a rate rule.
        ids: tolerances of single states by id, in the unit of the model of the
            state; an override of a concentration species is an amount. Any
            mapping is normalized to a sorted tuple, so that tolerances are
            hashable and compare by value.
    """

    amount: float = 1e-10
    concentration: float = 1e-10
    other: float = 1e-10
    ids: Mapping[str, float] | tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        """Check the tolerances and normalize the overrides."""
        for kind in StateKind:
            object.__setattr__(
                self, kind.value, _checked(kind.value, getattr(self, kind.value))
            )
        overrides = tuple(
            sorted(
                (str(sid), _checked(str(sid), v)) for sid, v in dict(self.ids).items()
            )
        )
        object.__setattr__(self, "ids", overrides)

    @classmethod
    def of(cls, value: float | AbsoluteTolerance) -> AbsoluteTolerance:
        """Get the tolerance of a setting, a float is the same for every kind."""
        if isinstance(value, AbsoluteTolerance):
            return value
        value = float(value)
        return cls(amount=value, concentration=value, other=value)

    @property
    def overrides(self) -> dict[str, float]:
        """Get the tolerances of single states by id."""
        return dict(self.ids)

    def of_kind(self, kind: StateKind) -> float:
        """Get the tolerance of a kind of state."""
        return float(getattr(self, kind.value))

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary of JSON serializable values."""
        return {
            "amount": self.amount,
            "concentration": self.concentration,
            "other": self.other,
            "ids": self.overrides,
        }

    @classmethod
    def from_dict(cls, d: float | Mapping[str, Any]) -> AbsoluteTolerance:
        """Create a tolerance from `to_dict`, or from the float of stored settings."""
        if isinstance(d, Mapping):
            return cls(
                amount=d["amount"],
                concentration=d["concentration"],
                other=d["other"],
                ids=d.get("ids", {}),
            )
        return cls.of(d)

    def __str__(self) -> str:
        """Get the tolerances per kind and the number of overrides."""
        text = (
            f"amount {self.amount:.1e}, concentration {self.concentration:.1e}, "
            f"other {self.other:.1e}"
        )
        if self.ids:
            text += f", {len(self.ids)} by id"
        return text


@dataclass(frozen=True)
class StateTolerance:
    """The absolute tolerance of a state.

    Attributes:
        sid: id of the state.
        kind: kind of the state.
        compartment: compartment of a concentration species, else `None`.
        volume: reference volume of the compartment of a concentration
            species, else `None`.
        volume_raised: whether the initial volume was raised to the floor.
        absolute_tolerance: the tolerance in the unit of the model of the
            state, an amount for a species.
    """

    sid: str
    kind: StateKind
    compartment: str | None
    volume: float | None
    volume_raised: bool
    absolute_tolerance: float


def state_kinds(states: Iterable[str], symbols: ModelSymbols) -> dict[str, StateKind]:
    """Get the kind of every state.

    Args:
        states: ids of the states which the integrator integrates.
        symbols: symbols of the model.

    Returns:
        The kind by id.
    """
    kinds: dict[str, StateKind] = {}
    for sid in states:
        if sid not in symbols.species:
            kinds[sid] = StateKind.OTHER
        elif sid in symbols.only_substance:
            kinds[sid] = StateKind.AMOUNT
        else:
            kinds[sid] = StateKind.CONCENTRATION
    return kinds


def _reference_volumes(
    initial_volumes: Mapping[str, float],
) -> tuple[dict[str, float], set[str]]:
    """Get the reference volume of every compartment and the raised ones."""
    finite = [v for v in initial_volumes.values() if math.isfinite(v) and v > 0]
    floor = VOLUME_FLOOR * max(finite, default=1.0)
    volumes: dict[str, float] = {}
    raised: set[str] = set()
    for cid, volume in initial_volumes.items():
        if math.isfinite(volume) and volume >= floor:
            volumes[cid] = volume
        else:
            volumes[cid] = floor
            raised.add(cid)
    return volumes, raised


def state_tolerances(
    states: Sequence[str],
    symbols: ModelSymbols,
    initial_volumes: Mapping[str, float],
    tolerance: AbsoluteTolerance,
) -> list[StateTolerance]:
    """Get the absolute tolerance of every state, see the module.

    Args:
        states: ids of the states which the integrator integrates.
        symbols: symbols of the model.
        initial_volumes: initial volume of every compartment.
        tolerance: the tolerances per kind and by id.

    Returns:
        The tolerance of every state, in the order of `states`.

    Raises:
        ValueError: if an override is not a state.
    """
    overrides = tolerance.overrides
    unknown = sorted(set(overrides) - set(states))
    if unknown:
        raise ValueError(
            f"The absolute tolerances of {unknown} are overrides of ids which are "
            f"not a state, the states are {sorted(states)}."
        )
    volumes, raised = _reference_volumes(initial_volumes)
    result: list[StateTolerance] = []
    for sid, kind in state_kinds(states, symbols).items():
        compartment: str | None = None
        volume: float | None = None
        value = tolerance.of_kind(kind)
        if kind is StateKind.CONCENTRATION:
            compartment = symbols.species_compartment[sid]
            volume = volumes.get(compartment, VOLUME_FLOOR)
            value *= volume
        result.append(
            StateTolerance(
                sid=sid,
                kind=kind,
                compartment=compartment,
                volume=volume,
                volume_raised=compartment in raised,
                absolute_tolerance=overrides.get(sid, value),
            )
        )
    return result
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `.venv/bin/pytest -n 0 -q tests/model/test_tolerances.py`
Expected: all pass. Then `.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests && .venv/bin/ty check` with zero diagnostics.

- [ ] **Step 6: Commit**

```bash
git add src/sbmlsim/model/tolerances.py tests/model/test_tolerances.py tests/simulator/models.py
git commit -m "The absolute tolerance of a state follows from its kind"
```

---

### Task 2: The model sets the tolerances of its states

**Files:**
- Modify: `src/sbmlsim/model/model_roadrunner.py` (`__init__`, `set_integrator_settings`, remove `_tolerance_volume_factor` and `set_default_settings`, add `state_ids`, `tolerances`)
- Modify: `src/sbmlsim/simulator/simulation_serial.py:81-90` (`set_model`, `set_integrator_settings`)
- Modify: `src/sbmlsim/fit/optimization.py:999-1009`
- Test: `tests/simulator/test_simulator_serial.py`

**Interfaces:**
- Consumes: `AbsoluteTolerance`, `StateTolerance`, `state_tolerances` of Task 1.
- Produces:
  - `RoadrunnerSBMLModel.set_integrator_settings(self, **kwargs: float | int | bool | AbsoluteTolerance) -> roadrunner.Integrator` (instance method).
  - `RoadrunnerSBMLModel.state_ids(self) -> list[str]`.
  - `RoadrunnerSBMLModel.tolerances(self) -> pd.DataFrame` with columns `sid`, `kind`, `compartment`, `volume`, `absolute_tolerance`.
  - `RoadrunnerSBMLModel.absolute_tolerance: AbsoluteTolerance` (the tolerance set last, `AbsoluteTolerance()` after the load).

- [ ] **Step 1: Write the failing tests**

Replace the two tests `test_integrator_settings_are_passed_on_and_kept` and `test_an_unknown_integrator_setting_is_an_error` at the end of `tests/simulator/test_simulator_serial.py` with:

```python
def _vector(simulator: SimulatorSerial) -> list[float]:
    """Get the sorted absolute tolerances which CVODE uses."""
    integrator = simulator.r_loaded.getIntegrator()
    return sorted(float(v) for v in integrator.getAbsoluteToleranceVector())


def test_integrator_settings_are_passed_on_and_kept(tmp_path) -> None:
    """Every setting of the integrator reaches roadrunner, also for a later model."""
    path = tmp_path / "probe.xml"
    path.write_text(sbml())
    simulator = SimulatorSerial(model=path, initial_time_step=1e-9)
    simulator.set_integrator_settings(maximum_num_steps=1234)
    # a new model, as for every task of an experiment
    simulator.set_model(path)
    integrator = simulator.r_loaded.getIntegrator()
    assert integrator.getValue("initial_time_step") == pytest.approx(1e-9)
    assert integrator.getValue("maximum_num_steps") == 1234


def test_an_unknown_integrator_setting_is_an_error(simulator: SimulatorSerial) -> None:
    """A setting the integrator does not have is not dropped silently."""
    with pytest.raises(ValueError, match="has no settings"):
        simulator.set_integrator_settings(initial_step=1e-9)


def test_the_tolerance_of_every_state_reaches_cvode(tmp_path) -> None:
    """The tolerances per state are the vector of CVODE, after a new model too."""
    path = tmp_path / "tolerances.xml"
    path.write_text(sbml(TOLERANCE_PROBE))
    tolerance = AbsoluteTolerance(amount=1e-9, concentration=1e-8, other=1e-7)
    simulator = SimulatorSerial(model=path, absolute_tolerance=tolerance)
    # A and S are concentration species in C = 2 and U = 1e-12 (raised to
    # 2e-6), X an amount species, D a parameter with a rate rule
    expected = sorted([1e-8 * 2, 1e-8 * 2e-6, 1e-9, 1e-7])
    assert _vector(simulator) == pytest.approx(expected)
    simulator.set_model(path)
    assert _vector(simulator) == pytest.approx(expected)
    table = simulator.model_loaded.tolerances()
    assert list(table["sid"]) == simulator.model_loaded.state_ids()
    assert set(table["kind"]) == {"amount", "concentration", "other"}


def test_a_float_tolerance_is_the_same_for_every_kind(tmp_path) -> None:
    """The scaling of roadrunner by the initial values is not used."""
    path = tmp_path / "tolerances.xml"
    path.write_text(sbml(TOLERANCE_PROBE))
    simulator = SimulatorSerial(model=path, absolute_tolerance=1e-10)
    expected = sorted([1e-10 * 2, 1e-10 * 2e-6, 1e-10, 1e-10])
    assert _vector(simulator) == pytest.approx(expected)


def test_a_degenerate_compartment_is_logged(tmp_path, caplog) -> None:
    """A compartment whose volume is raised to the floor is logged once per model."""
    path = tmp_path / "tolerances.xml"
    path.write_text(sbml(TOLERANCE_PROBE))
    with caplog.at_level("WARNING"):
        simulator = SimulatorSerial(model=path)
        simulator.set_integrator_settings(absolute_tolerance=1e-9)
    messages = [r.getMessage() for r in caplog.records if "'U'" in r.getMessage()]
    assert len(messages) == 1
```

Change the imports at the top of the file to:

```python
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.result import TimecourseResult
from sbmlsim.simulation import Change, Dimension, ScanSim, Simulation
from sbmlsim.simulator import SimulatorSerial
from tests.simulator.models import TOLERANCE_PROBE, sbml
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/pytest -n 0 -q tests/simulator/test_simulator_serial.py`
Expected: FAIL in `test_the_tolerance_of_every_state_reaches_cvode` (the vector of roadrunner's scaling differs) and in `test_a_degenerate_compartment_is_logged`.

- [ ] **Step 3: Implement**

In `src/sbmlsim/model/model_roadrunner.py`:

1. Import: `from sbmlsim.model.tolerances import AbsoluteTolerance, StateTolerance, state_tolerances`.
2. In `__init__`, replace

```python
        # set integrator settings
        # logger.info("set integrator settings")
        if settings:
            RoadrunnerSBMLModel.set_integrator_settings(self.r, **settings)
```

with

```python
        #: the absolute tolerance set last and the tolerances of the states,
        #: see `set_integrator_settings`
        self.absolute_tolerance: AbsoluteTolerance = AbsoluteTolerance()
        self._state_tolerances: list[StateTolerance] = []
        #: compartments whose volume was reported as raised to the floor
        self._raised_reported: set[str] = set()
        self.set_integrator_settings(
            **{"absolute_tolerance": AbsoluteTolerance(), **(settings or {})}
        )
```

3. Replace the static `set_integrator_settings`, `_tolerance_volume_factor` and `set_default_settings` with:

```python
def set_integrator_settings(
    self, **kwargs: float | int | bool | AbsoluteTolerance
) -> roadrunner.Integrator:
    """Set settings of the integrator.

    Every setting of the integrator of roadrunner is passed on, for CVODE
    e.g. `relative_tolerance`, `stiff`, `variable_step_size`,
    `initial_time_step`, `minimum_time_step`, `maximum_time_step` and
    `maximum_num_steps`. `absolute_tolerance`, a float or an
    `AbsoluteTolerance`, is set as one tolerance per state, see
    `sbmlsim.model.tolerances`.

    Args:
        **kwargs: the settings by their names in roadrunner.

    Returns:
        The integrator.

    Raises:
        ValueError: if the integrator has no setting of a name, or an
            override of the absolute tolerance is not a state.
    """
    integrator: roadrunner.Integrator = self.r_loaded.getIntegrator()
    names = set(integrator.getSettings())
    unknown = sorted(set(kwargs) - names)
    if unknown:
        raise ValueError(
            f"The integrator '{integrator.getName()}' has no settings "
            f"{unknown}, its settings are {sorted(names)}."
        )
    for key, value in kwargs.items():
        if key == "absolute_tolerance":
            if isinstance(value, bool):
                raise ValueError("The absolute tolerance is a number.")
            self._set_absolute_tolerance(AbsoluteTolerance.of(value))
        else:
            integrator.setValue(key, value)
            logger.debug("Integrator setting: '%s = %s'", key, value)
    return integrator


def state_ids(self) -> list[str]:
    """Get the ids of the states which the integrator integrates."""
    r = self.r_loaded
    n = len(r.getIntegrator().getAbsoluteToleranceVector())
    return [r.model.getStateVectorId(k) for k in range(n)]


def _set_absolute_tolerance(self, tolerance: AbsoluteTolerance) -> None:
    """Set the absolute tolerance of every state, see `tolerances`.

    roadrunner turns a single value into a vector by its own scaling, so
    the value of the kind `other` is set first and every state is set by
    its id afterwards; a later single value would replace the vector.
    """
    r = self.r_loaded
    volumes = dict(
        zip(
            r.model.getCompartmentIds(),
            (float(v) for v in r.model.getCompartmentInitVolumes()),
            strict=True,
        )
    )
    states = state_tolerances(self.state_ids(), self.symbols, volumes, tolerance)
    for state in states:
        if (
            state.volume_raised
            and state.compartment is not None
            and state.compartment not in self._raised_reported
        ):
            self._raised_reported.add(state.compartment)
            logger.warning(
                "The compartment '%s' of the model '%s' has the initial volume "
                "%s; the absolute tolerances of its species use the volume %s.",
                state.compartment,
                self.sid or r.model.getModelName(),
                volumes.get(state.compartment),
                state.volume,
            )
    integrator: roadrunner.Integrator = r.getIntegrator()
    integrator.setValue("absolute_tolerance", tolerance.other)
    for state in states:
        integrator.setIndividualTolerance(state.sid, state.absolute_tolerance)
    self.absolute_tolerance = tolerance
    self._state_tolerances = states


def tolerances(self) -> pd.DataFrame:
    """Get the absolute tolerance of every state.

    Returns:
        A row per state with `sid`, `kind`, `compartment`, `volume` (the
        reference volume of a concentration species) and
        `absolute_tolerance`.
    """
    return pd.DataFrame(
        [
            {
                "sid": s.sid,
                "kind": s.kind.value,
                "compartment": s.compartment,
                "volume": s.volume,
                "absolute_tolerance": s.absolute_tolerance,
            }
            for s in self._state_tolerances
        ],
        columns=["sid", "kind", "compartment", "volume", "absolute_tolerance"],
    )
```

4. In `src/sbmlsim/simulator/simulation_serial.py`, `set_model`: replace `self.set_integrator_settings(**self.integrator_settings)` by `self.model.set_integrator_settings(**self.integrator_settings)`; `set_integrator_settings`:

```python
def set_integrator_settings(
    self, **kwargs: float | int | bool | AbsoluteTolerance
) -> None:
    """Set settings of the integrator.

    See `RoadrunnerSBMLModel.set_integrator_settings`. The settings apply
    to the loaded model and to every model set later, e.g. the models of
    the tasks of an experiment.
    """
    if self.model is not None:
        self.model.set_integrator_settings(**kwargs)
    self.integrator_settings.update(kwargs)
```

with `from sbmlsim.model.tolerances import AbsoluteTolerance` and the annotation of `self.integrator_settings: dict[str, float | int | bool | AbsoluteTolerance]`.

5. In `src/sbmlsim/fit/optimization.py` replace

```python
            RoadrunnerSBMLModel.set_integrator_settings(
                model.r_loaded, **simulator.integrator_settings
            )
```

with `model.set_integrator_settings(**simulator.integrator_settings)`.

6. `rg -n "set_integrator_settings\(|_tolerance_volume_factor|set_default_settings" src tests examples docs` and fix every remaining call of the static form.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest -n 0 -q tests/simulator tests/model tests/fit/test_optimization.py tests/fit/test_fit.py`
Expected: pass. Then `.venv/bin/pytest -q` (the whole suite, in parallel) and the lint and type commands of Task 1; a test which compared values to roadrunner's default tolerances may need `pytest.approx` and is fixed in place.

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "The model sets the absolute tolerance of every state"
```

---

### Task 3: The settings of a fit carry the tolerance per kind

**Files:**
- Modify: `src/sbmlsim/fit/options.py` (`FitSettings`)
- Modify: `src/sbmlsim/fit/display.py:205-225` (`settings_table`)
- Modify: `src/sbmlsim/fit/report.py:876-879` and `src/sbmlsim/resources/templates/fit_report.html` (the tolerances of the states)
- Test: `tests/fit/test_fit.py`, `tests/fit/test_display.py`, `tests/fit/test_report.py`, `tests/fit/test_petab_v2.py`

**Interfaces:**
- Consumes: `AbsoluteTolerance` (Task 1), `RoadrunnerSBMLModel.tolerances()` (Task 2).
- Produces: `FitSettings.absolute_tolerance: float | AbsoluteTolerance`, an `AbsoluteTolerance` after `__post_init__`; `FitSettings.to_dict()["absolute_tolerance"]` is `AbsoluteTolerance.to_dict()`.

- [ ] **Step 1: Write the failing tests**

In `tests/fit/test_fit.py` append:

```python
def test_settings_tolerance_per_kind() -> None:
    """The absolute tolerance of the settings is one per kind of state."""
    tolerance = AbsoluteTolerance(amount=1e-9, concentration=1e-8, other=1e-7)
    settings = FitSettings(absolute_tolerance=tolerance)
    assert FitSettings.from_dict(settings.to_dict()) == settings
    # a float is the same for every kind and equals its normalized form
    assert FitSettings(absolute_tolerance=1e-6) == FitSettings(
        absolute_tolerance=AbsoluteTolerance.of(1e-6)
    )


def test_stored_settings_with_a_float_tolerance() -> None:
    """Stored settings of sbmlsim 0.8.4 have a float absolute tolerance."""
    stored = FitSettings().to_dict()
    stored["absolute_tolerance"] = 1e-7
    settings = FitSettings.from_dict(stored)
    assert settings.absolute_tolerance == AbsoluteTolerance.of(1e-7)
```

with `from sbmlsim.model.tolerances import AbsoluteTolerance` added to the imports.

In `tests/fit/test_display.py`, in `test_settings_table`, replace the token `"1.0e-06"` check by `"concentration 1.0e-06"`.

In `tests/fit/test_report.py` append (the fixtures `op_hctz_pk` and `fit_settings` come from `tests/fit/conftest.py`, as in `test_a_metric_which_is_not_defined_is_a_dash`):

```python
def test_the_report_lists_the_tolerances_of_the_states(
    tmp_path: Path, op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The settings section lists the absolute tolerance of every state."""
    op_hctz_pk.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=op_hctz_pk.parameter_set_model(),
        mapping_figures=False,
    )
    context = report.html_context(tmp_path, "report")
    rows = context["tolerances"]
    assert rows
    assert {"model", "time", "sid", "kind", "volume", "absolute tolerance"} == set(
        rows[0]
    )
    assert "concentration" in context["settings"]["absolute tolerance"]
```

In `tests/fit/test_petab_v2.py`, next to the assertion `FitSettings.from_dict(extension.settings) == fit_settings_module`, nothing changes; it covers the round trip of the extension with the new dictionary. Add the old form:

```python
def test_extension_of_0_8_4_reads() -> None:
    """The settings of a problem exported by 0.8.4 carry a float tolerance."""
    settings = FitSettings().to_dict()
    settings["absolute_tolerance"] = 1e-6
    assert FitSettings.from_dict(settings).absolute_tolerance == AbsoluteTolerance.of(
        1e-6
    )
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/pytest -n 0 -q tests/fit/test_fit.py tests/fit/test_display.py tests/fit/test_report.py tests/fit/test_petab_v2.py`
Expected: FAIL (`FitSettings` has a float).

- [ ] **Step 3: Implement**

In `src/sbmlsim/fit/options.py`:

```python
    absolute_tolerance: float | AbsoluteTolerance = 1e-6
```

in `__post_init__` add

```python
        object.__setattr__(
            self, "absolute_tolerance", AbsoluteTolerance.of(self.absolute_tolerance)
        )
```

docstring: `absolute_tolerance: absolute tolerance of the simulator, one per kind of state, see sbmlsim.model.tolerances; a float is the same for every kind.`; `to_dict`: `"absolute_tolerance": AbsoluteTolerance.of(self.absolute_tolerance).to_dict()`; `from_dict`: `absolute_tolerance=AbsoluteTolerance.from_dict(d.get("absolute_tolerance", 1e-6))`.

In `src/sbmlsim/fit/display.py`, the row becomes `("absolute tolerance", str(AbsoluteTolerance.of(settings.absolute_tolerance)))`.

In `src/sbmlsim/fit/report.py`, the settings of the context:

```python
            "settings": {
                key.replace("_", " "): (
                    str(AbsoluteTolerance.of(self.settings.absolute_tolerance))
                    if key == "absolute_tolerance"
                    else value
                )
                for key, value in self.settings.to_dict().items()
            },
            "tolerances": self._tolerances(),
```

and the method

```python
    def _tolerances(self) -> list[dict[str, str]]:
        """Get the absolute tolerance of every state of the models of the fit."""
        rows: list[dict[str, str]] = []
        for model in {id(m): m for m in self.problem.models}.values():
            local = "local" if not model.symbols.time_dependent else "absolute"
            for _, row in model.tolerances().iterrows():
                rows.append(
                    {
                        "model": model.sid or "",
                        "time": local,
                        "sid": str(row["sid"]),
                        "kind": str(row["kind"]),
                        "volume": "" if row["volume"] is None else f"{row['volume']:.3g}",
                        "absolute tolerance": f"{row['absolute_tolerance']:.2e}",
                    }
                )
        return rows
```

(`symbols.time_dependent` is added in Task 4; until then use `getattr(model.symbols, "time_dependent", False)` and replace it in Task 4.) Read how `self.problem` is named in `FitReport` before writing (`rg -n "self.problem" src/sbmlsim/fit/report.py`).

In `fit_report.html`, below the table of the settings (`{% for key, value in settings.items() %}`), add a collapsed table with the class the other tables of the template use:

```html
<details>
  <summary>Absolute tolerances of the states</summary>
  <table class="sortable">
    <thead><tr><th>model</th><th>time</th><th>state</th><th>kind</th><th>reference volume</th><th>absolute tolerance</th></tr></thead>
    <tbody>
      {% for row in tolerances %}
      <tr><td>{{ row.model }}</td><td>{{ row.time }}</td><td>{{ row.sid }}</td><td>{{ row.kind }}</td><td>{{ row.volume }}</td><td>{{ row["absolute tolerance"] }}</td></tr>
      {% endfor %}
    </tbody>
  </table>
</details>
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest -n 0 -q tests/fit/test_fit.py tests/fit/test_display.py tests/fit/test_report.py tests/fit/test_petab_v2.py`, then `.venv/bin/pytest -q tests/fit`, lint and types.
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "The settings of a fit carry the absolute tolerance per kind of state"
```

---

### Task 4: Restarts in local time

**Files:**
- Modify: `src/sbmlsim/model/symbols.py` (`ModelSymbols.time_dependent`, `_reads_time`)
- Modify: `src/sbmlsim/simulator/executor.py` (`execute`, `_simulate`)
- Modify: `src/sbmlsim/fit/report.py` (replace the `getattr` of Task 3)
- Test: `tests/model/test_time_dependent.py`, `tests/simulator/test_local_time.py`

**Interfaces:**
- Consumes: `ModelSymbols`, `execute(plan, model, selections)`, `compile_simulation(simulation, symbols, uinfo)`.
- Produces: `ModelSymbols.time_dependent: bool` (default `False`).

- [ ] **Step 1: Write the failing tests of the detection**

Create `tests/model/test_time_dependent.py`:

```python
"""A model is time dependent if its math reads the time."""

import pytest

from sbmlsim.model.symbols import ModelSymbols
from tests.simulator.models import PROBE, sbml

CASES = {
    "rule": "model m\n  A = 1; k := 1 + time\n  J: A -> ; k*A\nend",
    "kinetic law": "model m\n  A = 1\n  J: A -> ; time*A\nend",
    "event trigger": "model m\n  A = 1\n  E: at time > 2: A = 5\nend",
    "event assignment": "model m\n  A = 1\n  E: at A < 0.5: A = time\n  J: A -> ; A\nend",
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_math_which_reads_the_time(case: str) -> None:
    """Every place a model reads the time makes it time dependent."""
    assert ModelSymbols.from_sbml(sbml(CASES[case])).time_dependent


def test_the_probe_is_not_time_dependent() -> None:
    """A model without the time is integrated in local time."""
    assert not ModelSymbols.from_sbml(sbml(PROBE)).time_dependent


def test_a_parameter_named_time_is_not_the_time() -> None:
    """The csymbol decides, not the identifier (case 01820)."""
    import libsbml

    doc = libsbml.readSBMLFromString(sbml(PROBE))
    model = doc.getModel()
    p = model.createParameter()
    p.setId("time_")
    p.setConstant(True)
    p.setValue(1.0)
    assert not ModelSymbols.from_sbml(libsbml.writeSBMLToString(doc)).time_dependent
```

SBML does not allow the time in a function definition (it must be an argument), so function definitions are not a case; the detection reads them anyway.

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/pytest -n 0 -q tests/model/test_time_dependent.py`
Expected: FAIL with `AttributeError: 'ModelSymbols' object has no attribute 'time_dependent'`

- [ ] **Step 3: Implement the detection**

In `src/sbmlsim/model/symbols.py` add the attribute to the docstring (`time_dependent: whether a math of the model reads the csymbol time or delay, see sbmlsim.simulator.executor`), the field `time_dependent: bool = False` after `events`, and the function

```python
def _reads_time(math: libsbml.ASTNode | None) -> bool:
    """Get whether a math reads the time or a delay (the csymbols, not an id)."""
    if math is None:
        return False
    if math.getType() in (libsbml.AST_NAME_TIME, libsbml.AST_FUNCTION_DELAY):
        return True
    return any(_reads_time(math.getChild(k)) for k in range(math.getNumChildren()))
```

and in `from_sbml`, before `return cls(`:

```python
        maths: list[libsbml.ASTNode | None] = [r.getMath() for r in rules]
        maths += [
            reaction.getKineticLaw().getMath()
            for reaction in model.getListOfReactions()
            if reaction.isSetKineticLaw()
        ]
        maths += [a.getMath() for a in model.getListOfInitialAssignments()]
        # SBML does not allow the time in a function definition, a model may
        # still have one
        maths += [f.getMath() for f in model.getListOfFunctionDefinitions()]
        for event in model.getListOfEvents():
            maths.append(event.getTrigger().getMath() if event.isSetTrigger() else None)
            maths.append(event.getDelay().getMath() if event.isSetDelay() else None)
            maths.append(event.getPriority().getMath() if event.isSetPriority() else None)
            maths += [a.getMath() for a in event.getListOfEventAssignments()]
```

and pass `time_dependent=any(_reads_time(m) for m in maths)` to `cls(...)`.

- [ ] **Step 4: Run the detection tests**

Run: `.venv/bin/pytest -n 0 -q tests/model/test_time_dependent.py tests/model/test_symbols.py`
Expected: pass.

- [ ] **Step 5: Write the failing tests of the local time**

Create `tests/simulator/test_local_time.py`:

```python
"""A model which does not read the time is integrated in local time."""

from dataclasses import replace

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import Change, Simulation
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.simulator.executor import execute
from sbmlsim.simulator.plan import compile_simulation
from tests.simulator.models import sbml

SEL = ["time", "[A]", "[B]", "X", "C", "k1"]

EVENT = """
model events
  compartment C = 1;
  species A in C; species B in C;
  A = 1; B = 0; k = 0.5
  J: A -> B; k*A
  E: at A < 0.2: A = 1
end
"""

TIME_RULE = """
model clock
  compartment C = 1;
  species A in C;
  A = 1; k := 1 + time
  J: A -> ; 0*A
end
"""


def _run(model: RoadrunnerSBMLModel, sim: Simulation, local: bool, sel: list[str]):
    """Run a plan in local or in absolute time."""
    model.symbols = replace(model.symbols, time_dependent=not local)
    plan = compile_simulation(sim, model.symbols, model.uinfo)
    return execute(plan, model, sel)


@pytest.mark.parametrize(
    "sim",
    [
        Simulation(end=30, changes=[Change([10, 20], {"[A]": 2.0})], steps=30),
        Simulation(start=-20, end=10, changes=[Change(-10, {"X": 0.0})], steps=30),
        Simulation(end=30, changes=[Change(10, {"[A]": "[A] + 1"})]),
    ],
)
def test_local_time_equals_absolute_time(sim: Simulation) -> None:
    """The result of the local time is the one of the absolute time."""
    local = _run(RoadrunnerSBMLModel(source=sbml()), sim, True, SEL)
    absolute = _run(RoadrunnerSBMLModel(source=sbml()), sim, False, SEL)
    if sim.steps is not None:
        np.testing.assert_array_equal(local["time"], absolute["time"])
        np.testing.assert_allclose(local.values, absolute.values, rtol=1e-6, atol=1e-9)
    else:
        assert local["time"][0] == pytest.approx(absolute["time"][0])
        assert local["time"][-1] == pytest.approx(absolute["time"][-1])
        assert local["[A]"][-1] == pytest.approx(absolute["[A]"][-1], rel=1e-6)


def test_the_output_times_are_the_absolute_times() -> None:
    """Exact output times come back as they were asked for."""
    times = [0.0, 9.999, 10.0, 10.5, 25.0]
    sim = Simulation(end=25, changes=[Change(10, {"[A]": 2.0})], times=times)
    res = _run(RoadrunnerSBMLModel(source=sbml()), sim, True, SEL)
    assert list(res["time"]) == times
    assert res["[A]"][2] == pytest.approx(2.0)


def test_a_change_formula_reads_the_absolute_time() -> None:
    """`time` in the formula of a change is the time of the change."""
    sim = Simulation(end=20, changes=[Change(10, {"[A]": "time"})], times=[10, 20])
    res = _run(RoadrunnerSBMLModel(source=sbml()), sim, True, SEL)
    assert res["[A]"][0] == pytest.approx(10.0)


def test_an_event_fires_in_local_time() -> None:
    """An event which does not read the time fires as in absolute time."""
    sim = Simulation(end=40, changes=[Change(20, {"[A]": 0.1})], steps=400)
    sel = ["time", "[A]", "[B]"]
    local = _run(RoadrunnerSBMLModel(source=sbml(EVENT)), sim, True, sel)
    absolute = _run(RoadrunnerSBMLModel(source=sbml(EVENT)), sim, False, sel)
    np.testing.assert_allclose(local["[A]"], absolute["[A]"], rtol=1e-5, atol=1e-8)
    # the change to 0.1 made the trigger true: A is reset to 1 at the change
    assert local["[A]"][200] == pytest.approx(1.0)


def test_a_model_which_reads_the_time_keeps_the_absolute_time() -> None:
    """`k := 1 + time` is the absolute time after a change."""
    model = RoadrunnerSBMLModel(source=sbml(TIME_RULE))
    assert model.symbols.time_dependent
    plan = compile_simulation(
        Simulation(end=30, changes=[Change(10, {"[A]": 2.0})], times=[0, 15, 30]),
        model.symbols,
        model.uinfo,
    )
    res = execute(plan, model, ["time", "k"])
    np.testing.assert_allclose(res["k"], [1.0, 16.0, 31.0])


def test_no_warning_after_a_late_dose(capfd) -> None:
    """A dose into an empty state at a late time does not make CVODE warn."""
    from examples.hctz_fitting import MODEL_PATH

    simulator = SimulatorSerial(model=MODEL_PATH)
    simulator.run_simulation(
        Simulation(
            time_unit="hr",
            start=-24,
            end=72,
            steps=96,
            changes=[
                Change([-24, 0, 24, 48], {"PODOSE_hctz": Q(25, "mg")}),
                Change([24, 48], {"Aurine_hctz": Q(0, "mmole")}),
            ],
        )
    )
    assert "t + h = t" not in capfd.readouterr().err
```

- [ ] **Step 6: Run them to see them fail**

Run: `.venv/bin/pytest -n 0 -q tests/simulator/test_local_time.py`
Expected: the equality tests pass already (absolute time on both sides), `test_no_warning_after_a_late_dose` FAILS with the CVODE warning in stderr.

- [ ] **Step 7: Implement the local time**

In `src/sbmlsim/simulator/executor.py`:

1. Module docstring, append an item: `6. A model which does not read the time (ModelSymbols.time_dependent) is integrated in local time: every segment starts at the time 0 of roadrunner and its output is shifted back, so the first step of CVODE after a change is never below the resolution of the time.`
2. In `execute`, compute `local = not model.symbols.time_dependent`; the warning about events and a start which is not 0 is logged only `if not local and plan.start != 0.0 and ...`; call `_simulate(plan, model, r, columns, local)` and, for `plan.steady_state_output`, keep `start=plan.end` in absolute time and use `start=0.0` in local time.
3. `_simulate(plan, model, r, columns, local: bool)`; inside the loop over the segments:

```python
a, b = points[k], points[k + 1]
last = k == len(points) - 2
if a in events:
    if model_events and (a > plan.start or plan.steady_state is not None):
        # the triggers at the end of the integration before the change,
        # the one of the steady state for the change at the start
        triggers = _triggers(model_events, r, a if local else float(r.model.getTime()))
        _apply(events[a], r, plan)
        _fire_events(model_events, triggers, r, a, model)
    else:
        # at the start after a reset roadrunner evaluates the triggers
        # itself
        _apply(events[a], r, plan)

# the segment in the time of roadrunner: from 0 in local time
offset = a if local else 0.0
start, end = a - offset, b - offset
# an event of the model at the time of the next change fires after
# the change (PEtab v2, reinitialization): the integration stops just
# before it, where roadrunner does not fire it, see `_fire_events`
end_stop = float(np.nextafter(end, -np.inf)) if model_events and not last else end
if plan.output is OutputMode.INTEGRATOR:
    integrator.setValue(VARIABLE_STEP_SIZE, True)
    block = np.array(r.simulate(start, end_stop), dtype=float)
    block[:, 0] += offset
    if not last:
        # the state at `b` is the one before the change at `b`
        block = block[:-1]
else:
    integrator.setValue(VARIABLE_STEP_SIZE, False)
    wanted = times[(times >= a) & ((times <= b) if last else (times < b))]
    shifted = wanted - offset
    grid = np.unique(np.concatenate([[start], shifted, [end_stop]]))
    result = np.array(r.simulate(times=grid.tolist()), dtype=float)
    block = result[np.isin(grid, shifted)]
    # the output times are the ones asked for, not the shifted ones
    block[:, 0] = wanted
blocks.append(block)
```

and for the change at the end: `triggers = _triggers(model_events, r, plan.end if local else float(r.model.getTime()))`.

`block[:, 0] = wanted` assumes that `shifted` has no two values which round to the same float; add `if block.shape[0] != wanted.size: raise RuntimeError(...)` with a message naming the segment, which a test never reaches but makes a silent misalignment impossible.

4. In `src/sbmlsim/fit/report.py` replace `getattr(model.symbols, "time_dependent", False)` with `model.symbols.time_dependent`.

- [ ] **Step 8: Run the tests**

Run: `.venv/bin/pytest -n 0 -q tests/simulator tests/model`, then `.venv/bin/pytest -q`, lint and types.
Expected: pass; the warning test passes.

- [ ] **Step 9: Commit**

```bash
git add src tests
git commit -m "A model which does not read the time is integrated in local time"
```

---

### Task 5: The test suites, the documentation and the amendments of the spec

**Files:**
- Modify: `docs/simulation.md`, `docs/models.md`, `docs/fitting.md`, `CLAUDE.md`, `docs/superpowers/specs/2026-10-08-integrator-tolerances-design.md`
- Possibly modify: `tests/data/testsuite_baseline.json`, `tests/data/petab_baseline.json`, `tests/data/benchmark_baseline.json` (only with the evidence of Step 1)

- [ ] **Step 1: Run the suites**

Run, one after the other, and keep the summaries:

```bash
.venv/bin/pytest -q
.venv/bin/pytest -q -m testsuite tests/testsuite        # SBML Test Suite, cached under ~/.cache/sbmlsim/test-suite
tox r -e petab                                          # PEtab v2 test suite, downloads the cases (or: pytest -m petab_testsuite tests/fit with SBMLSIM_PETAB_SUITE_PATH)
tox r -e benchmark                                      # benchmark collection, on demand; takes long
```

Expected: no regression. A case of the SBML Test Suite which changes its status is a finding: read the case (its tolerances, its model), decide whether the new tolerances are the cause (e.g. a model with a tiny compartment) and report it; do not edit a baseline without that analysis. If `tox r -e benchmark` cannot run (network, time), say so in the report of the task.

- [ ] **Step 2: Documentation**

- `docs/simulation.md`, section "Selections and integrator settings": replace the paragraph which introduces `initial_time_step` against the warning by: the absolute tolerance is a float or an `AbsoluteTolerance` (kinds, reference volume, overrides, an example), every model which does not read the time is integrated in local time, and `initial_time_step` remains the remedy of a model which reads the time. Keep the code examples runnable (`tests/docs` runs them).
- `docs/models.md:55`: replace the sentence on the scaling by the smallest volume by the tolerance per state and `RoadrunnerSBMLModel.tolerances()`.
- `docs/fitting.md`: `FitSettings.absolute_tolerance` takes an `AbsoluteTolerance`; the report lists the tolerances of the states.
- `CLAUDE.md`, the architecture paragraph of `model/`: `set_integrator_settings` is a method of the model, the tolerances per state (`model/tolerances.py`), the local time of the executor.
- The spec: section B, the states are the state ids of roadrunner (`RoadrunnerSBMLModel.state_ids`), their kinds come from the symbols, so `ModelSymbols` gets no `boundary`; there is no model which is derived with a new roadrunner instance, the tolerances are set at the load and with every `set_integrator_settings`; section D, the console of a fit shows the tolerance per kind and the report the tolerance of every state; `set_default_settings`, which nothing called, is removed.

- [ ] **Step 3: Check and commit**

Run: `.venv/bin/pytest -q tests/docs`, `.venv/bin/ruff format --check .`, `.venv/bin/ruff check`, `.venv/bin/ty check`.

```bash
git add docs CLAUDE.md
git commit -m "The documentation of the tolerances per state and of the local time"
```

---

### Task 6: pkdb_models uses the tolerances of sbmlsim

**Files (repository `/home/mkoenig/git/pkdb_models`, branch `sbmlsim-0.8.4`):**
- Modify: `pkdb_models/models/hctz/fitting/fitting.py` (remove `initial_time_step=1e-10` and its comment)
- Modify: `pkdb_models/models/hctz/helpers.py` (remove `initial_time_step=1e-10` and its comment)

Environment: `S=/tmp/claude-1000/-home-mkoenig-git-pkdb-models/5185c4a8-8859-43e8-8bf0-d39f6928aa59/scratchpad`, `export LD_LIBRARY_PATH=/home/mkoenig/.local/share/uv/python/cpython-3.14.4-linux-x86_64-gnu/lib MPLBACKEND=Agg`, the branch of sbmlsim on the path with `PYTHONPATH=/home/mkoenig/git/sbmlsim-fix-084/src`, python `$S/venv084/bin/python`.

- [ ] **Step 1: Remove the workaround**

Delete the lines `initial_time_step=1e-10,` and the comments above them in both files; `ruff format` and `ruff check` on `pkdb_models/models/hctz`.

- [ ] **Step 2: Verify**

```bash
for m in hctz albuterol lenvatinib rapamycin; do PYTHONPATH=/home/mkoenig/git/sbmlsim-fix-084/src $S/venv084/bin/python $S/dump.py $m $S/tol_$m.pkl > $S/tol_$m.log 2>&1; echo "$m warn=$(grep -c 't + h = t' $S/tol_$m.log) err=$(grep -c '^ERROR' $S/tol_$m.log)"; done
for m in hctz albuterol lenvatinib rapamycin; do $S/venv084/bin/python $S/hctz_port/compare_common.py $S/base_$m.pkl $S/tol_$m.pkl $(grep ': ok' $S/tol_$m.log | sed 's/: ok//' | tr '\n' ' ') | sort -k2 -g -r | head -5; done
```

Expected: 0 warnings and 0 errors for every model; the differences against the baseline of sbmlsim 0.8.3 on the shared times are the known ones (hctz: the urine volume of 1e-6 l and the cardiac impairments) and below 1e-4 otherwise. Then the fits: `fit_hctz` with `--subset=PK` and `--subset=PD`, `--runs=3 --cores=1 --seed=1234 --method=LSQ --strategy=ALL --no-identifiability`, 0 warnings and a cost within 1 % of 150.771 (PK) and 91.991 (PD); the end to end run `run_hctz --action simulate --experiments all` into a directory of the scratchpad with 0 warnings; `ty check --extra-search-path /home/mkoenig/git/sbmlsim-fix-084/src`; `pytest -q`.

- [ ] **Step 3: Commit**

```bash
git add pkdb_models/models/hctz
git commit -m "hctz no longer sets the first step of the integrator"
```

with a body which says that sbmlsim sets the tolerance of every state and integrates in local time, the numbers of Step 2, and that the floor of pkdb_models stays `sbmlsim>=0.8.5`.
