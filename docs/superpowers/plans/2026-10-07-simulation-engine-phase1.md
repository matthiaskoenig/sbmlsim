# Simulation engine, phase 1: the engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `Timecourse`/`TimecourseSim` by `Simulation`/`Change`/`SteadyState`, compiled into a unit free `Plan` and run by an executor with the initialization and change semantics of PEtab v2, and move the simulation experiments, scans, fits, the PEtab layer, the SBML Test Suite runner, the examples and the tests onto it.

**Architecture:** `sbmlsim.simulation.definition` holds the definition with units. `sbmlsim.simulator.plan` compiles a definition against a loaded model into a frozen, picklable `Plan` of floats and formula strings. `sbmlsim.simulator.executor` runs a `Plan` on roadrunner; `SimulatorSerial` loads models and runs simulations and scans with it. Results are ragged (`_point`), `XResult.interpolate` puts them on a common grid.

**Tech Stack:** python 3.13, libroadrunner 2.10, python-libsbml, pint, numpy, sympy (`petab.v2.math.sympify_petab`), xarray, pytest, ruff, ty.

**Spec:** `docs/superpowers/specs/2026-10-07-simulation-engine-design.md`

## Global Constraints

- `Timecourse`, `TimecourseSim`, `AbstractSim`, `time_offset`, `Timecourse.model_changes`, `TimecourseSim.selections`, `ScanSim.mapping` and `SimulationExperiment.Q_` are removed, no deprecation shims.
- Symbol convention everywhere: `S` is the amount of a species, `[S]` its concentration, as in roadrunner.
- Formulas are strings of the math of PEtab in the units of the model and carry no units.
- One unit registry: `sbmlsim.units.ureg`, `sbmlsim.Q = ureg.Quantity`. No id of a model unit definition is defined in the registry.
- No pint, `deepcopy`, `xarray` or `pandas` in `executor.py`.
- Every module, class and function has full annotations and a google docstring; ty with zero diagnostics; ruff clean; logging with `%s`, never f-strings in log calls.
- Markdown without hard wraps; never the em dash; no agent attribution in commits or pull requests.
- Commits are on the branch `simulation-engine`.

## Review Focus

- A `Change` at exactly `start`, and an output time equal to the time of a change: the output is the state after the change. Pinned in Task 6.
- `ScanSim` on a simulation whose dose is set in a `Change` with several times: the scan value replaces the dose at every time, it does not add a pre-initialization change. Pinned in Task 9.
- A fit parameter which feeds an initial assignment (the old silent bug): the initial value follows the parameter. Pinned in Task 6 and Task 12.
- Two models in one experiment which define the same unit id differently: each converts with its own definition. Pinned in Task 1.
- A simulation run twice on one roadrunner instance, the first with a pre-initialization change of `k`, the second without: the second uses the value of the model. Pinned in Task 6.

---

### Task 1: One unit registry, units of models as expressions

**Files:**
- Modify: `src/sbmlsim/units.py`
- Modify: `src/sbmlsim/__init__.py`
- Modify: every caller of `UnitsInformation._default_ureg()` and `UnitRegistry(` in `src/` (`experiment/runner.py`, `experiment/experiment.py`, `model/model_roadrunner.py`, `fit/petab_v2/reader.py`, `simulation/scan.py`)
- Test: `tests/test_units.py` (extend the existing file if present, otherwise create)

**Interfaces:**
- Produces: `sbmlsim.units.ureg: pint.UnitRegistry`, `sbmlsim.units.Q`, `sbmlsim.Q`; `UnitsInformation.from_sbml(sbml, ureg=None)` with `ureg` defaulting to `sbmlsim.units.ureg`; `UnitsInformation.model_uid_dict(model, ureg)` returns expressions and defines nothing in `ureg`.

- [ ] **Step 1: Write the failing tests**

```python
import libsbml
import pytest

from sbmlsim import Q
from sbmlsim.units import UnitsInformation, ureg


def _model_with_unit(uid: str, kind: int, exponent: int = 1, scale: int = 0) -> str:
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    model.setId(f"m_{uid}_{kind}")
    udef = model.createUnitDefinition()
    udef.setId(uid)
    unit = udef.createUnit()
    unit.setKind(kind)
    unit.setExponent(exponent)
    unit.setScale(scale)
    unit.setMultiplier(1.0)
    p = model.createParameter()
    p.setId("p")
    p.setValue(1.0)
    p.setConstant(True)
    p.setUnits(uid)
    return libsbml.writeSBMLToString(doc)


def test_q_is_quantity_of_package_registry() -> None:
    assert Q(1, "mg")._REGISTRY is ureg


def test_same_unit_id_two_models_converts_per_model() -> None:
    gram_model = _model_with_unit("u1", libsbml.UNIT_KIND_GRAM, scale=-3)  # mg
    mole_model = _model_with_unit("u1", libsbml.UNIT_KIND_MOLE, scale=-3)  # mmol
    u_gram = UnitsInformation.from_sbml(gram_model)
    u_mole = UnitsInformation.from_sbml(mole_model)
    assert Q(1, "g").to(u_gram["p"]).magnitude == pytest.approx(1000.0)
    assert Q(1, "mol").to(u_mole["p"]).magnitude == pytest.approx(1000.0)


def test_model_unit_ids_are_not_defined_in_registry() -> None:
    UnitsInformation.from_sbml(
        _model_with_unit("u_not_in_registry", libsbml.UNIT_KIND_GRAM)
    )
    with pytest.raises(Exception):
        ureg("u_not_in_registry")
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest -n 0 tests/test_units.py -q`
Expected: FAIL, `ImportError: cannot import name 'Q' from 'sbmlsim'`.

- [ ] **Step 3: Implement**

In `src/sbmlsim/units.py`, add the module registry after the imports and make `_default_ureg` return it:

```python
def _create_registry() -> UnitRegistry:
    """Create the registry of the package with the units sbmlsim defines."""
    registry = pint.UnitRegistry(on_redefinition="ignore")
    registry.define("none = count")
    registry.define("item = count")
    registry.define("percent = 0.01*count")
    registry.define("IU = 0.0347 * mg")
    registry.define("IU_per_ml = 0.0347 * mg/ml")
    return registry


#: the unit registry of the package, every model, experiment and fit uses it
ureg: UnitRegistry = _create_registry()

#: quantity of the registry of the package
Q = ureg.Quantity
```

Remove `UnitsInformation._default_ureg` and replace every call by `ureg`. In `model_uid_dict`, drop every `ureg.define(...)`: the predefined Level 2 units map to their expression (`uid_dict[uid] = unit_str`), and a unit definition maps to `Units.udef_to_str(udef)` unless pint already knows the id as the identical unit (keep the existing comparison, but compare `ureg(unit_str)` with `ureg(uid)` only inside `try`, and use `unit_str` otherwise). Every value of `uid_dict` is then parseable by `ureg`. In `from_sbml_doc`, every place that stored a uid as the unit of an entity stores `uid_dict[uid]` instead. In `src/sbmlsim/__init__.py` add `from sbmlsim.units import Q` (import placed after `__version__` so hatchling still reads the version) and `__all__ = ["Q"]`. Replace every `UnitRegistry(...)` in `src/` by the package `ureg`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -n 0 tests/test_units.py tests/model -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sbmlsim/units.py src/sbmlsim/__init__.py src/sbmlsim tests/test_units.py
git commit -m "One unit registry for the package, the units of a model are expressions"
```

### Task 2: The definition: `Simulation`, `Change`, `SteadyState`

**Files:**
- Create: `src/sbmlsim/simulation/definition.py`
- Modify: `src/sbmlsim/simulation/__init__.py`
- Delete: `src/sbmlsim/simulation/simulation.py`, `src/sbmlsim/task/task_new.py` (SED-ML leftovers without callers)
- Test: `tests/simulation/test_definition.py`

**Interfaces:**
- Produces:
  - `Change(times: float | Sequence[float] | Quantity, values: Mapping[str, float | Quantity | str])` with attributes `times: tuple[float | Quantity, ...]`, `values: dict[str, float | Quantity | str]`.
  - `SteadyState(preinit_changes: Mapping[str, float | Quantity] | None = None, absolute_tolerance: float = 1e-8, relative_tolerance: float = 1e-6, max_time: float = 1e8)`.
  - `Simulation(*, end, start=0.0, time_unit: str | None = None, preinit_changes=None, changes=None, presimulation: SteadyState | None = None, times=None, steps: int | None = None, time_shift=0.0)`, attributes of the same names; `Simulation.changes: list[Change]`; `Simulation.to_dict() -> dict[str, Any]`, `Simulation.from_dict(d) -> Simulation`, `Simulation.to_json(path=None)`, `Simulation.from_json(info)`; `Simulation.targets() -> set[str]` (every target of `preinit_changes` and of every `Change`); `Simulation.with_values(values: Mapping[str, float | Quantity]) -> Simulation` (the override semantics of the spec: replace wherever the target is set, else add to `preinit_changes`).

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from sbmlsim import Q
from sbmlsim.simulation import Change, Simulation, SteadyState


def test_change_scalar_and_vector_times() -> None:
    assert Change(10, {"k": 1.0}).times == (10,)
    assert Change([0, 24, 48], {"PODOSE": Q(10, "mg")}).times == (0, 24, 48)
    assert Change(Q([0, 1], "hr"), {"k": 1.0}).times == (Q(0, "hr"), Q(1, "hr"))


def test_change_requires_values() -> None:
    with pytest.raises(ValueError, match="no values"):
        Change(0, {})


def test_simulation_validates_interval_and_output() -> None:
    with pytest.raises(ValueError, match="start"):
        Simulation(start=10, end=5)
    with pytest.raises(ValueError, match="times.*steps"):
        Simulation(end=10, times=[0, 5], steps=10)
    with pytest.raises(ValueError, match="formula"):
        Simulation(end=10, preinit_changes={"k": "2 * a"})


def test_simulation_rejects_conflicting_presimulation_targets() -> None:
    with pytest.raises(ValueError, match="'k'"):
        Simulation(
            end=10,
            preinit_changes={"k": 1.0},
            presimulation=SteadyState(preinit_changes={"k": 2.0}),
        )


def test_with_values_replaces_everywhere_or_adds_preinit() -> None:
    sim = Simulation(
        end=72,
        preinit_changes={"BW": Q(70, "kg")},
        changes=[Change([0, 24, 48], {"PODOSE": Q(10, "mg")})],
    )
    scanned = sim.with_values({"PODOSE": Q(20, "mg"), "k": 2.0, "BW": Q(80, "kg")})
    assert scanned.changes[0].values["PODOSE"] == Q(20, "mg")
    assert scanned.preinit_changes == {"BW": Q(80, "kg"), "k": 2.0}
    assert sim.changes[0].values["PODOSE"] == Q(10, "mg")


def test_json_round_trip() -> None:
    sim = Simulation(
        time_unit="hr",
        start=-72,
        end=48,
        preinit_changes={"BW": Q(70, "kg")},
        changes=[
            Change([-72, 0], {"PODOSE": Q(10, "mg")}),
            Change(10, {"[glc]": "[glc] + 5"}),
        ],
        presimulation=SteadyState(preinit_changes={"ins": 0.0}),
        steps=100,
    )
    again = Simulation.from_json(sim.to_json())
    assert again.to_dict() == sim.to_dict()
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest -n 0 tests/simulation/test_definition.py -q`
Expected: FAIL, `ImportError: cannot import name 'Change'`.

- [ ] **Step 3: Implement `definition.py`**

```python
"""The definition of a simulation: what is simulated, with units.

A `Simulation` is the interval of a simulation, the changes applied to the
model before it is initialized (`preinit_changes`), the changes at times
(`changes`, a list of `Change`), an optional pre-equilibration
(`SteadyState`) and the output. It is compiled against a model into a
`sbmlsim.simulator.plan.Plan`, which is what the simulator runs; the
semantics are the ones of PEtab v2, see the design
`docs/superpowers/specs/2026-10-07-simulation-engine-design.md`.

A value is a `Quantity`, a number in the unit of its target in the model, or
a formula: a string of the math of PEtab over the symbols of the model, in the
units of the model, evaluated with the state at the time of the change. `S`
is the amount of a species and `[S]` its concentration.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sbmlsim.units import Quantity, ureg

Time = float | Quantity
Value = float | Quantity | str


def _times(times: float | Sequence[float] | Quantity) -> tuple[Time, ...]:
    """Get the times of a change as a tuple of scalars."""
    if isinstance(times, Quantity):
        magnitudes = np.atleast_1d(np.asarray(times.magnitude, dtype=float))
        return tuple(ureg.Quantity(float(m), times.units) for m in magnitudes)
    if isinstance(times, int | float | np.integer | np.floating):
        return (times,)
    return tuple(times)


@dataclass(frozen=True, init=False)
class Change:
    """Changes of the model at one or several times.

    A change at several times is the same change at each of them, e.g. the
    doses of a multiple dosing.

    Attributes:
        times: the times, numbers in the `time_unit` of the simulation or
            quantities.
        values: target -> value, see the module.
    """

    times: tuple[Time, ...]
    values: dict[str, Value]

    def __init__(
        self, times: float | Sequence[float] | Quantity, values: Mapping[str, Value]
    ) -> None:
        """Create a change, see the class."""
        if not values:
            raise ValueError(f"The change at {times} has no values.")
        resolved = _times(times)
        if not resolved:
            raise ValueError(f"The change of {sorted(values)} has no times.")
        object.__setattr__(self, "times", resolved)
        object.__setattr__(self, "values", dict(values))
```

Continue in the same file with `SteadyState` (a frozen dataclass with `preinit_changes: dict[str, float | Quantity] = field(default_factory=dict)` and the three tolerances of the interface, `__post_init__` refusing a formula value with `ValueError("... 'k' is a formula, pre-initialization changes are numbers or quantities")`), and `Simulation`:

```python
class Simulation:
    """A simulation, see the module."""

    def __init__(
        self,
        *,
        end: Time,
        start: Time = 0.0,
        time_unit: str | None = None,
        preinit_changes: Mapping[str, float | Quantity] | None = None,
        changes: Sequence[Change] | None = None,
        presimulation: SteadyState | None = None,
        times: Sequence[float] | Quantity | None = None,
        steps: int | None = None,
        time_shift: Time = 0.0,
    ) -> None:
        """Create a simulation, see the module and the design for the semantics.

        Raises:
            ValueError: if `start >= end`, if both `times` and `steps` are
                given, if `steps < 1`, if a pre-initialization change is a
                formula, or if a target is a pre-initialization change of the
                simulation and of the presimulation.
        """
```

Validation compares `start` and `end` after converting a quantity to `time_unit` (or to the unit of the other bound when `time_unit` is `None`), never against model units, which the definition does not know. `targets()`, `with_values()`, `to_dict()` and `from_dict()` serialize a quantity as `{"value": magnitude, "unit": str(units)}` and a formula as a string; `to_json`/`from_json` follow the existing `TimecourseSim.to_json` contract (a path writes and returns `None`, no path returns the string). `__repr__` lists start, end, the number of changes and the output mode.

In `src/sbmlsim/simulation/__init__.py` export `Change`, `Dimension`, `ScanSim`, `Simulation`, `SteadyState` and nothing else. Delete `simulation/simulation.py` and `task/task_new.py` (`rg -n "task_new|simulation.simulation" src tests examples` must be empty afterwards).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -n 0 tests/simulation/test_definition.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sbmlsim/simulation tests/simulation/test_definition.py src/sbmlsim/task
git commit -m "The definition of a simulation: Simulation, Change and SteadyState"
```

### Task 3: The symbols of a model and the formulas

**Files:**
- Create: `src/sbmlsim/simulator/symbols.py`
- Create: `src/sbmlsim/simulator/formula.py`
- Test: `tests/simulator/test_symbols.py`, `tests/simulator/test_formula.py`
- Create: `tests/simulator/models.py` (the probe models of the design, as antimony strings converted to SBML in a fixture)

**Interfaces:**
- Produces:
  - `TargetKind(StrEnum)`: `PARAMETER`, `COMPARTMENT`, `SPECIES_AMOUNT`, `SPECIES_CONCENTRATION`.
  - `ModelSymbols` (frozen dataclass) with `from_sbml(sbml: str | Path) -> ModelSymbols`, fields `parameters: frozenset[str]`, `compartments: frozenset[str]`, `species: frozenset[str]`, `species_compartment: dict[str, str]`, `only_substance: frozenset[str]`, `initial_assignments: frozenset[str]`, `assignment_rules: frozenset[str]`, `time_unit` is not here (units stay in `UnitsInformation`); methods `kind(target: str) -> TargetKind` (raises `ValueError` naming the target for an unknown id and for the target of an assignment rule), `entity(target: str) -> str` (`[S]` -> `S`).
  - `CompiledFormula` (frozen dataclass) with `formula: str`, `symbols: tuple[str, ...]` (roadrunner selections, e.g. `S`, `[S]`, `time`), `evaluate(values: Sequence[float]) -> float`; `compile_formula(formula: str) -> CompiledFormula` cached with `functools.lru_cache(maxsize=None)`.

- [ ] **Step 1: Write the probe models**

`tests/simulator/models.py`:

```python
"""Probe models of the simulation engine, see the design of the engine."""

import antimony

PROBE = """
model probe
  compartment C = 2;
  species A in C; species B in C;
  substanceOnly species X in C;
  a0 = 1; b0 = 1; k1 = 0.8; k2 = 0.6; f = 2
  kk := 3*f
  pinit = 2*f
  A = a0; B = b0; X = 3*pinit
  J1: A -> B; k1*A
  J2: B -> A; k2*B
end
"""


def sbml(model: str = PROBE) -> str:
    """Get the SBML of an antimony model."""
    antimony.clearPreviousLoads()
    if antimony.loadAntimonyString(model) < 0:
        raise ValueError(antimony.getLastError())
    return antimony.getSBMLString(antimony.getMainModuleName())
```

Add `antimony` to the `dev` extra in `pyproject.toml` if `uv run python -c "import antimony"` fails (it is installed today through `sbmlutils`; declare it in `dev` regardless, the tests import it).

- [ ] **Step 2: Write the failing tests**

```python
import pytest

from sbmlsim.simulator.formula import compile_formula
from sbmlsim.simulator.symbols import ModelSymbols, TargetKind
from tests.simulator.models import sbml


def test_kinds_of_probe() -> None:
    symbols = ModelSymbols.from_sbml(sbml())
    assert symbols.kind("k1") is TargetKind.PARAMETER
    assert symbols.kind("C") is TargetKind.COMPARTMENT
    assert symbols.kind("A") is TargetKind.SPECIES_AMOUNT
    assert symbols.kind("[A]") is TargetKind.SPECIES_CONCENTRATION
    assert "X" in symbols.only_substance
    assert "pinit" in symbols.initial_assignments
    with pytest.raises(ValueError, match="'kk'.*assignment rule"):
        symbols.kind("kk")
    with pytest.raises(ValueError, match="'nope'"):
        symbols.kind("nope")


def test_formula_with_concentration_and_time() -> None:
    f = compile_formula("[A] + 2 * time + k1")
    assert f.symbols == ("[A]", "k1", "time")
    assert f.evaluate([1.0, 0.5, 3.0]) == pytest.approx(1.0 + 3.0 + 0.5)


def test_formula_petab_math() -> None:
    assert compile_formula("log(exp(2))").evaluate([]) == pytest.approx(2.0)
    assert compile_formula("piecewise(1, time > 5, 0)").evaluate([6.0]) == 1.0
```

- [ ] **Step 3: Run them to see them fail**

Run: `uv run pytest -n 0 tests/simulator -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'sbmlsim.simulator.symbols'`.

- [ ] **Step 4: Implement**

`symbols.py` reads the SBML with libsbml once (`libsbml.readSBMLFromString` for a string which starts with `<`, `readSBMLFromFile` otherwise), fills the sets from `getListOfParameters/Compartments/Species`, `getHasOnlySubstanceUnits()`, `getListOfInitialAssignments()` (`getSymbol()`) and the `AssignmentRule`s of `getListOfRules()` (`getVariable()`). `kind` strips `[...]`: a bracketed id must be a species and is `SPECIES_CONCENTRATION`; an unbracketed species is `SPECIES_AMOUNT`.

`formula.py`:

```python
"""Formulas of the changes of a simulation.

A formula is a string of the math of PEtab whose symbols are selections of
roadrunner: `S` is the amount of a species, `[S]` its concentration and
`time` the time of the change. The brackets are not math, so `[S]` is replaced
by an identifier before the formula is parsed and mapped back afterwards.
"""

import functools
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import sympy as sp
from petab.v2.math import sympify_petab

_BRACKETS = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*)\]")
_PREFIX = "__concentration__"


@dataclass(frozen=True)
class CompiledFormula:
    """A formula compiled to a numpy function of its symbols.

    Attributes:
        formula: the formula as it was given.
        symbols: the selections the formula reads, sorted.
    """

    formula: str
    symbols: tuple[str, ...]
    _function: Callable[..., float] = field(repr=False, compare=False)

    def evaluate(self, values: Sequence[float]) -> float:
        """Evaluate the formula on the values of its symbols, in their order."""
        return float(self._function(*values))


@functools.lru_cache(maxsize=None)
def compile_formula(formula: str) -> CompiledFormula:
    """Compile a formula, see the module.

    Raises:
        ValueError: if the formula is not valid math of PEtab.
    """
    escaped = _BRACKETS.sub(lambda m: f"{_PREFIX}{m.group(1)}", formula)
    try:
        expression = sympify_petab(escaped)
    except Exception as err:
        raise ValueError(f"The formula '{formula}' is not valid math: {err}") from err
    ordered = sorted(expression.free_symbols, key=lambda s: _selection(str(s)))
    symbols = tuple(_selection(str(s)) for s in ordered)
    function = sp.lambdify(ordered, expression, modules="numpy")
    return CompiledFormula(formula=formula, symbols=symbols, _function=function)


def _selection(name: str) -> str:
    """Map an escaped symbol back to its selection."""
    if name.startswith(_PREFIX):
        return f"[{name.removeprefix(_PREFIX)}]"
    return name
```

`sympify_petab` returns `t` for the PEtab time symbol `time`; check with `sympify_petab("time").free_symbols` and map the PEtab time symbol to `time` in `_selection` if it is named differently.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -n 0 tests/simulator -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sbmlsim/simulator tests/simulator pyproject.toml uv.lock
git commit -m "The symbols of a model and the formulas of the changes of a simulation"
```

### Task 4: The plan: compiling a simulation against a model

**Files:**
- Create: `src/sbmlsim/simulator/plan.py`
- Test: `tests/simulator/test_plan.py`

**Interfaces:**
- Consumes: `Simulation`, `Change`, `SteadyState` (Task 2), `ModelSymbols`, `TargetKind`, `compile_formula` (Task 3), `UnitsInformation` (Task 1).
- Produces:
  - `Assignment` (frozen dataclass): `target: str`, `kind: TargetKind`, `value: float | None`, `formula: str | None`.
  - `PlanEvent` (frozen dataclass): `time: float`, `assignments: tuple[Assignment, ...]`.
  - `OutputMode(StrEnum)`: `INTEGRATOR`, `TIMES`.
  - `Plan` (frozen dataclass): `start: float`, `end: float`, `preinit: tuple[Assignment, ...]`, `events: tuple[PlanEvent, ...]` (sorted, merged per time), `steady_state: SteadyStatePlan | None`, `output: OutputMode`, `times: tuple[float, ...]` (empty for `INTEGRATOR`), `time_shift: float`; method `with_values(values: Mapping[str, float]) -> Plan` (the override semantics on model units, for the fit).
  - `SteadyStatePlan` (frozen dataclass): `preinit: tuple[Assignment, ...]`, `absolute_tolerance: float`, `relative_tolerance: float`, `max_time: float`.
  - `compile_simulation(simulation: Simulation, symbols: ModelSymbols, uinfo: UnitsInformation) -> Plan`.

- [ ] **Step 1: Write the failing tests**

```python
import pickle

import pytest

from sbmlsim import Q
from sbmlsim.simulation import Change, Simulation
from sbmlsim.simulator.plan import OutputMode, compile_simulation
from sbmlsim.simulator.symbols import ModelSymbols, TargetKind
from sbmlsim.units import UnitsInformation
from tests.simulator.models import sbml


@pytest.fixture(scope="module")
def probe() -> tuple[ModelSymbols, UnitsInformation]:
    s = sbml()
    return ModelSymbols.from_sbml(s), UnitsInformation.from_sbml(s)


def test_events_are_sorted_and_merged(probe) -> None:
    symbols, uinfo = probe
    plan = compile_simulation(
        Simulation(
            start=-10,
            end=10,
            changes=[
                Change([0, -10], {"k1": 1.0}),
                Change(0, {"[A]": "[A] + 1"}),
            ],
        ),
        symbols,
        uinfo,
    )
    assert [e.time for e in plan.events] == [-10.0, 0.0]
    assert {a.target for a in plan.events[1].assignments} == {"k1", "[A]"}
    formula = next(a for a in plan.events[1].assignments if a.target == "[A]")
    assert formula.kind is TargetKind.SPECIES_CONCENTRATION
    assert formula.formula == "[A] + 1" and formula.value is None
    assert plan.output is OutputMode.INTEGRATOR


def test_two_values_for_one_target_at_one_time_raise(probe) -> None:
    symbols, uinfo = probe
    with pytest.raises(ValueError, match="'k1'.*0"):
        compile_simulation(
            Simulation(end=1, changes=[Change(0, {"k1": 1.0}), Change(0, {"k1": 2.0})]),
            symbols,
            uinfo,
        )


def test_change_outside_interval_raises(probe) -> None:
    symbols, uinfo = probe
    with pytest.raises(ValueError, match="outside"):
        compile_simulation(
            Simulation(end=1, changes=[Change(2, {"k1": 1.0})]), symbols, uinfo
        )


def test_time_unit_and_steps(probe) -> None:
    symbols, uinfo = probe
    # the probe model has no time unit, i.e. dimensionless: a time unit of
    # the simulation which is not dimensionless cannot be converted
    plan = compile_simulation(Simulation(end=4, steps=4), symbols, uinfo)
    assert plan.output is OutputMode.TIMES
    assert plan.times == (0.0, 1.0, 2.0, 3.0, 4.0)


def test_plan_is_picklable_and_overridable(probe) -> None:
    symbols, uinfo = probe
    plan = compile_simulation(
        Simulation(
            end=1, preinit_changes={"k1": 1.0}, changes=[Change([0, 0.5], {"f": 3.0})]
        ),
        symbols,
        uinfo,
    )
    again = pickle.loads(pickle.dumps(plan))
    assert again == plan
    overridden = plan.with_values({"f": 4.0, "k2": 0.1})
    assert all(a.value == 4.0 for e in overridden.events for a in e.assignments)
    assert {a.target: a.value for a in overridden.preinit} == {"k1": 1.0, "k2": 0.1}
```

Add one test with a model with time unit `min` (antimony `unit time_unit = 60 second` is not antimony syntax; build it with libsbml: set `model.setTimeUnits("minute")` with a unit definition `minute`) asserting that `Simulation(time_unit="hr", end=2)` compiles to `end == 120.0`, and that `Change(Q(30, "min"), ...)` in a simulation with `time_unit="hr"` compiles to `30.0`.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest -n 0 tests/simulator/test_plan.py -q`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement `plan.py`**

Conversion rules: a time number is in `simulation.time_unit` (model time unit when `None`) and is converted with the factor `ureg.Quantity(1, time_unit).to(uinfo["time"]).magnitude`; a time quantity converts directly. A value quantity converts with `.to(uinfo[target]).magnitude` (a `DimensionalityError` is re-raised as `ValueError` naming the target, the given unit and the model unit); a number is taken as is (model units, no warning); a string is a formula, checked by `compile_formula` at compile time so a syntax error surfaces early, and every symbol of the formula must be `time` or resolvable by `symbols.kind`. `steps` becomes `OutputMode.TIMES` with `np.linspace(start, end, steps + 1)`, `times` must lie in `[start, end]`. Events outside `[start, end]` raise. `with_values` replaces the value of every `Assignment` of the target in `preinit` and in every event (a formula assignment becomes a value assignment), and appends a `PARAMETER`-kind preinit assignment for a target which is set nowhere; the kind of an added target is resolved by the compile step, so `Plan` keeps a `kinds: dict[str, TargetKind]` field for all symbols of the model and `with_values` reads it (a target missing from it raises `ValueError`).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -n 0 tests/simulator/test_plan.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sbmlsim/simulator/plan.py tests/simulator/test_plan.py
git commit -m "A simulation is compiled against a model into a plan without units"
```

### Task 5: The model a plan runs on: the initial values and their restore

**Files:**
- Modify: `src/sbmlsim/model/model_roadrunner.py`
- Test: `tests/model/test_model_roadrunner_init.py`

**Interfaces:**
- Consumes: `ModelSymbols` (Task 3).
- Produces on `RoadrunnerSBMLModel`: `symbols: ModelSymbols` (built at load), `set_initial_values(assignments: Sequence[Assignment]) -> None` (records the original initial value of every target the first time it is touched, restores every recorded target that `assignments` does not set, sets `init(...)` for the others), `free_initial_assignments(pids: Collection[str]) -> None` (derives and reloads the model once so that the parameters `pids`, which have an initial assignment, can be set before the initialization; see the design), `derived_initial: dict[str, str]` (parameter -> its `<p>__initial` helper).

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulator.plan import Assignment
from sbmlsim.simulator.symbols import TargetKind
from tests.simulator.models import sbml


def _a(
    target: str, value: float, kind: TargetKind = TargetKind.PARAMETER
) -> Assignment:
    return Assignment(target=target, kind=kind, value=value, formula=None)


def test_preinit_reaches_initial_assignments_and_is_restored() -> None:
    model = RoadrunnerSBMLModel(source=sbml())
    r = model.r
    model.set_initial_values([_a("f", 5.0), _a("b0", 0.0)])
    r.reset()
    assert r["[B]"] == pytest.approx(0.0)
    assert r["pinit"] == pytest.approx(10.0)
    assert r["X"] == pytest.approx(30.0)
    model.set_initial_values([])
    r.reset()
    assert r["[B]"] == pytest.approx(1.0)
    assert r["pinit"] == pytest.approx(4.0)


def test_parameter_with_initial_assignment_can_be_set() -> None:
    model = RoadrunnerSBMLModel(source=sbml())
    model.free_initial_assignments({"pinit"})
    model.set_initial_values([_a("pinit", 7.0)])
    model.r.reset()
    assert model.r["pinit"] == pytest.approx(7.0)
    assert model.r["X"] == pytest.approx(21.0)
    model.set_initial_values([_a("f", 3.0)])
    model.r.reset()
    assert model.r["pinit"] == pytest.approx(6.0)
    assert model.r["X"] == pytest.approx(18.0)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest -n 0 tests/model/test_model_roadrunner_init.py -q`
Expected: FAIL, `AttributeError: 'RoadrunnerSBMLModel' object has no attribute 'set_initial_values'`.

- [ ] **Step 3: Implement**

`RoadrunnerSBMLModel.__init__` builds `self.symbols = ModelSymbols.from_sbml(...)` from the same source, `self._init_original: dict[str, float] = {}` and `self.derived_initial: dict[str, str] = {}`. The key of an assignment is `init(<target>)` (`init([S])` for a concentration). `set_initial_values`:

```python
def set_initial_values(self, assignments: Sequence[Assignment]) -> None:
    """Set the initial values of a plan and restore the ones it does not set.

    roadrunner keeps a value set with `init(...)` across `resetToOrigin`, so
    the model records the original initial value of every target the first
    time it is set and writes it back before a plan which does not set it.
    A parameter of `derived_initial` gets the value of its helper when the
    plan does not set it, which needs a `reset` in between.

    Args:
        assignments: the pre-initialization assignments of a plan, values only.
    """
```

Restoring and setting go through `r.setValue(key, value)`; a target of `derived_initial` which the plan does not set is handled after the others: `r.reset()`, read `r[helper]`, `r.setValue(f"init({p})", value)`. `free_initial_assignments(pids)` takes the ids not yet in `derived_initial`, edits the SBML with libsbml (remove the `InitialAssignment` of `p`, add a parameter `<p>__initial` with `constant=true`, no value, and an initial assignment with the removed math, set `p` to value `0` so roadrunner accepts it), reloads `self.r` from the edited string, applies the integrator settings and timecourse selections the instance had, and records the helper; it is a no-op for an id without an initial assignment. Every selection list built from `getGlobalParameterIds()` excludes the helpers.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -n 0 tests/model -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sbmlsim/model/model_roadrunner.py tests/model/test_model_roadrunner_init.py
git commit -m "The model sets the initial values of a plan and restores what it does not set"
```

### Task 6: The executor

**Files:**
- Create: `src/sbmlsim/simulator/executor.py`
- Test: `tests/simulator/test_executor.py`

**Interfaces:**
- Consumes: `Plan`, `PlanEvent`, `Assignment`, `OutputMode`, `SteadyStatePlan` (Task 4), `RoadrunnerSBMLModel.set_initial_values`, `free_initial_assignments`, `symbols` (Task 5), `compile_formula` (Task 3), `TimecourseResult` (`result/timecourse.py`).
- Produces: `execute(plan: Plan, model: RoadrunnerSBMLModel, selections: Sequence[str]) -> TimecourseResult` (column `time` first, time shifted by `plan.time_shift`); `SteadyStateError(RuntimeError)`; `steady_state(model, plan: SteadyStatePlan) -> float` (time used) for reuse by phase 2.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import Change, Simulation, SteadyState
from sbmlsim.simulator.executor import SteadyStateError, execute
from sbmlsim.simulator.plan import compile_simulation
from tests.simulator.models import sbml

SEL = ["time", "[A]", "[B]", "X", "C", "k1"]


def run(sim: Simulation, model: RoadrunnerSBMLModel | None = None):
    model = model or RoadrunnerSBMLModel(source=sbml())
    plan = compile_simulation(sim, model.symbols, model.uinfo)
    return execute(plan, model, SEL)


def test_preinit_reaches_initial_assignment() -> None:
    res = run(Simulation(end=10, preinit_changes={"b0": 0.0}, times=[0, 10]))
    assert res["[B]"][0] == pytest.approx(0.0)
    # A + B = 1 at steady state, A = k2 / (k1 + k2)
    assert res["[A]"][-1] == pytest.approx(0.6 / 1.4, rel=1e-4)


def test_second_run_on_same_instance_uses_model_value() -> None:
    model = RoadrunnerSBMLModel(source=sbml())
    run(Simulation(end=1, preinit_changes={"b0": 0.0}), model)
    res = run(Simulation(end=1, times=[0, 1]), model)
    assert res["[B]"][0] == pytest.approx(1.0)


def test_change_at_start_and_output_at_change_time() -> None:
    res = run(
        Simulation(
            end=2,
            changes=[Change(0, {"[A]": 5.0}), Change(1, {"[A]": "[A] + 10"})],
            times=[0, 1, 2],
        )
    )
    assert res["[A]"][0] == pytest.approx(5.0)
    before = run(Simulation(end=1, changes=[Change(0, {"[A]": 5.0})], times=[0, 1]))
    assert res["[A]"][1] == pytest.approx(before["[A]"][1] + 10.0)


def test_simultaneous_formulas_use_old_values() -> None:
    res = run(
        Simulation(
            end=1, changes=[Change(0.5, {"[A]": "[B]", "[B]": "[A]"})], times=[0.5]
        )
    )
    ref = run(Simulation(end=0.5, times=[0.5]))
    assert res["[A]"][0] == pytest.approx(ref["[B]"][0])
    assert res["[B]"][0] == pytest.approx(ref["[A]"][0])


def test_compartment_change_keeps_concentration_and_amount_species_amount() -> None:
    res = run(Simulation(end=1, changes=[Change(0.5, {"C": 4.0})], times=[0.5]))
    ref = run(Simulation(end=0.5, times=[0.5]))
    assert res["[A]"][0] == pytest.approx(ref["[A]"][0])
    assert res["X"][0] == pytest.approx(ref["X"][0])
    assert res["C"][0] == pytest.approx(4.0)


def test_negative_start_and_time_shift() -> None:
    res = run(Simulation(start=-5, end=5, times=[-5, 0, 5], time_shift=5))
    np.testing.assert_allclose(res.time, [0.0, 5.0, 10.0])


def test_integrator_output_has_change_time_once() -> None:
    res = run(Simulation(end=2, changes=[Change(1, {"k1": 2.0})]))
    times = res.time
    assert np.all(np.diff(times) > 0)
    assert np.any(np.isclose(times, 1.0))


def test_multiple_dosing() -> None:
    res = run(
        Simulation(
            end=3, changes=[Change([0, 1, 2], {"[A]": "[A] + 1"})], times=[0, 1, 2, 3]
        )
    )
    assert res["[A]"][0] == pytest.approx(2.0)


def test_steady_state_presimulation() -> None:
    res = run(Simulation(end=1, presimulation=SteadyState(), times=[0]))
    assert res["[A]"][0] == pytest.approx(0.6 / 1.4 * 2.0, rel=1e-5)


def test_steady_state_not_reached_raises() -> None:
    growth = sbml("model g\n x' = 1\n x = 0\nend")
    model = RoadrunnerSBMLModel(source=growth)
    plan = compile_simulation(
        Simulation(end=1, presimulation=SteadyState(max_time=100)),
        model.symbols,
        model.uinfo,
    )
    with pytest.raises(SteadyStateError, match="100"):
        execute(plan, model, ["time", "x"])


def test_parameter_with_initial_assignment_preinit() -> None:
    res = run(Simulation(end=1, preinit_changes={"pinit": 7.0}, times=[0]))
    assert res["X"][0] == pytest.approx(21.0)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest -n 0 tests/simulator/test_executor.py -q`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement `executor.py`**

```python
def execute(
    plan: Plan, model: RoadrunnerSBMLModel, selections: Sequence[str]
) -> TimecourseResult:
    """Run a plan on a loaded model.

    1. The parameters with an initial assignment which the plan sets before
       the initialization are freed once (`free_initial_assignments`).
    2. The pre-initialization values are set (`set_initial_values`) and the
       model is initialized with `reset()`.
    3. A steady state presimulation integrates until the rates of change
       vanish, see `steady_state`.
    4. The interval is split at the times of the events. At the start of a
       segment the events of its time are applied: every value is evaluated
       first, then every target is set; a compartment keeps the
       concentration of the concentration species in it.
    5. A segment is integrated with the output of the plan, the rows of a
       segment end which is the start of the next segment are dropped, so a
       time of a change appears once, after the change.

    Raises:
        SteadyStateError: if the presimulation does not reach a steady state.
        RuntimeError: if roadrunner fails to integrate.
    """
```

Key implementation points, in this order:

- `r = model.r`; `r.timeCourseSelections = list(selections)` only when it differs (comparison against a cached tuple on the model, setting it is not free).
- Preinit: `model.free_initial_assignments({a.target for a in plan.preinit + steady.preinit if a.kind is PARAMETER and a.target in model.symbols.initial_assignments})`, then `model.set_initial_values(plan.preinit + steady.preinit)`, then `r.reset()`. `reset()` resets the time, the floating species and the rate rule targets to their initial values (the probe tests pin it).
- Steady state: integrate `r.simulate(0, horizon)` with `variable_step_size=True` for horizons `1, 10, 100, ...` up to `max_time`, after each check `rates = r.model.getStateVectorRate()`, `x = r.model.getStateVector()`, converged if `np.all(np.abs(rates) <= atol + rtol * np.abs(x))`; raise `SteadyStateError(f"no steady state up to the time {max_time}")` otherwise.
- Applying an event: `values = [r[a.target] ...]` is not needed; for every formula assignment evaluate `compile_formula(a.formula).evaluate([t if s == "time" else r[s] for s in f.symbols])`; collect `(target, value)`; then compartments first: for each compartment target, remember `r[f"[{s}]"]` of every species of it which is not in `only_substance` and not itself a target, set the compartment (`r[c] = v`), set every remembered species back with `r[f"[{s}]"] = conc`; then species and parameters with `r[target] = value`. Setting `r[...]` at the current time does not change `init(...)`.
- Output per segment `[a, b]`: `INTEGRATOR` sets `variable_step_size=True` and `r.simulate(a, b)`; `TIMES` sets `variable_step_size=False` and `r.simulate(times=[a, *inner, b])` with `inner` the output times in `(a, b)`, then keeps the rows of output times in `[a, b)` (the last segment keeps `b`). A segment of zero length (an event at `end`) only applies the event, and the last row of the previous segment is replaced by the state after it when `end` is an output time.
- Restore the integrator setting `variable_step_size` the model was configured with after the run.
- Concatenate with `TimecourseResult.concatenate`, add `plan.time_shift` to the column `time`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -n 0 tests/simulator -q`
Expected: PASS. A failing semantic test is a finding about roadrunner, fix the executor, never the test expectation (the expectations are the PEtab semantics).

- [ ] **Step 5: Commit**

```bash
git add src/sbmlsim/simulator/executor.py tests/simulator/test_executor.py
git commit -m "The executor runs a plan with the initialization and change semantics of PEtab v2"
```

### Task 7: Ragged results and their interpolation

**Files:**
- Modify: `src/sbmlsim/result/xresult.py`
- Modify: `src/sbmlsim/plot/serialization_matplotlib.py`, `src/sbmlsim/plot/serialization_plotly.py` (the `[:, 0]` selection drops the `NaN` padding)
- Modify or delete: `src/sbmlsim/result/datagenerator.py`, `examples/datagenerator.py` (reduce on an interpolated result, or delete both if `rg DataGenerator src examples tests` shows no other user)
- Test: `tests/result/test_xresult.py`

**Interfaces:**
- Produces: `XResult.from_timecourses(results, scan=None, uinfo=None)` with the dimension `_point` first, a variable `time` like every selection, `NaN` padding, no scan dimension without a scan; `XResult.interpolate(times: ArrayLike) -> XResult` with the dimension `_time` and coordinate `times`; `XResult.dim_mean(key, times=None)` (and `dim_std`, `dim_min`, `dim_max`) interpolating onto `times` or the union of the time points; `XResult.is_ragged() -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from sbmlsim.result import TimecourseResult, XResult
from sbmlsim.simulation import Dimension, ScanSim, Simulation


def _tc(times, values) -> TimecourseResult:
    return TimecourseResult(
        columns=("time", "y"), values=np.column_stack([times, values])
    )


def test_single_result_has_no_scan_dimension() -> None:
    xres = XResult.from_timecourses([_tc([0, 1, 3], [0, 1, 3])])
    assert xres["y"].dims == ("_point",)
    np.testing.assert_allclose(xres["time"].values, [0, 1, 3])


def test_ragged_scan_is_padded_and_interpolated() -> None:
    scan = ScanSim(
        Simulation(end=4),
        dimensions=[
            Dimension("d", index=np.arange(2), changes={"k": np.array([1.0, 2.0])})
        ],
    )
    xres = XResult.from_timecourses(
        [_tc([0, 4], [0, 4]), _tc([0, 1, 4], [0, 2, 8])], scan=scan
    )
    assert xres["y"].shape == (3, 2)
    assert np.isnan(xres["y"].values[2, 0])
    grid = xres.interpolate([0, 2, 4])
    np.testing.assert_allclose(
        grid["y"].values, [[0, 0], [2, 4.6666666667], [4, 8]], rtol=1e-6
    )
    mean = xres.dim_mean("y", times=[4])
    np.testing.assert_allclose(mean.magnitude, [6.0])
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest -n 0 tests/result/test_xresult.py -q`
Expected: FAIL (the old layout has `_time` and requires equal lengths).

- [ ] **Step 3: Implement**

`from_timecourses` builds `data = np.full((len(columns), n_point, *scan_shape), np.nan)` with `n_point = max(len(r) for r in results)` and assigns `data[(slice(None), slice(0, len(r)), *indices[k])] = r.values.T`; without a scan `scan_shape` is empty and there is exactly one result. `interpolate` uses `np.interp` per simulation on its non-`NaN` points (column `time` as x) and returns a new `XResult` whose dataset has `_time` (coordinate `times`) instead of `_point` and no variable `time`. `_redop_dims` excludes `_point` and `_time`. The plot serializers take `values[:, 0]` for a 2-d array as today and drop the trailing `NaN`s with `values[~np.isnan(values)]` on x and the same mask on y.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -n 0 tests/result tests/plot -q`
Expected: PASS for `tests/result/test_xresult.py`; failures in other result tests which construct `TimecourseSim` are migrated in Task 14.

- [ ] **Step 5: Commit**

```bash
git add src/sbmlsim/result src/sbmlsim/plot tests/result/test_xresult.py
git commit -m "Results keep the time points of every simulation, interpolation is on request"
```

### Task 8: `ScanSim` on `Simulation`

**Files:**
- Modify: `src/sbmlsim/simulation/scan.py`, `src/sbmlsim/simulation/range.py` (`Dimension.at`)
- Test: `tests/simulation/test_scan_definition.py`

**Interfaces:**
- Consumes: `Simulation.with_values`, `Change` (Task 2).
- Produces: `ScanSim(simulation: Simulation, dimensions: list[Dimension] | None = None)`; `Dimension(dimension, index=None, changes=None, at: float | Quantity | None = None)`; `ScanSim.to_simulations() -> tuple[list[tuple[int, ...]], list[Simulation]]`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from sbmlsim import Q
from sbmlsim.simulation import Change, Dimension, ScanSim, Simulation


def test_scan_replaces_dose_at_every_time() -> None:
    sim = Simulation(end=72, changes=[Change([0, 24, 48], {"PODOSE": Q(10, "mg")})])
    scan = ScanSim(
        sim, [Dimension("dose", changes={"PODOSE": Q(np.array([5.0, 20.0]), "mg")})]
    )
    _, sims = scan.to_simulations()
    assert [s.changes[0].values["PODOSE"] for s in sims] == [
        Q(5.0, "mg"),
        Q(20.0, "mg"),
    ]
    assert all(s.preinit_changes == {} for s in sims)


def test_scan_at_time_adds_a_change() -> None:
    sim = Simulation(end=10)
    scan = ScanSim(sim, [Dimension("k", changes={"k1": np.array([1.0, 2.0])}, at=5)])
    _, sims = scan.to_simulations()
    assert [s.changes[-1].times for s in sims] == [(5,), (5,)]
    assert [s.changes[-1].values["k1"] for s in sims] == [1.0, 2.0]


def test_scan_of_two_dimensions() -> None:
    sim = Simulation(end=1)
    scan = ScanSim(
        sim,
        [
            Dimension("a", changes={"k1": np.array([1.0, 2.0])}),
            Dimension("b", changes={"k2": np.array([3.0, 4.0, 5.0])}),
        ],
    )
    indices, sims = scan.to_simulations()
    assert len(sims) == 6 and indices[5] == (1, 2)
    assert sims[5].preinit_changes == {"k1": 2.0, "k2": 5.0}
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest -n 0 tests/simulation/test_scan_definition.py -q`
Expected: FAIL (`ScanSim` requires a `TimecourseSim`).

- [ ] **Step 3: Implement**

`to_simulations` builds, per index tuple, the values of every dimension without `at` and calls `simulation.with_values(values)`; the values of a dimension with `at` become an appended `Change(at, values)`. Remove `mapping`, `normalize` (units are converted when a plan is compiled) and the `__main__` block of `scan.py`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -n 0 tests/simulation -q`
Expected: PASS for the new tests.

- [ ] **Step 5: Commit**

```bash
git add src/sbmlsim/simulation tests/simulation/test_scan_definition.py
git commit -m "Scans replace the value of a target wherever the simulation sets it"
```

### Task 9: `SimulatorSerial` on the engine

**Files:**
- Modify: `src/sbmlsim/simulator/simulation_serial.py`
- Test: `tests/simulator/test_simulator_serial.py`

**Interfaces:**
- Consumes: `compile_simulation` (Task 4), `execute` (Task 6), `XResult.from_timecourses` (Task 7), `ScanSim.to_simulations` (Task 8).
- Produces: `SimulatorSerial.run_simulation(simulation: Simulation) -> XResult`, `SimulatorSerial.run_scan(scan: ScanSim) -> XResult`, `SimulatorSerial.simulate(simulation: Simulation) -> TimecourseResult` (one simulation, no `XResult`, for the testsuite runner and the fit), `SimulatorSerial.compile(simulation) -> Plan`. `run_timecourse` and `_timecourse(s)` are removed.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from sbmlsim.simulation import Change, Dimension, ScanSim, Simulation
from sbmlsim.simulator import SimulatorSerial
from tests.simulator.models import sbml
import numpy as np


def test_run_simulation_and_scan(tmp_path) -> None:
    path = tmp_path / "probe.xml"
    path.write_text(sbml())
    simulator = SimulatorSerial(model=path)
    xres = simulator.run_simulation(
        Simulation(end=1, preinit_changes={"b0": 0.0}, steps=10)
    )
    assert xres["[B]"].values[0] == pytest.approx(0.0)
    scan = ScanSim(
        Simulation(end=1, steps=10),
        [Dimension("d", changes={"b0": np.array([0.0, 2.0])})],
    )
    xres = simulator.run_scan(scan)
    assert xres["[B]"].values[0].tolist() == pytest.approx([0.0, 2.0])
```

- [ ] **Step 2: Run it to see it fail**

Run: `uv run pytest -n 0 tests/simulator/test_simulator_serial.py -q`
Expected: FAIL, `AttributeError: ... 'run_simulation'`.

- [ ] **Step 3: Implement**

`compile` uses `self.model_loaded.symbols` and `self.model_loaded.uinfo`; `simulate` executes with `self.model_loaded.selections`; `run_simulation` wraps one result; `run_scan` compiles each simulation of `to_simulations()` and executes it. `set_timecourse_selections` stays and stores the selections on the model. Remove the imports of `Timecourse`, `TimecourseSim`, `ModelChange`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -n 0 tests/simulator -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sbmlsim/simulator tests/simulator/test_simulator_serial.py
git commit -m "The serial simulator compiles simulations into plans and runs them"
```

### Task 10: Simulation experiments, tasks and data on `Simulation`

**Files:**
- Modify: `src/sbmlsim/experiment/experiment.py`, `src/sbmlsim/experiment/runner.py`, `src/sbmlsim/task/task.py`, `src/sbmlsim/data.py`, `src/sbmlsim/model/model.py` (`AbstractModel.changes` are merged into `preinit_changes` of every simulation of a task, the simulation wins), `src/sbmlsim/model/model_change.py` (`clamp_species` becomes `AbstractModel.manipulations`, applied when the model is loaded)
- Test: `tests/experiment/test_experiment_run.py` (migrate), `tests/experiment/test_model_changes_merge.py` (new)

**Interfaces:**
- Consumes: `SimulatorSerial.run_simulation`, `run_scan` (Task 9).
- Produces: `SimulationExperiment.simulations() -> dict[str, Simulation | ScanSim]`; `SimulationExperiment.ureg` is `sbmlsim.units.ureg`; no `Q_`.

- [ ] **Step 1: Write the failing test**

```python
from sbmlsim import Q
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import Simulation
from sbmlsim.task import Task
from tests.simulator.models import sbml


def test_model_changes_merge_into_preinit(tmp_path) -> None:
    path = tmp_path / "probe.xml"
    path.write_text(sbml())

    class Exp(SimulationExperiment):
        def models(self):
            return {"m": AbstractModel(source=path, changes={"b0": 0.0, "a0": 2.0})}

        def simulations(self):
            return {"s": Simulation(end=1, steps=2, preinit_changes={"a0": 3.0})}

        def tasks(self):
            return {"t": Task(model="m", simulation="s")}

    runner = ExperimentRunner([Exp], base_path=tmp_path, data_path=tmp_path)
    exp = runner.experiments["Exp"]
    exp.run(runner.simulator)
    xres = exp.results["t"]
    assert xres["[B]"].values[0] == 0.0
    assert xres["[A]"].values[0] == 3.0
```

- [ ] **Step 2: Run it to see it fail**

Run: `uv run pytest -n 0 tests/experiment/test_model_changes_merge.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

`_run_tasks` builds `simulation.with_values(model.changes | simulation.preinit_changes)` restricted to the model changes not set by the simulation (use `{k: v for k, v in model.changes.items() if k not in simulation.targets()}` then `with_values`), and calls `run_simulation` or `run_scan`; the `normalize`/`deepcopy` calls go away. `Data.get_data` drops `NaN` padding of a 1-d result (no scan). Remove `self.Q_` and every `Q_` in `src/`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -n 0 tests/experiment -q`
Expected: PASS after migrating `tests/experiment/test_experiment_run.py` to `Simulation`.

- [ ] **Step 5: Commit**

```bash
git add src/sbmlsim tests/experiment
git commit -m "Simulation experiments run Simulation and ScanSim, the changes of a model are pre-initialization changes"
```

### Task 11: Sensitivity scans on `Simulation`

**Files:**
- Modify: `src/sbmlsim/simulation/sensitivity.py`
- Test: existing tests of `ModelSensitivity` (`rg -l ModelSensitivity tests`), migrated

**Interfaces:**
- Produces: `ModelSensitivity.difference_sensitivity_scan(model, simulation: Simulation, ...)` and `distribution_sensitivity_scan(...)` reading `simulation.preinit_changes` where they read `timecourses[0].changes`, and building `ScanSim(simulation, [dim])` without `mapping`.

- [ ] **Step 1: Migrate the tests to `Simulation`, run them, see them fail**

Run: `uv run pytest -n 0 $(rg -l ModelSensitivity tests) -q`
Expected: FAIL on the old attribute.

- [ ] **Step 2: Implement and run again**

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/sbmlsim/simulation/sensitivity.py tests
git commit -m "The sensitivity scans are scans of a Simulation"
```

### Task 12: The fit on plans

**Files:**
- Modify: `src/sbmlsim/fit/optimization.py`, `src/sbmlsim/fit/parameter_mapping.py`, `src/sbmlsim/fit/derived.py`
- Test: `tests/fit/test_optimization.py`, `tests/fit/test_derived_changes.py`, `tests/fit/test_parameter_mapping.py` (migrate), `tests/fit/test_initial_assignment_parameter.py` (new)

**Interfaces:**
- Consumes: `SimulatorSerial.compile`, `execute`, `Plan.with_values`.
- Produces: `OptimizationProblem.plans: list[Plan]` (one per mapping group, compiled in `initialize` with `times` = the sorted union of the x references of the group when every mapping of the group has the x `time`, the output of the simulation otherwise); `OptimizationProblem.defined_changes` is the `preinit_changes` of the simulation; an evaluation is `execute(plan.with_values(values), model, selections)` with `values` = the parameter values in model units plus the derived changes.

- [ ] **Step 1: Write the failing test**

```python
"""A fit parameter which feeds an initial assignment reaches it."""

import numpy as np
import pandas as pd

from sbmlsim.data import DataSet
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import FitData, FitMapping, FitParameter
from sbmlsim.fit.objects import FitMappingCollection
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import Simulation
from sbmlsim.task import Task
from tests.simulator.models import sbml

PATH = None  # set by the fixture below


class IAExperiment(SimulationExperiment):
    def models(self):
        return {"m": AbstractModel(source=PATH)}

    def datasets(self):
        df = pd.DataFrame(
            {
                "time": [0.0],
                "time_unit": "dimensionless",
                "B": [0.0],
                "B_unit": "dimensionless",
            }
        )
        return {"d": DataSet.from_df(df, ureg=self.ureg)}

    def simulations(self):
        return {"s": Simulation(end=1)}

    def tasks(self):
        return {"t": Task(model="m", simulation="s")}

    def fit_mappings(self):
        return {
            "fm": FitMapping(
                self,
                reference=FitData(self, dataset="d", xid="time", yid="B"),
                observable=FitData(self, task="t", xid="time", yid="[B]"),
            )
        }


def test_fit_parameter_reaches_initial_assignment(tmp_path) -> None:
    global PATH
    PATH = tmp_path / "probe.xml"
    PATH.write_text(sbml())
    problem = OptimizationProblem(
        opid="ia",
        mapping_collections=[
            FitMappingCollection(experiment=IAExperiment, mappings=["fm"])
        ],
        fit_parameters=[
            FitParameter(
                pid="b0",
                lower_bound=0.0,
                upper_bound=2.0,
                start_value=1.0,
                unit="dimensionless",
            )
        ],
        base_path=tmp_path,
        data_path=tmp_path,
    )
    problem.initialize(FitSettings(parameter_scale="LIN"))
    predictions = problem.predictions(np.array([0.0]))
    assert predictions[0][0] == 0.0
```

Adjust the imports and the `FitSettings` argument to the actual names (`rg -n "class FitSettings|parameter_scale" src/sbmlsim/fit/options.py`), keeping the assertion.

- [ ] **Step 2: Run it to see it fail**

Run: `uv run pytest -n 0 tests/fit/test_initial_assignment_parameter.py -q`
Expected: FAIL (the simulation is not a `TimecourseSim`).

- [ ] **Step 3: Implement**

In `initialize`, accept `Simulation` (refuse `ScanSim` with the existing message adapted), compile one plan per mapping group after `_group_mappings`, with `simulation` replaced by a copy whose output is `times` when the group observes the time. `ParameterMapping.changes_for` returns quantities as today; convert once per evaluation with factors computed in `initialize` (`factor[ix] = Q(1, punit).to(model unit of the target).magnitude`) so an evaluation multiplies instead of calling pint, and the derived changes (`evaluate_derived_changes`) are converted the same way with a cached factor per key. `_check_shared_simulation_bindings` and the mutation of `timecourses[0].changes` are removed: plans are immutable and `with_values` builds the evaluated plan. `_interpolate` stays (a mapping whose x is not the time interpolates on the output of its simulation).

- [ ] **Step 4: Run the fit tests**

Run: `uv run pytest tests/fit -q -x`
Expected: PASS after migrating every test which builds a `TimecourseSim` (Task 14 lists them); the HCTZ costs of `tests/fit/test_optimization.py` may change because of the initial assignment fix, update an expected value only after checking that the change is caused by a parameter which feeds an initial assignment (print the parameters of `op_hctz` and `rg -n initialAssignment` the model).

- [ ] **Step 5: Commit**

```bash
git add src/sbmlsim/fit tests/fit
git commit -m "The fit compiles its simulations into plans and evaluates them without units"
```

### Task 13: The PEtab layer and the SBML Test Suite runner on `Simulation`

**Files:**
- Modify: `src/sbmlsim/fit/petab_v2/reader.py`, `export.py`, `extension.py`, `gaps.py`, `src/sbmlsim/testsuite/runner.py`, `src/sbmlsim/testsuite/submission.py`, `src/sbmlsim/fit/petab_v2/sciml.py` (if it builds timecourses: `rg -n Timecourse src/sbmlsim/fit/petab_v2 src/sbmlsim/sciml`)
- Test: `tests/fit/test_petab_v2*.py`, `tests/sciml/*`, `tests/testsuite/*` (migrate)

**Interfaces:**
- Produces: `PetabReader.simulations() -> dict[str, Simulation]` with the periods as `Change`s at their absolute times, `start` the time of the first finite period, the measurements as `times`, a period at `-inf` as `SteadyState`; the extension block `experiments.<id>` stores `{"simulation": Simulation.to_dict()}` (extension version `0.3.0`; the reader still reads `timecourses` of `0.2.0` by converting relative timecourses to absolute changes: the k-th timecourse starts at `time_offset + sum(end of the timecourses before)`, its changes become a `Change` at that time, the changes of the first one become `preinit_changes`); `run_case` of the SBML Test Suite builds `Simulation(start=start, end=end, steps=steps)` with the initial values of the case as `preinit_changes`.

- [ ] **Step 1: Migrate the tests of these modules to `Simulation` and run them**

Run: `uv run pytest tests/fit/test_petab_v2.py tests/fit/test_petab_v2_reader.py tests/fit/test_petab_v2_dosing.py tests/fit/test_petab_v2_definition.py tests/sciml tests/testsuite -q`
Expected: FAIL on the removed classes.

- [ ] **Step 2: Implement the reader, the export, the extension and the runner**

The export writes a `Simulation`: the `preinit_changes` are the condition of the first period at `start`, every distinct time of the `changes` is a period with a condition of the values at that time, a formula value is written as the `targetValue` formula (the PEtab math of the ids of PEtab, translated with `symbols.py`), a `SteadyState` is a period at `-inf` with a condition of its `preinit_changes`. `gaps.py` drops the gaps of `time_offset` and of a pre-simulation of a finite duration.

- [ ] **Step 3: Run the tests**

Run: same command as Step 1.
Expected: PASS. Then run the SBML Test Suite and compare with the baseline:

Run: `uv run python scripts/testsuite.py download && uv run pytest -m testsuite -q`
Expected: no case changes its outcome against `tests/data/testsuite_baseline.json`, a case which starts to pass is removed from the baseline with `uv run python scripts/testsuite.py baseline`.

Run: `uv run python scripts/sciml_testsuite.py download && uv run pytest -m sciml_testsuite tests/sciml -q`
Expected: no case changes its outcome against `tests/data/sciml_baseline.json`.

- [ ] **Step 4: Commit**

```bash
git add src/sbmlsim tests scripts tests/data
git commit -m "The PEtab layer and the SBML Test Suite runner use Simulation"
```

### Task 14: Remove `Timecourse`, migrate examples and the remaining tests

**Files:**
- Delete: `src/sbmlsim/simulation/timecourse.py`
- Modify: every file of `rg -l "TimecourseSim|Timecourse\b|AbstractSim|time_offset|\.timecourses|Q_" src tests examples scripts`
- Modify: `examples/README.md` (the list of failing examples stays accurate)

- [ ] **Step 1: List the files**

Run: `rg -l "TimecourseSim|Timecourse\b|AbstractSim|time_offset|\.timecourses|\bQ_\b" src tests examples scripts`

- [ ] **Step 2: Migrate each file with these rules**

- `TimecourseSim([Timecourse(start=0, end=e1, steps=n, changes=c1), Timecourse(start=0, end=e2, steps=n, changes=c2), ...], time_offset=o)` becomes `Simulation(start=o, end=o + e1 + e2 + ..., preinit_changes=c1, changes=[Change(o + e1, c2), ...])`; equal changes at several times become one `Change` with the vector of times; an example which plots on a grid keeps `steps=` with the total number of steps, a fit keeps no output (the fit sets its times).
- A change which resets a state during the simulation (e.g. `Aurine_hctz = 0` in `weir1998.py`) stays a `Change` at its times; a dose at the start time stays a `Change` at `start` when it is a dose, so the time vector of the doses is one vector.
- `Q_ = self.Q_` and `Q_(...)` become `from sbmlsim import Q` and `Q(...)`.
- `run_timecourse(` becomes `run_simulation(`.
- `ScanSim(..., mapping={...: k})` with `k > 0` becomes a `Dimension(..., at=<absolute time of timecourse k>)`.

- [ ] **Step 3: Run everything**

Run: `uv run pytest -q`
Expected: PASS.
Run: `uv run pytest -q tests/examples`
Expected: PASS for the examples listed as working in `examples/README.md`.
Run: `rg -n "TimecourseSim|Timecourse\b|AbstractSim|time_offset|\.timecourses|\bQ_\b" src tests examples scripts`
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add -A src tests examples scripts
git commit -m "Timecourse and TimecourseSim are removed, the examples and tests use Simulation"
```

### Task 15: Documentation, release notes, CLAUDE.md and the full verification

**Files:**
- Modify: `docs/simulation.md` (rename the page title to "Simulations"; the definition, the semantics with the table of the spec, multiple dosing, steady state, output, ragged results), `docs/scans.md`, `docs/units.md` (`Q`, one registry), `docs/experiments.md`, `docs/fitting.md` (where it shows timecourses), `zensical.toml` (nav label "Simulations"), `CLAUDE.md` (the architecture paragraph of `simulation/`, `simulator/`, `result/` and `units.py`)
- Create: `release-notes/0.9.0.md` (the breaking change, the initial assignment fix and its effect on results, the migration rules of Task 14)

- [ ] **Step 1: Write the documentation**

Every code block of the documentation runs: copy each block into `tests/docs/test_simulation_docs.py` as a test which executes it against the probe model or `REPRESSILATOR_SBML`.

- [ ] **Step 2: Verify**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check`
Expected: no findings.
Run: `uv run pytest -q`
Expected: PASS.
Run: `uv run zensical build --clean`
Expected: `No issues found`.

- [ ] **Step 3: Commit and open the pull request**

```bash
git add -A docs CLAUDE.md release-notes zensical.toml tests/docs
git commit -m "The documentation of the simulation engine and the release notes of 0.9.0"
git push -u origin simulation-engine
```

Open the pull request with `gh-axi pr create --base develop --head simulation-engine --title "The simulation engine: one way to set up simulations with the semantics of PEtab v2 (phase 1)" --body-file <body>`; the body lists what changed, the breaking changes, the verification commands and their results, without agent attribution.
