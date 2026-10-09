# The scan core, phase 2: observables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A scan computes observables from every simulation, `Simulator.run(model, scan, observables, keep=...)`: formulas with the reductions `max`, `min`, `mean` and `at` over the time of each simulation, the non-compartmental analysis of pkpdutils (`PK`) and custom functions, evaluated in the workers on the native solutions; the result hands a PK analysis and a timecourse over to pkpdutils.

**Architecture:** The definitions `Formula`, `PK` and `Custom` (`simulation/observables.py`) are frozen and picklable. `simulator/observables.py` compiles them against the first model of a run into an `ObservableGraph` (ordered by what they read, kinds and units derived, free of pint), which every chunk carries into the workers; a run without observables has the graph of the selections of its model. The worker stacks the native solutions of the points of a chunk into arrays `(n_points, n_rows)` padded with `NaN`, evaluates the graph on them (formulas through the reductions of `simulator/formula.py`, `PK` through `simulator/pk.py` and pkpdutils, `Custom` per simulation), then interpolates the kept timecourses onto the grid of `time=`; the parent writes timecourses over `(*dims, time)` or `(*dims, _point)` and values per simulation over the scan dimensions into the `ScanResult`.

**Tech Stack:** python 3.13/3.14, uv, libroadrunner (CVODE), numpy, sympy (through `compile_formula`), pint (units in the parent only), pkpdutils >= 1.3 (lazily imported), xarray, h5netcdf, pytest with xdist, ruff, ty, zensical.

**Spec:** `docs/superpowers/specs/2026-10-08-scan-core-design.md` (phase 2 of its "Phases": "The observables: `Formula` with the reductions (the pre-pass moved from `data.py`), `PK` with pkpdutils, `Custom`, `keep`, `nca`/`to_timecourses`, `observables.md`"; the sections "Observables", "Execution" (observables, `keep`, selections), "The result" (`nca`, `to_timecourses`) and "Testing"). Phase 1 is merged into `develop` (`cf21ae55`).

## Global Constraints

- Phase 2 starts from `develop` at `cf21ae55` (phase 1 merged). Branch `scan-core-phase2`, one pull request to `develop`, one commit per task.
- `Simulator.run(model, scan, observables=None, *, time=None, keep=None, on_error="raise", progress=None) -> ScanResult`; a run without observables keeps the behavior of phase 1: the selections of the model, each a timecourse of its own name, and every existing test passes unchanged except where a task says otherwise.
- `Observable` definitions live in `simulation/observables.py`, are frozen and pickle; their evaluation in `simulator/observables.py` (and `simulator/pk.py`) uses no pint and no xarray. `Formula(id, formula, unit=None)`, `PK(id, selection, *, dose=None, route=None, options=None, parameters=None)`, `Custom(id, function, unit, *, symbols, kind=SCALAR)`; `ObservableKind` is `TIMECOURSE` or `SCALAR`.
- The reductions of a formula are `max(x)`, `min(x)` (ignoring `NaN`), `mean(x)` (the trapezoidal integral divided by the time between the first and the last time point) and `at(x, t)` (linear interpolation, the value after a change at its time, `NaN` outside); `max` and `min` with two or more arguments stay the elementwise functions of PEtab. They reduce the time of each simulation, not the whole array; `Data` formulas use the same pre-pass.
- A formula whose free symbols are all values per simulation is a value per simulation, otherwise a timecourse; a value per simulation broadcasts over the time (`ins / ins0`).
- The unit of a formula is derived by applying it to quantities of one in the units of its symbols; the reductions keep the unit of their argument; with `unit` given and a derived unit the values are converted into `unit` and an incompatible unit raises; where no unit can be derived `unit` is taken as declared and without it the compile step raises with the formula and asks for `unit=`.
- An observable reads the selections of the model and other observables by id; a cycle or an unknown symbol raises when the scan is compiled; an observable id which is a selection of the model, a changed target or a dimension id raises.
- The selections a run asks roadrunner for are `time` and the union of what the (kept and needed) observables read.
- `PK`: every parameter pkpdutils derives is a value per simulation `<id>.<parameter>` with the unit pkpdutils gives it; the dose amounts are the values the plan of a point assigns to the dose target and the dose times the times of these assignments (`start` for a `preinit_changes` value); a formula value of the target raises; a `Quantity` is a fixed dose at `start`; pkpdutils (`>=1.3.0`) is a dependency and imported only when a `PK` observable is compiled or evaluated.
- `Custom.function(time, values)` is called once per simulation with its native time points; it must be a function of a module; a lambda or a closure raises.
- `keep` is the list of observables in the result, all by default; the others are evaluated as intermediates and dropped.
- The observables are computed on the native solution, before any interpolation onto `time`; the result does not depend on `n_workers` or the chunk size.
- `on_error="flag"` sets every observable of a failing point to `NaN` and records it in `status`.
- `ScanResult.nca(id)` gives the `pkpdutils.NCAResult` of a PK observable, `ScanResult.to_timecourses(id)` a timecourse as `pkpdutils.Timecourses`; the result layout `(*dims, time)` is the layout of pkpdutils.
- Never use the em dash character, use a plain dash `-`.
- No agent attribution anywhere: no `Co-Authored-By` trailer, no "Generated with Claude Code" line in commits, the pull request, docs or code.
- Commit messages are full sentences which describe the outcome, in the style of `git log` (e.g. "The fit runs its repeats in the pools of sbmlsim.parallel"), with a body that explains what and why; no conventional commit prefixes.
- Never edit `CHANGELOG.md` or auto-generated files; no release notes (they belong to the release commit).
- Markdown has no hard line wraps: a paragraph, list item or table row is one line.
- Every module, class and function of the package has full type annotations and a google style docstring (ruff `D`); `tests/` and `examples/` are exempt from docstrings. A subclass marks overrides with `typing.override`.
- ty stays at zero diagnostics (`uv run ty check`); suppress only with a rule specific `# ty: ignore[rule-name]`.
- Library code logs with `logging.getLogger(__name__)` and lazy `%s` formatting, it never prints.
- Every commit passes `uv run ruff check`, `uv run ruff format --check`, `uv run ty check` and `uv run pytest -q` (the default suite runs in parallel with xdist; `filterwarnings = error` turns every warning into a failure). Use `uv run` for every command.

## Decisions this plan takes where the spec is silent

- The base class `Observable` has the `id` and `reads` (the symbols it reads) and `to_dict`; the kind of a `Formula` is derived when the scan is compiled, so it is no attribute of the definition.
- An observable id is an identifier of letters, digits and underscores (no dot, no brackets); the parameters of a PK observable are `<id>.<parameter>`, and `<id>.flags` (the flags of pkpdutils, `pkpdutils.NCAFlag`) is always one of them, so `nca` can build the `NCAResult`. `compile_formula` reads an identifier with a dot as one symbol.
- The time points of a reduction are the finite times: the padding (`NaN`) and the steady state after the end (`inf`) are none. A value per simulation is constant in time, so `max`, `min`, `mean` and `at` of it are the value itself. `mean` of a simulation with a single time point is the value at it.
- A formula of data (`Data` of type FUNCTION) reduces with `max` and `min` along the last axis, the time of each simulation; `mean` and `at` need the times, which data has not, and raise with a hint to the observables. A formula of data whose identifiers are all reductions or numbers is a value per simulation (a number for a single simulation).
- Floating point errors of a formula (a division by zero) give `inf` or `NaN` without a warning.
- A selection without a unit counts as dimensionless; the conversion into a declared unit is a factor (no offset units).
- The function of a `Custom` is checked when the definition is created, which is before the scan is compiled; a symbol which is a value per simulation is passed as a float.
- The timecourse of a `PK` is a selection of the model or the id of a timecourse observable (e.g. a mass concentration as a `Formula`). The doses are the positive values the plan assigns to the target, a change at the start replaces the value before the initialization, the values of a presimulation are no doses, and the times are shifted by the `time_shift` of the simulation. A dose needs its `route`; an infusion (`iv_infusion`), whose duration a change does not carry, raises.
- The parameters of a `PK` and their units are found when the scan is compiled by an analysis of pkpdutils of a synthetic timecourse with the dosing of the first point of every plan, so every point has the same variables; a parameter pkpdutils does not give for a point is `NaN`. Without a dose the parameters which need one are left out, not `NaN` (their unit would need the unit of a dose).
- The points are analysed by pkpdutils in groups of the same number of doses and of time points, so no row is padded and a parameter does not depend on the chunking.
- Inside of the graph every observable keeps its natural (derived) unit; the factor into a declared `unit` is applied only to the result. A formula which mixes units of one dimension at different scales raises when the scan is compiled (a comparison of mixed scales is not detected). A declared unit is taken as declared when the formula derives no unit, or derives dimensionless from symbols which all have no unit or from numbers only. The time of `at` is a number (in the time unit of the model) or has the time unit of the model.
- `PK(parameters=...)` and `Custom(symbols=...)` take any sequence of names (annotated `Sequence[str]`, stored as tuples). `mean` sums its trapezoids sequentially, independent of the padding.
- With observables, `keep` names observables (the id of a PK observable keeps all its parameters); the selections are intermediates. Without observables, `keep` names selections. An observable no kept observable needs is not evaluated, and its selections are not selected.
- A result whose kept observables are all values per simulation has no time dimension and no variable or coordinate `time`.
- The observables are evaluated on the points which ran, on the chunk at once; after a failure the points are evaluated one at a time, so a point fails exactly when its own evaluation fails. With `on_error="raise"` the first failing point in the order of the scan is raised, an observable failure of an earlier point before a later execution failure.
- `nca` gives a failed point the flags `0` and `NaN` parameters; `to_timecourses(id, **kwargs)` passes `dose`, `route` and the other arguments of `Timecourses.from_arrays` on.
- `RoadrunnerSBMLModel.has_selection(name)` checks a selection (`r.getValue`), which the fit used as `_is_selection` and the observables need; the fit uses the method.
- `FLAGS = "flags"` is a constant of `sbmlsim.result.scan`, which `simulator/pk.py` imports, so the result does not import the simulator.

## Review Focus

- A flagged failing point in a scan with `Formula`, `Custom` and `PK` observables: every observable of that point is `NaN`, the other points are unchanged and pkpdutils never sees the failed row; covered in Task 6 (`test_a_failing_point_is_nan_for_every_observable`).
- A ragged scan (the steps of the integrator) whose points have different numbers of time points: the reductions and the PK analysis ignore the padding; covered in Task 3 (`test_ragged_rows_are_analysed_without_their_padding`) and Task 6 (`test_reductions_on_the_steps_of_the_integrator`).
- `keep` which drops an intermediate another kept observable needs gives the same values as the run without `keep`; covered in Task 6 (`test_keep_drops_intermediates_but_evaluates_them`).
- A dose which only a dimension gives (the base simulation does not set the target, the dimension adds it before the initialization) is read per point at the start; covered in Task 6 (`test_a_dose_only_a_dimension_gives`).
- Observables of a model whose default selections exclude what they read still read it; covered in Task 6 (`test_observables_read_what_the_model_does_not_select`).

---

## File Structure

- Modify `src/sbmlsim/simulator/formula.py`: identifiers with a dot; `Reduction`, `ReducedFormula`, `reduce_formula`, `evaluate_reduced`, `REDUCTIONS` (the pre-pass moved from `data.py`, extended by `mean` and `at`, reducing the last axis).
- Modify `src/sbmlsim/data.py`: `evaluate_function` uses the shared pre-pass, the reductions of data are per simulation.
- Create `src/sbmlsim/simulation/observables.py`: `ObservableKind`, `Observable`, `Formula`, `PK`, `Custom`.
- Create `src/sbmlsim/simulator/pk.py`: `DoseSpec`, `PKNode`, `doses_of`, `compile_pk`, `evaluate_pk`.
- Create `src/sbmlsim/simulator/observables.py`: `FormulaNode`, `CustomNode`, `ObservableError`, `ObservableGraph`, `identity_graph`, `compile_observables`.
- Modify `src/sbmlsim/model/model_roadrunner.py`: `RoadrunnerSBMLModel.has_selection`; `src/sbmlsim/fit/optimization.py` uses it.
- Modify `src/sbmlsim/simulator/worker.py`: `Chunk.graph`, `point_plan`, `ChunkResult.scalars`, the evaluation of the graph on the stacked native solutions.
- Modify `src/sbmlsim/simulator/simulator.py`: `run(..., observables, keep)`, the compile of the graph, the assembly of timecourses and values per simulation.
- Modify `src/sbmlsim/result/scan.py`: `FLAGS`, `ScanResult.nca`, `ScanResult.to_timecourses`.
- Create `docs/observables.md`, `examples/observables.py`, API pages; modify `docs/scans.md`, `docs/data.md`, `zensical.toml`, `CLAUDE.md`, `examples/README.md`, `pyproject.toml`.

---

### Task 1: The reductions over the time of each simulation

**Files:**
- Modify: `src/sbmlsim/simulator/formula.py` (whole module, see Step 3)
- Modify: `src/sbmlsim/data.py:29-142` (the pre-pass moves out, `evaluate_function` uses it)
- Modify: `docs/data.md:67`
- Test: `tests/simulator/test_formula_reductions.py` (new), `tests/test_data_function.py`

**Interfaces:**
- Consumes: `sbmlsim.result.timecourse.grid_weights(time, grid) -> GridWeights` and `apply_weights(w, values) -> np.ndarray` (phase 1; the right-most value at a duplicated time, `NaN` outside, non-finite times ignored).
- Produces:
  - `compile_formula(formula: str) -> CompiledFormula` reads `a.b` as one symbol `"a.b"`.
  - `REDUCTIONS: dict[str, str]` (`"max": "max(x)"`, ..., `"at": "at(x, t)"`).
  - `Reduction(symbol: str, function: str, arguments: tuple[str, ...])`, frozen.
  - `ReducedFormula(formula: str, outer: str, reductions: tuple[Reduction, ...])`, frozen, with the properties `placeholders: frozenset[str]`, `symbols: tuple[str, ...]` (every identifier read, sorted, without placeholders), `outer_symbols: tuple[str, ...]` (the identifiers outside of the reductions, without placeholders), `time_symbols: tuple[str, ...]` (the identifiers of the times of the `at` reductions, without placeholders).
  - `reduce_formula(formula: str) -> ReducedFormula` (cached; raises `ValueError`).
  - `evaluate_reduced(formula: str, values: Mapping[str, Any], time: np.ndarray | None = None) -> Any`.

- [ ] **Step 1: Write the failing tests**

Create `tests/simulator/test_formula_reductions.py`:

```python
"""The reductions of a formula over the time of each simulation."""

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.simulator.formula import compile_formula, evaluate_reduced, reduce_formula

#: two simulations: the second has a change at 1 (a duplicated time) and padding
T = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 1.0, np.nan]])
X = np.array([[1.0, 3.0, 2.0, 0.0], [5.0, 4.0, 6.0, np.nan]])


def test_max_and_min_reduce_every_simulation() -> None:
    np.testing.assert_array_equal(evaluate_reduced("max(x)", {"x": X}, T), [[3.0], [6.0]])
    np.testing.assert_array_equal(evaluate_reduced("min(x)", {"x": X}, T), [[0.0], [4.0]])


def test_a_reduction_broadcasts_over_the_time() -> None:
    np.testing.assert_allclose(
        evaluate_reduced("x / max(x)", {"x": X}, T), X / np.array([[3.0], [6.0]])
    )


def test_the_mean_is_weighted_by_the_time() -> None:
    # (1+3)/2 + (3+2)/2 + (2+0)/2 = 5.5 over 3; (5+4)/2 over 1, the change adds 0
    np.testing.assert_allclose(
        evaluate_reduced("mean(x)", {"x": X}, T), [[5.5 / 3.0], [4.5]]
    )


def test_the_mean_does_not_depend_on_the_steps() -> None:
    coarse = np.linspace(0.0, 2.0, 3)[None, :]
    fine = np.linspace(0.0, 2.0, 101)[None, :]
    for t in (coarse, fine):
        mean = evaluate_reduced("mean(x)", {"x": 2.0 * t + 1.0}, t)
        assert mean[0, 0] == pytest.approx(3.0)


def test_at_interpolates_and_takes_the_value_after_a_change() -> None:
    np.testing.assert_allclose(
        evaluate_reduced("at(x, 1.5)", {"x": X}, T), [[2.5], [np.nan]]
    )
    np.testing.assert_allclose(evaluate_reduced("at(x, 1)", {"x": X}, T), [[3.0], [6.0]])
    assert np.isnan(evaluate_reduced("at(x, 5)", {"x": X}, T)).all()


def test_the_time_of_at_is_a_value_per_simulation() -> None:
    when = np.array([[0.5], [0.0]])
    np.testing.assert_allclose(
        evaluate_reduced("at(x, when)", {"x": X, "when": when}, T), [[2.0], [5.0]]
    )


def test_the_padding_and_the_steady_state_are_no_time_points() -> None:
    t = np.array([[0.0, 1.0, np.inf]])
    x = np.array([[1.0, 2.0, 100.0]])
    assert evaluate_reduced("max(x)", {"x": x}, t)[0, 0] == 2.0
    assert evaluate_reduced("mean(x)", {"x": x}, t)[0, 0] == pytest.approx(1.5)
    assert np.isnan(evaluate_reduced("at(x, 2)", {"x": x}, t)[0, 0])


def test_a_simulation_without_values_reduces_to_nan_without_a_warning() -> None:
    t = np.full((1, 3), np.nan)
    x = np.full((1, 3), np.nan)
    for formula in ("max(x)", "min(x)", "mean(x)", "at(x, 1)"):
        assert np.isnan(evaluate_reduced(formula, {"x": x}, t)).all()


def test_a_single_time_point_is_its_own_mean() -> None:
    t = np.array([[2.0, np.nan]])
    x = np.array([[7.0, np.nan]])
    assert evaluate_reduced("mean(x)", {"x": x}, t)[0, 0] == 7.0


def test_a_value_per_simulation_is_constant_in_time() -> None:
    s = np.array([[2.0], [3.0]])
    np.testing.assert_array_equal(
        evaluate_reduced("max(s) + mean(s) + at(s, 9)", {"s": s}, T), [[6.0], [9.0]]
    )


def test_an_inner_reduction_is_reduced_first() -> None:
    np.testing.assert_allclose(
        evaluate_reduced("max(x - min(x))", {"x": X}, T), [[3.0], [2.0]]
    )


def test_a_reduction_keeps_the_unit() -> None:
    q = Q(X, "mmol/l")
    assert evaluate_reduced("max(x)", {"x": q}, T).units == q.units
    assert evaluate_reduced("mean(x)", {"x": q}, T).units == q.units
    assert evaluate_reduced("at(x, 1)", {"x": q}, T).units == q.units


@pytest.mark.parametrize("formula", ["mean(x)", "at(x, 1)"])
def test_mean_and_at_need_the_time(formula: str) -> None:
    with pytest.raises(ValueError, match="observable"):
        evaluate_reduced(formula, {"x": X})


@pytest.mark.parametrize(
    "formula", ["max()", "mean(x, 2)", "at(x)", "at(x, 1, 2)", "max(x", "x +"]
)
def test_a_wrong_formula_is_reported(formula: str) -> None:
    with pytest.raises(ValueError):
        reduce_formula(formula)


def test_an_identifier_without_a_value_is_reported() -> None:
    with pytest.raises(ValueError, match="'y'"):
        evaluate_reduced("x + max(y)", {"x": X}, T)


def test_the_symbols_of_a_reduced_formula() -> None:
    reduced = reduce_formula("ins / at(ins, t0) + max([glc])")
    assert reduced.symbols == ("[glc]", "ins", "t0")
    assert reduced.outer_symbols == ("ins",)
    assert reduced.time_symbols == ("t0",)
    assert len(reduced.reductions) == 2


def test_an_identifier_with_a_dot_is_one_symbol() -> None:
    compiled = compile_formula("hctz.auc_inf_obs / hctz.cmax + 1.5")
    assert compiled.symbols == ("hctz.auc_inf_obs", "hctz.cmax")
    assert compiled.evaluate([10.0, 2.0]) == pytest.approx(6.5)
```

Append to `tests/test_data_function.py`:

```python
def test_a_reduction_of_a_scan_is_per_simulation() -> None:
    y = np.array([[1.0, 2.0, 4.0], [1.0, 1.0, 2.0]])
    np.testing.assert_allclose(
        evaluate_function("Y/max(Y)", {"Y": y}), [[0.25, 0.5, 1.0], [0.5, 0.5, 1.0]]
    )
    np.testing.assert_allclose(evaluate_function("max(Y)", {"Y": y}), [4.0, 2.0])


def test_a_reduction_of_a_simulation_is_a_number() -> None:
    value = evaluate_function("max(Y) + k", {"Y": np.array([1.0, 3.0]), "k": 1.0})
    assert np.ndim(value) == 0
    assert value == pytest.approx(4.0)


@pytest.mark.parametrize("formula", ["mean(Y)", "at(Y, 1)"])
def test_mean_and_at_are_no_reductions_of_data(formula: str) -> None:
    with pytest.raises(ValueError, match="observable"):
        evaluate_function(formula, {"Y": np.array([1.0, 2.0])})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/simulator/test_formula_reductions.py tests/test_data_function.py`
Expected: FAIL with `ImportError: cannot import name 'evaluate_reduced'` (and the new data tests fail on the reduction over the whole array).

- [ ] **Step 3: Write the implementation**

Replace `src/sbmlsim/simulator/formula.py` with:

```python
"""Formulas of the math of PEtab over the selections of roadrunner.

A formula is a string of the math of PEtab whose symbols are selections of
roadrunner: `S` is the amount of a species, `[S]` its concentration and `time`
the time. The brackets are not math, so `[S]` is replaced by an identifier
before the formula is parsed and mapped back afterwards; so is an identifier
with a dot, the parameter of a PK observable, e.g. `hctz.cmax`.

A formula is compiled to a numpy function once and cached, a `CompiledFormula`
is not pickled: a plan keeps the formula as a string.

The formulas of observables and of data extend the math by four reductions
over the time of one simulation, see `reduce_formula` and `evaluate_reduced`:
`max(x)` and `min(x)` with a single argument (with two or more they are the
elementwise functions of PEtab), `mean(x)`, the time weighted mean, and
`at(x, t)`, the value at a time. A value is an array whose last axis is the
time of a simulation, so a scan of many simulations reduces every simulation
on its own.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

from sbmlsim.result.timecourse import apply_weights, grid_weights

#: a concentration `[S]` in a formula
_BRACKETS = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*)\]")

#: prefix of the identifier which stands for a concentration while parsing
_PREFIX = "sbmlsim_concentration__"

#: an identifier with a dot, e.g. the parameter `cmax` of the observable `hctz`
_DOTTED = re.compile(
    r"(?<![A-Za-z0-9_.])([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)(?![A-Za-z0-9_.])"
)

#: prefix of the identifier which stands for an identifier with a dot
_DOT = "sbmlsim_dotted__"

#: the reductions over the time of a simulation and how they are called
REDUCTIONS: dict[str, str] = {
    "max": "max(x)",
    "min": "min(x)",
    "mean": "mean(x)",
    "at": "at(x, t)",
}

#: a call of a reduction which is not the end of a longer identifier
_REDUCTION_CALL = re.compile(r"(?<![A-Za-z0-9_.])(max|min|mean|at)\s*\(")

#: prefix of the symbol which stands for the value of a reduction
_REDUCTION_PREFIX = "sbmlsim_reduction__"


@dataclass(frozen=True)
class CompiledFormula:
    """A formula compiled to a numpy function of its symbols.

    Attributes:
        formula: the formula as it was given.
        symbols: the selections the formula reads, sorted.
    """

    formula: str
    symbols: tuple[str, ...]
    _function: Callable[..., Any] = field(repr=False, compare=False)

    def evaluate(self, values: Sequence[float]) -> float:
        """Evaluate the formula.

        Args:
            values: the values of the symbols, in the order of `symbols`.

        Returns:
            The value of the formula.
        """
        return float(self._function(*values))

    def apply(self, values: Sequence[Any]) -> Any:
        """Evaluate the formula on values of any type numpy operates on.

        Unlike `evaluate`, the result is not converted to a float, arrays and
        the quantities of pint keep their shape and their units.

        Args:
            values: the values of the symbols, in the order of `symbols`.

        Returns:
            The value of the formula.
        """
        return self._function(*values)

    def evaluate_array(self, values: Sequence[np.ndarray], size: int) -> np.ndarray:
        """Evaluate the formula on arrays of the values of its symbols.

        Args:
            values: the values of the symbols, in the order of `symbols`, one
                array of `size` values per symbol.
            size: the number of values, which a formula without symbols needs.

        Returns:
            The `size` values of the formula.
        """
        value = np.asarray(self._function(*values), dtype=float)
        return np.broadcast_to(value, (size,)).copy()


@functools.cache
def compile_formula(formula: str) -> CompiledFormula:
    """Compile a formula, see the module.

    Args:
        formula: the formula.

    Returns:
        The compiled formula.

    Raises:
        ValueError: if the formula is not valid math of PEtab.
    """
    # petab.v2 imports its SciML extension and torch, which costs seconds;
    # only a simulation with a formula pays it
    from petab.v2.math import sympify_petab

    dotted: dict[str, str] = {}

    def escape(match: re.Match[str]) -> str:
        return dotted.setdefault(match.group(1), f"{_DOT}{len(dotted)}")

    escaped = _DOTTED.sub(escape, formula)
    escaped = _BRACKETS.sub(lambda m: f"{_PREFIX}{m.group(1)}", escaped)
    names = {identifier: name for name, identifier in dotted.items()}

    def selection(name: str) -> str:
        if name in names:
            return names[name]
        if name.startswith(_PREFIX):
            return f"[{name.removeprefix(_PREFIX)}]"
        return name

    try:
        expression = sympify_petab(escaped)
    except Exception as err:
        raise ValueError(f"The formula '{formula}' is not valid math: {err}") from err
    ordered = sorted(expression.free_symbols, key=lambda s: selection(str(s)))
    symbols = tuple(selection(str(s)) for s in ordered)
    function = sp.lambdify(ordered, expression, modules="numpy")
    return CompiledFormula(formula=formula, symbols=symbols, _function=function)


@dataclass(frozen=True)
class Reduction:
    """A reduction over the time of a simulation, see `reduce_formula`.

    Attributes:
        symbol: the identifier which stands for its value in the formula.
        function: `max`, `min`, `mean` or `at`.
        arguments: the formulas of its arguments, which read the symbols of
            the reductions inside of them.
    """

    symbol: str
    function: str
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class ReducedFormula:
    """A formula whose reductions are replaced by symbols, see `reduce_formula`.

    Attributes:
        formula: the formula as it was given.
        outer: the formula with every reduction replaced by its symbol.
        reductions: the reductions, an inner one before the one whose
            argument it is.
    """

    formula: str
    outer: str
    reductions: tuple[Reduction, ...]

    @property
    def placeholders(self) -> frozenset[str]:
        """Get the symbols which stand for the reductions."""
        return frozenset(reduction.symbol for reduction in self.reductions)

    @property
    def symbols(self) -> tuple[str, ...]:
        """Get the identifiers the formula reads, also inside of the reductions."""
        parts = (self.outer, *(a for r in self.reductions for a in r.arguments))
        found = {s for part in parts for s in compile_formula(part).symbols}
        return tuple(sorted(found - self.placeholders))

    @property
    def outer_symbols(self) -> tuple[str, ...]:
        """Get the identifiers the formula reads outside of its reductions."""
        return tuple(
            s for s in compile_formula(self.outer).symbols if s not in self.placeholders
        )

    @property
    def time_symbols(self) -> tuple[str, ...]:
        """Get the identifiers the times of the `at` reductions read."""
        found = {
            s
            for r in self.reductions
            if r.function == "at"
            for s in compile_formula(r.arguments[1]).symbols
        }
        return tuple(sorted(found - self.placeholders))


@functools.cache
def reduce_formula(formula: str) -> ReducedFormula:
    """Find the reductions of a formula, see the module.

    Args:
        formula: the formula.

    Returns:
        The formula with its reductions.

    Raises:
        ValueError: if the parentheses are not balanced, a reduction has not
            its number of arguments or a part of the formula is not valid math.
    """
    reductions: list[Reduction] = []
    outer = _replace_reductions(formula, formula, reductions)
    for part in (outer, *(a for r in reductions for a in r.arguments)):
        try:
            compile_formula(part)
        except ValueError as err:
            raise ValueError(f"The formula '{formula}' is not valid math: {err}") from err
    return ReducedFormula(formula=formula, outer=outer, reductions=tuple(reductions))


def _closing_parenthesis(text: str, start: int, formula: str) -> int:
    """Find the parenthesis which closes the one opened before `start`.

    Raises:
        ValueError: if the parentheses of the formula are not balanced.
    """
    depth = 1
    for k in range(start, len(text)):
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
            if depth == 0:
                return k
    raise ValueError(f"The parentheses of the formula '{formula}' are not balanced.")


def _split_arguments(text: str) -> list[str]:
    """Split the arguments of a call at the commas outside of parentheses."""
    arguments: list[str] = []
    depth = 0
    start = 0
    for k, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            arguments.append(text[start:k])
            start = k + 1
    arguments.append(text[start:])
    return arguments


def _replace_reductions(text: str, formula: str, reductions: list[Reduction]) -> str:
    """Replace every reduction of a text by a symbol, inner reductions first.

    Raises:
        ValueError: if the parentheses are not balanced or a reduction has not
            its number of arguments.
    """
    parts: list[str] = []
    position = 0
    while (match := _REDUCTION_CALL.search(text, position)) is not None:
        end = _closing_parenthesis(text, match.end(), formula)
        arguments = [
            _replace_reductions(argument, formula, reductions)
            for argument in _split_arguments(text[match.end() : end])
        ]
        name = match.group(1)
        parts.append(text[position : match.start()])
        if name in ("max", "min") and len(arguments) > 1:
            parts.append(f"{name}({','.join(arguments)})")
        else:
            expected = 2 if name == "at" else 1
            if len(arguments) != expected or not all(a.strip() for a in arguments):
                raise ValueError(
                    f"The reduction '{name}' of the formula '{formula}' is called "
                    f"as {REDUCTIONS[name]}."
                )
            symbol = f"{_REDUCTION_PREFIX}{len(reductions)}"
            reductions.append(
                Reduction(symbol=symbol, function=name, arguments=tuple(arguments))
            )
            parts.append(symbol)
        position = end + 1
    parts.append(text[position:])
    return "".join(parts)


def evaluate_reduced(
    formula: str, values: Mapping[str, Any], time: np.ndarray | None = None
) -> Any:
    """Evaluate a formula with reductions over the time of every simulation.

    A value is a number, a numpy array or a quantity of pint. The last axis of
    an array is the time of a simulation and the axes before it are the
    simulations, e.g. `(n_points, n_rows)` padded with `NaN` in a worker or
    `(*dims, time)` of the data of a scan. A number, or an array whose last
    axis has one element, is a value per simulation, which is constant in
    time. A reduction keeps its axis with one element, so its value broadcasts
    against the timecourses, and keeps the unit of its argument:

    - `max(x)`, `min(x)`: the largest and the smallest value, ignoring `NaN`;
    - `mean(x)`: the trapezoidal integral divided by the time between the
      first and the last time point, the value itself for a single time point;
    - `at(x, t)`: the value at the time `t`, a number or a value per
      simulation, interpolated linearly; the value after a change at its time
      and `NaN` outside of the time points.

    The times which are not finite, the padding (`NaN`) and the steady state
    after the end (`inf`), are no time points of a reduction.

    Args:
        formula: the formula.
        values: the values of its identifiers.
        time: the time points of the timecourses, of their shape; `None` where
            there are no times, e.g. for data, which allows `max` and `min`.

    Returns:
        The value of the formula.

    Raises:
        ValueError: if the formula is not valid, reads an identifier without a
            value, or reduces a timecourse with `mean` or `at` without `time`.
    """
    reduced = reduce_formula(formula)
    scope = dict(values)
    for reduction in reduced.reductions:
        x = _apply(reduction.arguments[0], scope, formula)
        if reduction.function == "at":
            when = _apply(reduction.arguments[1], scope, formula)
            scope[reduction.symbol] = _at(x, when, time, formula)
        elif reduction.function == "mean":
            scope[reduction.symbol] = _mean(x, time, formula)
        else:
            scope[reduction.symbol] = _extreme(reduction.function, x, time)
    return _apply(reduced.outer, scope, formula)


def _apply(part: str, values: Mapping[str, Any], formula: str) -> Any:
    """Evaluate a part of a formula without reductions on the values.

    Raises:
        ValueError: if the part reads an identifier which has no value.
    """
    compiled = compile_formula(part)
    missing = [symbol for symbol in compiled.symbols if symbol not in values]
    if missing:
        raise ValueError(f"The formula '{formula}' reads {missing}, which have no values.")
    return compiled.apply([values[symbol] for symbol in compiled.symbols])


def _split(x: Any) -> tuple[np.ndarray, Any]:
    """Split a value into a float array and its unit, `None` without one."""
    units = getattr(x, "units", None)
    if units is not None and hasattr(x, "magnitude"):
        return np.asarray(x.magnitude, dtype=float), units
    return np.asarray(x, dtype=float), None


def _join(magnitude: np.ndarray, units: Any) -> Any:
    """Give an array its unit again."""
    return magnitude if units is None else magnitude * units


def _constant(magnitude: np.ndarray) -> bool:
    """Check whether a value is constant in time: a number or a value per simulation."""
    return magnitude.ndim == 0 or magnitude.shape[-1] == 1


def _times(magnitude: np.ndarray, time: np.ndarray | None) -> np.ndarray:
    """Get the time points of a reduction: the finite times, every point without times."""
    if time is None:
        return np.ones(magnitude.shape, dtype=bool)
    return np.broadcast_to(np.isfinite(np.asarray(time, dtype=float)), magnitude.shape)


def _extreme(function: str, x: Any, time: np.ndarray | None) -> Any:
    """Reduce to the largest or the smallest value, ignoring `NaN`."""
    magnitude, units = _split(x)
    if _constant(magnitude):
        return x
    masked = np.where(_times(magnitude, time), magnitude, np.nan)
    ufunc = np.fmax if function == "max" else np.fmin
    # fmax and fmin ignore NaN and give NaN for a row of NaN, without a warning
    return _join(ufunc.reduce(masked, axis=-1, keepdims=True), units)


def _no_time(function: str, formula: str) -> ValueError:
    """Get the error of a reduction which needs the times and has none."""
    return ValueError(
        f"'{function}' in the formula '{formula}' needs the time points of the "
        f"simulation, which data has not; it is a reduction of the Formula "
        f"observables of a scan."
    )


def _mean(x: Any, time: np.ndarray | None, formula: str) -> Any:
    """Reduce to the time weighted mean, see `evaluate_reduced`."""
    magnitude, units = _split(x)
    if _constant(magnitude):
        return x
    if time is None:
        raise _no_time("mean", formula)
    t = np.broadcast_to(np.asarray(time, dtype=float), magnitude.shape)
    valid = np.isfinite(t)
    segment = valid[..., 1:] & valid[..., :-1]
    with np.errstate(invalid="ignore", over="ignore"):
        pieces = 0.5 * (magnitude[..., 1:] + magnitude[..., :-1]) * np.diff(t, axis=-1)
    area = np.where(segment, pieces, 0.0).sum(axis=-1, keepdims=True)
    finite = np.where(valid, t, np.nan)
    span = np.fmax.reduce(finite, axis=-1, keepdims=True) - np.fmin.reduce(
        finite, axis=-1, keepdims=True
    )
    last = _last(magnitude, valid)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(span > 0, area / np.where(span > 0, span, 1.0), last)
    return _join(mean, units)


def _last(magnitude: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Get the value at the last time point of every simulation, `NaN` without one."""
    index = valid.shape[-1] - 1 - np.argmax(valid[..., ::-1], axis=-1)
    value = np.take_along_axis(magnitude, index[..., None], axis=-1)
    return np.where(valid.any(axis=-1, keepdims=True), value, np.nan)


def _at(x: Any, when: Any, time: np.ndarray | None, formula: str) -> Any:
    """Reduce to the value at a time, see `evaluate_reduced`."""
    magnitude, units = _split(x)
    if _constant(magnitude):
        return x
    if time is None:
        raise _no_time("at", formula)
    target, _ = _split(when)
    shape = magnitude.shape
    times = np.broadcast_to(np.asarray(time, dtype=float), shape).reshape(-1, shape[-1])
    rows = magnitude.reshape(-1, shape[-1])
    targets = np.broadcast_to(target, (*shape[:-1], 1)).reshape(-1)
    out = np.array(
        [
            apply_weights(grid_weights(times[k], targets[k : k + 1]), rows[k])[0]
            for k in range(rows.shape[0])
        ]
    )
    return _join(out.reshape(*shape[:-1], 1), units)
```

In `src/sbmlsim/data.py`, delete `_REDUCTION_CALL`, `_REDUCTION_PREFIX`, `_REDUCTIONS`, `_closing_parenthesis`, `_split_arguments`, `_replace_reductions` and `_evaluate` (lines 29-117), import `compile_formula`, `evaluate_reduced` and `reduce_formula` from `sbmlsim.simulator.formula` (keep only the names used) and replace `evaluate_function` with:

```python
def evaluate_function(formula: str, variables: Mapping[str, Any]) -> Any:
    """Evaluate the formula of a `Data` of type FUNCTION on its data.

    The formula is the math of PEtab, see `sbmlsim.simulator.formula`, with
    the reductions of `evaluate_reduced`: `max` and `min` of a single argument
    reduce it along its last axis, the time of a simulation, and ignore `NaN`,
    the padding of a scan, so `Y/max(Y)` normalizes every simulation of a scan
    to its own maximum. With two or more arguments they are the elementwise
    maximum and minimum of PEtab. `mean` and `at` need the time points of a
    simulation, which data has not; they are reductions of the observables of
    a scan. A formula whose identifiers are all reductions or numbers is a
    value per simulation, a number for a single simulation.

    Args:
        formula: the formula.
        variables: the values of the identifiers of the formula, the arrays
            or quantities of the data and the numbers of the parameters.

    Returns:
        The value of the formula, a quantity if the variables are quantities.

    Raises:
        ValueError: if the formula is not valid math, reads an identifier
            which is not a variable, or uses `mean` or `at` on an array.
    """
    value = evaluate_reduced(formula, variables)
    reduced = reduce_formula(formula)
    per_simulation = bool(reduced.reductions) and all(
        symbol in reduced.placeholders
        or np.ndim(getattr(variables[symbol], "magnitude", variables[symbol])) == 0
        for symbol in compile_formula(reduced.outer).symbols
    )
    if per_simulation and np.ndim(getattr(value, "magnitude", value)) > 0:
        return value[..., 0]
    return value
```

In `docs/data.md:67`, replace the sentence about `max` and `min` with: "One extension serves data: `max` and `min` of a single argument reduce it along the time of every simulation and ignore the `NaN` of padding, so `Y/max(Y)` normalizes every simulation of a scan to its own maximum; with two or more arguments they are the elementwise maximum and minimum. The reductions `mean` and `at` need the time points of a simulation and are part of the observables of a scan, see [Observables](observables.md)." (The page `observables.md` is created in Task 8; the docs build of this commit still passes, since zensical warns about a missing link only in strict mode - check with `uv run zensical build --clean` and, if it fails, link to `scans.md` here and change the link to `observables.md` in Task 8.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/simulator/test_formula_reductions.py tests/test_data_function.py tests/simulator/test_formula.py tests/test_data.py`
Expected: PASS. Then `uv run pytest -q` (all tests; the scan regression compares results, not data, so it is unaffected; `examples/repressilator/repressilator_scans.py` now normalizes every simulation to its own maximum, which is the fix).

- [ ] **Step 5: Lint, types, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check`

```bash
git add src/sbmlsim/simulator/formula.py src/sbmlsim/data.py docs/data.md tests/simulator/test_formula_reductions.py tests/test_data_function.py
git commit -m "Formulas reduce over the time of each simulation with max, min, mean and at" -m "The pre-pass of the reductions of data moves to sbmlsim.simulator.formula and reduces the last axis, the time of a simulation, so the max of a Data of a scan is the maximum of every simulation and not of all of them. mean (the time weighted mean) and at (the value at a time, the one after a change) join max and min for the observables of phase 2, which need the time points; data has none and raises for them. compile_formula reads an identifier with a dot, the parameter of a PK observable, as one symbol."
```

---

### Task 2: The definitions of the observables

**Files:**
- Create: `src/sbmlsim/simulation/observables.py`
- Modify: `src/sbmlsim/simulation/__init__.py` (exports)
- Modify: `pyproject.toml:60-62` (pkpdutils)
- Modify: `tests/simulator/models.py` (custom functions, numpy import)
- Test: `tests/simulation/test_observables.py` (new)

**Interfaces:**
- Consumes: `sbmlsim.simulator.formula.reduce_formula` (Task 1), imported inside the methods (the simulator package imports the definitions, a module level import would be circular); `sbmlsim.simulation.scan.RESERVED`; `sbmlsim.units.Quantity`, `ureg`.
- Produces:
  - `ObservableKind(StrEnum)`: `TIMECOURSE = "timecourse"`, `SCALAR = "scalar"`.
  - `Observable` (frozen dataclass, field `id: str`; property `reads -> tuple[str, ...]`; `to_dict() -> dict[str, Any]`).
  - `Formula(id: str, formula: str, unit: str | None = None)`.
  - `PK(id: str, selection: str, *, dose: str | Quantity | None = None, route: str | None = None, options: NCAOptions | None = None, parameters: tuple[str, ...] | None = None)` (a sequence of parameters is stored as a tuple).
  - `Custom(id: str, function: Callable[[np.ndarray, dict[str, Any]], Any], unit: str, *, symbols: tuple[str, ...], kind: ObservableKind = ObservableKind.SCALAR)`.
  - `sbmlsim.simulation` exports `Custom`, `Formula`, `Observable`, `ObservableKind`, `PK`.
  - `tests/simulator/models.py`: `PK_MODEL`, `sbml_pk(ke: float = 0.2) -> str` are added in Task 3; this task adds `auc_of_c`, `doubled`, `fails_for_large_k1`, `last_value`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/simulator/models.py` (add `from typing import Any` and `import numpy as np` to its imports):

```python
def auc_of_c(time: np.ndarray, values: dict[str, Any]) -> float:
    """A custom observable: the trapezoidal area under `[C]`."""
    return float(np.trapezoid(values["[C]"], time))


def doubled(time: np.ndarray, values: dict[str, Any]) -> np.ndarray:
    """A custom timecourse: twice `[C]`."""
    return 2.0 * values["[C]"]


def last_value(time: np.ndarray, values: dict[str, Any]) -> float:
    """A custom observable: the last value of `S`."""
    return float(values["S"][-1])


def fails_for_large_k1(time: np.ndarray, values: dict[str, Any]) -> float:
    """A custom observable which fails for a point whose `k1` is above one."""
    if values["k1"][0] > 1.0:
        raise ValueError("k1 is too large")
    return 0.0
```

Create `tests/simulation/test_observables.py`:

```python
"""The definitions of the observables."""

import json
import pickle

import pytest
from pkpdutils import NCAOptions

from sbmlsim import Q
from sbmlsim.simulation import PK, Custom, Formula, Observable, ObservableKind
from tests.simulator.models import auc_of_c


def test_a_formula_reads_its_symbols() -> None:
    formula = Formula("ins_rel", "ins / at(ins, 0) + max([glc])")
    assert isinstance(formula, Observable)
    assert formula.reads == ("[glc]", "ins")


@pytest.mark.parametrize("formula", ["", "max(", "at(x)", "x +"])
def test_an_invalid_formula_raises(formula: str) -> None:
    with pytest.raises(ValueError):
        Formula("f", formula)


@pytest.mark.parametrize(
    "name", ["", "1a", "a.b", "[X]", "time", "status", "_point", "statistic"]
)
def test_an_invalid_id_raises(name: str) -> None:
    with pytest.raises(ValueError):
        Formula(name, "1")


def test_a_unit_pint_does_not_know_raises() -> None:
    with pytest.raises(ValueError, match="unit"):
        Formula("f", "1", unit="furlongs_per_blob")


def test_a_pk_observable() -> None:
    pk = PK(
        "hctz",
        "[Cve_hctz]",
        dose="PODOSE_hctz",
        route="oral",
        parameters=["cmax", "auc_inf_obs"],
    )
    assert pk.parameters == ("cmax", "auc_inf_obs")
    assert pk.reads == ("[Cve_hctz]",)


def test_a_dose_needs_its_route() -> None:
    with pytest.raises(ValueError, match="route"):
        PK("p", "[C]", dose="PODOSE")


def test_a_fixed_dose_is_one_amount() -> None:
    with pytest.raises(ValueError, match="one amount"):
        PK("p", "[C]", dose=Q([1.0, 2.0], "mg"), route="oral")


def test_the_parameters_are_a_sequence_of_names() -> None:
    with pytest.raises(TypeError):
        PK("p", "[C]", parameters="cmax")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError):
        PK("p", "[C]", parameters=["cmax", "cmax"])


def test_a_custom_observable() -> None:
    custom = Custom("auc", auc_of_c, "mg*hr/l", symbols=["[C]"])
    assert custom.kind is ObservableKind.SCALAR
    assert custom.symbols == ("[C]",)
    assert custom.reads == ("[C]",)


def test_a_lambda_or_a_closure_is_refused() -> None:
    with pytest.raises(ValueError, match="module"):
        Custom("r", lambda t, v: 1.0, "dimensionless", symbols=[])

    def inner(t: object, v: object) -> float:
        return 1.0

    with pytest.raises(ValueError, match="module"):
        Custom("r", inner, "dimensionless", symbols=[])


def test_the_definitions_pickle_and_serialize() -> None:
    observables = [
        Formula("f", "max([C])", unit="mg/l"),
        PK("p", "[C]", dose=Q(10.0, "mg"), route="oral", options=NCAOptions()),
        PK("q", "[C]", dose="PODOSE", route="iv_bolus", parameters=["cmax"]),
        Custom("c", auc_of_c, "mg*hr/l", symbols=["[C]"]),
    ]
    for observable in observables:
        again = pickle.loads(pickle.dumps(observable))
        assert again.to_dict() == observable.to_dict()
    data = json.loads(json.dumps([o.to_dict() for o in observables]))
    assert data[0] == {"type": "Formula", "id": "f", "formula": "max([C])", "unit": "mg/l"}
    assert data[1]["dose"] == {"value": 10.0, "unit": "milligram"}
    assert data[3]["function"] == "tests.simulator.models:auc_of_c"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/simulation/test_observables.py`
Expected: FAIL with `ImportError: cannot import name 'PK' from 'sbmlsim.simulation'` (or `ModuleNotFoundError: No module named 'pkpdutils'` before the dependency is added).

- [ ] **Step 3: Write the implementation**

In `pyproject.toml`, after `"SALib>=1.6.0",` add:

```toml
	# the non-compartmental analysis of the PK observables of a scan, imported
	# when a PK observable is compiled
	"pkpdutils>=1.3.0",
```

Run `uv sync --extra dev` (uv.lock is not tracked).

Create `src/sbmlsim/simulation/observables.py`:

```python
"""Observables: what a scan computes from every simulation.

An observable reads the selections of roadrunner (`S` amount, `[S]`
concentration, `time`, parameters, compartments, reactions) and other
observables by id:

- `Formula(id, formula, unit=None)`: the math of PEtab with the reductions
  over the time of a simulation, `max`, `min`, `mean` and `at`, see
  `sbmlsim.simulator.formula`; a timecourse, or a value per simulation when
  it reads only values per simulation, e.g. `max([glc])`;
- `PK(id, selection, *, dose=None, route=None, options=None, parameters=None)`:
  the non-compartmental analysis of pkpdutils of a timecourse, a value per
  simulation `<id>.<parameter>` for every parameter, e.g. `hctz.cmax`;
- `Custom(id, function, unit, *, symbols, kind=SCALAR)`: a function of a
  module, called with the time points and the values of its symbols of every
  simulation.

The definitions are frozen and pickle. `Simulator.run(model, scan,
observables)` compiles them against the models of the run, see
`sbmlsim.simulator.observables`, and evaluates them in the workers on the
native solution of every simulation.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, override

import numpy as np

from sbmlsim.simulation.scan import RESERVED
from sbmlsim.units import Quantity, ureg

if TYPE_CHECKING:
    from pkpdutils import NCAOptions

#: the id of an observable
_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ObservableKind(StrEnum):
    """What an observable is per simulation."""

    #: a value at every time point
    TIMECOURSE = "timecourse"
    #: one value
    SCALAR = "scalar"


def _check_id(name: str, kind: str) -> None:
    """Check the id of an observable.

    Raises:
        ValueError: if the id is no identifier or a name of the result.
    """
    if not isinstance(name, str) or not _ID.fullmatch(name):
        raise ValueError(
            f"The id of a {kind} observable is an identifier of letters, digits "
            f"and underscores, not {name!r}."
        )
    if name in RESERVED:
        raise ValueError(
            f"The id '{name}' is a name of the result ({sorted(RESERVED)}), choose "
            f"another one."
        )


def _check_unit(unit: str, name: str) -> None:
    """Check that pint reads a unit.

    Raises:
        ValueError: if pint does not read it.
    """
    try:
        ureg.parse_units(unit)
    except Exception as err:
        raise ValueError(
            f"The unit '{unit}' of the observable '{name}' is no unit: {err}"
        ) from err


def _check_names(names: Any, what: str, name: str) -> tuple[str, ...]:
    """Check a sequence of unique names.

    Raises:
        TypeError: if the names are a string.
        ValueError: if a name is empty or appears twice.
    """
    if isinstance(names, str):
        raise TypeError(
            f"The {what} of the observable '{name}' are a sequence of names, not "
            f"the string {names!r}."
        )
    names = tuple(names)
    if len(set(names)) != len(names) or not all(
        isinstance(n, str) and n for n in names
    ):
        raise ValueError(
            f"The {what} of the observable '{name}' are unique names, not {names}."
        )
    return names


def _check_function(function: Any, name: str) -> None:
    """Check that a function is defined at the top level of a module.

    Raises:
        TypeError: if it is not callable.
        ValueError: if it is a lambda, a closure or not found in its module.
    """
    if not callable(function):
        raise TypeError(
            f"The function of the observable '{name}' is not callable: {function!r}."
        )
    qualname = getattr(function, "__qualname__", "")
    module = sys.modules.get(getattr(function, "__module__", None) or "")
    found: Any = module
    for part in qualname.split("."):
        found = getattr(found, part, None)
    if module is None or "<" in qualname or found is not function:
        raise ValueError(
            f"The function '{qualname}' of the observable '{name}' is no function "
            f"of a module but a lambda or a closure; define it at the top level "
            f"of a module, so that it pickles for the workers of a scan."
        )


@dataclass(frozen=True)
class Observable:
    """An observable, the base of `Formula`, `PK` and `Custom`.

    Attributes:
        id: the id, which the result and the other observables use.
    """

    id: str

    @property
    def reads(self) -> tuple[str, ...]:
        """Get the selections and the observables it reads."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Get the definition as JSON types, the provenance of a result."""
        raise NotImplementedError


@dataclass(frozen=True)
class Formula(Observable):
    """A formula of the math of PEtab with reductions over time, see the module.

    Attributes:
        formula: the formula over selections and the ids of other observables.
        unit: the unit of the values: the unit the formula has is converted
            into it; where pint cannot derive one, e.g. for `piecewise` or a
            comparison, it is the unit of the formula, which then needs it.
    """

    formula: str
    unit: str | None = None

    def __post_init__(self) -> None:
        """Check the definition.

        Raises:
            ValueError: if the id, the formula or the unit is not valid.
        """
        _check_id(self.id, "Formula")
        if not isinstance(self.formula, str) or not self.formula.strip():
            raise ValueError(
                f"The formula of the observable '{self.id}' is a non-empty string, "
                f"not {self.formula!r}."
            )
        if self.unit is not None:
            _check_unit(self.unit, self.id)
        # the simulator package imports the definitions
        from sbmlsim.simulator.formula import reduce_formula

        reduce_formula(self.formula)

    @property
    @override
    def reads(self) -> tuple[str, ...]:
        """Get the selections and the observables the formula reads."""
        from sbmlsim.simulator.formula import reduce_formula

        return reduce_formula(self.formula).symbols

    @override
    def to_dict(self) -> dict[str, Any]:
        """Get the definition as JSON types."""
        return {"type": "Formula", "id": self.id, "formula": self.formula, "unit": self.unit}


@dataclass(frozen=True)
class PK(Observable):
    """The non-compartmental analysis of a timecourse with pkpdutils.

    Every parameter pkpdutils derives is a value per simulation
    `<id>.<parameter>`, e.g. `hctz.cmax`, `hctz.auc_inf_obs`, `hctz.thalf`,
    with the unit pkpdutils gives it, and `<id>.flags` the flags of the
    analysis (`pkpdutils.NCAFlag`).

    Attributes:
        selection: the timecourse, a selection of the model or the id of a
            timecourse observable, e.g. a concentration.
        dose: the target of the model which the simulations dose: its values
            in the plan of every point are the doses and the times of these
            values their times, so the dose of a dimension and the changes of
            a multiple dosing are found; a quantity is a fixed dose at the
            start; `None` analyses without a dose and leaves out the
            parameters which need one.
        route: the route of the dose, a `pkpdutils.Route` or its name, e.g.
            `"oral"` or `"iv_bolus"`; a dose needs it.
        options: the options of the analysis, a `pkpdutils.NCAOptions`.
        parameters: the parameters to keep, every parameter by default.
    """

    selection: str
    dose: str | Quantity | None = field(default=None, kw_only=True)
    route: str | None = field(default=None, kw_only=True)
    options: NCAOptions | None = field(default=None, kw_only=True)
    parameters: tuple[str, ...] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Check the definition.

        Raises:
            TypeError: if the dose, the route or the parameters have the wrong type.
            ValueError: if the id or the selection is not valid, a fixed dose is
                no single amount, a dose has no route or a parameter repeats.
        """
        _check_id(self.id, "PK")
        if not isinstance(self.selection, str) or not self.selection:
            raise ValueError(
                f"The timecourse of the observable '{self.id}' is a selection or "
                f"an observable, not {self.selection!r}."
            )
        if self.dose is not None:
            if isinstance(self.dose, Quantity):
                if np.ndim(self.dose.magnitude) != 0:
                    raise ValueError(
                        f"The fixed dose of the observable '{self.id}' is one "
                        f"amount, not {self.dose}."
                    )
            elif not isinstance(self.dose, str) or not self.dose:
                raise TypeError(
                    f"The dose of the observable '{self.id}' is a target of the "
                    f"model or a quantity, not {self.dose!r}."
                )
            if self.route is None:
                raise ValueError(
                    f"The dose of the observable '{self.id}' needs its route, e.g. "
                    f"route='oral'."
                )
        if self.route is not None and not isinstance(self.route, str):
            raise TypeError(
                f"The route of the observable '{self.id}' is a name of a route of "
                f"pkpdutils, not {self.route!r}."
            )
        if self.parameters is not None:
            parameters = _check_names(self.parameters, "parameters", self.id)
            if not parameters:
                raise ValueError(f"The observable '{self.id}' keeps no parameters.")
            object.__setattr__(self, "parameters", parameters)

    @property
    @override
    def reads(self) -> tuple[str, ...]:
        """Get the timecourse the analysis reads."""
        return (self.selection,)

    @override
    def to_dict(self) -> dict[str, Any]:
        """Get the definition as JSON types."""
        dose: Any = self.dose
        if isinstance(self.dose, Quantity):
            dose = {"value": float(self.dose.magnitude), "unit": str(self.dose.units)}
        return {
            "type": "PK",
            "id": self.id,
            "selection": self.selection,
            "dose": dose,
            "route": None if self.route is None else str(self.route),
            "options": None
            if self.options is None
            else self.options.model_dump(mode="json"),
            "parameters": None if self.parameters is None else list(self.parameters),
        }


@dataclass(frozen=True)
class Custom(Observable):
    """A function of a module, called once per simulation.

    `function(time, values)` gets the time points of a simulation, without
    padding, and `values`, its symbols to their values: an array of the length
    of `time` for a timecourse and a float for a value per simulation. It
    returns a float for a value per simulation and an array of the length of
    `time` for a timecourse.

    Attributes:
        function: the function, defined at the top level of a module so that
            it pickles for the workers of a scan.
        unit: the unit of its values.
        symbols: the selections and observables it reads.
        kind: a value per simulation (`SCALAR`) or a timecourse.
    """

    function: Callable[[np.ndarray, dict[str, Any]], Any]
    unit: str
    symbols: tuple[str, ...] = field(kw_only=True)
    kind: ObservableKind = field(default=ObservableKind.SCALAR, kw_only=True)

    def __post_init__(self) -> None:
        """Check the definition.

        Raises:
            TypeError: if the function is not callable or the symbols are a string.
            ValueError: if the id, the function, the unit or the symbols are not
                valid.
        """
        _check_id(self.id, "Custom")
        _check_function(self.function, self.id)
        if not isinstance(self.unit, str):
            raise TypeError(
                f"The unit of the observable '{self.id}' is a string, not {self.unit!r}."
            )
        _check_unit(self.unit, self.id)
        object.__setattr__(
            self, "symbols", _check_names(self.symbols, "symbols", self.id)
        )
        object.__setattr__(self, "kind", ObservableKind(self.kind))

    @property
    @override
    def reads(self) -> tuple[str, ...]:
        """Get the symbols the function reads."""
        return self.symbols

    @override
    def to_dict(self) -> dict[str, Any]:
        """Get the definition as JSON types; the function as `module:name`."""
        function = f"{self.function.__module__}:{self.function.__qualname__}"
        return {
            "type": "Custom",
            "id": self.id,
            "function": function,
            "unit": self.unit,
            "symbols": list(self.symbols),
            "kind": str(self.kind),
        }
```

In `src/sbmlsim/simulation/__init__.py` add `from .observables import PK, Custom, Formula, Observable, ObservableKind` (after the import of `.scan`) and the five names to `__all__` (sorted).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/simulation/test_observables.py`
Expected: PASS. If `test_the_definitions_pickle_and_serialize` reports the unit of the fixed dose in another spelling than `"milligram"`, the string is what `str(Q(10.0, "mg").units)` gives in the registry of sbmlsim; assert that string.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add pyproject.toml src/sbmlsim/simulation/observables.py src/sbmlsim/simulation/__init__.py tests/simulator/models.py tests/simulation/test_observables.py
git commit -m "Formula, PK and Custom define what a scan computes from every simulation" -m "The definitions of the observables are frozen and pickle, they check their ids, formulas, units, doses and functions when they are created: a Custom refuses a lambda or a closure, which would not pickle for the workers. pkpdutils is a dependency again for the PK observable and is imported only when one is compiled."
```

---

### Task 3: The PK analysis of a chunk with pkpdutils

**Files:**
- Create: `src/sbmlsim/simulator/pk.py`
- Modify: `src/sbmlsim/result/scan.py:46-57` (the constant `FLAGS`)
- Modify: `tests/simulator/models.py` (`PK_MODEL`, `sbml_pk`)
- Test: `tests/simulator/test_pk.py` (new)

**Interfaces:**
- Consumes: `PK` (Task 2); `Plan` with `start`, `end`, `time_shift`, `preinit: tuple[Assignment, ...]`, `events: tuple[PlanEvent, ...]` (`PlanEvent.time`, `.assignments`), `Assignment.target`, `.value: float | None`, `.formula: str | None` (phase 1, `simulator/plan.py`); `Simulator().load(model)` and `Simulator().compile(loaded, simulation) -> Plan`.
- Produces:
  - `sbmlsim.result.scan.FLAGS = "flags"`.
  - `DoseSpec(target: str | None, amount: float, unit: str)`, frozen.
  - `PKNode(id: str, selection: str, time_unit: str, unit: str, dose: DoseSpec | None, route: str | None, options: Any, parameters: tuple[str, ...])`, frozen, with `output(parameter: str) -> str` (`"<id>.<parameter>"`) and the property `outputs: tuple[str, ...]`.
  - `doses_of(plan: Plan, dose: DoseSpec) -> tuple[np.ndarray, np.ndarray]` (times, amounts).
  - `compile_pk(observable: PK, *, unit: str, time_unit: str, dose_unit: str | None, plans: Sequence[Plan]) -> tuple[PKNode, dict[str, str]]` (the node and output id -> unit).
  - `evaluate_pk(node: PKNode, time: np.ndarray, values: np.ndarray, plans: Sequence[Plan]) -> dict[str, np.ndarray]` (output id -> `(n_points, 1)`).
  - `tests/simulator/models.py`: `PK_MODEL` and `sbml_pk(ke: float = 0.2) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/simulator/models.py`:

```python
#: a one-compartment model with a first-order absorption from the depot
#: `PODOSE`: for a dose D at 0, C(t) = D ka / (V (ka - ke)) (exp(-ke t) - exp(-ka t))
PK_MODEL = """
model onecomp
  compartment V = 10
  species C in V = 0
  ka = 1; ke = {ke}
  PODOSE = 0
  PODOSE' = -ka*PODOSE
  absorption: -> C; ka*PODOSE
  elimination: C -> ; ke*C*V
end
"""


def sbml_pk(ke: float = 0.2) -> str:
    """Get the one-compartment model in hours, mg and litres."""
    import libsbml

    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml(PK_MODEL.format(ke=ke)))
    model: libsbml.Model = doc.getModel()
    for uid, kind, scale, multiplier, exponent in (
        ("hr", libsbml.UNIT_KIND_SECOND, 0, 3600.0, 1),
        ("mg", libsbml.UNIT_KIND_GRAM, -3, 1.0, 1),
        ("per_hr", libsbml.UNIT_KIND_SECOND, 0, 3600.0, -1),
    ):
        definition = model.createUnitDefinition()
        definition.setId(uid)
        unit = definition.createUnit()
        unit.setKind(kind)
        unit.setScale(scale)
        unit.setMultiplier(multiplier)
        unit.setExponent(exponent)
    model.setTimeUnits("hr")
    model.setSubstanceUnits("mg")
    model.setExtentUnits("mg")
    model.setVolumeUnits("litre")
    model.getCompartment("V").setUnits("litre")
    model.getSpecies("C").setSubstanceUnits("mg")
    for pid, uid in (("ka", "per_hr"), ("ke", "per_hr"), ("PODOSE", "mg")):
        model.getParameter(pid).setUnits(uid)
    return libsbml.writeSBMLToString(doc)
```

Create `tests/simulator/test_pk.py`:

```python
"""The PK analysis of a chunk with pkpdutils."""

import numpy as np
import pkpdutils as pk
import pytest

from sbmlsim import Q
from sbmlsim.simulation import PK, Change, Simulation, SteadyState
from sbmlsim.simulator import Simulator
from sbmlsim.simulator.pk import DoseSpec, compile_pk, doses_of, evaluate_pk
from sbmlsim.units import ureg
from tests.simulator.models import sbml_pk

KA, KE, V = 1.0, 0.2, 10.0
DOSE = DoseSpec(target="PODOSE", amount=0.0, unit="mg")


def concentration(t: np.ndarray, dose: float, at: float = 0.0) -> np.ndarray:
    tau = np.clip(t - at, 0.0, None)
    return np.where(
        t >= at,
        dose * KA / (V * (KA - KE)) * (np.exp(-KE * tau) - np.exp(-KA * tau)),
        0.0,
    )


@pytest.fixture(scope="module")
def plans() -> dict[str, object]:
    simulator = Simulator()
    model = simulator.load(sbml_pk())

    def plan(simulation: Simulation) -> object:
        return simulator.compile(model, simulation)

    return {
        "single": plan(Simulation(end=48, changes=[Change(0, {"PODOSE": Q(100, "mg")})])),
        "late": plan(Simulation(end=48, changes=[Change(10, {"PODOSE": 50.0})])),
        "multiple": plan(
            Simulation(end=48, changes=[Change([0, 24], {"PODOSE": Q(100, "mg")})])
        ),
        "preinit": plan(
            Simulation(
                end=48,
                preinit_changes={"PODOSE": 5.0},
                changes=[Change(0, {"PODOSE": 100.0})],
            )
        ),
        "shifted": plan(
            Simulation(end=48, time_shift=-12, changes=[Change(0, {"PODOSE": 100.0})])
        ),
        "formula": plan(Simulation(end=48, changes=[Change(0, {"PODOSE": "2 * ka"})])),
        "none": plan(Simulation(end=48)),
        "presimulation": plan(
            Simulation(
                end=48,
                presimulation=SteadyState(preinit_changes={"PODOSE": 3.0}),
                changes=[Change(0, {"PODOSE": 100.0})],
            )
        ),
    }


def test_the_doses_are_the_values_the_plan_assigns(plans: dict) -> None:
    times, amounts = doses_of(plans["single"], DOSE)
    assert times.tolist() == [0.0] and amounts.tolist() == [100.0]
    times, amounts = doses_of(plans["late"], DOSE)
    assert times.tolist() == [10.0] and amounts.tolist() == [50.0]
    times, amounts = doses_of(plans["multiple"], DOSE)
    assert times.tolist() == [0.0, 24.0] and amounts.tolist() == [100.0, 100.0]


def test_a_change_at_the_start_replaces_the_value_before_the_initialization(
    plans: dict,
) -> None:
    times, amounts = doses_of(plans["preinit"], DOSE)
    assert times.tolist() == [0.0] and amounts.tolist() == [100.0]


def test_the_dose_times_are_shifted_like_the_result(plans: dict) -> None:
    times, _ = doses_of(plans["shifted"], DOSE)
    assert times.tolist() == [-12.0]


def test_the_values_of_a_presimulation_are_no_doses(plans: dict) -> None:
    times, amounts = doses_of(plans["presimulation"], DOSE)
    assert times.tolist() == [0.0] and amounts.tolist() == [100.0]


def test_no_dose_and_a_fixed_dose(plans: dict) -> None:
    times, amounts = doses_of(plans["none"], DOSE)
    assert times.size == 0 and amounts.size == 0
    fixed = DoseSpec(target=None, amount=7.0, unit="mg")
    times, amounts = doses_of(plans["none"], fixed)
    assert times.tolist() == [0.0] and amounts.tolist() == [7.0]


def test_a_formula_dose_raises(plans: dict) -> None:
    with pytest.raises(ValueError, match="formula"):
        doses_of(plans["formula"], DOSE)


def _compile(observable: PK, *plan_list: object) -> tuple:
    dose_unit = "mg" if isinstance(observable.dose, str) else None
    return compile_pk(
        observable, unit="mg/l", time_unit="hr", dose_unit=dose_unit, plans=plan_list
    )


def test_the_parameters_and_their_units_come_from_pkpdutils(plans: dict) -> None:
    node, units = _compile(PK("c", "[C]", dose="PODOSE", route="oral"), plans["single"])
    for parameter in ("cmax", "tmax", "auc_inf_obs", "thalf", "cl_f", "flags"):
        assert f"c.{parameter}" in units
    assert ureg.Quantity(1.0, units["c.cmax"]).to("mg/l").magnitude == pytest.approx(1.0)
    assert ureg.Quantity(1.0, units["c.tmax"]).to("hr").magnitude == pytest.approx(1.0)
    assert node.outputs == tuple(units)


def test_a_multiple_dosing_adds_the_parameters_of_its_interval(plans: dict) -> None:
    _, single = _compile(PK("c", "[C]", dose="PODOSE", route="oral"), plans["single"])
    _, both = _compile(
        PK("c", "[C]", dose="PODOSE", route="oral"), plans["single"], plans["multiple"]
    )
    assert "c.tau" not in single
    assert "c.tau" in both and set(single) < set(both)


def test_without_a_dose_the_parameters_of_the_dose_are_left_out(plans: dict) -> None:
    _, units = _compile(PK("c", "[C]"), plans["single"])
    assert "c.cmax" in units and "c.cl_f" not in units


def test_the_parameters_to_keep_and_the_flags(plans: dict) -> None:
    node, units = _compile(
        PK("c", "[C]", dose="PODOSE", route="oral", parameters=["cmax"]), plans["single"]
    )
    assert node.parameters == ("cmax", "flags")
    assert list(units) == ["c.cmax", "c.flags"]
    with pytest.raises(ValueError, match="nope"):
        _compile(
            PK("c", "[C]", dose="PODOSE", route="oral", parameters=["nope"]),
            plans["single"],
        )


@pytest.mark.parametrize(
    ("route", "match"), [("intranasal", "route"), ("iv_infusion", "infusion")]
)
def test_a_route_pkpdutils_cannot_take_raises(plans: dict, route: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _compile(PK("c", "[C]", dose="PODOSE", route=route), plans["single"])


def test_a_timecourse_or_a_dose_without_a_unit_raises(plans: dict) -> None:
    with pytest.raises(ValueError, match="unit"):
        compile_pk(PK("c", "[C]"), unit="", time_unit="hr", dose_unit=None, plans=[plans["single"]])
    with pytest.raises(ValueError, match="unit"):
        compile_pk(
            PK("c", "[C]", dose="PODOSE", route="oral"),
            unit="mg/l",
            time_unit="hr",
            dose_unit="",
            plans=[plans["single"]],
        )


def _direct(time: np.ndarray, values: np.ndarray, dose_times: np.ndarray, amounts: np.ndarray) -> pk.NCAResult:
    timecourses = pk.Timecourses.from_arrays(
        time,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("_sim",),
        dose={"amount": amounts, "time": dose_times, "unit": "mg"},
        route="oral",
    )
    return pk.nca(timecourses)


def test_the_analysis_equals_pkpdutils_on_the_same_arrays(plans: dict) -> None:
    node, _ = _compile(PK("c", "[C]", dose="PODOSE", route="oral"), plans["single"])
    t = np.linspace(0.0, 48.0, 97)
    time = np.vstack([t, t])
    values = np.vstack([concentration(t, 100.0), concentration(t, 50.0, at=10.0)])
    out = evaluate_pk(node, time, values, [plans["single"], plans["late"]])
    direct = _direct(time, values, np.array([[0.0], [10.0]]), np.array([[100.0], [50.0]]))
    for parameter in ("cmax", "tmax", "auc_inf_obs", "thalf", "cl_f", "flags"):
        np.testing.assert_allclose(
            out[f"c.{parameter}"][:, 0], np.asarray(direct.ds[parameter].values, float)
        )
    assert out["c.auc_inf_obs"][0, 0] == pytest.approx(100.0 / (V * KE), rel=1e-2)
    assert out["c.thalf"][0, 0] == pytest.approx(np.log(2.0) / KE, rel=1e-2)


def test_ragged_rows_are_analysed_without_their_padding(plans: dict) -> None:
    node, _ = _compile(PK("c", "[C]", dose="PODOSE", route="oral"), plans["single"])
    t = np.linspace(0.0, 48.0, 97)
    short = np.full(97, np.nan)
    short[:65] = t[:65]
    time = np.vstack([t, short])
    values = concentration(np.where(np.isfinite(time), time, 0.0), 100.0)
    values[1, 65:] = np.nan
    out = evaluate_pk(node, time, values, [plans["single"], plans["single"]])
    alone = evaluate_pk(node, short[None, :65], values[1:, :65], [plans["single"]])
    np.testing.assert_allclose(out["c.cmax"][1], alone["c.cmax"][0])
    np.testing.assert_allclose(out["c.auc_last"][1], alone["c.auc_last"][0])


def test_points_with_different_dosings_are_analysed_on_their_own(plans: dict) -> None:
    node, _ = _compile(
        PK("c", "[C]", dose="PODOSE", route="oral"), plans["single"], plans["multiple"]
    )
    t = np.linspace(0.0, 48.0, 97)
    single = concentration(t, 100.0)
    multiple = single + concentration(t, 100.0, at=24.0)
    out = evaluate_pk(node, np.vstack([t, t]), np.vstack([single, multiple]), [plans["single"], plans["multiple"]])
    direct = _direct(t[None, :], multiple[None, :], np.array([[0.0, 24.0]]), np.array([[100.0, 100.0]]))
    assert out["c.tau"][1, 0] == pytest.approx(float(direct.ds["tau"].values[0]))
    assert np.isnan(out["c.tau"][0, 0])
    assert out["c.cmax"][0, 0] == pytest.approx(single.max(), rel=1e-12)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/simulator/test_pk.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbmlsim.simulator.pk'`.

- [ ] **Step 3: Write the implementation**

In `src/sbmlsim/result/scan.py`, after `STATUS = "status"` add:

```python
#: the variable of the flags of the analysis of a PK observable, `<id>.flags`
FLAGS = "flags"
```

Create `src/sbmlsim/simulator/pk.py`:

```python
"""The non-compartmental analysis of a `PK` observable, with pkpdutils.

A PK observable analyses one timecourse of every simulation of a chunk with
`pkpdutils.nca`: the timecourses are handed over as `Timecourses.from_arrays`
with the time points of every simulation, padded with `NaN`, and the doses
the plan of every point assigns to the dose target. Every parameter pkpdutils
derives is a value per simulation `<id>.<parameter>`, and `<id>.flags` the
flags of the analysis (`pkpdutils.NCAFlag`).

Which parameters pkpdutils derives depends on the dosing (a multiple dosing
adds the parameters of the dosing interval, a dose the clearance and the
volume) and on the options. `compile_pk` finds them and their units by an
analysis of a synthetic timecourse with the dosing of every plan of a run,
so every point of the result has the same variables; the points of a chunk
are analysed in groups of the same number of doses.

pkpdutils is imported when a PK observable is compiled or evaluated, not on
the import of sbmlsim: it imports pandas, scipy and xarray, about a second.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sbmlsim.result.scan import FLAGS
from sbmlsim.simulation.observables import PK
from sbmlsim.simulator.plan import Plan

#: the time points of the synthetic timecourse which finds the parameters
_PROBE_POINTS = 97


@dataclass(frozen=True)
class DoseSpec:
    """Where the doses of a PK observable come from.

    Attributes:
        target: the target of the model whose values the plans assign, `None`
            for a fixed dose.
        amount: the fixed dose at the start, ignored for a target.
        unit: the unit of the amounts, for a target its unit in the model.
    """

    target: str | None
    amount: float
    unit: str


@dataclass(frozen=True)
class PKNode:
    """A PK observable compiled against the models of a run.

    Attributes:
        id: the id of the observable.
        selection: the timecourse it analyses.
        time_unit: the unit of the time.
        unit: the unit of the timecourse.
        dose: where its doses come from, `None` without doses.
        route: the route of the doses, the name of a `pkpdutils.Route`.
        options: the `pkpdutils.NCAOptions`, `None` for the defaults.
        parameters: the parameters it gives, `flags` among them.
    """

    id: str
    selection: str
    time_unit: str
    unit: str
    dose: DoseSpec | None
    route: str | None
    options: Any
    parameters: tuple[str, ...]

    def output(self, parameter: str) -> str:
        """Get the id of the value of a parameter, `<id>.<parameter>`."""
        return f"{self.id}.{parameter}"

    @property
    def outputs(self) -> tuple[str, ...]:
        """Get the ids of the values of its parameters."""
        return tuple(self.output(parameter) for parameter in self.parameters)


def doses_of(plan: Plan, dose: DoseSpec) -> tuple[np.ndarray, np.ndarray]:
    """Get the times and the amounts of the doses of the plan of a point.

    A fixed dose is one dose at the start. The doses of a target are the
    positive values the plan assigns to it: a value before the initialization
    at the start and the values of its changes at their times, where a change
    at the start replaces the value before the initialization; the values of a
    presimulation are no doses. The times are the times of the result, i.e.
    shifted by the `time_shift` of the plan.

    Args:
        plan: the plan of a point, with its values.
        dose: where the doses come from.

    Returns:
        The times and the amounts, sorted by time.

    Raises:
        ValueError: if the plan assigns a formula to the target.
    """
    if dose.target is None:
        return (
            np.array([plan.start + plan.time_shift]),
            np.array([dose.amount], dtype=float),
        )
    found: dict[float, float] = {}
    assignments = [(plan.start, a) for a in plan.preinit] + [
        (event.time, a) for event in plan.events for a in event.assignments
    ]
    for time, assignment in assignments:
        if assignment.target != dose.target:
            continue
        if assignment.value is None:
            raise ValueError(
                f"The dose '{dose.target}' is the formula '{assignment.formula}' in a "
                f"simulation; a PK observable reads the dose as a number."
            )
        found[time] = assignment.value
    doses = sorted((time, value) for time, value in found.items() if value > 0)
    times = np.array([time for time, _ in doses], dtype=float) + plan.time_shift
    amounts = np.array([value for _, value in doses], dtype=float)
    return times, amounts


def _analyse(
    *,
    time_unit: str,
    unit: str,
    dose_unit: str | None,
    route: str | None,
    options: Any,
    time: np.ndarray,
    values: np.ndarray,
    dose_times: np.ndarray,
    dose_amounts: np.ndarray,
) -> Any:
    """Run the analysis of pkpdutils on timecourses `(n, n_rows)` and doses `(n, n_doses)`."""
    import pkpdutils as pk

    dose = None
    if dose_unit is not None and dose_times.shape[-1] > 0:
        dose = {"amount": dose_amounts, "time": dose_times, "unit": dose_unit}
    timecourses = pk.Timecourses.from_arrays(
        time,
        values,
        time_unit=time_unit,
        unit=unit,
        dims=("_sim",),
        dose=dose,
        route=route if dose is not None else None,
    )
    return pk.nca(timecourses, options=options)


def _probe(plan: Plan, times: np.ndarray, route: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Get a timecourse of a one-compartment model dosed at the times, over a plan."""
    start, end = plan.start + plan.time_shift, plan.end + plan.time_shift
    t = np.linspace(start, end, _PROBE_POINTS)
    ke = 8.0 / (end - start)
    ka = 4.0 * ke
    c = np.zeros_like(t)
    for at in times if times.size else np.array([start]):
        tau = np.clip(t - at, 0.0, None)
        if route == "iv_bolus":
            shape = np.exp(-ke * tau)
        else:
            shape = np.exp(-ke * tau) - np.exp(-ka * tau)
        c += np.where(t >= at, shape, 0.0)
    return t, c


def compile_pk(
    observable: PK,
    *,
    unit: str,
    time_unit: str,
    dose_unit: str | None,
    plans: Sequence[Plan],
) -> tuple[PKNode, dict[str, str]]:
    """Compile a PK observable, see the module.

    Args:
        observable: the definition.
        unit: the unit of its timecourse.
        time_unit: the unit of the time of the model.
        dose_unit: the unit of the dose target in the model; `None` for a fixed
            dose or without a dose.
        plans: the plans of the run, each with the values of a point, whose
            dosing decides the parameters.

    Returns:
        The compiled observable and the unit of each of its outputs.

    Raises:
        ValueError: if the timecourse or the dose target has no unit, the
            route is none of pkpdutils or an infusion, a plan doses by a
            formula, a simulation has no time span, there is no plan, or a
            parameter to keep is none pkpdutils derives.
    """
    import pkpdutils as pk

    if not unit:
        raise ValueError(
            f"The timecourse '{observable.selection}' of the observable "
            f"'{observable.id}' has no unit, which the analysis needs; give the "
            f"model units or analyse a Formula observable with a unit."
        )
    if not plans:
        raise ValueError(f"The observable '{observable.id}' has no simulation to analyse.")
    route = None
    if observable.route is not None:
        try:
            route = str(pk.Route(observable.route))
        except ValueError as err:
            raise ValueError(
                f"The route '{observable.route}' of the observable '{observable.id}' "
                f"is none of pkpdutils: {[str(r) for r in pk.Route]}."
            ) from err
        if route == str(pk.Route.IV_INFUSION):
            raise ValueError(
                f"The observable '{observable.id}' has the route of an infusion, "
                f"whose duration the changes of a simulation do not carry; analyse "
                f"it as 'iv_bolus' or 'oral'."
            )
    dose: DoseSpec | None = None
    if isinstance(observable.dose, str):
        if not dose_unit:
            raise ValueError(
                f"The dose '{observable.dose}' of the observable '{observable.id}' "
                f"has no unit in the model."
            )
        dose = DoseSpec(target=observable.dose, amount=0.0, unit=dose_unit)
    elif observable.dose is not None:
        dose = DoseSpec(
            target=None,
            amount=float(observable.dose.magnitude),
            unit=str(observable.dose.units),
        )
    units: dict[str, str] = {}
    for plan in plans:
        if plan.end <= plan.start:
            raise ValueError(
                f"The observable '{observable.id}' analyses a simulation without a "
                f"time span, [{plan.start}, {plan.end}]."
            )
        times, amounts = (
            (np.empty(0), np.empty(0)) if dose is None else doses_of(plan, dose)
        )
        t, c = _probe(plan, times, route)
        result = _analyse(
            time_unit=time_unit,
            unit=unit,
            dose_unit=None if dose is None else dose.unit,
            route=route,
            options=observable.options,
            time=t[None, :],
            values=c[None, :],
            dose_times=times[None, :],
            dose_amounts=amounts[None, :],
        )
        for parameter in (*result.parameters, FLAGS):
            units.setdefault(parameter, result.units(parameter))
    parameters = tuple(units)
    if observable.parameters is not None:
        unknown = [p for p in observable.parameters if p not in units]
        if unknown:
            raise ValueError(
                f"The observable '{observable.id}' keeps the parameters {unknown}, "
                f"which pkpdutils does not derive for its dosing: {sorted(units)}."
            )
        parameters = tuple(dict.fromkeys((*observable.parameters, FLAGS)))
    node = PKNode(
        id=observable.id,
        selection=observable.selection,
        time_unit=time_unit,
        unit=unit,
        dose=dose,
        route=route,
        options=observable.options,
        parameters=parameters,
    )
    return node, {node.output(p): units[p] for p in parameters}


def evaluate_pk(
    node: PKNode, time: np.ndarray, values: np.ndarray, plans: Sequence[Plan]
) -> dict[str, np.ndarray]:
    """Analyse the timecourses of the points of a chunk.

    Args:
        node: the observable.
        time: the time points `(n_points, n_rows)`, padded with `NaN`.
        values: the timecourse, of the shape of `time`.
        plans: the plan of every point, with its values.

    Returns:
        The output id of every parameter -> its values `(n_points, 1)`, `NaN`
        where pkpdutils gives no value for a point.
    """
    finite = np.isfinite(time)
    time = np.where(finite, time, np.nan)
    values = np.where(finite, values, np.nan)
    n = time.shape[0]
    out = {output: np.full((n, 1), np.nan) for output in node.outputs}
    empty = (np.empty(0), np.empty(0))
    doses = [empty if node.dose is None else doses_of(p, node.dose) for p in plans]
    counts = np.array([times.size for times, _ in doses], dtype=int)
    for count in np.unique(counts):
        rows = np.flatnonzero(counts == count)
        result = _analyse(
            time_unit=node.time_unit,
            unit=node.unit,
            dose_unit=None if node.dose is None else node.dose.unit,
            route=node.route,
            options=node.options,
            time=time[rows],
            values=values[rows],
            dose_times=np.array([doses[r][0] for r in rows]).reshape(rows.size, count),
            dose_amounts=np.array([doses[r][1] for r in rows]).reshape(rows.size, count),
        )
        for parameter in node.parameters:
            if parameter in result.ds:
                out[node.output(parameter)][rows, 0] = np.asarray(
                    result.ds[parameter].values, dtype=float
                )
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/simulator/test_pk.py`
Expected: PASS. If pkpdutils warns (the suite turns warnings into errors), find the cause in its message and fix the input (e.g. a time point which is not finite), never filter the warning.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/simulator/pk.py src/sbmlsim/result/scan.py tests/simulator/models.py tests/simulator/test_pk.py
git commit -m "A PK observable analyses the timecourses of a chunk with pkpdutils" -m "The doses are the values the plan of every point assigns to the dose target, at their times and shifted like the result, so a dose of a dimension and a multiple dosing need no further input. The parameters and their units are found when a scan is compiled, by an analysis of a synthetic timecourse with the dosing of every plan, so every point has the same variables; the points of a chunk are analysed in groups of the same number of doses."
```

---

### Task 4: The observable graph: compile and evaluate

**Files:**
- Create: `src/sbmlsim/simulator/observables.py`
- Modify: `src/sbmlsim/model/model_roadrunner.py` (`has_selection`, next to `set_selections` at line 470)
- Modify: `src/sbmlsim/fit/optimization.py:144-150,167,198` (use `model.has_selection`, delete `_is_selection`)
- Test: `tests/simulator/test_observables.py` (new)

**Interfaces:**
- Consumes: `evaluate_reduced`, `reduce_formula` (Task 1); `Formula`, `PK`, `Custom`, `Observable`, `ObservableKind` (Task 2); `PKNode`, `compile_pk`, `evaluate_pk` (Task 3); `RoadrunnerSBMLModel.uinfo`, `.selections`, `.r_loaded`.
- Produces:
  - `RoadrunnerSBMLModel.has_selection(selection: str) -> bool`.
  - `FormulaNode(id: str, formula: str, kind: ObservableKind, factor: float)`, `CustomNode(id: str, function: Callable, kind: ObservableKind, symbols: tuple[str, ...])`, `Node = FormulaNode | CustomNode | PKNode`.
  - `ObservableError(row: int | None, message: str)` (`RuntimeError`, pickles).
  - `ObservableGraph(nodes: tuple[Node, ...], selections: tuple[str, ...], kinds: dict[str, ObservableKind], units: dict[str, str], keep: tuple[str, ...], outputs: tuple[str, ...], doses: tuple[str, ...] = ())`, frozen, with properties `timecourses`, `scalars` (the kept ids by kind) and `evaluate(time, columns, plans) -> dict[str, np.ndarray]`.
  - `identity_graph(selections: Sequence[str], units: Mapping[str, str] | None = None, keep: Sequence[str] | None = None) -> ObservableGraph`.
  - `compile_observables(observables: Sequence[Observable] | None, model: RoadrunnerSBMLModel, *, keep: Sequence[str] | None = None, plans: Sequence[Plan] = ()) -> ObservableGraph`.

- [ ] **Step 1: Write the failing tests**

Create `tests/simulator/test_observables.py`:

```python
"""The observables of a run, compiled against its model and evaluated."""

import pickle

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import PK, Change, Custom, Formula, ObservableKind, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.simulator.observables import (
    ObservableError,
    compile_observables,
    identity_graph,
)
from sbmlsim.units import ureg
from tests.simulator.models import auc_of_c, doubled, sbml, sbml_pk

SCALAR, TIMECOURSE = ObservableKind.SCALAR, ObservableKind.TIMECOURSE


@pytest.fixture(scope="module")
def pk_model() -> RoadrunnerSBMLModel:
    return Simulator().load(sbml_pk())


@pytest.fixture(scope="module")
def plan(pk_model: RoadrunnerSBMLModel) -> object:
    simulation = Simulation(end=48, changes=[Change(0, {"PODOSE": Q(100, "mg")})])
    return Simulator().compile(pk_model, simulation)


def _equal(unit: str, expected: str) -> bool:
    return ureg.Quantity(1.0, unit).to(expected).magnitude == pytest.approx(1.0)


def test_has_selection(pk_model: RoadrunnerSBMLModel) -> None:
    for name in ("time", "C", "[C]", "ke", "V", "absorption"):
        assert pk_model.has_selection(name)
    assert not pk_model.has_selection("nope")


def test_the_kinds_follow_what_a_formula_reads(pk_model: RoadrunnerSBMLModel) -> None:
    graph = compile_observables(
        [
            Formula("c", "[C]"),
            Formula("cmax", "max(c)"),
            Formula("rel", "c / cmax"),
            Formula("two", "2"),
            Formula("late", "at(c, 10) + cmax"),
        ],
        pk_model,
    )
    assert graph.kinds["c"] is TIMECOURSE and graph.kinds["rel"] is TIMECOURSE
    assert graph.kinds["cmax"] is SCALAR and graph.kinds["two"] is SCALAR
    assert graph.kinds["late"] is SCALAR
    assert graph.timecourses == ("c", "rel") and graph.scalars == ("cmax", "two", "late")
    assert graph.selections == ("[C]",)


def test_the_units_are_derived(pk_model: RoadrunnerSBMLModel) -> None:
    graph = compile_observables(
        [
            Formula("c", "[C]"),
            Formula("rel", "c / max(c)"),
            Formula("exposure", "mean(c) * time"),
        ],
        pk_model,
    )
    assert _equal(graph.units["c"], "mg/l")
    assert _equal(graph.units["rel"], "dimensionless")
    assert _equal(graph.units["exposure"], "mg*hr/l")


def test_a_declared_unit_converts_the_values(pk_model: RoadrunnerSBMLModel) -> None:
    graph = compile_observables([Formula("c", "[C]", unit="ng/ml")], pk_model)
    (node,) = graph.nodes
    assert graph.units["c"] == "ng/ml"
    assert node.factor == pytest.approx(1000.0)
    with pytest.raises(ValueError, match="cannot be converted"):
        compile_observables([Formula("c", "[C]", unit="hr")], pk_model)


def test_a_unit_which_cannot_be_derived_is_declared(pk_model: RoadrunnerSBMLModel) -> None:
    with pytest.raises(ValueError, match="unit="):
        compile_observables([Formula("high", "piecewise(1, [C] > 2, 0)")], pk_model)
    graph = compile_observables(
        [Formula("high", "piecewise(1, [C] > 2, 0)", unit="dimensionless")], pk_model
    )
    assert graph.units["high"] == "dimensionless"


def test_the_symbols_are_checked(pk_model: RoadrunnerSBMLModel) -> None:
    with pytest.raises(ValueError, match="'nope'"):
        compile_observables([Formula("x", "nope * 2")], pk_model)
    with pytest.raises(ValueError, match="selections of the model"):
        compile_observables([Formula("ke", "2")], pk_model)
    with pytest.raises(ValueError, match="Two observables"):
        compile_observables([Formula("x", "1"), Formula("x", "2")], pk_model)


def test_a_cycle_is_reported(pk_model: RoadrunnerSBMLModel) -> None:
    with pytest.raises(ValueError, match="cycle: a -> b -> a"):
        compile_observables([Formula("a", "b + 1"), Formula("b", "a * 2")], pk_model)
    with pytest.raises(ValueError, match="cycle"):
        compile_observables([Formula("a", "a + 1")], pk_model)


def test_the_time_of_at_is_no_timecourse(pk_model: RoadrunnerSBMLModel) -> None:
    with pytest.raises(ValueError, match="at"):
        compile_observables([Formula("x", "at([C], [C])")], pk_model)


def test_keep_and_the_observables_it_needs(pk_model: RoadrunnerSBMLModel) -> None:
    observables = [
        Formula("c", "[C]"),
        Formula("cmax", "max(c)"),
        Formula("rel", "c / cmax"),
        Formula("unused", "ke * 2"),
    ]
    graph = compile_observables(observables, pk_model, keep=["rel"])
    assert graph.keep == ("rel",)
    assert [node.id for node in graph.nodes] == ["c", "cmax", "rel"]
    assert graph.selections == ("[C]",)
    with pytest.raises(ValueError, match="'nope'"):
        compile_observables(observables, pk_model, keep=["nope"])
    with pytest.raises(TypeError):
        compile_observables(observables, pk_model, keep="rel")


def test_the_id_of_a_pk_observable_keeps_its_parameters(
    pk_model: RoadrunnerSBMLModel, plan: object
) -> None:
    observables = [
        PK("p", "[C]", dose="PODOSE", route="oral"),
        Formula("ratio", "p.auc_inf_obs / p.cmax"),
    ]
    graph = compile_observables(observables, pk_model, keep=["p"], plans=[plan])
    assert "p.cmax" in graph.keep and "p.flags" in graph.keep
    assert "ratio" not in graph.keep
    assert graph.doses == ("PODOSE",)
    full = compile_observables(observables, pk_model, plans=[plan])
    assert _equal(full.units["ratio"], "hr")
    with pytest.raises(ValueError, match="p.nope"):
        compile_observables(
            [*observables, Formula("x", "p.nope")], pk_model, plans=[plan]
        )


def test_the_graph_of_a_run_without_observables(pk_model: RoadrunnerSBMLModel) -> None:
    graph = compile_observables(None, pk_model)
    expected = tuple(s for s in pk_model.selections or [] if s != "time")
    assert graph.selections == expected and graph.keep == expected
    assert graph.nodes == () and graph.scalars == ()
    assert identity_graph(["[C]", "ke"], keep=["ke"]).keep == ("ke",)


T = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, np.nan]])
C = np.array([[0.0, 4.0, 2.0, 1.0], [0.0, 8.0, 4.0, np.nan]])


def test_the_graph_evaluates_on_the_arrays_of_a_chunk(
    pk_model: RoadrunnerSBMLModel, plan: object
) -> None:
    graph = compile_observables(
        [
            Formula("c", "[C]", unit="ng/ml"),
            Formula("cmax", "max([C])"),
            Formula("rel", "[C] / cmax"),
            Custom("auc", auc_of_c, "mg*hr/l", symbols=["[C]"]),
            Custom("twice", doubled, "mg/l", symbols=["[C]"], kind=TIMECOURSE),
        ],
        pk_model,
    )
    out = graph.evaluate(T, {"[C]": C}, [plan, plan])
    np.testing.assert_allclose(out["c"], C * 1000.0)
    np.testing.assert_allclose(out["cmax"], [4.0, 8.0])
    np.testing.assert_allclose(out["rel"], C / np.array([[4.0], [8.0]]))
    np.testing.assert_allclose(out["auc"], [np.trapezoid(C[0], T[0]), np.trapezoid(C[1, :3], T[1, :3])])
    np.testing.assert_allclose(out["twice"][:, :3], 2.0 * C[:, :3])
    assert np.isnan(out["twice"][1, 3])


def test_a_division_by_zero_gives_inf_without_a_warning(pk_model: RoadrunnerSBMLModel) -> None:
    graph = compile_observables([Formula("x", "1 / at([C], 0)")], pk_model)
    out = graph.evaluate(T, {"[C]": C}, [])
    assert np.isinf(out["x"]).all()


def test_a_custom_which_fails_names_its_row(pk_model: RoadrunnerSBMLModel) -> None:
    graph = compile_observables(
        [Custom("auc", auc_of_c, "mg*hr/l", symbols=["[C]"])], pk_model
    )
    broken = {"[C]": np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, np.nan]])}
    out = graph.evaluate(T, broken, [])
    assert np.isfinite(out["auc"]).all()
    with pytest.raises(ObservableError) as info:
        graph.evaluate(T, {}, [])
    assert info.value.row == 0
    assert "auc" in info.value.message


def test_a_pk_observable_in_the_graph(pk_model: RoadrunnerSBMLModel, plan: object) -> None:
    graph = compile_observables(
        [PK("p", "[C]", dose="PODOSE", route="oral"), Formula("ratio", "p.auc_inf_obs / p.cmax")],
        pk_model,
        plans=[plan],
    )
    t = np.linspace(0.0, 48.0, 97)[None, :]
    c = 12.5 * (np.exp(-0.2 * t) - np.exp(-t))
    out = graph.evaluate(t, {"[C]": c}, [plan])
    assert out["ratio"][0] == pytest.approx(out["p.auc_inf_obs"][0] / out["p.cmax"][0])


def test_the_graph_pickles(pk_model: RoadrunnerSBMLModel, plan: object) -> None:
    graph = compile_observables(
        [
            Formula("c", "[C]"),
            PK("p", "[C]", dose="PODOSE", route="oral"),
            Custom("auc", auc_of_c, "mg*hr/l", symbols=["[C]"]),
        ],
        pk_model,
        plans=[plan],
    )
    again = pickle.loads(pickle.dumps(graph))
    assert again.keep == graph.keep and again.units == graph.units


def test_a_model_without_units_is_dimensionless() -> None:
    model = Simulator().load(sbml())
    graph = compile_observables([Formula("a", "[A] + 1")], model)
    assert graph.units["a"] == "dimensionless"
```

Note on `test_a_custom_which_fails_names_its_row`: `graph.evaluate(T, {}, [])` fails because the column `[C]` is missing for the first row; the error is an `ObservableError` with `row == 0`, since the function is called per row and the `KeyError` happens inside the loop. Build the arguments of the function inside the `try` of `_custom` (see Step 3) so that this holds.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/simulator/test_observables.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbmlsim.simulator.observables'`.

- [ ] **Step 3: Write the implementation**

In `src/sbmlsim/model/model_roadrunner.py`, add after `set_selections`:

```python
    def has_selection(self, selection: str) -> bool:
        """Check whether roadrunner has a selection of a name in the model.

        Args:
            selection: the name, e.g. `S`, `[S]`, a parameter or `time`.

        Returns:
            Whether the loaded model can select it.
        """
        try:
            self.r_loaded.getValue(selection)
        except RuntimeError:
            return False
        return True
```

In `src/sbmlsim/fit/optimization.py`, delete `_is_selection` and replace its two calls `_is_selection(model, symbol)` and `_is_selection(model, s)` with `model.has_selection(symbol)` and `model.has_selection(s)`.

Create `src/sbmlsim/simulator/observables.py`:

```python
"""The observables of a run, compiled against its models and evaluated.

`compile_observables` orders the observables of a run by what they read,
checks every symbol against the model, derives the kind (a timecourse or a
value per simulation) and the unit of every observable and gives an
`ObservableGraph`: frozen, picklable and free of pint, so the chunks of a scan
carry it into the workers. A run without observables has the graph of the
selections of its model, each a timecourse of its own name (`identity_graph`).

`ObservableGraph.evaluate` computes the observables on the native solutions
of the points of a chunk, stacked into arrays `(n_points, n_rows)` padded
with `NaN`: a timecourse is such an array, a value per simulation an array
`(n_points, 1)` while the graph is evaluated and `(n_points,)` in its result.
The observables are evaluated before any interpolation onto a grid, so the
reductions and the analysis of a PK observable are as exact as the output of
the simulation. An observable which no kept observable needs is not
evaluated, and the selections only it reads are not selected.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sbmlsim.model.model_roadrunner import RoadrunnerSBMLModel
from sbmlsim.result.scan import TIME
from sbmlsim.simulation.observables import (
    PK,
    Custom,
    Formula,
    Observable,
    ObservableKind,
)
from sbmlsim.simulator.formula import evaluate_reduced, reduce_formula
from sbmlsim.simulator.pk import PKNode, compile_pk, evaluate_pk
from sbmlsim.simulator.plan import Plan
from sbmlsim.units import Quantity, ureg

SCALAR = ObservableKind.SCALAR
TIMECOURSE = ObservableKind.TIMECOURSE


@dataclass(frozen=True)
class FormulaNode:
    """A Formula observable compiled against the models of a run.

    Attributes:
        id: the id of the observable.
        formula: the formula.
        kind: a timecourse or a value per simulation.
        factor: the factor into the unit of the observable, `1` without a
            conversion.
    """

    id: str
    formula: str
    kind: ObservableKind
    factor: float


@dataclass(frozen=True)
class CustomNode:
    """A Custom observable compiled against the models of a run.

    Attributes:
        id: the id of the observable.
        function: the function of a module.
        kind: a timecourse or a value per simulation.
        symbols: what it reads.
    """

    id: str
    function: Callable[[np.ndarray, dict[str, Any]], Any]
    kind: ObservableKind
    symbols: tuple[str, ...]


#: an observable compiled against the models of a run
Node = FormulaNode | CustomNode | PKNode


class ObservableError(RuntimeError):
    """An observable failed on the points of a chunk.

    Attributes:
        row: the row of the point which failed in the arrays of the chunk,
            `None` for every point.
        message: the error.
    """

    def __init__(self, row: int | None, message: str) -> None:
        """Create the error, the arguments pickle."""
        super().__init__(row, message)
        self.row = row
        self.message = message

    def __str__(self) -> str:
        """Get the message."""
        return self.message


@dataclass(frozen=True)
class ObservableGraph:
    """The observables of a run, see the module.

    Attributes:
        nodes: the observables the kept ones need, in the order of what they
            read.
        selections: the selections of roadrunner the nodes read, without
            `time`.
        kinds: the kind of every symbol and output: `time` and the selections
            are timecourses.
        units: the unit of every symbol and output.
        keep: the outputs of the result, in order.
        outputs: every output: the ids of the formulas and customs and the
            parameters of the PK observables; the selections without
            observables.
        doses: the dose targets of the PK observables the run evaluates,
            whose units the models of a run share.
    """

    nodes: tuple[Node, ...]
    selections: tuple[str, ...]
    kinds: dict[str, ObservableKind]
    units: dict[str, str]
    keep: tuple[str, ...]
    outputs: tuple[str, ...]
    doses: tuple[str, ...] = ()

    @property
    def timecourses(self) -> tuple[str, ...]:
        """Get the kept timecourses."""
        return tuple(k for k in self.keep if self.kinds[k] is TIMECOURSE)

    @property
    def scalars(self) -> tuple[str, ...]:
        """Get the kept values per simulation."""
        return tuple(k for k in self.keep if self.kinds[k] is SCALAR)

    def evaluate(
        self,
        time: np.ndarray,
        columns: Mapping[str, np.ndarray],
        plans: Sequence[Plan],
    ) -> dict[str, np.ndarray]:
        """Evaluate the observables on the points of a chunk, see the module.

        Args:
            time: the time points `(n_points, n_rows)`, padded with `NaN`.
            columns: the selections, arrays of the shape of `time`.
            plans: the plan of every point, with its values, which gives the
                doses of the PK observables.

        Returns:
            The kept outputs: a timecourse `(n_points, n_rows)`, a value per
            simulation `(n_points,)`.

        Raises:
            ObservableError: if an observable fails for a point or for all.
        """
        values: dict[str, np.ndarray] = {TIME: time, **columns}
        for node in self.nodes:
            if isinstance(node, FormulaNode):
                values[node.id] = _formula(node, time, values)
            elif isinstance(node, CustomNode):
                values[node.id] = _custom(node, time, values, self.kinds)
            else:
                try:
                    values.update(evaluate_pk(node, time, values[node.selection], plans))
                except Exception as err:
                    raise ObservableError(
                        None,
                        f"The PK observable '{node.id}' failed: "
                        f"{type(err).__name__}: {err}",
                    ) from err
        return {
            key: values[key][:, 0] if self.kinds[key] is SCALAR else values[key]
            for key in self.keep
        }


def _formula(
    node: FormulaNode, time: np.ndarray, values: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Evaluate a formula; a floating point error gives `inf` or `NaN`.

    Raises:
        ObservableError: if the formula fails, for every point.
    """
    try:
        with np.errstate(all="ignore"):
            value = np.asarray(evaluate_reduced(node.formula, values, time), dtype=float)
            if node.factor != 1.0:
                value = value * node.factor
    except Exception as err:
        raise ObservableError(
            None,
            f"The formula of the observable '{node.id}' failed: "
            f"{type(err).__name__}: {err}",
        ) from err
    shape = (time.shape[0], 1) if node.kind is SCALAR else time.shape
    return np.broadcast_to(value, shape).copy()


def _custom(
    node: CustomNode,
    time: np.ndarray,
    values: Mapping[str, np.ndarray],
    kinds: Mapping[str, ObservableKind],
) -> np.ndarray:
    """Call the function of a custom observable once per point.

    Raises:
        ObservableError: if the function fails for a point, with its row.
    """
    scalar = node.kind is SCALAR
    out = np.full((time.shape[0], 1) if scalar else time.shape, np.nan)
    for row in range(time.shape[0]):
        valid = np.isfinite(time[row])
        try:
            arguments = {
                symbol: float(values[symbol][row, 0])
                if kinds[symbol] is SCALAR
                else values[symbol][row, valid]
                for symbol in node.symbols
            }
            result = node.function(time[row, valid], arguments)
            if scalar:
                out[row, 0] = float(result)
            else:
                array = np.asarray(result, dtype=float)
                if array.shape != (int(valid.sum()),):
                    raise ValueError(
                        f"the function returned the shape {array.shape} for "
                        f"{int(valid.sum())} time points"
                    )
                out[row, valid] = array
        except Exception as err:
            raise ObservableError(
                row,
                f"The function of the observable '{node.id}' failed: "
                f"{type(err).__name__}: {err}",
            ) from err
    return out


def identity_graph(
    selections: Sequence[str],
    units: Mapping[str, str] | None = None,
    keep: Sequence[str] | None = None,
) -> ObservableGraph:
    """Get the graph of a run without observables.

    Every selection is a timecourse of its own name.

    Args:
        selections: the selections, without `time`.
        units: the unit of the time and of every selection, `""` by default.
        keep: the selections of the result, every one by default.

    Returns:
        The graph.

    Raises:
        ValueError: if `keep` names no selection.
    """
    names = tuple(dict.fromkeys(s for s in selections if s != TIME))
    units = dict(units or {})
    return ObservableGraph(
        nodes=(),
        selections=names,
        kinds={TIME: TIMECOURSE, **dict.fromkeys(names, TIMECOURSE)},
        units={TIME: units.get(TIME, ""), **{s: units.get(s, "") for s in names}},
        keep=_keep(keep, names, {}),
        outputs=names,
    )


def compile_observables(
    observables: Sequence[Observable] | None,
    model: RoadrunnerSBMLModel,
    *,
    keep: Sequence[str] | None = None,
    plans: Sequence[Plan] = (),
) -> ObservableGraph:
    """Compile the observables of a run against its first model, see the module.

    Args:
        observables: the observables; `None` or none for the selections of
            the model, see `identity_graph`.
        model: the loaded model, whose selections and units they read.
        keep: the outputs of the result, every output by default; the id of a
            PK observable keeps all its parameters.
        plans: the plans of the run, each with the values of a point, whose
            dosing decides the parameters of the PK observables.

    Returns:
        The graph.

    Raises:
        TypeError: if the observables are no sequence of observables or `keep`
            is a string.
        ValueError: for two observables of one id; an id which is a selection
            of the model; a symbol which is neither an observable nor a
            selection; a cycle; a time of `at` which is a timecourse; a unit
            which cannot be derived without `unit=` or not be converted into
            it; a PK observable which does not fit the model (see
            `compile_pk`); or a `keep` which names no output.
    """
    uinfo = model.uinfo
    time_unit = uinfo.get(TIME, "") or ""
    if not observables:
        selections = [s for s in model.selections or [] if s != TIME]
        units = {s: uinfo.get(s, "") or "" for s in selections}
        return identity_graph(selections, {TIME: time_unit, **units}, keep)
    if isinstance(observables, Observable):
        raise TypeError("The observables of a run are a sequence of observables.")
    definitions: dict[str, Observable] = {}
    for observable in observables:
        if not isinstance(observable, Observable):
            raise TypeError(f"{observable!r} is no observable (Formula, PK, Custom).")
        if observable.id in definitions:
            raise ValueError(f"Two observables have the id '{observable.id}'.")
        definitions[observable.id] = observable
    clash = sorted(name for name in definitions if model.has_selection(name))
    if clash:
        raise ValueError(
            f"The observable ids {clash} are selections of the model: an observable "
            f"and a selection share no name, choose other ids."
        )
    kinds: dict[str, ObservableKind] = {TIME: TIMECOURSE}
    units: dict[str, str] = {TIME: time_unit}
    selections: dict[str, None] = {}
    compiled: dict[str, Node] = {}
    outputs: dict[str, tuple[str, ...]] = {}
    for name in _order(definitions):
        observable = definitions[name]
        for symbol in observable.reads:
            _resolve(symbol, name, model, definitions, outputs, kinds, units, selections)
        if isinstance(observable, Formula):
            node, unit = _compile_formula(observable, kinds, units)
            kinds[name], units[name] = node.kind, unit
            outputs[name] = (name,)
        elif isinstance(observable, Custom):
            node = CustomNode(
                id=name,
                function=observable.function,
                kind=observable.kind,
                symbols=observable.symbols,
            )
            kinds[name], units[name] = observable.kind, observable.unit
            outputs[name] = (name,)
        elif isinstance(observable, PK):
            node, pk_units = _compile_pk(observable, model, kinds, units, plans)
            for output, unit in pk_units.items():
                kinds[output], units[output] = SCALAR, unit
            outputs[name] = tuple(pk_units)
        else:
            raise TypeError(f"{observable!r} is no observable (Formula, PK, Custom).")
        compiled[name] = node
    every = tuple(o for name in definitions for o in outputs[name])
    kept = _keep(keep, every, outputs)
    needed = _needed(kept, definitions)
    nodes = tuple(node for name, node in compiled.items() if name in needed)
    read = {s for name in needed for s in definitions[name].reads}
    doses = tuple(
        node.dose.target
        for node in nodes
        if isinstance(node, PKNode) and node.dose is not None and node.dose.target
    )
    return ObservableGraph(
        nodes=nodes,
        selections=tuple(s for s in selections if s in read),
        kinds=kinds,
        units=units,
        keep=kept,
        outputs=every,
        doses=tuple(dict.fromkeys(doses)),
    )


def _owner(symbol: str, definitions: Mapping[str, Observable]) -> str | None:
    """Get the observable a symbol is an output of, `None` for a selection."""
    if symbol in definitions:
        return symbol
    head, dot, _ = symbol.partition(".")
    if dot and isinstance(definitions.get(head), PK):
        return head
    return None


def _order(definitions: Mapping[str, Observable]) -> list[str]:
    """Order the observables so that every one comes after what it reads.

    Raises:
        ValueError: if the observables read each other in a cycle.
    """
    order: list[str] = []
    state: dict[str, bool] = {}

    def visit(name: str, path: tuple[str, ...]) -> None:
        if state.get(name) is True:
            return
        if state.get(name) is False:
            cycle = (*path[path.index(name) :], name)
            raise ValueError(
                f"The observables read each other in a cycle: {' -> '.join(cycle)}."
            )
        state[name] = False
        for symbol in definitions[name].reads:
            owner = _owner(symbol, definitions)
            if owner is not None:
                visit(owner, (*path, name))
        state[name] = True
        order.append(name)

    for name in definitions:
        visit(name, ())
    return order


def _resolve(
    symbol: str,
    name: str,
    model: RoadrunnerSBMLModel,
    definitions: Mapping[str, Observable],
    outputs: Mapping[str, tuple[str, ...]],
    kinds: dict[str, ObservableKind],
    units: dict[str, str],
    selections: dict[str, None],
) -> None:
    """Resolve a symbol an observable reads: an output or a selection.

    Raises:
        ValueError: if it is a parameter a PK observable does not give, or
            neither an observable nor a selection of the model.
    """
    if symbol in kinds:
        return
    owner = _owner(symbol, definitions)
    if owner is not None:
        raise ValueError(
            f"The observable '{name}' reads '{symbol}', which the PK observable "
            f"'{owner}' does not give: {list(outputs.get(owner, ()))}."
        )
    if not model.has_selection(symbol):
        raise ValueError(
            f"The observable '{name}' reads '{symbol}', which is neither an "
            f"observable nor a selection of the model."
        )
    kinds[symbol] = TIMECOURSE
    units[symbol] = model.uinfo.get(symbol, "") or ""
    selections[symbol] = None


def _derive_unit(formula: str, units: Mapping[str, str]) -> str | None:
    """Derive the unit of a formula, `None` where pint cannot.

    The formula is applied to quantities of one in the units of its symbols;
    a symbol without a unit is dimensionless. A warning of pint, e.g. a unit
    which numpy strips, means that no unit can be derived.
    """
    reduced = reduce_formula(formula)
    try:
        values = {
            symbol: ureg.Quantity(1.0, units[symbol] or "dimensionless")
            for symbol in reduced.symbols
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with np.errstate(all="ignore"):
                value = evaluate_reduced(formula, values)
    except Exception:
        return None
    if isinstance(value, Quantity):
        return str(value.units)
    try:
        float(value)
    except (TypeError, ValueError):
        return None
    return "dimensionless"


def _compile_formula(
    observable: Formula,
    kinds: Mapping[str, ObservableKind],
    units: Mapping[str, str],
) -> tuple[FormulaNode, str]:
    """Compile a formula: its kind, its unit and the factor into it.

    Raises:
        ValueError: if a time of `at` is a timecourse, or the unit cannot be
            derived without `unit=` or not be converted into it.
    """
    reduced = reduce_formula(observable.formula)
    for symbol in reduced.time_symbols:
        if kinds[symbol] is TIMECOURSE:
            raise ValueError(
                f"The time of 'at' in the formula of the observable "
                f"'{observable.id}' reads the timecourse '{symbol}'; it is a number "
                f"or a value per simulation."
            )
    kind = (
        SCALAR if all(kinds[s] is SCALAR for s in reduced.outer_symbols) else TIMECOURSE
    )
    derived = _derive_unit(observable.formula, units)
    if observable.unit is None:
        if derived is None:
            raise ValueError(
                f"The unit of the formula '{observable.formula}' of the observable "
                f"'{observable.id}' cannot be derived, e.g. of a comparison or of "
                f"piecewise; give it as unit=."
            )
        return FormulaNode(observable.id, observable.formula, kind, 1.0), derived
    factor = 1.0
    if derived is not None:
        try:
            factor = float(ureg.Quantity(1.0, derived).to(observable.unit).magnitude)
        except Exception as err:
            raise ValueError(
                f"The formula of the observable '{observable.id}' has the unit "
                f"'{derived}', which cannot be converted into its unit "
                f"'{observable.unit}'."
            ) from err
    return FormulaNode(observable.id, observable.formula, kind, factor), observable.unit


def _compile_pk(
    observable: PK,
    model: RoadrunnerSBMLModel,
    kinds: Mapping[str, ObservableKind],
    units: Mapping[str, str],
    plans: Sequence[Plan],
) -> tuple[PKNode, dict[str, str]]:
    """Compile a PK observable against the model, see `compile_pk`.

    Raises:
        ValueError: if its timecourse is a value per simulation, the dose is no
            target of the model, or `compile_pk` raises.
    """
    if kinds[observable.selection] is not TIMECOURSE:
        raise ValueError(
            f"The PK observable '{observable.id}' analyses '{observable.selection}', "
            f"which is a value per simulation, not a timecourse."
        )
    dose_unit = None
    if isinstance(observable.dose, str):
        if not model.has_selection(observable.dose):
            raise ValueError(
                f"The dose '{observable.dose}' of the observable '{observable.id}' is "
                f"no target of the model."
            )
        dose_unit = model.uinfo.get(observable.dose, "") or ""
    return compile_pk(
        observable,
        unit=units[observable.selection],
        time_unit=units[TIME],
        dose_unit=dose_unit,
        plans=plans,
    )


def _keep(
    keep: Sequence[str] | None,
    outputs: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Get the kept outputs; a group, the id of a PK observable, keeps all of its.

    Raises:
        TypeError: if `keep` is a string.
        ValueError: if `keep` names no output or keeps nothing.
    """
    if keep is None:
        return tuple(outputs)
    if isinstance(keep, str):
        raise TypeError(f"'keep' is a sequence of ids, not the string {keep!r}.")
    kept: dict[str, None] = {}
    for key in keep:
        names = groups.get(key) or ([key] if key in outputs else None)
        if names is None:
            raise ValueError(
                f"'keep' names '{key}', which is no observable of the run: "
                f"{list(outputs)}."
            )
        kept.update(dict.fromkeys(names))
    if not kept:
        raise ValueError("'keep' keeps nothing; name at least one observable.")
    return tuple(kept)


def _needed(kept: Sequence[str], definitions: Mapping[str, Observable]) -> set[str]:
    """Get the observables the kept outputs need, the kept ones among them."""
    needed: set[str] = set()
    stack = [owner for k in kept if (owner := _owner(k, definitions)) is not None]
    while stack:
        name = stack.pop()
        if name in needed:
            continue
        needed.add(name)
        stack.extend(
            owner
            for symbol in definitions[name].reads
            if (owner := _owner(symbol, definitions)) is not None
        )
    return needed
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/simulator/test_observables.py tests/fit`
Expected: PASS (the fit tests cover the replaced `_is_selection`).

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/simulator/observables.py src/sbmlsim/model/model_roadrunner.py src/sbmlsim/fit/optimization.py tests/simulator/test_observables.py
git commit -m "The observables of a run compile into a graph which evaluates them on a chunk" -m "compile_observables orders the observables by what they read, checks every symbol against the model, finds cycles, derives the kind of a formula (a timecourse or a value per simulation) and its unit, converts it into a declared one and keeps only the observables the kept ones need. The graph is free of pint and pickles, so a chunk carries it into a worker, where it evaluates the formulas, the custom functions and the PK analyses on the stacked native solutions. RoadrunnerSBMLModel.has_selection replaces the check of the fit."
```

---

### Task 5: The worker and the assembly run on the graph

This task changes no behavior: a run without observables evaluates the graph of the selections of its model, and every existing test and the scan regression pass.

**Files:**
- Modify: `src/sbmlsim/simulator/worker.py:1-20` (docstring), `:170-290` (`Chunk`, `ChunkResult`, `_run_chunk`)
- Modify: `src/sbmlsim/simulator/simulator.py:278-337` (`_compile`), `:439-626` (`_Compiled`, `chunks`, `assemble`), `:640-645` (delete `_columns`)
- Modify: `tests/simulator/test_worker.py:42-62` (the helper `_chunk`)
- Test: `tests/simulator/test_worker.py`, all existing tests

**Interfaces:**
- Consumes: `ObservableGraph`, `ObservableError`, `identity_graph`, `compile_observables` (Task 4).
- Produces:
  - `point_plan(plan: Plan, values: Mapping[str, np.ndarray], timed: Mapping[float, Mapping[str, np.ndarray]], k: int) -> Plan` in `worker.py` (the body of `Chunk.plan_of`, which calls it).
  - `Chunk(indices, plan, model, graph: ObservableGraph, values, timed, time, on_error="raise")` (the field `graph` replaces `selections`), property `selections -> tuple[str, ...]` (`(TIME, *graph.selections)`).
  - `ChunkResult(indices, values, scalars: np.ndarray, status, errors)`: `values` is `(point, row, column)` with the columns `(TIME, *graph.timecourses)`, `scalars` is `(point, scalar)` with the columns `graph.scalars`.
  - `_Compiled.graph: ObservableGraph` replaces `_Compiled.selections`.

- [ ] **Step 1: Adapt the test helper**

In `tests/simulator/test_worker.py`, import `identity_graph` from `sbmlsim.simulator.observables` and change the `Chunk(...)` of `_chunk` to pass `graph=identity_graph(selections)` instead of `selections=selections` (the helper keeps its parameter `selections`, `identity_graph` drops `time`). Add a test of the scalars of a chunk:

```python
def test_a_chunk_without_observables_has_no_scalars(model: RoadrunnerSBMLModel) -> None:
    result = run_chunk(_chunk(model, Simulation(end=2, steps=4)), model)
    assert result.scalars.shape == (2, 0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/simulator/test_worker.py`
Expected: FAIL with `TypeError: Chunk.__init__() got an unexpected keyword argument 'graph'`.

- [ ] **Step 3: Write the implementation**

In `src/sbmlsim/simulator/worker.py`:

- Replace the first paragraph of the module docstring with: "`run_chunk` is the same function serially and in a worker process: for every point of a chunk it applies the values of the point to the plan of the chunk, `Plan.with_values`, and runs the plan with `execute`; the native solutions of the points which ran are stacked into arrays padded with `NaN`, the observables of the run are evaluated on them, see `sbmlsim.simulator.observables`, and the kept timecourses are interpolated onto the grid of times of the chunk, if it has one. Nothing in here uses pint or xarray, and a chunk and its result are numbers, strings, a plan and the graph of the observables, so they pickle."
- Import `Mapping` from `collections.abc`, `Callable` as well, and `from sbmlsim.simulator.observables import ObservableError, ObservableGraph` and `from sbmlsim.result.scan import TIME`.
- Replace `Chunk`, `ChunkResult` and `_run_chunk` with:

```python
def point_plan(
    plan: Plan,
    values: Mapping[str, np.ndarray],
    timed: Mapping[float, Mapping[str, np.ndarray]],
    k: int,
) -> Plan:
    """Get the plan of the `k`-th point of values.

    The values of the dimensions without a time are applied first, then the
    changes of the dimensions with one in time order, so a target in both has
    the value of the change at its time.

    Args:
        plan: the plan of the points.
        values: target -> value of every point.
        timed: time -> target -> value of every point.
        k: the point.

    Returns:
        The plan with the values of the point.
    """
    plan = plan.with_values({t: float(v[k]) for t, v in values.items()})
    for at in sorted(timed):
        changes = timed[at]
        plan = plan.with_values({t: float(v[k]) for t, v in changes.items()}, at=at)
    return plan


@dataclass(frozen=True)
class Chunk:
    """Points of a scan which share a plan and a model.

    Attributes:
        indices: the flat indices of the points in the scan, in C order.
        plan: the plan of the points.
        model: the index of the model of the points among the models of the
            run.
        graph: the observables of the run, which give the selections.
        values: target -> value of every point, which replaces the target
            wherever the plan sets it and is a change before the
            initialization otherwise.
        timed: time -> target -> value of every point, a change at that time.
        time: the grid of times to interpolate the kept timecourses onto,
            `None` for the time points of the simulation.
        on_error: what to do about a point which fails.
    """

    indices: np.ndarray
    plan: Plan
    model: int
    graph: ObservableGraph
    values: dict[str, np.ndarray]
    timed: dict[float, dict[str, np.ndarray]]
    time: np.ndarray | None
    on_error: OnError = "raise"

    @property
    def selections(self) -> tuple[str, ...]:
        """Get the selections of roadrunner, `time` first."""
        return (TIME, *self.graph.selections)

    def plan_of(self, k: int) -> Plan:
        """Get the plan of the `k`-th point of the chunk, see `point_plan`."""
        return point_plan(self.plan, self.values, self.timed, k)


@dataclass(frozen=True)
class ChunkResult:
    """The answer of a chunk.

    Attributes:
        indices: the flat indices of the points, those of the chunk.
        values: the kept timecourses, `(point, row, column)` with the time
            first and the timecourses of the graph, padded with `NaN`.
        scalars: the kept values per simulation, `(point, column)` with the
            scalars of the graph.
        status: `0` for a point which ran, `1` for one which failed.
        errors: the flat index and the error of the first `MAX_ERRORS`
            points which failed, in the order of the scan.
    """

    indices: np.ndarray
    values: np.ndarray
    scalars: np.ndarray
    status: np.ndarray
    errors: tuple[tuple[int, str], ...]
```

and, after `run_chunk`:

```python
def _run_chunk(chunk: Chunk, model: RoadrunnerSBMLModel) -> ChunkResult:
    """Run the points of a chunk, see `run_chunk`."""
    n = len(chunk.indices)
    status = np.zeros(n, dtype=np.int8)
    errors: list[tuple[int, str]] = []

    def fail(k: int, message: str, cause: BaseException) -> None:
        index = int(chunk.indices[k])
        if chunk.on_error == "raise":
            raise ScanPointError(index, message) from cause
        status[k] = 1
        errors.append((index, message))

    points: list[int] = []
    solutions: list[np.ndarray] = []
    plans: list[Plan] = []
    for k in range(n):
        # a definition which does not fit the plan is no failed point
        plan = chunk.plan_of(k)
        try:
            result = execute(plan, model, chunk.selections)
        except Exception as err:
            fail(k, f"{type(err).__name__}: {err}", err)
            continue
        points.append(k)
        solutions.append(result.values)
        plans.append(plan)
    time, outputs = _observe(chunk, points, solutions, plans, fail)
    return _pack(chunk, n, points, time, outputs, status, errors)


def _observe(
    chunk: Chunk,
    points: list[int],
    solutions: list[np.ndarray],
    plans: list[Plan],
    fail: Callable[[int, str, BaseException], None],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Evaluate the observables on the native solutions of the points which ran.

    A point whose observable fails fails and is dropped, and the observables
    are evaluated again on the others; an observable which fails for every
    point fails them all. `points`, `solutions` and `plans` keep the points
    which are left.

    Returns:
        The time points `(n_points, n_rows)` padded with `NaN` and the kept
        outputs of the points which are left.
    """
    while points:
        rows = max(solution.shape[0] for solution in solutions)
        stacked = np.full((len(points), rows, len(chunk.selections)), np.nan)
        for r, solution in enumerate(solutions):
            stacked[r, : solution.shape[0]] = solution
        time = stacked[:, :, 0]
        columns = {
            name: stacked[:, :, j] for j, name in enumerate(chunk.selections) if j
        }
        try:
            return time, chunk.graph.evaluate(time, columns, plans)
        except ObservableError as err:
            failing = list(range(len(points))) if err.row is None else [err.row]
            for r in failing:
                fail(points[r], err.message, err)
            for r in reversed(failing):
                del points[r], solutions[r], plans[r]
    return np.empty((0, 0)), {}


def _pack(
    chunk: Chunk,
    n: int,
    points: list[int],
    time: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    status: np.ndarray,
    errors: list[tuple[int, str]],
) -> ChunkResult:
    """Write the kept outputs of the points which ran into the arrays of a chunk.

    The timecourses are interpolated onto the grid of the chunk, if it has
    one; a failed point is `NaN`, with the time of the grid.
    """
    names = chunk.graph.timecourses
    scalars = chunk.graph.scalars
    if chunk.time is not None:
        n_rows = chunk.time.size
    else:
        n_rows = time.shape[1] if points else 0
    values = np.full((n, n_rows, 1 + len(names)), np.nan)
    if chunk.time is not None:
        values[:, :, 0] = chunk.time
    table = np.full((n, len(scalars)), np.nan)
    for r, k in enumerate(points):
        rows = np.stack([time[r], *(outputs[name][r] for name in names)])
        if chunk.time is not None:
            rows = apply_weights(grid_weights(time[r], chunk.time), rows)
            rows[0] = chunk.time
        values[k] = rows.T
        for c, name in enumerate(scalars):
            table[k, c] = outputs[name][r]
    return ChunkResult(
        indices=chunk.indices,
        values=values,
        scalars=table,
        status=status,
        errors=tuple(sorted(errors)[:MAX_ERRORS]),
    )
```

In `src/sbmlsim/simulator/simulator.py`:

- Import `compile_observables` and `ObservableGraph` from `sbmlsim.simulator.observables`.
- In `_compile`, compute the graph from the first model and use its selections: replace the line `selections = (TIME, *(s for s in first.selections or [] if s != TIME))` with `graph = compile_observables(None, first)` and `selections = (TIME, *graph.selections)`; keep the checks which follow on `selections` as they are; pass `graph=graph` instead of `selections=selections` to `_Compiled`.
- In `_Compiled`, replace the field `selections: tuple[str, ...]` (and its docstring line) with `graph: ObservableGraph` ("the observables of the run"); in `chunks`, pass `graph=self.graph` instead of `selections=self.selections`.
- Replace the part of `assemble` from `n_rows = (` to the end of the loop over the variables (the loop over `_columns(self.selections)`) with:

```python
        n, shape = self.size, self.scan.shape
        names = self.graph.timecourses
        scalars = self.graph.scalars
        n_rows = (
            self.grid.size
            if self.grid is not None
            else max((r.values.shape[1] for r in results), default=0)
        )
        cube = np.full((1 + len(names), n, n_rows), np.nan)
        table = np.full((len(scalars), n), np.nan)
        status = np.zeros(n, dtype=np.int8)
        errors: list[tuple[int, str]] = []
        for result in results:
            rows = result.values.shape[1]
            cube[:, result.indices, :rows] = np.moveaxis(result.values, 2, 0)
            table[:, result.indices] = result.scalars.T
            status[result.indices] = result.status
            errors.extend(result.errors)

        first = self.models[0]
        dims = list(self.scan.dims)
        tdim = TIME if self.grid is not None else POINT
        units: dict[str, str] = {}
        data_vars: dict[str, Any] = {}
        if names:
            units[TIME] = first.uinfo.get(TIME, "") or ""
            if self.grid is None:
                data_vars[TIME] = ([*dims, POINT], cube[0].reshape(*shape, n_rows))
        for j, name in enumerate(names, 1):
            data_vars[name] = ([*dims, tdim], cube[j].reshape(*shape, n_rows))
            units[name] = self.graph.units[name]
        for j, name in enumerate(scalars):
            data_vars[name] = (dims, table[j].reshape(shape))
            units[name] = self.graph.units[name]
```

  and replace `if self.grid is not None:` before `coords[TIME] = self.grid` with `if self.grid is not None and names:`.
- Delete `_columns`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/simulator/test_worker.py tests/simulator/test_simulator.py tests/simulator/test_scan_regression.py`, then `uv run pytest -q`
Expected: PASS, every existing test unchanged apart from the helper `_chunk`. A test which reads `_Compiled.selections` reads `compiled.graph.selections` instead.

- [ ] **Step 5: Lint, types, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check`

```bash
git add src/sbmlsim/simulator/worker.py src/sbmlsim/simulator/simulator.py tests/simulator/test_worker.py
git commit -m "A chunk evaluates the observables of its run on the native solutions" -m "The worker stacks the native solutions of the points which ran, evaluates the observable graph on them and only then interpolates the kept timecourses onto the grid, so the observables of phase 2 see the output of the simulation. A run without observables has the graph of the selections of its model and gives the results of phase 1; a chunk answers with its timecourses and its values per simulation, which the assembly writes over the scan dimensions."
```

---

### Task 6: Simulator.run computes observables

**Files:**
- Modify: `src/sbmlsim/simulator/simulator.py` (`run`, `_compile`, `_Compiled`, `chunks`, `assemble`, the module docstring)
- Test: `tests/simulator/test_simulator_observables.py` (new)

**Interfaces:**
- Consumes: `compile_observables`, `ObservableGraph` (Task 4); `point_plan` (Task 5); `Observable` (Task 2).
- Produces:
  - `Simulator.run(model, scan, observables: Sequence[Observable] | None = None, *, time=None, keep: Sequence[str] | None = None, on_error="raise", progress=None) -> ScanResult`.
  - `Simulator._compile(model, scan, time, observables=None, keep=None) -> _Compiled`.
  - `_Compiled.observables: tuple[Observable, ...]`; `attrs["observables"]` of a result is the list of `to_dict()` of the observables of a run with observables.
  - module functions `_positions(scan) -> np.ndarray`, `_plan_points(scan, positions, s, m) -> np.ndarray`, `_values_of(scan, positions, vectors, at_times, part) -> tuple[dict, dict]` (the parts of `_Compiled.chunks` which the first plan of every simulation and model needs as well).

- [ ] **Step 1: Write the failing tests**

Create `tests/simulator/test_simulator_observables.py`:

```python
"""A scan computes observables from every simulation."""

import json
from pathlib import Path

import numpy as np
import pkpdutils as pk
import pytest
import xarray as xr

import sbmlsim.simulator.simulator as simulator_module
from sbmlsim import Q
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.result import ScanResult
from sbmlsim.simulation import PK, Change, Custom, Dimension, Formula, Scan, Simulation
from sbmlsim.simulation.observables import ObservableKind
from sbmlsim.simulator import ScanError, Simulator
from sbmlsim.units import ureg
from tests.simulator.models import (
    BLOWUP,
    auc_of_c,
    doubled,
    fails_for_large_k1,
    last_value,
    sbml,
    sbml_pk,
)

KA, KE, V = 1.0, 0.2, 10.0


def concentration(t: np.ndarray, dose: float) -> np.ndarray:
    return dose * KA / (V * (KA - KE)) * (np.exp(-KE * t) - np.exp(-KA * t))


@pytest.fixture(scope="module")
def pk_sbml() -> str:
    return sbml_pk()


def dosed(steps: int | None = 480, times: tuple[float, ...] = (0.0,)) -> Simulation:
    return Simulation(
        end=48, steps=steps, changes=[Change(list(times), {"PODOSE": Q(100, "mg")})]
    )


def dose_scan(steps: int | None = 480, doses: tuple[float, ...] = (50.0, 100.0, 200.0)) -> Scan:
    return Scan(dosed(steps), [Dimension("dose", values={"PODOSE": Q(list(doses), "mg")})])


REDUCTIONS = [
    Formula("c", "[C]"),
    Formula("cmax", "max(c)"),
    Formula("cmin", "min(c)"),
    Formula("cmean", "mean(c)"),
    Formula("c10", "at(c, 10)"),
    Formula("rel", "c / cmax"),
]


def test_reductions_against_the_analytic_solution(pk_sbml: str) -> None:
    res = Simulator().run(pk_sbml, dose_scan(), REDUCTIONS)
    grid = res["time"].values
    assert res["cmax"].dims == ("dose",) and res["c"].dims == ("dose", "time")
    for k, dose in enumerate((50.0, 100.0, 200.0)):
        exact = concentration(grid, dose)
        assert float(res["cmax"][k]) == pytest.approx(exact.max(), rel=1e-6)
        assert float(res["cmin"][k]) == pytest.approx(exact.min(), abs=1e-9)
        assert float(res["cmean"][k]) == pytest.approx(
            np.trapezoid(exact, grid) / 48.0, rel=1e-6
        )
        assert float(res["c10"][k]) == pytest.approx(concentration(10.0, dose), rel=1e-6)
    np.testing.assert_allclose(res["rel"].values.max(axis=1), 1.0)
    assert ureg.Quantity(1.0, res.units["cmax"]).to("mg/l").magnitude == pytest.approx(1.0)
    assert res.units["rel"] == "dimensionless"


def test_a_baseline_formula_broadcasts_over_the_time(pk_sbml: str) -> None:
    observables = [
        Formula("c", "[C]"),
        Formula("c1", "at(c, 1)"),
        Formula("rel", "c / c1"),
    ]
    res = Simulator().run(pk_sbml, dose_scan(steps=48), observables)
    np.testing.assert_allclose(res["rel"].sel(time=1.0).values, 1.0)


def test_reductions_on_the_steps_of_the_integrator(pk_sbml: str) -> None:
    res = Simulator().run(pk_sbml, dose_scan(steps=None, doses=(50.0, 200.0)), REDUCTIONS)
    assert res.ragged
    for k, dose in enumerate((50.0, 200.0)):
        time = res["time"].values[k]
        time = time[np.isfinite(time)]
        exact = concentration(time, dose)
        assert float(res["cmax"][k]) == pytest.approx(exact.max(), rel=1e-6)
        assert float(res["cmean"][k]) == pytest.approx(
            np.trapezoid(exact, time) / 48.0, rel=1e-6
        )


def test_pk_equals_pkpdutils_on_the_same_arrays(pk_sbml: str) -> None:
    observables = [Formula("conc", "[C]"), PK("c", "[C]", dose="PODOSE", route="oral")]
    res = Simulator().run(pk_sbml, dose_scan(doses=(50.0, 100.0)), observables)
    timecourses = pk.Timecourses.from_arrays(
        res["time"].values,
        res["conc"].values,
        time_unit="hr",
        unit="mg/l",
        dims=("_sim",),
        dose={"amount": np.array([[50.0], [100.0]]), "time": np.zeros((2, 1)), "unit": "mg"},
        route="oral",
    )
    direct = pk.nca(timecourses)
    for parameter in ("cmax", "tmax", "auc_inf_obs", "thalf", "cl_f"):
        np.testing.assert_allclose(
            res[f"c.{parameter}"].values, direct.ds[parameter].values, rtol=1e-9
        )
    assert float(res["c.auc_inf_obs"][1]) == pytest.approx(100.0 / (V * KE), rel=2e-3)
    assert float(res["c.thalf"][1]) == pytest.approx(np.log(2.0) / KE, rel=1e-3)
    assert float(res["c.tmax"][1]) == pytest.approx(np.log(KA / KE) / (KA - KE), abs=0.1)
    assert float(res["c.cl_f"][0]) == pytest.approx(V * KE, rel=2e-3)
    assert float(res["c.cl_f"][1]) == pytest.approx(V * KE, rel=2e-3)


def test_a_multiple_dosing(pk_sbml: str) -> None:
    observables = [
        PK("c", "[C]", dose="PODOSE", route="oral"),
        Formula("depot", "at(PODOSE, 24)"),
    ]
    res = Simulator().run(pk_sbml, dosed(times=(0.0, 24.0)), observables)
    assert float(res["c.tau"]) == pytest.approx(24.0)
    # at the time of a change, the value after it: the new dose in the depot
    assert float(res["depot"]) == pytest.approx(100.0)


def test_pk_without_a_dose(pk_sbml: str) -> None:
    res = Simulator().run(pk_sbml, dosed(), [PK("c", "[C]")])
    assert "c.cmax" in res and "c.cl_f" not in res


def test_a_dose_only_a_dimension_gives(pk_sbml: str) -> None:
    scan = Scan(
        Simulation(end=48, steps=480),
        [Dimension("dose", values={"PODOSE": Q([50.0, 100.0], "mg")})],
    )
    res = Simulator().run(pk_sbml, scan, [PK("c", "[C]", dose="PODOSE", route="oral")])
    np.testing.assert_allclose(res["c.cl_f"].values, V * KE, rtol=2e-3)


def test_custom_observables(pk_sbml: str) -> None:
    observables = [
        Formula("c", "[C]"),
        Custom("auc", auc_of_c, "mg*hr/l", symbols=["[C]"]),
        Custom("twice", doubled, "mg/l", symbols=["[C]"], kind=ObservableKind.TIMECOURSE),
    ]
    res = Simulator().run(pk_sbml, dose_scan(), observables)
    np.testing.assert_allclose(
        res["auc"].values, np.trapezoid(res["c"].values, res["time"].values, axis=1)
    )
    np.testing.assert_allclose(res["twice"].values, 2.0 * res["c"].values)


def test_keep_drops_intermediates_but_evaluates_them(pk_sbml: str) -> None:
    full = Simulator().run(pk_sbml, dose_scan(), REDUCTIONS)
    kept = Simulator().run(pk_sbml, dose_scan(), REDUCTIONS, keep=["rel", "c10"])
    assert set(kept.ds.data_vars) == {"rel", "c10"}
    xr.testing.assert_equal(kept["rel"], full["rel"])
    xr.testing.assert_equal(kept["c10"], full["c10"])


def test_a_result_of_values_per_simulation_has_no_time(pk_sbml: str, tmp_path: Path) -> None:
    observables = [PK("c", "[C]", dose="PODOSE", route="oral"), Formula("cmax", "max([C])")]
    res = Simulator().run(pk_sbml, dose_scan(), observables, keep=["c", "cmax"])
    assert "time" not in res.ds.dims and "time" not in res.ds.variables
    assert res["cmax"].dims == ("dose",)
    again = ScanResult.from_netcdf(_write(res, tmp_path))
    xr.testing.assert_equal(again.ds, res.ds)
    summary = res.summary("dose")
    assert "statistic" in summary.ds.dims
    stored = json.loads(json.dumps(res.ds.attrs["observables"]))
    assert [o["id"] for o in stored] == ["c", "cmax"]


def _write(res: ScanResult, tmp_path: Path) -> Path:
    path = tmp_path / "result.nc"
    res.to_netcdf(path)
    return path


def test_observables_read_what_the_model_does_not_select(pk_sbml: str) -> None:
    model = RoadrunnerSBMLModel(source=pk_sbml)
    model.set_selections(["time"])
    res = Simulator().run(model, dosed(), [Formula("c", "[C]")])
    assert float(res["c"].max()) == pytest.approx(concentration(np.linspace(0, 48, 481), 100.0).max(), rel=1e-6)


def test_the_result_does_not_depend_on_the_workers_or_the_chunks(
    pk_sbml: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    observables = [
        *REDUCTIONS,
        PK("p", "[C]", dose="PODOSE", route="oral"),
        Custom("auc", auc_of_c, "mg*hr/l", symbols=["[C]"]),
    ]
    scan = dose_scan(doses=tuple(np.linspace(10.0, 200.0, 12)))
    reference = Simulator(n_workers=1).run(pk_sbml, scan, observables)
    for workers in (2, 4):
        res = Simulator(n_workers=workers).run(pk_sbml, scan, observables)
        xr.testing.assert_equal(res.ds, reference.ds)
    monkeypatch.setattr(simulator_module, "_chunk_size", lambda n, w: 1)
    xr.testing.assert_equal(Simulator(n_workers=2).run(pk_sbml, scan, observables).ds, reference.ds)
    monkeypatch.setattr(simulator_module, "_chunk_size", lambda n, w: 1000)
    xr.testing.assert_equal(Simulator(n_workers=1).run(pk_sbml, scan, observables).ds, reference.ds)


def test_a_failing_point_is_nan_for_every_observable() -> None:
    scan = Scan(Simulation(end=1, steps=4), [Dimension("rate", values={"k": [0.1, 2.0]})])
    observables = [
        Formula("smax", "max(S)"),
        Formula("s", "S"),
        Custom("last", last_value, "dimensionless", symbols=["S"]),
    ]
    res = Simulator().run(sbml(BLOWUP), scan, observables, on_error="flag")
    assert res["status"].values.tolist() == [0, 1]
    assert np.isfinite(res["smax"].values[0]) and np.isnan(res["smax"].values[1])
    assert np.isnan(res["last"].values[1]) and np.isnan(res["s"].values[1]).all()
    with pytest.raises(ScanError, match="k=2.0"):
        Simulator().run(sbml(BLOWUP), scan, observables)


def test_an_observable_which_fails_fails_its_point() -> None:
    scan = Scan(Simulation(end=1, steps=4), [Dimension("d", values={"k1": [0.5, 2.0]})])
    observables = [Custom("check", fails_for_large_k1, "dimensionless", symbols=["k1"])]
    res = Simulator().run(sbml(), scan, observables, on_error="flag")
    assert res["status"].values.tolist() == [0, 1]
    assert "k1 is too large" in res.ds.attrs["errors"][0]
    with pytest.raises(ScanError, match="k1 is too large"):
        Simulator().run(sbml(), scan, observables)


def test_the_observables_are_checked_when_the_scan_is_compiled(pk_sbml: str) -> None:
    with pytest.raises(ValueError, match="'nope'"):
        Simulator().run(pk_sbml, dose_scan(), [Formula("x", "nope")])
    scan = Scan(dosed(), [Dimension("cmax", values={"PODOSE": Q([1.0, 2.0], "mg")})])
    with pytest.raises(ValueError, match="cmax"):
        Simulator().run(pk_sbml, scan, [Formula("cmax", "max([C])")])


def test_a_dimension_of_models_with_observables() -> None:
    models = {"slow": sbml_pk(ke=0.2), "fast": sbml_pk(ke=0.3)}
    scan = Scan(dosed(), [Dimension("model", models=models)])
    res = Simulator().run(None, scan, [PK("c", "[C]", dose="PODOSE", route="oral")])
    np.testing.assert_allclose(
        res["c.thalf"].values, np.log(2.0) / np.array([0.2, 0.3]), rtol=1e-3
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/simulator/test_simulator_observables.py`
Expected: FAIL with `TypeError: Simulator.run() takes 3 positional arguments but 4 were given`.

- [ ] **Step 3: Write the implementation**

In `src/sbmlsim/simulator/simulator.py`:

- Import `Observable` from `sbmlsim.simulation.observables` and `point_plan` from `sbmlsim.simulator.worker`.
- Extend step 1 of the module docstring by "the observables of the run, ordered, checked against the first model and with their kinds and units, see `sbmlsim.simulator.observables`" and step 3 by "the observables are evaluated on the native solutions of the points of a chunk before the kept timecourses are interpolated onto a grid".
- Change `run` to:

```python
    def run(
        self,
        model: ModelLike | None,
        scan: Scan | Simulation,
        observables: Sequence[Observable] | None = None,
        *,
        time: ArrayLike | Quantity | None = None,
        keep: Sequence[str] | None = None,
        on_error: OnError = "raise",
        progress: bool | None = None,
    ) -> ScanResult:
```

  and its docstring: the first paragraph becomes "The variables of the result are the kept observables; without observables they are the selections of the model, of the first model of a dimension of models, each a timecourse of its own name.", and add, after `scan`:

```
            observables: what the run computes from every simulation, see
                `sbmlsim.simulation.observables`; they are evaluated on the
                native solution of every simulation, also with `time`.
```

  and after `time`:

```
            keep: the observables of the result, every one by default; the id
                of a PK observable keeps all its parameters; without
                observables the selections of the result. The others are
                evaluated where a kept one needs them and dropped.
```

  Pass them on: `compiled = self._compile(model, Scan.of(scan), time, observables, keep)`.
- Move the parts of `_Compiled.chunks` which find the points of a plan and their values into module functions and use them in `chunks`:

```python
def _positions(scan: Scan) -> np.ndarray:
    """Get the index of every point along every dimension, a row per point."""
    if not scan.dimensions:
        return np.zeros((1, 0), dtype=int)
    return np.stack(np.unravel_index(np.arange(scan.size), scan.shape), axis=1)


def _axis(scan: Scan, kind: DimensionKind) -> int | None:
    """Get the position of the dimension of a kind, `None` without one."""
    return next((k for k, d in enumerate(scan.dimensions) if d.kind is kind), None)


def _plan_points(scan: Scan, positions: np.ndarray, s: int, m: int) -> np.ndarray:
    """Get the flat indices of the points of a simulation and a model."""
    mask = np.ones(scan.size, dtype=bool)
    sim_axis = _axis(scan, DimensionKind.SIMULATIONS)
    model_axis = _axis(scan, DimensionKind.MODELS)
    if sim_axis is not None:
        mask &= positions[:, sim_axis] == s
    if model_axis is not None:
        mask &= positions[:, model_axis] == m
    return np.flatnonzero(mask)


def _values_of(
    scan: Scan,
    positions: np.ndarray,
    vectors: Sequence[Mapping[str, np.ndarray]],
    at_times: Sequence[float | None],
    part: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[float, dict[str, np.ndarray]]]:
    """Get the values of points: target -> values, and time -> target -> values."""
    values: dict[str, np.ndarray] = {}
    timed: dict[float, dict[str, np.ndarray]] = {}
    for i in range(len(scan.dimensions)):
        at = at_times[i]
        for target, vector in vectors[i].items():
            point_values = vector[positions[part, i]]
            if at is None:
                values[target] = point_values
            else:
                timed.setdefault(at, {})[target] = point_values
    return values, timed
```

  `_Compiled._positions` and `_Compiled._axis` are deleted; `chunks` becomes:

```python
    def chunks(self, workers: int, on_error: OnError) -> list[Chunk]:
        """Cut the points into chunks which share a plan, see the module.

        Returns:
            The chunks in the order of their first point.
        """
        size = _chunk_size(self.size, workers)
        positions = _positions(self.scan)
        chunks: list[Chunk] = []
        for (s, m), plan in self.plans.items():
            indices = _plan_points(self.scan, positions, s, m)
            for start in range(0, indices.size, size):
                part = indices[start : start + size]
                values, timed = _values_of(
                    self.scan, positions, self.vectors[m], self.at_times[(s, m)], part
                )
                chunks.append(
                    Chunk(
                        indices=part,
                        plan=plan,
                        model=m,
                        graph=self.graph,
                        values=values,
                        timed=timed,
                        time=self.grid if self.interpolate else None,
                        on_error=on_error,
                    )
                )
        chunks.sort(key=lambda chunk: int(chunk.indices[0]))
        return chunks
```

- Change `_compile` to take `observables: Sequence[Observable] | None = None, keep: Sequence[str] | None = None`, document them in its docstring (and add to its `Raises`: "an observable which does not fit the first model, see `compile_observables`; an observable id which is a dimension id or a target the scan changes"), and compile the graph after the plans and the vectors, with the first plan of every simulation and model:

```python
        models, labels = self._models(model, scan)
        first = models[0]
        plans: dict[tuple[int, int], Plan] = {}
        at_times: dict[tuple[int, int], list[float | None]] = {}
        for s, simulation in enumerate(scan.simulations()):
            for m, loaded in enumerate(models):
                try:
                    plan = self.compile(loaded, simulation)
                    at_times[(s, m)] = _at_times(scan, simulation, loaded, plan)
                except ValueError as err:
                    raise ValueError(f"{_where(scan, s, labels[m])}: {err}") from err
                plans[(s, m)] = plan
        vectors = [
            _vectors(scan, loaded, label)
            for loaded, label in zip(models, labels, strict=True)
        ]
        positions = _positions(scan)
        point_plans: list[Plan] = []
        for (s, m), plan in plans.items():
            indices = _plan_points(scan, positions, s, m)
            if indices.size:
                values, timed = _values_of(
                    scan, positions, vectors[m], at_times[(s, m)], indices[:1]
                )
                point_plans.append(point_plan(plan, values, timed, 0))
        graph = compile_observables(observables, first, keep=keep, plans=point_plans)
        selections = (TIME, *graph.selections)
        if not observables:
            reserved = sorted((set(selections[1:]) & RESERVED) - {TIME})
            if reserved:
                raise ValueError(
                    f"The selections {reserved} are names of the result "
                    f"({sorted(RESERVED)}), select other entities, see "
                    f"`RoadrunnerSBMLModel.set_selections`."
                )
        for loaded, label in zip(models[1:], labels[1:], strict=True):
            _check_model(loaded, label, first, (*selections, *graph.doses))
        clash = sorted(set(scan.dims) & set(graph.outputs))
        if clash:
            raise ValueError(
                f"The dimension ids {clash} are selections of the model or "
                f"observables of the run: a dimension and a variable of the result "
                f"share no name, choose other ids."
            )
        targets = {t for dimension in scan.dimensions for t in dimension.values}
        changed = sorted(targets & set(graph.outputs)) if observables else []
        if changed:
            raise ValueError(
                f"The observables {changed} are targets the scan changes, which are "
                f"coordinates of the result; choose other ids."
            )
        grid, interpolate = _grid(plans, time, first)
        return _Compiled(
            scan=scan,
            models=models,
            plans=plans,
            vectors=vectors,
            at_times=at_times,
            graph=graph,
            grid=grid,
            interpolate=interpolate,
            observables=tuple(observables or ()),
        )
```

  (The order of the checks changes: the plans are compiled before the selections are checked. If a test of phase 1 expects the error of a selection before the error of a plan, keep the test and move the check of the reserved selections before the loop of the plans for a run without observables, using `compile_observables(None, first)` there.)
- Add the field `observables: tuple[Observable, ...] = ()` ("the observables of the run, their definitions are the provenance of the result") to `_Compiled`, and in `assemble` add to `attrs`, after `"integrator_settings"`: `if self.observables: attrs["observables"] = [o.to_dict() for o in self.observables]`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/simulator/test_simulator_observables.py`, then `uv run pytest -q`
Expected: PASS. Tolerances: the integrator runs at `relative_tolerance=1e-10`; a reduction compared with the analytic solution at the same time points agrees to `1e-6` relative, a PK parameter against the analytic value to the error of the trapezoids of pkpdutils on a grid of 0.1 h (`2e-3`).

- [ ] **Step 5: Lint, types, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check`

```bash
git add src/sbmlsim/simulator/simulator.py tests/simulator/test_simulator_observables.py
git commit -m "Simulator.run computes observables and keeps the ones asked for" -m "run(model, scan, observables, keep=...) compiles the observables against the first model with the dosing of the first point of every simulation, selects only what the needed observables read and checks them against the dimensions and the changed targets. The result keeps timecourses over the scan dimensions and the time and values per simulation over the scan dimensions only, with their units and the definitions of the observables as provenance; it does not depend on the number of workers or the size of the chunks."
```

---

### Task 7: The result hands a PK analysis and a timecourse to pkpdutils

**Files:**
- Modify: `src/sbmlsim/result/scan.py` (`ScanResult.nca`, `ScanResult.to_timecourses`, module docstring)
- Test: `tests/result/test_scan_result_pk.py` (new)

**Interfaces:**
- Consumes: `FLAGS` (Task 3); a result of Task 6 with PK outputs `<id>.<parameter>` and `<id>.flags`.
- Produces:
  - `ScanResult.nca(observable: str) -> pkpdutils.NCAResult`.
  - `ScanResult.to_timecourses(key: str, **kwargs: Any) -> pkpdutils.Timecourses`.

- [ ] **Step 1: Write the failing tests**

Create `tests/result/test_scan_result_pk.py`:

```python
"""A result hands a PK analysis and a timecourse over to pkpdutils."""

import numpy as np
import pkpdutils as pk
import pytest

from sbmlsim import Q
from sbmlsim.simulation import PK, Change, Dimension, Formula, Scan, Simulation
from sbmlsim.simulator import Simulator
from tests.simulator.models import sbml_pk

OBSERVABLES = [Formula("conc", "[C]"), PK("c", "[C]", dose="PODOSE", route="oral")]


@pytest.fixture(scope="module")
def pk_sbml() -> str:
    return sbml_pk()


def simulation(steps: int | None = 480) -> Simulation:
    return Simulation(end=48, steps=steps, changes=[Change(0, {"PODOSE": Q(100, "mg")})])


def scan(steps: int | None = 480) -> Scan:
    return Scan(simulation(steps), [Dimension("dose", values={"PODOSE": Q([50.0, 100.0], "mg")})])


def test_nca_is_the_analysis_of_pkpdutils(pk_sbml: str) -> None:
    res = Simulator().run(pk_sbml, scan(), OBSERVABLES)
    nca = res.nca("c")
    assert isinstance(nca, pk.NCAResult)
    assert nca.sample_dims == ("dose",)
    np.testing.assert_array_equal(nca["cmax"].values, res["c.cmax"].values)
    assert nca.units("cmax") == res.units["c.cmax"]
    assert nca["flags"].dtype.kind == "i"


def test_to_timecourses_hands_over_a_timecourse(pk_sbml: str) -> None:
    res = Simulator().run(pk_sbml, scan(), OBSERVABLES)
    timecourses = res.to_timecourses(
        "conc", dose={"amount": np.array([50.0, 100.0]), "unit": "mg"}, route="oral"
    )
    again = pk.nca(timecourses)
    np.testing.assert_allclose(
        again["auc_inf_obs"].values, res["c.auc_inf_obs"].values, rtol=1e-9
    )


def test_a_ragged_result_and_a_single_simulation(pk_sbml: str) -> None:
    ragged = Simulator().run(pk_sbml, scan(steps=None), OBSERVABLES)
    assert ragged.ragged
    timecourses = ragged.to_timecourses("conc")
    assert pk.nca(timecourses)["cmax"].shape == (2,)
    single = Simulator().run(pk_sbml, simulation(), OBSERVABLES)
    assert single.nca("c").sample_dims == ()
    assert float(pk.nca(single.to_timecourses("conc"))["cmax"]) == pytest.approx(
        float(single["c.cmax"])
    )


def test_nca_needs_the_flags_and_to_timecourses_a_timecourse(pk_sbml: str) -> None:
    res = Simulator().run(pk_sbml, scan(), OBSERVABLES, keep=["conc", "c.cmax"])
    with pytest.raises(KeyError, match="c.flags"):
        res.nca("c")
    with pytest.raises(ValueError, match="c.cmax"):
        res.to_timecourses("c.cmax")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/result/test_scan_result_pk.py`
Expected: FAIL with `AttributeError: 'ScanResult' object has no attribute 'nca'`.

- [ ] **Step 3: Write the implementation**

In `src/sbmlsim/result/scan.py`, add to the imports `from typing import TYPE_CHECKING` (next to `Any`) and

```python
if TYPE_CHECKING:
    from pkpdutils import NCAResult, Timecourses
```

add a sentence to the module docstring: "`nca` and `to_timecourses` hand a PK analysis and a timecourse over to pkpdutils, whose layout `(*dims, time)` the result has.", and add to `ScanResult` (after `quantity`):

```python
    def nca(self, observable: str) -> NCAResult:
        """Get the analysis of a PK observable as the result of pkpdutils.

        The kept parameters of the observable, `<id>.<parameter>`, with their
        units and its flags over the dimensions of the scan. A point which
        failed has the flags `0` and `NaN` parameters.

        Args:
            observable: the id of the PK observable.

        Returns:
            The `pkpdutils.NCAResult`.

        Raises:
            KeyError: if the result does not keep the flags of the observable.
        """
        from pkpdutils import NCAResult

        prefix = f"{observable}."
        flags = f"{prefix}{FLAGS}"
        if flags not in self.ds.data_vars:
            raise KeyError(
                f"The result does not keep the PK observable '{observable}': keep "
                f"its flags '{flags}', e.g. keep=['{observable}']."
            )
        data_vars: dict[str, xr.DataArray] = {}
        for name in self.ds.data_vars:
            name = str(name)
            if not name.startswith(prefix):
                continue
            values = self.ds[name]
            parameter = name.removeprefix(prefix)
            if parameter == FLAGS:
                values = values.fillna(0).astype(int)
            data_vars[parameter] = values.assign_attrs(units=self.units.get(name, ""))
        return NCAResult(xr.Dataset(data_vars))

    def to_timecourses(self, key: str, **kwargs: Any) -> Timecourses:
        """Get a timecourse of the result as the timecourses of pkpdutils.

        Args:
            key: the variable, a timecourse.
            **kwargs: the other arguments of `Timecourses.from_arrays`, e.g.
                `dose` and `route`.

        Returns:
            The `pkpdutils.Timecourses` over the dimensions of the scan.

        Raises:
            ValueError: if the variable is no timecourse of the result.
        """
        from pkpdutils import Timecourses

        tdim = POINT if self.ragged else TIME
        if key not in self.ds.data_vars or tdim not in self.ds[key].dims:
            raise ValueError(f"'{key}' is no timecourse of the result.")
        dims = tuple(self.dims)
        values = self.ds[key].transpose(*dims, tdim).values
        time = (
            self.ds[TIME].transpose(*dims, POINT).values
            if self.ragged
            else self.ds[TIME].values
        )
        coords = {dim: self.ds[dim].values for dim in dims}
        return Timecourses.from_arrays(
            time,
            values,
            time_unit=self.units.get(TIME) or "dimensionless",
            unit=self.units.get(key) or "dimensionless",
            dims=dims,
            coords=coords,
            **kwargs,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/result/test_scan_result_pk.py tests/result`
Expected: PASS. If `NCAResult` refuses the coordinates of the changed targets (`PODOSE(dose)`), drop the coordinates which are not dimensions with `xr.Dataset(data_vars).reset_coords(drop=True)` and assert that in the test; if `Timecourses.from_arrays` refuses `dims=()` with `coords={}`, pass `coords=None` for a result without dimensions.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/result/scan.py tests/result/test_scan_result_pk.py
git commit -m "A ScanResult hands a PK analysis and a timecourse over to pkpdutils" -m "nca(id) builds the NCAResult of pkpdutils from the kept parameters and the flags of a PK observable over the scan dimensions, and to_timecourses(key) gives a timecourse of the result as Timecourses of pkpdutils, ragged or on a grid, with the arguments of a dose passed on; the layout (*dims, time) of the result needs no transpose."
```

---

### Task 8: Docs, example and the description of the core

**Files:**
- Create: `docs/observables.md`, `examples/observables.py`, `docs/api/simulation.observables.md`, `docs/api/simulator.observables.md`, `docs/api/simulator.pk.md`
- Modify: `zensical.toml:24-27` (nav: page), `:61-71` (nav: API pages); `docs/scans.md` (a section that links the page); `docs/data.md:67` (link to `observables.md` if Task 1 linked `scans.md`); `tests/docs/test_docs_code.py:17` (`PAGES`); `tests/examples/test_example_scripts.py` (the example); `examples/README.md` (the table); `CLAUDE.md` (dependencies, architecture); `docs/api/index.md` if it lists the modules (check with `rg -n "simulator.worker" docs/api/index.md`).

**Interfaces:**
- Consumes: everything of Tasks 1-7; the packaged model `sbmlsim.resources.MIDAZOLAM_SBML` (dose target `PODOSE_mid` in mg, concentration `[Cve_mid]` in mmol/l, molar mass `Mr_mid` in g/mol, time in min) and `REPRESSILATOR_SBML`.
- Produces: `examples.observables.time_above(time, values) -> float` (a `Custom` function the page imports).

- [ ] **Step 1: The example**

Create `examples/observables.py`:

```python
"""Observables of a scan: the plasma concentration of midazolam for three doses.

A Formula gives the mass concentration from the molar one, a PK observable
its non-compartmental analysis with pkpdutils, a Custom the time above a
threshold; the result keeps the values per dose and the timecourses.
"""

from typing import Any

import numpy as np

from sbmlsim import Q
from sbmlsim.resources import MIDAZOLAM_SBML
from sbmlsim.simulation import PK, Change, Custom, Dimension, Formula, Scan, Simulation
from sbmlsim.simulator import Simulator

#: the threshold of `time_above`, in ng/ml
THRESHOLD = 10.0


def time_above(time: np.ndarray, values: dict[str, Any]) -> float:
    """Get the time the plasma concentration is above the threshold.

    The time is in the time unit of the model, minutes for midazolam.
    """
    above = values["mid"] > THRESHOLD
    return float(np.sum(np.diff(time)[above[:-1]]))


def run() -> Any:
    """Run the scan over the dose and print the PK parameters per dose."""
    simulation = Simulation(
        time_unit="hr",
        end=24,
        steps=480,
        changes=[Change(0, {"PODOSE_mid": Q(7.5, "mg")})],
    )
    scan = Scan(
        simulation, [Dimension("dose", values={"PODOSE_mid": Q([5.0, 7.5, 15.0], "mg")})]
    )
    observables = [
        Formula("mid", "[Cve_mid] * Mr_mid", unit="ng/ml"),
        Formula("mid_rel", "mid / max(mid)"),
        PK("pk", "mid", dose="PODOSE_mid", route="oral"),
        Custom("t_above", time_above, "min", symbols=["mid"]),
    ]
    res = Simulator().run(
        MIDAZOLAM_SBML,
        scan,
        observables,
        keep=["mid", "pk.cmax", "pk.tmax", "pk.auc_inf_obs", "pk.thalf", "t_above"],
    )
    for name in ("pk.cmax", "pk.tmax", "pk.auc_inf_obs", "pk.thalf", "t_above"):
        print(name, res[name].values.round(3), res.units[name])
    return res


if __name__ == "__main__":
    run()
```

The time of the result is in the time unit of the model (min for midazolam, `res.units["time"]`), so `t_above` is in minutes. Run `uv run python -m examples.observables` and check that the values are plausible (cmax rising with the dose, tmax and thalf equal for the doses). Add `"examples.observables"` to the list of examples in `tests/examples/test_example_scripts.py` (next to `"examples.scan"`) and a row to the table of `examples/README.md`: `| examples/observables.py | observables of a scan: a formula of the mass concentration, the PK analysis of pkpdutils and a custom function of midazolam over three doses |`.

- [ ] **Step 2: The page**

Create `docs/observables.md` with the content below (every `python` block runs in `tests/docs/test_docs_code.py`; add `"observables.md"` to `PAGES`). The fences of the page are shown here with four backticks only to nest them in the plan; the page uses three.

````markdown
# Observables

An observable is what a scan computes from every simulation: a timecourse, such as a concentration normalized to its maximum, or a value per simulation, such as the maximum itself or the area under the curve. `Simulator.run(model, scan, observables, keep=...)` evaluates them in the workers on the native solution of every simulation, before any interpolation onto a grid, so they are as exact as the output of the simulation. There are three kinds: a `Formula` of the math of PEtab with reductions over the time, the non-compartmental analysis of pkpdutils (`PK`) and a `Custom` function; `keep` chooses which of them the result keeps.

## Formulas and reductions

A `Formula` reads the selections of the model (`S` an amount, `[S]` a concentration, `time`, parameters, reactions) and other observables by their id:

```python
import numpy as np

from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Dimension, Formula, Scan, Simulation
from sbmlsim.simulator import Simulator

scan = Scan(
    Simulation(end=500, steps=500),
    [Dimension("hill", values={"n": np.linspace(1.5, 4.0, 6)})],
)
observables = [
    Formula("px", "PX"),
    Formula("px_max", "max(px)"),
    Formula("px_mean", "mean(px)"),
    Formula("px_250", "at(px, 250)"),
    Formula("px_rel", "px / px_max"),
]
res = Simulator().run(REPRESSILATOR_SBML, scan, observables)
print(res["px_max"].values)  # a value per point of the scan
print(res["px_rel"].dims)  # ('hill', 'time')
```

Four reductions reduce the time of every simulation on its own:

| reduction | value |
| --- | --- |
| `max(x)`, `min(x)` | the largest and the smallest value, ignoring `NaN` |
| `mean(x)` | the time weighted mean: the trapezoidal integral divided by the time between the first and the last time point, which does not depend on the steps of the integrator |
| `at(x, t)` | the value at the time `t` in the time unit of the model, interpolated linearly; at the time of a change the value after it, outside of the simulated times `NaN` |

A formula which reads only values per simulation is a value per simulation, otherwise a timecourse; a value per simulation in a timecourse is the same at every time, which is how `px / px_max` normalizes every simulation to its own maximum and `ins / at(ins, 0)` to its baseline. `max` and `min` with two or more arguments are the elementwise maximum and minimum of PEtab. The padding of a ragged scan and the steady state after the end are no time points of a reduction.

## Units

The unit of a formula is derived from the units of what it reads, the reductions keep the unit of their argument. A formula with `unit=` is converted into it:

```python
from sbmlsim import Q
from sbmlsim.resources import MIDAZOLAM_SBML
from sbmlsim.simulation import PK, Change, Custom

simulation = Simulation(
    time_unit="hr",
    end=24,
    steps=480,
    changes=[Change(0, {"PODOSE_mid": Q(7.5, "mg")})],
)
mass = Formula("mid", "[Cve_mid] * Mr_mid", unit="ng/ml")
res = Simulator().run(MIDAZOLAM_SBML, simulation, [mass, Formula("mid_max", "max(mid)")])
print(res.units["mid"], res.units["mid_max"])
```

Where pint cannot derive a unit, e.g. for `piecewise` or a comparison, the formula needs `unit=`, which is then taken as declared: `Formula("high", "piecewise(1, mid > 100, 0)", unit="dimensionless")`. The compile step of a scan raises for a formula without one, for a unit which cannot be converted, for a symbol which is neither a selection nor an observable and for observables which read each other in a cycle, before any simulation runs.

## PK

`PK` is the non-compartmental analysis of a timecourse with pkpdutils, every parameter a value per simulation `<id>.<parameter>`:

```python
doses = Scan(
    simulation,
    [Dimension("dose", values={"PODOSE_mid": Q([5.0, 7.5, 15.0], "mg")})],
)
observables = [
    mass,
    PK("pk", "mid", dose="PODOSE_mid", route="oral"),
    Formula("auc_per_cmax", "pk.auc_inf_obs / pk.cmax"),
]
res = Simulator().run(MIDAZOLAM_SBML, doses, observables)
print(res["pk.cmax"].values, res.units["pk.cmax"])
print(res["auc_per_cmax"].values, res.units["auc_per_cmax"])
nca = res.nca("pk")  # the NCAResult of pkpdutils
timecourses = res.to_timecourses("mid")  # the Timecourses of pkpdutils
```

The doses are read from every point of the scan: the values the simulation assigns to the dose target are the doses and the times of these values their times, so a dose of a dimension and a multiple dosing by `Change([0, 12], ...)` need no further input; a value before the initialization is a dose at the start. A quantity is a fixed dose at the start, and without a dose the parameters which need one (clearance, volume) are left out. The parameters are those pkpdutils derives for the dosing, e.g. `pk.cmax`, `pk.tmax`, `pk.auc_inf_obs`, `pk.thalf`, `pk.cl_f`, and `pk.flags`, the flags of the analysis; `parameters=[...]` keeps a subset and `options=pkpdutils.NCAOptions(...)` sets the analysis. A formula reads a parameter by its name. The analysis is as exact as the time points of the simulation: the steps of the integrator can be coarse in a smooth elimination phase, so a non-compartmental analysis is best run on `steps` or `times`.

## Custom functions

A `Custom` observable is a function of a module, `function(time, values)`, called once per simulation with its time points and the values of its symbols; it returns a float, or an array of the length of `time` with `kind=ObservableKind.TIMECOURSE`:

```python
from examples.observables import time_above

res = Simulator().run(
    MIDAZOLAM_SBML,
    doses,
    [mass, Custom("t_above", time_above, "min", symbols=["mid"])],
    keep=["t_above"],
)
print(res["t_above"].values)
```

The function must be defined at the top level of a module, so that it pickles for the workers of a scan; a lambda or a closure raises.

## keep and memory

`keep` is the list of observables in the result, all by default; the id of a PK observable keeps all its parameters. The other observables are evaluated where a kept one needs them and dropped, and an observable no kept one needs is not evaluated at all:

```python
res = Simulator().run(MIDAZOLAM_SBML, doses, observables, keep=["pk"])
print(sorted(res.ds.data_vars)[:3], "time" in res.ds.dims)
```

`keep` is the control of the memory of a large scan: 1e5 points with 481 time points and two timecourses are 0.8 GB, the same points with ten values per simulation 8 MB. A result whose kept observables are all values per simulation has no time dimension.

## Errors

With `on_error="flag"` every observable of a point which fails is `NaN` and the variable `status` marks the point, see [Parameter scans](scans.md). A `Custom` function which raises fails its point, a `Formula` or `PK` which fails fails the points it was evaluated on with it.
````

- [ ] **Step 3: The navigation, the API pages and the links**

- `docs/api/simulation.observables.md` with the single line `::: sbmlsim.simulation.observables`, `docs/api/simulator.observables.md` with `::: sbmlsim.simulator.observables`, `docs/api/simulator.pk.md` with `::: sbmlsim.simulator.pk`.
- `zensical.toml`: `{ "Observables" = "observables.md" },` after `{ "Parameter scans" = "scans.md" },`; `{ "observables" = "api/simulation.observables.md" },` after the `scan` entry of the simulation group; `{ "observables" = "api/simulator.observables.md" },` and `{ "pk" = "api/simulator.pk.md" },` after the `worker` entry of the simulator group.
- `docs/scans.md`: a section `## Observables` before the section on the result (or at the end if there is none): "A scan computes observables from every simulation, e.g. the maximum of a concentration or the parameters of a non-compartmental analysis, with `Simulator.run(model, scan, observables, keep=...)`; see [Observables](observables.md)." with a short runnable block (`Formula("px_max", "max(PX)")` on the 1d scan of the page).
- `docs/data.md:67`: the link points to `observables.md`.
- `docs/api/index.md`: add the three modules if the page lists the modules.

- [ ] **Step 4: CLAUDE.md**

- In the sentence of the runtime dependencies, add `pkpdutils` ("`pkpdutils` (the non-compartmental analysis of a `PK` observable, imported when one is compiled)") after `SALib`.
- In the paragraph of the simulation core, after the description of `Simulator(...)`: describe `run(model, scan, observables, *, time, keep, on_error, progress)`; the observables `Formula`, `PK`, `Custom` (`simulation/observables.py`, frozen, picklable; a `Custom` function of a module); `simulator/observables.py` (`compile_observables` orders them, checks the symbols against the first model, derives kinds and units, prunes to what `keep` needs, gives the picklable `ObservableGraph`; a run without observables has `identity_graph` of the selections); `simulator/pk.py` (doses from the plans, parameters found by a probe analysis per dosing, analysis per group of dose counts, pkpdutils imported lazily); the worker evaluates the graph on the stacked native solutions before it interpolates; the result keeps timecourses over `(*dims, time|_point)` and values per simulation over `dims`, `attrs["observables"]`; `ScanResult.nca`/`to_timecourses`; the reductions of `simulator/formula.py` (`reduce_formula`, `evaluate_reduced`, `max`/`min`/`mean`/`at` over the last axis, identifiers with a dot).
- In the paragraph of the experiments: a `Data` of type FUNCTION reduces with `max`/`min` per simulation (along the time), `mean`/`at` are for observables.
- In the paragraph of the formulas: `simulator/formula.py` holds the reductions shared by observables and `Data`.

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest -q -n 0 tests/docs tests/examples/test_example_scripts.py -k "observables or docs"`, `uv run zensical build --clean` (no warnings about missing pages or links), `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add docs/observables.md docs/scans.md docs/data.md docs/api/simulation.observables.md docs/api/simulator.observables.md docs/api/simulator.pk.md zensical.toml examples/observables.py examples/README.md tests/docs/test_docs_code.py tests/examples/test_example_scripts.py CLAUDE.md
git commit -m "The observables have a page, an example and their place in the description of the core" -m "docs/observables.md explains formulas with the reductions, units, the PK analysis with pkpdutils, custom functions, keep and the errors, with code which the docs test runs; examples/observables.py scans the dose of midazolam with all three kinds. The navigation, the API reference and CLAUDE.md describe the new modules."
```

---

### Task 9: Verification and the pull request

**Files:** none new; the pull request.

**Interfaces:**
- Consumes: the branch after Task 8.
- Produces: a pull request to `develop`.

- [ ] **Step 1: The whole suite and the checks**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`, `uv run zensical build --clean`, and the pool tests under `spawn` in one process:

```bash
uv run python -c "
import multiprocessing as m, sys
m.set_start_method('spawn')
import pytest
sys.exit(pytest.main(['-q', '-n', '0', 'tests/simulator/test_simulator_observables.py', 'tests/simulator/test_simulator_pool.py']))"
```

Expected: all pass, no warnings.

- [ ] **Step 2: The speed of a run without observables**

Run: `uv run pytest -m benchmark -n 0 -s tests/simulator/test_benchmark.py`
Expected: the gates of phase 1 hold (the serial scan of 1e3 points within its relative bound, the pool of 1e4 points at least 2.5 times faster on 4 workers). Note the numbers for the pull request.

- [ ] **Step 3: The pull request**

Push the branch and create the pull request with `gh-axi` (never `gh`), base `develop`, title "Observables of a scan: Formula, PK and Custom (#249, phase 2)". The body describes, without any agent attribution: the new API (`Formula`, `PK`, `Custom`, `Simulator.run(..., observables, keep=...)`, the reductions `max`/`min`/`mean`/`at`, `ScanResult.nca`/`to_timecourses`), the change of `Data` (reductions per simulation), the decisions of this plan where the spec is silent (the list above), pkpdutils as a dependency, the numbers of Step 2, and that the analyses (sub-project 3) and the observables of the experiments (sub-project 4) follow. Wait for the checks `tests`, `ruff`, `ty` and `docs`.
