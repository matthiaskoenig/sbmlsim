# Cleanup of sbmlsim to its core (#250) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the leftovers of SED-ML, the dead code and the shims of sbmlsim, give `Data` formulas the math of PEtab, drop unused dependencies and make the tests and CI faster, in one pull request with one commit per task.

**Architecture:** Deletions are verified by searching every caller (`src/`, `tests/`, `examples/`, `docs/`, `scripts/` and `~/git/pkdb_models/pkdb_models`) before the code goes. New code is small: `evaluate_function` in `data.py` (PEtab math plus the reductions `max`/`min` of one argument), `CompiledFormula.apply`, a lazy import of petab in `simulator/formula.py`, the serial path of the fit runner for one worker and a `dpi` of the sensitivity analyses.

**Tech Stack:** python 3.13/3.14, uv, pytest with xdist, ruff, ty, zensical, libroadrunner, libsbml, sympy, petab, pint.

**Spec:** `docs/superpowers/specs/2026-10-08-core-cleanup-design.md`

## Global Constraints

- Branch `core-cleanup` (stacked on `examples-overrides`, PR #251); the pull request goes to `develop`.
- Never use the em dash character, use a plain dash `-`.
- No agent attribution anywhere: no `Co-Authored-By` trailer, no "Generated with Claude Code" line in commits, the pull request, docs or code.
- Commit messages are full sentences describing the outcome, in the style of `git log` (e.g. "The examples mark their overrides, ..."), with a body that explains what and why. No conventional commit prefixes.
- Never edit `CHANGELOG.md` or auto-generated files; no release notes (they belong to a release commit).
- Markdown has no hard line wraps: a paragraph, list item or table row is one line.
- Every module, class and function of the package has full type annotations and a google style docstring (ruff `D`); `tests/` and `examples/` are exempt from docstrings. A subclass marks overrides with `typing.override`.
- ty must stay at zero diagnostics (`uv run ty check`); suppress only with a rule specific `# ty: ignore[rule-name]`.
- Library code logs with `logging.getLogger(__name__)` and lazy `%s` formatting, it never prints.
- Every commit passes `uv run ruff check`, `uv run ruff format --check`, `uv run ty check` and `uv run pytest -q` (the default suite, about 80 s on 20 cores). The pre-commit hooks run ruff and ty on commit.
- Stays untouched apart from receiving moved code: everything of PEtab (`fit/petab_v2/`, `fit/petab_omex.py`), PEtab SciML (`sciml/`, `fit/derived.py`, `model/provenance.py`), `comparison/`, `examples/comparison`, the SBML Test Suite (`testsuite/`), `simulation/sensitivity.py`, the `sensitivity/` package.
- Before a public name is deleted, `rg -n "<name>" ~/git/pkdb_models/pkdb_models` must show no use outside comments and outside files which import a module of sbmlsim that no longer exists (e.g. `plotting_deprecated_matplotlib`, `simulation_ray`, `fit.analysis`). A name pkdb_models uses stays.
- The zensical navigation lists every api page in `zensical.toml` (`nav`, section "API"); deleting a page in `docs/api/` deletes its nav entry and its row in `docs/api/index.md`.
- Use `uv run` for every command. After a change of `pyproject.toml` run `uv lock` and `uv sync --extra dev`.

## Review Focus

- A `Data` formula of quantities with units (`x / y` of `mmol/l` and `mmol/l`, `x * y` of `mmol/l` and `l`) keeps the units as the old evaluation did; covered in Task 3.
- A reduction over data padded with `NaN` (a scan result where simulations have different numbers of points) ignores the `NaN`: `Y/max(Y)` of `[1, 2, nan]` is `[0.5, 1, nan]`; covered in Task 3.
- Nested and mixed calls: `max(max(Y), 2)`, `Y/max(Y + Z)`, `max(Y, Z)` elementwise of two arrays of the same unit, an identifier containing `max` (e.g. `Ymax/max(Y)`) is not taken for a call; covered in Task 3.
- `Data("[X]", task=...)` and `Data("X", task=...)` keep their `sid`, `selection` and `name` after the parameter `symbol` is removed, because figures, fit mappings and stored results key on them; covered in Task 7.
- A fit with `n_cores=1` (now serial) gives the same fitted parameters as `serial=True` with the same seed, and the tests which test the worker pool still start one; covered in Task 11.

---

### Task 0: Baseline

No commit. Record the numbers the pull request reports.

- [ ] **Step 1: Time the default suite on 4 CPUs**

Run: `cd /home/mkoenig/git/sbmlsim && /usr/bin/time -f "%e s" taskset -c 0-3 uv run pytest -q -n 4 -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass; note the wall time (about 173 s) and the test count in `$SCRATCH/baseline.txt` where `$SCRATCH=/tmp/claude-1000/-home-mkoenig-git-sbmlsim/b77b65ba-d50e-4d6a-8e6c-54bb838158ae/scratchpad`.

- [ ] **Step 2: Record the import time and the source size**

Run:
```bash
cd /home/mkoenig/git/sbmlsim
for m in sbmlsim.simulator sbmlsim.experiment sbmlsim.fit; do uv run python -X importtime -c "import $m" 2>&1 | tail -1; done
find src/sbmlsim -name "*.py" | xargs cat | wc -l
find tests -name "*.py" | xargs cat | wc -l
```
Append the output to `$SCRATCH/baseline.txt`.

- [ ] **Step 3: Write the import check of pkdb_models**

Create `$SCRATCH/pkdb_imports.py`:

```python
"""Import every sbmlsim name the live files of pkdb_models import.

A file is live if all its imports of sbmlsim resolve against the installed
sbmlsim; the script prints the live files and the names which fail.
"""

import ast
import importlib
import sys
from pathlib import Path

ROOT = Path.home() / "git" / "pkdb_models" / "pkdb_models"


def imports(path: Path) -> list[tuple[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("sbmlsim"):
            names.extend((node.module, a.name) for a in node.names)
        elif isinstance(node, ast.Import):
            names.extend((a.name, "") for a in node.names if a.name.startswith("sbmlsim"))
    return names


def resolves(module: str, name: str) -> bool:
    try:
        mod = importlib.import_module(module)
    except Exception:
        return False
    if not name:
        return True
    if hasattr(mod, name):
        return True
    try:
        importlib.import_module(f"{module}.{name}")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    out = Path(sys.argv[1])
    live = []
    for path in sorted(ROOT.rglob("*.py")):
        names = imports(path)
        if names and all(resolves(m, n) for m, n in names):
            live.append(str(path.relative_to(ROOT)))
    out.write_text("\n".join(live) + "\n", encoding="utf-8")
    print(len(live), "live files")
```

Run: `cd /home/mkoenig/git/sbmlsim && uv run python $SCRATCH/pkdb_imports.py $SCRATCH/pkdb_live_before.txt`
Expected: prints the number of live files (about 325).

---

### Task 1: The SED-ML objects of `simulation/` are removed

**Files:**
- Delete: `src/sbmlsim/simulation/base.py`, `src/sbmlsim/simulation/algorithm.py`, `src/sbmlsim/simulation/calculation.py`, `src/sbmlsim/simulation/change.py`
- Modify: `src/sbmlsim/simulation/range.py` (keep only `Dimension`)
- Delete: `tests/simulation/test_algorithm.py`, `docs/api/simulation.base.md`, `docs/api/simulation.algorithm.md`, `docs/api/simulation.calculation.md`, `docs/api/simulation.change.md`
- Modify: `zensical.toml` (nav entries `change`, `algorithm`, `calculation`, `base` of section simulation), `docs/api/index.md:43-46`, `docs/references.md:126` (the KISAO paragraph points at `sbmlsim.simulation.algorithm`), `src/sbmlsim/fit/petab_v2/gaps.py` (the mention near line 357 of `simulation.algorithm`, reword to `RoadrunnerSBMLModel.set_integrator_settings`)

**Interfaces:**
- Produces: `sbmlsim.simulation.range` contains only `Dimension` (same constructor `Dimension(dimension, index=None, changes=None, at=None)`, `__len__`, `indices_from_dimensions`); `sbmlsim.simulation` exports stay `Change`, `Dimension`, `ScanSim`, `Simulation`, `SteadyState`.

- [ ] **Step 1: Confirm no caller**

Run: `rg -n "simulation\.(base|algorithm|calculation|change)\b|simulation\.range import (?!Dimension)|\b(BaseObject|AlgorithmParameter|KISAOType|Calculation|ComputeChange|AppliedDimension|DependentVariable|VectorRange|UniformRange|UniformRangeType|DataRange|FunctionalRange)\b" -P src tests examples docs scripts --glob '!docs/superpowers/**' --glob '!src/sbmlsim/simulation/{base,algorithm,calculation,change,range}.py'`
Expected: only the docs/api pages, `docs/api/index.md`, `docs/references.md`, `gaps.py` and `tests/simulation/test_algorithm.py`. Run the same names against `~/git/pkdb_models/pkdb_models` and expect no hit outside stale files.

- [ ] **Step 2: Delete the modules and trim `range.py`**

Delete the four modules and the test. In `range.py` remove the imports of `BaseObject`, `Calculation`, `Parameter`, `Variable`, `abstractmethod`, `Enum`, `auto`, the classes `Range` through `FunctionalRange`, the comment `# Dimension is basically a ComputeChange;` and the `__main__` block. The module docstring becomes `"""The dimensions of a scan."""`. Keep `itertools`, `logging`, `Iterable`, `Any`, `numpy`.

- [ ] **Step 3: Update the docs**

Delete the four api pages, their nav entries and their rows in `docs/api/index.md`; delete the KISAO paragraph in `docs/references.md` if it only serves `sbmlsim.simulation.algorithm`; reword the mention in `gaps.py`.

- [ ] **Step 4: Verify**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q tests/simulation tests/simulator tests/experiment && uv run zensical build --clean 2>&1 | tail -3`
Expected: all pass, the build reports no missing page.

- [ ] **Step 5: Full suite and commit**

Run: `uv run pytest -q`
```bash
git add -A src/sbmlsim/simulation tests/simulation docs zensical.toml src/sbmlsim/fit/petab_v2/gaps.py
git commit -m "The SED-ML objects of the simulations are removed, a scan keeps only its Dimension" -m "Algorithm, AlgorithmParameter, Calculation, the SED-ML Change and ComputeChange, the ranges and their base objects were not used by any simulation: the integrator is set on the model and the changes are simulation.definition.Change. range.py keeps the Dimension of a scan."
```

---

### Task 2: Data generators, the SED-ML report and the hook `reports()` are removed

**Files:**
- Delete: `src/sbmlsim/result/datagenerator.py`, `src/sbmlsim/result/report.py`, `examples/datagenerator.py`, `tests/result/test_datagenerator.py`, `docs/api/result.datagenerator.md`, `docs/api/result.report.md`
- Modify: `src/sbmlsim/experiment/experiment.py` (`_reports` at about :110, :128, :149, :291, the method `reports` at about :223, docstrings at :182-207 which speak of "DataGenerators" and "datagenerators"), `examples/curve_types/experiment.py` (import at :13, method `reports` at :74), `tests/examples/test_example_scripts.py` (`examples.datagenerator` in `SCRIPTS`), `examples/README.md` (row of `examples/datagenerator.py`), `docs/data.md:119` (the paragraph on `DataGenerator`), `zensical.toml`, `docs/api/index.md:70-71`, `CLAUDE.md` (the hook list in the paragraph on `experiment/`)

**Interfaces:**
- Produces: `SimulationExperiment` has the hooks `datasets`, `models`, `simulations`, `tasks`, `data`, `fit_mappings`, `figures`; no `reports`, no `_reports`.

- [ ] **Step 1: Confirm no caller**

Run: `rg -n "datagenerator|DataGenerator|result\.report\b|result/report|_reports|def reports" src tests examples docs scripts CLAUDE.md --glob '!docs/superpowers/**' --glob '!tests/data/**'` and the same names on `~/git/pkdb_models/pkdb_models`.
Expected in pkdb_models: only `def datagenerators(self)` methods in stale midazolam experiments, which are not this hook.

- [ ] **Step 2: Delete and edit**

Delete the files. In `experiment.py` remove `self._reports`, its update in `initialize`, the line `f"{'reports':20} ..."` in the string representation, `"_reports"` in the list of attributes and the method `reports`. Replace "DataGenerators" in the docstrings of `data()` with "the data of the experiment, including functions of other data". In `examples/curve_types/experiment.py` remove the import and the method. Remove the README row, the `SCRIPTS` entry, the docs paragraph (no replacement sentence), nav entries and index rows. In `CLAUDE.md` remove `` `reports()` `` from the list of hooks.

- [ ] **Step 3: Verify**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q tests/experiment tests/report tests/result && uv run pytest -q tests/examples -k curve_types && uv run zensical build --clean 2>&1 | tail -3`
Expected: pass.

- [ ] **Step 4: Full suite and commit**

Run: `uv run pytest -q`
```bash
git add -A src tests examples docs zensical.toml CLAUDE.md
git commit -m "The data generators and the reports of SED-ML are removed with the hook reports() of an experiment" -m "The reports a SimulationExperiment collected were never rendered nor serialized, and the data generators were only used by their own example. No experiment of pkdb_models defines the hook."
```

---

### Task 3: The formula of a `Data` is the math of PEtab

**Files:**
- Modify: `src/sbmlsim/simulator/formula.py` (add `CompiledFormula.apply`)
- Modify: `src/sbmlsim/data.py` (add `evaluate_function` and helpers, use it in `get_data` at about :256-271, drop `from sbmlsim import mathml`)
- Create: `tests/test_data_function.py`
- Modify: `tests/test_mathml.py` (remove `test_max_min_reduce` style tests at about :93-115 which test `mathml.evaluate`; the rest of the file goes in Task 4)
- Modify: `docs/data.md` (the section on functions near line 55: the syntax is PEtab math, the reductions)

**Interfaces:**
- Consumes: `sbmlsim.simulator.formula.compile_formula(formula: str) -> CompiledFormula` with `symbols: tuple[str, ...]`.
- Produces: `CompiledFormula.apply(values: Sequence[Any]) -> Any`; `sbmlsim.data.evaluate_function(formula: str, variables: Mapping[str, Any]) -> Any`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data_function.py`:

```python
"""The formula of a Data of type FUNCTION is the math of PEtab."""

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.data import evaluate_function


def test_a_ratio_of_quantities_keeps_the_units() -> None:
    x = Q(np.array([1.0, 2.0]), "mmol/l")
    y = Q(np.array([2.0, 4.0]), "mmol/l")
    ratio = evaluate_function("x / y", {"x": x, "y": y})
    np.testing.assert_allclose(ratio.to("dimensionless").magnitude, [0.5, 0.5])
    amount = evaluate_function("x * v", {"x": x, "v": Q(2.0, "l")})
    assert amount.to("mmol").magnitude.tolist() == pytest.approx([2.0, 4.0])


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("Y/max(Y)", [0.25, 0.5, 1.0]),
        ("Y - min(Y)", [0.0, 1.0, 3.0]),
        ("max(Y, 2)", [2.0, 2.0, 4.0]),
        ("min(Y, 2)", [1.0, 2.0, 2.0]),
        ("Y/max(Y + Z)", [1 / 8, 2 / 8, 4 / 8]),
        ("max(max(Y), 2) + 0*Y", [4.0, 4.0, 4.0]),
        ("Ymax/max(Y)", [0.25, 0.25, 0.25]),
        ("Y^2 + ln(Z)", [1.0, 4.0, 16.0 + np.log(4.0)]),
        ("piecewise(1, Y > 1.5, 0)", [0.0, 1.0, 1.0]),
    ],
)
def test_reductions_and_petab_math(formula: str, expected: list[float]) -> None:
    variables = {
        "Y": np.array([1.0, 2.0, 4.0]),
        "Z": np.array([1.0, 1.0, 4.0]),
        "Ymax": np.array([1.0, 1.0, 1.0]),
    }
    np.testing.assert_allclose(evaluate_function(formula, variables), expected)


def test_a_reduction_ignores_the_padding() -> None:
    y = np.array([1.0, 2.0, np.nan])
    np.testing.assert_allclose(
        evaluate_function("Y/max(Y)", {"Y": y}), [0.5, 1.0, np.nan]
    )


def test_a_reduction_of_quantities_keeps_the_units() -> None:
    y = Q(np.array([1.0, 2.0, 4.0]), "mmol/l")
    normalized = evaluate_function("Y/max(Y)", {"Y": y})
    np.testing.assert_allclose(normalized.to("dimensionless").magnitude, [0.25, 0.5, 1.0])
    shifted = evaluate_function("Y - min(Y)", {"Y": y})
    assert str(shifted.units) == str(y.units)


def test_a_formula_of_parameters_is_a_number() -> None:
    assert evaluate_function("2 * k", {"k": 3.0}) == pytest.approx(6.0)


@pytest.mark.parametrize("formula", ["Y +", "max(Y", "foo(Y)"])
def test_invalid_math_is_reported(formula: str) -> None:
    with pytest.raises(ValueError):
        evaluate_function(formula, {"Y": np.array([1.0])})


def test_an_unknown_identifier_is_reported() -> None:
    with pytest.raises(ValueError, match="W"):
        evaluate_function("Y / W", {"Y": np.array([1.0])})
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest -q -n 0 tests/test_data_function.py`
Expected: FAIL with `ImportError: cannot import name 'evaluate_function'`.

- [ ] **Step 3: Add `CompiledFormula.apply`**

In `src/sbmlsim/simulator/formula.py`, add to `CompiledFormula` after `evaluate` (add `Any` to the `typing` imports):

```python
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
```

- [ ] **Step 4: Add `evaluate_function` to `data.py`**

In `src/sbmlsim/data.py` replace `from sbmlsim import mathml` with the imports `import re`, `from collections.abc import Callable, Mapping`, `from typing import Any`, `import numpy as np`, `from sbmlsim.simulator.formula import compile_formula` (merge with the existing import blocks, ruff sorts them), and add at module level, before `class Data`:

```python
#: a call of `max` or `min` which is not the end of a longer identifier
_REDUCTION_CALL = re.compile(r"(?<![A-Za-z0-9_])(max|min)\s*\(")

#: prefix of the symbol which stands for the value of a reduction
_REDUCTION_PREFIX = "sbmlsim_reduction__"

#: the reductions of a single argument, which ignore the padding of the data
_REDUCTIONS: dict[str, Callable[[Any], Any]] = {"max": np.nanmax, "min": np.nanmin}


def _closing_parenthesis(formula: str, start: int) -> int:
    """Find the parenthesis which closes the one opened before `start`.

    Raises:
        ValueError: if the parentheses of the formula are not balanced.
    """
    depth = 1
    for k in range(start, len(formula)):
        if formula[k] == "(":
            depth += 1
        elif formula[k] == ")":
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


def _replace_reductions(formula: str, values: dict[str, Any]) -> str:
    """Replace every `max` and `min` of a single argument by its value.

    The argument is evaluated on the data and reduced over it, the call is
    replaced by a symbol whose value is added to `values`. The arguments of a
    call are processed first, so an inner reduction is reduced first.
    """
    parts: list[str] = []
    position = 0
    while (match := _REDUCTION_CALL.search(formula, position)) is not None:
        end = _closing_parenthesis(formula, match.end())
        arguments = [
            _replace_reductions(argument, values)
            for argument in _split_arguments(formula[match.end() : end])
        ]
        parts.append(formula[position : match.start()])
        if len(arguments) == 1:
            count = sum(1 for key in values if key.startswith(_REDUCTION_PREFIX))
            symbol = f"{_REDUCTION_PREFIX}{count}"
            values[symbol] = _REDUCTIONS[match.group(1)](
                _evaluate(arguments[0], values)
            )
            parts.append(symbol)
        else:
            parts.append(f"{match.group(1)}({','.join(arguments)})")
        position = end + 1
    parts.append(formula[position:])
    return "".join(parts)


def _evaluate(formula: str, values: Mapping[str, Any]) -> Any:
    """Evaluate a formula of PEtab math without reductions on the values.

    Raises:
        ValueError: if the formula is not valid math or reads an identifier
            which has no value.
    """
    compiled = compile_formula(formula)
    missing = [symbol for symbol in compiled.symbols if symbol not in values]
    if missing:
        raise ValueError(
            f"The formula '{formula}' reads {missing}, which are neither "
            f"variables nor parameters of the data."
        )
    return compiled.apply([values[symbol] for symbol in compiled.symbols])


def evaluate_function(formula: str, variables: Mapping[str, Any]) -> Any:
    """Evaluate the formula of a `Data` of type FUNCTION on its data.

    The formula is the math of PEtab, see `sbmlsim.simulator.formula`, with
    one extension for data: `max` and `min` of a single argument reduce the
    argument over the data and ignore `NaN`, the padding of a scan, so
    `Y/max(Y)` is `Y` normalized to its maximum. With two or more arguments
    they are the elementwise maximum and minimum of PEtab.

    Args:
        formula: the formula.
        variables: the values of the identifiers of the formula, the arrays
            or quantities of the data and the numbers of the parameters.

    Returns:
        The value of the formula, a quantity if the variables are quantities.

    Raises:
        ValueError: if the formula is not valid math or reads an identifier
            which is not a variable.
    """
    values = dict(variables)
    return _evaluate(_replace_reductions(formula, values), values)
```

In `Data.get_data` replace the two lines with `mathml.formula_to_astnode(...)` and `mathml.evaluate(...)` by `x = evaluate_function(self.function, variables)` (keep the collection of `variables` and the wrapping of a non quantity result into `Q(x, "dimensionless")`).

If `import sbmlsim.data` raises a circular import, move `from sbmlsim.simulator.formula import compile_formula` into `_evaluate` with the comment `# the package of the simulator imports the models` as `model_roadrunner.py` does.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q -n 0 tests/test_data_function.py`
Expected: PASS. If `max(Y, 2)` or `max(Y, Z)` fails because lambdify maps `Max` to a numpy call which does not broadcast or does not accept quantities, pass `modules=[{"amax": _elementwise_max, "amin": _elementwise_min}, "numpy"]` in `compile_formula` only if the names lambdify emits are `amax`/`amin` (check with `sp.lambdify(..., modules="numpy")` and `inspect.getsource`), with `np.maximum.reduce(np.broadcast_arrays(*args))`; otherwise fix in `_replace_reductions` by rewriting `max(a, b)` to `piecewise(a, a >= b, b)`. Rerun until green, then run `uv run pytest -q tests/simulator` to confirm the simulator formulas are unchanged.

- [ ] **Step 6: Update the old tests, the examples and the docs**

Remove from `tests/test_mathml.py` the tests which call `mathml.evaluate` (they are covered by `tests/test_data_function.py`). Run the examples which use functions: `uv run pytest -q tests/examples -k "repressilator"`. In `docs/data.md` state after the code block: "The function is a formula of the math of PEtab, the same as the formulas of changes and observables. One extension serves data: `max` and `min` of a single argument reduce it over the data and ignore the `NaN` of padding, so `Y/max(Y)` normalizes `Y` to its maximum; with two or more arguments they are the elementwise maximum and minimum." Update `CLAUDE.md`, paragraph `mathml.py - formulas.`: the first sentence becomes "A `Data` of type `FUNCTION` (`data.py`) is a formula of PEtab math, compiled with `simulator.formula.compile_formula`; `evaluate_function` adds the reductions `max`/`min` of a single argument over the data."

- [ ] **Step 7: Full suite and commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`
```bash
git add -A src/sbmlsim/data.py src/sbmlsim/simulator/formula.py tests docs CLAUDE.md
git commit -m "The formula of a Data is the math of PEtab, max and min of one argument reduce over the data" -m "Data of type FUNCTION was parsed with libsedml as the L3 formula syntax and evaluated with a sympy evaluation of its own. It is now compiled with compile_formula like the formulas of changes and observables, so sbmlsim has one formula language. PEtab math has no reduction over an array; max and min of a single argument keep reducing the argument over the data, ignoring the NaN of padding, which the examples use to normalize curves."
```

---

### Task 4: The math of sbmlmath moves into `sciml`, `mathml.py` and libsedml are gone

**Files:**
- Create: `src/sbmlsim/sciml/formula.py` (the sbmlmath half of `mathml.py`, lines from `_MATHML_NAMESPACE` to the end before the `__main__` block, plus `TIME` and the imports it needs)
- Delete: `src/sbmlsim/mathml.py`, `tests/test_mathml.py`, `docs/api/mathml.md`
- Move: `tests/test_mathml_formula.py` to `tests/sciml/test_formula.py` (`git mv`), its import becomes `from sbmlsim.sciml.formula import ...`
- Modify: `src/sbmlsim/sciml/hybridization.py:36`, `src/sbmlsim/sciml/compiler.py:46`, `src/sbmlsim/fit/petab_v2/sciml.py:46`, `src/sbmlsim/fit/petab_v2/sciml_export.py:75`, `tests/sciml/test_export.py:33` (`sbmlsim.mathml` becomes `sbmlsim.sciml.formula`)
- Modify: `pyproject.toml` (remove `python-libsedml`, the dependency comment "SBML, COMBINE archives and formula parsing" becomes "SBML and its formulas"; the comment above `sbmlmath` names `sbmlsim.sciml.formula`), `uv.lock`, `zensical.toml` (nav `mathml` entry), `docs/api/index.md` (row of `mathml`), add `docs/api/sciml.formula.md` with `# sciml.formula` and `::: sbmlsim.sciml.formula` plus its nav entry in the sciml section and its row in `docs/api/index.md`, `CLAUDE.md` (paragraph `mathml.py - formulas.` becomes `**Formulas.**` and states the PEtab math of `simulator/formula.py`, `data.evaluate_function` and that `sciml/formula.py` reads and writes the math of SBML with sbmlmath for the network compiler)

**Interfaces:**
- Produces: `sbmlsim.sciml.formula` with `TIME`, `formula_expression`, `formula_symbols`, `evaluate_formula`, `expression_to_astnode`, `expression_to_formula`, unchanged signatures.

- [ ] **Step 1: Create the module**

`git mv` is not possible for half a file: create `src/sbmlsim/sciml/formula.py` with the module docstring "The math of SBML as sympy expressions, read with libsbml and sbmlmath, for the network compiler and the hybridization.", the imports the moved code uses (`functools`, `logging`, `Mapping`, `Any`, `libsbml`, `numpy`, `sympy`, `sbmlmath`, `sbmlmath.csymbol`, `from sympy import ...` only for what is used; ruff `F401` reports leftovers), `logger = logging.getLogger(__name__)`, `TIME = "time"` and the code from `_MATHML_NAMESPACE` up to the `__main__` block. Check that nothing in the moved code calls `expr_from_formula`, `replace_piecewise` or another function of the libsedml half: `rg -n "expr_from_formula|replace_piecewise|parse_formula|parse_astnode|formula_to_astnode" src/sbmlsim/sciml/formula.py` must be empty.

- [ ] **Step 2: Repoint the imports and delete `mathml.py`**

Run: `rg -l "sbmlsim.mathml|from sbmlsim import mathml" src tests examples docs scripts --glob '!docs/superpowers/**'` and change each to `sbmlsim.sciml.formula`. Delete `mathml.py`, `tests/test_mathml.py` (its remaining tests cover only the libsedml half), `docs/api/mathml.md`. `git mv tests/test_mathml_formula.py tests/sciml/test_formula.py` and fix its docstring reference to `sbmlsim.mathml.expr_from_formula` (drop the sentence).

- [ ] **Step 3: Drop libsedml**

Remove `"python-libsedml>=2.0.34",` from `pyproject.toml`, run `uv lock && uv sync --extra dev`, then `rg -n "libsedml" src tests examples docs scripts pyproject.toml --glob '!docs/superpowers/**'`.
Expected: no hit.

- [ ] **Step 4: Verify**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q && uv run zensical build --clean 2>&1 | tail -3`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add -A src tests docs zensical.toml pyproject.toml uv.lock CLAUDE.md
git commit -m "The math of SBML for the networks moves into sciml, mathml.py and the dependency on libsedml are removed" -m "After the formula of a Data became PEtab math, mathml.py held two unrelated parts: the libsedml evaluation nothing used any more, and the reading and writing of SBML math with sbmlmath which only the network compiler and the hybridization of sciml use. The latter is sbmlsim.sciml.formula."
```

---

### Task 5: Models are read from files and strings only, `ModelChange` is removed

**Files:**
- Modify: `src/sbmlsim/model/model_resources.py` (remove `import re`, `import requests`, `Union` if unused, `Source.is_path`, `is_urn`, `is_http`, `model_from_urn`, `model_from_url`, `parse_biomodels_mid`, `model_from_biomodels`, the branches `elif is_urn(...)` and `elif is_http(...)` of `Source.from_source`; module docstring "The source of a model, a file or the SBML itself.")
- Delete: `tests/models/test_biomodels.py`
- Modify: `src/sbmlsim/model/model.py` (`AbstractModel.SourceType` at about :32, `LanguageType.CELLML` at about :30)
- Delete: `src/sbmlsim/model/model_change.py`, `examples/model_change.py`, `tests/test_model_change.py`, `docs/api/model.model_change.md`
- Modify: `src/sbmlsim/model/__init__.py` (export of `ModelChange`), `src/sbmlsim/model/model_roadrunner.py` (`copy_roadrunner_model` at about :399), `tests/examples/test_example_scripts.py` (`examples.model_change`), `examples/README.md` (row), `zensical.toml`, `docs/api/index.md`, `CLAUDE.md` (the sentence "`ModelChange.clamp_species` changes a loaded roadrunner instance between simulations." and "(path relative to `base_path`, URL, BioModels URN, or the SBML itself)" becomes "(path relative to `base_path` or the SBML itself)")

- [ ] **Step 1: Confirm no caller**

Run: `rg -n "is_urn|is_http|model_from_|parse_biomodels|is_path\(|SourceType|CELLML|ModelChange|clamp_species|copy_roadrunner_model" src tests examples docs scripts --glob '!docs/superpowers/**'` and the same on `~/git/pkdb_models/pkdb_models`.
Expected: only the definitions, the files listed above and nothing in pkdb_models. A source string starting with `http` or `urn` now resolves as a path and fails with the existing `OSError` "Path ... does not exist".

- [ ] **Step 2: Add a test for the remaining sources**

Append to `tests/models/test_model_roadrunner.py`:

```python
def test_a_url_is_no_source(tmp_path: Path) -> None:
    """A model is read from a file or the SBML itself, not downloaded."""
    from sbmlsim.model.model_resources import Source

    with pytest.raises(OSError, match="does not exist"):
        Source.from_source("https://www.ebi.ac.uk/biomodels/BIOMD0000000012", base_dir=tmp_path)
```

(add `from pathlib import Path` and `import pytest` if the file lacks them). Run: `uv run pytest -q -n 0 tests/models/test_model_roadrunner.py -k url`. Expected before the change: FAIL (it downloads or raises another error); after: PASS.

- [ ] **Step 3: Delete and edit**

Apply the removals. `RoadrunnerSBMLModel` must not use `Source.is_path` (it uses `source.path is not None`); `rg -n "is_path" src` must be empty.

- [ ] **Step 4: Verify and commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q && uv run zensical build --clean 2>&1 | tail -3`
```bash
git add -A src tests examples docs zensical.toml CLAUDE.md
git commit -m "A model is read from a file or from its SBML, the downloads from BioModels and ModelChange are removed" -m "The sources from URLs and BioModels URNs imported requests, which is not a dependency, and were only tested against the network, where CI gets 403. ModelChange.clamp_species, copy_roadrunner_model, AbstractModel.SourceType and LanguageType.CELLML had no user."
```

---

### Task 6: Dead functions and the `__main__` blocks of library modules are removed

**Files:**
- Modify: `src/sbmlsim/plot/plotting.py` (`Figure.num_subplots` at about :1731, `Figure.from_plots` at about :1884), tests using `from_plots` (`rg -n "from_plots" tests`), `src/sbmlsim/result/xresult.py` (`is_ragged` about :199, `from_dfs` about :372, `from_netcdf` about :466) and their tests, `src/sbmlsim/simulation/scan.py` (`get_dimension` about :51), `src/sbmlsim/experiment/experiment.py` (`from_json` about :582), `src/sbmlsim/simulation/sensitivity.py` (`DistributionType`; `__main__` block about :274), `src/sbmlsim/utils.py` (`function_name` about :66), `src/sbmlsim/serialization.py` (`ObjectJSONEncoder.to_json` about :36), `src/sbmlsim/fit/helpers.py` (`mapping_kinds_info` about :323) with `tests/fit/test_mapping_kinds.py` (its tests of `mapping_kinds_info` only), `src/sbmlsim/fit/result.py` (`run_result` about :174) with `tests/fit/test_robustness.py::test_run_result`, `src/sbmlsim/units.py` (`__main__` block about :620), `src/sbmlsim/fit/cli.py` (`__main__` block about :19)

**Interfaces:**
- Keeps: `OptimizationResult.xopt_fit_parameters`, `SensitivityParameter.parameters_set_bounds`, `SensitivityParameter.parameter_to_latex`, `data.load_pkdb_dataframes_by_substance` (pkdb_models calls them), `identifiability.cost_threshold`, `likelihood.stencil`, `metrics.aic`/`bic`, `MappingSelection.kinds_of`.

- [ ] **Step 1: Confirm every name**

For each of `num_subplots from_plots is_ragged from_dfs from_netcdf get_dimension from_json DistributionType function_name mapping_kinds_info run_result` run `rg -n "\b<name>\b" src tests examples docs scripts --glob '!docs/superpowers/**'` and on `~/git/pkdb_models/pkdb_models`. `ObjectJSONEncoder.to_json`: `rg -n "\.to_json\(" src` must show no call on an encoder. `from_json` in pkdb_models is `serialization.from_json` and `dose_from_json`, not `SimulationExperiment.from_json`; keep `serialization.from_json`. Expected: callers only in their own tests. A name with another caller stays and is listed in the commit body.

- [ ] **Step 2: Delete the code, its tests and the `__main__` blocks**

A `__main__` block of `units.py` or `simulation/sensitivity.py` that demonstrates something not covered by an example is dropped without replacement; `fit/cli.py`'s block is dropped if `python -m sbmlsim.fit.cli` is not documented (`rg -n "sbmlsim.fit.cli" docs README.md`); if documented, keep it.

- [ ] **Step 3: Verify and commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`
```bash
git add -A src tests
git commit -m "The functions nothing calls and the __main__ blocks of the library modules are removed" -m "Figure.num_subplots and from_plots, XResult.is_ragged, from_dfs and from_netcdf, ScanSim.get_dimension, SimulationExperiment.from_json, which raised NotImplementedError, DistributionType, utils.function_name, ObjectJSONEncoder.to_json, mapping_kinds_info and OptimizationResult.run_result had no caller outside their own tests, neither in sbmlsim nor in pkdb_models."
```

---

### Task 7: The shims are removed, a `Data` is its selection

**Files:**
- Modify: `src/sbmlsim/data.py` (`Data.Symbols`, the parameter `symbol`, the FIXME block at about :61-73, the check at about :89, `selection`, `__repr__`, `to_dict`), `src/sbmlsim/fit/objects.py` (about :1151-1178 if it reads `symbol`), `src/sbmlsim/fit/runner.py` (the loop over `("fitting_type", ...)`, `("weighting_local", ...)` at about :232-239 and its `Raises:` lines), `src/sbmlsim/report/experiment_report.py` (`TEMPLATE_PATH = TEMPLATE_DIR` at :25, default `template_path: Path = TEMPLATE_DIR`; the FIXME at about :163 becomes the comment `# the results of several experiments as a list`)
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `Data(index: str, task=None, dataset=None, function=None, variables=None, parameters=None, sid=None)`; `Data.selection` is `index` as given; `Data.sid` and `Data.name` unchanged for every input.

- [ ] **Step 1: Pin the current identifiers in a test**

Append to `tests/test_data.py`:

```python
@pytest.mark.parametrize(
    ("index", "selection", "sid"),
    [
        ("[X]", "[X]", "task__X"),
        ("X", "X", "task__X"),
        ("time", "time", "task__time"),
    ],
)
def test_the_selection_and_sid_of_data(index: str, selection: str, sid: str) -> None:
    """A Data selects what its index names, its sid drops the brackets."""
    from sbmlsim.data import Data

    data = Data(index, task="task")
    assert data.selection == selection
    assert data.sid == sid
```

Run: `uv run pytest -q -n 0 tests/test_data.py -k selection_and_sid`
Expected: PASS before the change (it pins the behavior). Also run `uv run python -c "from sbmlsim.data import Data; d = Data('[X]', task='t'); print(d.name, d.to_dict())"` and note `name`.

- [ ] **Step 2: Remove `symbol`**

`Data.__init__` stores `self.index = index`; `selection` returns `self.index`; `sid` uses `self.index.strip("[]")` where it used `self.index` (task and dataset branches and the function branch keep their form); `name` must print the same as noted in Step 1 (adapt it if it used the stripped index). Remove `Data.Symbols`, the parameter `symbol`, the bracket check and `symbol` from `__repr__` and `to_dict`. `rg -n "\.symbol\b|Symbols\b" src/sbmlsim/data.py src/sbmlsim/fit/objects.py src/sbmlsim/experiment src/sbmlsim/plot` must show no use of the removed attribute; adapt `fit/objects.py` if it passes `symbol`.

- [ ] **Step 3: Remove the runner guard and the report aliases**

Delete the loop and the two `Raises:` lines on the removed parameters; `rg -n "TEMPLATE_PATH" src tests examples` must be empty after replacing the default.

- [ ] **Step 4: Verify and commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`
Expected: pass, including the pinning test.
```bash
git add -A src tests
git commit -m "The shims are removed: a Data is the selection it names, the guards of removed parameters and template aliases are gone" -m "Data encoded '[X]' as the index X with the symbol concentration and turned it back into the selection '[X]'; it now keeps the selection, its sid and name are unchanged. The runner no longer checks for fitting_type and weighting_local, which Python rejects as unexpected arguments anyway, and the report has one name for its template directory."
```

---

### Task 8: Stale examples, tests and fixtures are removed

**Files:**
- Delete: `examples/julia/`, `examples/interpolation/`, `tests/examples/test_interpolation.py`
- Modify: `tests/examples/test_example_scripts.py` (`examples.interpolation.interpolation_example` in `SCRIPTS`, the mention of `examples.julia` in the module docstring), `examples/README.md` (rows of `examples/interpolation/` and `examples/julia/`)
- Modify: `tests/test_sensitivity.py` (delete `test_sensitivity_example`, skipped with "no sensitivity support"), `tests/test_units.py` (delete `test_example_units`, the example runs in `test_example_scripts.py`; drop the then unused import of the example)
- Modify: `tests/test_data.py` (delete `TEST_PATH`, `DATA_DIR`, `MODEL_DIR`, `MODEL_*` if unused, they point at the missing `RESOURCES_DIR / "testdata"`)
- Delete: the fixtures of `tests/data` which nothing references (Step 2)

- [ ] **Step 1: Delete the examples and the duplicated tests**

Apply the deletions above. `rg -n "interpolation|julia" tests examples docs zensical.toml --glob '!docs/superpowers/**' --glob '!tests/data/**'` must show only unrelated uses (e.g. `interpolate` of `XResult`, `_area_interpolation_points`).

- [ ] **Step 2: Find the unreferenced fixtures**

Run:
```bash
cd /home/mkoenig/git/sbmlsim
for f in $(git ls-files tests/data/data tests/data/models tests/data/petab/icg_example1); do
  b=$(basename "$f"); s="${b%.*}"
  rg -q --fixed-strings "$s" src tests examples docs scripts --glob '!tests/data/**' --glob '!docs/superpowers/**' || echo "$f"
done
```
Expected: lists files like `numlData*.xml`, `reading-*.xml`, `oscli.*`, `parameter-from-data-csv.xml`, `asedml3repeat.xml`, `asedmlComplex.xml`, `app2sim.xml`, `BorisEJB.xml`, `curien.xml`, `lorenz.xml`, `BioModel1_*.xml`, `icg_example1/*`. `git rm` exactly the listed files; a file the loop does not list stays.

- [ ] **Step 3: Delete the untracked leftovers**

Run: `for d in src/sbmlsim/combine src/sbmlsim/interpolation tests/combine tests/interpolation tests/processing tests/comparison tests/data/combine tests/data/diff tests/data/data/omex; do [ -d "$d" ] && [ -z "$(git ls-files "$d")" ] && rm -rf "$d"; done; git status --short`
Expected: no change in `git status` from this step (the directories hold only `__pycache__` and are not tracked).

- [ ] **Step 4: Verify and commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`
```bash
git add -A examples tests
git commit -m "The examples on interpolation and julia, the skipped and duplicated tests and the fixtures of SED-ML are removed" -m "Neither example is about the core of sbmlsim. test_sensitivity_example was skipped, test_example_units ran an example the example tests run anyway, and the SED-ML and NuML fixtures in tests/data were referenced by nothing."
```

---

### Task 9: Unused dependencies are removed, pyyaml is declared

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, `CLAUDE.md` (the list of runtime dependencies in the first paragraph)

- [ ] **Step 1: Confirm**

Run: `rg -n "pkpdutils|sbml4humans|import yaml|from yaml" src tests examples scripts`
Expected: no `pkpdutils`/`sbml4humans`; `yaml` imported by `fit/petab_v2/testsuite.py`, `fit/petab_v2/benchmark.py`, `sciml/network.py`, `sciml/testsuite.py`.

- [ ] **Step 2: Edit `pyproject.toml`**

Remove `"pkpdutils>=1.3.0",` with its comment and `"sbml4humans>=0.12.2",` with its comment. Add `"pyyaml>=6.0.3",` to `dependencies` under a comment `# the files of the PEtab test suite, the benchmark collection and the networks`, and remove `"pyyaml>=6.0.3",` from the `sciml` extra. Run `uv lock && uv sync --extra dev`.

- [ ] **Step 3: Update `CLAUDE.md`**

In the first paragraph remove `pkpdutils` (and "(pharmacokinetic and pharmacodynamic analysis)") and `sbml4humans`, remove `python-libsedml` if still listed, add `pyyaml` next to `petab`.

- [ ] **Step 4: Verify and commit**

Run: `uv run ruff check && uv run ty check && uv run pytest -q`
```bash
git add pyproject.toml uv.lock CLAUDE.md
git commit -m "pkpdutils and sbml4humans are no longer dependencies, pyyaml is declared" -m "Nothing in sbmlsim imports pkpdutils or sbml4humans. pyyaml is imported by the PEtab test suite, the benchmark collection and the networks and was only installed through petab."
```

---

### Task 10: Importing sbmlsim no longer imports petab and torch

**Files:**
- Modify: `src/sbmlsim/simulator/formula.py` (import of `sympify_petab` inside `compile_formula`)
- Create: `tests/test_imports.py`
- Modify: any other module on the import path of `sbmlsim.experiment` or `sbmlsim.fit` which imports `petab` at the top (find them in Step 3)

- [ ] **Step 1: Write the failing test**

Create `tests/test_imports.py`:

```python
"""The core of sbmlsim imports neither petab nor torch.

petab.v2 imports its SciML extension and with it torch, which costs seconds in
every process: in every worker of the tests, of a parallel fit and in every
example. PEtab is imported when a formula is compiled or a PEtab problem is read.
"""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module", ["sbmlsim", "sbmlsim.simulator", "sbmlsim.experiment", "sbmlsim.fit"]
)
def test_the_core_does_not_import_petab(module: str) -> None:
    code = (
        f"import sys, {module}; "
        "print(sorted(m for m in ('petab', 'torch', 'petab_sciml') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
```

Run: `uv run pytest -q -n 0 tests/test_imports.py`
Expected: FAIL for `sbmlsim.simulator`, `sbmlsim.experiment`, `sbmlsim.fit` with `['petab', 'petab_sciml', 'torch']`.

- [ ] **Step 2: Import petab lazily in `formula.py`**

Remove `from petab.v2.math import sympify_petab` from the top; in `compile_formula` (which is cached with `functools.cache`, so the import runs once per formula and is cheap after the first) add before the `try`:

```python
    # petab.v2 imports its SciML extension and torch, which costs seconds;
    # only a simulation with a formula pays it
    from petab.v2.math import sympify_petab
```

- [ ] **Step 3: Find the other imports of petab**

Run: `uv run python -X importtime -c "import sbmlsim.fit" 2>&1 | grep -E "\| +(petab|torch)$|petab\.v2$" | head` and `uv run python -c "import sys, sbmlsim.fit; import importlib; [print(m) for m in sys.modules if m.startswith('sbmlsim') ]" | sort` to see which sbmlsim modules load, then `rg -n "^import petab|^from petab" <those modules>`. Move each import of petab in a module that `sbmlsim.fit` or `sbmlsim.experiment` loads into the function which needs it, with the same comment. `fit/petab_v2/*` modules which are only loaded by `to_petab`/`from_petab` keep their top level imports.

- [ ] **Step 4: Run**

Run: `uv run pytest -q -n 0 tests/test_imports.py && uv run ty check && uv run pytest -q`
Expected: PASS. Record `uv run python -X importtime -c "import sbmlsim.experiment" 2>&1 | tail -1` for the pull request.

- [ ] **Step 5: Commit**

```bash
git add -A src tests
git commit -m "Importing sbmlsim no longer imports petab and torch" -m "compile_formula imported petab.v2.math at the top of its module, and petab.v2 imports its SciML extension and with it torch: every import of sbmlsim.simulator took 2.2 s, in every test worker, every worker of a parallel fit and every example. petab is now imported when the first formula is compiled; a test guards the import of the core."
```

---

### Task 11: A fit with one worker runs without a process pool

**Files:**
- Modify: `src/sbmlsim/fit/runner.py` (`run_optimization`, about :262-300; docstring of `n_cores`)
- Test: `tests/fit/test_fit.py` (or the file where `run_optimization` is tested with `n_cores=1` and no `serial`)

**Interfaces:**
- Consumes: `run_optimization(problem, settings=None, size=5, algorithm=..., seed=None, n_cores=1, serial=False, show_progress=True, timeout=None, runs_dir=None, **kwargs) -> OptimizationResult`, fixtures `op_hctz_pk`, `fit_settings`, `short_fit` of `tests/fit/conftest.py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/fit/test_fit.py`:

```python
def test_one_worker_runs_without_a_pool(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    short_fit: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fit with one worker is serial and fits what a serial fit fits."""
    from sbmlsim.fit import runner

    serial = runner.run_optimization(
        problem=op_hctz_pk, settings=fit_settings, size=2, seed=1234,
        serial=True, show_progress=False, **short_fit,
    )

    def no_pool(**kwargs: Any) -> None:
        raise AssertionError("a fit with one worker started a pool")

    monkeypatch.setattr(runner, "_run_optimization_parallel", no_pool)
    one = runner.run_optimization(
        problem=op_hctz_pk, settings=fit_settings, size=2, seed=1234,
        n_cores=1, show_progress=False, **short_fit,
    )
    np.testing.assert_allclose(one.xopt, serial.xopt)
```

(import `Any`, `numpy as np`, `pytest`, `FitSettings`, `OptimizationProblem` at the top if missing). Run: `uv run pytest -q -n 0 tests/fit/test_fit.py -k one_worker`
Expected: FAIL with "a fit with one worker started a pool".

- [ ] **Step 2: Implement**

In `run_optimization` compute the workers before the branch and take the serial path for one worker:

```python
    # a worker without a repeat only costs the start of a process, and one
    # worker is a serial fit without the start of a process
    workers = 1 if serial else min(resolve_n_cores(n_cores), size)
    opt_result: OptimizationResult
    if workers <= 1:
        display.key_values({"runs": size, "workers": "1 (serial)"})
        ...  # the existing serial branch, unchanged
    else:
        ...  # the existing guard and parallel branch with n_cores=workers
```

Update the docstring of `n_cores`: "number of workers, `None` uses all available cores but one; one worker fits without starting a process."

- [ ] **Step 3: Keep the tests of the pool on the pool**

Run: `rg -n "n_cores=1" tests | grep -v serial` and read each test whose name or docstring speaks of a parallel fit, a pool, a worker, the guard or `GUARD_MESSAGE` (e.g. `test_fit_lsq_parallel`, `test_parallel_equals_serial`, the tests of a worker which fails to start). Set `n_cores=2` (with `size>=2`) in those, so they still test the pool. Leave the others, which now run faster.

- [ ] **Step 4: Run and commit**

Run: `uv run pytest -q -n 0 tests/fit/test_fit.py -k one_worker && uv run ty check && uv run pytest -q`
```bash
git add src/sbmlsim/fit/runner.py tests/fit
git commit -m "A fit with one worker runs serially instead of in a pool of one process" -m "run_optimization started a multiprocessing pool for n_cores=1, which costs the start of a process and the imports of sbmlsim in it, three times as much on Windows. One worker now takes the serial path, whose result is the same for the same seed; the tests of the pool use two workers."
```

---

### Task 12: The sensitivity analyses take a resolution, the examples run small in the tests

**Files:**
- Modify: `src/sbmlsim/sensitivity/analysis.py` (`SensitivityAnalysis.__init__` gets `dpi: int = 300`, stored as `self.dpi`), `src/sbmlsim/sensitivity/plots.py:124,216` (functions take `dpi: int = 300`), `src/sbmlsim/sensitivity/sensitivity_sampling.py:368`, `src/sbmlsim/sensitivity/sensitivity_morris.py:338` (use `self.dpi`), every call of the functions of `plots.py` in `sensitivity/` passes `dpi=self.dpi`
- Modify: `examples/sensitivity/sensitivity_example.py` (`--quick` passes `dpi=72`)
- Modify: `tests/examples/test_example_scripts.py` (`ARGUMENTS["examples.petab.benchmark"] = ["--no-identifiability"]`, after checking with `uv run python -m examples.petab.benchmark --help` which flags exist; add `--runs=2` if it has `--runs`)
- Test: `tests/sensitivity/test_analysis.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/sensitivity/test_analysis.py` (reuse the fixtures or imports of `tests/sensitivity/test_sensitivity_example.py`: `sensitivity_simulation`, `sensitivity_parameters`, `sensitivity_groups` from `examples.sensitivity.sensitivity_example`):

```python
def test_the_figures_have_the_resolution_of_the_analysis(tmp_path: Path) -> None:
    """The figures of an analysis are written at its resolution."""
    from matplotlib.image import imread

    from examples.sensitivity.sensitivity_example import (
        sensitivity_groups,
        sensitivity_parameters,
        sensitivity_simulation,
    )
    from sbmlsim.sensitivity import LocalSensitivityAnalysis

    analysis = LocalSensitivityAnalysis(
        sensitivity_simulation=sensitivity_simulation,
        parameters=sensitivity_parameters,
        groups=[sensitivity_groups[0]],
        results_path=tmp_path,
        difference=0.01,
        cache_results=False,
        n_cores=1,
        dpi=50,
    )
    analysis.execute()
    analysis.plot()
    pngs = sorted(tmp_path.rglob("*.png"))
    assert pngs
    small = imread(pngs[0]).shape
    analysis.dpi = 100
    analysis.plot()
    large = imread(sorted(tmp_path.rglob("*.png"))[0]).shape
    assert large[0] > 1.5 * small[0]
```

Check the constructor arguments of `LocalSensitivityAnalysis` in `tests/sensitivity/test_sensitivity_example.py` and use the same ones (only `dpi=50` is new). Run: `uv run pytest -q -n 0 tests/sensitivity/test_analysis.py -k resolution`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'dpi'`.

- [ ] **Step 2: Implement**

Add `dpi: int = 300` to `SensitivityAnalysis.__init__` (document it: "dpi: the resolution of the figures the analysis writes"), pass it through every subclass `__init__` that has its own signature (they forward `**kwargs` or list the arguments; add `dpi` where they list them), and replace each hardcoded `dpi=300` with `self.dpi` or the new `dpi` parameter of the plot function.

- [ ] **Step 3: Small examples in the tests**

In `examples/sensitivity/sensitivity_example.py` pass `dpi=72 if options.quick else 300` to every analysis it creates. Add the benchmark arguments to `ARGUMENTS`.

- [ ] **Step 4: Run, time and commit**

Run: `uv run pytest -q -n 0 tests/sensitivity && uv run pytest -q tests/examples -k "sensitivity or benchmark" --durations=5 && uv run ty check && uv run pytest -q`
Expected: pass; the durations of both examples are lower than in `$SCRATCH/baseline.txt` (22 s and 15 s on 20 cores).
```bash
git add -A src/sbmlsim/sensitivity examples/sensitivity tests
git commit -m "The sensitivity analyses take the resolution of their figures, the examples run small in the tests" -m "The figures were always written at 300 dpi, which was 70 percent of the quick run of the sensitivity example in the tests. The test of the benchmark example no longer analyses the identifiability."
```

---

### Task 13: The fit tests build their problems once, the timing assertions do not flake

**Files:**
- Modify: `tests/fit/conftest.py`
- Modify: `tests/model/test_model_roadrunner_init.py::test_initialize_is_fast` (about :98-105), `tests/sciml/test_compiler.py` (the load time assertions with `SOFTMAX_LOAD_TIME` and the like, about :350-365)

- [ ] **Step 1: Measure the fixtures**

Run: `uv run pytest -q -n 0 tests/fit --durations=0 2>&1 | grep -E "setup" | sort -k1 -nr | head -15`
Note the setup time per fixture of `op_hctz_pk`, `op_hctz_iv`, `definition_hctz_pk`, `definition_hctz_iv`.

- [ ] **Step 2: Build once, hand out copies**

For a fixture whose setup takes more than 0.2 s: add a `scope="session"` fixture that builds the object (e.g. `_op_hctz_pk_template`) and make the existing function scoped fixture return `copy.deepcopy(template)` if `copy.deepcopy` is faster than building (measure with `timeit` in a scratch script; an `OptimizationProblem` which is not initialized deep copies its definition). Keep the names and the docstrings of the existing fixtures, tests do not change. If deep copying is not faster, leave the fixture as it is.

- [ ] **Step 3: Make the timing assertions robust**

`test_initialize_is_fast` asserts "an initialization does not regenerate the model": measure one regeneration (`model.r_loaded.regenerateModel()` or a new `roadrunner.RoadRunner(sbml)` load, whichever the module documents as the 0.3 to 0.5 s cost) in the same test and assert the mean initialization is below a tenth of it, instead of the absolute 5 ms. For `tests/sciml/test_compiler.py`, raise the load time bound to three times the measured local time (the test documents that the bound guards against a model which grows with the square of the units; a factor of three still catches that) and state the reason in a comment.

- [ ] **Step 4: Run, time and commit**

Run: `uv run pytest -q tests/fit tests/model tests/sciml/test_compiler.py --durations=10 && uv run pytest -q`
```bash
git add tests
git commit -m "The fit tests build their problems once and the assertions on wall time compare against a reference" -m "The problems of the HCTZ fit were built for every test; a test now gets a copy of a problem built once. test_initialize_is_fast compared against 5 ms and the softmax load of the network compiler against 10 s, both failed on a loaded runner."
```

---

### Task 14: One job of the matrix saves the cache of uv

**Files:**
- Modify: `.github/actions/setup/action.yml` (input `save-cache`, passed to `astral-sh/setup-uv`)
- Modify: `.github/workflows/ci-cd.yml`, `.github/workflows/lint.yml`, `.github/workflows/docs.yml` (pass `save-cache` where the cache is saved)

- [ ] **Step 1: Check the inputs of setup-uv**

Run: `curl -sL https://raw.githubusercontent.com/astral-sh/setup-uv/v10.2.0/action.yml | grep -n -A3 "save-cache\|cache-suffix\|cache-dependency-glob"`
Expected: an input `save-cache` (boolean). If it does not exist, use `cache-suffix: ${{ github.job }}-${{ inputs.python-version }}` instead, so that the jobs do not race for one key, and skip Step 2's `save-cache`.

- [ ] **Step 2: Add the input**

In `action.yml`:

```yaml
  save-cache:
    description: >-
      whether the job saves the cache of uv; jobs which share a key would race
      for it and warn "Unable to reserve cache"
    required: false
    default: "false"
```

and `save-cache: ${{ inputs.save-cache }}` in the `with:` of setup-uv. In `ci-cd.yml` the `test` job passes `save-cache: "true"` (each matrix entry has its own os and python, so its own key); the other jobs keep the default and restore only.

- [ ] **Step 3: Validate and commit**

Run: `uv run pre-commit run check-yaml --all-files` (or `uv run python -c "import yaml,sys; [yaml.safe_load(open(f)) for f in sys.argv[1:]]" .github/actions/setup/action.yml .github/workflows/*.yml`)
```bash
git add .github
git commit -m "Only the jobs of the test matrix save the cache of uv" -m "Every job saved the cache under the key it shares with other jobs, which failed with Unable to reserve cache and cost up to 37 s in the post step. The other jobs restore it."
```

The effect is checked on the pull request in Task 16.

---

### Task 15: The plans are not published, the documentation describes the core

**Files:**
- Modify: `zensical.toml` (exclude `superpowers/`), `docs/api/index.md`, `CLAUDE.md`, `examples/README.md`

- [ ] **Step 1: Exclude the plans**

Zensical validates an `exclude` plugin with the options `enabled`, `glob`, `regex` (see `.venv/lib/python3.14/site-packages/zensical/config.py` around line 1960). Add to `zensical.toml` in the `[project.plugins]` table (create the table if missing, following the existing TOML structure; check how other plugins such as `mkdocstrings` are configured in the file):

```toml
[project.plugins.exclude]
glob = ["superpowers/*", "superpowers/**"]
```

Run: `uv run zensical build --clean 2>&1 | tail -3 && ls site | grep -c superpowers`
Expected: build passes, `0`. If the plugin form is rejected, read `config.py` lines 1950-1990 for the accepted form and use it.

- [ ] **Step 2: Sweep the documentation**

Run: `rg -n "mathml|libsedml|SED-ML|sedml|DataGenerator|datagenerator|ModelChange|clamp_species|BioModels|urn|interpolation|julia|pkpdutils|sbml4humans|reports\(\)" docs CLAUDE.md README.md examples/README.md --glob '!docs/superpowers/**'`
Fix every hit which describes removed code. The historical sentence in `CLAUDE.md` "SED-ML and COMBINE archive support (`combine/`) was removed in 0.7.0, a fit is exchanged as a PEtab problem instead." stays. Check that every page in `docs/api/` has a nav entry and every nav entry a page: `uv run zensical build --clean 2>&1 | grep -i "warn\|not found"` must be empty.

- [ ] **Step 3: Commit**

```bash
git add -A zensical.toml docs CLAUDE.md README.md examples/README.md
git commit -m "The plans and specs are no longer published with the documentation, which describes the core" -m "docs/superpowers holds the plans and designs of the development; it was built into the site because nothing excluded it."
```

---

### Task 16: Verification, measurement and the pull request

- [ ] **Step 1: The checks**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q && uv run zensical build --clean 2>&1 | tail -2`
Expected: all pass.

- [ ] **Step 2: The suites deselected by default**

Run, each only if its cases are cached (`ls ~/.cache/sbmlsim`): `uv run pytest -q -m testsuite`, `uv run pytest -q -m petab_testsuite tests/fit`, `uv run pytest -q -m sciml_testsuite`. Expected: pass against their baselines. If a cache is missing, run the tox environment (`tox r -e testsuite`, `tox r -e petab`, `tox r -e sciml`), which downloads the cases.

- [ ] **Step 3: pkdb_models**

Run: `uv run python $SCRATCH/pkdb_imports.py $SCRATCH/pkdb_live_after.txt && diff $SCRATCH/pkdb_live_before.txt $SCRATCH/pkdb_live_after.txt`
Expected: no line lost. A lost file means a removal broke a live import: restore the name in a fixup commit before continuing.

- [ ] **Step 4: Measure**

Run the commands of Task 0 Step 1 and Step 2 again; write a before/after table (pytest wall time on 4 CPUs, number of tests, import time of `sbmlsim.experiment`, lines of `src/sbmlsim` and of `tests`).

- [ ] **Step 5: Push and open the pull request**

```bash
git push -u origin core-cleanup
gh-axi pr create --base develop --head core-cleanup --title "Cleanup of sbmlsim to its core (#250)" --body-file $SCRATCH/pr_body.md
```

`$SCRATCH/pr_body.md` (normal prose, no agent attribution, no em dash): a summary of what is removed per commit, the decisions (PEtab math for `Data` with the reductions, what stays and why), the before/after table, the note that the branch is stacked on #251 so its first commits are those of #251 until #251 is merged, "Closes #250", and the follow-up issue to open for the migration of pkdb_models to the simulation engine.

- [ ] **Step 6: CI**

Watch the checks of the pull request (`gh-axi pr checks <number>`). A failing job is debugged from its log (`gh-axi run view <id> --log-failed`), fixed in a new commit and pushed. Done when every required check passes. Compare the wall time of the Windows test job with the baseline (6.5 to 8.5 min) and add it to the pull request description (`gh-axi pr edit <number> --body-file ...`).
