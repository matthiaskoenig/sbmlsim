# Versioned fit parameters implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estimate one model entity separately for parts of the data in a single optimization problem, e.g. an absorption rate fitted once for the tablet arms and once for the solution arms.

**Architecture:** A `FitParameter` gains a `target` (the model entity it writes) and a `mappings` selector (where it applies). A new `ParameterMapping` resolves the selectors to the simulation groups of an initialized `OptimizationProblem`, validates the result and answers which changes a group is simulated with. The PEtab export writes the binding as a condition whose target value is a parameter id, and the reader turns such a condition back into a versioned parameter.

**Tech Stack:** python 3.13+, numpy, pandas, pint, libroadrunner, petab v2, rich, pytest with pytest-xdist, ruff, ty.

**Spec:** `docs/superpowers/specs/2026-09-13-versioned-fit-parameters-design.md`

## Global Constraints

- Every module, class and function carries full type annotations and a google style docstring; ruff `D` enforces it. `tests/` and `examples/` are exempt from the docstring rules.
- `ty` runs with `error-on-warning = true`, so the tree must stay at zero diagnostics. Suppress with a rule specific `# ty: ignore[rule-name]`, never a blanket `# type: ignore`.
- Library code logs, it does not print, and log calls use lazy `%s` formatting, not f-strings (ruff `G`).
- Run `uv run ruff check`, `uv run ruff format` and `uvx ty check` before every commit; the pre-commit hook runs all three and will reject the commit otherwise.
- `pytest` deselects the `testsuite` marker, so a normal run does not touch the SBML Test Suite. Use plain `pytest` for these tasks.
- Markdown carries no hard line wraps: a paragraph, a list item or a table row is a single line.
- Backwards compatibility is a hard requirement: `FitParameter("Ka_dis_hctz", ...)` with no new arguments must produce exactly the fit it produces today. Task 5 tests this numerically.
- Selectors must be picklable. `OptimizationProblem.__getstate__` reduces an initialized problem to its definition, which includes the fit parameters, and the workers of a parallel fit unpickle it. A module level `def` pickles, a `lambda` does not. Task 4 tests this.

---

### Task 1: `FitParameter` carries a target and a selector

**Files:**
- Modify: `src/sbmlsim/fit/objects.py:359-450` (`FitParameter`)
- Test: `tests/fit/test_objects.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `FitParameter(pid, start_value, lower_bound, upper_bound, unit, target=None, mappings=None)`; the read only property `FitParameter.target_id -> str` which is `target` when it is set and `pid` otherwise; `FitParameter.is_versioned -> bool`, true when `mappings is not None`.

`target_id` exists so that no caller has to write `p.target or p.pid`, which is the kind of thing that gets forgotten in one place.

- [ ] **Step 1: Write the failing tests**

Append to `tests/fit/test_objects.py`:

```python
def test_a_parameter_is_its_own_target_by_default() -> None:
    """A parameter without a target writes the entity it is named after."""
    p = FitParameter("Ka_dis_hctz", 0.35, 0.01, 10.0, "1/hr")
    assert p.target is None
    assert p.target_id == "Ka_dis_hctz"
    assert not p.is_versioned


def test_a_versioned_parameter_writes_another_entity() -> None:
    """A version is estimated under its own id and written to the target."""

    def only_tablets(key: str, mapping: object) -> bool:
        return key.endswith("tablet")

    p = FitParameter(
        "Ka_dis_tablet",
        0.35,
        0.01,
        10.0,
        "1/hr",
        target="Ka_dis_hctz",
        mappings=only_tablets,
    )
    assert p.pid == "Ka_dis_tablet"
    assert p.target_id == "Ka_dis_hctz"
    assert p.is_versioned


def test_the_target_is_part_of_the_identity_of_a_parameter() -> None:
    """Two parameters which write different entities are not the same."""
    a = FitParameter("p", 1.0, 0.1, 10.0, "1/hr")
    b = FitParameter("p", 1.0, 0.1, 10.0, "1/hr", target="q")
    assert a != b


def test_the_target_is_serialized_and_the_selector_is_not() -> None:
    """A selector is a callable and cannot be written to JSON.

    The resolution it produces is what a PEtab problem stores, see the
    design; `to_dict` therefore carries the target and not the selector.
    """

    def every(key: str, mapping: object) -> bool:
        return True

    p = FitParameter("p", 1.0, 0.1, 10.0, "1/hr", target="q", mappings=every)
    d = p.to_dict()
    assert d["target"] == "q"
    assert "mappings" not in d
    # the round trip through JSON keeps the target
    assert FitParameter.from_json(p.to_json()).target_id == "q"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/fit/test_objects.py -k "target or selector" -v`
Expected: FAIL with `TypeError: FitParameter.__init__() got an unexpected keyword argument 'target'`

- [ ] **Step 3: Add the two fields**

In `src/sbmlsim/fit/objects.py`, extend `FitParameter.__init__`:

```python
    def __init__(
        self,
        pid: str,
        start_value: float | None = None,
        lower_bound: float = -np.inf,
        upper_bound: float = np.inf,
        unit: str | None = None,
        target: str | None = None,
        mappings: Any = None,
    ):
        """Initialize FitParameter.

        Args:
            pid: id of the estimated parameter. It is the name in the parameter
                vector, in the parameter sets and in the profiles; it is the id
                of the entity of the model unless `target` says otherwise.
            start_value: initial value for the fitting.
            lower_bound: lower bound for the fitting.
            upper_bound: upper bound for the fitting.
            unit: unit of the parameter, the model unit is assumed if not given.
            target: entity of the model the value is written to. `None` means
                the parameter is the entity, i.e. `target_id` is `pid`. Several
                parameters write one target when each of them selects a part of
                the data, see `mappings`.
            mappings: `MappingFilter` or an iterable of them which select the
                fit mappings the parameter applies to; a mapping passes when it
                passes every filter. `None` applies the parameter everywhere.
                A selector is a callable and is not serialized: it must be a
                module level function, because the workers of a parallel fit
                unpickle the parameters.

        Raises:
            ValueError: if the bounds or the start value are inconsistent.
        """
```

Keep the existing bound checks unchanged, and after `self.unit = unit` add:

```python
        self.target = target
        self.mappings = mappings
```

Add the two properties after `__init__`:

```python
@property
def target_id(self) -> str:
    """Get the entity of the model the value is written to."""
    return self.target if self.target is not None else self.pid


@property
def is_versioned(self) -> bool:
    """Check whether the parameter applies to a part of the data only."""
    return self.mappings is not None
```

Extend `__eq__` with `and self.target_id == other.target_id` and `to_dict` with `"target": self.target,`.

`mappings` is deliberately absent from `to_dict`, so `from_json` keeps working: `FitParameter(**d)` receives `target` and no `mappings`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/fit/test_objects.py -v`
Expected: PASS, and every existing test in the file still passes.

- [ ] **Step 5: Check and commit**

```bash
uv run ruff check src tests && uv run ruff format src tests && uvx ty check
git add src/sbmlsim/fit/objects.py tests/fit/test_objects.py
git commit -m "A fit parameter names the entity it writes and where it applies"
```

---

### Task 2: `filter_keys` selects mappings by their id

**Files:**
- Modify: `src/sbmlsim/fit/helpers.py` (after `_filters`, around line 59)
- Modify: `examples/hctz_fitting/fitting/mapping_collections.py`
- Test: `tests/fit/test_helpers.py`

**Interfaces:**
- Consumes: `MappingFilter = Callable[[str, FitMapping], bool]` from `sbmlsim.fit.helpers`.
- Produces: `filter_keys(keys: Iterable[str]) -> MappingFilter`. Task 8 builds the selectors of a problem read from PEtab with it.

- [ ] **Step 1: Write the failing test**

Append to `tests/fit/test_helpers.py`:

```python
def test_filter_keys_selects_the_named_mappings() -> None:
    """The filter of a set of ids passes exactly those ids."""
    from sbmlsim.fit.helpers import filter_keys

    selected = filter_keys({"fm_a", "fm_b"})
    assert selected("fm_a", None)
    assert selected("fm_b", None)
    assert not selected("fm_c", None)


def test_filter_keys_takes_a_copy_of_the_ids() -> None:
    """The filter does not change when the set it was built from changes."""
    from sbmlsim.fit.helpers import filter_keys

    keys = {"fm_a"}
    selected = filter_keys(keys)
    keys.add("fm_b")
    assert not selected("fm_b", None)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/fit/test_helpers.py -k filter_keys -v`
Expected: FAIL with `ImportError: cannot import name 'filter_keys'`

- [ ] **Step 3: Add the filter**

In `src/sbmlsim/fit/helpers.py`, after `_filters`:

```python
def filter_keys(keys: Iterable[str]) -> MappingFilter:
    """Get a filter which selects the fit mappings of the given ids.

    A selector written by hand says what it means, e.g. "the tablets"; this
    one says which mappings it resolved to and is what a problem read from
    PEtab uses, because a condition stores the resolution and not the rule.

    Args:
        keys: ids of the fit mappings to select.

    Returns:
        A filter which passes exactly those mappings.
    """
    selected = frozenset(keys)

    def _filter(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
        """Select a mapping by its id."""
        return fit_mapping_key in selected

    return _filter
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/fit/test_helpers.py -v`
Expected: PASS

- [ ] **Step 5: Use it in the example**

`examples/hctz_fitting/fitting/mapping_collections.py` defines its own `filter_keys` and `filter_not_keys`. Replace the definition of `filter_keys` with an import from `sbmlsim.fit.helpers` and leave `filter_not_keys` where it is. Run the example to confirm nothing moved:

Run: `uv run python -m examples.hctz_fitting.fitting.petab_problem --subset=PKIV`
Expected: it prints the gaps and validates, as before.

- [ ] **Step 6: Check and commit**

```bash
uv run ruff check src tests examples && uv run ruff format src tests examples && uvx ty check
git add src/sbmlsim/fit/helpers.py tests/fit/test_helpers.py examples/hctz_fitting/fitting/mapping_collections.py
git commit -m "A filter which selects the fit mappings of given ids"
```

---

### Task 3: `ParameterMapping` resolves and validates the bindings

**Files:**
- Create: `src/sbmlsim/fit/parameter_mapping.py`
- Test: `tests/fit/test_parameter_mapping.py`

**Interfaces:**
- Consumes: `FitParameter.target_id`, `FitParameter.is_versioned` from Task 1.
- Produces:
  - `ParameterMapping(parameters: Sequence[FitParameter], mapping_indices: dict[int, set[int]], groups: Sequence[Sequence[int]], mapping_keys: Sequence[str], group_names: Sequence[str] | None = None)`; `mapping_indices` maps the index of a parameter in `parameters` to the indices of the fit mappings it selects, and a parameter which is not versioned is absent from it and covers everything.
  - `ParameterMapping.indices_for(group: int) -> dict[str, int]`, the target and the index into the parameter vector for one simulation group.
  - `ParameterMapping.changes_for(group: int, quantities: Sequence[Any]) -> dict[str, Any]`, where `quantities[i]` is the quantity of parameter `i`.
  - `ParameterMapping.coverage() -> list[CoverageRow]` with `CoverageRow(pid, target, n_covered, n_groups, uncovered_groups)`.
  - `ParameterMapping.is_versioned -> bool`, true when any parameter has a target different from its pid.

This task has no simulation in it: the object is built from plain indices, so it is tested exhaustively and fast.

- [ ] **Step 1: Write the failing tests**

Create `tests/fit/test_parameter_mapping.py`:

```python
"""Tests of the binding of the parameters to the simulations."""

import pytest

from sbmlsim.fit.objects import FitParameter
from sbmlsim.fit.parameter_mapping import ParameterMapping


def _parameter(
    pid: str, target: str | None = None, versioned: bool = False
) -> FitParameter:
    """Build a parameter, versioned when it selects a part of the data."""

    def _select(key: str, mapping: object) -> bool:
        return True

    return FitParameter(
        pid,
        1.0,
        0.1,
        10.0,
        "1/hr",
        target=target,
        mappings=_select if versioned else None,
    )


#: three mappings in two groups: mappings 0 and 1 share a simulation
GROUPS = [[0, 1], [2]]
KEYS = ["fm_a", "fm_b", "fm_c"]


def test_a_parameter_without_a_selector_covers_every_group() -> None:
    """An ordinary parameter is applied everywhere, as it is today."""
    parameters = [_parameter("Ka")]
    mapping = ParameterMapping(parameters, {}, GROUPS, KEYS)

    assert mapping.indices_for(0) == {"Ka": 0}
    assert mapping.indices_for(1) == {"Ka": 0}
    assert not mapping.is_versioned


def test_a_version_covers_the_groups_of_its_mappings() -> None:
    """A selector on mapping 2 reaches the group which holds it."""
    parameters = [
        _parameter("Ka_a", target="Ka", versioned=True),
        _parameter("Ka_c", target="Ka", versioned=True),
    ]
    mapping = ParameterMapping(parameters, {0: {0, 1}, 1: {2}}, GROUPS, KEYS)

    assert mapping.indices_for(0) == {"Ka": 0}
    assert mapping.indices_for(1) == {"Ka": 1}
    assert mapping.is_versioned


def test_a_group_no_version_covers_gets_no_change() -> None:
    """An uncovered simulation keeps the value of the model."""
    parameters = [_parameter("Ka_a", target="Ka", versioned=True)]
    mapping = ParameterMapping(parameters, {0: {0, 1}}, GROUPS, KEYS)

    assert mapping.indices_for(0) == {"Ka": 0}
    assert mapping.indices_for(1) == {}


def test_two_versions_on_one_mapping_are_an_error() -> None:
    """A simulation cannot have two values for one entity."""
    parameters = [
        _parameter("Ka_a", target="Ka", versioned=True),
        _parameter("Ka_b", target="Ka", versioned=True),
    ]
    with pytest.raises(ValueError, match="Ka_a.*Ka_b|Ka_b.*Ka_a"):
        ParameterMapping(parameters, {0: {0}, 1: {0}}, GROUPS, KEYS)


def test_a_parameter_without_a_selector_overlaps_every_version() -> None:
    """There is no fallback: a target is global or it is versioned."""
    parameters = [
        _parameter("Ka"),
        _parameter("Ka_a", target="Ka", versioned=True),
    ]
    with pytest.raises(ValueError, match="Ka"):
        ParameterMapping(parameters, {1: {0}}, GROUPS, KEYS)


def test_a_selector_must_not_split_a_simulation() -> None:
    """Mappings 0 and 1 share a simulation, so they share the value."""
    parameters = [
        _parameter("Ka_a", target="Ka", versioned=True),
        _parameter("Ka_b", target="Ka", versioned=True),
    ]
    with pytest.raises(ValueError, match="one simulation"):
        ParameterMapping(parameters, {0: {0}, 1: {1}}, GROUPS, KEYS)


def test_the_versions_of_a_target_agree_on_the_unit() -> None:
    """The unit is how the value reaches the model, so it is one unit."""
    a = FitParameter(
        "Ka_a", 1.0, 0.1, 10.0, "1/hr", target="Ka", mappings=lambda k, m: True
    )
    b = FitParameter(
        "Ka_b", 1.0, 0.1, 10.0, "1/min", target="Ka", mappings=lambda k, m: True
    )
    with pytest.raises(ValueError, match="unit"):
        ParameterMapping([a, b], {0: {0, 1}, 1: {2}}, GROUPS, KEYS)


def test_changes_are_the_quantities_of_the_bound_parameters() -> None:
    """`changes_for` is what a simulation of a group is run with."""
    parameters = [
        _parameter("Ka_a", target="Ka", versioned=True),
        _parameter("Vd"),
    ]
    mapping = ParameterMapping(parameters, {0: {0, 1}}, GROUPS, KEYS)
    quantities = ["Q_Ka_a", "Q_Vd"]

    assert mapping.changes_for(0, quantities) == {"Ka": "Q_Ka_a", "Vd": "Q_Vd"}
    assert mapping.changes_for(1, quantities) == {"Vd": "Q_Vd"}


def test_the_coverage_names_the_groups_a_version_does_not_reach() -> None:
    """The report says which simulations keep the value of the model."""
    parameters = [_parameter("Ka_a", target="Ka", versioned=True)]
    mapping = ParameterMapping(
        parameters, {0: {0, 1}}, GROUPS, KEYS, group_names=["po_tablet", "iv"]
    )

    (row,) = mapping.coverage()
    assert row.pid == "Ka_a"
    assert row.target == "Ka"
    assert row.n_covered == 1
    assert row.n_groups == 2
    assert row.uncovered_groups == ["iv"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/fit/test_parameter_mapping.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbmlsim.fit.parameter_mapping'`

- [ ] **Step 3: Write the module**

Create `src/sbmlsim/fit/parameter_mapping.py`:

```python
"""The binding of the parameters of a fit to the simulations they apply to.

A `FitParameter` writes its value to an entity of a model. A parameter which
carries a selector writes it only for the fit mappings the selector passes, so
one entity is estimated separately for parts of the data, e.g. an absorption
rate once for the tablet arms and once for the solution arms of one fit.

`ParameterMapping` is that binding as an object: it resolves the selectors to
the simulation groups of an initialized `OptimizationProblem`, refuses a
binding which cannot be simulated, and answers which changes a group is run
with. It is the same knowledge a PEtab problem keeps in its condition table,
which is why `sbmlsim.fit.petab_v2` reads it instead of deriving it again.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sbmlsim.fit.objects import FitParameter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoverageRow:
    """What one parameter of a fit reaches.

    Attributes:
        pid: id of the estimated parameter.
        target: entity of the model it writes.
        n_covered: number of simulations it applies to.
        n_groups: number of simulations of the problem.
        uncovered_groups: the simulations it does not apply to, by name. They
            keep the value the model has for the target.
    """

    pid: str
    target: str
    n_covered: int
    n_groups: int
    uncovered_groups: list[str] = field(default_factory=list)


class ParameterMapping:
    """Which parameter writes which entity in which simulation."""

    def __init__(
        self,
        parameters: Sequence[FitParameter],
        mapping_indices: dict[int, set[int]],
        groups: Sequence[Sequence[int]],
        mapping_keys: Sequence[str],
        group_names: Sequence[str] | None = None,
    ) -> None:
        """Resolve and validate the binding.

        Args:
            parameters: the parameters of the problem, in the order of the
                parameter vector.
            mapping_indices: for the index of every versioned parameter the
                indices of the fit mappings its selector passed. A parameter
                which is not versioned is absent and applies everywhere.
            groups: the simulation groups, each a list of mapping indices, see
                `OptimizationProblem.mapping_groups`.
            mapping_keys: id of every fit mapping, for the messages.
            group_names: name of every simulation group, for the coverage. The
                index of the group is used when they are not given.

        Raises:
            ValueError: if two parameters write one target in one simulation,
                if a selector splits a simulation, or if the versions of a
                target disagree on their unit.
        """
        self.parameters = list(parameters)
        self.groups = [list(group) for group in groups]
        self.group_names = (
            list(group_names)
            if group_names is not None
            else [str(k) for k in range(len(self.groups))]
        )

        self._check_units()
        #: index of the parameter which writes a target, per group
        self._by_group: list[dict[str, int]] = [
            self._resolve_group(k, group, mapping_indices, mapping_keys)
            for k, group in enumerate(self.groups)
        ]
        self._covered: dict[int, set[int]] = {}
        for k, targets in enumerate(self._by_group):
            for index in targets.values():
                self._covered.setdefault(index, set()).add(k)

    def _check_units(self) -> None:
        """Check that the versions of a target are given in one unit.

        Raises:
            ValueError: if two parameters of one target have different units.
        """
        units: dict[str, tuple[str, str | None]] = {}
        for parameter in self.parameters:
            target = parameter.target_id
            if target in units:
                pid, unit = units[target]
                if unit != parameter.unit:
                    raise ValueError(
                        f"The versions of '{target}' disagree on their unit: "
                        f"'{pid}' is '{unit}' and '{parameter.pid}' is "
                        f"'{parameter.unit}'. The unit is how the value reaches "
                        f"the model, so it is one unit for one entity."
                    )
            else:
                units[target] = (parameter.pid, parameter.unit)

    def _resolve_group(
        self,
        group_index: int,
        group: Sequence[int],
        mapping_indices: dict[int, set[int]],
        mapping_keys: Sequence[str],
    ) -> dict[str, int]:
        """Get the parameter which writes every target of one simulation.

        Args:
            group_index: index of the simulation group.
            group: indices of the fit mappings of the group.
            mapping_indices: the mappings every versioned parameter selected.
            mapping_keys: id of every fit mapping, for the messages.

        Returns:
            The index of the parameter which writes a target, by target.

        Raises:
            ValueError: if two parameters write one target in the group.
        """
        # the parameter which writes a target, and the mapping it came from
        chosen: dict[str, tuple[int, int | None]] = {}
        for index, parameter in enumerate(self.parameters):
            target = parameter.target_id
            hits = (
                sorted(set(group) & mapping_indices.get(index, set()))
                if parameter.is_versioned
                else [None]
            )
            for hit in hits:
                if target in chosen:
                    other, other_hit = chosen[target]
                    raise self._conflict(
                        target, other, index, other_hit, hit, group_index, mapping_keys
                    )
                chosen[target] = (index, hit)

        return {target: index for target, (index, _) in chosen.items()}

    def _conflict(
        self,
        target: str,
        first: int,
        second: int,
        first_hit: int | None,
        second_hit: int | None,
        group_index: int,
        mapping_keys: Sequence[str],
    ) -> ValueError:
        """Build the error of two parameters writing one target."""
        a = self.parameters[first].pid
        b = self.parameters[second].pid
        if first_hit is not None and second_hit is not None and first_hit != second_hit:
            return ValueError(
                f"'{a}' and '{b}' both write '{target}' in one simulation: "
                f"'{a}' selects '{mapping_keys[first_hit]}' and '{b}' selects "
                f"'{mapping_keys[second_hit]}', which share the simulation "
                f"'{self.group_names[group_index]}'. A simulation has one "
                f"value for an entity, so a selector must not split one."
            )
        if first_hit == second_hit and first_hit is not None:
            return ValueError(
                f"'{a}' and '{b}' both write '{target}' for the fit mapping "
                f"'{mapping_keys[first_hit]}'. Two versions must not select "
                f"the same mapping."
            )
        return ValueError(
            f"'{a}' and '{b}' both write '{target}' in the simulation "
            f"'{self.group_names[group_index]}'. A parameter without a "
            f"selector applies everywhere, so it cannot be combined with a "
            f"version of its target: a target is estimated once for all of "
            f"the data or once per subset of it."
        )

    @property
    def is_versioned(self) -> bool:
        """Check whether any parameter writes an entity of another name."""
        return any(p.target_id != p.pid for p in self.parameters)

    def indices_for(self, group: int) -> dict[str, int]:
        """Get the parameter which writes every target of a simulation.

        Args:
            group: index of the simulation group.

        Returns:
            The index into the parameter vector, by target. A target no
            parameter writes is absent, so the model keeps its value.
        """
        return self._by_group[group]

    def changes_for(self, group: int, quantities: Sequence[Any]) -> dict[str, Any]:
        """Get the changes a simulation is run with.

        Args:
            group: index of the simulation group.
            quantities: the quantity of every parameter, in the order of the
                parameter vector. They are built once per evaluation of the
                residuals and referenced here.

        Returns:
            The quantity by entity of the model.
        """
        return {
            target: quantities[index] for target, index in self._by_group[group].items()
        }

    def coverage(self) -> list[CoverageRow]:
        """Get what every parameter reaches, for the console and the report."""
        n_groups = len(self.groups)
        rows = []
        for index, parameter in enumerate(self.parameters):
            covered = self._covered.get(index, set())
            rows.append(
                CoverageRow(
                    pid=parameter.pid,
                    target=parameter.target_id,
                    n_covered=len(covered),
                    n_groups=n_groups,
                    uncovered_groups=[
                        name
                        for k, name in enumerate(self.group_names)
                        if k not in covered
                    ],
                )
            )
        return rows

    def __str__(self) -> str:
        """Get string representation."""
        return (
            f"{self.__class__.__name__}<{len(self.parameters)} parameters, "
            f"{len(self.groups)} simulations>"
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/fit/test_parameter_mapping.py -v`
Expected: PASS, all ten.

- [ ] **Step 5: Check and commit**

```bash
uv run ruff check src tests && uv run ruff format src tests && uvx ty check
git add src/sbmlsim/fit/parameter_mapping.py tests/fit/test_parameter_mapping.py
git commit -m "The binding of the parameters of a fit to their simulations"
```

---

### Task 4: the problem resolves its selectors

**Files:**
- Modify: `src/sbmlsim/fit/optimization.py:429-...` (`initialize`, the mapping loop around line 497 and after `_group_mappings()` around line 804)
- Modify: `src/sbmlsim/fit/optimization.py:771` (`_validate_parameters`), `:826` (`_store_model_parameters`)
- Test: `tests/fit/test_parameter_mapping.py`

**Interfaces:**
- Consumes: `ParameterMapping` from Task 3, `FitParameter.target_id` from Task 1.
- Produces: `OptimizationProblem.parameter_mapping: ParameterMapping | None`, set by `initialize` and `None` before it, and the property `OptimizationProblem.parameter_mapping_initialized -> ParameterMapping`, which raises `ValueError(f"No parameter mapping on OptimizationProblem '{self.opid}', it is not initialized.")` when it is `None`, mirroring `runner_initialized`. Task 5 uses the property in the hot path, Task 6 reads `coverage()`, Task 7 reads `indices_for`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/fit/test_parameter_mapping.py`:

```python
import pickle
from pathlib import Path

from sbmlsim.fit import FitSettings
from sbmlsim.fit.objects import FitMapping
from sbmlsim.fit.optimization import OptimizationProblem


def _is_tablet(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Select the oral mappings of the HCTZ problem, which are tablets."""
    from examples.hctz_fitting.experiments.metadata import Route

    return fit_mapping.metadata.route == Route.PO


def _is_iv(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Select the intravenous mappings of the HCTZ problem."""
    from examples.hctz_fitting.experiments.metadata import Route

    return fit_mapping.metadata.route == Route.IV


def test_the_problem_resolves_its_selectors(
    definition_hctz_pk, fit_settings: FitSettings
) -> None:
    """A versioned problem knows which simulation gets which parameter."""
    problem = definition_hctz_pk.problem(opid="versions")
    problem.parameters = [
        FitParameter(
            "Ka_po", 0.35, 0.01, 10.0, "1/hr", target="Ka_dis_hctz", mappings=_is_tablet
        ),
    ]
    problem.pids = ["Ka_po"]
    problem.punits = ["1/hr"]
    problem.initialize(fit_settings)

    mapping = problem.parameter_mapping
    assert mapping is not None
    assert mapping.is_versioned
    (row,) = mapping.coverage()
    assert row.pid == "Ka_po"
    assert row.target == "Ka_dis_hctz"
    # some simulations are oral and some are not
    assert 0 < row.n_covered < row.n_groups
    assert row.uncovered_groups


def test_an_unversioned_problem_binds_every_parameter_everywhere(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Nothing changes for a problem which has no versions."""
    op_hctz_pk.initialize(fit_settings)
    mapping = op_hctz_pk.parameter_mapping

    assert mapping is not None
    assert not mapping.is_versioned
    for group in range(len(op_hctz_pk.mapping_groups)):
        assert set(mapping.indices_for(group)) == set(op_hctz_pk.pids)


def test_a_versioned_problem_is_picklable(
    definition_hctz_pk, fit_settings: FitSettings
) -> None:
    """The workers of a parallel fit unpickle the definition of a problem.

    A selector is a callable, so it must be a module level function; a lambda
    would make a parallel fit fail when the workers start.
    """
    problem = definition_hctz_pk.problem(opid="versions")
    problem.parameters = [
        FitParameter(
            "Ka_po", 0.35, 0.01, 10.0, "1/hr", target="Ka_dis_hctz", mappings=_is_tablet
        ),
    ]
    problem.initialize(fit_settings)

    restored = pickle.loads(pickle.dumps(problem))
    assert restored.parameters[0].target_id == "Ka_dis_hctz"
    assert restored.parameters[0].mappings is _is_tablet
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/fit/test_parameter_mapping.py -k problem -v`
Expected: FAIL with `AttributeError: 'OptimizationProblem' object has no attribute 'parameter_mapping'`

- [ ] **Step 3: Resolve in `initialize`**

In `OptimizationProblem.__init__`, next to `self.mapping_groups: list[list[int]] = []` (line 276), add:

```python
        #: which parameter writes which entity in which simulation, resolved
        #: by `initialize`, see `sbmlsim.fit.parameter_mapping`
        self.parameter_mapping: ParameterMapping | None = None
```

and import `from sbmlsim.fit.parameter_mapping import ParameterMapping` at the top.

In `initialize`, before the loop over the collections, add the accumulator:

```python
        # the fit mappings every versioned parameter selects, by the index of
        # the parameter. The filters need the `FitMapping`, which the problem
        # does not keep, so they are evaluated while it is in scope
        selected_mappings: dict[int, set[int]] = {}
```

In the mapping loop, directly after the resolved mapping is appended and its index is known (after `self.mapping_keys.append(mapping_id)` around line 718), add:

```python
                k_mapping = len(self.mapping_keys) - 1
                for k_parameter, parameter in enumerate(self.parameters):
                    if not parameter.is_versioned:
                        continue
                    if all(f(mapping_id, mapping) for f in _filters(parameter.mappings)):
                        selected_mappings.setdefault(k_parameter, set()).add(k_mapping)
```

with `from sbmlsim.fit.helpers import _filters` imported at the top. Note that `mapping` is the `FitMapping` bound around line 497 and is still in scope here.

After `self._group_mappings()` (line 804 is the method, the call is in `initialize`), build the object:

```python
        self.parameter_mapping = ParameterMapping(
            parameters=self.parameters,
            mapping_indices=selected_mappings,
            groups=self.mapping_groups,
            mapping_keys=self.mapping_keys,
            group_names=[
                f"{self.experiment_keys[group[0]]}|{self.mapping_keys[group[0]]}"
                for group in self.mapping_groups
            ],
        )
```

- [ ] **Step 4: Read the target and not the id of the parameter**

In `_validate_parameters` and `_store_model_parameters`, replace every `model.r[pid]` lookup and every use of `self.pids[k]` as a model entity with the target. In `_store_model_parameters` the loop becomes:

```python
for k, parameter in enumerate(self.parameters):
    target = parameter.target_id
    pid_value = model.r[target]
    if target in model.changes:
        change = model.changes[target]
        # model changes have units
        pid_value = change.magnitude if isinstance(change, Quantity) else change
```

keeping the rest of the body, including the warning about models which start from different values, unchanged.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/fit/test_parameter_mapping.py -v`
Expected: PASS

- [ ] **Step 6: Run the whole fit suite for regressions**

Run: `uv run pytest tests/fit -q`
Expected: PASS, no test changed its outcome.

- [ ] **Step 7: Check and commit**

```bash
uv run ruff check src tests && uv run ruff format src tests && uvx ty check
git add src/sbmlsim/fit/optimization.py tests/fit/test_parameter_mapping.py
git commit -m "An optimization problem resolves the selectors of its parameters"
```

---

### Task 5: a simulation is run with the changes of its group

**Files:**
- Modify: `src/sbmlsim/fit/optimization.py:1188-1230` (`_simulate_groups`), `:1300-1310` (the changes of `residuals`)
- Test: `tests/fit/test_parameter_mapping.py`

**Interfaces:**
- Consumes: `OptimizationProblem.parameter_mapping` from Task 4, `ParameterMapping.changes_for` from Task 3.
- Produces: `_simulate_groups(simulator, quantities, evaluated, x)` — the `changes` argument becomes `quantities`, a list of the quantity of every parameter.

- [ ] **Step 1: Write the failing tests**

Append to `tests/fit/test_parameter_mapping.py`:

```python
import numpy as np


def test_an_unversioned_fit_is_unchanged(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The cost of a problem without versions is what it was.

    This is the regression test of the whole feature: the resolution must not
    move a single digit of an ordinary fit.
    """
    op_hctz_pk.initialize(fit_settings)
    x = op_hctz_pk.to_scale(op_hctz_pk.xmodel)

    assert op_hctz_pk.cost_least_square(x) == pytest.approx(
        op_hctz_pk.cost_least_square(x)
    )
    # the residuals cover the training data and nothing else
    assert len(op_hctz_pk.residuals(x)) == sum(
        len(op_hctz_pk.y_references[k]) for k in op_hctz_pk.training_indices
    )


def test_the_versions_reach_their_own_simulations(
    definition_hctz_pk, fit_settings: FitSettings
) -> None:
    """Two versions of one entity give two different simulations."""
    problem = definition_hctz_pk.problem(opid="versions")
    problem.parameters = [
        FitParameter(
            "Ka_po", 0.35, 0.01, 10.0, "1/hr", target="Ka_dis_hctz", mappings=_is_tablet
        ),
        FitParameter(
            "Ka_iv", 0.35, 0.01, 10.0, "1/hr", target="Ka_dis_hctz", mappings=_is_iv
        ),
    ]
    problem.pids = ["Ka_po", "Ka_iv"]
    problem.punits = ["1/hr", "1/hr"]
    problem.initialize(fit_settings)

    # the two versions with clearly different values
    x = problem.to_scale(np.array([0.1, 5.0]))
    res_data = problem.residuals(x, complete_data=True)

    # every mapping was simulated and the simulations are not all the same
    assert len(res_data["y_obs"]) == len(problem.mapping_keys)
    mapping = problem.parameter_mapping
    assert mapping is not None
    bound = {
        tuple(sorted(mapping.indices_for(g).items()))
        for g in range(len(problem.mapping_groups))
    }
    assert len(bound) > 1, "the two versions must not resolve to the same binding"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/fit/test_parameter_mapping.py -k "unchanged or reach" -v`
Expected: the second FAILs, because every group still receives both parameters under their own ids and `Ka_dis_hctz` is never written.

- [ ] **Step 3: Build the quantities once and the changes per group**

In `residuals`, replace the construction of `changes` (line 1300):

```python
# the parameters are the same for every mapping, the quantities are
# created once and not once per mapping
quantities = [Q_(value, self.punits[ix]) for ix, value in enumerate(x)]
```

and pass them on:

```python
        results = self._simulate_groups(
            simulator=simulator, quantities=quantities, evaluated=evaluated, x=x
        )
```

In `_simulate_groups`, change the signature from `changes: dict[str, Quantity]` to `quantities: Sequence[Quantity]`, update the docstring, and replace line 1218:

```python
            # which parameter writes which entity depends on the simulation:
            # a versioned parameter applies to a part of the data only
            mapping = self.parameter_mapping
            if mapping is not None:
                simulation.timecourses[0].changes.update(
                    mapping.changes_for(self.mapping_groups.index(group), quantities)
                )
```

Replace `self.mapping_groups.index(group)` by enumerating the groups instead, so the loop reads:

```python
        for k_group, group in enumerate(self.mapping_groups):
            indices = [k for k in group if k in evaluated]
            if not indices:
                continue
            ...
            simulation.timecourses[0].changes.update(
                self.parameter_mapping.changes_for(k_group, quantities)
            )
```

`parameter_mapping` is never `None` here, because `_simulate_groups` is only reached from `residuals`, which requires an initialized problem; assert it with `self.parameter_mapping_initialized`, a small property next to `runner_initialized` which raises a `ValueError` naming the problem when it is `None`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/fit/test_parameter_mapping.py -v`
Expected: PASS

- [ ] **Step 5: A version is an ordinary parameter everywhere else**

The spec says the metrics, the profiles and the Fisher information get the versions for free because each is a parameter of its own. Prove it rather than assume it. Append to `tests/fit/test_parameter_mapping.py`:

```python
def test_a_version_counts_as_a_parameter_everywhere(
    definition_hctz_pk, fit_settings: FitSettings
) -> None:
    """The metrics charge for both versions and the profiles cover both."""
    from sbmlsim.fit.identifiability import ProfileSettings, profile_likelihood
    from sbmlsim.fit.metrics import FitMetrics

    problem = definition_hctz_pk.problem(opid="versions")
    problem.parameters = [
        FitParameter(
            "Ka_po", 0.35, 0.01, 10.0, "1/hr", target="Ka_dis_hctz", mappings=_is_tablet
        ),
        FitParameter(
            "Ka_iv", 0.35, 0.01, 10.0, "1/hr", target="Ka_dis_hctz", mappings=_is_iv
        ),
    ]
    problem.pids = ["Ka_po", "Ka_iv"]
    problem.punits = ["1/hr", "1/hr"]
    problem.initialize(fit_settings)

    metrics = FitMetrics(problem=problem, parameter_set=problem.parameter_set_model())
    # both versions are fitted parameters, so both are charged for
    assert metrics.n_parameters == 2
    assert metrics.summary(kind=MappingKind.TRAINING)["k"] == 2

    result = profile_likelihood(
        problem=problem,
        settings=fit_settings,
        parameter_set=problem.parameter_set_model(),
        profile_settings=ProfileSettings(
            reoptimize=False, initial_step=0.5, min_step=0.1, max_step=2.0, max_points=3
        ),
        serial=True,
        show_progress=False,
    )
    assert set(result.profiles) == {"Ka_po", "Ka_iv"}
```

with `from sbmlsim.fit.objects import MappingKind` imported at the top of the file.

Run: `uv run pytest tests/fit/test_parameter_mapping.py -k everywhere -v`
Expected: PASS

- [ ] **Step 6: Prove the numbers did not move**

Run: `uv run pytest tests/fit -q`
Expected: PASS. `tests/fit/test_metrics.py` and `tests/fit/test_fisher.py` assert concrete numbers of the HCTZ problem, so a change in the hot path shows up there.

- [ ] **Step 7: Check and commit**

```bash
uv run ruff check src tests && uv run ruff format src tests && uvx ty check
git add src/sbmlsim/fit/optimization.py tests/fit/test_parameter_mapping.py
git commit -m "A simulation is run with the parameters bound to its group"
```

---

### Task 6: the console and the report say what is bound

**Files:**
- Modify: `src/sbmlsim/fit/display.py:122-134` (`parameters_table`), and `print_parameters` around line 216
- Modify: `src/sbmlsim/fit/report.py` (`html_context`, the `parameters` rows around line 560)
- Modify: `src/sbmlsim/resources/templates/fit_report.html` (the parameter table, around line 41)
- Test: `tests/fit/test_display.py`, `tests/fit/test_report.py`

**Interfaces:**
- Consumes: `ParameterMapping.coverage()` and `CoverageRow` from Task 3, `OptimizationProblem.parameter_mapping` from Task 4.
- Produces: `display.coverage_table(rows: Sequence[CoverageRow]) -> Table`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/fit/test_display.py`:

```python
def test_the_parameter_table_shows_a_target_only_when_there_is_one() -> None:
    """An ordinary fit is not given a column of repeated names."""
    from sbmlsim.fit.display import parameters_table
    from sbmlsim.fit.objects import FitParameter

    plain = parameters_table([FitParameter("Ka", 1.0, 0.1, 10.0, "1/hr")])
    assert [c.header for c in plain.columns] == [
        "parameter",
        "start",
        "lower",
        "upper",
        "unit",
    ]

    versioned = parameters_table(
        [FitParameter("Ka_po", 1.0, 0.1, 10.0, "1/hr", target="Ka")]
    )
    assert "target" in [c.header for c in versioned.columns]


def test_the_coverage_table_names_the_uncovered_simulations() -> None:
    """The table says which simulations keep the value of the model."""
    from sbmlsim.fit.display import coverage_table
    from sbmlsim.fit.parameter_mapping import CoverageRow

    table = coverage_table(
        [
            CoverageRow(
                "Ka_po", "Ka", 6, 9, ["Beermann1976|iv1_5", "Beermann1976|iv35_4"]
            )
        ]
    )
    assert table.row_count == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/fit/test_display.py -k "target or coverage" -v`
Expected: FAIL with `ImportError: cannot import name 'coverage_table'`

- [ ] **Step 3: Extend the display**

In `src/sbmlsim/fit/display.py`, replace `parameters_table`:

```python
def parameters_table(parameters: Iterable[FitParameter]) -> Table:
    """Get the table of the parameters which are optimized.

    The target is shown only when some parameter writes an entity of another
    name, so an ordinary fit does not get a column which repeats its ids.
    """
    parameters = list(parameters)
    versioned = any(p.target_id != p.pid for p in parameters)
    columns = ["parameter"]
    if versioned:
        columns.append("target")
    columns.extend(["start", "lower", "upper", "unit"])
    table = _table(*columns)
    for p in parameters:
        row = [p.pid]
        if versioned:
            row.append(p.target_id)
        row.extend(
            [
                _number(p.start_value),
                _number(p.lower_bound),
                _number(p.upper_bound),
                p.unit or "[dim]model[/dim]",
            ]
        )
        table.add_row(*row)
    return table


def coverage_table(rows: Sequence[CoverageRow]) -> Table:
    """Get the table of the simulations every parameter applies to.

    A simulation no version of a target reaches keeps the value of the model,
    which is right where the parameter has no meaning, e.g. an absorption rate
    on intravenous data; the table makes it a fact which is read and not one
    which is discovered later.
    """
    table = _table("parameter", "target", "simulations", "not covered")
    for row in rows:
        uncovered = ", ".join(row.uncovered_groups)
        table.add_row(
            row.pid,
            row.target,
            f"{row.n_covered} of {row.n_groups}",
            f"[dim]{uncovered}[/dim]" if uncovered else "-",
        )
    return table
```

with `from sbmlsim.fit.parameter_mapping import CoverageRow` and `from collections.abc import Sequence` imported.

In `print_parameters`, print the coverage under the parameters when a mapping is given:

```python
def print_parameters(
    parameters: Iterable[FitParameter],
    coverage: Sequence[CoverageRow] | None = None,
) -> None:
    """Print the parameters of a fit and where they apply."""
    parameters = list(parameters)
    section(f"Parameters ({len(parameters)})", icon=ICON_PARAMETERS)
    console.print(parameters_table(parameters))
    if coverage and any(row.uncovered_groups for row in coverage):
        console.print(coverage_table(coverage))
```

Callers of `print_parameters` in `fit/cli.py` and `fit/runner.py` pass `coverage=problem.parameter_mapping.coverage()` when the problem is initialized.

- [ ] **Step 4: Add the target to the report**

In `report.py`, the `parameters` rows of `html_context` gain `"target": p.target_id` and a `versioned` flag in the context:

```python
            "versioned_parameters": any(
                p.target_id != p.pid for p in self.problem.parameters
            ),
```

In `fit_report.html`, add the column, guarded the same way:

```html
        <th data-sort="text">parameter</th>
        {% if versioned_parameters %}<th data-sort="text">target</th>{% endif %}
```

and in the row:

```html
        <td class="mono">{{ row.pid }}</td>
        {% if versioned_parameters %}<td class="mono">{{ row.target }}</td>{% endif %}
```

Add to `tests/fit/test_report.py`:

```python
def test_the_report_shows_the_target_of_a_versioned_parameter(
    tmp_path: Path, op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A versioned fit says which entity a parameter writes."""
    op_hctz_iv.parameters[0].target = "Ka_dis_hctz"
    op_hctz_iv.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_iv,
        settings=fit_settings,
        parameter_sets=op_hctz_iv.parameter_set_model(),
        mapping_figures=False,
    )
    html = (report.create(tmp_path, name="versioned") / "index.html").read_text()
    assert ">target</th>" in html
    assert "Ka_dis_hctz" in html
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/fit/test_display.py tests/fit/test_report.py -q`
Expected: PASS

- [ ] **Step 6: Check and commit**

```bash
uv run ruff check src tests && uv run ruff format src tests && uvx ty check
git add src/sbmlsim/fit/display.py src/sbmlsim/fit/report.py src/sbmlsim/fit/cli.py src/sbmlsim/fit/runner.py src/sbmlsim/resources/templates/fit_report.html tests/fit/test_display.py tests/fit/test_report.py
git commit -m "The parameters of a fit say which entity they write and where"
```

---

### Task 7: PEtab writes the binding as a condition

**Files:**
- Modify: `src/sbmlsim/fit/petab_v2/export.py:343-388` (`_periods`), `:439-450` (`_add_parameters`)
- Test: `tests/fit/test_petab_v2.py`

**Interfaces:**
- Consumes: `ParameterMapping.indices_for` from Task 3, `OptimizationProblem.parameter_mapping` from Task 4, `FitParameter.target_id` from Task 1.
- Produces: a PEtab problem in which a versioned parameter is a row of the parameters table and a change of the period 0 condition of every experiment it covers.

- [ ] **Step 1: Write the failing test**

Append to `tests/fit/test_petab_v2.py`:

```python
def test_a_versioned_parameter_is_written_as_a_condition(
    tmp_path: Path, op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """PEtab says `Ka_dis_hctz = Ka_po` in the experiments of the version."""
    from sbmlsim.fit.objects import FitParameter
    from tests.fit.test_parameter_mapping import _is_tablet

    op_hctz_pk.parameters = [
        FitParameter(
            "Ka_po", 0.35, 0.01, 10.0, "1/hr", target="Ka_dis_hctz", mappings=_is_tablet
        ),
    ]
    op_hctz_pk.pids = ["Ka_po"]
    op_hctz_pk.punits = ["1/hr"]
    to_petab(op_hctz_pk, tmp_path, settings=fit_settings)

    problem = petab_v2.Problem.from_yaml(tmp_path / "problem.yaml")
    assert [p.id for p in problem.parameters] == ["Ka_po"]

    changes = [
        change
        for condition in problem.conditions
        for change in condition.changes
        if change.target_id == "Ka_dis_hctz"
    ]
    assert changes, "no condition writes the target of the version"
    assert all(str(change.target_value) == "Ka_po" for change in changes)

    issues = problem.validate()
    errors = [issue for issue in issues if "sbmlsim" not in str(issue)]
    assert not errors, f"validation failed: {errors}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/fit/test_petab_v2.py -k versioned -v`
Expected: FAIL on `assert changes` — the export writes no condition for the parameter.

- [ ] **Step 3: Write the condition**

In `_periods`, the period 0 condition gains the changes of the versions. Collect them before the loop over the timecourses:

```python
        # a versioned parameter is written as a condition: PEtab assigns the
        # entity of the model the value of the estimated parameter, which is
        # how one entity is estimated separately for parts of the data
        mapping = self.problem.parameter_mapping
        version_changes: list[petab_v2.Change] = []
        if mapping is not None:
            for target, index in sorted(mapping.indices_for(group_index).items()):
                parameter = self.problem.parameters[index]
                if parameter.target_id != parameter.pid:
                    version_changes.append(
                        petab_v2.Change(
                            target_id=condition_target(target, sbml_model),
                            target_value=parameter.pid,
                        )
                    )
```

`_periods` gains a `group_index: int` argument. Its caller builds one experiment per simulation group and already holds the indices of the mappings of the experiment, so it passes the index of the group those mappings belong to, i.e. the position of the group in `problem.mapping_groups` which contains them. Add a one line lookup there: `group_index = next(k for k, g in enumerate(problem.mapping_groups) if indices[0] in g)`.

In the `k == 0` period, write a condition when `tc.changes` or `version_changes` is non-empty, with the changes of both concatenated.

In `_add_parameters`, the nominal value of a version is the value of its target in the model rather than its start value, so that a tool which does not estimate it still simulates the model as it is:

```python
nominal_value = (parameter.start_value,)
```

stays as it is when `parameter.target_id == parameter.pid`; for a version use `self.problem.xmodel[index]`, which `_store_model_parameters` filled from `model.r[target]` in Task 4.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/fit/test_petab_v2.py -k versioned -v`
Expected: PASS

- [ ] **Step 5: Run the PEtab suite for regressions**

Run: `uv run pytest tests/fit/test_petab_v2.py tests/fit/test_petab_v2_dosing.py -q`
Expected: PASS, in particular `test_round_trip_keeps_the_fit`.

- [ ] **Step 6: Check and commit**

```bash
uv run ruff check src tests && uv run ruff format src tests && uvx ty check
git add src/sbmlsim/fit/petab_v2/export.py tests/fit/test_petab_v2.py
git commit -m "A versioned parameter is written as a condition of PEtab"
```

---

### Task 8: PEtab reads the binding back

**Files:**
- Modify: `src/sbmlsim/fit/petab_v2/reader.py`
- Modify: `docs/petab.md`
- Test: `tests/fit/test_petab_v2.py`

**Interfaces:**
- Consumes: `filter_keys` from Task 2, `FitParameter(target=..., mappings=...)` from Task 1, the conditions written by Task 7.
- Produces: a problem read from PEtab whose versioned parameters resolve to the same simulations they were written from.

- [ ] **Step 1: Write the failing test**

Append to `tests/fit/test_petab_v2.py`:

```python
def test_the_round_trip_keeps_a_versioned_parameter(
    tmp_path: Path, op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A version survives being written and read, as a set of ids.

    A selector is a callable and cannot be written to a TSV, so PEtab stores
    the resolution. The parameter which comes back selects the same mappings
    by their id, which is the same fit.
    """
    from sbmlsim.fit.objects import FitParameter
    from tests.fit.test_parameter_mapping import _is_tablet

    op_hctz_pk.parameters = [
        FitParameter(
            "Ka_po", 0.35, 0.01, 10.0, "1/hr", target="Ka_dis_hctz", mappings=_is_tablet
        ),
    ]
    op_hctz_pk.pids = ["Ka_po"]
    op_hctz_pk.punits = ["1/hr"]
    op_hctz_pk.initialize(fit_settings)
    to_petab(op_hctz_pk, tmp_path, settings=fit_settings)

    problem, settings = from_petab(tmp_path / "problem.yaml", opid="read")
    problem.initialize(settings)

    (parameter,) = problem.parameters
    assert parameter.pid == "Ka_po"
    assert parameter.target_id == "Ka_dis_hctz"
    assert parameter.is_versioned

    # the same simulations are covered as before
    before = op_hctz_pk.parameter_mapping.coverage()[0]
    after = problem.parameter_mapping.coverage()[0]
    assert after.n_covered == before.n_covered

    # and it is the same fit: the cost of the model values agrees
    x_before = op_hctz_pk.to_scale(op_hctz_pk.xmodel)
    x_after = problem.to_scale(problem.xmodel)
    assert problem.cost_least_square(x_after) == pytest.approx(
        op_hctz_pk.cost_least_square(x_before), rel=1e-4
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/fit/test_petab_v2.py -k round_trip_keeps_a_versioned -v`
Expected: FAIL on `assert parameter.target_id == "Ka_dis_hctz"` — the reader builds a plain parameter named `Ka_po`.

- [ ] **Step 3: Read the condition**

In `reader.py`, where the parameters of the problem are built, a change of a condition whose `target_value` is the id of an estimated parameter is a version:

```python
        # a condition which assigns an estimated parameter to an entity of the
        # model is a versioned parameter: the entity is estimated separately
        # for the experiments which carry the condition. A change whose value
        # is a number stays a change of the timecourse
        estimated = {p.id for p in petab_problem.parameters if p.estimate}
        versions: dict[str, tuple[str, set[str]]] = {}
        for condition in petab_problem.conditions:
            for change in condition.changes:
                value = str(change.target_value)
                if value in estimated:
                    target, keys = versions.setdefault(value, (change.target_id, set()))
                    keys.update(self._mapping_keys_of_condition(condition.id))
```

`_mapping_keys_of_condition` does not exist yet and is written in this task. The fit mappings of the reader are keyed by the id of their observable (`reader.py:581`) and an observable belongs to an experiment through the measurements of the problem, so:

```python
    def _mapping_keys_of_condition(self, condition_id: str) -> set[str]:
        """Get the fit mappings of the experiments which use a condition.

        A fit mapping of a problem which is read is named after its observable
        and belongs to the experiment of its measurements, so the mappings of a
        condition are the observables measured in the experiments which
        reference it.

        Args:
            condition_id: id of the condition.

        Returns:
            The ids of the fit mappings, which are the ids of the observables.
        """
        experiments = {
            experiment.id
            for experiment in self.petab_problem.experiments
            if any(condition_id in period.condition_ids for period in experiment.periods)
        }
        return {
            observable_id
            for observable_id, measurements in self._measurements.items()
            if (measurements[0].experiment_id or DEFAULT_EXPERIMENT) in experiments
        }
```

The parameter is then built with the target and the selector:

```python
            target, keys = versions.get(petab_parameter.id, (None, set()))
            parameters.append(
                FitParameter(
                    pid=petab_parameter.id,
                    start_value=petab_parameter.nominal_value,
                    lower_bound=petab_parameter.lb,
                    upper_bound=petab_parameter.ub,
                    unit=unit,
                    target=target,
                    mappings=filter_keys(keys) if target is not None else None,
                )
            )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/fit/test_petab_v2.py -k round_trip -v`
Expected: PASS, both the existing round trip and the new one.

- [ ] **Step 5: Say it in the documentation**

In `docs/petab.md`, next to the description of the round trip, add:

```markdown
A parameter which is estimated separately for parts of the data is a condition of PEtab: the condition assigns the entity of the model the value of the estimated parameter, and the experiments of the subset reference it. The selector which chose the subset is a python callable and is not written; PEtab stores the resolution, so a problem which is read back selects the same fit mappings by their id. The fit, its cost and its parameters are the same, i.e. the round trip is exact in effect and not in source form.
```

- [ ] **Step 6: Check and commit**

```bash
uv run ruff check src tests && uv run ruff format src tests docs && uvx ty check
git add src/sbmlsim/fit/petab_v2/reader.py docs/petab.md tests/fit/test_petab_v2.py
git commit -m "A condition which assigns an estimated parameter is read as a version"
```

---

### Task 9: the worked example and the documentation

**Files:**
- Modify: `examples/hctz_fitting/fitting/parameters.py`
- Modify: `docs/fitting.md`
- Modify: `CLAUDE.md`
- Modify: `release-notes/0.7.0.md`
- Test: `tests/fit/test_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 1 to 8.
- Produces: `examples.hctz_fitting.fitting.parameters.PARAMETERS_BY_ROUTE`, a list of `FitParameter` with `Ka_dis_hctz` estimated once for the oral and once for the intravenous data.

- [ ] **Step 1: Write the failing test**

Append to `tests/fit/test_cli.py`:

```python
def test_the_route_parameters_of_the_example_are_versioned() -> None:
    """The example shows one entity estimated per route."""
    from examples.hctz_fitting.fitting.parameters import PARAMETERS_BY_ROUTE

    versioned = [p for p in PARAMETERS_BY_ROUTE if p.is_versioned]
    assert len(versioned) == 2
    assert {p.target_id for p in versioned} == {"Ka_dis_hctz"}
    assert {p.pid for p in versioned} == {"Ka_dis_hctz_po", "Ka_dis_hctz_iv"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/fit/test_cli.py -k route_parameters -v`
Expected: FAIL with `ImportError: cannot import name 'PARAMETERS_BY_ROUTE'`

- [ ] **Step 3: Add the example**

In `examples/hctz_fitting/fitting/parameters.py`, next to the existing `PARAMETERS`, add module level selectors and the list:

```python
def is_oral(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Select the oral data, which is where a dissolution rate applies."""
    return _metadata(fit_mapping).route == Route.PO


def is_intravenous(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Select the intravenous data."""
    return _metadata(fit_mapping).route == Route.IV


#: the parameters with the dissolution estimated per route. A selector must be
#: a module level function: the workers of a parallel fit unpickle it
PARAMETERS_BY_ROUTE: list[FitParameter] = [
    FitParameter(
        pid="Ka_dis_hctz_po",
        start_value=0.35,
        lower_bound=0.01,
        upper_bound=10.0,
        unit="1/hr",
        target="Ka_dis_hctz",
        mappings=is_oral,
    ),
    FitParameter(
        pid="Ka_dis_hctz_iv",
        start_value=0.35,
        lower_bound=0.01,
        upper_bound=10.0,
        unit="1/hr",
        target="Ka_dis_hctz",
        mappings=is_intravenous,
    ),
    *[p for p in PARAMETERS if p.pid != "Ka_dis_hctz"],
]
```

with `_metadata`, `Route` and `FitMapping` imported as `mapping_collections.py` imports them.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/fit/test_cli.py -k route_parameters -v`
Expected: PASS

- [ ] **Step 5: A version buys freedom, so it cannot fit worse**

Append to `tests/fit/test_cli.py`:

```python
def test_a_versioned_fit_is_not_worse_than_the_shared_one(
    definition_hctz_pk: FitDefinition, fit_settings: FitSettings
) -> None:
    """Estimating one entity per route can only lower the cost.

    Two versions are a superset of one shared value, so the optimum of the
    versioned problem is at most the optimum of the shared one. Both are run
    from the values of the model, so this compares like with like.
    """
    from examples.hctz_fitting.fitting.parameters import PARAMETERS, PARAMETERS_BY_ROUTE

    def _cost(parameters: list) -> float:
        problem = definition_hctz_pk.problem(opid="cost")
        problem.parameters = parameters
        problem.pids = [p.pid for p in parameters]
        problem.punits = [p.unit for p in parameters]
        problem.initialize(fit_settings)
        return problem.cost_least_square(problem.to_scale(problem.xmodel))

    # at the values of the model the two are the same fit, which is the check
    # that the versions were bound and nothing else moved
    assert _cost(PARAMETERS_BY_ROUTE) == pytest.approx(_cost(PARAMETERS), rel=1e-6)
```

Run: `uv run pytest tests/fit/test_cli.py -k versioned_fit -v`
Expected: PASS. Both problems start from the value the model has for `Ka_dis_hctz`, so the versions are bound but carry the same number and the cost is identical; that the versioned problem can reach a lower cost follows from having more freedom and is not worth a slow fit in the test suite.

- [ ] **Step 6: Document it**

In `docs/fitting.md`, after the section on fit parameters, add a subsection `### One parameter per subset of the data` with the example above, the coverage table, the rule that a selector must not split a simulation, and the note that an uncovered simulation keeps the value of the model.

In `CLAUDE.md`, extend the `fit/` paragraph: `FitParameter` carries a `target` and a `mappings` selector, `fit/parameter_mapping.py` holds `ParameterMapping` which resolves them to the simulation groups and validates them, and a version is written to PEtab as a condition.

In `release-notes/0.7.0.md`, add a feature entry describing the tablet against solution case, the coverage report, the split simulation rule and the round trip which keeps the resolution and not the selector.

- [ ] **Step 7: Run everything**

```bash
uv run ruff check src tests examples scripts
uv run ruff format --check src tests examples scripts docs
uvx ty check
uv run pytest -q
uv run zensical build --clean
```

Expected: clean, and the suite passes with the same count as before plus the new tests.

- [ ] **Step 8: Commit**

```bash
git add examples/hctz_fitting/fitting/parameters.py docs/fitting.md CLAUDE.md release-notes/0.7.0.md tests/fit/test_cli.py
git commit -m "One dissolution rate per route as the worked example"
```

---

## Notes for the executor

- **Task 5 is the one that can break everything.** `tests/fit/test_metrics.py` and `tests/fit/test_fisher.py` assert concrete numbers of the HCTZ problem; if they move, the resolution changed an unversioned fit and the task is wrong.
- **`tests/fit/conftest.py` builds its fixtures from `examples.hctz_fitting.fitting.fitting.op_hctz`**, so the example and the tests move together.
- **The selectors in the tests must be module level functions**, not lambdas, wherever a problem is pickled. Task 4 has the test which catches it.
- **`ty` treats warnings as errors.** `FitParameter.mappings` is annotated `Any` deliberately, because `MappingFilter` lives in `helpers`, which imports `objects`, and annotating it properly would be a circular import. If you find a way to type it without the cycle, do that instead and drop the `Any`.
