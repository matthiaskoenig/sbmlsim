# The analyses, phase 1: sampling and uncertainty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One sampler, `sbmlsim.simulation.sampling`, whose designs (`local`, `random`, `lhs`, `sobol`, `fast`, `morris`, `fit_parameters`, `profile_parameters`, `fit_repeats`, `population`) return a `Dimension` with a serializable design record, the start values of a fit on it, the plots of the uncertainty analysis, and the removal of `ModelSensitivity`.

**Architecture:** A design maps points of the unit cube through the inverse CDF (`ppf`) of a distribution per target, so every design works with every marginal; relative distributions read the reference of their target from the model after the pre-initialization. The design returns an ordinary `Dimension` of values with a `Design` record (method, distributions, options, seed, references), which `Scan.to_dict` writes into the provenance of the `ScanResult`, so phase 2 can compute indices from a result alone. The uncertainty analysis is a design, `Simulator.run` and `ScanResult.summary`, with two plot functions.

**Tech Stack:** python 3.13/3.14, uv, numpy, scipy (`stats`, `qmc`), SALib 1.6 (lazily imported), pint, xarray, matplotlib, pytest with xdist, ruff, ty, zensical.

**Spec:** `docs/superpowers/specs/2026-10-09-analyses-design.md` (phase 1 of its "Phases"; the sensitivity analyses `local`, `sobol`, `fast`, `morris` on a result and the removal of the old `sensitivity/` package are phase 2).

## Global Constraints

- Phase 1 starts from `develop` at `344e43ee` (the scan core with its observables). Branch `analyses-phase1`, one pull request to `develop`, one commit per task.
- `sbmlsim.simulation.sampling` is a package; its designs return `Dimension(id, values={target: values}, labels=..., design=Design(...))`; a grid stays a plain `Dimension(values=...)`.
- Distributions: `Uniform(lower, upper)` or `Uniform(relative=r)`, `LogUniform(lower, upper)` or `LogUniform(factor=f)`, `Normal(mean, sd)` or `Normal(cv=c)`, `LogNormal(median, cv)` or `LogNormal(cv=c)`, `Truncated(distribution, lower=None, upper=None)`, `Empirical(values)`, `Fixed(value)`; every distribution has `ppf(u, reference)`; the probabilities 0 and 1 of an unbounded distribution are clipped into the open interval; a relative distribution without a reference raises with the target and the remedy (`model=`).
- References: the value every target has after the pre-initialization of the simulation (`Simulation(end=1)` by default), compiled with the changes of the model as defaults (`Simulator.compile`), in the unit of the target in the model.
- Designs: `local(targets, delta=0.1, *, model, simulation=None, id="local")` with `2 k + 1` points labelled `reference`, `<target>+`, `<target>-`; `random` (`rng.random((n, d))`, with a correlation a Gaussian copula); `lhs` (`qmc.LatinHypercube(d, rng=rng).random(n)`, with a correlation the rank reordering of Iman and Conover); `sobol` (`n` a power of two, SALib, scrambled), `fast` (SALib, `n d` points), `morris` (SALib, levels mapped to the centres of their strata); a correlation is refused for `sobol`, `fast` and `morris`.
- Every random design takes `seed`; `None` draws a seed which the record keeps.
- `fit_parameters` draws from the normal of the Fisher covariance in the space of the `parameter_scale`; `profile_parameters` draws every parameter from `exp(-(cost - cost_min))` on its profile, interpolated in the space of the `parameter_scale`, independent across parameters, a side which stays below the threshold reaching the bound of the parameter; `fit_repeats` takes the best parameter sets of a fit.
- `fit/sampling.py` keeps `SamplingType` and `create_samples(parameters, size, sampling, seed, min_bound)` with the same values for the same seed.
- `ModelSensitivity` (`simulation/sensitivity.py`) is removed; the old `sensitivity/` package stays until phase 2.
- Never use the em dash character, use a plain dash `-`.
- No agent attribution anywhere: no `Co-Authored-By` trailer, no "Generated with Claude Code" line in commits, the pull request, docs or code.
- Commit messages are full sentences which describe the outcome, in the style of `git log`, with a body that explains what and why; no conventional commit prefixes.
- Never edit `CHANGELOG.md` or auto-generated files; no release notes (they belong to the release commit).
- Markdown has no hard line wraps: a paragraph, list item or table row is one line.
- Every module, class and function of the package has full type annotations and a google style docstring (ruff `D`); `tests/` and `examples/` are exempt from docstrings. A subclass marks overrides with `typing.override`.
- ty stays at zero diagnostics (`uv run ty check`, which checks `src`, `tests`, `examples` and `scripts`); suppress only with a rule specific `# ty: ignore[rule-name]`.
- Library code logs with `logging.getLogger(__name__)` and lazy `%s` formatting, it never prints.
- Every commit passes `uv run ruff check`, `uv run ruff format --check`, `uv run ty check` and `uv run pytest -q` (in parallel with xdist; `filterwarnings = error`). Use `uv run` for every command.

## Decisions this plan takes where the spec is silent

- `Design` lives in `sbmlsim/simulation/scan.py` next to `Dimension` (the sampler imports the scan, the scan does not import the sampler) and `sbmlsim.simulation.sampling` re-exports it.
- `Dimension` gains `coordinates=`: arrays along the dimension which the result carries as coordinates but which are never set on a model, e.g. the covariates of a population; the assembly writes them like the changed targets, with their units.
- The package `sbmlsim.simulation.sampling` has the modules `distributions`, `references`, `designs`, `fit` and `population`; `sbmlsim.simulation` does not import it, `from sbmlsim.simulation import sampling` loads it.
- A plain number next to a quantity in one distribution is in the unit of the quantity; the values of a distribution are a quantity when it has a unit (of a quantity or of its reference) and plain floats (the unit of the target in the model) otherwise.
- `local` takes `0 < delta < 1`; its points are the reference, then `+` and `-` for every target in the given order.
- `random` without a correlation draws `rng.random((n, d))` and with one `rng.standard_normal((n, d))`; `lhs` with a correlation needs more points than targets.
- `fast` needs `n > 4 m²` (the condition of SALib).
- The designs of a fit write a parameter to its `pid` unless `targets` maps it to another target; `fit_parameters` draws with the eigendecomposition of the covariance (negative eigenvalues of a rank deficient covariance clipped to zero), which needs no warning of numpy.
- `profile_parameters` drops the points of a profile which did not converge, extends a side whose confidence bound is open to the bound of the parameter with the density of its last point (a finite bound only), and inverts the cumulative trapezoidal integral of the density linearly.
- `fit_repeats` labels its points with the ids of the parameter sets.
- `population` uses `random` or `lhs` for the covariates; the covariates are `coordinates` of the dimension, the targets the function returns its values; its function is checked as the function of a `Custom` (a function of a module).
- `references` initializes the model it reads, which every simulation of the model does anyway; a loaded `RoadrunnerSBMLModel` is used as it is (its integrator settings are not touched), any other model is loaded by `Simulator.load`.
- `parameters_of` lists the constant parameters of the model without the helper parameters sbmlsim adds (`RoadrunnerSBMLModel.parameters`), and with `species=True` the species; `exclude` is a callable or a collection of ids, `exclude_zero` drops targets whose reference is below `1e-8` in magnitude.
- The plots of the uncertainty analysis live in `sbmlsim/sensitivity/uncertainty.py` (the old analyses of the package stay until phase 2).

## Review Focus

- A relative distribution on a target whose reference is zero or negative: `LogNormal(cv=...)` and `LogUniform(factor=...)` raise with the target, `Uniform(relative=...)` and `Normal(cv=...)` give an interval of the right order; covered in Task 2 (`test_a_log_distribution_needs_a_positive_reference`, `test_a_negative_reference`).
- A design whose distributions are quantities in another unit than the model unit of their target: the run converts them (the model sees the value in its unit); covered in Task 4 (`test_a_design_runs_in_a_scan`).
- `seed=None` reproduces from the record: running the design again with the seed of its record gives the same values; covered in Task 4 (`test_the_seed_of_the_record_reproduces_the_design`).
- A population function which returns no mapping, arrays of the wrong length or a covariate as a target raises clearly; covered in Task 7 (`test_a_wrong_population_function_raises`).
- The start values of a fit do not change for a seed and a sampling type (reproducible fits across the change); covered in Task 8 (`test_the_start_values_did_not_change`).

---

## File Structure

- Modify `src/sbmlsim/simulation/scan.py`: `Design`; `Dimension(..., design=None, coordinates=None)`, `to_dict`.
- Modify `src/sbmlsim/simulator/simulator.py`: the assembly writes the `coordinates` of a dimension.
- Create `src/sbmlsim/simulation/sampling/__init__.py`, `distributions.py`, `references.py`, `designs.py`, `fit.py`, `population.py`.
- Modify `src/sbmlsim/fit/sampling.py`: `create_samples` on the sampler.
- Create `src/sbmlsim/sensitivity/uncertainty.py`: `plot_bands`, `plot_distribution`.
- Delete `src/sbmlsim/simulation/sensitivity.py`, `tests/test_sensitivity.py`, `docs/api/simulation.sensitivity.md`.
- Create `docs/sampling.md`, API pages; modify `docs/scans.md`, `zensical.toml`, `examples/model_sensitivity.py`, `examples/demo/demo.py`, `examples/README.md`, `tests/docs/test_docs_code.py`, `CLAUDE.md`.
- Tests: `tests/simulation/test_design_record.py`, `tests/simulation/sampling/` (`__init__.py`, `test_distributions.py`, `test_references.py`, `test_designs.py`, `test_salib_designs.py`, `test_fit_designs.py`, `test_population.py`), `tests/fit/test_sampling.py` (extended), `tests/sensitivity/test_uncertainty.py`.

---

### Task 1: The design record and the coordinates of a dimension

**Files:**
- Modify: `src/sbmlsim/simulation/scan.py` (`Design` before `Dimension`; `Dimension` fields, `__init__`, `__repr__`, `to_dict`)
- Modify: `src/sbmlsim/simulator/simulator.py:626-640` (the coordinates of the assembly)
- Test: `tests/simulation/test_design_record.py` (new)

**Interfaces:**
- Produces:
  - `Design(method: str, distributions: dict[str, dict[str, Any]] = {}, options: dict[str, Any] = {}, references: dict[str, dict[str, Any]] = {})`, frozen, JSON types only, `to_dict() -> dict[str, Any]`, `Design.from_dict(d) -> Design`, equality by `to_dict`.
  - `Dimension(id, *, values=None, simulations=None, models=None, at=None, labels=None, design: Design | None = None, coordinates: Mapping[str, Any] | None = None)`; attributes `design` and `coordinates` (a read-only mapping of read-only arrays or quantities, empty by default); `to_dict()` has the keys `"design"` and `"coordinates"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/simulation/test_design_record.py`:

```python
"""The record of a design and the coordinates of a dimension."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.result import ScanResult
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulation.scan import Design
from sbmlsim.simulator import Simulator
from tests.simulator.models import sbml

DESIGN = Design(
    method="lhs",
    distributions={"k1": {"type": "Uniform", "lower": 0.5, "upper": 1.0}},
    options={"n": 3, "seed": 7},
    references={},
)


def test_a_design_is_json_and_compares_by_its_content() -> None:
    again = Design.from_dict(json.loads(json.dumps(DESIGN.to_dict())))
    assert again == DESIGN
    assert again is not DESIGN
    with pytest.raises(ValueError, match="JSON"):
        Design(method="lhs", options={"rng": np.random.default_rng(1)})


def test_a_dimension_carries_its_design_and_coordinates() -> None:
    dimension = Dimension(
        "d",
        values={"k1": [0.5, 0.75, 1.0]},
        design=DESIGN,
        coordinates={"BW": Q([60.0, 70.0, 80.0], "kg")},
    )
    assert dimension.design == DESIGN
    assert dimension.coordinates["BW"].magnitude.tolist() == [60.0, 70.0, 80.0]
    assert "lhs" in repr(dimension)
    data = dimension.to_dict()
    assert data["design"] == DESIGN.to_dict()
    assert data["coordinates"] == {"BW": {"value": [60.0, 70.0, 80.0], "unit": "kilogram"}}
    again = pickle.loads(pickle.dumps(dimension))
    assert again.design == DESIGN and set(again.coordinates) == {"BW"}


def test_a_design_and_coordinates_need_a_dimension_of_values() -> None:
    with pytest.raises(ValueError, match="values"):
        Dimension("d", simulations={"a": Simulation(end=1)}, design=DESIGN)
    with pytest.raises(ValueError, match="length"):
        Dimension("d", values={"k1": [1.0, 2.0]}, coordinates={"BW": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="target"):
        Dimension("d", values={"k1": [1.0, 2.0]}, coordinates={"k1": [1.0, 2.0]})
    with pytest.raises(TypeError):
        Dimension("d", values={"k1": [1.0]}, design={"method": "lhs"})  # ty: ignore[invalid-argument-type]


def test_the_result_carries_the_record_and_the_coordinates(tmp_path: Path) -> None:
    dimension = Dimension(
        "d",
        values={"k1": [0.5, 0.75, 1.0]},
        design=DESIGN,
        coordinates={"BW": Q([60.0, 70.0, 80.0], "kg")},
    )
    res = Simulator().run(sbml(), Scan(Simulation(end=1, steps=2), [dimension]))
    assert res.ds["BW"].dims == ("d",)
    assert res.units["BW"] == "kilogram"
    path = tmp_path / "r.nc"
    res.to_netcdf(path)
    again = ScanResult.from_netcdf(path)
    (stored,) = again.ds.attrs["scan"]["dimensions"]
    assert Design.from_dict(stored["design"]) == DESIGN
    np.testing.assert_array_equal(again.ds["BW"].values, [60.0, 70.0, 80.0])
```

If `ScanResult.from_netcdf` stores the provenance under another key than `attrs["scan"]["dimensions"]`, read `Scan.to_dict` and the netCDF code of `result/scan.py` and use the key they use (the record must be in the provenance; do not change where the provenance lives).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/simulation/test_design_record.py`
Expected: FAIL with `ImportError: cannot import name 'Design'`.

- [ ] **Step 3: Write the implementation**

In `src/sbmlsim/simulation/scan.py`, add `import json` and, before `Dimension`:

```python
@dataclass(frozen=True, eq=False)
class Design:
    """The record of the design of a dimension, see `sbmlsim.simulation.sampling`.

    It is made of JSON types, so `Dimension.to_dict` writes it into the
    provenance of a result and an analysis reads it from there, also from a
    result stored as netCDF. Two records are equal when their contents are.

    Attributes:
        method: the design, e.g. `local`, `random`, `lhs`, `sobol`, `fast`,
            `morris`, `fit_parameters`, `profile_parameters`, `fit_repeats`
            or `population`.
        distributions: target -> its distribution as a dictionary.
        options: the options of the design, e.g. `n` and `seed`.
        references: target -> `{"value": ..., "unit": ...}`, the references
            the design resolved.
    """

    method: str
    distributions: dict[str, dict[str, Any]] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    references: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check that the record is made of JSON types.

        Raises:
            ValueError: if a part of the record is no JSON type.
        """
        try:
            json.dumps(self.to_dict())
        except TypeError as err:
            raise ValueError(
                f"The record of the design '{self.method}' must be made of JSON "
                f"types: {err}"
            ) from err

    def __eq__(self, other: object) -> bool:
        """Compare two records by their contents."""
        return isinstance(other, Design) and self.to_dict() == other.to_dict()

    def to_dict(self) -> dict[str, Any]:
        """Get the record as a dictionary of JSON types."""
        return {
            "method": self.method,
            "distributions": self.distributions,
            "options": self.options,
            "references": self.references,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Design:
        """Create a record from its dictionary.

        Args:
            d: the dictionary of `to_dict`.

        Returns:
            The record.
        """
        return cls(
            method=str(d["method"]),
            distributions=dict(d.get("distributions") or {}),
            options=dict(d.get("options") or {}),
            references=dict(d.get("references") or {}),
        )
```

In `Dimension`:

- add to the attribute docstring: "design: the record of the design which created the dimension, `None` for a dimension given by hand, see `sbmlsim.simulation.sampling`." and "coordinates: name -> a read-only array or quantity along the dimension, which the result carries as a coordinate and which is never set on a model, e.g. the covariates of a population.";
- add the fields `design: Design | None` and `coordinates: Mapping[str, Any]` after `labels`;
- add the keywords `design: Design | None = None, coordinates: Mapping[str, Any] | None = None` to `__init__` and document them in its `Raises` ("a design or coordinates of a dimension which is no dimension of values; coordinates of another length than the dimension or with the name of a target; a design which is no `Design`"); after the labels are checked:

```python
        if design is not None and not isinstance(design, Design):
            raise TypeError(
                f"The design of the dimension '{id}' is a Design, not {design!r}."
            )
        extra: dict[str, Any] = {}
        if (design is not None or coordinates) and kind is not DimensionKind.VALUES:
            raise ValueError(
                f"The dimension '{id}' of {kind} has a design or coordinates, which "
                f"only a dimension of values has."
            )
        for name, column in (coordinates or {}).items():
            if name in arrays:
                raise ValueError(
                    f"The coordinate '{name}' of the dimension '{id}' is a target of "
                    f"its values; a coordinate is never set on a model."
                )
            array = _array(name, column)
            if len(array) != len(keys):
                raise ValueError(
                    f"The coordinate '{name}' of the dimension '{id}' has the length "
                    f"{len(array)}, the dimension has {len(keys)} points."
                )
            extra[name] = array
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "coordinates", MappingProxyType(extra))
```

  (`_array` makes read-only arrays or quantities of numbers, as for the values.)
- `__getstate__`/`__setstate__`: treat `"coordinates"` like the other mappings.
- `__repr__`: append `, design={self.design.method}` when there is a design.
- `to_dict`: add

```python
            "design": None if self.design is None else self.design.to_dict(),
            "coordinates": {
                name: _encode(values) if isinstance(values, Quantity) else values.tolist()
                for name, values in self.coordinates.items()
            },
```

In `src/sbmlsim/simulator/simulator.py`, in `assemble`, after the loop over `dimension.values.items()` inside the loop over the dimensions, add:

```python
            for name, values in dimension.coordinates.items():
                if name in data_vars or name in coords:
                    continue
                if isinstance(values, Quantity):
                    coords[name] = (dimension.id, np.array(values.magnitude))
                    units[name] = str(values.units)
                else:
                    coords[name] = (dimension.id, np.array(values))
                    units[name] = ""
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/simulation/test_design_record.py tests/simulation/test_scan_definition.py tests/simulator/test_simulator.py`
Expected: PASS.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/simulation/scan.py src/sbmlsim/simulator/simulator.py tests/simulation/test_design_record.py
git commit -m "A dimension carries the record of its design and coordinates which no model sees" -m "Design is the JSON record of the design of a dimension (method, distributions, options, references), which Dimension.to_dict writes into the provenance of a result, so an analysis reads it from a result, also from netCDF. Dimension(coordinates=...) carries arrays along the dimension, e.g. the covariates of a population, which the result has as coordinates and which are never set on a model."
```

---

### Task 2: The distributions

**Files:**
- Create: `src/sbmlsim/simulation/sampling/__init__.py`, `src/sbmlsim/simulation/sampling/distributions.py`
- Test: `tests/simulation/sampling/__init__.py` (empty), `tests/simulation/sampling/test_distributions.py`

**Interfaces:**
- Produces: `Distribution` (abstract: `is_relative: bool`, `ppf(u, reference=None) -> np.ndarray | Quantity`, `cdf(x, reference=None) -> np.ndarray`, `to_dict() -> dict[str, Any]`); `Uniform(lower=None, upper=None, relative=None)`, `LogUniform(lower=None, upper=None, factor=None)`, `Normal(mean=None, sd=None, cv=None)`, `LogNormal(median=None, cv=...)`, `Truncated(distribution, lower=None, upper=None)`, `Empirical(values)`, `Fixed(value)`; `EPS = 1e-12`; `Number = float | Quantity`. The package `__init__` re-exports them (later tasks add their names).

- [ ] **Step 1: Write the failing tests**

Create `tests/simulation/sampling/__init__.py` (empty) and `tests/simulation/sampling/test_distributions.py`:

```python
"""The distributions of the sampler."""

import json

import numpy as np
import pytest
from scipy import stats

from sbmlsim import Q
from sbmlsim.simulation.sampling import (
    Empirical,
    Fixed,
    LogNormal,
    LogUniform,
    Normal,
    Truncated,
    Uniform,
)

U = np.array([0.0, 0.1, 0.5, 0.9, 1.0])


def test_uniform_and_relative_uniform() -> None:
    np.testing.assert_allclose(Uniform(2.0, 4.0).ppf(U), 2.0 + 2.0 * U)
    np.testing.assert_allclose(Uniform(relative=0.5).ppf(U, 10.0), 5.0 + 10.0 * U)
    assert Uniform(relative=0.5).is_relative and not Uniform(2.0, 4.0).is_relative


def test_log_uniform_is_uniform_in_log10() -> None:
    values = LogUniform(1e-2, 1e2).ppf(U)
    np.testing.assert_allclose(np.log10(values), -2.0 + 4.0 * U)
    np.testing.assert_allclose(LogUniform(factor=10.0).ppf(U, 5.0), 10 ** (np.log10(0.5) + 2.0 * U))


def test_normal_and_lognormal_against_scipy() -> None:
    u = U[1:-1]
    np.testing.assert_allclose(Normal(3.0, 2.0).ppf(u), stats.norm(3.0, 2.0).ppf(u))
    np.testing.assert_allclose(Normal(cv=0.1).ppf(u, 10.0), stats.norm(10.0, 1.0).ppf(u))
    sigma = np.sqrt(np.log(1.0 + 0.2**2))
    np.testing.assert_allclose(
        LogNormal(5.0, 0.2).ppf(u), stats.lognorm(sigma, scale=5.0).ppf(u)
    )
    np.testing.assert_allclose(
        LogNormal(cv=0.2).ppf(u, 5.0), stats.lognorm(sigma, scale=5.0).ppf(u)
    )


def test_the_probabilities_zero_and_one_are_clipped_for_unbounded_distributions() -> None:
    values = Normal(0.0, 1.0).ppf(np.array([0.0, 1.0]))
    assert np.isfinite(values).all()
    assert values[0] < -6.0 and values[1] > 6.0


def test_truncated_restricts_the_probabilities() -> None:
    truncated = Truncated(Normal(0.0, 1.0), lower=0.0)
    values = truncated.ppf(np.linspace(0.0, 1.0, 11))
    assert values.min() >= 0.0
    np.testing.assert_allclose(truncated.ppf(np.array([0.5])), stats.halfnorm.ppf(0.5))


def test_empirical_and_fixed() -> None:
    empirical = Empirical([3.0, 1.0, 2.0])
    np.testing.assert_array_equal(empirical.ppf(np.array([0.0, 0.4, 0.99, 1.0])), [1.0, 2.0, 3.0, 3.0])
    np.testing.assert_array_equal(Fixed(7.0).ppf(U), np.full(5, 7.0))


def test_quantities_keep_their_unit() -> None:
    values = Normal(Q(75.0, "kg"), Q(12.0, "kg")).ppf(np.array([0.5]))
    assert values.units == Q(1.0, "kg").units
    assert values.magnitude[0] == pytest.approx(75.0)
    mixed = Truncated(Normal(Q(75.0, "kg"), Q(12.0, "kg")), lower=40.0)
    assert mixed.ppf(np.array([0.0])).magnitude[0] >= 40.0
    grams = Uniform(Q(1.0, "kg"), Q(2000.0, "g")).ppf(np.array([1.0]))
    assert grams.to("kg").magnitude[0] == pytest.approx(2.0)
    relative = LogNormal(cv=0.1).ppf(np.array([0.5]), Q(5.0, "mg"))
    assert str(relative.units) == "milligram"


def test_a_relative_distribution_needs_a_reference() -> None:
    with pytest.raises(ValueError, match="model="):
        LogNormal(cv=0.1).ppf(U)


def test_a_log_distribution_needs_a_positive_reference() -> None:
    with pytest.raises(ValueError, match="positive"):
        LogNormal(cv=0.1).ppf(U, -1.0)
    with pytest.raises(ValueError, match="positive"):
        LogUniform(factor=2.0).ppf(U, 0.0)


def test_a_negative_reference() -> None:
    values = Uniform(relative=0.5).ppf(np.array([0.0, 1.0]), -10.0)
    np.testing.assert_allclose(values, [-15.0, -5.0])
    assert Normal(cv=0.1).ppf(np.array([0.5]), -10.0)[0] == pytest.approx(-10.0)


@pytest.mark.parametrize(
    "make",
    [
        lambda: Uniform(4.0, 2.0),
        lambda: Uniform(2.0),
        lambda: Uniform(2.0, 4.0, relative=0.1),
        lambda: LogUniform(0.0, 1.0),
        lambda: LogUniform(factor=1.0),
        lambda: Normal(1.0),
        lambda: Normal(sd=1.0, cv=0.1),
        lambda: Normal(1.0, -1.0),
        lambda: LogNormal(cv=-0.1),
        lambda: Empirical([]),
        lambda: Truncated(Normal(0.0, 1.0), lower=2.0, upper=1.0),
    ],
)
def test_invalid_distributions_raise(make: object) -> None:
    with pytest.raises(ValueError):
        make()  # ty: ignore[call-non-callable]


def test_the_distributions_serialize() -> None:
    distributions = [
        Uniform(1.0, 2.0),
        Uniform(relative=0.1),
        LogUniform(factor=3.0),
        Normal(Q(75.0, "kg"), Q(12.0, "kg")),
        LogNormal(cv=0.2),
        Truncated(Normal(0.0, 1.0), lower=0.0),
        Empirical([1.0, 2.0]),
        Fixed(Q(3.0, "mg")),
    ]
    data = json.loads(json.dumps([d.to_dict() for d in distributions]))
    assert data[0] == {"type": "Uniform", "lower": 1.0, "upper": 2.0, "relative": None}
    assert data[3]["mean"] == {"value": 75.0, "unit": "kilogram"}
    assert data[5]["distribution"]["type"] == "Normal"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/simulation/sampling/test_distributions.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbmlsim.simulation.sampling'`.

- [ ] **Step 3: Write the implementation**

Create `src/sbmlsim/simulation/sampling/__init__.py`:

```python
"""The sampler: distributions and the designs of a scan.

A design is a dimension of a scan whose values are drawn from distributions,
see `sbmlsim.simulation.sampling.designs`; it carries the record of how it was
drawn, `Design`, which the result keeps as its provenance. A distribution
gives the values of a target at probabilities of the unit cube, so every
design works with every marginal; a distribution without a location is
relative to the reference of its target, see `references`.
"""

from sbmlsim.simulation.sampling.distributions import (
    Distribution,
    Empirical,
    Fixed,
    LogNormal,
    LogUniform,
    Normal,
    Truncated,
    Uniform,
)
from sbmlsim.simulation.scan import Design

__all__ = [
    "Design",
    "Distribution",
    "Empirical",
    "Fixed",
    "LogNormal",
    "LogUniform",
    "Normal",
    "Truncated",
    "Uniform",
]
```

Create `src/sbmlsim/simulation/sampling/distributions.py`:

```python
"""The distributions of the sampler.

A distribution gives the values of a target at probabilities `u` of the unit
cube, `ppf(u, reference)`, the inverse of its cumulative distribution
function, so every design (random draws, Latin hypercubes, the designs of
SALib) works with every marginal. Its numbers are floats in the unit of the
target in the model or quantities; a plain number next to a quantity is in the
unit of the quantity. A distribution without a location (`Uniform(relative=)`,
`LogUniform(factor=)`, `Normal(cv=)`, `LogNormal(cv=)`) is relative to the
reference of its target, the value the model gives it after the
pre-initialization, see `sbmlsim.simulation.sampling.references`.

The values are a quantity when the distribution has a unit (of a quantity or
of its reference) and plain floats otherwise. The probabilities 0 and 1 of an
unbounded distribution are clipped to `EPS` from the bounds of the unit
interval, so its values are finite.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, override

import numpy as np
from scipy import stats

from sbmlsim.simulation.definition import _encode
from sbmlsim.units import Quantity, ureg

#: the distance of the clipped probabilities 0 and 1 from the bounds
EPS: float = 1e-12

#: a number in the unit of a target in the model, or a quantity
Number = float | Quantity


def _unit_of(*values: Any) -> str | None:
    """Get the unit of the first quantity, `None` without one."""
    return next((str(v.units) for v in values if isinstance(v, Quantity)), None)


def _magnitude(value: Number, unit: str | None) -> float:
    """Get a number in a unit; a plain number is in it already."""
    if isinstance(value, Quantity):
        return float(value.to(unit).magnitude) if unit else float(value.magnitude)
    return float(value)


def _values(magnitudes: np.ndarray, unit: str | None) -> np.ndarray | Quantity:
    """Give the values their unit, plain floats without one."""
    return ureg.Quantity(magnitudes, unit) if unit else magnitudes


def _reference(reference: Number | None, distribution: Distribution) -> tuple[float, str | None]:
    """Split a reference into its number and its unit.

    Raises:
        ValueError: without a reference.
    """
    if reference is None:
        raise ValueError(
            f"{distribution!r} is relative to the reference of its target; give the "
            f"design the model (model=) to read it."
        )
    if isinstance(reference, Quantity):
        return float(reference.magnitude), str(reference.units)
    return float(reference), None


def _positive(value: float, distribution: Distribution) -> float:
    """Check that a location of a logarithmic distribution is positive.

    Raises:
        ValueError: if it is not.
    """
    if not value > 0.0:
        raise ValueError(f"{distribution!r} needs a positive location, not {value}.")
    return value


def _clip(u: Any) -> np.ndarray:
    """Clip probabilities into the open unit interval."""
    return np.clip(np.asarray(u, dtype=float), EPS, 1.0 - EPS)


class Distribution(ABC):
    """A distribution of the values of a target, see the module."""

    @property
    def is_relative(self) -> bool:
        """Check whether the location is the reference of the target."""
        return False

    @abstractmethod
    def ppf(self, u: Any, reference: Number | None = None) -> np.ndarray | Quantity:
        """Get the values at probabilities of the unit interval.

        Args:
            u: the probabilities, an array.
            reference: the reference of the target, which a relative
                distribution needs.

        Returns:
            The values, a quantity where the distribution has a unit.

        Raises:
            ValueError: if a relative distribution has no reference or its
                reference does not fit.
        """

    @abstractmethod
    def cdf(self, x: Any, reference: Number | None = None) -> np.ndarray:
        """Get the probabilities of values, the inverse of `ppf`.

        Args:
            x: the values, numbers in the unit of the distribution.
            reference: the reference of the target.

        Returns:
            The probabilities.
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Get the distribution as a dictionary of JSON types."""


@dataclass(frozen=True)
class Uniform(Distribution):
    """Uniform in `[lower, upper]`, or in the reference times `[1 - relative, 1 + relative]`.

    Attributes:
        lower: the lower bound.
        upper: the upper bound, not below `lower`.
        relative: the relative half width around the reference.
    """

    lower: Number | None = None
    upper: Number | None = None
    relative: float | None = None

    def __post_init__(self) -> None:
        """Check the definition.

        Raises:
            ValueError: unless either both bounds or a positive `relative`
                are given, or if the bounds are no interval.
        """
        bounds = self.lower is not None and self.upper is not None
        if self.relative is not None:
            if self.lower is not None or self.upper is not None:
                raise ValueError(f"{self!r} has bounds or a relative width, not both.")
            if not self.relative > 0.0:
                raise ValueError(f"{self!r} needs a positive relative width.")
        elif not bounds:
            raise ValueError(f"{self!r} needs a lower and an upper bound, or relative=.")
        else:
            lower, upper = self._bounds(None)
            if lower > upper:
                raise ValueError(f"{self!r}: the lower bound is above the upper one.")

    @property
    @override
    def is_relative(self) -> bool:
        """Check whether the bounds are relative to the reference."""
        return self.relative is not None

    def _bounds(self, reference: Number | None) -> tuple[float, float]:
        """Get the bounds in the unit of the distribution."""
        if self.relative is not None:
            value, _ = _reference(reference, self)
            a, b = value * (1.0 - self.relative), value * (1.0 + self.relative)
            return min(a, b), max(a, b)
        unit = _unit_of(self.lower, self.upper)
        assert self.lower is not None and self.upper is not None
        return _magnitude(self.lower, unit), _magnitude(self.upper, unit)

    def _unit(self, reference: Number | None) -> str | None:
        """Get the unit of the values."""
        if self.relative is not None:
            return _reference(reference, self)[1]
        return _unit_of(self.lower, self.upper)

    @override
    def ppf(self, u: Any, reference: Number | None = None) -> np.ndarray | Quantity:
        """Get the values, `lower + u (upper - lower)`."""
        lower, upper = self._bounds(reference)
        return _values(lower + np.asarray(u, dtype=float) * (upper - lower), self._unit(reference))

    @override
    def cdf(self, x: Any, reference: Number | None = None) -> np.ndarray:
        """Get the probabilities of values."""
        lower, upper = self._bounds(reference)
        span = upper - lower
        x = np.asarray(x, dtype=float)
        if span == 0.0:
            return (x >= lower).astype(float)
        return np.clip((x - lower) / span, 0.0, 1.0)

    @override
    def to_dict(self) -> dict[str, Any]:
        """Get the distribution as a dictionary of JSON types."""
        return {
            "type": "Uniform",
            "lower": _encode(self.lower),
            "upper": _encode(self.upper),
            "relative": self.relative,
        }
```

Replace the `assert` in `_bounds` (ruff and the conventions of the package allow no `assert` for type narrowing in library code): use `if self.lower is None or self.upper is None: raise ValueError(...)` instead; it never fires after `__post_init__`. Write the other distributions in the same pattern:

- `LogUniform(lower=None, upper=None, factor=None)`: either both bounds (positive, `lower <= upper`) or `factor > 1` (the bounds `reference / factor` and `reference * factor`, the reference positive, see `_positive`); `ppf(u) = 10 ** (log10(lower) + u * (log10(upper) - log10(lower)))` (this formula exactly, the start values of a fit depend on it, Task 8); `cdf` in log10 space; `is_relative` when `factor` is given; `to_dict` with `"type": "LogUniform"`.
- `Normal(mean=None, sd=None, cv=None)`: exactly one of `sd` (positive) and `cv` (positive); `mean` defaults to the reference (then `is_relative`); with `cv` the standard deviation is `cv * |mean|`; `ppf(u) = mean + sd * stats.norm.ppf(_clip(u))`; `cdf(x) = stats.norm.cdf((x - mean) / sd)`.
- `LogNormal(median=None, cv=...)`: `cv` positive and required (`cv: float = field(kw_only=True)` is not needed, give `cv: float | None = None` and raise when it is missing); `median` defaults to the reference (then `is_relative`) and must be positive; `sigma = sqrt(ln(1 + cv²))`; `ppf(u) = median * exp(sigma * stats.norm.ppf(_clip(u)))`; `cdf(x) = stats.norm.cdf(log(x / median) / sigma)` for positive `x`, `0` otherwise.
- `Truncated(distribution, lower=None, upper=None)`: at least one bound; `lower < upper`; the bounds in the unit of the inner distribution (`_unit_of(lower, upper)` or the unit of the inner values); `is_relative` of the inner one; `ppf(u) = inner.ppf(F_lo + u (F_hi - F_lo))` with `F_lo = inner.cdf(lower)` (`0` without), `F_hi = inner.cdf(upper)` (`1` without); raise when `F_hi <= F_lo` (an interval without probability); `cdf` accordingly; `to_dict` with the inner distribution's dictionary under `"distribution"`.
- `Empirical(values)`: a non-empty sequence of numbers or a quantity array; the sorted values; `ppf(u) = values[min(floor(u n), n - 1)]`; `cdf(x) = searchsorted(values, x, side="right") / n`; store the values as a tuple of floats and the unit separately (`field(init=False)` set in `__post_init__` with `object.__setattr__`), so the dataclass stays frozen and hashable.
- `Fixed(value)`: `ppf(u) = full(shape of u, value)`; `cdf(x) = (x >= value)`.

Add the seven classes and `Distribution` to the package `__init__` (as above).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/simulation/sampling/test_distributions.py`
Expected: PASS.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/simulation/sampling tests/simulation/sampling
git commit -m "Distributions give the values of a target at probabilities of the unit cube" -m "Uniform, LogUniform, Normal, LogNormal, Truncated, Empirical and Fixed map probabilities through their inverse CDF, so every design of the sampler works with every marginal. A distribution without a location is relative to the reference of its target; numbers are in the unit of the target or quantities, and the values keep the unit."
```

---

### Task 3: The references of targets

**Files:**
- Create: `src/sbmlsim/simulation/sampling/references.py`
- Modify: `src/sbmlsim/simulation/sampling/__init__.py` (exports)
- Test: `tests/simulation/sampling/test_references.py`

**Interfaces:**
- Consumes: `Simulator().load(model)`, `Simulator().compile(loaded, simulation) -> Plan` (merges the changes of the model as defaults), `RoadrunnerSBMLModel.initialize(assignments)`, `sbmlsim.simulator.plan.preinit_targets(plan)`, `RoadrunnerSBMLModel.has_selection(name)`, `.uinfo`, `.parameters`, `.r_loaded`.
- Produces: `references(model: ModelLike, targets: Sequence[str], simulation: Simulation | None = None) -> dict[str, Number]`; `parameters_of(model: ModelLike, *, species: bool = False, exclude: Callable[[str], bool] | Collection[str] | None = None, exclude_zero: bool = True, simulation: Simulation | None = None) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/simulation/sampling/test_references.py`:

```python
"""The references of the targets of a design."""

import pytest

from sbmlsim import Q
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import Simulation
from sbmlsim.simulation.sampling import parameters_of, references
from sbmlsim.simulator import Simulator
from tests.simulator.models import sbml, sbml_minutes


def test_the_references_are_the_values_after_the_preinitialization() -> None:
    model = Simulator().load(sbml())
    refs = references(model, ["k1", "pinit", "X"])
    assert refs == {"k1": 0.8, "pinit": 4.0, "X": 12.0}
    changed = references(model, ["k1", "pinit"], Simulation(end=1, preinit_changes={"f": 5.0, "k1": 2.0}))
    # pinit = 2 f is an initial assignment of the changed f
    assert changed == {"k1": 2.0, "pinit": 10.0}


def test_the_changes_of_the_model_are_defaults() -> None:
    model = RoadrunnerSBMLModel(source=sbml(), changes={"f": 3.0})
    assert references(model, ["pinit"])["pinit"] == pytest.approx(6.0)


def test_a_reference_has_the_unit_of_its_target() -> None:
    refs = references(Simulator().load(sbml_minutes()), ["f"])
    assert refs["f"] == Q(2.0, "mg")


def test_an_unknown_target_raises() -> None:
    with pytest.raises(ValueError, match="'nope'"):
        references(Simulator().load(sbml()), ["nope"])


def test_the_parameters_of_a_model() -> None:
    model = Simulator().load(sbml())
    parameters = parameters_of(model)
    assert {"a0", "b0", "k1", "k2", "f"} <= set(parameters)
    assert not any(p.endswith("__initial") for p in parameters)
    assert "k1" not in parameters_of(model, exclude={"k1"})
    assert "k1" not in parameters_of(model, exclude=lambda pid: pid.startswith("k"))
    assert {"A", "B", "X"} <= set(parameters_of(model, species=True))
    zero = RoadrunnerSBMLModel(source=sbml(), changes={"k2": 0.0})
    assert "k2" not in parameters_of(zero)
    assert "k2" in parameters_of(zero, exclude_zero=False)


def test_a_model_path_is_loaded() -> None:
    assert references(sbml(), ["k1"]) == {"k1": 0.8}
```

The values follow from the probe model of `tests/simulator/models.py` (`k1 = 0.8`, `f = 2`, `pinit = 2*f`, `X = 3*pinit`); `sbml_minutes()` gives `f` the unit `mg`. If `RoadrunnerSBMLModel` takes its changes under another keyword than `changes`, use the one `AbstractModel` documents.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/simulation/sampling/test_references.py`
Expected: FAIL with `ImportError: cannot import name 'references'`.

- [ ] **Step 3: Write the implementation**

Create `src/sbmlsim/simulation/sampling/references.py`:

```python
"""The references of the targets of a design.

The reference of a target is the value the model gives it after the
pre-initialization of a simulation: the changes of the model and of the
simulation are set, and the initial assignments which read them are evaluated
again. A relative distribution is centred on it, and a local design varies
around it.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence

import libsbml
import numpy as np

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation.definition import Simulation
from sbmlsim.simulation.sampling.distributions import Number
from sbmlsim.simulator.plan import preinit_targets
from sbmlsim.simulator.simulator import ModelLike, Simulator
from sbmlsim.units import ureg

#: a reference below this magnitude is zero for `parameters_of`
ZERO: float = 1e-8


def _loaded(model: ModelLike) -> RoadrunnerSBMLModel:
    """Get a loaded model; a loaded one is used as it is, with its settings."""
    if isinstance(model, RoadrunnerSBMLModel) and model.r is not None:
        return model
    return Simulator().load(model)


def references(
    model: ModelLike,
    targets: Sequence[str],
    simulation: Simulation | None = None,
) -> dict[str, Number]:
    """Get the value every target has after the pre-initialization.

    The simulation is compiled with the changes of the model as defaults and
    the model is initialized with its plan, which every simulation of the
    model does anyway, so the model is left initialized.

    Args:
        model: the model, loaded or a source.
        targets: the targets, selections of roadrunner, e.g. `k1`, `S` or `[S]`.
        simulation: the simulation whose pre-initialization counts,
            `Simulation(end=1)` by default.

    Returns:
        Target -> its value, a quantity in the unit of the target in the model
        or a float without a unit.

    Raises:
        ValueError: if a target is no selection of the model.
    """
    loaded = _loaded(model)
    plan = Simulator().compile(loaded, simulation or Simulation(end=1.0))
    loaded.initialize(preinit_targets(plan))
    values: dict[str, Number] = {}
    for target in targets:
        if not loaded.has_selection(target):
            raise ValueError(f"'{target}' is no target of the model.")
        value = float(loaded.r_loaded.getValue(target))
        unit = loaded.uinfo.get(target, "") or ""
        values[target] = ureg.Quantity(value, unit) if unit else value
    return values


def parameters_of(
    model: ModelLike,
    *,
    species: bool = False,
    exclude: Callable[[str], bool] | Collection[str] | None = None,
    exclude_zero: bool = True,
    simulation: Simulation | None = None,
) -> list[str]:
    """List the constant parameters of a model, which an analysis of all parameters varies.

    Args:
        model: the model, loaded or a source.
        species: list the species as well (their amounts).
        exclude: ids, or a function which is true for an id, to leave out.
        exclude_zero: leave out a target whose reference is below `ZERO` in
            magnitude, which a relative change does not change.
        simulation: the simulation whose pre-initialization gives the
            references of `exclude_zero`.

    Returns:
        The ids, sorted, without the helper parameters sbmlsim adds.
    """
    loaded = _loaded(model)
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(loaded.r_loaded.getSBML())
    sbml_model: libsbml.Model = doc.getModel()
    ids = [p.getId() for p in sbml_model.getListOfParameters() if p.getConstant()]
    if species:
        ids.extend(s.getId() for s in sbml_model.getListOfSpecies())
    helpers = set(loaded.parameters)
    excluded: Callable[[str], bool]
    if exclude is None:
        excluded = lambda sid: False  # noqa: E731
    elif callable(exclude):
        excluded = exclude
    else:
        excluded = set(exclude).__contains__
    ids = sorted(sid for sid in ids if sid not in helpers and not excluded(sid))
    if exclude_zero:
        refs = references(loaded, ids, simulation)
        ids = [
            sid
            for sid in ids
            if abs(float(getattr(refs[sid], "magnitude", refs[sid]))) >= ZERO
        ]
    return ids
```

Replace the lambda by a small module function (`_nothing(sid) -> bool`) if ruff flags it. If `RoadrunnerSBMLModel.parameters` is not the collection of the helper parameters, find the attribute `set_selections` excludes (`exclude=set(self.parameters)`) and use it. Export `references` and `parameters_of` from the package `__init__`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/simulation/sampling/test_references.py`
Expected: PASS.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/simulation/sampling tests/simulation/sampling/test_references.py
git commit -m "The sampler reads the references of targets after the pre-initialization" -m "references(model, targets, simulation) gives the value every target has once the changes of the model and of the simulation are set and the initial assignments are evaluated again, in the unit of the target; parameters_of lists the constant parameters (and species) an analysis of all parameters varies, without the helpers of sbmlsim and the zeros."
```

---

### Task 4: The designs local, random and lhs

**Files:**
- Create: `src/sbmlsim/simulation/sampling/designs.py`
- Modify: `src/sbmlsim/simulation/sampling/__init__.py` (exports)
- Test: `tests/simulation/sampling/test_designs.py`

**Interfaces:**
- Consumes: `Distribution.ppf`, `.to_dict`, `.is_relative` (Task 2); `references` (Task 3); `Design`, `Dimension(..., design=, coordinates=)` (Task 1).
- Produces:
  - `local(targets: Sequence[str], delta: float = 0.1, *, model: ModelLike, simulation: Simulation | None = None, id: str = "local") -> Dimension`.
  - `random(distributions: Mapping[str, Distribution], n: int, *, seed: int | None = None, correlation: ArrayLike | None = None, model: ModelLike | None = None, simulation: Simulation | None = None, id: str = "random") -> Dimension`.
  - `lhs(...)` with the signature of `random` and `id="lhs"`.
  - Helpers the later tasks use: `_seed(seed) -> int`, `_count(n, name) -> int`, `_resolve(distributions, model, simulation) -> dict[str, Number]`, `_values(distributions, refs, u) -> dict[str, Any]`, `_record(method, distributions, options, refs) -> Design`, `_encode_reference(value) -> dict[str, Any]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/simulation/sampling/test_designs.py`:

```python
"""The designs local, random and lhs."""

import numpy as np
import pytest
from scipy import stats

from sbmlsim import Q
from sbmlsim.simulation import Dimension, Formula, Scan, Simulation
from sbmlsim.simulation.sampling import (
    LogNormal,
    Normal,
    Uniform,
    lhs,
    local,
    random,
)
from sbmlsim.simulator import Simulator
from tests.simulator.models import sbml, sbml_minutes


@pytest.fixture(scope="module")
def model() -> object:
    return Simulator().load(sbml())


def test_local_varies_every_target_around_its_reference(model: object) -> None:
    dimension = local(["k1", "k2"], delta=0.1, model=model)
    assert dimension.labels.tolist() == ["reference", "k1+", "k1-", "k2+", "k2-"]
    np.testing.assert_allclose(dimension.values["k1"], [0.8, 0.88, 0.72, 0.8, 0.8])
    np.testing.assert_allclose(dimension.values["k2"], [0.6, 0.6, 0.6, 0.66, 0.54])
    assert dimension.design.method == "local"
    assert dimension.design.options == {"delta": 0.1, "targets": ["k1", "k2"]}
    assert dimension.design.references["k1"] == {"value": 0.8, "unit": ""}


def test_local_needs_a_delta_below_one(model: object) -> None:
    with pytest.raises(ValueError, match="delta"):
        local(["k1"], delta=1.0, model=model)


def test_random_draws_from_every_marginal() -> None:
    dimension = random({"a": Uniform(0.0, 1.0), "b": Normal(5.0, 1.0)}, 2000, seed=3)
    assert len(dimension) == 2000 and dimension.labels.tolist()[:3] == [0, 1, 2]
    assert stats.kstest(dimension.values["a"], "uniform").pvalue > 0.01
    assert stats.kstest(dimension.values["b"], stats.norm(5.0, 1.0).cdf).pvalue > 0.01
    assert dimension.design.options["seed"] == 3 and dimension.design.options["n"] == 2000


def test_random_and_lhs_use_the_draws_of_numpy_and_scipy() -> None:
    u = np.random.default_rng(5).random((4, 2))
    dimension = random({"a": Uniform(0.0, 1.0), "b": Uniform(0.0, 1.0)}, 4, seed=5)
    np.testing.assert_array_equal(np.column_stack([dimension.values["a"], dimension.values["b"]]), u)
    from scipy.stats import qmc

    v = qmc.LatinHypercube(d=2, rng=np.random.default_rng(5)).random(n=4)
    dimension = lhs({"a": Uniform(0.0, 1.0), "b": Uniform(0.0, 1.0)}, 4, seed=5)
    np.testing.assert_array_equal(np.column_stack([dimension.values["a"], dimension.values["b"]]), v)


def test_lhs_has_one_point_per_stratum() -> None:
    dimension = lhs({"a": Uniform(0.0, 1.0), "b": Uniform(0.0, 1.0)}, 50, seed=1)
    for target in ("a", "b"):
        strata = np.floor(np.asarray(dimension.values[target]) * 50).astype(int)
        assert sorted(strata.tolist()) == list(range(50))


def test_a_correlation_is_reached() -> None:
    correlation = [[1.0, 0.7], [0.7, 1.0]]
    for design in (random, lhs):
        dimension = design(
            {"a": LogNormal(1.0, 0.3), "b": Uniform(0.0, 1.0)}, 2000, seed=2, correlation=correlation
        )
        rho = stats.spearmanr(dimension.values["a"], dimension.values["b"]).statistic
        assert rho == pytest.approx(0.7, abs=0.05)
    correlated = lhs({"a": Uniform(0.0, 1.0), "b": Uniform(0.0, 1.0)}, 40, seed=2, correlation=correlation)
    strata = np.floor(np.asarray(correlated.values["a"]) * 40).astype(int)
    assert sorted(strata.tolist()) == list(range(40))


@pytest.mark.parametrize(
    "correlation", [[[1.0, 0.5], [0.4, 1.0]], [[1.0, 2.0], [2.0, 1.0]], [[2.0, 0.0], [0.0, 1.0]], [[1.0]]]
)
def test_an_invalid_correlation_raises(correlation: list[list[float]]) -> None:
    with pytest.raises(ValueError, match="correlation"):
        random({"a": Uniform(0.0, 1.0), "b": Uniform(0.0, 1.0)}, 10, seed=1, correlation=correlation)


def test_relative_distributions_need_the_model(model: object) -> None:
    with pytest.raises(ValueError, match="model="):
        random({"k1": LogNormal(cv=0.1)}, 5, seed=1)
    dimension = random({"k1": LogNormal(cv=0.1)}, 2000, seed=1, model=model)
    assert np.median(dimension.values["k1"]) == pytest.approx(0.8, rel=0.02)
    assert dimension.design.references["k1"] == {"value": 0.8, "unit": ""}


def test_the_seed_of_the_record_reproduces_the_design() -> None:
    first = random({"a": Normal(0.0, 1.0)}, 5)
    seed = first.design.options["seed"]
    assert isinstance(seed, int)
    again = random({"a": Normal(0.0, 1.0)}, 5, seed=seed)
    np.testing.assert_array_equal(first.values["a"], again.values["a"])


def test_a_design_runs_in_a_scan() -> None:
    model = Simulator().load(sbml_minutes())
    design = random({"f": Uniform(Q(1000.0, "ug"), Q(3000.0, "ug"))}, 6, seed=4)
    doses = Dimension("k", values={"k1": [0.5, 1.0]})
    res = Simulator().run(
        model, Scan(Simulation(end=1, steps=2), [design, doses]), [Formula("f_mg", "f")]
    )
    # the model sees f in mg: the values of the design are converted
    np.testing.assert_allclose(
        res["f_mg"].isel(k=0, time=0).values, np.asarray(design.values["f"].to("mg").magnitude)
    )
    assert res.ds.attrs["scan"]["dimensions"][0]["design"]["method"] == "random"
```

The formula `f` is a timecourse of the selection `f` in mg; the point of the test is that the model gets the converted value. Type the fixture `model` as `RoadrunnerSBMLModel` (ty checks tests).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/simulation/sampling/test_designs.py`
Expected: FAIL with `ImportError: cannot import name 'lhs'`.

- [ ] **Step 3: Write the implementation**

Create `src/sbmlsim/simulation/sampling/designs.py`:

```python
"""The designs of a scan: dimensions whose values follow a design.

Every design returns a `Dimension` of values with a `Design` record, which a
scan writes into the provenance of its result:

- `local`: the reference point and every target alone at `1 + delta` and
  `1 - delta` times its reference;
- `random`: independent draws, with a correlation a Gaussian copula;
- `lhs`: a Latin hypercube, with a correlation the rank reordering of Iman
  and Conover, which keeps one point per stratum;
- `sobol`, `fast`, `morris`: the designs of SALib for the sensitivity
  analyses.

A design draws points of the unit cube and maps them through the inverse CDF
of the distribution of every target, see
`sbmlsim.simulation.sampling.distributions`. A relative distribution reads the
reference of its target from the model, see `references`. Every random design
takes `seed`; `None` draws a seed, which the record keeps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.stats import qmc

from sbmlsim.simulation.definition import Simulation
from sbmlsim.simulation.sampling.distributions import Distribution, Number
from sbmlsim.simulation.sampling.references import references
from sbmlsim.simulation.scan import Design, Dimension
from sbmlsim.simulator.simulator import ModelLike
from sbmlsim.units import Quantity, ureg


def _seed(seed: int | None) -> int:
    """Get the seed of a design; `None` draws one, which the record keeps.

    Raises:
        TypeError: if the seed is no integer.
    """
    if seed is None:
        return int(np.random.SeedSequence().generate_state(1)[0])
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
        raise TypeError(f"The seed of a design is an integer, not {seed!r}.")
    return int(seed)


def _count(n: int, name: str = "n") -> int:
    """Check a number of points.

    Raises:
        ValueError: if it is no positive integer.
    """
    if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 1:
        raise ValueError(f"'{name}' of a design is a positive integer, not {n!r}.")
    return int(n)


def _check(distributions: Mapping[str, Distribution]) -> None:
    """Check the distributions of a design.

    Raises:
        TypeError: if they are no mapping of targets to distributions.
        ValueError: if there is none.
    """
    if not isinstance(distributions, Mapping):
        raise TypeError(
            f"The distributions of a design map targets to distributions, not "
            f"{distributions!r}."
        )
    if not distributions:
        raise ValueError("A design needs at least one distribution.")
    for target, distribution in distributions.items():
        if not isinstance(distribution, Distribution):
            raise TypeError(f"'{target}': {distribution!r} is no distribution.")


def _resolve(
    distributions: Mapping[str, Distribution],
    model: ModelLike | None,
    simulation: Simulation | None,
) -> dict[str, Number]:
    """Read the references of the relative distributions.

    Raises:
        ValueError: if a distribution is relative and there is no model.
    """
    _check(distributions)
    relative = [t for t, d in distributions.items() if d.is_relative]
    if not relative:
        return {}
    if model is None:
        raise ValueError(
            f"The distributions of {relative} are relative to the references of "
            f"their targets: give the design the model (model=)."
        )
    return references(model, relative, simulation)


def _values(
    distributions: Mapping[str, Distribution],
    refs: Mapping[str, Number],
    u: np.ndarray,
) -> dict[str, Any]:
    """Map the points of the unit cube through the distributions, a column per target.

    Raises:
        ValueError: if a distribution does not fit its reference.
    """
    values: dict[str, Any] = {}
    for k, (target, distribution) in enumerate(distributions.items()):
        try:
            values[target] = distribution.ppf(u[:, k], refs.get(target))
        except ValueError as err:
            raise ValueError(f"'{target}': {err}") from err
    return values


def _encode_reference(value: Number) -> dict[str, Any]:
    """Encode a reference for the record."""
    if isinstance(value, Quantity):
        return {"value": float(value.magnitude), "unit": str(value.units)}
    return {"value": float(value), "unit": ""}


def _record(
    method: str,
    distributions: Mapping[str, Distribution],
    options: Mapping[str, Any],
    refs: Mapping[str, Number],
) -> Design:
    """Create the record of a design."""
    return Design(
        method=method,
        distributions={t: d.to_dict() for t, d in distributions.items()},
        options=dict(options),
        references={t: _encode_reference(v) for t, v in refs.items()},
    )


def _correlation(correlation: ArrayLike | None, d: int) -> np.ndarray | None:
    """Get the Cholesky factor of a correlation matrix, `None` without one.

    Raises:
        ValueError: if it is no symmetric, positive definite `d x d` matrix with
            ones on the diagonal.
    """
    if correlation is None:
        return None
    matrix = np.asarray(correlation, dtype=float)
    if (
        matrix.shape != (d, d)
        or not np.allclose(matrix, matrix.T)
        or not np.allclose(np.diag(matrix), 1.0)
    ):
        raise ValueError(
            f"The correlation of a design is a symmetric {d} x {d} matrix with ones "
            f"on the diagonal, not {matrix.tolist()}."
        )
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as err:
        raise ValueError(
            f"The correlation {matrix.tolist()} of a design is not positive definite."
        ) from err


def _iman_conover(u: np.ndarray, factor: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reorder the columns of a sample to the ranks of correlated scores.

    The scores are the van der Waerden scores `Φ⁻¹(i / (n + 1))` in random
    orders, freed of their sample correlation and given the target one; every
    column of the sample is sorted into their ranks, so its values, and the
    strata of a Latin hypercube, stay.

    Raises:
        ValueError: if the sample has not more points than columns.
    """
    n, d = u.shape
    if d == 1:
        return u
    if n <= d:
        raise ValueError(
            f"A correlated Latin hypercube of {d} targets needs more than {d} "
            f"points, it has {n}."
        )
    scores = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
    s = np.column_stack([rng.permutation(scores) for _ in range(d)])
    q = np.linalg.cholesky(np.corrcoef(s, rowvar=False))
    target = s @ np.linalg.inv(q).T @ factor.T
    out = np.empty_like(u)
    for k in range(d):
        ranks = np.argsort(np.argsort(target[:, k]))
        out[:, k] = np.sort(u[:, k])[ranks]
    return out


def local(
    targets: Sequence[str],
    delta: float = 0.1,
    *,
    model: ModelLike,
    simulation: Simulation | None = None,
    id: str = "local",
) -> Dimension:
    """Get the local design: the reference and every target alone at `1 ± delta` times it.

    Args:
        targets: the targets, e.g. the parameters of `parameters_of`.
        delta: the relative change, `0 < delta < 1`.
        model: the model, whose references the design varies.
        simulation: the simulation whose pre-initialization gives the references.
        id: the id of the dimension.

    Returns:
        The dimension of `2 k + 1` points, labelled `reference`, `<target>+`,
        `<target>-`.

    Raises:
        ValueError: if `delta` is not in `(0, 1)`, there is no target or a target
            is no target of the model.
    """
    if isinstance(targets, str):
        raise TypeError(f"The targets of a local design are a sequence, not {targets!r}.")
    names = list(dict.fromkeys(targets))
    if not names:
        raise ValueError("A local design needs at least one target.")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"The delta of a local design is in (0, 1), not {delta}.")
    refs = references(model, names, simulation)
    n = 2 * len(names) + 1
    values: dict[str, Any] = {}
    for j, target in enumerate(names):
        reference = refs[target]
        magnitude = float(getattr(reference, "magnitude", reference))
        column = np.full(n, magnitude)
        column[1 + 2 * j] = magnitude * (1.0 + delta)
        column[2 + 2 * j] = magnitude * (1.0 - delta)
        unit = str(reference.units) if isinstance(reference, Quantity) else None
        values[target] = ureg.Quantity(column, unit) if unit else column
    labels = ["reference", *(f"{t}{sign}" for t in names for sign in "+-")]
    record = _record("local", {}, {"delta": delta, "targets": names}, refs)
    return Dimension(id, values=values, labels=labels, design=record)


def random(
    distributions: Mapping[str, Distribution],
    n: int,
    *,
    seed: int | None = None,
    correlation: ArrayLike | None = None,
    model: ModelLike | None = None,
    simulation: Simulation | None = None,
    id: str = "random",
) -> Dimension:
    """Get `n` random draws of the distributions.

    Without a correlation the points of the unit cube are `rng.random((n, d))`;
    with one they are standard normals with the Cholesky factor of the
    correlation, mapped through the normal CDF (a Gaussian copula), so the
    rank correlation of the values is the asked one for any marginals.

    Args:
        distributions: target -> its distribution.
        n: the number of points.
        seed: the seed; `None` draws one, which the record keeps.
        correlation: the correlation of the targets, in the order of
            `distributions`.
        model: the model, which a relative distribution needs.
        simulation: the simulation whose pre-initialization gives the references.
        id: the id of the dimension.

    Returns:
        The dimension.

    Raises:
        TypeError: if the distributions or the seed have the wrong type.
        ValueError: if `n` is no positive integer, the correlation is not
            valid, or a relative distribution has no model.
    """
    n = _count(n)
    refs = _resolve(distributions, model, simulation)
    factor = _correlation(correlation, len(distributions))
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    if factor is None:
        u = rng.random((n, len(distributions)))
    else:
        u = stats.norm.cdf(rng.standard_normal((n, len(distributions))) @ factor.T)
    options = {
        "n": n,
        "seed": seed,
        "correlation": None if correlation is None else np.asarray(correlation, dtype=float).tolist(),
    }
    return Dimension(
        id,
        values=_values(distributions, refs, u),
        design=_record("random", distributions, options, refs),
    )


def lhs(
    distributions: Mapping[str, Distribution],
    n: int,
    *,
    seed: int | None = None,
    correlation: ArrayLike | None = None,
    model: ModelLike | None = None,
    simulation: Simulation | None = None,
    id: str = "lhs",
) -> Dimension:
    """Get a Latin hypercube of `n` points of the distributions.

    The points of the unit cube are `qmc.LatinHypercube(d, rng=rng).random(n)`,
    one point per stratum of every target; with a correlation the columns are
    reordered by the method of Iman and Conover, which keeps the strata.

    Args:
        distributions: target -> its distribution.
        n: the number of points, more than the number of targets with a
            correlation.
        seed: the seed; `None` draws one, which the record keeps.
        correlation: the correlation of the targets, in the order of
            `distributions`.
        model: the model, which a relative distribution needs.
        simulation: the simulation whose pre-initialization gives the references.
        id: the id of the dimension.

    Returns:
        The dimension.

    Raises:
        TypeError: if the distributions or the seed have the wrong type.
        ValueError: see `random`, and a correlated hypercube with too few points.
    """
    n = _count(n)
    refs = _resolve(distributions, model, simulation)
    factor = _correlation(correlation, len(distributions))
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    u = qmc.LatinHypercube(d=len(distributions), rng=rng).random(n=n)
    if factor is not None:
        u = _iman_conover(u, factor, rng)
    options = {
        "n": n,
        "seed": seed,
        "correlation": None if correlation is None else np.asarray(correlation, dtype=float).tolist(),
    }
    return Dimension(
        id,
        values=_values(distributions, refs, u),
        design=_record("lhs", distributions, options, refs),
    )
```

The correlation check runs before the seed is drawn, so a correlation of the wrong size raises before anything is drawn; `[[1.0]]` for two targets fails the shape check. Export `local`, `random` and `lhs` from the package `__init__`. `random` shadows the module `random` of the standard library only inside `sbmlsim.simulation.sampling`, which does not use it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/simulation/sampling/test_designs.py`
Expected: PASS. The Kolmogorov-Smirnov and correlation tests are seeded and deterministic.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/simulation/sampling tests/simulation/sampling/test_designs.py
git commit -m "The designs local, random and lhs return dimensions with their record" -m "local varies every target alone around its reference, random draws independent or correlated points (a Gaussian copula) and lhs a Latin hypercube whose correlation the rank reordering of Iman and Conover gives without losing the strata. Every design records its method, distributions, options, seed and references, and a seed of None is drawn and recorded."
```

---

### Task 5: The designs of SALib: sobol, fast, morris

**Files:**
- Modify: `src/sbmlsim/simulation/sampling/designs.py`, `src/sbmlsim/simulation/sampling/__init__.py`
- Test: `tests/simulation/sampling/test_salib_designs.py`

**Interfaces:**
- Consumes: the helpers of Task 4.
- Produces: `sobol(distributions, n, *, seed=None, second_order=False, model=None, simulation=None, id="sobol")`, `fast(distributions, n, *, m=4, seed=None, model=None, simulation=None, id="fast")`, `morris(distributions, trajectories, *, levels=4, seed=None, model=None, simulation=None, id="morris")`; `unit_cube(design: Design, d: int) -> np.ndarray` (the points of the unit cube of a recorded SALib design, which phase 2 recreates).

- [ ] **Step 1: Write the failing tests**

Create `tests/simulation/sampling/test_salib_designs.py`:

```python
"""The designs of SALib."""

import numpy as np
import pytest
from SALib.sample import fast_sampler
from SALib.sample import morris as morris_sampler
from SALib.sample import sobol as sobol_sampler

from sbmlsim.simulation.sampling import LogNormal, Normal, Uniform, fast, morris, sobol
from sbmlsim.simulation.sampling.designs import unit_cube

PROBLEM = {"num_vars": 2, "names": ["x0", "x1"], "bounds": [[2.0, 5.0], [0.0, 1.0]]}
UNIFORM = {"a": Uniform(2.0, 5.0), "b": Uniform(0.0, 1.0)}


def _stack(dimension: object) -> np.ndarray:
    return np.column_stack([np.asarray(dimension.values[t]) for t in ("a", "b")])  # ty: ignore[unresolved-attribute]


def test_sobol_equals_salib() -> None:
    dimension = sobol(UNIFORM, 16, seed=3)
    expected = sobol_sampler.sample(PROBLEM, 16, calc_second_order=False, scramble=True, seed=3)
    np.testing.assert_allclose(_stack(dimension), expected)
    assert len(dimension) == 16 * (2 + 2)
    assert len(sobol(UNIFORM, 16, seed=3, second_order=True)) == 16 * (2 * 2 + 2)


def test_sobol_needs_a_power_of_two() -> None:
    with pytest.raises(ValueError, match="power of two"):
        sobol(UNIFORM, 10, seed=1)


def test_fast_equals_salib() -> None:
    dimension = fast(UNIFORM, 65, m=4, seed=2)
    np.testing.assert_allclose(_stack(dimension), fast_sampler.sample(PROBLEM, 65, M=4, seed=2))
    with pytest.raises(ValueError, match="4"):
        fast(UNIFORM, 64, m=4, seed=2)


def test_morris_maps_the_levels_to_the_centres_of_their_strata() -> None:
    dimension = morris(UNIFORM, 10, levels=4, seed=1)
    grid = morris_sampler.sample(
        {"num_vars": 2, "names": ["x0", "x1"], "bounds": [[0.0, 1.0]] * 2}, 10, num_levels=4, seed=1
    )
    u = (grid * 3 + 0.5) / 4
    np.testing.assert_allclose(_stack(dimension), np.column_stack([2.0 + 3.0 * u[:, 0], u[:, 1]]))


def test_unbounded_marginals_stay_finite() -> None:
    distributions = {"a": Normal(0.0, 1.0), "b": LogNormal(1.0, 0.5)}
    for dimension in (sobol(distributions, 8, seed=1), fast(distributions, 65, seed=1), morris(distributions, 4, seed=1)):
        assert np.isfinite(_stack_of(dimension, ("a", "b"))).all()


def _stack_of(dimension: object, targets: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([np.asarray(dimension.values[t]) for t in targets])  # ty: ignore[unresolved-attribute]


def test_a_correlation_is_refused() -> None:
    for design in (sobol, fast, morris):
        with pytest.raises(TypeError):
            design(UNIFORM, 8, seed=1, correlation=[[1.0, 0.5], [0.5, 1.0]])  # ty: ignore[unknown-argument]


def test_the_record_recreates_the_unit_cube() -> None:
    for dimension in (sobol(UNIFORM, 8, seed=4), fast(UNIFORM, 65, seed=4), morris(UNIFORM, 5, seed=4)):
        u = unit_cube(dimension.design, 2)
        np.testing.assert_allclose(2.0 + 3.0 * u[:, 0], np.asarray(dimension.values["a"]))
```

The designs of SALib take no `correlation` (their indices assume independence), so passing one is a `TypeError` of the call; the spec's error list names this case.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/simulation/sampling/test_salib_designs.py`
Expected: FAIL with `ImportError: cannot import name 'fast'`.

- [ ] **Step 3: Write the implementation**

Add to `src/sbmlsim/simulation/sampling/designs.py`:

```python
def _problem(d: int) -> dict[str, Any]:
    """Get the problem of SALib on the unit cube of `d` dimensions."""
    return {"num_vars": d, "names": [f"x{k}" for k in range(d)], "bounds": [[0.0, 1.0]] * d}


def unit_cube(design: Design, d: int) -> np.ndarray:
    """Get the points of the unit cube of a recorded design of SALib.

    The points follow from the method, the options and the seed of the
    record, so a result of the design needs not store them; phase 2 creates
    them again for the analysis.

    Args:
        design: the record of a `sobol`, `fast` or `morris` design.
        d: the number of targets.

    Returns:
        The points, a row per point of the dimension.

    Raises:
        ValueError: if the record is of another method.
    """
    options = design.options
    if design.method == "sobol":
        from SALib.sample import sobol as sampler

        return sampler.sample(
            _problem(d),
            options["n"],
            calc_second_order=options["second_order"],
            scramble=True,
            seed=options["seed"],
        )
    if design.method == "fast":
        from SALib.sample import fast_sampler

        return fast_sampler.sample(_problem(d), options["n"], M=options["m"], seed=options["seed"])
    if design.method == "morris":
        from SALib.sample import morris as sampler

        levels = options["levels"]
        grid = sampler.sample(
            _problem(d), options["trajectories"], num_levels=levels, seed=options["seed"]
        )
        # the centre of the stratum of every level, so an unbounded marginal is finite
        return (grid * (levels - 1) + 0.5) / levels
    raise ValueError(f"The design '{design.method}' is no design of SALib.")


def _salib(
    method: str,
    distributions: Mapping[str, Distribution],
    options: dict[str, Any],
    model: ModelLike | None,
    simulation: Simulation | None,
    id: str,
) -> Dimension:
    """Create a design of SALib: its unit cube mapped through the distributions."""
    refs = _resolve(distributions, model, simulation)
    record = _record(method, distributions, options, refs)
    u = unit_cube(record, len(distributions))
    return Dimension(id, values=_values(distributions, refs, u), design=record)


def sobol(
    distributions: Mapping[str, Distribution],
    n: int,
    *,
    seed: int | None = None,
    second_order: bool = False,
    model: ModelLike | None = None,
    simulation: Simulation | None = None,
    id: str = "sobol",
) -> Dimension:
    """Get the design of Saltelli for the Sobol indices.

    `SALib.sample.sobol.sample` on the unit cube, scrambled, mapped through the
    distributions: `n (d + 2)` points, `n (2 d + 2)` with second order.

    Args:
        distributions: target -> its distribution; independent.
        n: the base number of points, a power of two.
        seed: the seed; `None` draws one, which the record keeps.
        second_order: include the points of the second order indices.
        model: the model, which a relative distribution needs.
        simulation: the simulation whose pre-initialization gives the references.
        id: the id of the dimension.

    Returns:
        The dimension.

    Raises:
        ValueError: if `n` is no power of two, or see `random`.
    """
    n = _count(n)
    if n < 2 or n & (n - 1):
        raise ValueError(f"The n of a Sobol design is a power of two, not {n}.")
    options = {"n": n, "seed": _seed(seed), "second_order": bool(second_order)}
    return _salib("sobol", distributions, options, model, simulation, id)
```

and `fast` (`n > 4 m²`, else `ValueError` naming the condition; options `{"n", "seed", "m"}`) and `morris` (`trajectories` a positive integer, `levels >= 2`; options `{"trajectories", "seed", "levels"}`) with the same structure and docstrings that name the number of points (`n d` for FAST, `trajectories (d + 1)` for Morris) and the mapping of the Morris levels. Export `sobol`, `fast` and `morris`. If a sampler of SALib warns (the suite turns warnings into errors), find the cause in the input (e.g. a non power of two), never filter it; SALib may need its `skip_values` default for scrambled Sobol sequences, keep the defaults of SALib apart from the arguments named here.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/simulation/sampling/test_salib_designs.py`
Expected: PASS.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/simulation/sampling tests/simulation/sampling/test_salib_designs.py
git commit -m "The designs of SALib map their unit cube through any distribution" -m "sobol, fast and morris draw the designs of SALib on the unit cube and map them through the inverse CDF of every distribution, so the global sensitivity analyses of phase 2 are not tied to uniform bounds; the levels of Morris are the centres of their strata, so an unbounded marginal stays finite. unit_cube recreates the points from the record, which a result therefore need not store."
```

---

### Task 6: The designs of a fit

**Files:**
- Create: `src/sbmlsim/simulation/sampling/fit.py`
- Modify: `src/sbmlsim/simulation/sampling/__init__.py`
- Test: `tests/simulation/sampling/test_fit_designs.py`

**Interfaces:**
- Consumes: `FisherInformation` (`pids`, `values` in model units, `units`, `covariance` in the scaled space, `to_scale`, `from_scale`, `opid`, `sid`, `alpha`, `parameter_scales`); `IdentifiabilityResult` (`profiles: dict[str, ParameterProfile]`, `parameters: list[FitParameter]`, `fit_settings.parameter_scale`, `settings.alpha`, `cost_min`, `opid`, `parameter_set`); `ParameterProfile` (`values` in model units ascending, `costs`, `converged`, `index_optimum`, `ci_lower`, `ci_upper`); `FitParameter` (`pid`, `scale`, `lower_bound`, `upper_bound`, `unit`); `ParameterScaleType.to_scale`/`from_scale` (vectorized); `OptimizationResult.parameter_sets(size) -> ParameterSets` (`ParameterSet.sid`, `.values`, `.units`, `.cost`).
- Produces: `fit_parameters(fisher, n, *, seed=None, targets=None, id="fit") -> Dimension`; `profile_parameters(identifiability, n, *, seed=None, targets=None, id="profile") -> Dimension`; `fit_repeats(result, size, *, targets=None, id="repeats") -> Dimension`.

- [ ] **Step 1: Write the failing tests**

Create `tests/simulation/sampling/test_fit_designs.py`:

```python
"""The designs of a fit: Fisher, profiles, repeats."""

import numpy as np
import pytest
from scipy import stats

from sbmlsim.fit.fisher import FisherInformation
from sbmlsim.fit.identifiability import IdentifiabilityResult, ParameterProfile, ProfileSettings
from sbmlsim.fit.objects import FitParameter
from sbmlsim.fit.options import FitSettings, ParameterScaleType
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.simulation.sampling import fit_parameters, fit_repeats, profile_parameters


def _fisher(matrix: np.ndarray) -> FisherInformation:
    return FisherInformation(
        opid="op",
        sid="best",
        pids=["k1", "k2"],
        values=np.array([1.0, 10.0]),
        scale=ParameterScaleType.LOG10,
        matrix=matrix,
        cost=1.0,
        n=102,
        units=["1/min", None],
    )


def test_fit_parameters_follow_the_fisher_covariance() -> None:
    fisher = _fisher(np.array([[400.0, 100.0], [100.0, 900.0]]))
    dimension = fit_parameters(fisher, 20000, seed=1)
    logs = np.column_stack(
        [np.log10(np.asarray(dimension.values["k1"].magnitude)), np.log10(dimension.values["k2"])]
    )
    np.testing.assert_allclose(np.cov(logs, rowvar=False), fisher.covariance, rtol=0.05, atol=1e-6)
    np.testing.assert_allclose(np.median(logs, axis=0), [0.0, 1.0], atol=0.01)
    assert str(dimension.values["k1"].units) == "1 / minute"
    assert dimension.design.method == "fit_parameters"
    assert dimension.design.options["opid"] == "op"


def test_a_rank_deficient_covariance_warns_once_and_draws(caplog: pytest.LogCaptureFixture) -> None:
    fisher = _fisher(np.array([[1.0, 1.0], [1.0, 1.0]]))
    dimension = fit_parameters(fisher, 100, seed=1)
    assert np.isfinite(np.asarray(dimension.values["k2"])).all()
    assert sum("rank" in r.message for r in caplog.records) == 1


def test_targets_map_parameters_and_versions_raise() -> None:
    fisher = _fisher(np.eye(2) * 100.0)
    assert set(fit_parameters(fisher, 5, seed=1, targets={"k1": "kcat"}).values) == {"kcat", "k2"}
    with pytest.raises(ValueError, match="target"):
        fit_parameters(fisher, 5, seed=1, targets={"k1": "k", "k2": "k"})


def _profile(values: np.ndarray, costs: np.ndarray, lower: float | None, upper: float | None) -> ParameterProfile:
    return ParameterProfile(
        pid="k1",
        values=values,
        costs=costs,
        paths=values[:, None],
        converged=np.ones(len(values), dtype=bool),
        index_optimum=int(np.argmin(costs)),
        ci_lower=lower,
        ci_upper=upper,
    )


def _identifiability(profile: ParameterProfile, bounds: tuple[float, float]) -> IdentifiabilityResult:
    parameter = FitParameter(pid="k1", start_value=1.0, lower_bound=bounds[0], upper_bound=bounds[1])
    return IdentifiabilityResult(
        opid="op",
        parameter_set=ParameterSet(sid="best", values={"k1": 1.0}, cost=0.0),
        parameters=[parameter],
        settings=ProfileSettings(),
        fit_settings=FitSettings(parameter_scale=ParameterScaleType.LOG10),
        cost=0.0,
        profiles={"k1": profile},
    )


def test_profile_parameters_follow_the_likelihood_of_the_profile() -> None:
    # a quadratic profile in log10 space: the likelihood ratio is a normal of sd 0.1
    x = np.linspace(-0.5, 0.5, 201)
    profile = _profile(10.0**x, 0.5 * (x / 0.1) ** 2, 10.0**-0.2, 10.0**0.2)
    dimension = profile_parameters(_identifiability(profile, (1e-3, 1e3)), 4000, seed=2)
    logs = np.log10(np.asarray(dimension.values["k1"]))
    assert stats.kstest(logs, stats.norm(0.0, 0.1).cdf).pvalue > 0.01
    assert dimension.design.options["alpha"] == pytest.approx(0.95)


def test_a_flat_side_reaches_the_bound() -> None:
    # flat above the optimum: the likelihood stays high up to the upper bound
    x = np.linspace(-0.5, 0.5, 101)
    costs = np.where(x < 0.0, 0.5 * (x / 0.1) ** 2, 0.0)
    profile = _profile(10.0**x, costs, 10.0**-0.2, None)
    dimension = profile_parameters(_identifiability(profile, (1e-3, 1e2)), 4000, seed=3)
    values = np.asarray(dimension.values["k1"])
    assert values.max() > 10.0**0.5 and values.max() <= 1e2


def test_a_profile_without_a_converged_optimum_raises() -> None:
    x = np.linspace(-0.5, 0.5, 11)
    profile = _profile(10.0**x, x**2, None, None)
    profile.converged[profile.index_optimum] = False
    with pytest.raises(ValueError, match="converged"):
        profile_parameters(_identifiability(profile, (1e-3, 1e3)), 10, seed=1)


class _Result:
    """A stand-in of OptimizationResult with three repeats."""

    def parameter_sets(self, size: int = 1) -> list[ParameterSet]:
        sets = [
            ParameterSet(sid=f"run{k}", values={"k1": 1.0 + k, "k2": 2.0 * k}, units={"k1": "1/min", "k2": None}, cost=float(k))
            for k in range(3)
        ]
        return sets[:size]


def test_fit_repeats_take_the_best_sets() -> None:
    dimension = fit_repeats(_Result(), 2)  # ty: ignore[invalid-argument-type]
    assert dimension.labels.tolist() == ["run0", "run1"]
    np.testing.assert_allclose(dimension.values["k1"].magnitude, [1.0, 2.0])
    np.testing.assert_allclose(dimension.values["k2"], [0.0, 2.0])
    assert dimension.design.options["costs"] == [0.0, 1.0]
```

If the constructors of `FisherInformation`, `ParameterProfile`, `IdentifiabilityResult`, `ProfileSettings`, `FitSettings` or `FitParameter` need other arguments than the ones used here, read their definitions and pass what they need; keep the numbers. Replace `_Result` by a real `OptimizationResult` if one is easy to build from the fit tests' fixtures; otherwise keep the stand-in (it has the one method the design calls) and type the parameter of `fit_repeats` with a `Protocol` (`parameter_sets(size) -> Iterable[ParameterSet]`) so no ignore is needed.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/simulation/sampling/test_fit_designs.py`
Expected: FAIL with `ImportError: cannot import name 'fit_parameters'`.

- [ ] **Step 3: Write the implementation**

Create `src/sbmlsim/simulation/sampling/fit.py`:

```python
"""The designs of a fit: draws of what a fit knows about its parameters.

- `fit_parameters`: the multivariate normal of the Fisher covariance around
  the fitted values, in the space of the `parameter_scale` (correlated, a
  local approximation);
- `profile_parameters`: every parameter from its profile likelihood (follows
  asymmetric and open profiles, independent across parameters, since the
  profiles carry no joint information);
- `fit_repeats`: the best parameter sets of the repeats of a fit.

A fitted parameter is written to its `pid` unless `targets` maps it to another
target; two parameters of one target (the versions of a parameter) raise,
since a scan sets a target once per point.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from sbmlsim.fit.fisher import FisherInformation
from sbmlsim.fit.identifiability import IdentifiabilityResult, ParameterProfile
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.simulation.sampling.designs import _count, _seed
from sbmlsim.simulation.scan import Design, Dimension
from sbmlsim.units import ureg

logger = logging.getLogger(__name__)


class _Repeats(Protocol):
    """What `fit_repeats` reads of a result of a fit."""

    def parameter_sets(self, size: int = 1) -> Iterable[ParameterSet]:
        """Get the best parameter sets."""
        ...


def _targets(pids: Sequence[str], targets: Mapping[str, str] | None) -> list[str]:
    """Get the target of every parameter.

    Raises:
        ValueError: if two parameters have one target.
    """
    names = [(targets or {}).get(pid, pid) for pid in pids]
    twice = sorted({t for t in names if names.count(t) > 1})
    if twice:
        raise ValueError(
            f"The fitted parameters set the targets {twice} more than once (the "
            f"versions of a parameter); a scan sets a target once per point, "
            f"map them to different targets with targets=."
        )
    return names


def _column(values: np.ndarray, unit: str | None) -> Any:
    """Give a column its unit, plain floats without one."""
    return ureg.Quantity(values, unit) if unit else values


def fit_parameters(
    fisher: FisherInformation,
    n: int,
    *,
    seed: int | None = None,
    targets: Mapping[str, str] | None = None,
    id: str = "fit",
) -> Dimension:
    """Draw the fitted parameters from the normal of their Fisher covariance.

    The draws are normal around the fitted values in the space of the
    `parameter_scale` with the covariance of the Fisher information, drawn
    with its eigendecomposition (the eigenvalues of a rank deficient
    covariance clipped at zero), and transformed back into the units of the
    model. A covariance which is rank deficient warns once, see
    `FisherInformation.covariance`.

    Args:
        fisher: the Fisher information of the fitted parameters.
        n: the number of points.
        seed: the seed; `None` draws one, which the record keeps.
        targets: pid -> the target of the model, the pid by default.
        id: the id of the dimension.

    Returns:
        The dimension.

    Raises:
        ValueError: if `n` is no positive integer or two parameters have one
            target.
    """
    n = _count(n)
    names = _targets(fisher.pids, targets)
    seed = _seed(seed)
    mean = fisher.to_scale(fisher.values)
    eigenvalues, eigenvectors = np.linalg.eigh(fisher.covariance)
    scale = np.sqrt(np.clip(eigenvalues, 0.0, None))
    rng = np.random.default_rng(seed)
    draws = mean + (rng.standard_normal((n, fisher.k)) * scale) @ eigenvectors.T
    values = np.array([fisher.from_scale(row) for row in draws])
    units = list(fisher.units) or [None] * fisher.k
    record = Design(
        method="fit_parameters",
        options={
            "n": n,
            "seed": seed,
            "opid": fisher.opid,
            "sid": fisher.sid,
            "alpha": fisher.alpha,
            "pids": list(fisher.pids),
            "scales": [str(s) for s in fisher.parameter_scales],
        },
        references={
            target: {"value": float(value), "unit": unit or ""}
            for target, value, unit in zip(names, fisher.values, units, strict=True)
        },
    )
    return Dimension(
        id,
        values={
            target: _column(values[:, k], units[k]) for k, target in enumerate(names)
        },
        design=record,
    )
```

Write `profile_parameters` and `fit_repeats` in the same module:

- `profile_parameters(identifiability, n, *, seed=None, targets=None, id="profile")`: for every profile in the order of `identifiability.profiles`: raise a `ValueError` naming the parameter and "converged" when `profile.converged[profile.index_optimum]` is false; keep the converged points with finite costs; the scale of the parameter is `FitParameter.scale` or `identifiability.fit_settings.parameter_scale`; `x = scale.to_scale(values)`; `density = exp(-(costs - identifiability.cost_min))`; a side whose `ci_lower`/`ci_upper` is `None` and whose parameter bound is finite is extended by one point at `scale.to_scale(bound)` with the density of its outermost point (skip when the bound equals the outermost value); the cumulative trapezoidal integral of the density on `x`, normalized to 1; `x_draws = np.interp(rng.random(n), cdf, x)` (one `rng.random(n)` per parameter, in order, from one generator of `seed`); values `scale.from_scale(x_draws)` with the unit of the `FitParameter` (`unit`, a quantity) or plain floats; the record `Design("profile_parameters", options={"n", "seed", "opid", "alpha": identifiability.settings.alpha, "pids"}, references=the values of identifiability.parameter_set)`.
- `fit_repeats(result: _Repeats, size, *, targets=None, id="repeats")`: `sets = list(result.parameter_sets(size=size))`; the pids of the first set (all sets have the same); a column per pid with the unit of `ParameterSet.units[pid]`; labels the `sid`s; the record `Design("fit_repeats", options={"size": size, "costs": [s.cost for s in sets], "sids": [...]})`; raise when there is no set.

Export the three designs from the package `__init__`. The module imports `sbmlsim.fit`, which imports the simulator; `sbmlsim.simulation.sampling.__init__` importing `fit.py` at the top is fine because `sbmlsim.simulation` itself does not import the sampler; if an import cycle appears all the same (e.g. through `sbmlsim.fit.__init__`), import the fit types under `TYPE_CHECKING` and keep the module free of runtime imports of `sbmlsim.fit`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/simulation/sampling/test_fit_designs.py`
Expected: PASS.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/simulation/sampling tests/simulation/sampling/test_fit_designs.py
git commit -m "Designs draw the parameters of a fit from its Fisher information, its profiles or its repeats" -m "fit_parameters draws the correlated normal of the Fisher covariance in the space of the parameter scale, profile_parameters draws every parameter from the likelihood ratio of its profile, which follows asymmetric intervals and spreads a parameter that is not identifiable up to its bound, and fit_repeats takes the best parameter sets of the repeats of a fit; versions of one target raise."
```

---

### Task 7: Virtual populations

**Files:**
- Create: `src/sbmlsim/simulation/sampling/population.py`
- Modify: `src/sbmlsim/simulation/sampling/__init__.py`; `tests/simulator/models.py` (a population function)
- Test: `tests/simulation/sampling/test_population.py`

**Interfaces:**
- Consumes: `random`, `lhs` (Task 4); `sbmlsim.simulation.observables._check_function(function, name)`; `Dimension(coordinates=)` (Task 1).
- Produces: `population(function: Callable[[dict[str, Any]], Mapping[str, Any]], covariates: Mapping[str, Distribution], n: int, *, seed: int | None = None, method: str = "random", model: ModelLike | None = None, simulation: Simulation | None = None, id: str = "population") -> Dimension`; in `tests/simulator/models.py` the module function `clearance_of(covariates: dict[str, Any]) -> dict[str, Any]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/simulator/models.py`:

```python
def clearance_of(covariates: dict[str, Any]) -> dict[str, Any]:
    """A population function: k1 scales with the body weight to the power 0.75."""
    bw = np.asarray(getattr(covariates["BW"], "magnitude", covariates["BW"]), dtype=float)
    return {"k1": 0.8 * (bw / 70.0) ** 0.75}


def no_mapping(covariates: dict[str, Any]) -> Any:
    """A wrong population function, which returns a list."""
    return [1.0]


def wrong_length(covariates: dict[str, Any]) -> dict[str, Any]:
    """A wrong population function, which returns one value too many."""
    return {"k1": np.ones(len(covariates["BW"]) + 1)}


def covariate_as_target(covariates: dict[str, Any]) -> dict[str, Any]:
    """A wrong population function, which returns a covariate."""
    return {"BW": covariates["BW"]}
```

Create `tests/simulation/sampling/test_population.py`:

```python
"""Virtual populations."""

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.simulation import Formula, Scan, Simulation
from sbmlsim.simulation.sampling import Normal, Truncated, population
from sbmlsim.simulator import Simulator
from tests.simulator.models import (
    clearance_of,
    covariate_as_target,
    no_mapping,
    sbml,
    wrong_length,
)

COVARIATES = {"BW": Truncated(Normal(Q(70.0, "kg"), Q(15.0, "kg")), lower=Q(40.0, "kg"))}


def test_a_population_maps_its_covariates_to_targets() -> None:
    dimension = population(clearance_of, COVARIATES, 200, seed=1)
    bw = np.asarray(dimension.coordinates["BW"].magnitude)
    assert bw.min() >= 40.0
    np.testing.assert_allclose(dimension.values["k1"], 0.8 * (bw / 70.0) ** 0.75)
    assert dimension.design.method == "population"
    assert dimension.design.options["function"] == "tests.simulator.models:clearance_of"
    assert dimension.design.options["covariates"] == ["BW"]
    lhs = population(clearance_of, COVARIATES, 50, seed=1, method="lhs")
    assert lhs.design.options["method"] == "lhs"


def test_the_result_has_the_covariates_as_coordinates() -> None:
    dimension = population(clearance_of, COVARIATES, 5, seed=2)
    res = Simulator().run(sbml(), Scan(Simulation(end=1, steps=2), [dimension]), [Formula("k", "k1")])
    assert res.ds["BW"].dims == ("population",)
    assert res.units["BW"] == "kilogram"


def test_a_wrong_population_function_raises() -> None:
    with pytest.raises(ValueError, match="mapping"):
        population(no_mapping, COVARIATES, 5, seed=1)
    with pytest.raises(ValueError, match="length"):
        population(wrong_length, COVARIATES, 5, seed=1)
    with pytest.raises(ValueError, match="covariate"):
        population(covariate_as_target, COVARIATES, 5, seed=1)
    with pytest.raises(ValueError, match="module"):
        population(lambda c: {"k1": c["BW"]}, COVARIATES, 5, seed=1)
    with pytest.raises(ValueError, match="method"):
        population(clearance_of, COVARIATES, 5, seed=1, method="sobol")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/simulation/sampling/test_population.py`
Expected: FAIL with `ImportError: cannot import name 'population'`.

- [ ] **Step 3: Write the implementation**

Create `src/sbmlsim/simulation/sampling/population.py`:

```python
"""Virtual populations: covariates drawn and mapped to the values of targets.

The covariates (e.g. the body weight, the age) are drawn by `random` or
`lhs`, and a function of a module maps them to the values of targets of the
model, `function(covariates) -> {target: values}`, vectorized over the points.
The covariates are coordinates of the dimension, which the result carries and
no model sees, so an observable can be plotted against them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from sbmlsim.simulation.definition import Simulation
from sbmlsim.simulation.observables import _check_function
from sbmlsim.simulation.sampling.designs import lhs, random
from sbmlsim.simulation.sampling.distributions import Distribution
from sbmlsim.simulation.scan import Design, Dimension
from sbmlsim.simulator.simulator import ModelLike

#: the designs which draw the covariates of a population
METHODS = {"random": random, "lhs": lhs}


def population(
    function: Callable[[dict[str, Any]], Mapping[str, Any]],
    covariates: Mapping[str, Distribution],
    n: int,
    *,
    seed: int | None = None,
    method: str = "random",
    model: ModelLike | None = None,
    simulation: Simulation | None = None,
    id: str = "population",
) -> Dimension:
    """Get a virtual population, see the module.

    Args:
        function: a function of a module, `covariates -> {target: values}`, the
            values of every target for every point.
        covariates: covariate -> its distribution.
        n: the number of individuals.
        seed: the seed; `None` draws one, which the record keeps.
        method: `random` or `lhs`, the design of the covariates.
        model: the model, which a relative distribution of a covariate needs.
        simulation: the simulation whose pre-initialization gives the references.
        id: the id of the dimension.

    Returns:
        The dimension, the targets as its values and the covariates as its
        coordinates.

    Raises:
        ValueError: if the function is no function of a module or returns no
            mapping of targets to `n` values, a covariate as a target, or the
            method is unknown.
    """
    _check_function(function, id)
    if method not in METHODS:
        raise ValueError(f"The method of a population is one of {sorted(METHODS)}, not '{method}'.")
    drawn = METHODS[method](covariates, n, seed=seed, model=model, simulation=simulation, id=id)
    values = dict(drawn.values)
    targets = function(values)
    if not isinstance(targets, Mapping):
        raise ValueError(
            f"The function of the population '{id}' returns a mapping of targets to "
            f"values, not {targets!r}."
        )
    for target, column in targets.items():
        if target in values:
            raise ValueError(
                f"The function of the population '{id}' returns the covariate "
                f"'{target}' as a target; a covariate is never set on a model."
            )
        length = len(np.atleast_1d(getattr(column, "magnitude", column)))
        if length != len(drawn):
            raise ValueError(
                f"The function of the population '{id}' returns {length} values of "
                f"'{target}' for {len(drawn)} individuals; the values must have the "
                f"length of the population."
            )
    record = drawn.design
    if record is None:
        raise ValueError(f"The covariates of the population '{id}' have no record.")
    function_name = f"{getattr(function, '__module__', '')}:{getattr(function, '__qualname__', '')}"
    design = Design(
        method="population",
        distributions=record.distributions,
        options={
            **record.options,
            "method": method,
            "function": function_name,
            "covariates": list(covariates),
        },
        references=record.references,
    )
    return Dimension(id, values=dict(targets), coordinates=values, design=design)
```

If `_check_function` is private to `simulation/observables.py` by its name, it is still the one check of a function of a module in the package; keep the import (it is within the package) rather than copying it. Export `population` from the package `__init__`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/simulation/sampling/test_population.py`
Expected: PASS.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/simulation/sampling tests/simulation/sampling/test_population.py tests/simulator/models.py
git commit -m "A virtual population maps drawn covariates to the values of targets" -m "population draws covariates with random or lhs and maps them with a function of a module to the values of targets of the model; the covariates are coordinates of the dimension, which the result carries and no model sees, and the record names the function and the design of the covariates."
```

---

### Task 8: The start values of a fit on the sampler

**Files:**
- Modify: `src/sbmlsim/fit/sampling.py` (`create_samples`)
- Test: `tests/fit/test_sampling.py` (a test of unchanged values)

**Interfaces:**
- Consumes: `random`, `lhs` (Task 4); `Uniform`, `LogUniform`, `Fixed` (Task 2).
- Produces: `create_samples(parameters, size, sampling=LOGUNIFORM, seed=None, min_bound=1e-10) -> pd.DataFrame` with the values of before for every seed.

- [ ] **Step 1: Write the failing test**

Append to `tests/fit/test_sampling.py` a reference implementation of the old sampling (the code of `create_samples` at `344e43ee`, which this task replaces) and the comparison:

```python
def _reference_samples(
    parameters: list[FitParameter], size: int, sampling: SamplingType, seed: int
) -> np.ndarray:
    """The start values of fit/sampling.py before the sampler (344e43ee)."""
    from scipy.stats import qmc

    rng = np.random.default_rng(seed)
    if sampling.is_lhs:
        x = qmc.LatinHypercube(d=len(parameters), rng=rng).random(n=size)
    else:
        x = rng.random(size=(size, len(parameters)))
    for k, p in enumerate(parameters):
        if np.isinf(p.lower_bound) or np.isinf(p.upper_bound):
            x[:, k] = p.start_value
            continue
        is_log = sampling.is_log and p.scale is not ParameterScaleType.LINEAR
        lb, ub = float(p.lower_bound), float(p.upper_bound)
        if is_log and lb <= 0.0:
            lb = 1e-10
        if is_log:
            x[:, k] = np.power(10, np.log10(lb) + x[:, k] * (np.log10(ub) - np.log10(lb)))
        else:
            x[:, k] = lb + x[:, k] * (ub - lb)
    return x


@pytest.mark.parametrize(
    "sampling",
    [SamplingType.LOGUNIFORM, SamplingType.UNIFORM, SamplingType.LOGUNIFORM_LHS, SamplingType.UNIFORM_LHS],
)
def test_the_start_values_did_not_change(sampling: SamplingType) -> None:
    parameters = [
        FitParameter(pid="a", start_value=1.0, lower_bound=1e-3, upper_bound=1e3),
        FitParameter(pid="b", start_value=0.5, lower_bound=0.0, upper_bound=2.0),
        FitParameter(pid="c", start_value=3.0, lower_bound=-np.inf, upper_bound=np.inf),
        FitParameter(pid="d", start_value=0.0, lower_bound=-1.0, upper_bound=1.0, scale="LINEAR"),
    ]
    with warnings.catch_warnings():
        # the non-positive lower bound of b is replaced for the logarithmic samplings
        warnings.simplefilter("ignore")
        df = create_samples(parameters, size=7, sampling=sampling, seed=11)
    np.testing.assert_array_equal(df.values, _reference_samples(parameters, 7, sampling, 11))
    assert list(df.columns) == ["a", "b", "c", "d"]
```

The replacement of a non-positive lower bound logs a warning through `logging` (not `warnings`); drop the `warnings` block if nothing warns. Add the imports the test needs (`warnings`, `numpy as np`, `pytest`, `FitParameter`, `ParameterScaleType`, `SamplingType`, `create_samples`) if the module does not have them. If `FitParameter(scale="LINEAR")` takes another spelling, use the one `FitParameter` documents. Run it now: it passes on the old code (this is the guard of the refactoring) - record that in the report; it must still pass after Step 3.

- [ ] **Step 2: Run the test on the old code**

Run: `uv run pytest -q -n 0 tests/fit/test_sampling.py`
Expected: PASS (the reference is the old code); the guard is in place before the change.

- [ ] **Step 3: Write the implementation**

In `src/sbmlsim/fit/sampling.py`, keep `SamplingType`, `_start_samples` and `_sampling_bounds`; replace the body of `create_samples` after the `START` branch with:

```python
    distributions: dict[str, Distribution] = {}
    for p in parameters:
        if np.isinf(p.lower_bound) or np.isinf(p.upper_bound):
            if p.start_value is None:
                raise ValueError(
                    f"'{p.pid}': a parameter with the infinite bounds "
                    f"[{p.lower_bound} - {p.upper_bound}] is not sampled and "
                    f"requires a 'start_value'."
                )
            # the column is drawn and replaced, so the others keep their draws
            distributions[p.pid] = Fixed(float(p.start_value))
            continue
        is_log = sampling.is_log and p.scale is not ParameterScaleType.LINEAR
        lb, ub = _sampling_bounds(parameter=p, is_log=is_log, min_bound=min_bound)
        distributions[p.pid] = LogUniform(lb, ub) if is_log else Uniform(lb, ub)
    if sampling.is_lhs:
        design = lhs(distributions, size, seed=seed)
    elif sampling in {SamplingType.UNIFORM, SamplingType.LOGUNIFORM}:
        design = random(distributions, size, seed=seed)
    else:
        raise ValueError(f"Unsupported SamplingType: '{sampling}'")
    return pd.DataFrame(
        {p.pid: np.asarray(design.values[p.pid], dtype=float) for p in parameters}
    )
```

and import `Fixed`, `LogUniform`, `Uniform`, `Distribution`, `lhs` and `random` from `sbmlsim.simulation.sampling` (drop the `qmc` import). Update the module docstring: "Sampling of the start values of a fit, on the designs of `sbmlsim.simulation.sampling`; the values for a seed are the ones of the sampling before it." `Uniform(lb, ub)` and `LogUniform(lb, ub)` must accept `lb == ub` (Task 2 allows `lower <= upper`); if `LogUniform` refused equal bounds, allow them there (a degenerate interval is a valid fit bound).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/fit/test_sampling.py tests/fit/test_fit.py`
Expected: PASS, the start values equal to the reference for every sampling type.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/fit/sampling.py tests/fit/test_sampling.py
git commit -m "The start values of a fit come from the designs of the sampler" -m "create_samples maps every FitParameter to a LogUniform, a Uniform or a Fixed and draws them with random or lhs, so the fit and the analyses share one sampler. The start values for a seed and a sampling type are the ones of before, which a test against the former implementation guards."
```

---

### Task 9: The plots of the uncertainty analysis

**Files:**
- Create: `src/sbmlsim/sensitivity/uncertainty.py`
- Test: `tests/sensitivity/test_uncertainty.py`

**Interfaces:**
- Consumes: `ScanResult.summary(dims, statistics, quantiles)` (the quantile `q` is the statistic `q<q>`, e.g. `q0.05`; a timecourse keeps its `time` dimension on a grid), `ScanResult.__getitem__`, `.units`, `.dims`; `random` with `LogNormal` (Task 4).
- Produces: `plot_bands(summary: ScanResult, key: str, *, lower: str = "q0.05", center: str = "q0.5", upper: str = "q0.95", ax: Axes | None = None, alpha: float = 0.3) -> Figure`; `plot_distribution(result: ScanResult, key: str, *, dim: str, kind: str = "hist", ax: Axes | None = None, bins: int = 30) -> Figure`.

- [ ] **Step 1: Write the failing tests**

Create `tests/sensitivity/test_uncertainty.py`:

```python
"""The uncertainty analysis: bands and distributions."""

import numpy as np
import pytest
from matplotlib.figure import Figure
from scipy import stats

from sbmlsim.simulation import Dimension, Formula, Scan, Simulation
from sbmlsim.simulation.sampling import LogNormal, random
from sbmlsim.sensitivity.uncertainty import plot_bands, plot_distribution
from sbmlsim.simulator import Simulator
from tests.simulator.models import sbml


@pytest.fixture(scope="module")
def result() -> object:
    draws = random({"k1": LogNormal(0.8, 0.3)}, 2000, seed=1)
    doses = Dimension("a", values={"a0": [1.0, 2.0]})
    observables = [Formula("rate", "k1 * a0 + 0 * time"), Formula("k", "max(k1)")]
    scan = Scan(Simulation(end=1, steps=4), [draws, doses])
    return Simulator(n_workers=1).run(sbml(), scan, observables)


def test_the_bands_of_a_lognormal_parameter_are_its_quantiles(result: object) -> None:
    summary = result.summary("random", quantiles=[0.05, 0.5, 0.95])  # ty: ignore[unresolved-attribute]
    sigma = np.sqrt(np.log(1.0 + 0.3**2))
    expected = stats.lognorm(sigma, scale=0.8).ppf([0.05, 0.5, 0.95])
    # the first label of the dimension a is a0 = 1, so the rate is k1
    band = summary["rate"].isel(a=0, time=0)
    np.testing.assert_allclose(band.sel(statistic=["q0.05", "q0.5", "q0.95"]).values, expected, rtol=0.05)


def test_plot_bands_draws_a_band_per_label(result: object) -> None:
    summary = result.summary("random", quantiles=[0.05, 0.5, 0.95])  # ty: ignore[unresolved-attribute]
    figure = plot_bands(summary, "rate")
    assert isinstance(figure, Figure)
    (ax,) = figure.axes
    assert len(ax.collections) == 2 and len(ax.lines) == 2
    assert ax.get_xlabel().startswith("time")


def test_plot_distribution_per_label(result: object) -> None:
    figure = plot_distribution(result, "k", dim="random")  # ty: ignore[invalid-argument-type]
    assert isinstance(figure, Figure)
    assert len(figure.axes[0].patches) > 0
    box = plot_distribution(result, "k", dim="random", kind="box")  # ty: ignore[invalid-argument-type]
    assert isinstance(box, Figure)
    with pytest.raises(ValueError, match="kind"):
        plot_distribution(result, "k", dim="random", kind="violin")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="time"):
        plot_distribution(result, "rate", dim="random")  # ty: ignore[invalid-argument-type]
```

Type the fixture as `ScanResult` so the ignores are not needed. `Formula("rate", "k1 * a0 + 0 * time")` is a timecourse (it reads `time`) of a constant, `Formula("k", "max(k1)")` a value per simulation; the probe model has no units, so both are dimensionless. The run is serial (`n_workers=1`) since 4000 points would start a pool.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/sensitivity/test_uncertainty.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbmlsim.sensitivity.uncertainty'`.

- [ ] **Step 3: Write the implementation**

Create `src/sbmlsim/sensitivity/uncertainty.py`:

```python
"""The uncertainty analysis: bands of timecourses and distributions of values.

The uncertainty of a prediction is a scan over draws of the uncertain
parameters, see `sbmlsim.simulation.sampling` (`random`, `lhs`,
`fit_parameters`, `profile_parameters`, `fit_repeats`, `population`), run by
`Simulator.run`, and its summary over the dimension of the draws,
`ScanResult.summary(dim, quantiles=[0.05, 0.5, 0.95])`. The functions here
draw it: `plot_bands` a timecourse as a median and a band per label of the
other dimensions, `plot_distribution` a value per simulation as a histogram
or a box per label. They return the figure and never show it.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sbmlsim.result.scan import STATISTIC, TIME, ScanResult


def _figure(ax: Axes | None) -> tuple[Figure, Axes]:
    """Get the figure and the axes to draw into."""
    if ax is not None:
        figure = ax.get_figure()
        if not isinstance(figure, Figure):
            raise ValueError("The axes of a plot must belong to a figure.")
        return figure, ax
    figure = Figure(figsize=(6, 4), layout="constrained")
    return figure, figure.add_subplot()


def _labels(data: Any, skip: set[str]) -> list[dict[str, Any]]:
    """Get every combination of the labels of the other dimensions."""
    dims = [str(d) for d in data.dims if str(d) not in skip]
    return [
        dict(zip(dims, values, strict=True))
        for values in itertools.product(*(data[d].values.tolist() for d in dims))
    ]


def _text(selection: dict[str, Any]) -> str:
    """Get the label of a curve."""
    return ", ".join(f"{d}={v}" for d, v in selection.items())


def plot_bands(
    summary: ScanResult,
    key: str,
    *,
    lower: str = "q0.05",
    center: str = "q0.5",
    upper: str = "q0.95",
    ax: Axes | None = None,
    alpha: float = 0.3,
) -> Figure:
    """Draw a timecourse of a summary as a center line and a band per label.

    Args:
        summary: a summary of a scan, `ScanResult.summary(..., quantiles=...)`.
        key: the timecourse.
        lower: the statistic of the lower bound of the band.
        center: the statistic of the line.
        upper: the statistic of the upper bound of the band.
        ax: the axes to draw into, a new figure without.
        alpha: the opacity of the band.

    Returns:
        The figure.

    Raises:
        ValueError: if the variable is no timecourse of a summary or a
            statistic is missing.
    """
    data = summary[key]
    if STATISTIC not in data.dims or TIME not in data.dims:
        raise ValueError(f"'{key}' is no timecourse of a summary; summarize a scan first.")
    missing = [s for s in (lower, center, upper) if s not in data[STATISTIC].values.tolist()]
    if missing:
        raise ValueError(f"The summary has not the statistics {missing}, add their quantiles.")
    figure, axes = _figure(ax)
    time = data[TIME].values
    for selection in _labels(data, {STATISTIC, TIME}):
        curve = data.sel(selection)
        lines = axes.plot(time, curve.sel({STATISTIC: center}).values, label=_text(selection) or key)
        axes.fill_between(
            time,
            curve.sel({STATISTIC: lower}).values,
            curve.sel({STATISTIC: upper}).values,
            color=lines[0].get_color(),
            alpha=alpha,
            linewidth=0,
        )
    units = summary.units
    axes.set_xlabel(f"time [{units.get(TIME, '')}]")
    axes.set_ylabel(f"{key} [{units.get(key, '')}]")
    if len(axes.lines) > 1:
        axes.legend()
    return figure


def plot_distribution(
    result: ScanResult,
    key: str,
    *,
    dim: str,
    kind: str = "hist",
    ax: Axes | None = None,
    bins: int = 30,
) -> Figure:
    """Draw a value per simulation over the draws as a histogram or a box per label.

    Args:
        result: the result of a scan over draws.
        key: a value per simulation, e.g. the parameter of a PK observable.
        dim: the dimension of the draws.
        kind: `hist` or `box`.
        ax: the axes to draw into, a new figure without.
        bins: the bins of a histogram.

    Returns:
        The figure.

    Raises:
        ValueError: if the variable is a timecourse, the dimension is not one
            of it, or the kind is unknown.
    """
    if kind not in ("hist", "box"):
        raise ValueError(f"The kind of a distribution plot is 'hist' or 'box', not '{kind}'.")
    data = result[key]
    if TIME in data.dims or "_point" in data.dims:
        raise ValueError(f"'{key}' is a timecourse; draw a value per simulation, e.g. at(x, t).")
    if dim not in data.dims:
        raise ValueError(f"'{dim}' is no dimension of '{key}': {list(data.dims)}.")
    figure, axes = _figure(ax)
    selections = _labels(data, {dim})
    samples = [np.asarray(data.sel(s).values, dtype=float) for s in selections]
    samples = [s[np.isfinite(s)] for s in samples]
    names = [_text(s) or key for s in selections]
    if kind == "hist":
        for values, name in zip(samples, names, strict=True):
            axes.hist(values, bins=bins, alpha=0.5, label=name)
        axes.set_xlabel(f"{key} [{result.units.get(key, '')}]")
        axes.set_ylabel("count")
        if len(samples) > 1:
            axes.legend()
    else:
        axes.boxplot(samples, tick_labels=names)
        axes.set_ylabel(f"{key} [{result.units.get(key, '')}]")
    return figure
```

The package `sbmlsim.sensitivity` keeps its old analyses until phase 2; its `__init__` does not need to export the new module (it is imported by its path). If matplotlib's `boxplot` of the installed version takes `labels=` instead of `tick_labels=`, use the one it takes without a deprecation warning.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/sensitivity/test_uncertainty.py`
Expected: PASS.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/sensitivity/uncertainty.py tests/sensitivity/test_uncertainty.py
git commit -m "The uncertainty of a scan is drawn as bands of timecourses and distributions of values" -m "plot_bands draws the median and a quantile band of a summarized timecourse for every label of the other dimensions, and plot_distribution a value per simulation over the draws as a histogram or a box; the uncertainty analysis itself is a design of draws, a run and ScanResult.summary."
```

---

### Task 10: ModelSensitivity goes, examples and docs

**Files:**
- Delete: `src/sbmlsim/simulation/sensitivity.py`, `tests/test_sensitivity.py`, `docs/api/simulation.sensitivity.md`
- Modify: `examples/model_sensitivity.py`, `examples/demo/demo.py:19,51`, `docs/scans.md:160-181`, `zensical.toml` (nav), `examples/README.md`, `tests/docs/test_docs_code.py:17` (`PAGES`), `CLAUDE.md`, `docs/api/index.md` (if it lists the modules), `docs/index.md` and `README.md` (if they name `ModelSensitivity`)
- Create: `docs/sampling.md`, `docs/api/simulation.sampling.md`, `docs/api/sensitivity.uncertainty.md`

**Interfaces:**
- Consumes: everything of Tasks 1-9.

- [ ] **Step 1: Find every user of ModelSensitivity**

Run: `rg -n "ModelSensitivity|simulation.sensitivity|simulation/sensitivity" --glob '!docs/superpowers/**' --glob '!release-notes/**'`
Expected: `examples/model_sensitivity.py`, `examples/demo/demo.py`, `docs/scans.md`, `tests/test_sensitivity.py`, `docs/api/simulation.sensitivity.md`, `zensical.toml`, `CLAUDE.md` and possibly `docs/index.md`/`README.md`. Every one is migrated below; the release notes stay as they are (history).

- [ ] **Step 2: Migrate the examples**

In `examples/model_sensitivity.py`, replace the import of `ModelSensitivity` with `from sbmlsim.simulation import Scan, Simulation, sampling` and the body of `run_sensitivity` with:

```python
def run_sensitivity() -> None:
    """Parameter sensitivity simulations: a local design and lognormal draws."""
    simulator = Simulator()
    model = simulator.load(REPRESSILATOR_SBML)
    model.set_selections(["time", "[X]", "[Y]", "[Z]"])
    tcsim = Simulation(end=200, steps=2000)
    parameters = sampling.parameters_of(model)

    # the parameters drawn from lognormal distributions around their references
    draws = sampling.random(
        {pid: sampling.LogNormal(cv=0.03) for pid in parameters},
        50,
        seed=1234,
        model=model,
    )
    res_distrib_scan = simulator.run(model, Scan(tcsim, [draws]))

    # every parameter alone 10 % up and down
    local = sampling.local(parameters, delta=0.1, model=model)
    res_diff_scan = simulator.run(model, Scan(tcsim, [local]))
```

and keep the rest (the figures). In `examples/demo/demo.py`, replace `ModelSensitivity.create_difference_dimension(model, difference=0.5)` with `sampling.local(sampling.parameters_of(model), delta=0.5, model=model, id="dim_sens")` and the import accordingly (the local design adds the reference point; the demo plots all points, which is fine). Run both examples (`uv run python -m examples.model_sensitivity`, `uv run python -m examples.demo.demo` from a scratch directory) and the example tests.

- [ ] **Step 3: The docs**

Replace the section "Sensitivity scans" of `docs/scans.md` with:

````markdown
## Designs

A design is a dimension whose values follow a design: the sampler `sbmlsim.simulation.sampling` creates the local design of all parameters, random draws, Latin hypercubes, the designs of the global sensitivity analyses, draws of the parameters of a fit and virtual populations, see [Sampling and uncertainty](sampling.md):

```python
from sbmlsim.simulation import sampling

simulation = Simulation(end=100, steps=100)
parameters = sampling.parameters_of(model)
local = sampling.local(parameters, delta=0.1, model=model)
res = simulator.run(model, Scan(simulation, [local]))
print(res["PX"].sizes)

draws = sampling.lhs({pid: sampling.LogNormal(cv=0.05) for pid in parameters}, 10, seed=1, model=model)
res = simulator.run(model, Scan(simulation, [draws]))
print(res["PX"].sizes)
```

The local design varies every constant parameter alone up and down by the relative `delta` around its reference, the value the model gives it after the pre-initialization of the simulation; the Latin hypercube draws 10 points of lognormal distributions around the references. The record of a design is part of the provenance of the result.
````

Create `docs/sampling.md` (every `python` block runs in the docs test; add `"sampling.md"` to `PAGES`):

````markdown
# Sampling and uncertainty

The sampler `sbmlsim.simulation.sampling` creates designs: dimensions of a scan whose values follow a design. Every design maps points of the unit cube through the inverse cumulative distribution function of a distribution per target, so any marginal works with every design, and it records how it was drawn (the method, the distributions, the options, the seed and the references), which the result keeps as its provenance.

## Distributions

```python
import numpy as np

from sbmlsim import Q
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Formula, Scan, Simulation, sampling
from sbmlsim.simulation.sampling import LogNormal, Normal, Truncated, Uniform
from sbmlsim.simulator import Simulator

simulator = Simulator()
model = simulator.load(REPRESSILATOR_SBML)
print(Uniform(1.0, 2.0).ppf(np.array([0.0, 0.5, 1.0])))
print(Truncated(Normal(Q(75, "kg"), Q(12, "kg")), lower=Q(40, "kg")).ppf(np.array([0.01, 0.5])))
```

| distribution | values |
| --- | --- |
| `Uniform(lower, upper)`, `Uniform(relative=r)` | uniform in the bounds, or in the reference times `[1 - r, 1 + r]` |
| `LogUniform(lower, upper)`, `LogUniform(factor=f)` | uniform in log10, or between the reference divided and multiplied by `f` |
| `Normal(mean, sd)`, `Normal(cv=c)` | normal, around the reference with `sd = c * |reference|` |
| `LogNormal(median, cv)`, `LogNormal(cv=c)` | lognormal, around the reference |
| `Truncated(distribution, lower, upper)` | the distribution restricted to the interval |
| `Empirical(values)` | the values with equal weights |
| `Fixed(value)` | one value for every point |

A distribution without a location is relative to the reference of its target: the value the model gives the target after the pre-initialization of the simulation, i.e. the changes of the model and of the simulation and the initial assignments; the design then needs the model, `model=`. Numbers are in the unit of the target in the model, quantities are converted by the run.

## Designs

```python
parameters = sampling.parameters_of(model)
simulation = Simulation(end=200, steps=200)

local = sampling.local(parameters, delta=0.1, model=model)
draws = sampling.lhs({pid: LogNormal(cv=0.1) for pid in parameters}, 50, seed=1, model=model)
print(local.labels[:3], len(draws), draws.design.method, draws.design.options["seed"])
```

| design | points |
| --- | --- |
| `local(targets, delta, model=...)` | the reference and every target alone at `1 + delta` and `1 - delta` times its reference |
| `random(distributions, n, seed=..., correlation=...)` | independent draws, or correlated ones (a Gaussian copula) |
| `lhs(distributions, n, seed=..., correlation=...)` | a Latin hypercube; with a correlation the rank reordering of Iman and Conover keeps its strata |
| `sobol(distributions, n, seed=...)`, `fast(...)`, `morris(...)` | the designs of SALib for the global sensitivity analyses |
| `fit_parameters(fisher, n, seed=...)` | the parameters of a fit from the normal of its Fisher covariance |
| `profile_parameters(identifiability, n, seed=...)` | every parameter of a fit from its profile likelihood, which follows asymmetric and open confidence intervals |
| `fit_repeats(result, size)` | the best parameter sets of the repeats of a fit |
| `population(function, covariates, n, seed=...)` | covariates drawn and mapped to the values of targets by a function of a module; the covariates are coordinates of the result |

A design is a dimension like any other, so it combines with the other dimensions of a scan, e.g. doses or conditions. `seed=None` draws a seed and records it, so a result can always be drawn again.

## Uncertainty

The uncertainty of a prediction is a scan over draws of the uncertain parameters and its summary over their dimension:

```python
from sbmlsim.sensitivity.uncertainty import plot_bands, plot_distribution

res = simulator.run(model, Scan(simulation, [draws]), [Formula("px", "PX"), Formula("px_max", "max(PX)")])
bands = res.summary("lhs", quantiles=[0.05, 0.5, 0.95])
figure = plot_bands(bands, "px")
figure.savefig("bands.png")
plot_distribution(res, "px_max", dim="lhs").savefig("px_max.png")
```

`plot_bands` draws the median and the band between the 5 % and the 95 % quantile of every label of the other dimensions, `plot_distribution` the distribution of a value per simulation as a histogram or a box. The draws of the parameters of a fit, `fit_parameters` and `profile_parameters`, give the uncertainty of the predictions of a fitted model.
````

Navigation and API pages: `{ "Sampling and uncertainty" = "sampling.md" },` after `{ "Parameter scans" = "scans.md" },` in `zensical.toml`; `docs/api/simulation.sampling.md` with `::: sbmlsim.simulation.sampling` followed by `::: sbmlsim.simulation.sampling.distributions`, `::: sbmlsim.simulation.sampling.references`, `::: sbmlsim.simulation.sampling.designs`, `::: sbmlsim.simulation.sampling.fit`, `::: sbmlsim.simulation.sampling.population` (one per line, a blank line between); `docs/api/sensitivity.uncertainty.md` with `::: sbmlsim.sensitivity.uncertainty`; the nav entries `{ "sampling" = "api/simulation.sampling.md" },` in the simulation group (instead of the removed `sensitivity` entry) and `{ "uncertainty" = "api/sensitivity.uncertainty.md" },` in the sensitivity group. Delete `docs/api/simulation.sensitivity.md` and its nav entry. `examples/README.md`: the row of `examples/model_sensitivity.py` says "a local design and lognormal draws of all parameters of the repressilator, with the uncertainty bands". `docs/sensitivity.md` keeps describing the old analyses until phase 2; add one sentence at its top: "The designs of these analyses come from `sbmlsim.simulation.sampling` from the next release on, see [Sampling and uncertainty](sampling.md)." only if the docs build or a reader would otherwise find a contradiction; otherwise leave it for phase 2.

- [ ] **Step 4: CLAUDE.md**

In the paragraph of the simulation core, replace "`simulation/sensitivity.py` builds the `Scan`s of `ModelSensitivity`." with: "`simulation/sampling/` is the sampler: distributions (`Uniform`, `LogUniform`, `Normal`, `LogNormal`, `Truncated`, `Empirical`, `Fixed`, each with `ppf(u, reference)`; a distribution without a location is relative to the reference of its target, `references(model, targets, simulation)`, the value after the pre-initialization) and designs which return a `Dimension` with a `Design` record (`local`, `random`, `lhs` with correlations by a Gaussian copula and by Iman and Conover, `sobol`, `fast`, `morris` on the unit cube of SALib, `fit_parameters` from the Fisher covariance, `profile_parameters` from the profile likelihoods, `fit_repeats`, `population` with covariates as `coordinates` of the dimension); the record is in the provenance of the result (`Scan.to_dict`), so an analysis needs only the result, and `seed=None` draws a recorded seed. `fit/sampling.py` draws the start values of a fit with it (the same values for a seed). `sensitivity/uncertainty.py` draws the bands and distributions of a scan over draws." In the paragraph of `sensitivity/`, add: "Its designs move to `sbmlsim.simulation.sampling`; the analyses are rebuilt on `Simulator.run` in phase 2 of the analyses design." In the list of dependencies nothing changes.

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest -q -n 0 tests/docs tests/examples/test_example_scripts.py`, `uv run zensical build --clean` (no warnings), `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add -A src/sbmlsim/simulation tests docs examples zensical.toml CLAUDE.md README.md
git commit -m "ModelSensitivity gives way to the sampler, which the docs and examples describe" -m "The difference and distribution scans of ModelSensitivity are the local design and random draws of relative distributions now, with a seed and lognormal draws which stay positive. docs/sampling.md explains the distributions, the designs and the uncertainty analysis with code which the docs test runs; the examples, the navigation, the API reference and CLAUDE.md follow."
```

(Check `git status` before `git add -A`: only the files of this task.)

---

### Task 11: Verification and the pull request

**Files:** none new; the pull request.

- [ ] **Step 1: The whole suite and the checks**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`, `uv run zensical build --clean`, and the pool under `spawn` for a scan with a design: `uv run python -c "import multiprocessing as m, sys; m.set_start_method('spawn'); import pytest; sys.exit(pytest.main(['-q', '-n', '0', 'tests/simulation/sampling', 'tests/simulator/test_simulator_pool.py']))"`.
Expected: all pass, no warnings.

- [ ] **Step 2: The pull request**

Push the branch and create the pull request with `gh-axi` (never `gh`), base `develop`, title "The sampler: designs of a scan and the uncertainty analysis (#249, analyses phase 1)". The body describes, without any agent attribution: the distributions and the references, the designs with their records, the designs of a fit (Fisher, profiles, repeats) and populations, the start values of a fit unchanged for a seed, the uncertainty plots, the removal of `ModelSensitivity`, the decisions of this plan where the spec is silent, and that the sensitivity analyses on a result (phase 2) follow. Wait for the checks `tests`, `ruff`, `ty` and `docs`.
