# The analyses, phase 2: sensitivity on a result Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The sensitivity analyses are functions of a `ScanResult`: `sensitivity.local`, `sobol`, `fast` and `morris` read the design record of a dimension and compute their indices for every observable, every label of the other dimensions and every time point, into a `SensitivityResult` with plots; the old analysis classes of `sbmlsim.sensitivity` are removed.

**Architecture:** A scan with a design of phase 1 (`sampling.local`, `sobol`, `fast`, `morris`) runs with `Simulator.run` and observables; the result keeps the design record in `attrs["scan"]`. An analysis finds the dimension of its method, recreates what it needs from the record (the references and `delta` of a local design, the unit cube of a SALib design with `sampling.designs.unit_cube`), moves that dimension to the front of every observable and computes the indices per remaining array element (SALib's `analyze` per element for the global methods, central differences for the local one). The indices go into an `xarray.Dataset` wrapped by `SensitivityResult`, `<observable>.<index>` over `(parameter, *other dims, [time])`, with units and netCDF; plots and the classification of today work on it.

**Tech Stack:** python 3.13/3.14, uv, numpy, xarray, SALib 1.6 (lazily imported), pint, matplotlib, seaborn (the heatmap of today), pytest with xdist, ruff, ty, zensical.

**Spec:** `docs/superpowers/specs/2026-10-09-analyses-design.md` (phase 2 of its "Phases": the section "Sensitivity" and the testing of phase 2). Phase 1 (the sampler, the uncertainty plots) is the pull request #258 on the branch `analyses-phase1`.

## Global Constraints

- Phase 2 starts from the branch `analyses-phase1` (the head of #258); after #258 is merged it is rebased onto `develop`. Branch `analyses-phase2`, one pull request to `develop`, one commit per task.
- `local(result, *, dim=None, observables=None)`, `sobol(result, *, dim=None, observables=None, conf_level=0.95, num_resamples=100)`, `fast(...)` and `morris(...)` read the record of the dimension of their method (`dim=` chooses one of several, none raises) and the kept observables of the result (or `observables=`), and compute the indices of every array element: every label of the other dimensions and, for a timecourse on a grid, every time point; a ragged timecourse raises and names `time=`.
- `local`: `raw = (y(+) - y(-)) / (2 delta p_ref)` in the unit of the observable per unit of the parameter, `normalized = (y(+) - y(-)) / (2 delta y_ref)`; `sobol`: `S1`, `ST` and with second order `S2`, each with `_conf`; `fast`: `S1`, `ST` with `_conf`; `morris`: `mu`, `mu_star`, `sigma`, `mu_star_conf`, on the unit cube the record recreates.
- An element whose points contain a failed simulation (`NaN`) has `NaN` indices, one warning counts them; an element which is constant has `NaN` indices of variance (Sobol, FAST) and zero effects (local, Morris).
- `SensitivityResult` wraps an `xarray.Dataset`: one variable per observable and index, `<observable>.<index>`, over `(parameter, *other dimensions, [time])`, with `attrs["units"]` and the method, its options and the provenance of the scan in `attrs`; `index(name, observables=None)`, `sel`/`isel`, `to_dataframe`, `to_netcdf`/`from_netcdf`, `classify(name)`.
- Plots are functions which return a figure and save it only with a path: `plot_heatmap(result, index, ...)`, `plot_indices(result, observable)`, `plot_morris(result, observable)`.
- The analysis classes, `SensitivitySimulation`, `SensitivityOutput`, `AnalysisGroup`, `SensitivityParameter`, `ParameterType` and `parameters.py` are removed; `classification.py` stays.
- Never use the em dash character, use a plain dash `-`.
- No agent attribution anywhere: no `Co-Authored-By` trailer, no "Generated with Claude Code" line in commits, the pull request, docs or code.
- Commit messages are full sentences which describe the outcome, in the style of `git log`, with a body that explains what and why; no conventional commit prefixes.
- Never edit `CHANGELOG.md` or auto-generated files; no release notes (they belong to the release commit).
- Markdown has no hard line wraps.
- Every module, class and function of the package has full type annotations and a google style docstring (ruff `D`); `tests/` and `examples/` are exempt from docstrings. A subclass marks overrides with `typing.override`.
- ty stays at zero diagnostics (`uv run ty check` checks `src`, `tests`, `examples`, `scripts`; unused ignore comments are warnings); suppress only with a rule specific `# ty: ignore[rule-name]`.
- Library code logs with `logging.getLogger(__name__)` and lazy `%s` formatting, it never prints.
- Every commit passes `uv run ruff check`, `uv run ruff format --check`, `uv run ty check` and `uv run pytest -q` (xdist, `filterwarnings = error`). Use `uv run` for every command.

## Decisions this plan takes where the spec is silent

- The analyses live in `src/sbmlsim/sensitivity/indices.py` (the four functions and their helpers), `SensitivityResult` in `src/sbmlsim/sensitivity/result.py`, the plots in `src/sbmlsim/sensitivity/plots.py` (rewritten on `SensitivityResult`, the clustered heatmap of today kept as its engine); `sbmlsim.sensitivity` exports `local`, `sobol`, `fast`, `morris`, `SensitivityResult`, `plot_heatmap`, `plot_indices`, `plot_morris`.
- The parameters of the indices are the targets of the design (the keys of its distributions, or the targets of a local design), in their order; the dimension `parameter` has them as labels.
- A SALib analysis checks that the dimension has as many points as the recreated unit cube; the values themselves are not compared (the result holds them as coordinates in the unit of the model, the unit cube is in probabilities).
- Sobol and FAST are given `seed` (the seed of the design, so the bootstrap intervals are reproducible) and `num_resamples`; Morris gets `num_levels` from the record and `scaled=False`.
- `y_ref` of `normalized` is the value at the label `reference`; a zero `y_ref` or a zero reference of a parameter gives `NaN`.
- Second order Sobol indices (`S2`) are over `(parameter, parameter_2, ...)`.
- `classify(name)` applies `sensitivity_classification` to every value of the index (an array of `SensitivityClassification` values as strings), for local normalized indices as today.
- The example `examples/sensitivity/sensitivity_example.py` keeps its `--quick` and `--cores` arguments (`--cores` is the `n_workers` of the simulator).

## Review Focus

- A scan whose design dimension is not the first dimension (e.g. doses first, the design second): the indices are per dose, with the design's axis taken from its position; covered in Task 2 (`test_the_design_dimension_need_not_be_first`).
- A timecourse observable on a grid together with a scalar observable: the timecourse has indices per time, the scalar none, both in one result; covered in Task 3 (`test_timecourses_and_scalars`).
- A flagged run with a failed point: the elements touching it are `NaN` with one warning, the others unchanged; covered in Task 3 (`test_a_failed_point_gives_nan_indices`).
- A result read back from netCDF gives the same indices as the result in memory; covered in Task 3 (`test_a_stored_result_is_analysed`).
- Two design dimensions of the same method: without `dim=` the analysis raises and names them; covered in Task 2 (`test_two_designs_need_dim`).

---

## File Structure

- Create `src/sbmlsim/sensitivity/result.py` (`SensitivityResult`), `src/sbmlsim/sensitivity/indices.py` (`local`, `sobol`, `fast`, `morris`, `design_of`).
- Rewrite `src/sbmlsim/sensitivity/plots.py` (`plot_heatmap`, `plot_indices`, `plot_morris`), `src/sbmlsim/sensitivity/__init__.py`.
- Delete `src/sbmlsim/sensitivity/analysis.py`, `parameters.py`, `sensitivity_local.py`, `sensitivity_sampling.py`, `sensitivity_sobol.py`, `sensitivity_fast.py`, `sensitivity_morris.py`; their tests `tests/sensitivity/test_analysis.py`, `test_parameters.py`, `test_sensitivity_example.py` (replaced); their API pages.
- Rewrite `examples/sensitivity/sensitivity_example.py`, `docs/sensitivity.md`; modify `examples/model_sensitivity.py` (local indices), `zensical.toml`, `CLAUDE.md`, `tests/docs/test_docs_code.py` (`PAGES` gains `sensitivity.md`), `tests/examples/test_example_scripts.py`.
- Tests: `tests/sensitivity/test_result.py`, `tests/sensitivity/test_local.py`, `tests/sensitivity/test_global.py`, `tests/sensitivity/test_plots.py`, `tests/sensitivity/models.py` (the Ishigami and power-law models).

---

### Task 1: SensitivityResult and finding a design in a result

**Files:**
- Create: `src/sbmlsim/sensitivity/result.py`, `src/sbmlsim/sensitivity/indices.py` (only `design_of` in this task)
- Test: `tests/sensitivity/test_result.py`

**Interfaces:**
- Consumes: `ScanResult` (`ds`, `attrs["scan"]["dimensions"]`: dicts with `id`, `design` (the record of `Design.to_dict()` or `None`), `values`, `labels`), `Design.from_dict` (phase 1).
- Produces:
  - `design_of(result: ScanResult, methods: Collection[str], dim: str | None = None) -> tuple[str, Design]` (the id of the dimension and its record).
  - `SensitivityResult(ds: xr.Dataset)` with `units`, `method`, `parameters` (the labels of `parameter`), `observables`, `__getitem__`, `__contains__`, `index(name, observables=None) -> xr.DataArray`, `sel(**indexers)`, `isel(**indexers)`, `to_dataframe(name) -> pd.DataFrame`, `to_netcdf(path)`, `from_netcdf(path)`, `classify(name) -> xr.DataArray`; the constants `PARAMETER = "parameter"`, `PARAMETER_2 = "parameter_2"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/sensitivity/test_result.py`:

```python
"""The result of a sensitivity analysis and finding the design of a result."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from sbmlsim.result import ScanResult
from sbmlsim.sensitivity.indices import design_of
from sbmlsim.sensitivity.result import PARAMETER, SensitivityResult
from sbmlsim.simulation import Dimension, Formula, Scan, Simulation, sampling
from sbmlsim.simulation.sampling import Uniform
from sbmlsim.simulator import Simulator
from tests.simulator.models import sbml


def _result() -> SensitivityResult:
    ds = xr.Dataset(
        {
            "auc.ST": ((PARAMETER, "dose"), np.array([[0.6, 0.7], [0.05, 0.3]])),
            "auc.S1": ((PARAMETER, "dose"), np.array([[0.5, 0.6], [0.04, 0.2]])),
            "c.ST": ((PARAMETER, "dose", "time"), np.ones((2, 2, 3))),
        },
        coords={PARAMETER: ["k1", "k2"], "dose": [0, 1], "time": [0.0, 1.0, 2.0]},
        attrs={"units": {"auc.ST": "dimensionless", "auc.S1": "dimensionless", "c.ST": "dimensionless"}, "method": "sobol", "options": {"n": 8}},
    )
    return SensitivityResult(ds)


def test_a_result_has_its_variables_and_metadata() -> None:
    s = _result()
    assert s.method == "sobol"
    assert s.parameters == ["k1", "k2"]
    assert s.observables == ["auc", "c"]
    assert "auc.ST" in s and s["auc.ST"].dims == (PARAMETER, "dose")


def test_index_stacks_the_scalar_observables() -> None:
    stacked = _result().index("ST")
    assert stacked.dims == (PARAMETER, "observable", "dose")
    assert stacked["observable"].values.tolist() == ["auc"]
    with pytest.raises(KeyError, match="nope"):
        _result().index("nope")


def test_classify_and_to_dataframe() -> None:
    classes = _result().classify("auc.ST")
    assert classes.sel({PARAMETER: "k1", "dose": 0}).item() == "high"
    assert classes.sel({PARAMETER: "k2", "dose": 0}).item() == "negligible"
    df = _result().to_dataframe("auc.ST")
    assert list(df.index) == ["k1", "k2"]


def test_a_result_survives_netcdf(tmp_path: Path) -> None:
    s = _result()
    path = tmp_path / "s.nc"
    s.to_netcdf(path)
    again = SensitivityResult.from_netcdf(path)
    xr.testing.assert_equal(again.ds, s.ds)
    assert again.method == "sobol" and again.units == s.units


def test_design_of_finds_the_dimension_of_a_method() -> None:
    model = Simulator().load(sbml())
    design = sampling.sobol({"k1": Uniform(0.5, 1.0), "k2": Uniform(0.5, 1.0)}, 8, seed=1)
    doses = Dimension("dose", values={"a0": [1.0, 2.0]})
    res = Simulator().run(model, Scan(Simulation(end=1, steps=2), [doses, design]), [Formula("k", "max(k1)")])
    dim, record = design_of(res, {"sobol"})
    assert dim == "sobol" and record.method == "sobol"
    with pytest.raises(ValueError, match="morris"):
        design_of(res, {"morris"})


def test_two_designs_need_dim() -> None:
    model = Simulator().load(sbml())
    first = sampling.local(["k1"], 0.1, model=model, id="a")
    second = sampling.local(["k2"], 0.1, model=model, id="b")
    res = Simulator().run(model, Scan(Simulation(end=1, steps=2), [first, second]), [Formula("k", "max(k1)")])
    with pytest.raises(ValueError, match="'a'.*'b'|dim="):
        design_of(res, {"local"})
    assert design_of(res, {"local"}, dim="b")[0] == "b"
    assert isinstance(res, ScanResult)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/sensitivity/test_result.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbmlsim.sensitivity.result'`.

- [ ] **Step 3: Write the implementation**

Create `src/sbmlsim/sensitivity/result.py`:

```python
"""The result of a sensitivity analysis.

`SensitivityResult` wraps an `xarray.Dataset` with one variable per observable
and index, `<observable>.<index>` (e.g. `auc.ST`, `auc.ST_conf`,
`[S2].normalized`), over `(parameter, *other dimensions, [time])`: the
parameters of the design, every label of the other dimensions of the scan
(e.g. doses, conditions) and, for a timecourse on a grid, every time point.
`attrs` carry the units of every variable, the method, its options and the
provenance of the scan.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from sbmlsim.result.scan import TIME
from sbmlsim.sensitivity.classification import sensitivity_classification

#: the dimension of the parameters of the design
PARAMETER = "parameter"

#: the second dimension of the parameters of a second order index
PARAMETER_2 = "parameter_2"

#: the attribute which carries the attributes as JSON in a netCDF file
NETCDF_ATTRS = "sbmlsim"


class SensitivityResult:
    """The indices of a sensitivity analysis, see the module.

    Attributes:
        ds: the dataset.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        """Wrap a dataset of indices.

        Raises:
            ValueError: if the dataset has no dimension `parameter`.
        """
        if PARAMETER not in ds.dims:
            raise ValueError(f"A sensitivity result has the dimension '{PARAMETER}'.")
        self.ds = ds

    @property
    def units(self) -> dict[str, str]:
        """Get the unit of every variable."""
        return dict(self.ds.attrs.get("units", {}))

    @property
    def method(self) -> str:
        """Get the method of the analysis, e.g. `sobol`."""
        return str(self.ds.attrs.get("method", ""))

    @property
    def parameters(self) -> list[str]:
        """Get the parameters of the indices."""
        return [str(p) for p in self.ds[PARAMETER].values.tolist()]

    @property
    def observables(self) -> list[str]:
        """Get the observables, in the order of their first variable."""
        return list(dict.fromkeys(str(name).rsplit(".", 1)[0] for name in self.ds.data_vars))

    def __getitem__(self, key: str) -> xr.DataArray:
        """Get the variable of an index of an observable, e.g. `auc.ST`."""
        return self.ds[key]

    def __contains__(self, key: object) -> bool:
        """Check whether the result has a variable."""
        return key in self.ds.data_vars

    def index(self, name: str, observables: Sequence[str] | None = None) -> xr.DataArray:
        """Stack an index of the scalar observables into `(parameter, observable, ...)`.

        Args:
            name: the index, e.g. `ST`.
            observables: the observables, every scalar observable which has
                the index by default.

        Returns:
            The index over `(parameter, observable, *other dimensions)`.

        Raises:
            KeyError: if no scalar observable has the index.
        """
        chosen = [
            o
            for o in (observables or self.observables)
            if f"{o}.{name}" in self.ds.data_vars and TIME not in self.ds[f"{o}.{name}"].dims
        ]
        if not chosen:
            raise KeyError(f"No scalar observable of the result has the index '{name}'.")
        stacked = xr.concat([self.ds[f"{o}.{name}"] for o in chosen], dim=pd.Index(chosen, name="observable"))
        dims = [PARAMETER, "observable", *(d for d in stacked.dims if d not in (PARAMETER, "observable"))]
        return stacked.transpose(*dims)

    def sel(self, **indexers: Any) -> SensitivityResult:
        """Select labels, see `xarray.Dataset.sel`."""
        return SensitivityResult(self.ds.sel(**indexers))

    def isel(self, **indexers: Any) -> SensitivityResult:
        """Select positions, see `xarray.Dataset.isel`."""
        return SensitivityResult(self.ds.isel(**indexers))

    def to_dataframe(self, name: str) -> pd.DataFrame:
        """Get a variable as a table, a row per parameter.

        Args:
            name: the variable, e.g. `auc.ST`.

        Returns:
            The table; the other dimensions are its columns.
        """
        data = self.ds[name]
        others = [d for d in data.dims if d != PARAMETER]
        frame = data.to_dataframe(name=name)[name]
        if others:
            return frame.unstack(others)  # ty: ignore[invalid-return-type]
        return frame.to_frame()

    def classify(self, name: str) -> xr.DataArray:
        """Classify every value of an index, see `sensitivity_classification`.

        Args:
            name: the variable, e.g. `auc.normalized`.

        Returns:
            The classes as strings (`high`, `medium`, `low`, `negligible`), `""`
            for `NaN`.
        """
        data = self.ds[name]
        classes = np.vectorize(
            lambda v: "" if np.isnan(v) else str(sensitivity_classification(float(v))),
            otypes=[object],
        )(data.values)
        return data.copy(data=classes.astype(str))

    def to_netcdf(self, path: str | Path) -> None:
        """Write the result as netCDF, the attributes as JSON."""
        ds = self.ds.copy()
        ds.attrs = {NETCDF_ATTRS: json.dumps(self.ds.attrs)}
        ds.to_netcdf(path, engine="h5netcdf")

    @classmethod
    def from_netcdf(cls, path: str | Path) -> SensitivityResult:
        """Read a result written by `to_netcdf`."""
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            loaded = ds.load()
        loaded.attrs = json.loads(loaded.attrs.pop(NETCDF_ATTRS, "{}"))
        return cls(loaded)
```

Replace the `ty: ignore` in `to_dataframe` by a typed variable if ty accepts it; keep the JSON encoding of numpy values consistent with `ScanResult.to_netcdf` (reuse its `_json_default` from `sbmlsim.result.scan` if the attributes can hold numpy numbers; the attributes this phase writes are JSON types already).

Create `src/sbmlsim/sensitivity/indices.py` with its module docstring (see Task 2, which adds the analyses) and:

```python
def design_of(
    result: ScanResult, methods: Collection[str], dim: str | None = None
) -> tuple[str, Design]:
    """Find the dimension of a design of a method in a result.

    Args:
        result: the result of a scan with a design of `sbmlsim.simulation.sampling`.
        methods: the methods an analysis takes, e.g. `{"sobol"}`.
        dim: the id of the dimension, needed when there are several.

    Returns:
        The id of the dimension and its record.

    Raises:
        ValueError: if no dimension has a design of the methods, several have
            and `dim` is not given, or `dim` has none.
    """
    found = {
        str(d["id"]): Design.from_dict(d["design"])
        for d in result.ds.attrs.get("scan", {}).get("dimensions", [])
        if d.get("design") and d["design"]["method"] in methods
    }
    if dim is not None:
        if dim not in found:
            raise ValueError(f"The dimension '{dim}' has no design of {sorted(methods)}: {sorted(found)}.")
        return dim, found[dim]
    if not found:
        raise ValueError(
            f"The result has no dimension with a design of {sorted(methods)}; create "
            f"one with sbmlsim.simulation.sampling."
        )
    if len(found) > 1:
        raise ValueError(f"The result has the designs {sorted(found)} of {sorted(methods)}; choose one with dim=.")
    ((name, design),) = found.items()
    return name, design
```

(If the provenance key of the dimensions differs, read `Scan.to_dict` and use its key; phase 1 verified `attrs["scan"]["dimensions"]`.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/sensitivity/test_result.py`
Expected: PASS.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/sensitivity/result.py src/sbmlsim/sensitivity/indices.py tests/sensitivity/test_result.py
git commit -m "A sensitivity result holds the indices of every observable over the parameters" -m "SensitivityResult wraps a dataset with one variable per observable and index over the parameters of the design, the other dimensions of the scan and the time, with units, netCDF, a stack of the scalar observables for heatmaps and the classification of today. design_of finds the dimension of a design of a method in the provenance of a result."
```

---

### Task 2: The local analysis

**Files:**
- Modify: `src/sbmlsim/sensitivity/indices.py` (`local` and the shared helpers)
- Create: `tests/sensitivity/models.py` (the power-law and Ishigami models)
- Test: `tests/sensitivity/test_local.py`

**Interfaces:**
- Consumes: `design_of` (Task 1); `SensitivityResult`, `PARAMETER` (Task 1); `ScanResult` (`ds`, `units`, `ragged`); `sampling.local` (phase 1: labels `reference`, `<t>+`, `<t>-`, options `delta`, `targets`, references `{t: {"value", "unit"}}`).
- Produces: `local(result: ScanResult, *, dim: str | None = None, observables: Sequence[str] | None = None) -> SensitivityResult`; helpers `_observables(result, observables) -> list[str]` (the kept variables, a ragged timecourse raises), `_moved(result, name, dim) -> tuple[np.ndarray, list[str]]` (the values with the design axis first, and the other dims), `_assemble(result, dim, method, options, parameters, indices, units) -> SensitivityResult`.

- [ ] **Step 1: Write the failing tests**

Create `tests/sensitivity/models.py`:

```python
"""Models with known sensitivities."""

#: y = a^2 * b^-1 * c^0.5: the normalized local sensitivities are 2, -1 and 0.5
POWER_LAW = """
model power
  a = 2; b = 3; c = 4
  y := a^2 * b^(-1) * c^0.5
end
"""

#: the Ishigami function of x1, x2, x3 uniform in [-pi, pi], a = 7, b = 0.1:
#: S1 = (0.314, 0.442, 0), ST = (0.558, 0.442, 0.244)
ISHIGAMI = """
model ishigami
  x1 = 0; x2 = 0; x3 = 0
  y := sin(x1) + 7 * sin(x2)^2 + 0.1 * x3^4 * sin(x1)
end
"""
```

Create `tests/sensitivity/test_local.py`:

```python
"""The local sensitivity analysis."""

import numpy as np
import pytest

from sbmlsim import sensitivity
from sbmlsim.sensitivity.result import PARAMETER
from sbmlsim.simulation import Dimension, Formula, Scan, Simulation, sampling
from sbmlsim.simulator import Simulator
from tests.sensitivity.models import POWER_LAW
from tests.simulator.models import sbml


@pytest.fixture(scope="module")
def model() -> object:
    return Simulator().load(sbml(POWER_LAW))


def test_the_normalized_sensitivities_are_the_exponents(model: object) -> None:
    design = sampling.local(["a", "b", "c"], 0.001, model=model)
    res = Simulator().run(model, Scan(Simulation(end=1, steps=1), [design]), [Formula("y_max", "max(y)")])
    s = sensitivity.local(res)
    np.testing.assert_allclose(s["y_max.normalized"].values, [2.0, -1.0, 0.5], rtol=1e-5)
    assert s.parameters == ["a", "b", "c"]
    # raw: dy/dp = normalized * y / p
    y = 2.0**2 / 3.0 * 4.0**0.5
    np.testing.assert_allclose(s["y_max.raw"].values, [2.0 * y / 2.0, -y / 3.0, 0.5 * y / 4.0], rtol=1e-5)
    assert s.method == "local"


def test_the_design_dimension_need_not_be_first(model: object) -> None:
    design = sampling.local(["b"], 0.001, model=model)
    scale = Dimension("scale", values={"c": [4.0, 16.0]})
    res = Simulator().run(model, Scan(Simulation(end=1, steps=1), [scale, design]), [Formula("y_max", "max(y)")])
    s = sensitivity.local(res)
    assert s["y_max.normalized"].dims == (PARAMETER, "scale")
    np.testing.assert_allclose(s["y_max.normalized"].values, [[-1.0, -1.0]], rtol=1e-5)


def test_a_timecourse_has_indices_per_time(model: object) -> None:
    design = sampling.local(["a"], 0.001, model=model)
    res = Simulator().run(model, Scan(Simulation(end=1, steps=2), [design]), [Formula("y", "y")])
    s = sensitivity.local(res)
    assert s["y.normalized"].dims == (PARAMETER, "time")
    np.testing.assert_allclose(s["y.normalized"].values, 2.0, rtol=1e-5)


def test_a_ragged_timecourse_raises() -> None:
    model = Simulator().load(sbml())
    design = sampling.local(["k1"], 0.1, model=model)
    res = Simulator().run(model, Scan(Simulation(end=1), [design]), [Formula("a", "[A]")])
    with pytest.raises(ValueError, match="time="):
        sensitivity.local(res)
```

Type the fixture as `RoadrunnerSBMLModel`. If antimony does not take `y := ...` without a species, give the model a dummy compartment; the probe model helper `sbml(model)` of tests/simulator/models.py compiles antimony.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/sensitivity/test_local.py`
Expected: FAIL with `AttributeError: module 'sbmlsim.sensitivity' has no attribute 'local'`.

- [ ] **Step 3: Write the implementation**

Write the module docstring of `src/sbmlsim/sensitivity/indices.py`:

```python
"""The sensitivity analyses: indices computed on the result of a scan.

A scan with a design of `sbmlsim.simulation.sampling` (`local`, `sobol`,
`fast`, `morris`) runs with `Simulator.run` and its observables; an analysis
reads the record of the design from the result, moves the dimension of the
design to the front of every kept observable and computes the indices of
every remaining array element: every label of the other dimensions of the
scan (doses, conditions) and every time point of a timecourse on a grid. A
ragged timecourse has no common time points; run the scan with `time=`.

- `local`: central differences at the reference, `raw` (the change of the
  observable per change of the parameter) and `normalized` (`∂ ln y / ∂ ln p`);
- `sobol`: the first order and total Sobol indices `S1`, `ST` (and `S2`)
  with their confidence intervals, SALib;
- `fast`: `S1`, `ST` of the extended FAST, SALib;
- `morris`: the elementary effects `mu`, `mu_star`, `sigma`, `mu_star_conf`,
  SALib on the unit cube the record recreates.

An element whose points contain a failed simulation (`NaN`) has `NaN`
indices, one warning counts them.
"""
```

and add `local` with the helpers (imports: `logging`, `Collection`, `Sequence`, `numpy`, `xarray`, `ScanResult`, `POINT`, `TIME`, `Design`, `SensitivityResult`, `PARAMETER`, `ureg`):

```python
def _observables(result: ScanResult, observables: Sequence[str] | None) -> list[str]:
    """Get the observables of an analysis; a ragged timecourse raises.

    Raises:
        ValueError: if an observable is no variable of the result or a ragged
            timecourse.
    """
    names = list(observables) if observables is not None else list(result.variables)
    for name in names:
        if name not in result.ds.data_vars:
            raise ValueError(f"'{name}' is no variable of the result: {list(result.variables)}.")
        if POINT in result.ds[name].dims:
            raise ValueError(
                f"'{name}' is a ragged timecourse, which has no common time points; "
                f"run the scan with time= or a simulation with steps or times."
            )
    return names


def _moved(result: ScanResult, name: str, dim: str) -> tuple[np.ndarray, list[str]]:
    """Get the values of a variable with the axis of the design first, and the other dims."""
    data = result.ds[name]
    others = [str(d) for d in data.dims if d != dim]
    return np.asarray(data.transpose(dim, *others).values, dtype=float), others


def _assemble(
    result: ScanResult,
    dim: str,
    method: str,
    options: dict[str, Any],
    parameters: list[str],
    indices: dict[str, tuple[list[str], np.ndarray]],
    units: dict[str, str],
) -> SensitivityResult:
    """Build the result: a variable per observable and index, the coords of the scan."""
    data_vars = {name: (dims, values) for name, (dims, values) in indices.items()}
    used = {d for dims, _ in indices.values() for d in dims} - {PARAMETER, "parameter_2"}
    coords: dict[str, Any] = {PARAMETER: parameters}
    for name in used:
        if name in result.ds.coords:
            coords[name] = result.ds.coords[name].values
    ds = xr.Dataset(
        data_vars,
        coords=coords,
        attrs={
            "units": units,
            "method": method,
            "options": options,
            "dim": dim,
            "scan": result.ds.attrs.get("scan", {}),
        },
    )
    return SensitivityResult(ds)


def _unit_ratio(numerator: str, denominator: str) -> str:
    """Get the unit of a ratio of two units, `""` where one is unknown."""
    if not numerator or not denominator:
        return ""
    return str((ureg.Unit(numerator) / ureg.Unit(denominator)))


def local(
    result: ScanResult,
    *,
    dim: str | None = None,
    observables: Sequence[str] | None = None,
) -> SensitivityResult:
    """Compute the local sensitivities of a scan with a local design, see the module.

    `raw = (y(+) - y(-)) / (2 delta p_ref)` and `normalized = (y(+) - y(-)) /
    (2 delta y_ref)`, with `y_ref` the value at the reference; a zero `y_ref`
    gives `NaN`.

    Args:
        result: the result of a scan with a `sampling.local` design.
        dim: the dimension of the design, needed when there are several.
        observables: the observables, every variable of the result by default.

    Returns:
        The indices `raw` and `normalized` of every observable.

    Raises:
        ValueError: if the result has no local design or several without
            `dim`, or an observable is a ragged timecourse.
    """
    dim, design = design_of(result, {"local"}, dim)
    delta = float(design.options["delta"])
    targets = [str(t) for t in design.options["targets"]]
    labels = [str(label) for label in result.ds[dim].values.tolist()]
    reference = labels.index("reference")
    indices: dict[str, tuple[list[str], np.ndarray]] = {}
    units: dict[str, str] = {}
    for name in _observables(result, observables):
        values, others = _moved(result, name, dim)
        y_ref = values[reference]
        raw = []
        normalized = []
        for target in targets:
            up, down = values[labels.index(f"{target}+")], values[labels.index(f"{target}-")]
            p_ref = float(design.references[target]["value"])
            with np.errstate(divide="ignore", invalid="ignore"):
                raw.append((up - down) / (2.0 * delta * p_ref))
                normalized.append(np.where(y_ref != 0.0, (up - down) / (2.0 * delta * y_ref), np.nan))
        dims = [PARAMETER, *others]
        indices[f"{name}.raw"] = (dims, np.stack(raw))
        indices[f"{name}.normalized"] = (dims, np.stack(normalized))
        unit = result.units.get(name, "")
        target_units = {design.references[t]["unit"] for t in targets}
        units[f"{name}.raw"] = _unit_ratio(unit, target_units.pop()) if len(target_units) == 1 else ""
        units[f"{name}.normalized"] = "dimensionless"
    options = {"delta": delta, "targets": targets}
    return _assemble(result, dim, "local", options, targets, indices, units)
```

The unit of `raw` differs per parameter when the targets have different units; a variable has one unit, so `raw` gets a unit only when all targets share one (else `""`) - write that in the docstring. Export `local` from `src/sbmlsim/sensitivity/__init__.py` next to the old classes for now (Task 5 rewrites the package `__init__`): add `from .indices import local` and `"local"` to `__all__`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/sensitivity/test_local.py`
Expected: PASS.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/sensitivity tests/sensitivity/models.py tests/sensitivity/test_local.py
git commit -m "The local sensitivities are computed on the result of a scan with a local design" -m "sensitivity.local reads the references and the delta of the design from the result and gives the raw and the normalized central differences of every observable, for every label of the other dimensions and every time point of a timecourse on a grid; the normalized sensitivities of a power law are its exponents."
```

---

### Task 3: The global analyses: sobol, fast, morris

**Files:**
- Modify: `src/sbmlsim/sensitivity/indices.py`, `src/sbmlsim/sensitivity/__init__.py`
- Test: `tests/sensitivity/test_global.py`

**Interfaces:**
- Consumes: the helpers of Task 2; `sampling.designs.unit_cube(design, d)` (phase 1); SALib 1.6: `SALib.analyze.sobol.analyze(problem, Y, calc_second_order, num_resamples, conf_level, print_to_console=False, seed=...)`, `SALib.analyze.fast.analyze(problem, Y, M, num_resamples, conf_level, print_to_console=False, seed=...)`, `SALib.analyze.morris.analyze(problem, X, Y, num_resamples, conf_level, scaled=False, print_to_console=False, num_levels, seed=...)`.
- Produces: `sobol(result, *, dim=None, observables=None, conf_level=0.95, num_resamples=100) -> SensitivityResult`, `fast(...)`, `morris(...)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/sensitivity/test_global.py`:

```python
"""The global sensitivity analyses."""

from pathlib import Path

import numpy as np
import pytest
from SALib.analyze import sobol as salib_sobol

from sbmlsim import sensitivity
from sbmlsim.result import ScanResult
from sbmlsim.sensitivity.result import PARAMETER
from sbmlsim.simulation import Dimension, Formula, Scan, Simulation, sampling
from sbmlsim.simulation.sampling import Uniform
from sbmlsim.simulation.sampling.designs import unit_cube
from sbmlsim.simulator import Simulator
from tests.sensitivity.models import ISHIGAMI
from tests.simulator.models import BLOWUP, sbml

PI = np.pi
BOUNDS = {"x1": Uniform(-PI, PI), "x2": Uniform(-PI, PI), "x3": Uniform(-PI, PI)}
OBSERVABLES = [Formula("y_max", "max(y)")]


def _run(design: Dimension, *others: Dimension) -> ScanResult:
    model = Simulator().load(sbml(ISHIGAMI))
    return Simulator(n_workers=1).run(model, Scan(Simulation(end=1, steps=1), [*others, design]), OBSERVABLES)


def test_sobol_of_the_ishigami_function() -> None:
    res = _run(sampling.sobol(BOUNDS, 1024, seed=1))
    s = sensitivity.sobol(res)
    np.testing.assert_allclose(s["y_max.S1"].values, [0.314, 0.442, 0.0], atol=0.05)
    np.testing.assert_allclose(s["y_max.ST"].values, [0.558, 0.442, 0.244], atol=0.05)
    assert "y_max.ST_conf" in s and s.method == "sobol"


def test_sobol_equals_salib_on_the_same_values() -> None:
    design = sampling.sobol(BOUNDS, 64, seed=2)
    res = _run(design)
    s = sensitivity.sobol(res)
    problem = {"num_vars": 3, "names": ["x1", "x2", "x3"], "bounds": [[0.0, 1.0]] * 3}
    y = res["y_max"].values
    expected = salib_sobol.analyze(problem, y, calc_second_order=False, num_resamples=100, conf_level=0.95, print_to_console=False, seed=design.design.options["seed"])
    np.testing.assert_allclose(s["y_max.S1"].values, expected["S1"])
    np.testing.assert_allclose(s["y_max.ST"].values, expected["ST"])


def test_fast_and_morris_rank_the_ishigami_parameters() -> None:
    f = sensitivity.fast(_run(sampling.fast(BOUNDS, 1025, seed=3)))
    np.testing.assert_allclose(f["y_max.S1"].values, [0.314, 0.442, 0.0], atol=0.06)
    m = sensitivity.morris(_run(sampling.morris(BOUNDS, 100, seed=4)))
    mu_star = m["y_max.mu_star"].values
    assert mu_star[0] > mu_star[2] > 0.0 and mu_star[1] > 0.0
    assert set(m.observables) == {"y_max"}
    assert {"y_max.mu", "y_max.sigma", "y_max.mu_star_conf"} <= set(m.ds.data_vars)


def test_indices_per_label_of_another_dimension() -> None:
    shift = Dimension("shift", values={"x3": [0.0, 0.0]})  # changes nothing, two labels
    res = _run(sampling.sobol({"x1": Uniform(-PI, PI), "x2": Uniform(-PI, PI)}, 64, seed=5), shift)
    s = sensitivity.sobol(res)
    assert s["y_max.S1"].dims == (PARAMETER, "shift")


def test_timecourses_and_scalars() -> None:
    model = Simulator().load(sbml(ISHIGAMI))
    design = sampling.sobol(BOUNDS, 16, seed=6)
    res = Simulator(n_workers=1).run(model, Scan(Simulation(end=1, steps=2), [design]), [Formula("y", "y"), Formula("y_max", "max(y)")])
    s = sensitivity.sobol(res)
    assert s["y.ST"].dims == (PARAMETER, "time") and s["y_max.ST"].dims == (PARAMETER,)


def test_a_failed_point_gives_nan_indices(caplog: pytest.LogCaptureFixture) -> None:
    model = Simulator().load(sbml(BLOWUP))
    design = sampling.sobol({"k": Uniform(0.1, 3.0)}, 8, seed=7)
    res = Simulator(n_workers=1).run(model, Scan(Simulation(end=1, steps=2), [design]), [Formula("s_max", "max(S)")], on_error="flag")
    assert res["status"].values.any()
    s = sensitivity.sobol(res)
    assert np.isnan(s["s_max.S1"].values).all()
    assert sum("NaN" in r.message for r in caplog.records) == 1


def test_a_stored_result_is_analysed(tmp_path: Path) -> None:
    res = _run(sampling.sobol(BOUNDS, 32, seed=8))
    path = tmp_path / "r.nc"
    res.to_netcdf(path)
    again = sensitivity.sobol(ScanResult.from_netcdf(path))
    np.testing.assert_allclose(again["y_max.ST"].values, sensitivity.sobol(res)["y_max.ST"].values)


def test_a_wrong_number_of_points_raises() -> None:
    design = sampling.sobol(BOUNDS, 16, seed=9)
    assert unit_cube(design.design, 3).shape[0] == len(design)  # ty: ignore[possibly-unbound-attribute]
```

Replace the last test with a check that tampering the record (e.g. a result whose record says `n=32` for 16-point values, built by hand from a `ScanResult` with an edited `attrs["scan"]`) raises a `ValueError` naming the points; type `design.design` with an `assert ... is not None` instead of the ignore.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/sensitivity/test_global.py`
Expected: FAIL with `AttributeError: module 'sbmlsim.sensitivity' has no attribute 'sobol'`.

- [ ] **Step 3: Write the implementation**

Add to `src/sbmlsim/sensitivity/indices.py`:

```python
def _unit_points(result: ScanResult, dim: str, design: Design) -> tuple[list[str], np.ndarray]:
    """Get the parameters and the unit cube of a SALib design, checked against the result.

    Raises:
        ValueError: if the dimension has another number of points than the cube.
    """
    parameters = list(design.distributions)
    cube = unit_cube(design, len(parameters))
    points = result.ds.sizes[dim]
    if cube.shape[0] != points:
        raise ValueError(
            f"The design of the dimension '{dim}' has {cube.shape[0]} points by its "
            f"record, the result has {points}."
        )
    return parameters, cube


def _global(
    result: ScanResult,
    methods: set[str],
    dim: str | None,
    observables: Sequence[str] | None,
    analyze: Callable[[Design, np.ndarray, np.ndarray], dict[str, np.ndarray]],
    keys: Sequence[str],
) -> SensitivityResult:
    """Compute the indices of a SALib method for every element, see the module.

    `analyze(design, cube, y)` gives the indices of one element (`y` the values
    of the points); an element with a `NaN` value gives `NaN` indices.
    """
    dim, design = design_of(result, methods, dim)
    parameters, cube = _unit_points(result, dim, design)
    indices: dict[str, tuple[list[str], np.ndarray]] = {}
    units: dict[str, str] = {}
    failed = 0
    for name in _observables(result, observables):
        values, others = _moved(result, name, dim)
        flat = values.reshape(values.shape[0], -1)
        out = {key: np.full((len(parameters), flat.shape[1]), np.nan) for key in keys}
        for k in range(flat.shape[1]):
            y = flat[:, k]
            if not np.isfinite(y).all():
                failed += 1
                continue
            for key, value in analyze(design, cube, y).items():
                out[key][:, k] = np.asarray(value, dtype=float)
        shape = (len(parameters), *values.shape[1:])
        for key in keys:
            indices[f"{name}.{key}"] = ([PARAMETER, *others], out[key].reshape(shape))
            units[f"{name}.{key}"] = "dimensionless" if not key.startswith("mu") and key != "sigma" else result.units.get(name, "")
    if failed:
        logger.warning(
            "%s elements of the result contain a failed simulation (NaN); their "
            "indices are NaN.",
            failed,
        )
    return _assemble(result, dim, design.method, dict(design.options), parameters, indices, units)
```

and the three analyses, each with a full docstring (Args, Returns, Raises as `local`), e.g.:

```python
def sobol(
    result: ScanResult,
    *,
    dim: str | None = None,
    observables: Sequence[str] | None = None,
    conf_level: float = 0.95,
    num_resamples: int = 100,
) -> SensitivityResult:
    """Compute the Sobol indices of a scan with a Sobol design, see the module."""
    from SALib.analyze import sobol as analyzer

    def analyze(design: Design, cube: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        second = bool(design.options["second_order"])
        found = analyzer.analyze(
            _problem(cube.shape[1]),
            y,
            calc_second_order=second,
            num_resamples=num_resamples,
            conf_level=conf_level,
            print_to_console=False,
            seed=design.options["seed"],
        )
        return {key: found[key] for key in ("S1", "S1_conf", "ST", "ST_conf")}

    return _global(result, {"sobol"}, dim, observables, analyze, ("S1", "S1_conf", "ST", "ST_conf"))
```

- `fast`: `SALib.analyze.fast.analyze(problem, y, M=design.options["m"], num_resamples=..., conf_level=..., print_to_console=False, seed=design.options["seed"])`, keys `S1`, `S1_conf`, `ST`, `ST_conf`.
- `morris`: `SALib.analyze.morris.analyze(problem, cube, y, num_resamples=..., conf_level=..., scaled=False, print_to_console=False, num_levels=design.options["levels"], seed=design.options["seed"])`, keys `mu`, `mu_star`, `sigma`, `mu_star_conf`; the unit of `mu`, `mu_star`, `sigma`, `mu_star_conf` is the unit of the observable per step of the unit cube (write it as the unit of the observable and document it).
- `_problem(d)` is the problem of the unit cube; import it from `sbmlsim.simulation.sampling.designs` (where phase 1 defines it) rather than repeating it.
- Second order Sobol indices: when `second_order` is true, add `S2` and `S2_conf` over `(parameter, parameter_2, *others)` (SALib gives a `d x d` matrix per element); extend `_global` with an optional second-order path or handle `S2` inside `sobol` after `_global`; test it with `second_order=True` on Ishigami (`S2[x1, x3]` about 0.244, atol 0.06).
- A SALib analysis may warn (e.g. a constant output gives a division by zero); a constant element must give `NaN` Sobol/FAST indices without a warning reaching the test run (compute the variance first and skip the element with `NaN` when it is zero, which also keeps the warnings away), and zero Morris effects.

Export `sobol`, `fast` and `morris` from the package `__init__`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/sensitivity/test_global.py`
Expected: PASS (the Ishigami runs take a few seconds serially).

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/sensitivity tests/sensitivity/test_global.py
git commit -m "The Sobol, FAST and Morris indices are computed on the result of a scan" -m "sobol, fast and morris recreate the unit cube of their design from the record of the result and run the analyses of SALib on every element of every observable: every label of the other dimensions and every time point of a timecourse on a grid. The Ishigami function gives its known indices; an element with a failed simulation gives NaN indices and one warning; a result read from netCDF gives the same indices."
```

---

### Task 4: The plots of a sensitivity result

**Files:**
- Rewrite: `src/sbmlsim/sensitivity/plots.py`
- Test: `tests/sensitivity/test_plots.py`

**Interfaces:**
- Consumes: `SensitivityResult` (Task 1), the clustered heatmap of today (`heatmap(df, ...)` in plots.py, kept as a private engine `_heatmap`).
- Produces: `plot_heatmap(result: SensitivityResult, index: str, *, observables=None, cutoff=0.1, cluster_rows=True, title=None, cmap="seismic", vmin=None, vmax=None, path=None, dpi=300, **selection) -> Figure`; `plot_indices(result, observable, *, path=None, **selection) -> Figure` (S1 and ST bars with their intervals, Sobol/FAST); `plot_morris(result, observable, *, path=None, **selection) -> Figure` (`mu_star` against `sigma`, one point per parameter, labelled).

- [ ] **Step 1: Write the failing tests**

Create `tests/sensitivity/test_plots.py`:

```python
"""The plots of a sensitivity result."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from matplotlib.figure import Figure

from sbmlsim.sensitivity import plot_heatmap, plot_indices, plot_morris
from sbmlsim.sensitivity.result import PARAMETER, SensitivityResult


def _sobol() -> SensitivityResult:
    rng = np.random.default_rng(1)
    data = {}
    for o in ("auc", "cmax"):
        for key in ("S1", "ST", "S1_conf", "ST_conf"):
            data[f"{o}.{key}"] = ((PARAMETER, "dose"), rng.random((3, 2)))
    units = dict.fromkeys(data, "dimensionless")
    return SensitivityResult(xr.Dataset(data, coords={PARAMETER: ["a", "b", "c"], "dose": [0, 1]}, attrs={"units": units, "method": "sobol"}))


def _morris() -> SensitivityResult:
    data = {f"y.{key}": ((PARAMETER,), np.array([1.0, 0.5, 0.1])) for key in ("mu", "mu_star", "sigma", "mu_star_conf")}
    return SensitivityResult(xr.Dataset(data, coords={PARAMETER: ["a", "b", "c"]}, attrs={"units": dict.fromkeys(data, ""), "method": "morris"}))


def test_plot_heatmap(tmp_path: Path) -> None:
    figure = plot_heatmap(_sobol(), "ST", dose=0, path=tmp_path / "h.png", cutoff=None)
    assert isinstance(figure, Figure) and (tmp_path / "h.png").exists()
    with pytest.raises(ValueError, match="dose"):
        plot_heatmap(_sobol(), "ST")  # two doses: choose one


def test_plot_indices_and_morris() -> None:
    figure = plot_indices(_sobol(), "auc", dose=1)
    assert isinstance(figure, Figure) and len(figure.axes[0].patches) == 6
    figure = plot_morris(_morris(), "y")
    assert isinstance(figure, Figure) and len(figure.axes[0].texts) == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/sensitivity/test_plots.py`
Expected: FAIL with `ImportError: cannot import name 'plot_heatmap'`.

- [ ] **Step 3: Write the implementation**

Rewrite `src/sbmlsim/sensitivity/plots.py`: keep the body of today's `heatmap` as `_heatmap(df, parameter_labels, output_labels, cutoff, annotate_values, cluster_rows, cluster_cols, title, cmap, vcenter, vmin, vmax, fig_path, dpi) -> Figure` (unchanged), drop `plot_S1_ST_indices` and `S1_ST_barplot` (their bar plot becomes `plot_indices`), and add:

- `_selected(data: xr.DataArray, selection: Mapping[str, Any], keep: set[str]) -> xr.DataArray`: `data.sel(selection)`, then raise a `ValueError` naming every dimension outside `keep` (e.g. `dose`) which has more than one label ("choose one label of 'dose' with dose=...").
- `plot_heatmap`: `result.index(index, observables)` stacked to `(parameter, observable, ...)`, selected down to `(parameter, observable)`, as a DataFrame (rows parameters, columns observables) into `_heatmap`; `vmin`/`vmax` default to `-2, 2` for `normalized` and `0, 1` otherwise, `vcenter` the middle; saved to `path` with `dpi` if given; returns the figure.
- `plot_indices`: the variables `<observable>.S1`, `.ST` and their `_conf`, selected down to `(parameter,)`; two bars per parameter (S1, ST) with error bars of the intervals; axis labels; legend; a `Figure` of `matplotlib.figure.Figure` (no pyplot).
- `plot_morris`: `mu_star` against `sigma` of the observable, a point per parameter annotated with its name (`ax.annotate`), axis labels with the unit of the observable.

Each with a full docstring; no `plt.show()`. Export `plot_heatmap`, `plot_indices`, `plot_morris` from the package `__init__`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/sensitivity/test_plots.py`
Expected: PASS.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/sensitivity/plots.py src/sbmlsim/sensitivity/__init__.py tests/sensitivity/test_plots.py
git commit -m "A sensitivity result is drawn as a heatmap, bars of indices or the plane of Morris" -m "plot_heatmap draws an index of the scalar observables over the parameters with the clustered heatmap of before, plot_indices the S1 and ST bars with their intervals and plot_morris mu_star against sigma; a dimension with several labels must be chosen, and every plot returns its figure and saves it only with a path."
```

---

### Task 5: The old analyses go, the examples and the docs follow

**Files:**
- Delete: `src/sbmlsim/sensitivity/analysis.py`, `parameters.py`, `sensitivity_local.py`, `sensitivity_sampling.py`, `sensitivity_sobol.py`, `sensitivity_fast.py`, `sensitivity_morris.py`; `tests/sensitivity/test_analysis.py`, `test_parameters.py`, `test_sensitivity_example.py`; `docs/api/sensitivity.analysis.md`, `sensitivity.parameters.md`, `sensitivity.sensitivity_local.md`, `sensitivity.sensitivity_sampling.md`, `sensitivity.sensitivity_sobol.md`, `sensitivity.sensitivity_fast.md`, `sensitivity.sensitivity_morris.md`
- Rewrite: `src/sbmlsim/sensitivity/__init__.py`, `examples/sensitivity/sensitivity_example.py`, `docs/sensitivity.md`
- Modify: `examples/model_sensitivity.py`, `zensical.toml` (nav), `tests/docs/test_docs_code.py` (`PAGES` gains `"sensitivity.md"`), `tests/examples/test_example_scripts.py` (the arguments of the example), `CLAUDE.md`, `examples/README.md`, `docs/api/index.md` if it lists the modules, `docs/index.md`/`README.md` if they name the old classes
- Create: `docs/api/sensitivity.indices.md`, `docs/api/sensitivity.result.md`

**Interfaces:**
- Consumes: everything of Tasks 1-4 and phase 1.

- [ ] **Step 1: Find every user of the old analyses**

Run: `rg -n "SensitivitySimulation|SensitivityAnalysis|LocalSensitivityAnalysis|SobolSensitivityAnalysis|FASTSensitivityAnalysis|MorrisSensitivityAnalysis|SamplingSensitivityAnalysis|SensitivityParameter|AnalysisGroup|SensitivityOutput|parameters_from_sbml|plot_S1_ST_indices" --glob '!docs/superpowers/**' --glob '!release-notes/**'`
Expected: the files of this task only. Every one is removed or migrated.

- [ ] **Step 2: The package and the example**

`src/sbmlsim/sensitivity/__init__.py`: a module docstring which describes the analyses on a result (the design of `sbmlsim.simulation.sampling`, `Simulator.run`, then `local`/`sobol`/`fast`/`morris`, the `SensitivityResult`, the plots and the classification, the uncertainty plots of `sbmlsim.sensitivity.uncertainty`), and the exports `fast`, `local`, `morris`, `sobol`, `SensitivityResult`, `plot_heatmap`, `plot_indices`, `plot_morris`.

Rewrite `examples/sensitivity/sensitivity_example.py`:

```python
"""Sensitivity analyses of a simple chain: local, Sobol, FAST and Morris.

The chain S1 -> S2 -> S3 with the rates k1 and k2 is simulated for three
initial concentrations of S1 (a dimension of the scan); the observables are
the AUC of every species and the maximum and the time of the maximum of S2.
Every analysis is a design of the sampler, a run and the indices on the
result.
"""

import argparse
from pathlib import Path

import numpy as np

from sbmlsim import sensitivity
from sbmlsim.simulation import Dimension, Formula, Scan, Simulation, sampling
from sbmlsim.simulator import Simulator

MODEL = Path(__file__).parent / "simple_chain.xml"

OBSERVABLES = [
    Formula("S1_auc", "mean([S1]) * 1000"),
    Formula("S2_auc", "mean([S2]) * 1000"),
    Formula("S3_auc", "mean([S3]) * 1000"),
    Formula("S2_max", "max([S2])"),
]


def run(quick: bool, cores: int | None) -> dict[str, sensitivity.SensitivityResult]:
    """Run the four analyses and draw their figures into the working directory."""
    simulator = Simulator(n_workers=cores)
    model = simulator.load(MODEL)
    simulation = Simulation(end=1000, steps=1000)
    conditions = Dimension("S1_0", values={"[S1]": [0.1, 1.0, 10.0]})
    parameters = sampling.parameters_of(model)
    bounds = {pid: sampling.Uniform(relative=0.15) for pid in parameters}
    n = 32 if quick else 1024

    results = {}
    local = sampling.local(parameters, 0.01, model=model)
    results["local"] = sensitivity.local(simulator.run(model, Scan(simulation, [conditions, local]), OBSERVABLES))
    designs = {
        "sobol": (sampling.sobol(bounds, n, seed=1, model=model), sensitivity.sobol),
        "fast": (sampling.fast(bounds, max(n, 65), seed=1, model=model), sensitivity.fast),
        "morris": (sampling.morris(bounds, 10 if quick else 100, seed=1, model=model), sensitivity.morris),
    }
    for name, (design, analysis) in designs.items():
        results[name] = analysis(simulator.run(model, Scan(simulation, [conditions, design]), OBSERVABLES))
    sensitivity.plot_heatmap(results["local"], "normalized", S1_0=1, path=Path("local.png"), dpi=72 if quick else 300)
    sensitivity.plot_indices(results["sobol"], "S2_auc", S1_0=1, path=Path("sobol_S2_auc.png"))
    sensitivity.plot_morris(results["morris"], "S2_auc", S1_0=1, path=Path("morris_S2_auc.png"))
    for name, result in results.items():
        print(name, np.round(result.index(list(result.ds.data_vars)[0].rsplit(".", 1)[1]).sel(S1_0=1).values, 3).tolist())
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small designs and figures, for the tests")
    parser.add_argument("--cores", type=int, default=None, help="the workers of the simulator, all from 256 points by default")
    options = parser.parse_args()
    run(options.quick, options.cores)
```

Adapt what does not run as written (e.g. the selection of the condition `S1_0=1` is the label 1 of the dimension `S1_0`, whose labels are `0, 1, 2`; the time of the maximum of S2 is a `Custom` observable in phase 2 of the scan core and can be left out or written as `Formula` if the reductions allow it); keep the four analyses, the conditions dimension and the three figures. Keep the arguments `--quick` and `--cores=1` of `tests/examples/test_example_scripts.py`.

In `examples/model_sensitivity.py`, compute the local indices of the local design with `sensitivity.local(res_diff_scan, observables=[...])` of a scalar observable (e.g. `Formula("x_max", "max([X])")` added to the runs) and print them; keep the figures.

- [ ] **Step 3: The docs**

Rewrite `docs/sensitivity.md` (its `python` blocks run in the docs test; add `"sensitivity.md"` to `PAGES`): an introduction (an analysis is a design, a run and the indices on the result; the indices per label of the other dimensions and per time point), a section per analysis with a runnable block on the Ishigami model written inline as antimony through `sbmlutils`/`antimony` if available in the docs environment, else on the packaged `REPRESSILATOR_SBML` with two or three parameters (local: `sampling.local` + `sensitivity.local`; Sobol: `sampling.sobol(..., n=64)` + `sensitivity.sobol`, `s.index("ST")`; FAST and Morris in one block), the `SensitivityResult` (variables, `index`, `to_dataframe`, `classify`, netCDF), the plots (`plot_heatmap`, `plot_indices`, `plot_morris`), and a pointer to [Sampling and uncertainty](sampling.md) for the designs and the uncertainty analysis. Keep every block short (a few seconds).

Navigation: replace the nav entries of the deleted API pages by `{ "indices" = "api/sensitivity.indices.md" },` and `{ "result" = "api/sensitivity.result.md" },` (keep `uncertainty`, `classification`, `plots`); create the two API pages (`::: sbmlsim.sensitivity.indices`, `::: sbmlsim.sensitivity.result`) and delete the old ones.

- [ ] **Step 4: CLAUDE.md and the README of the examples**

In `CLAUDE.md`, replace the paragraph of `sensitivity/` with: "**`sensitivity/` - sensitivity analysis.** The analyses are functions of a `ScanResult` whose scan has a design of `sbmlsim.simulation.sampling`: `local` (central differences at the reference, `raw` and `normalized`), `sobol` (`S1`, `ST`, `S2` with intervals), `fast` and `morris` (`indices.py`) read the record of the design from the result (`design_of`), recreate the unit cube of a SALib design (`unit_cube`) and compute the indices of every observable for every label of the other dimensions and every time point of a timecourse on a grid (a ragged timecourse raises); `SensitivityResult` (`result.py`) holds them as `<observable>.<index>` over `(parameter, *dims, [time])` with units and netCDF, `index(name)` stacks the scalar observables, `classify` applies the classification of the IPCS (`classification.py`); `plots.py` draws `plot_heatmap`, `plot_indices` and `plot_morris`, `uncertainty.py` the bands and distributions of a scan over draws. There is no simulation loop and no pool in the package: every simulation runs through `Simulator.run`." Remove the sentence of phase 1 which says the analyses are rebuilt in phase 2, and in the conventions the note that `sensitivity/analysis.py` creates `process_context().Pool(...)` (only `scripts/petab_benchmark.py` remains). `examples/README.md`: the row of `examples/sensitivity/` describes the four analyses on the scan core.

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest -q -n 0 tests/docs tests/examples/test_example_scripts.py -k "sensitivity or docs"`, `uv run zensical build --clean` (no warnings), `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`, and the example `uv run python -m examples.sensitivity.sensitivity_example --quick --cores=1` from a scratch directory.

```bash
git add src/sbmlsim/sensitivity examples docs/sensitivity.md docs/api zensical.toml tests CLAUDE.md
git commit -m "The sensitivity analyses run on the scan core and the old analysis classes are gone" -m "SensitivitySimulation with its simulate(r, changes), the five analysis classes, their pool, their pickle cache and their files are replaced by a design of the sampler, Simulator.run and the indices on the result. The example of the simple chain runs the four analyses with observables and a dimension of conditions, docs/sensitivity.md explains them with code the docs test runs, and the navigation, the API reference and CLAUDE.md follow."
```

(Add files by path; never add an untracked file you did not create.)

---

### Task 6: Verification and the pull request

**Files:** none new; the pull request.

- [ ] **Step 1: The whole suite and the checks**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`, `uv run zensical build --clean`, and `rg -n "SensitivitySimulation|SensitivityAnalysis" src tests examples docs --glob '!docs/superpowers/**'` (no hit).

- [ ] **Step 2: The pull request**

Push and create the pull request with `gh-axi`, base `develop` (after #258 is merged and the branch rebased onto `develop`; until then, base `analyses-phase1` with a note), title "Sensitivity analyses on the result of a scan (#249, analyses phase 2)". The body describes, without any agent attribution: the four analyses and the `SensitivityResult`, the plots, the removal of the old classes, the decisions of this plan, and the verification (Ishigami, power law, netCDF, failed points). Wait for the checks.
