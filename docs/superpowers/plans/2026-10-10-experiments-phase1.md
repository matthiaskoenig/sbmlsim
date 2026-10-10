# Observables in experiments and Data as labelled arrays, implementation plan (experiments phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A simulation experiment declares observables (`observables()`), every task computes the observables its data read, and `Data.get_data` returns labelled arrays which keep the dimensions of a scan, so nothing downstream guesses which point of a scan it holds.

**Architecture:** The scan core learns to keep selections next to observables (`keep` names a selection of the model). `Data` resolves to an `xarray.DataArray` (dims, coords, `attrs["units"]`), with `sel=` by label, dataset columns over a dimension `row` and function data broadcast by dimension name; callers get a pint quantity through `to_quantity`. The figures draw one line per curve from the broadcast arrays and raise for a dimension of a scan which is not selected (`over=` is phase 2). The experiment classifies the index of every task `Data` (observable, coordinate, `time`, selection) at `initialize()` and runs a task with the observables and selections it needs.

**Not in this phase:** `over=`, colours, legends and `Plot.band` (phase 2, with the figures of the glucose example, the HCTZ observables and their figure of cmax and AUC, and the figures of `examples/experiment_scans.py`), the fits of scalar observables (phase 3).

**Tech Stack:** Python 3.13/3.14, xarray, numpy, pint, pandas, libroadrunner, pkpdutils (PK observables), matplotlib and plotly (figures), pytest with xdist and `filterwarnings = error`, ruff, ty, zensical.

**Spec:** `docs/superpowers/specs/2026-10-10-experiments-design.md` (phase 1: the sections "Observables in an experiment" and "Data as labelled arrays", the padding part of "Figures over scan dimensions", the callers of these).

## Global Constraints

- Python >= 3.13; no new runtime dependency.
- `ty check` stays at zero diagnostics (`error-on-warning = true`); suppress only with a rule specific `# ty: ignore[rule]`, never `# type: ignore`.
- Every module, class and function of the package has full type annotations and a google style docstring (ruff `D`); `tests/` and `examples/` are exempt from the docstring rules.
- A subclass marks every overridden method with `typing.override`; a `SimulationExperiment` defines its methods in the order `datasets`, `models`, `simulations`, `observables`, `tasks`, `data`, `fit_mappings`, `figures`.
- Library code logs with `logging.getLogger(__name__)` and lazy `%s` formatting, never prints, never calls `plt.show()`.
- `filterwarnings = error`: fix the cause of a warning, never filter it.
- Never use the em dash character anywhere (code, docs, commits). Markdown has no hard line wraps.
- Commit messages are full sentences with a body, no conventional prefixes, no attribution lines of any kind (no Co-Authored-By, no "Generated with").
- Add files by path (`git add <paths>`), never `git add -A`, `git add .` or a directory add: an untracked `docs/README.md` belongs to someone else and must never be committed.
- Do not edit `CHANGELOG.md` or release notes.
- No API compatibility: `Data.get_data` returns `xr.DataArray` instead of a quantity; `first_curve` is removed.

## Review Focus

1. A ragged result (`(*dims, _point)`, padded with `NaN`): `sel` picks one simulation of time and y alike, and a curve of it is drawn without the padding. Test: Task 2 `test_sel_of_a_ragged_result_picks_one_simulation`, Task 3 `test_a_curve_of_a_ragged_point_drops_the_padding`.
2. A function of a dataset column and a task timecourse, e.g. `x / max(d)`: `max(d)` reduces over the rows of the dataset, not over the time of `x`. Test: Task 2 `test_a_reduction_of_a_dataset_runs_over_its_rows`.
3. The same `sel` given to the time and the y of a curve of a scan on a common grid, where the time has no dimension of the scan: the time is constant along it and the dimension is skipped, while a dimension the result has not raises. Test: Task 2 `test_sel_skips_a_dimension_of_the_scan_the_data_has_not`.
4. A task whose data read only values per simulation while `add_selections_data(["time", ...])` registered `time` for every task: the run keeps no timecourse and does not fail. Test: Task 4 `test_a_task_of_values_per_simulation_runs_with_time_registered`.
5. `to_units` on an array keeps its dimensions and coordinates and sets `attrs["units"]`; an incompatible unit raises `DimensionalityError`. Test: Task 2 `test_to_units_keeps_the_dimensions`.

---

### Task 1: `keep` names selections next to observables

**Files:**
- Modify: `src/sbmlsim/simulator/observables.py` (`compile_observables`, `_keep`, a new `_kept_selections`)
- Modify: `docs/observables.md` (section "keep and memory")
- Test: `tests/simulator/test_observables.py`, `tests/simulator/test_simulator_observables.py`

**Interfaces:**
- Consumes: `compile_observables(observables, model, *, keep=None, plans=())`, `ObservableGraph` (`nodes`, `selections`, `kinds`, `units`, `keep`, `outputs`, `doses`), `RoadrunnerSBMLModel.has_selection`.
- Produces: `Simulator.run(model, scan, observables, keep=[..., "<selection>"])` gives the selection as a timecourse of its own name in the unit of the model next to the observables. Task 4 relies on it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/simulator/test_observables.py`:

```python
def test_keep_names_selections_next_to_observables(
    pk_model: RoadrunnerSBMLModel,
) -> None:
    graph = compile_observables(
        [Formula("cmax", "max([C])")], pk_model, keep=["cmax", "[C]", "C"]
    )
    assert graph.keep == ("cmax", "[C]", "C")
    assert graph.kinds["[C]"] is TIMECOURSE and graph.kinds["C"] is TIMECOURSE
    assert set(graph.selections) == {"[C]", "C"}
    assert graph.units["[C]"] == (pk_model.uinfo.get("[C]", "") or "")
    assert graph.timecourses == ("[C]", "C") and graph.scalars == ("cmax",)
    with pytest.raises(ValueError, match="neither an observable of the run nor a selection"):
        compile_observables(
            [Formula("cmax", "max([C])")], pk_model, keep=["cmax", "nope"]
        )
```

Append to `tests/simulator/test_simulator_observables.py`:

```python
def test_a_run_keeps_selections_next_to_observables(pk_sbml: str) -> None:
    res = Simulator().run(
        pk_sbml, dose_scan(), [Formula("cmax", "max([C])")], keep=["cmax", "[C]"]
    )
    assert set(res.ds.data_vars) == {"cmax", "[C]"}
    assert res["[C]"].dims == ("dose", "time")
    assert res["cmax"].dims == ("dose",)
    np.testing.assert_allclose(res["cmax"].values, res["[C]"].max("time").values)
    model = Simulator().load(pk_sbml)
    assert res.units["[C]"] == (model.uinfo.get("[C]", "") or "")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/simulator/test_observables.py tests/simulator/test_simulator_observables.py -k "selections_next_to"`
Expected: FAIL with `ValueError: 'keep' names '[C]', which is no observable of the run`.

- [ ] **Step 3: Implement**

In `src/sbmlsim/simulator/observables.py`, in `compile_observables` replace the block from `every = tuple(...)` to the `return ObservableGraph(...)` with:

```python
    every = tuple(o for name in definitions for o in outputs[name])
    kept_selections = _kept_selections(keep, every, model)
    for selection in kept_selections:
        kinds[selection] = TIMECOURSE
        units[selection] = uinfo.get(selection, "") or ""
        selections[selection] = None
    every = (*every, *kept_selections)
    kept = _keep(keep, every, outputs)
    needed = _needed(kept, definitions)
    nodes = tuple(node for name, node in compiled.items() if name in needed)
    read = {s for name in needed for s in definitions[name].reads}
    read.update(kept_selections)
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
```

Add after `_keep`:

```python
def _kept_selections(
    keep: Sequence[str] | None,
    outputs: Sequence[str],
    model: RoadrunnerSBMLModel,
) -> tuple[str, ...]:
    """Get the selections of the model which `keep` names next to the observables.

    A kept selection is an output of the graph, a timecourse of its own name,
    as in a run without observables; an observable shares no name with a
    selection, so the name is unambiguous.
    """
    if keep is None or isinstance(keep, str):
        return ()
    return tuple(
        dict.fromkeys(
            k for k in keep if k not in outputs and k != TIME and model.has_selection(k)
        )
    )
```

In `_keep`, change the message of the unknown key to:

```python
            raise ValueError(
                f"'keep' names '{key}', which is neither an observable of the run "
                f"nor a selection of the model: {list(outputs)}."
            )
```

Update the docstring of `compile_observables`: the description of `keep` becomes "the outputs of the result, every output by default; the id of a PK observable keeps all its parameters, a selection of the model is kept as a timecourse of its own name next to the observables." Update the existing test `test_keep_and_the_observables_it_needs` only if its `match="'nope'"` no longer matches (it still does).

In `docs/observables.md`, section "keep and memory", append this paragraph:

```markdown
`keep` may also name selections of the model next to the observables, e.g. `keep=["cmax", "[C]"]`: a kept selection is a timecourse of its own name in the unit of the model, as in a run without observables, so one run gives the values per simulation and the timecourses a figure draws.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/simulator/test_observables.py tests/simulator/test_simulator_observables.py`
Expected: PASS.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/simulator/observables.py docs/observables.md tests/simulator/test_observables.py tests/simulator/test_simulator_observables.py
git commit -m "A run with observables keeps the selections which keep names" -m "keep may name a selection of the model next to the observables; it is an output of the graph, a timecourse of its own name in the unit of the model, so one run gives the values per simulation and the timecourses. A key which is neither an observable nor a selection still raises."
```

---

### Task 2: `Data` resolves to labelled arrays

**Files:**
- Modify: `src/sbmlsim/data.py` (`ROW`, `REDUCED_DIMS`, `to_quantity`, `evaluate_function`, helpers, `Data.__init__(sel=)`, `Data.get_data`, `Data.to_dict`)
- Modify: `src/sbmlsim/fit/objects.py:1141` and `:1191` (the counts and `FitData.get_data`)
- Modify: `src/sbmlsim/plot/plotting.py:1573` (the counts of `Plot.add_data`)
- Modify: `src/sbmlsim/plot/serialization_matplotlib.py:146-168,255-264` and `src/sbmlsim/plot/serialization_plotly.py:85-96` (`.magnitude` becomes `.values`)
- Modify: `docs/data.md` (section "Resolving data" and a new section "Selecting points")
- Test: `tests/test_data_function.py` (rewritten), `tests/test_data_arrays.py` (new), `tests/experiment/test_experiment_run.py` (updated)

**Interfaces:**
- Consumes: `ScanResult` (`ds`, `units`, `__getitem__`, `__contains__`), `DataSet` (`uinfo`), `reduce_formula`, `compile_formula`, `sbmlsim.result.scan.TIME`, `POINT`.
- Produces:
  - `sbmlsim.data.ROW = "row"`, `REDUCED_DIMS = (TIME, POINT, ROW)`.
  - `to_quantity(array: xr.DataArray, ureg: UnitRegistry) -> Quantity`.
  - `evaluate_function(formula: str, variables: Mapping[str, xr.DataArray | float], ureg: UnitRegistry) -> xr.DataArray`.
  - `Data(index, task=None, dataset=None, function=None, variables=None, parameters=None, sid=None, sel=None)`, attribute `sel: dict[str, Any]`.
  - `Data.get_data(experiment, to_units=None) -> xr.DataArray` named by the sid, with `attrs["units"]` (`None` for labels).

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_data_function.py` by:

```python
"""The formula of a Data of type FUNCTION is the math of PEtab on labelled arrays."""

import numpy as np
import pytest
import xarray as xr

from sbmlsim.data import evaluate_function
from sbmlsim.units import ureg


def _tc(values: list[float], unit: str | None = "dimensionless") -> xr.DataArray:
    return xr.DataArray(np.array(values), dims=("time",), attrs={"units": unit})


def test_a_ratio_of_quantities_keeps_the_units() -> None:
    x = _tc([1.0, 2.0], "mmol/l")
    y = _tc([2.0, 4.0], "mmol/l")
    ratio = evaluate_function("x / y", {"x": x, "y": y}, ureg)
    assert ratio.dims == ("time",)
    q = ureg.Quantity(ratio.values, ratio.attrs["units"])
    np.testing.assert_allclose(q.to("dimensionless").magnitude, [0.5, 0.5])
    v = xr.DataArray(2.0, attrs={"units": "l"})
    amount = evaluate_function("x * v", {"x": x, "v": v}, ureg)
    q = ureg.Quantity(amount.values, amount.attrs["units"])
    assert q.to("mmol").magnitude.tolist() == pytest.approx([2.0, 4.0])


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
        "Y": _tc([1.0, 2.0, 4.0], None),
        "Z": _tc([1.0, 1.0, 4.0], None),
        "Ymax": _tc([1.0, 1.0, 1.0], None),
    }
    np.testing.assert_allclose(evaluate_function(formula, variables, ureg).values, expected)


def test_a_reduction_ignores_the_padding() -> None:
    y = xr.DataArray([1.0, 2.0, np.nan], dims=("_point",), attrs={"units": None})
    np.testing.assert_allclose(
        evaluate_function("Y/max(Y)", {"Y": y}, ureg).values, [0.5, 1.0, np.nan]
    )


def test_a_reduction_of_quantities_keeps_the_units() -> None:
    y = _tc([1.0, 2.0, 4.0], "mmol/l")
    shifted = evaluate_function("Y - min(Y)", {"Y": y}, ureg)
    assert ureg.Unit(shifted.attrs["units"]) == ureg.Unit("mmol/l")


def test_a_formula_of_parameters_is_a_number() -> None:
    value = evaluate_function("2 * k", {"k": 3.0}, ureg)
    assert value.dims == () and float(value) == pytest.approx(6.0)
    assert value.attrs["units"] == "dimensionless"


@pytest.mark.parametrize("formula", ["Y +", "max(Y", "foo(Y)"])
def test_invalid_math_is_reported(formula: str) -> None:
    with pytest.raises(ValueError):
        evaluate_function(formula, {"Y": _tc([1.0])}, ureg)


def test_an_unknown_identifier_is_reported() -> None:
    with pytest.raises(ValueError, match="W"):
        evaluate_function("Y / W", {"Y": _tc([1.0])}, ureg)


def test_a_reduction_of_a_scan_is_per_simulation_by_name() -> None:
    # the time is first here: the reduction finds it by its name
    y = xr.DataArray(
        np.array([[1.0, 1.0], [2.0, 1.0], [4.0, 2.0]]),
        dims=("time", "dose"),
        attrs={"units": None},
    )
    normalized = evaluate_function("Y/max(Y)", {"Y": y}, ureg)
    assert set(normalized.dims) == {"time", "dose"}
    np.testing.assert_allclose(
        normalized.transpose("dose", "time").values, [[0.25, 0.5, 1.0], [0.5, 0.5, 1.0]]
    )
    peak = evaluate_function("max(Y)", {"Y": y}, ureg)
    assert peak.dims == ("dose",)
    np.testing.assert_allclose(peak.values, [4.0, 2.0])


def test_arrays_broadcast_by_dimension_name() -> None:
    y = xr.DataArray(np.ones((2, 3)), dims=("dose", "time"), attrs={"units": "mM"})
    dose = xr.DataArray([1.0, 2.0], dims=("dose",), attrs={"units": "mg"})
    per_dose = evaluate_function("y / d", {"y": y, "d": dose}, ureg)
    assert per_dose.dims == ("dose", "time")
    np.testing.assert_allclose(per_dose.values[:, 0], [1.0, 0.5])


def test_different_coordinates_of_a_dimension_raise() -> None:
    a = xr.DataArray([1.0, 2.0], dims=("time",), coords={"time": [0.0, 1.0]}, attrs={"units": None})
    b = xr.DataArray([1.0, 2.0], dims=("time",), coords={"time": [0.0, 2.0]}, attrs={"units": None})
    with pytest.raises(ValueError, match="coordinates"):
        evaluate_function("a + b", {"a": a, "b": b}, ureg)


@pytest.mark.parametrize("formula", ["mean(Y)", "at(Y, 1)"])
def test_mean_and_at_are_no_reductions_of_data(formula: str) -> None:
    with pytest.raises(ValueError, match="observable"):
        evaluate_function(formula, {"Y": _tc([1.0, 2.0])}, ureg)
```

Create `tests/test_data_arrays.py`:

```python
"""A Data resolves to a labelled array with the dimensions of its source."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sbmlsim.data import ROW, Data, DataSet, to_quantity
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task
from sbmlsim.units import DimensionalityError


class ArrayExperiment(SimulationExperiment):
    """A scan of the initial amount of X on a grid, a ragged scan and a dataset."""

    def datasets(self) -> dict:
        df = pd.DataFrame(
            {"group": ["a", "b", "b"], "time": [0.0, 5.0, 10.0], "X": [1.0, 4.0, 2.0]}
        )
        return {
            "ds": DataSet.from_df(
                df, udict={"time": "second", "X": "dimensionless"}, ureg=self.ureg
            )
        }

    def models(self) -> dict:
        return {"m": AbstractModel(source=REPRESSILATOR_SBML)}

    def simulations(self) -> dict:
        dim = Dimension("d", values={"X": np.array([1.0, 2.0, 30.0])}, labels=["lo", "mid", "hi"])
        return {
            "grid": Scan(Simulation(end=10, steps=10), [dim]),
            "ragged": Scan(Simulation(end=10), [dim]),
        }

    def tasks(self) -> dict:
        return {"grid": Task(model="m", simulation="grid"), "ragged": Task(model="m", simulation="ragged")}

    def data(self) -> dict:
        self.add_selections_data(["time", "[Y]"])
        return {}


@pytest.fixture(scope="module")
def experiment() -> SimulationExperiment:
    runner = ExperimentRunner(
        experiment_classes=[ArrayExperiment],
        simulator=Simulator(),
        base_path=Path("."),
        data_path=Path("."),
    )
    experiment = runner.experiments["ArrayExperiment"]
    experiment.run(runner.simulator)
    return experiment


def test_task_data_keeps_the_dimensions_and_coordinates(experiment: SimulationExperiment) -> None:
    y = Data("[Y]", task="grid").get_data(experiment)
    assert isinstance(y, xr.DataArray)
    assert y.dims == ("d", "time") and y.shape == (3, 11)
    assert y.name == "grid__Y"
    assert y["d"].values.tolist() == ["lo", "mid", "hi"]
    np.testing.assert_allclose(y["X"].values, [1.0, 2.0, 30.0])
    assert y.attrs["units"] is not None
    x = Data("X", task="grid").get_data(experiment)
    assert x.dims == ("d",)
    np.testing.assert_allclose(x.values, [1.0, 2.0, 30.0])
    labels = Data("d", task="grid").get_data(experiment)
    assert labels.values.tolist() == ["lo", "mid", "hi"] and labels.attrs["units"] is None


def test_sel_by_label(experiment: SimulationExperiment) -> None:
    one = Data("[Y]", task="grid", sel={"d": "mid"}).get_data(experiment)
    assert one.dims == ("time",)
    two = Data("[Y]", task="grid", sel={"d": ["lo", "hi"]}).get_data(experiment)
    assert two.dims == ("d", "time") and two["d"].values.tolist() == ["lo", "hi"]
    with pytest.raises(ValueError, match=r"dimension 'nope'.*'time'"):
        Data("[Y]", task="grid", sel={"nope": 0}).get_data(experiment)
    with pytest.raises(ValueError, match=r"\['zz'\].*'d'.*\['lo', 'mid', 'hi'\]"):
        Data("[Y]", task="grid", sel={"d": "zz"}).get_data(experiment)


def test_sel_skips_a_dimension_of_the_scan_the_data_has_not(experiment: SimulationExperiment) -> None:
    time = Data("time", task="grid", sel={"d": "mid"}).get_data(experiment)
    assert time.dims == ("time",)
    np.testing.assert_allclose(time.values, np.linspace(0, 10, 11))


def test_sel_of_a_ragged_result_picks_one_simulation(experiment: SimulationExperiment) -> None:
    time = Data("time", task="ragged", sel={"d": "lo"}).get_data(experiment)
    y = Data("[Y]", task="ragged", sel={"d": "lo"}).get_data(experiment)
    assert time.dims == y.dims == ("_point",)
    np.testing.assert_array_equal(np.isnan(time.values), np.isnan(y.values))
    native = Simulator().simulate(
        experiment._models["m"], Simulation(end=10, preinit_changes={"X": 1.0})
    )
    keep = ~np.isnan(time.values)
    np.testing.assert_allclose(time.values[keep], native.time)
    np.testing.assert_allclose(y.values[keep], native["[Y]"])


def test_dataset_data_is_over_rows(experiment: SimulationExperiment) -> None:
    x = Data("X", dataset="ds").get_data(experiment)
    assert x.dims == (ROW,) and x[ROW].values.tolist() == [0, 1, 2]
    assert x.attrs["units"] == "dimensionless"
    b = Data("X", dataset="ds", sel={"group": "b"}).get_data(experiment)
    assert b.values.tolist() == [4.0, 2.0] and b[ROW].values.tolist() == [1, 2]
    a = Data("X", dataset="ds", sel={"group": ["a"]}).get_data(experiment)
    assert a.dims == (ROW,) and a.values.tolist() == [1.0]
    with pytest.raises(ValueError, match=r"column 'nope'"):
        Data("X", dataset="ds", sel={"nope": 1}).get_data(experiment)
    with pytest.raises(ValueError, match="no row"):
        Data("X", dataset="ds", sel={"group": "c"}).get_data(experiment)


def test_a_reduction_of_a_dataset_runs_over_its_rows(experiment: SimulationExperiment) -> None:
    ratio = Data(
        "ratio",
        function="x / max(d)",
        variables={"x": Data("[X]", task="grid", sel={"d": "lo"}), "d": Data("X", dataset="ds")},
    ).get_data(experiment)
    x = Data("[X]", task="grid", sel={"d": "lo"}).get_data(experiment)
    assert ratio.dims == ("time",)
    np.testing.assert_allclose(ratio.values, x.values / 4.0)


def test_to_units_keeps_the_dimensions(experiment: SimulationExperiment) -> None:
    time = Data("time", task="grid").get_data(experiment, to_units="minute")
    assert time.dims == ("time",) and time.attrs["units"] == "minute"
    np.testing.assert_allclose(time.values, np.linspace(0, 10, 11) / 60.0)
    with pytest.raises(DimensionalityError):
        Data("time", task="grid").get_data(experiment, to_units="mol")


def test_to_quantity(experiment: SimulationExperiment) -> None:
    time = Data("time", task="grid").get_data(experiment)
    quantity = to_quantity(time, experiment.ureg)
    assert quantity.to("second").magnitude.tolist() == pytest.approx(np.linspace(0, 10, 11).tolist())
    with pytest.raises(ValueError, match="labels"):
        to_quantity(Data("d", task="grid").get_data(experiment), experiment.ureg)


def test_the_selection_is_serialized() -> None:
    assert Data("[Y]", task="grid", sel={"d": "lo"}).to_dict()["sel"] == {"d": "lo"}
    assert Data("[Y]", task="grid").to_dict()["sel"] is None
```

In `tests/experiment/test_experiment_run.py` update the existing tests to the arrays:

- `test_the_data_of_a_scan_has_the_time_last`: `np.shape(y.magnitude)` becomes `y.shape`, `str(y.units) == "dimensionless"` becomes `y.attrs["units"] == "dimensionless"`, every `.magnitude` becomes `.values`.
- `test_the_data_of_a_task_is_in_the_registry_of_the_experiment`: replace `assert x._REGISTRY is ureg` by `assert experiment.ureg is ureg` and `to_quantity(x, experiment.ureg)._REGISTRY is ureg` (import `to_quantity` from `sbmlsim.data`); `ratio.magnitude` and `x.magnitude` become `.values`.
- `test_the_data_of_a_ragged_scan_has_its_points_last`: every `.magnitude` becomes `.values` (the `first_curve` call stays until Task 3).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/test_data_function.py tests/test_data_arrays.py`
Expected: FAIL with `ImportError: cannot import name 'ROW'` and `TypeError: evaluate_function() takes 2 positional arguments but 3 were given`.

- [ ] **Step 3: Implement `data.py`**

Imports of `src/sbmlsim/data.py`: add `import xarray as xr` and `from sbmlsim.result.scan import POINT, TIME`; keep `compile_formula` and `reduce_formula`, drop `evaluate_reduced` (no longer used here).

Replace `evaluate_function` by:

```python
#: the dimension of the rows of a dataset
ROW = "row"

#: the dimensions a reduction of data runs along: the first one an array has
REDUCED_DIMS = (TIME, POINT, ROW)


def to_quantity(array: xr.DataArray, ureg: UnitRegistry) -> Quantity:
    """Get the values of data as a quantity.

    Args:
        array: the values of a `Data`, see `Data.get_data`.
        ureg: the registry of the quantity, e.g. the one of the experiment.

    Returns:
        The values with the unit of `attrs["units"]`.

    Raises:
        ValueError: if the array holds labels or has no unit.
    """
    if array.dtype.kind not in "fiub":
        raise ValueError(
            f"'{array.name}' holds labels, not values, so it is no quantity."
        )
    unit = array.attrs.get("units")
    if unit is None:
        raise ValueError(f"'{array.name}' has no unit, so it is no quantity.")
    return ureg.Quantity(np.asarray(array.values, dtype=float), unit)


def evaluate_function(
    formula: str,
    variables: Mapping[str, xr.DataArray | float],
    ureg: UnitRegistry,
) -> xr.DataArray:
    """Evaluate the formula of a `Data` of type FUNCTION on its data.

    The formula is the math of PEtab, see `sbmlsim.simulator.formula`. The
    arrays are broadcast by the names of their dimensions, so `y / dose` over
    `(dose, time)` and `(dose,)` needs no reshaping; two arrays must have the
    same coordinates of a dimension they share. `max` and `min` of a single
    argument reduce it along its time (`time`, or `_point` of a ragged
    result) or, without one, along the rows of a dataset, ignoring `NaN`, the
    padding of a scan; the other dimensions stay, so `Y/max(Y)` normalizes
    every simulation of a scan to its own maximum. With two or more arguments
    they are the elementwise maximum and minimum of PEtab. `mean` and `at`
    need the time points of a simulation, they are reductions of the
    observables of a scan. An array without a unit is evaluated as plain
    numbers, a formula without units is dimensionless.

    Args:
        formula: the formula.
        variables: the arrays of the data and the numbers of the parameters,
            by the identifiers of the formula.
        ureg: the registry of the units.

    Returns:
        The value of the formula over the broadcast dimensions, with its unit.

    Raises:
        ValueError: if the formula is not valid math, reads an identifier
            which is not a variable, uses `mean` or `at`, or combines arrays
            with different coordinates of one dimension.
    """
    reduced = reduce_formula(formula)
    scope: dict[str, xr.DataArray | float] = dict(variables)
    for reduction in reduced.reductions:
        if reduction.function in ("mean", "at"):
            raise ValueError(
                f"'{reduction.function}' in the formula '{formula}' needs the time "
                f"points of a simulation: it is a reduction of the observables of a "
                f"scan, see sbmlsim.simulation.observables."
            )
        x = _evaluate_part(reduction.arguments[0], scope, formula, ureg)
        scope[reduction.symbol] = _extreme(reduction.function, x)
    return _evaluate_part(reduced.outer, scope, formula, ureg)


def _evaluate_part(
    part: str,
    scope: Mapping[str, xr.DataArray | float],
    formula: str,
    ureg: UnitRegistry,
) -> xr.DataArray:
    """Evaluate a part of a formula without reductions, broadcasting by name.

    Raises:
        ValueError: if the part reads an identifier without a value, or two
            arrays have different coordinates of one dimension.
    """
    compiled = compile_formula(part)
    missing = [symbol for symbol in compiled.symbols if symbol not in scope]
    if missing:
        raise ValueError(
            f"The formula '{formula}' reads {missing}, which have no values."
        )
    arrays = {
        symbol: value
        for symbol in compiled.symbols
        if isinstance(value := scope[symbol], xr.DataArray)
    }
    try:
        aligned = xr.align(*arrays.values(), join="exact") if arrays else ()
    except ValueError as err:
        raise ValueError(
            f"The data of the formula '{formula}' has different coordinates of a "
            f"dimension: {err}"
        ) from err
    broadcast = (
        dict(zip(arrays, xr.broadcast(*aligned), strict=True)) if arrays else {}
    )
    arguments = [
        _argument(broadcast[symbol], ureg) if symbol in broadcast else scope[symbol]
        for symbol in compiled.symbols
    ]
    value = compiled.apply(arguments)
    if isinstance(value, Quantity):
        magnitude, unit = np.asarray(value.magnitude, dtype=float), str(value.units)
    else:
        magnitude, unit = np.asarray(value, dtype=float), "dimensionless"
    template = next(iter(broadcast.values()), None)
    if template is None:
        return xr.DataArray(magnitude, attrs={"units": unit})
    return xr.DataArray(
        np.broadcast_to(magnitude, template.shape).copy(),
        dims=template.dims,
        coords=template.coords,
        attrs={"units": unit},
    )


def _argument(array: xr.DataArray, ureg: UnitRegistry) -> Any:
    """Get the value of an array for a formula: a quantity, plain numbers without a unit."""
    if array.attrs.get("units") is None:
        return np.asarray(array.values, dtype=float)
    return to_quantity(array, ureg)


def _extreme(function: str, x: xr.DataArray) -> xr.DataArray:
    """Reduce data to its largest or smallest value along its reduced dimension.

    The dimension is the first of `REDUCED_DIMS` the array has; an array
    without one is a value per simulation and stays. `NaN`, the padding of a
    ragged result, is ignored, and only `NaN` gives `NaN`.
    """
    dim = next((d for d in REDUCED_DIMS if d in x.dims), None)
    if dim is None:
        return x
    ufunc = np.fmax if function == "max" else np.fmin
    reduced = x.reduce(lambda values, axis: ufunc.reduce(values, axis=axis), dim=dim)
    reduced.attrs = dict(x.attrs)
    return reduced


def _select(
    array: xr.DataArray,
    sel: Mapping[str, Any],
    data: Data,
    source: xr.Dataset | xr.DataArray,
) -> xr.DataArray:
    """Select the labels of `sel` from data, see `Data`.

    A dimension of the source (the result of a task, or the array itself for
    a function) which the array has not is skipped: the array is constant
    along it. A dimension without labels is selected by position.

    Raises:
        ValueError: if a dimension is not one of the source, or a label is not
            one of its dimension.
    """
    by_label: dict[str, Any] = {}
    by_position: dict[str, Any] = {}
    for dim, label in sel.items():
        if dim not in source.sizes:
            raise ValueError(
                f"{data} selects the dimension '{dim}', which its source has not: "
                f"{[str(d) for d in source.sizes]}."
            )
        labelled = dim in source.coords
        labels = (
            source[dim].values.tolist() if labelled else list(range(source.sizes[dim]))
        )
        wanted = list(label) if isinstance(label, list | tuple | np.ndarray) else [label]
        unknown = [w for w in wanted if w not in labels]
        if unknown:
            raise ValueError(
                f"{data} selects {unknown} of the dimension '{dim}', whose labels "
                f"are {labels}."
            )
        if dim in array.dims:
            value = list(label) if isinstance(label, tuple | np.ndarray) else label
            (by_label if labelled else by_position)[dim] = value
    if by_label:
        array = array.sel(by_label)
    if by_position:
        array = array.isel(by_position)
    return array


def _rows(dset: pd.DataFrame, sel: Mapping[str, Any], data: Data) -> pd.DataFrame:
    """Select the rows of a dataset whose columns have the values of `sel`.

    Raises:
        ValueError: if a column is not one of the dataset, or no row is left.
    """
    rows = dset
    for column, value in sel.items():
        if column not in dset.columns:
            raise ValueError(
                f"{data} selects rows by the column '{column}', which the dataset "
                f"has not: {list(dset.columns)}."
            )
        values = list(value) if isinstance(value, list | tuple | np.ndarray) else [value]
        rows = rows[rows[column].isin(values)]
    if sel and rows.empty:
        raise ValueError(f"{data} selects {dict(sel)}, which no row of the dataset has.")
    return rows
```

In `Data.__init__`, add the parameter `sel: Mapping[str, Any] | None = None` after `sid`, document it ("labels of dimensions to select, `{dim: label}` keeps one point and drops the dimension, `{dim: [labels]}` keeps the dimension; for a dataset the values of columns whose rows are kept, see `get_data`; the sid does not depend on it, give `sid` to tell apart two data of one index with different selections"), and set `self.sel: dict[str, Any] = dict(sel) if sel else {}`. In `Data.to_dict` add `"sel": self.sel or None`.

Replace `Data.get_data` by:

```python
    def get_data(
        self,
        experiment: SimulationExperiment,
        to_units: str | None = None,
    ) -> xr.DataArray:
        """Get the values of the data from an experiment which ran.

        The values are a labelled array named by the sid of the data, with the
        unit in `attrs["units"]` (`None` for labels), see `to_quantity`:

        - a task: the variable or coordinate of its `ScanResult` with its
          coordinates, a timecourse over `(*dims, time)` or `(*dims, _point)`
          for a ragged result padded with `NaN`, a value per simulation over
          `(*dims)`; `time` is the time, a changed target or a coordinate of a
          dimension is over its dimension, a dimension id gives its labels;
        - a dataset: the column over the dimension `row`, whose coordinate is
          the index of the dataset;
        - a function: its formula on its variables and parameters, see
          `evaluate_function`.

        `sel` selects labels of the dimensions of a task or a function (a
        dimension of the scan which the data has not is skipped) and the rows
        of a dataset by the values of its columns. `unit` is set to the unit of
        the values.

        Args:
            experiment: the experiment whose datasets and results are read.
            to_units: the unit to convert the values to, their own unit
                without.

        Returns:
            The values.

        Raises:
            KeyError: if the dataset has no column of the index or no unit of
                it, or the result of the task has no variable or coordinate of
                the selection.
            ValueError: if the dataset is no `DataSet`, the result of the task
                is no `ScanResult` or its selection has no unit, a function has
                no formula, or `sel` names a dimension, label or column which
                does not exist.
            DimensionalityError: if the values cannot be converted to
                `to_units`.
        """
        if self.dtype == Data.Types.DATASET:
            array = self._dataset_array(experiment)
        elif self.dtype == Data.Types.TASK:
            array = self._task_array(experiment)
        else:
            array = self._function_array(experiment)
        array.name = self.sid
        self.unit = array.attrs.get("units")
        if to_units is not None:
            try:
                quantity = to_quantity(array, experiment.ureg).to(to_units)
            except DimensionalityError:
                logger.error("Could not convert '%s' to units '%s'.", self, to_units)
                raise
            array = array.copy(data=np.asarray(quantity.magnitude, dtype=float))
            array.attrs = {"units": to_units}
        return array

    def _dataset_array(self, experiment: SimulationExperiment) -> xr.DataArray:
        """Get the column of the dataset over its rows, see `get_data`."""
        if not experiment._datasets:
            experiment._datasets = experiment.datasets()
        dset = experiment._datasets[str(self.dset_id)]
        if not isinstance(dset, DataSet):
            raise ValueError(
                f"DataSet '{self.dset_id}' is not a DataSet, but type '{type(dset)}'"
            )
        if dset.empty:
            logger.error("Adding empty dataset '%s' for '%s'.", dset, self.dset_id)
        uindex = self.index[:-3] if self.index.endswith(("_se", "_sd")) else self.index
        if self.index not in dset.columns:
            error_msg = (
                f"Data column with key '{self.index}' does not exist in dataset: "
                f"'{self.dset_id}'."
            )
            logger.error(error_msg)
            raise KeyError(error_msg)
        try:
            unit = dset.uinfo[uindex]
        except KeyError:
            logger.error(
                "Units missing for key '%s' in dataset: '%s'. Add missing units to "
                "dataset.",
                uindex,
                self.dset_id,
            )
            raise
        rows = _rows(dset, self.sel, self)
        return xr.DataArray(
            np.asarray(rows[self.index].values),
            dims=(ROW,),
            coords={ROW: np.asarray(rows.index.values)},
            attrs={"units": unit},
        )

    def _task_array(self, experiment: SimulationExperiment) -> xr.DataArray:
        """Get the variable or coordinate of the result of the task, see `get_data`."""
        result = experiment.results[str(self.task_id)]
        if not isinstance(result, ScanResult):
            raise ValueError(
                f"The result of the task '{self.task_id}' is no ScanResult: "
                f"{type(result)}."
            )
        if self.selection not in result:
            raise KeyError(
                f"'{self.selection}' is not in the result of the task "
                f"'{self.task_id}', its variables are {result.variables}: add "
                f"it to the selections of the experiment."
            )
        array = result[self.selection]
        unit = result.units.get(self.selection)
        if unit is None and array.dtype.kind in "fiub":
            raise ValueError(
                f"'{self.selection}' of the task '{self.task_id}' has no unit in "
                f"the result."
            )
        array = _select(array, self.sel, self, result.ds)
        return xr.DataArray(
            array.values, dims=array.dims, coords=array.coords, attrs={"units": unit}
        )

    def _function_array(self, experiment: SimulationExperiment) -> xr.DataArray:
        """Evaluate the function on its variables and parameters, see `get_data`."""
        if self.function is None:
            raise ValueError(f"Data '{self}' has no function.")
        variables: dict[str, xr.DataArray | float] = {}
        for key, variable in self.variables.items():
            d = experiment._data[variable] if isinstance(variable, str) else variable
            variables[key] = d.get_data(experiment=experiment)
        variables.update(self.parameters)
        array = evaluate_function(self.function, variables, experiment.ureg)
        return _select(array, self.sel, self, array)
```

Remove the old `# todo: dimensions, data type` comment block above `to_dict` (dimensions are now part of the data).

- [ ] **Step 4: Migrate the callers**

- `src/sbmlsim/fit/objects.py`: import `to_quantity` from `sbmlsim.data`; in `_resolve_count` replace `np.unique(counts.magnitude)` by `np.unique(np.asarray(counts.values))`; in `FitData.get_data` replace `setattr(result, key, d.get_data(self.experiment))` by `setattr(result, key, to_quantity(d.get_data(self.experiment), self.experiment.ureg))`, so `FitDataInitialized` keeps its quantities (phase 3 of the spec changes the fits).
- `src/sbmlsim/plot/plotting.py`: in `Plot.add_data` replace `np.unique(counts.magnitude)` by `np.unique(np.asarray(counts.values))`.
- `src/sbmlsim/plot/serialization_matplotlib.py`: the eight `x.magnitude`, `y.magnitude`, `xerr.magnitude`, `yerr.magnitude`, `yfrom.magnitude`, `yto.magnitude` inside the `first_curve(...)` calls become `.values`.
- `src/sbmlsim/plot/serialization_plotly.py` `_values`: rename `quantity` to `array` and return `first_curve(np.asarray(array.values))`; adapt its docstring ("The values, ...").
- `docs/data.md`, section "Resolving data": the first sentence becomes "`Data.get_data(experiment)` returns the values of the data in a run experiment as a labelled array, an `xarray.DataArray` with the dimensions of its source, their coordinates and the unit in `attrs["units"]`, optionally converted to other units; `sbmlsim.data.to_quantity(array, ureg)` gives the pint quantity:" and the two `print` lines of the block become `print(time.attrs["units"], time.values[:3])` and `print(x.attrs["units"], x.values[:3])`. Append a section:

```markdown
## Selecting points

The data of a task has the dimensions of its scan: a timecourse is over `(*dims, time)`, or `(*dims, _point)` for a ragged result whose simulations keep their own time points padded with `NaN`, a value per simulation is over `(*dims)`, and a changed target is over its dimension. `sel` selects labels: `Data("[X]", task="task_scan", sel={"dose": "high"})` keeps one point and drops the dimension, `sel={"dose": ["low", "high"]}` keeps the dimension with two labels. A dimension of the scan which the data has not, e.g. the dimension of a scan for the time on a common grid, is skipped, so the same `sel` serves the time and the values of a curve; a dimension or label which does not exist raises with the ones which do. The data of a dataset is a column over the dimension `row`, and `sel={"group": "b"}` keeps the rows whose column `group` has the value. A function broadcasts its data by the names of their dimensions, and `max` and `min` of a single argument reduce along the time of a timecourse, or along the rows of a dataset.
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/test_data_function.py tests/test_data_arrays.py tests/test_data.py tests/experiment tests/plot tests/fit -x`
Expected: PASS.

- [ ] **Step 6: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/data.py src/sbmlsim/fit/objects.py src/sbmlsim/plot/plotting.py src/sbmlsim/plot/serialization_matplotlib.py src/sbmlsim/plot/serialization_plotly.py docs/data.md tests/test_data_function.py tests/test_data_arrays.py tests/experiment/test_experiment_run.py
git commit -m "A Data resolves to a labelled array with the dimensions of its source" -m "Data.get_data returns an xarray.DataArray with the dimensions of the scan, their coordinates and the unit in attrs, so nothing downstream guesses which point of a scan it holds. sel selects labels, a dataset column is over the dimension row and selects rows by column values, a function broadcasts by dimension name and reduces max and min along the time or the rows; to_quantity gives the pint quantity the fits and the conversions need."
```

---

### Task 3: A curve draws one line and refuses a dimension of a scan

**Files:**
- Modify: `src/sbmlsim/plot/padding.py` (`first_curve` removed, `line_values` added, module docstring)
- Modify: `src/sbmlsim/plot/serialization_matplotlib.py`, `src/sbmlsim/plot/serialization_plotly.py` (use `line_values`)
- Modify: `src/sbmlsim/plot/plotting.py` (`Plot.add_data(..., sel=None)`)
- Modify: `examples/demo/demo.py` (the curves select one point), `examples/README.md` (the sentence on curves of a scan)
- Test: `tests/plot/test_padding.py` (rewritten), `tests/experiment/test_experiment_run.py`

**Interfaces:**
- Consumes: `Data.get_data` arrays and `Data(sel=...)` (Task 2), `sbmlsim.data.ROW`, `sbmlsim.result.scan.TIME`, `POINT`.
- Produces: `line_values(sid: str, *arrays: xr.DataArray | None) -> tuple[Any, ...]` (the values of a curve as one line, without padding); `Plot.add_data(..., sel: Mapping[str, Any] | None = None)` passes `sel` to every `Data` it builds.

- [ ] **Step 1: Write the failing tests**

Replace `tests/plot/test_padding.py` by:

```python
"""A curve is one line of broadcast arrays, without the padding of a ragged result."""

import numpy as np
import pytest
import xarray as xr

from sbmlsim.plot.padding import line_values, without_padding


def _time(values: list[float]) -> xr.DataArray:
    return xr.DataArray(np.array(values), dims=("time",), coords={"time": np.array(values)})


def test_one_line_of_a_timecourse() -> None:
    t = _time([0.0, 1.0, 2.0])
    y = xr.DataArray([1.0, 2.0, 3.0], dims=("time",), coords={"time": t.values})
    x, values, err = line_values("c", t, y, None)
    assert x.tolist() == [0.0, 1.0, 2.0] and values.tolist() == [1.0, 2.0, 3.0]
    assert err is None


def test_a_curve_of_a_ragged_point_drops_the_padding() -> None:
    t = xr.DataArray([0.0, 1.0, np.nan], dims=("_point",))
    y = xr.DataArray([1.0, np.nan, np.nan], dims=("_point",))
    x, values = line_values("c", t, y)
    assert x.tolist() == [0.0, 1.0]
    np.testing.assert_array_equal(values, [1.0, np.nan])


def test_a_dimension_of_a_scan_raises() -> None:
    t = _time([0.0, 1.0, 2.0])
    y = xr.DataArray(
        np.ones((2, 3)), dims=("dose", "time"), coords={"time": t.values, "dose": [0, 1]}
    )
    with pytest.raises(ValueError, match=r"curve 'c'.*'dose'.*Data\(sel=\.\.\.\)"):
        line_values("c", t, y)


def test_a_value_broadcasts_against_rows() -> None:
    x = xr.DataArray([1.0, 2.0], dims=("row",))
    y = xr.DataArray(5.0)
    xs, ys = line_values("c", x, y)
    assert xs.tolist() == [1.0, 2.0] and ys.tolist() == [5.0, 5.0]


def test_different_coordinates_raise() -> None:
    with pytest.raises(ValueError, match="curve 'c'.*coordinates"):
        line_values("c", _time([0.0, 1.0]), _time([0.0, 2.0]))


def test_nothing_is_nothing() -> None:
    assert line_values("c", None, None) == (None, None)


def test_without_padding_keeps_a_nan_of_y() -> None:
    x, y = without_padding(np.array([0.0, np.nan, 2.0]), np.array([1.0, 2.0, np.nan]))
    assert x.tolist() == [0.0, 2.0]
    np.testing.assert_array_equal(y, [1.0, np.nan])
```

In `tests/experiment/test_experiment_run.py`:

- drop the import of `first_curve` (keep `without_padding` only if still used); in `test_the_data_of_a_ragged_scan_has_its_points_last` replace the `first_curve` line by a selection of the first point: `x, curve = without_padding(time.values[0], y.values[0])`.
- add an experiment with a figure of the scan and two tests:

```python
class ScanFigureExperiment(ScanExperiment):
    """The scan with a figure of Y, of one point or of all points."""

    SEL: dict | None = None

    def figures(self) -> dict:
        figure = Figure(experiment=self, sid="fig", num_rows=1, num_cols=1)
        plot = figure.create_plots(xaxis=Axis("time"), yaxis=Axis("Y"))[0]
        sel = type(self).SEL
        plot.curve(
            x=Data("time", task="task", sel=sel),
            y=Data("[Y]", task="task", sel=sel),
            label="Y",
        )
        return {"fig": figure}


class OnePointFigureExperiment(ScanFigureExperiment):
    SEL = {"d": 1}


def test_a_curve_of_a_scan_without_a_selection_raises() -> None:
    runner = _runner(ScanFigureExperiment)
    experiment = runner.experiments["ScanFigureExperiment"]
    experiment.run(runner.simulator)
    with pytest.raises(ValueError, match=r"'d'.*Data\(sel=\.\.\.\)"):
        experiment.create_mpl_figures()


def test_a_curve_of_one_point_of_a_scan_is_drawn(tmp_path: Path) -> None:
    runner = _runner(OnePointFigureExperiment)
    experiment = runner.experiments["OnePointFigureExperiment"]
    experiment.run(runner.simulator, output_path=tmp_path, figure_formats=["png", "html"])
    assert (tmp_path / f"{experiment.sid}_fig.png").exists()
```

`save_mpl_figures` writes `<sid>_<figure key>.<format>`. The `html` format needs plotly from the `dev` extra, which the tests have.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/plot/test_padding.py tests/experiment/test_experiment_run.py -k "padding or line or curve or ragged"`
Expected: FAIL with `ImportError: cannot import name 'line_values'`.

- [ ] **Step 3: Implement**

Replace `src/sbmlsim/plot/padding.py` by:

```python
"""The values of a curve as one line, without the padding of ragged results.

The values of a curve are the labelled arrays of its data, see
`Data.get_data`; they are broadcast by the names of their dimensions and must
leave one dimension, along which the line is drawn: the time of a timecourse,
the points of a ragged result or the rows of a dataset. A dimension of a scan
is selected with `Data(sel=...)`. In the ragged layout every simulation keeps
its own time points and one with fewer points is padded with `NaN`: the
points whose x is `NaN` are the padding, a `NaN` of y is a gap of the data and
is kept.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from sbmlsim.data import ROW
from sbmlsim.result.scan import POINT, TIME


def line_values(sid: str, *arrays: xr.DataArray | None) -> tuple[Any, ...]:
    """Get the values of a curve as one line, without the padding.

    Args:
        sid: the id of the curve, for the errors.
        arrays: the x, y and error arrays of the curve, `None` for a missing
            one.

    Returns:
        The values of every array along the one dimension which is left,
        `None` stays. The values are typed `Any`: they are handed to the
        functions of matplotlib and plotly, which take any array.

    Raises:
        ValueError: if two arrays have different coordinates of a dimension,
            or more than one dimension is left after broadcasting.
    """
    given = [a for a in arrays if a is not None]
    if not given:
        return tuple(arrays)
    try:
        aligned = xr.align(*given, join="exact")
    except ValueError as err:
        raise ValueError(
            f"The data of the curve '{sid}' has different coordinates of a "
            f"dimension: {err}"
        ) from err
    broadcast = iter(xr.broadcast(*aligned))
    lines = [None if a is None else next(broadcast) for a in arrays]
    dims = next(line for line in lines if line is not None).dims
    if len(dims) > 1:
        scan = [str(d) for d in dims if d not in (TIME, POINT, ROW)] or [str(d) for d in dims]
        raise ValueError(
            f"The curve '{sid}' has the dimensions {[str(d) for d in dims]}; a curve "
            f"draws one line: select one label of {scan} with Data(sel=...)."
        )
    return without_padding(
        *[None if line is None else np.asarray(line.values) for line in lines]
    )


def without_padding(x: Any, *others: Any) -> tuple[Any, ...]:
    """Drop the points of a curve whose x is the padding of a ragged result.

    Args:
        x: the x values.
        others: arrays of the same points, e.g. y and the errors, or `None`.

    Returns:
        The x values and the others without the padded points, `None` stays.
    """
    if x is None:
        return (x, *others)
    xs = np.asarray(x)
    if not np.issubdtype(xs.dtype, np.floating):
        return (x, *others)
    keep = ~np.isnan(xs)
    if keep.all():
        return (x, *others)
    return (
        xs[keep],
        *[
            None
            if other is None
            else (np.asarray(other)[keep] if np.size(other) == keep.size else other)
            for other in others
        ],
    )
```

If importing `sbmlsim.data` from `sbmlsim.plot.padding` creates an import cycle (`sbmlsim.data` imports nothing of `sbmlsim.plot`, so it should not), define `ROW` where both can import it without a cycle and record it in the report.

`src/sbmlsim/plot/serialization_matplotlib.py`: import `line_values` instead of `first_curve, without_padding`; the curve block becomes

```python
                    x_data, y_data, xerr_data, yerr_data = line_values(
                        curve.sid, x, y, xerr, yerr
                    )
```

(drop the comment about the first point of a scan) and the area block

```python
                    x_data, yfrom_data, yto_data = line_values(area.sid, x, yfrom, yto)
```

`src/sbmlsim/plot/serialization_plotly.py`: import `line_values`; `_values` returns the array itself (`data.get_data(experiment=experiment, to_units=unit)`, typed `xr.DataArray | None`, docstring "The values of the data, or `None`."); the area becomes `x, yfrom, yto = line_values(area.sid, _values(area.x, ...), _values(area.yfrom, ...), _values(area.yto, ...))` and the curve `x, y, yerr, xerr = line_values(curve.sid, _values(curve.x, ...), _values(curve.y, ...), _values(curve.yerr, ...), _values(curve.xerr, ...))`.

`src/sbmlsim/plot/plotting.py` `Plot.add_data`: add the keyword `sel: Mapping[str, Any] | None = None` (document it: "labels of the dimensions of a task, or column values of the rows of a dataset, for every `Data` of the curve, see `Data`"), and pass `sel=sel` to every `Data(...)` the method builds (x, y, the four error data and the count data).

`examples/demo/demo.py`: above the class define

```python
#: the point of the scan a curve draws: the middle initial value of A and the
#: reference of the local design
POINT_SEL = {"dim_init": 5, "dim_sens": "reference"}
```

and the curves become `x=Data("time", task=task_id, sel=POINT_SEL)`, `y=Data(key, task=task_id, sel=POINT_SEL)`; replace the comment "a curve of a scan draws the first point of the scan" by "a curve draws one point of the scan, selected by its labels".

`examples/README.md`: the last sentence becomes "A curve draws one line: the point of a scan it shows is selected with `Data(sel=...)`; `examples/demo`, `examples/glucose` and `examples/repressilator` run and are tested."

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/plot tests/experiment tests/examples/test_example_scripts.py -k "padding or curve or ragged or serialization or demo"`
Expected: PASS.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/plot/padding.py src/sbmlsim/plot/serialization_matplotlib.py src/sbmlsim/plot/serialization_plotly.py src/sbmlsim/plot/plotting.py examples/demo/demo.py examples/README.md tests/plot/test_padding.py tests/experiment/test_experiment_run.py
git commit -m "A curve draws one line of its data and never a silent point of a scan" -m "first_curve, which drew the first point of a scan without a word, is replaced by line_values: the arrays of a curve are broadcast by dimension name and must leave one dimension, a dimension of a scan raises with the advice to select a label with Data(sel=...). Plot.add_data passes sel to its data and the demo selects the point it draws."
```

---

### Task 4: Observables of an experiment

**Files:**
- Modify: `src/sbmlsim/experiment/experiment.py` (`observables`, `initialize`, `_check_keys`, `_check_types`, `_figure_data`, `_task_data`, `_index_kind`, `_check_task_data`, `_task_outputs`, `_needed_observables`, `_selections_of_model`, `_run_tasks`, `to_dict`, `__str__`)
- Test: `tests/experiment/test_experiment_observables.py` (new)

**Interfaces:**
- Consumes: `Simulator.run(model, scan, observables, keep=[observables and selections])` (Task 1), `Data.get_data` arrays and `Data(sel=...)` (Task 2), `line_values` (Task 3, through figures), `sbmlsim.simulation.observables.Observable`, `PK`, `Scan`, `Dimension` (`id`, `values`, `coordinates`), `Figure.get_plots()`, `Plot.curves`, `Plot.areas`, `RoadrunnerSBMLModel.has_selection`, `RoadrunnerSBMLModel.r`.
- Produces: `SimulationExperiment.observables() -> dict[str, Observable]`; `SimulationExperiment._observables`; the JSON keys `"observables"` (experiment) and `tasks[k]["observables"]` (the ids a task computes).

- [ ] **Step 1: Write the failing tests**

Create `tests/experiment/test_experiment_observables.py`:

```python
"""An experiment declares observables and every task computes the ones its data read."""

import json
from pathlib import Path

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Axis, Figure
from sbmlsim.simulation import PK, Change, Dimension, Formula, Observable, Scan, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task
from tests.simulator.models import sbml_pk


class PKExperiment(SimulationExperiment):
    """A dosed one-compartment model, once and over three doses."""

    def models(self) -> dict:
        return {"m": AbstractModel(source=sbml_pk())}

    def simulations(self) -> dict:
        simulation = Simulation(end=48, steps=96, changes=[Change(0, {"PODOSE": Q(100, "mg")})])
        doses = Dimension("dose", values={"PODOSE": Q([50.0, 100.0, 200.0], "mg")})
        return {"sim": simulation, "scan": Scan(simulation, [doses])}

    def observables(self) -> dict[str, Observable]:
        return {
            "pk": PK("pk", "[C]", dose="PODOSE", route="oral"),
            "cmax": Formula("cmax", "max([C])"),
            # never read by a data: never compiled, so its unknown symbol is no error
            "unused": Formula("unused", "nope * 2"),
        }

    def tasks(self) -> dict:
        return {"task_sim": Task(model="m", simulation="sim"), "task_scan": Task(model="m", simulation="scan")}

    def data(self) -> dict:
        self.add_selections_data(["time", "[C]"])
        return {
            "cmax_scan": Data("cmax", task="task_scan"),
            "pk_cmax_scan": Data("pk.cmax", task="task_scan"),
            "dose_scan": Data("PODOSE", task="task_scan"),
        }


def _runner(experiment_class: type[SimulationExperiment]) -> ExperimentRunner:
    return ExperimentRunner(
        experiment_classes=[experiment_class],
        simulator=Simulator(),
        base_path=Path("."),
        data_path=Path("."),
    )


@pytest.fixture(scope="module")
def experiment() -> SimulationExperiment:
    runner = _runner(PKExperiment)
    experiment = runner.experiments["PKExperiment"]
    experiment.run(runner.simulator)
    return experiment


def test_a_task_computes_the_observables_its_data_read(experiment: SimulationExperiment) -> None:
    scan = experiment.results["task_scan"]
    assert set(scan.ds.data_vars) == {"cmax", "pk.cmax", "[C]"}
    assert scan["cmax"].dims == ("dose",) and scan["[C]"].dims == ("dose", "time")
    np.testing.assert_allclose(scan["cmax"].values, scan["[C]"].max("time").values)
    single = experiment.results["task_sim"]
    assert "cmax" not in single.ds.data_vars and "[C]" in single.ds.data_vars


def test_the_data_of_an_observable_is_a_labelled_array(experiment: SimulationExperiment) -> None:
    cmax = Data("pk.cmax", task="task_scan").get_data(experiment)
    assert cmax.dims == ("dose",)
    np.testing.assert_allclose(cmax["PODOSE"].values, [50.0, 100.0, 200.0])
    assert np.all(np.diff(cmax.values) > 0)
    assert cmax.attrs["units"]
    middle = Data("cmax", task="task_scan", sel={"dose": 1}).get_data(experiment)
    assert middle.dims == ()


def test_an_unknown_index_raises_at_initialize() -> None:
    class Unknown(PKExperiment):
        def data(self) -> dict:
            return {"x": Data("nope", task="task_sim")}

    with pytest.raises(ValueError, match=r"'nope'.*observable.*task_sim.*model 'm'"):
        _runner(Unknown)


def test_a_parameter_of_an_observable_which_is_no_pk_raises() -> None:
    class NotPK(PKExperiment):
        def data(self) -> dict:
            return {"x": Data("cmax.value", task="task_sim")}

    with pytest.raises(ValueError, match=r"'cmax\.value'"):
        _runner(NotPK)


def test_the_key_of_an_observable_is_its_id() -> None:
    class WrongKey(PKExperiment):
        def observables(self) -> dict[str, Observable]:
            return {"peak": Formula("cmax", "max([C])")}

    with pytest.raises(ValueError, match=r"key 'peak'.*id 'cmax'"):
        _runner(WrongKey)


def test_an_observable_id_must_not_clash_with_a_key() -> None:
    class Clash(PKExperiment):
        def observables(self) -> dict[str, Observable]:
            return {"task_sim": Formula("task_sim", "max([C])")}

    with pytest.raises(ValueError, match="Duplicate key 'task_sim'"):
        _runner(Clash)


def test_without_reduced_selections_every_selection_is_kept(experiment: SimulationExperiment) -> None:
    runner = _runner(PKExperiment)
    full = runner.experiments["PKExperiment"]
    full.run(runner.simulator, reduced_selections=False)
    names = set(full.results["task_scan"].ds.data_vars)
    assert {"cmax", "pk.cmax", "[C]"} <= names and len(names) > 3


class ValuesOnly(PKExperiment):
    """A task whose data read values per simulation only, with time registered."""

    def data(self) -> dict:
        self.add_selections_data(["time"], task_ids=["task_scan"])
        return {"cmax_scan": Data("cmax", task="task_scan")}


def test_a_task_of_values_per_simulation_runs_with_time_registered() -> None:
    runner = _runner(ValuesOnly)
    experiment = runner.experiments["ValuesOnly"]
    experiment.run(runner.simulator)
    result = experiment.results["task_scan"]
    assert set(result.ds.data_vars) == {"cmax"} and "time" not in result.ds.dims


class FigureData(PKExperiment):
    """A figure reads a selection which data() does not register."""

    def data(self) -> dict:
        return {}

    def figures(self) -> dict:
        figure = Figure(experiment=self, sid="fig", num_rows=1, num_cols=1)
        plot = figure.create_plots(xaxis=Axis("time"), yaxis=Axis("C"))[0]
        plot.curve(x=Data("time", task="task_sim"), y=Data("[C]", task="task_sim"))
        return {"fig": figure}


def test_the_data_of_figures_count() -> None:
    runner = _runner(FigureData)
    experiment = runner.experiments["FigureData"]
    experiment.run(runner.simulator)
    assert "[C]" in experiment.results["task_sim"].ds.data_vars


def test_the_json_has_the_observables(experiment: SimulationExperiment) -> None:
    d = json.loads(experiment.to_json())
    assert set(d["observables"]) == {"pk", "cmax", "unused"}
    assert d["observables"]["pk"]["type"]
    assert d["tasks"]["task_scan"]["observables"] == ["pk", "cmax"]
    assert d["tasks"]["task_sim"]["observables"] == []
```

`Observable.to_dict` writes the key `"type"` (`"Formula"`, `"PK"`, `"Custom"`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/experiment/test_experiment_observables.py`
Expected: FAIL (the runner raises `ValueError: Duplicate key` or `KeyError` from `self._observables`, or `"cmax"` becomes a roadrunner selection).

- [ ] **Step 3: Implement**

In `src/sbmlsim/experiment/experiment.py`:

Imports: `from sbmlsim.simulation.observables import PK, Observable` and `from sbmlsim.result.scan import TIME`; `RoadrunnerSBMLModel` from `sbmlsim.model.model_roadrunner` if not imported yet.

`__init__`: `self._observables: dict[str, Observable] = {}` next to the other dicts.

`initialize`: after `self._simulations.update(self.simulations())` add `self._observables.update(self.observables())`; after `self._check_types()` add `self._check_task_data()`.

`__str__`: add `f"{'observables':20} {list(self._observables.keys())}",` after the simulations line.

After `simulations()` add:

```python
    def observables(self) -> dict[str, Observable]:
        """Define the observables of the experiment, by their id.

        A `Formula`, `PK` or `Custom` of `sbmlsim.simulation.observables`. A
        `Data` of a task reads an observable by its id and a parameter of a
        `PK` observable as `<id>.<parameter>`; a task computes the observables
        its data read, see `_run_tasks`. The child classes fill out the
        information.
        """
        return {}
```

`_check_keys`: add `"_observables"` to the list of fields after `"_simulations"`.

`_check_types`: after the simulations loop add

```python
        for key, observable in self._observables.items():
            if not isinstance(observable, Observable):
                raise ValueError(
                    f"observables must be of type Formula, PK or Custom, but "
                    f"observable '{key}' has type: '{type(observable)}'"
                )
            if observable.id != key:
                raise ValueError(
                    f"The observable of the key '{key}' has the id "
                    f"'{observable.id}': the key of an observable is its id."
                )
```

Replace `_task_data` and `_selections_of_model`, and add the helpers:

```python
    def _figure_data(self) -> Iterator[Data]:
        """Iterate the data the curves and areas of the figures read."""
        for figure in self._figures.values():
            for plot in figure.get_plots():
                for curve in plot.curves:
                    for d in (curve.x, curve.y, curve.xerr, curve.yerr):
                        if d is not None:
                            yield d
                for area in plot.areas:
                    yield from (area.x, area.yfrom, area.yto)

    def _task_data(self) -> Iterator[Data]:
        """Iterate the data of the experiment which comes from a task.

        The data of `data()`, of every fit mapping and of every figure, and
        the variables of a function among them, i.e. everything a run has to
        simulate. A `FitData` builds its `Data` when it is created and does
        not register it, so a fit mapping is asked for its data here rather
        than looked up in `self._data`.

        Yields:
            Every `Data` of the experiment which reads a task.
        """

        def walk(d: Data) -> Iterator[Data]:
            if d.is_task():
                yield d
            elif d.is_function():
                for variable in d.variables.values():
                    found = self._data.get(variable) if isinstance(variable, str) else variable
                    if isinstance(found, Data):
                        yield from walk(found)

        sources: list[Data] = list(self._data.values())
        for mapping in self._fit_mappings.values():
            for fit_data in (mapping.reference, mapping.observable):
                for key in FitDataInitialized.KEYS:
                    d = getattr(fit_data, key, None)
                    if isinstance(d, Data):
                        sources.append(d)
        sources.extend(self._figure_data())
        for d in sources:
            yield from walk(d)

    def _index_kind(self, d: Data) -> str:
        """Classify the index of task data.

        Returns:
            `"time"`, `"observable"` (an observable or a parameter of a `PK`
            observable), `"coordinate"` (a dimension of the scan of the task, a
            target it changes or a coordinate of a dimension) or `"selection"`.

        Raises:
            ValueError: if the data reads a task which does not exist, or an
                index which is none of these.
        """
        task = self._tasks.get(str(d.task_id))
        if task is None:
            raise ValueError(
                f"{d} reads the task '{d.task_id}', which is no task of the "
                f"experiment '{self.sid}': {sorted(self._tasks)}."
            )
        index = d.selection
        if index == TIME:
            return "time"
        if index in self._observables:
            return "observable"
        head, _, parameter = index.partition(".")
        if parameter and isinstance(self._observables.get(head), PK):
            return "observable"
        if index in _coordinates(self._simulations[task.simulation_id]):
            return "coordinate"
        model = self._models.get(task.model_id)
        if (
            isinstance(model, RoadrunnerSBMLModel)
            and model.r is not None
            and not model.has_selection(index)
        ):
            raise ValueError(
                f"{d} of the experiment '{self.sid}' reads '{index}', which is "
                f"neither an observable of the experiment "
                f"{sorted(self._observables)}, a coordinate of the scan of the "
                f"task '{d.task_id}' nor a selection of the model "
                f"'{task.model_id}'."
            )
        return "selection"

    def _check_task_data(self) -> None:
        """Check the index of every task data before anything is simulated.

        Raises:
            ValueError: see `_index_kind`.
        """
        for d in self._task_data():
            self._index_kind(d)

    def _task_outputs(self, task_key: str) -> tuple[list[str], list[str]]:
        """Get the observable outputs and the selections the data of a task read.

        Returns:
            The observable ids and `<id>.<parameter>` of PK observables, and the
            selections, each in the order of the data.
        """
        observed: dict[str, None] = {}
        selections: dict[str, None] = {}
        for d in self._task_data():
            if d.task_id != task_key:
                continue
            kind = self._index_kind(d)
            if kind == "observable":
                observed[d.selection] = None
            elif kind == "selection":
                selections[d.selection] = None
        return list(observed), list(selections)

    def _needed_observables(self, outputs: Iterable[str]) -> list[Observable]:
        """Get the observables the outputs need, in the order of `observables()`.

        An output is an observable id or `<id>.<parameter>` of a PK observable;
        an observable needs the observables its formula or function reads.
        """
        needed: set[str] = set()
        stack = [o if o in self._observables else o.partition(".")[0] for o in outputs]
        while stack:
            name = stack.pop()
            if name in needed or name not in self._observables:
                continue
            needed.add(name)
            for symbol in self._observables[name].reads:
                stack.append(
                    symbol if symbol in self._observables else symbol.partition(".")[0]
                )
        return [o for name, o in self._observables.items() if name in needed]

    def _selections_of_model(self, model_id: str) -> set[str]:
        """Get the selections a model has to be simulated with.

        Args:
            model_id: the model the tasks are run on.

        Returns:
            `time` and the selections the data of the tasks of the model read;
            observables and coordinates are no selections.
        """
        selections = {TIME}
        for task_key, task in self._tasks.items():
            if task.model_id == model_id:
                selections.update(self._task_outputs(task_key)[1])
        return selections
```

Add the module function (after the imports or at the end of the module):

```python
def _coordinates(simulation: Simulation | Scan) -> set[str]:
    """Get the names the result of a scan has as coordinates of its dimensions.

    The dimension ids, the targets the dimensions change and the coordinates
    of the dimensions; a simulation has none.
    """
    if not isinstance(simulation, Scan):
        return set()
    names: set[str] = set()
    for dimension in simulation.dimensions:
        names.add(dimension.id)
        names.update(dimension.values)
        names.update(dimension.coordinates)
    return names
```

In `_run_tasks`, the loop over the tasks of a model becomes:

```python
            for task_key in task_keys:
                task = self._tasks[task_key]
                scan = self._simulations[task.simulation_id]
                observed, selections = self._task_outputs(task_key)
                if not observed:
                    self._results[task_key] = simulator.run(model, scan)
                    continue
                if not reduced_selections:
                    selections = [s for s in model.selections or [] if s != TIME]
                self._results[task_key] = simulator.run(
                    model,
                    scan,
                    self._needed_observables(observed),
                    keep=[*observed, *selections],
                )
```

and its docstring gains: "A task whose data read observables runs with the observables they need and keeps them and the selections its data read, every selection of the model without `reduced_selections`; a task without runs with the selections of the model."

`to_dict`: the tasks entry becomes

```python
            "tasks": {
                k: {
                    **v.to_dict(),
                    "observables": [
                        o.id for o in self._needed_observables(self._task_outputs(k)[0])
                    ],
                }
                for k, v in self._tasks.items()
            },
            "observables": self._observables,
```

(`ObjectJSONEncoder` serializes an observable with its `to_dict`).

Update the docstring of `data()` and `figures()`: the data of figures counts for the selections as well, registering it in `data()` is no longer needed.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/experiment`
Expected: PASS.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/experiment/experiment.py tests/experiment/test_experiment_observables.py
git commit -m "A simulation experiment declares observables and its tasks compute the ones their data read" -m "observables() returns the Formula, PK and Custom observables of the scan core by their id. The index of every task data of data(), fit mappings and figures is classified at initialize as time, observable, coordinate of the scan or selection of the model, and an unknown one raises before anything is simulated. A task whose data read observables runs with the observables they need and keeps them next to its selections in one run; the JSON of the experiment holds the observables and what every task computes."
```

---

### Task 5: The example, the docs and CLAUDE.md

**Files:**
- Create: `examples/experiment_scans.py`
- Modify: `tests/examples/test_example_scripts.py` (`SCRIPTS`), `examples/README.md` (a row), `docs/experiments.md` (a section "Observables and scans"), `docs/data.md` (a runnable block of observables), `CLAUDE.md`

**Interfaces:**
- Consumes: everything of Tasks 1 to 4.

- [ ] **Step 1: Write the example**

Create `examples/experiment_scans.py`:

```python
"""Scans and observables in a simulation experiment: midazolam at three doses.

The experiment declares the observables of the scan core next to its
simulations, the mass concentration of midazolam in plasma and its
non-compartmental analysis, and its data are labelled arrays which keep the
dimension of the doses: the cmax of every dose is read by its label.
"""

from pathlib import Path
from typing import override

from sbmlsim import Q
from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.resources import MIDAZOLAM_SBML
from sbmlsim.simulation import PK, Change, Dimension, Formula, Observable, Scan, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task

#: the PK parameters the example prints
PARAMETERS = ("pk.cmax", "pk.tmax", "pk.auc_inf_obs")


class MidazolamDoses(SimulationExperiment):
    """Oral midazolam at three doses."""

    @override
    def models(self) -> dict[str, AbstractModel | Path]:
        return {"model": MIDAZOLAM_SBML}

    @override
    def simulations(self) -> dict[str, Scan]:
        simulation = Simulation(
            time_unit="hr",
            end=24,
            steps=480,
            changes=[Change(0, {"PODOSE_mid": Q(7.5, "mg")})],
        )
        doses = Dimension(
            "dose",
            values={"PODOSE_mid": Q([5.0, 7.5, 15.0], "mg")},
            labels=["low", "standard", "high"],
        )
        return {"doses": Scan(simulation, [doses])}

    @override
    def observables(self) -> dict[str, Observable]:
        return {
            "mid": Formula("mid", "[Cve_mid] * Mr_mid", unit="ng/ml"),
            "pk": PK("pk", "mid", dose="PODOSE_mid", route="oral"),
        }

    @override
    def tasks(self) -> dict[str, Task]:
        return {"task_doses": Task(model="model", simulation="doses")}

    @override
    def data(self) -> dict[str, Data]:
        indices = ("time", "mid", *PARAMETERS)
        return {i.replace(".", "_"): Data(i, task="task_doses") for i in indices}


def run(output_path: Path) -> SimulationExperiment:
    """Run the experiment and print the PK parameters per dose."""
    base_path = Path(__file__).parent
    runner = ExperimentRunner(
        MidazolamDoses, simulator=Simulator(), base_path=base_path, data_path=base_path
    )
    experiment = runner.run_experiments(output_path=output_path)[0].experiment
    for index in PARAMETERS:
        values = Data(index, task="task_doses").get_data(experiment)
        per_dose = dict(zip(values["dose"].values.tolist(), values.values.round(3).tolist(), strict=True))
        print(index, per_dose, values.attrs["units"])
    high = Data("mid", task="task_doses", sel={"dose": "high"}).get_data(experiment)
    print("mid of the high dose", high.dims, round(float(high.max()), 3), high.attrs["units"])
    return experiment


if __name__ == "__main__":
    run(Path.cwd() / "results")
```

`ExperimentRunner` takes one class or a list of classes and `run_experiments(output_path)` returns the `ExperimentResult`s, as in `examples/demo/demo.py`. Run it from a scratch directory: `cd $(mktemp -d) && PYTHONPATH=/home/mkoenig/git/sbmlsim uv run --project /home/mkoenig/git/sbmlsim python -W error -m examples.experiment_scans`; the cmax must grow with the dose.

Add `"examples.experiment_scans",` to `SCRIPTS` in `tests/examples/test_example_scripts.py` after `"examples.observables"`, and to `examples/README.md` after the row of `examples/observables.py`:

```markdown
| `examples/experiment_scans.py` | scans and observables in a simulation experiment: the PK parameters of midazolam over three doses as labelled arrays, read by the labels of the doses |
```

- [ ] **Step 2: The docs**

`docs/data.md`: after the section "Selecting points" of Task 2 append a section with a runnable block (the page runs in `tests/docs/test_docs_code.py`):

````markdown
## Data of observables

An experiment declares observables with `observables()`, the `Formula`, `PK` and `Custom` of [Observables](observables.md), and a `Data` of a task reads one by its id, the parameter of a `PK` observable as `<id>.<parameter>`. A task computes the observables its data read, in one run with the selections it reads:

```python
import numpy as np

from sbmlsim.simulation import Dimension, Formula, Observable, Scan


class ObservableExperiment(DataExperiment):
    @override
    def simulations(self) -> dict[str, Simulation | Scan]:
        return {
            "tc": Simulation(end=100, steps=100),
            "scan": Scan(
                Simulation(end=100, steps=100),
                [Dimension("x0", values={"X": np.array([10.0, 20.0, 40.0])})],
            ),
        }

    @override
    def observables(self) -> dict[str, Observable]:
        return {"xmax": Formula("xmax", "max([X])")}

    @override
    def tasks(self) -> dict[str, Task]:
        return {
            "task_tc": Task(model="model", simulation="tc"),
            "task_scan": Task(model="model", simulation="scan"),
        }

    @override
    def data(self) -> dict[str, Data]:
        return {"xmax": Data("xmax", task="task_scan"), "x": Data("[X]", task="task_scan")}


results = ExperimentRunner(
    [ObservableExperiment], simulator=Simulator(), base_path=Path.cwd(), data_path=Path.cwd()
).run_experiments(output_path=Path.cwd() / "results")
experiment = results[0].experiment

xmax = Data("xmax", task="task_scan").get_data(experiment)
print(xmax.dims, xmax["X"].values, xmax.values.round(3))
x = Data("[X]", task="task_scan", sel={"x0": 2}).get_data(experiment)
print(x.dims, float(x.max()) == float(xmax.sel(x0=2)))
```
````

The block reuses `DataExperiment`, `Path`, `override`, `Simulation`, `Task`, `Data`, `ExperimentRunner` and `Simulator` of the earlier blocks of the page (the blocks of a page share one namespace); check that and import what is missing. Run `uv run pytest -q -n 0 tests/docs -k data`.

`docs/experiments.md`: add a section "Observables and scans" before "Running an experiment":

```markdown
## Observables and scans

`observables()` declares the observables of the experiment, the `Formula`, `PK` and `Custom` of [Observables](observables.md), by their id; it is defined after `simulations()` and before `tasks()`. A `Data` of a task reads an observable by its id (`Data("pk.cmax", task="task_doses")` for a parameter of a `PK` observable), and every task computes the observables its data read, in one run with the selections they read; the data of `data()`, of the fit mappings and of the figures count. The index of every task data is checked when the experiment is initialized: an index which is neither `time`, an observable, a coordinate of the scan of the task nor a selection of its model raises before anything is simulated. The data of a task is a labelled array with the dimensions of its scan, see [Data](data.md); a curve draws one line, so the point of a scan it shows is selected with `Data(sel=...)`. `examples/experiment_scans.py` reads the PK parameters of midazolam over three doses.
```

- [ ] **Step 3: CLAUDE.md**

The user approved this plan, which includes these edits; change nothing else in `CLAUDE.md`:

- In the paragraph of `simulation/`, `simulator/`, `result/`, the sentence "The selections of a run without observables are the ones of the model (`RoadrunnerSBMLModel.set_selections`), with observables `time` and what the needed observables read;" becomes "The selections of a run without observables are the ones of the model (`RoadrunnerSBMLModel.set_selections`), with observables `time`, what the needed observables read and the selections `keep` names, which the result keeps as timecourses of their own names;".
- In the paragraph of `experiment/`, `data.py`, `task/`, `plot/`, `report/`: after "`SimulationExperiment` (`experiment/experiment.py`) is subclassed with `models()`, `datasets()`, `simulations()`," insert "`observables()` (the `Formula`, `PK` and `Custom` of the scan core by their id)," and replace the text from "`Data` (`data.py`) is a promise" up to and including "(`plot/padding.first_curve`, `without_padding`)." by: "`Data` (`data.py`) is a promise for data of a task (`Data("[X]", task=...)`, an observable `Data("pk.cmax", task=...)`), a dataset column or a function of other data; `DataSet` is a `pandas.DataFrame` with `uinfo`. `Task` pairs a model and a `Simulation` or a `Scan`; the results are `ScanResult`s, written as netCDF. The index of every task data of `data()`, the fit mappings and the figures is classified at `initialize()` as `time`, an observable, a coordinate of the scan or a selection of the model (an unknown one raises), and a task whose data read observables runs with the observables they need and keeps them next to its selections in one run. `Data.get_data` returns an `xarray.DataArray` with the dimensions of its source, their coordinates and `attrs["units"]` (`to_quantity` gives the pint quantity): a task variable over `(*dims, time)`, `(*dims, _point)` for a ragged result padded with `NaN`, or `(*dims)`, a dataset column over `row`, a function broadcast by dimension name; `sel=` selects labels (a dimension of the scan the data has not is skipped) or the rows of a dataset by column values. A curve draws one line of its broadcast arrays (`plot/padding.line_values`, without the padding) and raises for a dimension of a scan which is not selected."
- In the paragraph "**Formulas.**", "`data.evaluate_function` adds the reductions `max`/`min` of a single argument over the data, along the time of each simulation (`mean` and `at` are for observables, which have the times)" becomes "`data.evaluate_function` evaluates it on labelled arrays broadcast by dimension name and adds the reductions `max`/`min` of a single argument along the time dimension (`time` or `_point`), else the rows of a dataset (`mean` and `at` are for observables, which have the times)".
- In the conventions, "a `SimulationExperiment` defines its methods in the order of their dependencies: `datasets`, `models`, `simulations`, `tasks`, `data`, `fit_mappings`, `figures`; `tasks` reads `self._simulations` rather than calling `simulations()` again, and the selections are registered in `data`, not in `figures`." becomes "a `SimulationExperiment` defines its methods in the order of their dependencies: `datasets`, `models`, `simulations`, `observables`, `tasks`, `data`, `fit_mappings`, `figures`; `tasks` reads `self._simulations` rather than calling `simulations()` again; the data of `data()`, the fit mappings and the figures all count for the selections and observables of a task."

- [ ] **Step 4: Verify**

Run: `uv run pytest -q -n 0 tests/docs tests/examples/test_example_scripts.py -k "data or experiment_scans or demo"`, `uv run zensical build --clean` (no warnings), the example from a scratch directory as in Step 1.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add examples/experiment_scans.py examples/README.md tests/examples/test_example_scripts.py docs/data.md docs/experiments.md CLAUDE.md
git commit -m "The example and the docs of observables in simulation experiments" -m "examples/experiment_scans.py declares the observables of midazolam next to a scan over three doses and reads the PK parameters per dose as labelled arrays; docs/data.md shows observables, labelled arrays and sel with code the docs test runs, docs/experiments.md describes observables and scans in an experiment, and CLAUDE.md follows."
```

---

### Task 6: Verification and the pull request

**Files:** none new; the pull request.

- [ ] **Step 1: The whole suite and the checks**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`, `uv run zensical build --clean`, `rg -n "first_curve" src tests examples docs --glob '!docs/superpowers/**'` (no hit) and `rg -n "\.get_data\(" src examples` (every caller handles an array).

- [ ] **Step 2: The pull request**

Push and create the pull request with `gh-axi`, base `develop`, title "Observables in simulation experiments and Data as labelled arrays (#249, experiments phase 1)". The body describes, without any agent attribution: `observables()` and what a task computes, the selections `keep` names, `Data` as labelled arrays with `sel`, dataset rows and function broadcasting, `to_quantity`, the curve which refuses a silent point of a scan, the migrated callers and examples, the decisions of this plan, and what phases 2 and 3 bring. Wait for the checks and fix any failure.
