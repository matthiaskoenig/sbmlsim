# PEtab SciML Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A PEtab SciML problem is read into an `OptimizationProblem`, its networks run before the simulation or are compiled into the model, and the cases `sciml_problem_import` 001 to 039 of the PEtab SciML test suite reproduce the log-likelihood, the simulations and the gradient of the reference implementations, except the three cases with priors.

**Architecture:** `sbmlsim.sciml` gets the second half of its native layer: `SympyBackend` runs the unchanged interpreter on expressions, `compiler.py` writes a network of the right hand side or of an observable into a copy of the model as parameters with assignment rules (one rule per unit, one layer deep), and `hybridization.py` says where a network sits and evaluates a network before the simulation in numpy. `sbmlsim.fit` gets three generic changes which name no network: derived changes as a hook of `OptimizationProblem` (the protocol `sbmlsim.fit.derived.DerivedChanges`, which `Hybridization` implements), a scale per parameter (`FitParameter.scale`), and infinite bounds on the linear scale with parameters which are not entities of the model (targets `sciml:<id>`). `sbmlsim.fit.petab_v2.sciml` translates the extension `sciml` into these objects without `torch`, the reader compiles the models and hands the hybridizations to the problem, and `sciml/testsuite.py` compares the cases.

**Tech Stack:** python >= 3.13, `petab` 0.9.0 with `petab.v2.extensions.sciml`, `petab-sciml` 0.0.3, `h5py`, `pyyaml`, `sbmlmath` 0.4.1 (a declared dependency from this phase on), `sympy`, `libsbml`, `libroadrunner` 2.10, `numpy`, `scipy`; `torch` (CPU) only in the tests.

**Spec:** `docs/superpowers/specs/2026-09-30-petab-sciml-design.md`, including the section "Amendments after the phases 1 and 2". The notes of the controller (`phase3-notes.md`) are taken up in the tasks 1 (frozen `Network`, equality), 2 (`gelu` with `erf` on expressions), 8 (the step of the gradient next to a bound), 9 (elements of a layer the forward pass does not call), 11 (`sciml` as a known extension, `parameterScale`, reading without `torch`), and 12 (stale staging directories, the two gaps of the script, the sign of the gradient of the suite); every function which takes arrays annotates them as `np.ndarray`, never as `NDArray`.

## Global Constraints

- python >= 3.13; every module, class and function of `src/` carries full type annotations and a google style docstring (ruff `D`); `tests/` and `examples/` are exempt from the docstring rules.
- Zero diagnostics of `ruff check`, `ruff format --check` and `uvx ty check` after every task; a diagnostic of ty is suppressed only with a rule specific `# ty: ignore[rule]`, never with a blanket `# type: ignore`.
- Library code logs with `logging.getLogger(__name__)` and lazy `%s` formatting and never prints.
- Never the em dash character in any file, use "-".
- No attribution of agents in commits or files: no `Co-Authored-By`, no "Generated with" lines.
- Never edit `CHANGELOG.md` or `CLAUDE.md`; the passage for `CLAUDE.md` is collected in the last section of this plan, which no task applies.
- Markdown without hard line wraps: a paragraph, a list item and a table row are one line.
- Tests run with `uv run pytest -q -x <path>`; a test of a report uses `mapping_figures=False`.
- Commits on the current branch `petab-sciml-phase3`, never pushed.
- The documented commands `tox r -e sciml`, `tox r -e ty` and `tox r -e py3.14` work with the installed tox (tox 4.34.1 with tox-uv 1.29.0); no task changes `tox.ini`.
- `sbmlsim.sciml` knows nothing of PEtab: `sbmlsim.fit.petab_v2.sciml` translates. `sbmlsim.fit` imports `sbmlsim.sciml` lazily, `import sbmlsim.fit` and every module of it work without the extra `sciml`, which `tests/sciml/test_package.py::test_the_package_does_not_import_the_networks` pins.
- The changes to `sbmlsim.fit` are generic and none of them names a network. `FitParameter.scale=None` means the scale of the settings, the results of `tests/fit/` do not change (the one test whose expectation the spec changes is `test_an_infinite_bound_is_never_allowed`, see task 5), and the sampling of bounded parameters is tested against the behaviour of the release before.
- `OptimizationProblem` stays picklable, `__getstate__` gives the uninitialized definition with its hybridizations; a parallel fit with a network is a test (task 9).
- The compiled model is written to `derived_dir` (the directory of the model by default) as `<stem>_sciml.xml`, and `observables.py` runs on the compiled model afterwards (`<stem>_sciml_observables.xml`).
- Every new public function validates its inputs and raises errors which name the network, the node, the input, the output, the target or the fit mapping; the tests of every task pin them.
- The diffs of this plan are unified diffs against the file as it is when the task starts (the tasks in order, every task committed); they are applied with `git apply`. A new file is given completely.

## Review Focus

Five input classes the spec implies and the tests of the tasks now pin, one test each, most likely to bite first:

1. A PEtab SciML problem with two models: the reader says which models instead of reading the networks into the first one silently (`tests/sciml/test_reader.py::test_a_problem_with_two_models`, task 11).
2. A condition which sets the input of a network in the right hand side, i.e. one formula per condition for a rule of the model which has one formula: refused with the name of the input (`test_a_condition_which_sets_the_input_of_a_compiled_network`, task 11).
3. An output which the hybridization table assigns to an entity and an observable uses as well: refused instead of compiling one of the two (`test_an_output_in_the_right_hand_side_and_an_observable`, task 11).
4. A parameter whose bounds are closer than the steps of a difference, or which sits next to a bound: the gradient keeps its step and is one sided, or is the secant of the interval (`tests/fit/test_petab_v2_likelihood.py::test_the_difference_next_to_a_bound_keeps_its_step`, `test_the_difference_of_bounds_closer_than_the_steps`, task 8).
5. A problem which `petab` read with `torch`, i.e. a `PetabProblem` whose configuration holds a `SciMLConfig`: read from the configuration like one read from the YAML (`test_a_problem_which_petab_read_with_torch`, task 11).

## Facts measured with the prototype

The whole path of this plan was built in a scratch copy and run against the 39 cases of `sciml_problem_import` before the plan was written. What the tasks rely on:

- Reading: `petab.v2.Problem.from_yaml` accepts a dictionary of the YAML with the `sciml` block removed and reads the tables, the `array` nominal values and the extra column `parameterScale` (`Parameter.model_extra`); the conditions `cond1`/`cond2` of the cases 015 and 038 exist in no condition table, only in the array files, and the changes of a condition of case 003 set the inputs of the network, which are not entities of the model.
- The gradient of the suite is the gradient of the log-likelihood, built by `FiniteDifferences.central_fdm(5, 1)` of `llh + prior` at the nominal values, with the simulation of `Vern9` at the tolerances `1e-12`; case 001 gives `2538.78` for `alpha`, which `gradient(problem, order=4)` reproduces to `0.002`.
- The three point difference with `step=1e-6` misses `tol_grad = 0.1` on the cases 001, 010, 011, 012, 029 and 035 by `0.89` (the truncation error, it falls a hundredfold with a tenfold smaller step); the five point difference with `step=1e-6` and the tolerances `1e-13` of the integrator passes all 36 cases with the largest difference `0.0197` (case 001), and with the tolerances `1e-12` with `0.065`. With the tolerances `1e-10` 14 cases fail, with `1e-8` all of them: the error of the simulation enters the gradient multiplied by sensitivities of the order `1e4`.
- The simulations agree with the reference to `2.8e-8` and the log-likelihood to `1.2e-6` at the tolerances `1e-13`; the floor of `2e-8` is the output grid of the reader (`END_MARGIN = 1e-9` stretches the grid past the last measurement, so a measurement is interpolated between two points of the grid).
- roadrunner: an assignment rule follows a change of a parameter in every case, which is why the compiled network is rules and its elements are constant parameters a fit changes; the compiled model of case 001 gives the forward pass of the network at every time point to `1e-12` and the simulation of the reference to `4e-8`.
- libsbml writes a number with 15 significant digits: the nominal value of an element of a compiled network differs from the array by up to `1e-15` relative, which the tests take into account (`rel=1e-14`).
- `mathml.evaluate` cannot evaluate a formula with `beta`, `gamma`, `lambda`, `S`, `I` or `E` (see task 3), and the reader handed the text of a sympy expression to the L3 parser of libsbml, in which `log` is the decadic logarithm (see task 10).
- Compiled networks and time: a `Linear`-`tanh`-`Linear` network of 5 units per layer (51 elements, 11 units) compiles in `0.03 s`, loads in `0.12 s` and simulates 101 points in `1.4 ms` against `0.6 ms` without it; 20 units (501 elements) compile in `0.2 s`, load in `2 s` and simulate in `9 ms`; 50 units (2751 elements) compile in `3 s`, load in `44 s` and simulate in `32 ms`. The time to load grows faster than the number of units, roadrunner compiles every rule.
- The full test suite of the scratch copy after all tasks: 1471 passed, 2 skipped, the 12 warnings which exist before this branch.

## The baseline of phase 3

| case | outcome | reason |
| --- | --- | --- |
| 001 to 031, 035 to 039 | pass | largest differences: log-likelihood `1.2e-6`, simulations `7e-8`, gradient `0.0197` |
| 032, 033, 034 | unsupported | priors on the parameters of the network and of the model, the case states a log-posterior which the log-likelihood is not; the reader raises the gap `sciml-priors`, issue #190 |

`tests/data/sciml_baseline.json` records the four cases which do not pass (`ml_model_import/020` of phase 1 and these three) with their reasons, 92 of 96 cases pass.

## Where the plan departs from the spec

Every departure was forced by a case of the suite or by a finding of the prototype, and is named in the task which makes it:

- `NetworkInput` has `formulas` (one formula per condition) next to `formula` and `arrays`, and `Hybridization` has `constants`; a compiled network may have an input which is an array (task 7).
- `gradient` has `order` (2 or 4) and the comparison uses the five point difference; the difference next to a bound is one sided with the full step (task 8).
- The reader keys its fit mappings by observable and experiment and translates the math of an observable (task 10).
- `create_samples` loses `max_bound` (task 5).
- The formulas of the inputs are evaluated with the new `evaluate_formula` and not with `mathml.evaluate` (task 3).
- `network_fit_parameters` has `external` and leaves the elements of layers the forward pass does not call out (task 9).

## File structure

| file | responsibility |
| --- | --- |
| `src/sbmlsim/sciml/network.py` | `Network` (frozen, comparable) and the ids of elements, units, inputs and outputs |
| `src/sbmlsim/sciml/backend.py` | `NumpyBackend` and `SympyBackend` |
| `src/sbmlsim/sciml/hybridization.py` | `NetworkPattern`, `NetworkInput`, `Hybridization` (the `DerivedChanges` of a network) |
| `src/sbmlsim/sciml/compiler.py` | `compile_network`, `compiled_path` |
| `src/sbmlsim/sciml/parameters.py` | `network_fit_parameters` with `external` and the linear scale |
| `src/sbmlsim/sciml/errors.py` | `NetworkHybridizationError`, `NetworkCompilationError` |
| `src/sbmlsim/sciml/testsuite.py` | `ProblemImportCase.run` |
| `src/sbmlsim/mathml.py` | formulas of SBML as expressions and back |
| `src/sbmlsim/fit/derived.py` | the protocol `DerivedChanges` |
| `src/sbmlsim/fit/objects.py` | `FitParameter.scale`, `EXTERNAL_PREFIX`, `is_external`, `entity_id` |
| `src/sbmlsim/fit/optimization.py` | `hybridizations`, the scales, the derived changes of a simulation group, infinite bounds |
| `src/sbmlsim/fit/sampling.py` | parameters without bounds are not sampled |
| `src/sbmlsim/fit/parameter_mapping.py` | no change for an external target |
| `src/sbmlsim/fit/fisher.py`, `identifiability.py` | the scale of every parameter |
| `src/sbmlsim/fit/cli.py` | `FitDefinition.hybridizations` |
| `src/sbmlsim/fit/petab_v2/likelihood.py` | `stencil`, `gradient(order=...)` |
| `src/sbmlsim/fit/petab_v2/reader.py` | fit mappings per observable and experiment, the math of an observable, the compiled model, the SciML parts |
| `src/sbmlsim/fit/petab_v2/sciml.py` | `SciMLReader`: the translation of the extension |
| `src/sbmlsim/fit/petab_v2/extension.py` | `known_extensions`, the error of the missing extra |
| `src/sbmlsim/fit/petab_v2/gaps.py` | the five gaps of SciML |
| `src/sbmlsim/testsuite/cache.py` | stale staging directories |
| `scripts/sciml_testsuite.py` | the two gaps of the notes |
| `tests/data/models/lotka_volterra.xml` | the test model of the hybrid problems |
| `tests/sciml/hybrid.py`, `experiment.py`, `petab.py` | the networks, the experiment and the writer of PEtab problems of the tests |

---

### Task 1: `Network` is frozen and compares its architecture and arrays

**Files:**
- Modify: `src/sbmlsim/sciml/network.py`
- Test: `tests/sciml/test_layers_convolution.py`
- Test: `tests/sciml/test_network.py`

**Interfaces:**
- Consumes: `Network` of `src/sbmlsim/sciml/network.py` as it is.
- Produces: `Network.__eq__(other) -> bool` (id, architecture and arrays), `Network.__hash__()`, read-only arrays in `Network.parameters`; `dataclasses.replace(network, parameters=...)` stays the way to a network with other values.

`Network` is a mutable dataclass whose caches (`_array_specs`, `_used_layers`, `_parameter_ids`) depend on `sid` and `model`, and an assignment of `parameters` skips the validation of `__post_init__`. The hybridizations of phase 3 hold networks, a fit pickles them for its workers and tests compare them, so the network becomes `@dataclass(frozen=True, eq=False)` with an `__eq__` which compares the id, the architecture and the arrays element by element, an `__hash__` of the id, and arrays which are copies of the ones it was built from and are read-only (`setflags(write=False)`). `cached_property` writes into `__dict__` and works on a frozen dataclass without slots; a pickled copy rebuilds the caches on first use. The one test which assigned `network.parameters` builds the network with the arrays instead.

- [ ] **Step 1: Apply this patch to `tests/sciml/test_network.py`**

Apply this patch to `tests/sciml/test_network.py`:

```diff
diff --git a/tests/sciml/test_network.py b/tests/sciml/test_network.py
index dca25bb..6c56e7b 100644
--- a/tests/sciml/test_network.py
+++ b/tests/sciml/test_network.py
@@ -1,7 +1,8 @@
 """Tests of a network: its files, its ids and its forward pass."""

+import pickle
 from collections.abc import Mapping
-from dataclasses import replace
+from dataclasses import FrozenInstanceError, replace
 from pathlib import Path
 from typing import Any

@@ -438,3 +439,60 @@ def test_the_ids_of_a_file_in_the_column_major_layout(tmp_path: Path) -> None:
     expected = _parameters()["layer1"]["weight"]
     expected[2, 1] = 60.0
     np.testing.assert_array_equal(parameters["layer1"]["weight"], expected)
+
+
+def test_a_network_does_not_change() -> None:
+    """An attribute cannot be assigned and an array cannot be written."""
+    network = Network(sid="net1", model=_model(), parameters=_parameters())
+    with pytest.raises(FrozenInstanceError):
+        network.sid = "net2"  # ty: ignore[invalid-assignment]
+    with pytest.raises(FrozenInstanceError):
+        network.parameters = {}  # ty: ignore[invalid-assignment]
+    with pytest.raises(ValueError, match="read-only"):
+        network.parameters["layer1"]["weight"][0, 0] = 5.0
+    assert network.parameters["layer1"]["weight"][0, 0] == 1.0
+
+
+def test_the_arrays_of_a_network_are_its_own() -> None:
+    """Writing into the arrays a network was built from does not change it."""
+    parameters = _parameters()
+    network = Network(sid="net1", model=_model(), parameters=parameters)
+    parameters["layer1"]["weight"][0, 0] = 5.0
+    assert network.parameters["layer1"]["weight"][0, 0] == 1.0
+    assert network.with_values({})["layer1"]["weight"].flags.writeable
+
+
+def test_the_equality_of_networks() -> None:
+    """Networks with one architecture and equal arrays are equal."""
+    network = Network(sid="net1", model=_model(), parameters=_parameters())
+    assert network == Network(sid="net1", model=_model(), parameters=_parameters())
+    assert hash(network) == hash(
+        Network(sid="net1", model=_model(), parameters=_parameters())
+    )
+    assert network != Network(sid="net1", model=_model())
+    other = _model()
+    other.layers[1].args = {"in_features": 3, "out_features": 1, "bias": True}
+    assert network != Network(sid="net1", model=other)
+    assert network != "net1"
+
+    changed = _parameters()
+    changed["layer2"]["weight"][0, 1] = 1.0
+    assert network != Network(sid="net1", model=_model(), parameters=changed)
+    missing = _parameters()
+    del missing["layer1"]["bias"]
+    assert network != Network(sid="net1", model=_model(), parameters=missing)
+
+
+def test_a_network_is_pickled() -> None:
+    """A fit pickles its networks for the workers, the copy is equal."""
+    network = Network(sid="net1", model=_model(), parameters=_parameters())
+    ids = network.parameter_ids()
+    copy = pickle.loads(pickle.dumps(network))
+    assert copy == network
+    assert copy.parameter_ids() == ids
+    np.testing.assert_array_equal(
+        copy.forward(np.array([0.5, -0.5]))[0],
+        network.forward(np.array([0.5, -0.5]))[0],
+    )
+    with pytest.raises(ValueError, match="read-only"):
+        copy.parameters["layer1"]["weight"][0, 0] = 5.0
```

- [ ] **Step 2: Apply this patch to `tests/sciml/test_layers_convolution.py`**

Apply this patch to `tests/sciml/test_layers_convolution.py`:

```diff
diff --git a/tests/sciml/test_layers_convolution.py b/tests/sciml/test_layers_convolution.py
index a6fa5d8..acbb862 100644
--- a/tests/sciml/test_layers_convolution.py
+++ b/tests/sciml/test_layers_convolution.py
@@ -275,10 +275,12 @@ def test_an_input_with_the_wrong_number_of_axes(

 def test_a_weight_of_the_wrong_shape(layer_model: Callable[..., NNModel]) -> None:
     """An array which does not fit the layer is an error of the import."""
-    network = Network(sid="net1", model=layer_model("Conv2d", CONV2D))
-    network.parameters = {"layer1": {"weight": np.ones((1, 1, 3, 2))}}
     with pytest.raises(NetworkImportError, match=r"'weight'.*\(1, 1, 3, 2\)"):
-        network.forward(np.ones((1, 1, 5, 5)))
+        Network(
+            sid="net1",
+            model=layer_model("Conv2d", CONV2D),
+            parameters={"layer1": {"weight": np.ones((1, 1, 3, 2))}},
+        )


 def test_an_unknown_padding_mode(
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest -q -x tests/sciml/test_network.py tests/sciml/test_layers_convolution.py`
Expected: FAIL: `test_a_network_does_not_change` (no `FrozenInstanceError`), `test_the_equality_of_networks` (`ValueError: The truth value of an array ... is ambiguous`), `test_a_network_is_pickled` (the copy is not equal)

- [ ] **Step 4: Apply this patch to `src/sbmlsim/sciml/network.py`**

Apply this patch to `src/sbmlsim/sciml/network.py`:

```diff
diff --git a/src/sbmlsim/sciml/network.py b/src/sbmlsim/sciml/network.py
index 11bbf0c..5556ee9 100644
--- a/src/sbmlsim/sciml/network.py
+++ b/src/sbmlsim/sciml/network.py
@@ -118,16 +118,22 @@ def load_array_data(path: Path) -> ArrayData:
     return data


-@dataclass
+@dataclass(frozen=True, eq=False)
 class Network:
     """The architecture and the arrays of one network.

     A network is validated when it is created: the id, the forward pass and
     the nominal values are checked against the architecture, and every layer
-    needs an implementation. `sid` and `model` do not change afterwards, the
+    needs an implementation. A network does not change afterwards: its
+    attributes cannot be assigned and its arrays cannot be written, so the
     structures derived from them (`array_specs`, `used_layers`,
-    `parameter_ids`) are computed once. A network with other nominal values is
-    a new network, e.g. `dataclasses.replace(network, parameters=...)`.
+    `parameter_ids`) are computed once and stay true. A network with other
+    nominal values is a new network, e.g.
+    `dataclasses.replace(network, parameters=...)`.
+
+    Two networks are equal when they have the same id, the same architecture
+    and the same arrays, which is what the round trip of a problem and the
+    pickling of a fit compare.

     Attributes:
         sid: id of the network, an SBML `SId` which is the `nn_model_id` of
@@ -163,6 +169,40 @@ class Network:
             )
         self.check_forward()
         self.check_arrays(self.parameters, complete=False)
+        # the arrays are the ones of the network alone and are not written
+        parameters = copy_parameters(self.parameters)
+        for arrays in parameters.values():
+            for array in arrays.values():
+                array.setflags(write=False)
+        object.__setattr__(self, "parameters", parameters)
+
+    def __eq__(self, other: object) -> bool:
+        """Check whether two networks have the same architecture and arrays.
+
+        Args:
+            other: the object the network is compared with.
+
+        Returns:
+            Whether the ids, the architectures and the arrays are equal, the
+            arrays element by element.
+        """
+        if not isinstance(other, Network):
+            return NotImplemented
+        if self.sid != other.sid or self.model != other.model:
+            return False
+        if self.parameters.keys() != other.parameters.keys():
+            return False
+        for layer, arrays in self.parameters.items():
+            others = other.parameters[layer]
+            if arrays.keys() != others.keys():
+                return False
+            if not all(np.array_equal(a, others[name]) for name, a in arrays.items()):
+                return False
+        return True
+
+    def __hash__(self) -> int:
+        """Get the hash of the id, equal networks have the same id."""
+        return hash(self.sid)

     @classmethod
     def from_files(
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/sciml`
Expected: PASS

- [ ] **Step 6: Run the checks**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: `All checks passed!`, `... files already formatted`, `All checks passed!` (zero diagnostics)

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "sciml: Network is frozen and compares its architecture and arrays"
```

### Task 2: The ids of units, inputs and outputs, and `SympyBackend`

**Files:**
- Modify: `src/sbmlsim/sciml/backend.py`
- Modify: `src/sbmlsim/sciml/network.py`
- Test: `tests/sciml/conftest.py`
- Test: `tests/sciml/test_backend.py`
- Test: `tests/sciml/test_expressions.py`
- Test (create): `tests/sciml/test_ids.py`

**Interfaces:**
- Consumes: `element_id`, `ID_SEPARATOR`, `INDEX_SEPARATOR`, `_NOT_SID` of `network.py`; `Backend`, `BackendKind` of `backend.py`.
- Produces: `index_id(index) -> str`, `unit_id(network, node, index) -> str`, `input_id(network, k, index=None) -> str`, `output_id(network, k, index) -> str`, `parse_io_id(network, kind, sid) -> tuple[int, tuple[int, ...] | None]` (raises `ValueError` naming the network and the form of an id); `SympyBackend()` with `kind == BackendKind.SYMPY` and `dtype is object`.

The ids of the design (`<net>__<node>__<index>`, `<net>__input<k>__<index>`, `<net>__input<k>` for an array, `<net>__output<k>__<index>`) get their functions next to `element_id`, and `parse_io_id` reads the position and the index back, which the hybridization and the compiler share. A value without axes has the index `0`. `SympyBackend` moves from the conftest of the tests into the package: it evaluates the elementwise functions with `numpy.frompyfunc`, a function of a value without axes returns an array of shape `()` and not a scalar, `select` is a `sympy.Piecewise` whose numbers are integers where they are exact (`0` instead of `0.0`), and `stabilizer` is zero. `gelu` with `approximate="none"` is evaluated on expressions as `sympy.erf`, which the MathML of SBML does not have: the test pins that it evaluates, and task 9 pins that the compiler rejects it.

- [ ] **Step 1: Create `tests/sciml/test_ids.py`**

Create `tests/sciml/test_ids.py`:

```python
"""Tests of the ids of the units, the inputs and the outputs of a network."""

import re

import pytest

from sbmlsim.sciml.network import (
    element_id,
    index_id,
    input_id,
    output_id,
    parse_io_id,
    unit_id,
)

#: an SBML `SId`
SID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def test_the_ids_of_the_design() -> None:
    """The ids are the ones of the table of the design."""
    assert element_id("net1", "layer1", "weight", (0, 1)) == "net1__layer1__weight__0_1"
    assert unit_id("net1", "tanh_1", (3,)) == "net1__tanh_1__3"
    assert input_id("net1", 0, (1,)) == "net1__input0__1"
    assert input_id("net1", 0) == "net1__input0"
    assert output_id("net1", 0, (0,)) == "net1__output0__0"


def test_the_index_of_an_id() -> None:
    """The axes are joined by `_`, a value without axes has the index `0`."""
    assert index_id((0, 1, 2)) == "0_1_2"
    assert index_id((7,)) == "7"
    assert index_id(()) == "0"
    assert unit_id("net1", "flatten", ()) == "net1__flatten__0"


@pytest.mark.parametrize(
    "sid",
    [
        unit_id("net1", "block.0", (1, 2)),
        unit_id("net1", "layer-1", (0,)),
        input_id("net.1", 2, (0, 0)),
        output_id("net1", 1, (3,)),
    ],
)
def test_an_id_is_an_sid(sid: str) -> None:
    """A character which is not part of an `SId` is replaced."""
    assert SID.fullmatch(sid)


def test_a_negative_position_or_index() -> None:
    """An id of a negative position or axis does not exist."""
    with pytest.raises(ValueError, match=r"Network 'net1'.*input '-1'"):
        input_id("net1", -1, (0,))
    with pytest.raises(ValueError, match=r"Network 'net1'.*output '-2'"):
        output_id("net1", -2, (0,))
    with pytest.raises(ValueError, match=r"\(0, -1\).*negative"):
        unit_id("net1", "tanh", (0, -1))


@pytest.mark.parametrize(
    ("kind", "k", "index"),
    [
        ("input", 0, (1,)),
        ("input", 12, (0, 3, 10)),
        ("input", 1, None),
        ("output", 0, (0,)),
        ("output", 3, (2, 1)),
    ],
)
def test_an_id_is_read_back(kind: str, k: int, index: tuple[int, ...] | None) -> None:
    """The position and the index of an id are the ones it was built from."""
    sid = (
        input_id("net1", k, index)
        if kind == "input"
        else output_id("net1", k, index or ())
    )
    assert parse_io_id("net1", kind, sid) == (k, index)


@pytest.mark.parametrize(
    "sid",
    [
        "net1__input0__",
        "net1__input__0",
        "net1__inputs0__0",
        "net1__input0__a",
        "net1__input0__0_",
        "net1__input0__0__1",
        "net1_input0__0",
        "net2__input0__0",
        "xnet1__input0__0",
        "net1__output0__0",
        "prey",
        "",
    ],
)
def test_an_id_which_is_not_an_input(sid: str) -> None:
    """An id which is not the id of an input names the form of one."""
    with pytest.raises(ValueError, match=r"Network 'net1'.*net1__input0__1") as excinfo:
        parse_io_id("net1", "input", sid)
    assert f"'{sid}'" in str(excinfo.value)


def test_the_id_of_a_network_with_a_pattern_character() -> None:
    """The id of the network is text and not a pattern."""
    assert parse_io_id("net.1", "input", "net.1__input0__1") == (0, (1,))
    with pytest.raises(ValueError, match="is not the id of an input"):
        parse_io_id("net.1", "input", "netx1__input0__1")
```

- [ ] **Step 2: Apply this patch to `tests/sciml/test_backend.py`**

Apply this patch to `tests/sciml/test_backend.py`:

```diff
diff --git a/tests/sciml/test_backend.py b/tests/sciml/test_backend.py
index aa2eae6..a2de1cc 100644
--- a/tests/sciml/test_backend.py
+++ b/tests/sciml/test_backend.py
@@ -1,8 +1,10 @@
-"""Tests of the backend on arrays of numbers."""
+"""Tests of the backends on arrays of numbers and of expressions."""

 import inspect

 import numpy as np
+import pytest
+import sympy

 from sbmlsim.sciml.backend import (
     ALL_BACKENDS,
@@ -10,6 +12,7 @@ from sbmlsim.sciml.backend import (
     Backend,
     BackendKind,
     NumpyBackend,
+    SympyBackend,
 )


@@ -81,3 +84,76 @@ def test_the_shift_of_softmax() -> None:
     x = np.array([[1.0, 5.0, 3.0], [7.0, 2.0, 4.0]])
     np.testing.assert_array_equal(backend.stabilizer(x, 1), [[5.0], [7.0]])
     np.testing.assert_array_equal(backend.stabilizer(x, 0), [[7.0, 5.0, 4.0]])
+
+
+# --- THE BACKEND ON EXPRESSIONS ---
+
+
+def test_the_backend_on_expressions() -> None:
+    """The backend on expressions is the second kind of backend."""
+    assert SympyBackend.kind == BackendKind.SYMPY
+    assert SympyBackend.dtype is object
+    assert not inspect.isabstract(SympyBackend)
+    x = SympyBackend().asarray([sympy.Symbol("a"), 1.5])
+    assert x.dtype == object
+    assert x.shape == (2,)
+
+
+@pytest.mark.parametrize(
+    ("name", "expected"),
+    [
+        ("exp", sympy.exp),
+        ("log", sympy.log),
+        ("tanh", sympy.tanh),
+        ("sqrt", sympy.sqrt),
+        ("erf", sympy.erf),
+        ("absolute", sympy.Abs),
+    ],
+)
+def test_the_elementwise_functions_on_expressions(name: str, expected: object) -> None:
+    """A function is applied to every element and keeps the shape."""
+    backend = SympyBackend()
+    a, b = sympy.symbols("a b")
+    x = np.array([[a, b], [a + b, 2 * a]], dtype=object)
+    y = getattr(backend, name)(x)
+    assert y.dtype == object
+    assert y.shape == (2, 2)
+    assert y[1, 0] == expected(a + b)  # ty: ignore[call-non-callable]
+
+
+def test_a_function_of_a_value_without_axes_is_an_array() -> None:
+    """numpy returns a scalar for an array without axes, the backend an array."""
+    backend = SympyBackend()
+    x = np.array(sympy.Symbol("a"), dtype=object)
+    for y in (backend.tanh(x), backend.select(x, 0.0, 0.0, x)):
+        assert isinstance(y, np.ndarray)
+        assert y.shape == ()
+        assert y.dtype == object
+
+
+def test_a_condition_is_a_piecewise() -> None:
+    """`select` is `above` where `x > threshold` and `below` elsewhere."""
+    backend = SympyBackend()
+    a = sympy.Symbol("a")
+    (y,) = backend.select(np.array([a], dtype=object), 0.0, 0.0, np.array([a]))
+    assert y == sympy.Piecewise((a, a > 0), (0, True))
+    assert y.subs(a, 2.0) == 2.0
+    assert y.subs(a, 0.0) == 0
+    assert y.subs(a, -2.0) == 0
+    # a number which is an integer is written as one, the others as they are
+    assert y.atoms(sympy.Float) == set()
+    (z,) = backend.select(np.array([a], dtype=object), 0.5, -1.25, 6.0)
+    assert z == sympy.Piecewise((6, a > 0.5), (-1.25, True))
+
+
+def test_a_condition_on_a_number() -> None:
+    """An element which is a number is evaluated."""
+    backend = SympyBackend()
+    y = backend.select(np.array([2.0, -1.0], dtype=object), 0.0, 0.0, 7.0)
+    assert list(y) == [7, 0]
+
+
+def test_an_expression_needs_no_shift() -> None:
+    """`softmax` on expressions is not shifted by a maximum."""
+    x = np.array([sympy.Symbol("a"), sympy.Symbol("b")], dtype=object)
+    assert SympyBackend().stabilizer(x, 0) == 0.0
```

- [ ] **Step 3: Apply this patch to `tests/sciml/test_expressions.py`**

Apply this patch to `tests/sciml/test_expressions.py`:

```diff
diff --git a/tests/sciml/test_expressions.py b/tests/sciml/test_expressions.py
index 287bbbf..86f13bc 100644
--- a/tests/sciml/test_expressions.py
+++ b/tests/sciml/test_expressions.py
@@ -282,3 +282,38 @@ def test_a_network_is_compiled_node_by_node(
     np.testing.assert_allclose(
         observed, network.forward(point)[0], rtol=TOLERANCE, atol=TOLERANCE
     )
+
+
+@pytest.mark.parametrize("approximate", ["none", "tanh"])
+def test_gelu_on_expressions(
+    approximate: str,
+    sympy_backend: Backend,
+    symbolic: Callable[[str, tuple[int, ...]], np.ndarray],
+    rng: np.random.Generator,
+) -> None:
+    """Both forms of `gelu` are evaluated on expressions.
+
+    The form with the error function is an expression of sympy, which the
+    MathML of SBML does not have: the compilation of a network rejects it,
+    see `tests/sciml/test_compiler.py`.
+    """
+    model = NNModel(
+        nn_model_id="net1",
+        inputs=[Input(input_id="input0")],
+        layers=[],
+        forward=[
+            *_placeholders(1),
+            Node(
+                name="f",
+                op="call_function",
+                target="gelu",
+                args=["x0"],
+                kwargs={"approximate": approximate},
+            ),
+            _output("f"),
+        ],
+    )
+    x = symbolic("x0", (4,))
+    (expressions,) = evaluate(model, {}, [x], sympy_backend)
+    assert bool(expressions[0].has(sympy.erf)) == (approximate == "none")
+    _compare(model, {}, [x], sympy_backend, rng)
```

- [ ] **Step 4: Apply this patch to `tests/sciml/conftest.py`**

Apply this patch to `tests/sciml/conftest.py`:

```diff
diff --git a/tests/sciml/conftest.py b/tests/sciml/conftest.py
index 3d0218e..41d675c 100644
--- a/tests/sciml/conftest.py
+++ b/tests/sciml/conftest.py
@@ -6,14 +6,14 @@ tests which need it are skipped without it.
 """

 from collections.abc import Callable
-from typing import Any, ClassVar
+from typing import Any

 import numpy as np
 import pytest
 import sympy
 from petab_sciml import Input, Layer, NNModel, Node

-from sbmlsim.sciml.backend import Backend, BackendKind, NumpyBackend
+from sbmlsim.sciml.backend import NumpyBackend, SympyBackend
 from sbmlsim.sciml.interpreter import evaluate

 #: absolute and relative tolerance of the comparison with PyTorch, which runs
@@ -21,56 +21,6 @@ from sbmlsim.sciml.interpreter import evaluate
 TOLERANCE = 1e-10


-def _elementwise(function: Callable[..., Any], n_args: int = 1) -> np.ufunc:
-    """Apply a sympy function to every element of `object` arrays."""
-    return np.frompyfunc(function, n_args, 1)
-
-
-class SympyBackend(Backend):
-    """A backend on `object` arrays of sympy expressions.
-
-    It is the backend the compilation of a network into expressions needs,
-    reduced to what the tests of the layers and functions which declare
-    `BackendKind.SYMPY` evaluate with.
-    """
-
-    kind: ClassVar[BackendKind] = BackendKind.SYMPY
-    dtype: ClassVar[type] = object
-
-    def exp(self, x: np.ndarray) -> np.ndarray:
-        return _elementwise(sympy.exp)(x)
-
-    def log(self, x: np.ndarray) -> np.ndarray:
-        return _elementwise(sympy.log)(x)
-
-    def tanh(self, x: np.ndarray) -> np.ndarray:
-        return _elementwise(sympy.tanh)(x)
-
-    def sqrt(self, x: np.ndarray) -> np.ndarray:
-        return _elementwise(sympy.sqrt)(x)
-
-    def erf(self, x: np.ndarray) -> np.ndarray:
-        return _elementwise(sympy.erf)(x)
-
-    def absolute(self, x: np.ndarray) -> np.ndarray:
-        return _elementwise(sympy.Abs)(x)
-
-    def select(
-        self,
-        x: np.ndarray,
-        threshold: float,
-        below: np.ndarray | float,
-        above: np.ndarray | float,
-    ) -> np.ndarray:
-        def piecewise(value: Any, low: Any, high: Any) -> sympy.Expr:
-            return sympy.Piecewise((high, value > threshold), (low, True))
-
-        return _elementwise(piecewise, 3)(x, below, above)
-
-    def stabilizer(self, x: np.ndarray, axis: int) -> np.ndarray | float:
-        return 0.0
-
-
 def symbols(prefix: str, shape: tuple[int, ...]) -> np.ndarray:
     """Get an `object` array of symbols, named by the prefix and the index."""
     array = np.empty(shape, dtype=object)
```

- [ ] **Step 5: Run the tests to verify they fail**

Run: `uv run pytest -q -x tests/sciml/test_ids.py tests/sciml/test_backend.py tests/sciml/test_expressions.py`
Expected: FAIL at collection: `ImportError: cannot import name 'index_id' from 'sbmlsim.sciml.network'` and `cannot import name 'SympyBackend' from 'sbmlsim.sciml.backend'`

- [ ] **Step 6: Apply this patch to `src/sbmlsim/sciml/network.py`**

Apply this patch to `src/sbmlsim/sciml/network.py`:

```diff
diff --git a/src/sbmlsim/sciml/network.py b/src/sbmlsim/sciml/network.py
index 5556ee9..d231bef 100644
--- a/src/sbmlsim/sciml/network.py
+++ b/src/sbmlsim/sciml/network.py
@@ -6,7 +6,18 @@ in the PyTorch layout. `Network.forward` evaluates it with numpy.

 Every element of an array has an id, `<net>__<layer>__<array>__<index>` with
 the PyTorch index of the element and `_` between the axes, e.g.
-`net1__layer1__weight__0_1`. The id is a valid SBML `SId`.
+`net1__layer1__weight__0_1`. The units of the nodes of the forward pass, the
+inputs and the outputs have ids of the same form:
+
+| entity | id | function |
+| --- | --- | --- |
+| element of an array | `net1__layer1__weight__0_1` | `element_id` |
+| unit of a node | `net1__tanh_1__3` | `unit_id` |
+| input | `net1__input0__1`, `net1__input0` for the array | `input_id` |
+| output | `net1__output0__0` | `output_id` |
+
+Every id is a valid SBML `SId`, the compilation of a network into a model
+and the hybridization of a problem share these functions.
 """

 from __future__ import annotations
@@ -75,6 +86,120 @@ def element_id(network: str, layer: str, array: str, index: tuple[int, ...]) ->
     return _NOT_SID.sub("_", sid)


+def index_id(index: tuple[int, ...]) -> str:
+    """Get the part of an id which is the index of an element.
+
+    Args:
+        index: the PyTorch index of the element, empty for a value without
+            axes.
+
+    Returns:
+        The axes joined by `_`, e.g. `0_1`, and `0` for a value without axes.
+
+    Raises:
+        ValueError: if an axis of the index is negative.
+    """
+    if any(i < 0 for i in index):
+        raise ValueError(f"The index {index} has a negative axis")
+    return INDEX_SEPARATOR.join(str(int(i)) for i in index) if index else "0"
+
+
+def unit_id(network: str, node: str, index: tuple[int, ...]) -> str:
+    """Get the id of a unit of a node of the forward pass.
+
+    Args:
+        network: id of the network.
+        node: name of the node.
+        index: the index of the unit in the value of the node.
+
+    Returns:
+        The id, e.g. `net1__tanh_1__3`. A character which is not part of an
+        SBML `SId` is replaced by `_`.
+    """
+    return _NOT_SID.sub("_", ID_SEPARATOR.join([network, node, index_id(index)]))
+
+
+def _io_id(network: str, kind: str, k: int, index: tuple[int, ...] | None) -> str:
+    """Get the id of an input or an output, see `input_id` and `output_id`."""
+    if k < 0:
+        raise ValueError(f"Network '{network}': the {kind} '{k}' is negative")
+    parts = [network, f"{kind}{int(k)}"]
+    if index is not None:
+        parts.append(index_id(index))
+    return _NOT_SID.sub("_", ID_SEPARATOR.join(parts))
+
+
+def input_id(network: str, k: int, index: tuple[int, ...] | None = None) -> str:
+    """Get the id of an input of a network or of an element of it.
+
+    Args:
+        network: id of the network.
+        k: the position of the input in the inputs of the forward pass.
+        index: the index of the element, `None` for the input as an array.
+
+    Returns:
+        The id, e.g. `net1__input0__1` for an element and `net1__input0` for
+        the array.
+
+    Raises:
+        ValueError: if the position or an axis of the index is negative.
+    """
+    return _io_id(network, "input", k, index)
+
+
+def output_id(network: str, k: int, index: tuple[int, ...]) -> str:
+    """Get the id of an element of an output of a network.
+
+    Args:
+        network: id of the network.
+        k: the position of the output in the outputs of the forward pass.
+        index: the index of the element.
+
+    Returns:
+        The id, e.g. `net1__output0__0`.
+
+    Raises:
+        ValueError: if the position or an axis of the index is negative.
+    """
+    return _io_id(network, "output", k, index)
+
+
+def parse_io_id(
+    network: str, kind: str, sid: str
+) -> tuple[int, tuple[int, ...] | None]:
+    """Read the position and the index from the id of an input or an output.
+
+    Args:
+        network: id of the network.
+        kind: `input` or `output`.
+        sid: the id, e.g. `net1__input0__1` or `net1__input0`.
+
+    Returns:
+        The position and the index of the element, `None` as index for an
+        id without one, i.e. for an input as an array.
+
+    Raises:
+        ValueError: if the id is not the id of an input or output of the
+            network.
+    """
+    pattern = re.compile(
+        rf"{re.escape(network)}{ID_SEPARATOR}{kind}(?P<k>\d+)"
+        rf"(?:{ID_SEPARATOR}(?P<index>\d+(?:{INDEX_SEPARATOR}\d+)*))?"
+    )
+    match = pattern.fullmatch(sid)
+    if match is None:
+        raise ValueError(
+            f"Network '{network}': '{sid}' is not the id of an {kind}, which is "
+            f"'{network}{ID_SEPARATOR}{kind}<k>{ID_SEPARATOR}<index>', e.g. "
+            f"'{_io_id(network, kind, 0, (1,))}'"
+        )
+    index = match.group("index")
+    return (
+        int(match.group("k")),
+        None if index is None else tuple(int(i) for i in index.split(INDEX_SEPARATOR)),
+    )
+
+
 def copy_parameters(
     parameters: Mapping[str, Mapping[str, np.ndarray]],
 ) -> NetworkParameters:
```

- [ ] **Step 7: Apply this patch to `src/sbmlsim/sciml/backend.py`**

Apply this patch to `src/sbmlsim/sciml/backend.py`:

```diff
diff --git a/src/sbmlsim/sciml/backend.py b/src/sbmlsim/sciml/backend.py
index bb1776d..b56b582 100644
--- a/src/sbmlsim/sciml/backend.py
+++ b/src/sbmlsim/sciml/backend.py
@@ -6,19 +6,21 @@ A layer is implemented once. It uses the array operations of numpy (`@`,
 between the two from a `Backend`: the elementwise functions and the functions
 with a condition.

-`NumpyBackend` works on `float` arrays and is the forward pass. A backend on
-sympy expressions is the first half of the compilation of a network into an
-SBML model; it subclasses `Backend`, sets `kind` to `BackendKind.SYMPY` and
-implements the abstract methods, the layers do not change.
+`NumpyBackend` works on `float` arrays and is the forward pass.
+`SympyBackend` works on `object` arrays of sympy expressions and is the first
+half of the compilation of a network into an SBML model, see
+`sbmlsim.sciml.compiler`. The layers are the same for both.
 """

 from __future__ import annotations

 from abc import ABC, abstractmethod
+from collections.abc import Callable
 from enum import StrEnum
 from typing import Any, ClassVar

 import numpy as np
+import sympy
 from scipy import special


@@ -173,3 +175,90 @@ class NumpyBackend(Backend):
     def stabilizer(self, x: np.ndarray, axis: int) -> np.ndarray | float:
         """Get the maximum along the axis."""
         return np.max(x, axis=axis, keepdims=True)
+
+
+def _elementwise(function: Callable[..., Any], n_args: int = 1) -> np.ufunc:
+    """Get the function which applies a function to every element of arrays.
+
+    Args:
+        function: the function of `n_args` elements.
+        n_args: the number of arrays the function takes an element of.
+
+    Returns:
+        The function of `object` arrays, with the broadcasting of numpy.
+    """
+    return np.frompyfunc(function, n_args, 1)
+
+
+def _exact(value: Any) -> Any:
+    """Get a number as the integer it is, which keeps the expressions short.
+
+    Args:
+        value: a number or an expression.
+
+    Returns:
+        The integer for a float without a fraction, e.g. `0` for `0.0`, the
+        value otherwise.
+    """
+    if isinstance(value, float | np.floating) and float(value).is_integer():
+        return sympy.Integer(int(value))
+    return value
+
+
+class SympyBackend(Backend):
+    """The backend on `object` arrays of sympy expressions.
+
+    The elements of the arrays are symbols, numbers and expressions of them.
+    A function with a condition is a `sympy.Piecewise`, which is the
+    `piecewise` of the MathML of SBML. `erf` is evaluated as `sympy.erf`,
+    which the MathML of SBML does not have: the compiler rejects an expression
+    with it.
+    """
+
+    kind: ClassVar[BackendKind] = BackendKind.SYMPY
+    dtype: ClassVar[type] = object
+
+    def exp(self, x: np.ndarray) -> np.ndarray:
+        """Calculate the exponential function."""
+        return np.asarray(_elementwise(sympy.exp)(x), dtype=object)
+
+    def log(self, x: np.ndarray) -> np.ndarray:
+        """Calculate the natural logarithm."""
+        return np.asarray(_elementwise(sympy.log)(x), dtype=object)
+
+    def tanh(self, x: np.ndarray) -> np.ndarray:
+        """Calculate the hyperbolic tangent."""
+        return np.asarray(_elementwise(sympy.tanh)(x), dtype=object)
+
+    def sqrt(self, x: np.ndarray) -> np.ndarray:
+        """Calculate the square root."""
+        return np.asarray(_elementwise(sympy.sqrt)(x), dtype=object)
+
+    def erf(self, x: np.ndarray) -> np.ndarray:
+        """Calculate the error function."""
+        return np.asarray(_elementwise(sympy.erf)(x), dtype=object)
+
+    def absolute(self, x: np.ndarray) -> np.ndarray:
+        """Calculate the absolute value."""
+        return np.asarray(_elementwise(sympy.Abs)(x), dtype=object)
+
+    def select(
+        self,
+        x: np.ndarray,
+        threshold: float,
+        below: np.ndarray | float,
+        above: np.ndarray | float,
+    ) -> np.ndarray:
+        """Choose between two values by a condition on `x`, as a `Piecewise`."""
+        limit = _exact(threshold)
+
+        def piecewise(value: Any, low: Any, high: Any) -> sympy.Basic:
+            return sympy.Piecewise(
+                (_exact(high), sympy.sympify(value) > limit), (_exact(low), True)
+            )
+
+        return np.asarray(_elementwise(piecewise, 3)(x, below, above), dtype=object)
+
+    def stabilizer(self, x: np.ndarray, axis: int) -> np.ndarray | float:
+        """Get zero, an expression has no maximum and needs no shift."""
+        return 0.0
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/sciml`
Expected: PASS

- [ ] **Step 9: Run the checks**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: `All checks passed!`, `... files already formatted`, `All checks passed!` (zero diagnostics)

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "sciml: the ids of units, inputs and outputs, and the backend on expressions"
```

### Task 3: Formulas of SBML as sympy expressions, and `sbmlmath` as a dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/sbmlsim/mathml.py`
- Test (create): `tests/test_mathml_formula.py`

**Interfaces:**
- Consumes: `libsbml`, `sbmlmath.SBMLMathMLParser`, `sbmlmath.SBMLMathMLPrinter`, `sbmlmath.TimeSymbol`.
- Produces: `sbmlsim.mathml.TIME = "time"`, `formula_expression(formula) -> sympy.Basic`, `formula_symbols(formula) -> set[str]`, `evaluate_formula(formula, variables) -> Any`, `expression_to_astnode(expression) -> libsbml.ASTNode`, `expression_to_formula(expression) -> str`; every one raises `ValueError` which names the formula or the expression. `mathml.evaluate` stays as it is for `Data` of type `FUNCTION`.

The spec evaluates the formulas of the inputs of a network with `mathml.evaluate`. That function hands the text of a formula to `sympify`, which reads `beta`, `gamma`, `lambda`, `S`, `I` and `E` as the functions and constants of sympy: `mathml.evaluate(formula_to_astnode("beta*2"), {"beta": 2.0})` raises `TypeError` and `gamma + 1` as well, i.e. the parameters of the Lotka-Volterra model of every case of the test suite. The new functions read a formula with `libsbml.parseL3Formula` and the MathML parser of `sbmlmath`, which makes a symbol of every identifier; the way goes through the text of the MathML because libsbml and libsedml both wrap `ASTNode` and the python class of a node is the one of the library which was imported last. `expression_to_astnode` and `expression_to_formula` are the way back for the compiler and the reader: the math of PEtab differs from the math of a formula of SBML (`log` is the natural logarithm in PEtab and `log10` in a formula of SBML, and `log(a, b)` puts the base first in SBML), so a sympy expression is printed as MathML without units of its numbers (`literals_dimensionless=False`, valid in SBML level 2) and read by libsbml. The functions of a formula are compiled once (`functools.lru_cache`), a fit evaluates them in every simulation. `sbmlmath` was installed as a dependency of a dependency and becomes a declared one; `uv.lock` is not tracked, `uv sync --extra dev` picks it up.

- [ ] **Step 1: Create `tests/test_mathml_formula.py`**

Create `tests/test_mathml_formula.py`:

```python
"""Tests of the formulas of SBML which are read with libsbml and sbmlmath."""

from typing import Any

import libsbml
import numpy as np
import pytest
import sympy

from sbmlsim.mathml import (
    evaluate_formula,
    expression_to_astnode,
    expression_to_formula,
    formula_expression,
    formula_symbols,
)


@pytest.mark.parametrize(
    ("formula", "variables", "expected"),
    [
        ("alpha", {"alpha": 1.3}, 1.3),
        ("prey + (alpha - 1.3)", {"prey": 1.0, "alpha": 2.0}, 1.7),
        ("0.5", {}, 0.5),
        ("2 * x^2", {"x": 3.0}, 18.0),
        ("piecewise(1, x > 2, 0)", {"x": 3.0}, 1.0),
        ("piecewise(1, x > 2, 0)", {"x": 1.0}, 0.0),
        ("exp(x) + ln(y)", {"x": 0.0, "y": 1.0}, 1.0),
        ("log10(x)", {"x": 100.0}, 2.0),
        ("log(x)", {"x": 100.0}, 2.0),
        ("time / 2", {"time": 4.0}, 2.0),
        ("pi", {}, np.pi),
    ],
)
def test_a_formula_is_evaluated(
    formula: str, variables: dict[str, float], expected: float
) -> None:
    """A formula is evaluated on the values of its identifiers."""
    assert evaluate_formula(formula, variables) == pytest.approx(expected)


@pytest.mark.parametrize(
    "symbol", ["beta", "gamma", "lambda", "S", "I", "E", "N", "O", "Q", "zeta", "re"]
)
def test_an_identifier_which_is_a_name_of_sympy(symbol: str) -> None:
    """An identifier of a model is a symbol, whatever sympy calls by its name.

    `sympify` reads `beta` and `gamma` as functions and `S`, `I` and `E` as
    its registry, the imaginary unit and the number of Euler, which
    `sbmlsim.mathml.expr_from_formula` hands the text of a formula to.
    """
    assert formula_symbols(f"2 * {symbol} + 1") == {symbol}
    assert evaluate_formula(f"2 * {symbol} + 1", {symbol: 3.0}) == pytest.approx(7.0)


def test_the_values_of_a_formula_are_arrays() -> None:
    """A formula of arrays is an array."""
    values = evaluate_formula("a * time", {"a": 2.0, "time": np.array([1.0, 2.0])})
    np.testing.assert_allclose(values, [2.0, 4.0])


def test_the_symbols_of_a_formula() -> None:
    """The identifiers are the symbols, the time of the model is `time`."""
    assert formula_symbols("prey + (alpha - 1.3) * time") == {"prey", "alpha", "time"}
    assert formula_symbols("0.5") == set()
    assert formula_symbols("pi * 2") == set()


def test_values_which_are_not_used() -> None:
    """The values of other identifiers are ignored."""
    assert evaluate_formula("a", {"a": 1.0, "b": 2.0}) == 1.0


@pytest.mark.parametrize("formula", ["", "  ", "x +", "1 +* 2", "(x"])
def test_a_formula_which_is_not_math(formula: str) -> None:
    """A formula which cannot be read names itself."""
    with pytest.raises(ValueError, match=r"The formula '.*' is (empty|not valid math)"):
        formula_expression(formula)


@pytest.mark.parametrize("formula", [None, 1.0, ["x"]])
def test_a_formula_which_is_not_text(formula: Any) -> None:
    """A formula is text, a number is not read as one."""
    with pytest.raises(ValueError, match="is empty"):
        formula_expression(formula)


@pytest.mark.parametrize("formula", ["f(x)", "rateOf(x)", "delay(x, 1)"])
def test_a_function_without_a_value(formula: str) -> None:
    """A function which sympy cannot evaluate is an error and not a symbol."""
    with pytest.raises(ValueError, match="uses the functions"):
        formula_expression(formula)


def test_an_identifier_without_a_value() -> None:
    """An identifier without a value names itself and the values."""
    with pytest.raises(ValueError, match=r"'a \+ b' uses \['b'\].*\['a', 'c'\]"):
        evaluate_formula("a + b", {"a": 1.0, "c": 2.0})


@pytest.mark.parametrize(
    ("expression", "formula"),
    [
        (sympy.log(sympy.Symbol("x")), "ln(x)"),
        (sympy.tanh(sympy.Symbol("x")), "tanh(x)"),
        (sympy.Symbol("x") ** 2, "x^2"),
        (sympy.Float(1.5) * sympy.Symbol("net1__layer1__weight__0_1"), None),
        (sympy.Piecewise((sympy.Symbol("x"), sympy.Symbol("x") > 0), (0, True)), None),
        (sympy.Abs(sympy.Symbol("x")), "abs(x)"),
        (sympy.exp(-sympy.Symbol("x")), "exp(-x)"),
    ],
)
def test_an_expression_is_a_formula(
    expression: sympy.Basic, formula: str | None
) -> None:
    """An expression is written as a formula which has its value."""
    written = expression_to_formula(expression)
    if formula is not None:
        assert written == formula
    # no unit of a number, which the math of SBML level 2 does not have
    assert "dimensionless" not in written
    symbols = sorted(expression.free_symbols, key=str)
    for value in (-0.75, 0.5, 2.0):
        if expression.has(sympy.log) and value <= 0:
            continue
        variables = {str(symbol): value for symbol in symbols}
        expected = float(sympy.N(expression.subs(dict.fromkeys(symbols, value))))
        assert evaluate_formula(written, variables) == pytest.approx(expected)


def test_the_logarithm_of_a_formula_is_not_the_natural_logarithm() -> None:
    """`log` is the logarithm to the base 10 in a formula of SBML."""
    x = sympy.Symbol("x")
    assert expression_to_formula(sympy.log(x)) == "ln(x)"
    assert evaluate_formula("ln(x)", {"x": np.e}) == pytest.approx(1.0)
    assert evaluate_formula("log(x)", {"x": 10.0}) == pytest.approx(1.0)


def test_an_expression_is_math_of_sbml() -> None:
    """The syntax tree is what a rule of a model takes."""
    a, b = sympy.symbols("a b")
    astnode = expression_to_astnode(a * b + 1)
    assert astnode.isWellFormedASTNode()
    document = libsbml.SBMLDocument(2, 4)
    model = document.createModel()
    for sid in ("a", "b", "c"):
        parameter = model.createParameter()
        parameter.setId(sid)
        parameter.setConstant(sid != "c")
        parameter.setValue(2.0)
    rule = model.createAssignmentRule()
    rule.setVariable("c")
    assert rule.setMath(astnode) == libsbml.LIBSBML_OPERATION_SUCCESS
    assert "units" not in libsbml.writeSBMLToString(document)


def test_an_expression_without_mathml() -> None:
    """The error function is not a function of the MathML of SBML."""
    with pytest.raises(ValueError, match=r"'erf\(x\)' has no MathML.*\['erf'\]"):
        expression_to_astnode(sympy.erf(sympy.Symbol("x")))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -x tests/test_mathml_formula.py`
Expected: FAIL at collection: `ImportError: cannot import name 'evaluate_formula' from 'sbmlsim.mathml'`

- [ ] **Step 3: Apply this patch to `src/sbmlsim/mathml.py`**

Apply this patch to `src/sbmlsim/mathml.py`:

```diff
diff --git a/src/sbmlsim/mathml.py b/src/sbmlsim/mathml.py
index a803a0b..94755f6 100644
--- a/src/sbmlsim/mathml.py
+++ b/src/sbmlsim/mathml.py
@@ -4,16 +4,32 @@ A `Data` of type `FUNCTION` is a formula of other data, see `sbmlsim.data`.
 The formula is parsed into the abstract syntax tree of libsedml, which
 implements the L3 formula syntax of SBML, and evaluated on the arrays of the
 variables with sympy.
+
+`formula_expression`, `formula_symbols` and `evaluate_formula` read a formula
+with libsbml and `sbmlmath`, which make a symbol of every identifier.
+`expr_from_formula` hands the text of a formula to `sympify`, which reads
+`beta`, `gamma`, `lambda`, `S` or `I` as the functions and constants of sympy
+and not as identifiers of a model. `expression_to_astnode` and
+`expression_to_formula` are the way back, from a sympy expression to the math
+of SBML.
 """

+import functools
 import logging
+from collections.abc import Mapping
 from typing import Any

+import libsbml
 import libsedml
+import sympy
+from sbmlmath import SBMLMathMLParser, SBMLMathMLPrinter, TimeSymbol
 from sympy import lambdify, sympify

 logger = logging.getLogger(__name__)

+#: the name of the time of the model in the variables of a formula
+TIME = "time"
+

 def formula_to_astnode(formula: str) -> libsedml.ASTNode:
     """Parse ASTNode from formula."""
@@ -201,3 +217,195 @@ if __name__ == "__main__":
     * The Boolean function symbols '&&' (and), '||' (or), '!' (not),
     and '!=' (not equals) may be used.
     """
+
+
+def formula_expression(formula: str) -> sympy.Basic:
+    """Parse an L3 formula of SBML into a sympy expression.
+
+    Every identifier of the formula is a symbol of its name, also the
+    identifiers which are functions or constants of sympy (`beta`, `gamma`,
+    `lambda`, `S`, `I`). The time of the model is the symbol `time`.
+
+    Args:
+        formula: the formula, e.g. `prey + (alpha - 1.3)`.
+
+    Returns:
+        The expression.
+
+    Raises:
+        ValueError: if the formula is empty, is not valid math or uses a
+            function which is not a function of the MathML of SBML.
+    """
+    if not isinstance(formula, str) or not formula.strip():
+        raise ValueError(f"The formula '{formula}' is empty")
+    astnode: libsbml.ASTNode | None = libsbml.parseL3Formula(formula)
+    if astnode is None:
+        raise ValueError(
+            f"The formula '{formula}' is not valid math: "
+            f"{libsbml.getLastParseL3Error()}"
+        )
+    try:
+        # through the text of the MathML: libsbml and libsedml both wrap the
+        # syntax tree, and the class of a node is the one of the library which
+        # was imported last
+        mathml = libsbml.writeMathMLToString(astnode)
+        expression = sympy.sympify(
+            SBMLMathMLParser(ignore_units=True).parse_str(mathml)
+        )
+    except Exception as err:
+        raise ValueError(
+            f"The formula '{formula}' cannot be evaluated: {type(err).__name__}: {err}"
+        ) from err
+    functions = sorted(
+        {
+            str(function.func)
+            for function in expression.atoms(sympy.Function)
+            if isinstance(function, sympy.core.function.AppliedUndef)
+        }
+    )
+    if functions:
+        raise ValueError(
+            f"The formula '{formula}' uses the functions {functions}, which "
+            f"have no value: they are not functions of the MathML of SBML, or "
+            f"functions of a simulation"
+        )
+    return expression.subs(
+        {
+            symbol: sympy.Symbol(TIME)
+            for symbol in expression.free_symbols
+            if isinstance(symbol, TimeSymbol)
+        }
+    )
+
+
+def formula_symbols(formula: str) -> set[str]:
+    """Get the identifiers a formula uses.
+
+    Args:
+        formula: the formula, an L3 formula of SBML.
+
+    Returns:
+        The names of the symbols of the formula, `time` for the time of the
+        model.
+
+    Raises:
+        ValueError: if the formula is not valid math, see
+            `formula_expression`.
+    """
+    return {str(symbol) for symbol in formula_expression(formula).free_symbols}
+
+
+@functools.lru_cache(maxsize=1024)
+def _formula_function(formula: str) -> tuple[tuple[str, ...], Any]:
+    """Get the function of a formula and the names of its arguments.
+
+    The function is built once per formula: a fit evaluates the formulas of
+    its derived changes in every simulation.
+
+    Args:
+        formula: the formula.
+
+    Returns:
+        The names of the symbols of the formula and the function of their
+        values, in this order.
+
+    Raises:
+        ValueError: if the formula is not valid math, see
+            `formula_expression`.
+    """
+    expression = formula_expression(formula)
+    symbols = sorted(expression.free_symbols, key=str)
+    # the symbols are arguments by position: an identifier of a model is not
+    # always a name of python, e.g. `lambda`
+    arguments = [sympy.Dummy() for _ in symbols]
+    function = lambdify(
+        args=arguments,
+        expr=expression.xreplace(dict(zip(symbols, arguments, strict=True))),
+        modules="numpy",
+    )
+    return tuple(str(symbol) for symbol in symbols), function
+
+
+def evaluate_formula(formula: str, variables: Mapping[str, Any]) -> Any:
+    """Evaluate an L3 formula of SBML on the values of its identifiers.
+
+    Args:
+        formula: the formula.
+        variables: the value of every identifier of the formula, a number or
+            an array. Values of identifiers the formula does not use are
+            ignored.
+
+    Returns:
+        The value of the formula, an array if one of its values is one.
+
+    Raises:
+        ValueError: if the formula is not valid math, see
+            `formula_expression`, or if an identifier has no value.
+    """
+    if not isinstance(formula, str):
+        raise ValueError(f"The formula '{formula}' is empty")
+    symbols, function = _formula_function(formula)
+    missing = [symbol for symbol in symbols if symbol not in variables]
+    if missing:
+        raise ValueError(
+            f"The formula '{formula}' uses {missing}, which have no value. The "
+            f"values are given for {sorted(variables)}"
+        )
+    return function(*[variables[symbol] for symbol in symbols])
+
+
+def expression_to_astnode(expression: sympy.Basic) -> libsbml.ASTNode:
+    """Convert a sympy expression into the math of SBML.
+
+    The numbers of the expression carry no units, so the math is valid in
+    every level of SBML.
+
+    Args:
+        expression: the expression.
+
+    Returns:
+        The syntax tree of the MathML of the expression.
+
+    Raises:
+        ValueError: if the expression uses a function the MathML of SBML does
+            not have, e.g. the error function.
+    """
+    functions = sorted({str(f.func) for f in expression.atoms(sympy.Function)})
+    try:
+        mathml = SBMLMathMLPrinter(literals_dimensionless=False).doprint(expression)
+        astnode: libsbml.ASTNode | None = libsbml.readMathMLFromString(mathml)
+    except Exception as err:
+        raise ValueError(
+            f"The expression '{expression}' has no MathML of SBML, its "
+            f"functions are {functions}: {type(err).__name__}: {err}"
+        ) from err
+    if astnode is None or not astnode.isWellFormedASTNode():
+        raise ValueError(
+            f"The expression '{expression}' has no MathML of SBML, its "
+            f"functions are {functions}"
+        )
+    return astnode
+
+
+def expression_to_formula(expression: sympy.Basic) -> str:
+    """Write a sympy expression as an L3 formula of SBML.
+
+    Args:
+        expression: the expression.
+
+    Returns:
+        The formula, which `formula_expression` and libsbml read. The
+        natural logarithm is `ln`: `log` is the logarithm to the base 10 in a
+        formula of SBML.
+
+    Raises:
+        ValueError: if the expression has no MathML of SBML, see
+            `expression_to_astnode`.
+    """
+    settings = libsbml.L3ParserSettings()
+    settings.setParseUnits(False)
+    return str(
+        libsbml.formulaToL3StringWithSettings(
+            expression_to_astnode(expression), settings
+        )
+    )
```

- [ ] **Step 4: Apply this patch to `pyproject.toml`**

Apply this patch to `pyproject.toml`:

```diff
diff --git a/pyproject.toml b/pyproject.toml
index 707c137..70e5554 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -47,6 +47,8 @@ dependencies = [
 	"xarray>=2026.7.0",
 	"scipy>=1.18.1",
 	"sympy>=1.14.0",
+	# the formulas of a model and of a network as sympy expressions
+	"sbmlmath>=0.4.1",
 	"pint>=0.25.3",
 	"pydantic>=2.13.5",
 	# parameter fitting and sensitivity analysis
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/test_mathml_formula.py tests/test_mathml.py tests/test_data.py`
Expected: PASS

- [ ] **Step 6: Run the checks**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: `All checks passed!`, `... files already formatted`, `All checks passed!` (zero diagnostics)

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "mathml: formulas are read with libsbml and sbmlmath, expressions are written as math of SBML"
```

### Task 4: `FitParameter.scale`, the space the optimizer searches one parameter in

**Files:**
- Modify: `src/sbmlsim/fit/fisher.py`
- Modify: `src/sbmlsim/fit/identifiability.py`
- Modify: `src/sbmlsim/fit/objects.py`
- Modify: `src/sbmlsim/fit/optimization.py`
- Test: `tests/fit/test_parameter_scale.py`

**Interfaces:**
- Consumes: `ParameterScaleType` of `sbmlsim.fit.options`, `OptimizationProblem` and `FisherInformation` as they are.
- Produces: `FitParameter(..., scale=None)`, `FitParameter.scale`, `FitParameter.to_dict()["scale"]` (the name or `None`); `OptimizationProblem.scales: list[ParameterScaleType]` after `initialize`, `OptimizationProblem.scales_initialized`, `to_scale(x)`/`from_scale(x)` for one value per parameter (`ValueError` otherwise); `FisherInformation.scales`, `.parameter_scales`, `.to_scale(values)`, `.from_scale(values)`, `to_dict()["scales"]`.

`FitParameter` gains `scale: ParameterScaleType | None = None`, given as the enum or its name; `None` is the `parameter_scale` of the settings, so every existing definition means what it means today. The constructor also refuses bounds which are `None` or `nan` and start values which are not finite, which were silently accepted. `OptimizationProblem.initialize` resolves `self.scales`, one per parameter, and `to_scale`/`from_scale` transform one value per parameter with its own scale (one call of numpy when all scales agree, which is every existing fit); a scalar or a vector of another length is an error, which is why the bounds of differential evolution are transformed as vectors. `_validate_parameters` checks the bounds and the start value of every parameter against its own scale. `FisherInformation` gains `scales` and transforms the confidence intervals with them, `fisher_information` fills them from the problem, and the profile likelihood checks the positivity of a parameter against its own scale; both use `problem.to_scale`/`from_scale`. The existing results of `tests/fit/` do not change: the scale of every parameter is the one of the settings.

- [ ] **Step 1: Apply this patch to `tests/fit/test_parameter_scale.py`**

Apply this patch to `tests/fit/test_parameter_scale.py`:

```diff
diff --git a/tests/fit/test_parameter_scale.py b/tests/fit/test_parameter_scale.py
index 27fed4a..8b31901 100644
--- a/tests/fit/test_parameter_scale.py
+++ b/tests/fit/test_parameter_scale.py
@@ -8,11 +8,13 @@ of its parameter table.

 from copy import deepcopy
 from dataclasses import replace
+from typing import Any

 import numpy as np
 import pytest

-from sbmlsim.fit import FitSettings
+from sbmlsim.fit import FitParameter, FitSettings, ParameterSet
+from sbmlsim.fit.fisher import fisher_information
 from sbmlsim.fit.optimization import OptimizationProblem
 from sbmlsim.fit.options import ParameterScaleType

@@ -127,3 +129,164 @@ def test_an_infinite_bound_is_never_allowed(
     for scale in ParameterScaleType:
         with pytest.raises(ValueError, match="finite"):
             problem.initialize(replace(fit_settings, parameter_scale=scale), force=True)
+
+
+# --- THE SCALE OF A PARAMETER ---
+
+
+def test_a_parameter_has_the_scale_of_the_settings_by_default() -> None:
+    """Every existing definition means what it means without a scale."""
+    parameter = FitParameter("k", 1.0, 0.1, 10.0, unit="1/min")
+    assert parameter.scale is None
+    assert parameter.to_dict()["scale"] is None
+    assert FitParameter(**parameter.to_dict()) == parameter
+
+
+@pytest.mark.parametrize("scale", list(ParameterScaleType))
+def test_the_scale_of_a_parameter_is_stored(scale: ParameterScaleType) -> None:
+    """The scale is a part of the parameter and of its serialization."""
+    parameter = FitParameter("k", 1.0, 0.1, 10.0, unit="1/min", scale=scale)
+    assert parameter.scale is scale
+    assert parameter.to_dict()["scale"] == scale.name
+    assert FitParameter(**parameter.to_dict()).scale is scale
+    restored = FitParameter.from_json(str(parameter.to_json()))
+    assert restored.scale is scale
+    assert restored == parameter
+    assert parameter != FitParameter("k", 1.0, 0.1, 10.0, unit="1/min")
+    assert (
+        FitParameter("k", 1.0, 0.1, 10.0, unit="1/min", scale=scale.name) == parameter
+    )
+
+
+@pytest.mark.parametrize("scale", ["lin", "LOG2", 1, 2.0, ParameterScaleType])
+def test_a_scale_which_is_not_a_scale(scale: object) -> None:
+    """A scale is a `ParameterScaleType` or the name of one."""
+    with pytest.raises(ValueError, match=r"FitParameter 'k': the scale"):
+        FitParameter("k", 1.0, 0.1, 10.0, unit="1/min", scale=scale)  # ty: ignore[invalid-argument-type]
+
+
+@pytest.mark.parametrize(
+    ("kwargs", "message"),
+    [
+        ({"lower_bound": np.nan}, "the 'lower_bound' is 'nan'"),
+        ({"upper_bound": np.nan}, "the 'upper_bound' is 'nan'"),
+        ({"upper_bound": None}, "the 'upper_bound' is 'None'"),
+        ({"start_value": np.nan}, "the start value 'nan' is not a finite number"),
+        ({"start_value": np.inf}, "the start value 'inf' is not a finite number"),
+    ],
+)
+def test_the_values_of_a_parameter_are_numbers(kwargs: dict, message: str) -> None:
+    """A value which is no number is an error and not a bound which holds."""
+    arguments: dict[str, Any] = {
+        "start_value": 1.0,
+        "lower_bound": 0.1,
+        "upper_bound": 10.0,
+    }
+    arguments.update(kwargs)
+    with pytest.raises(ValueError, match=rf"FitParameter 'k': {message}"):
+        FitParameter("k", unit="1/min", **arguments)
+
+
+def _with_scales(
+    problem: OptimizationProblem, scales: list[ParameterScaleType | None]
+) -> OptimizationProblem:
+    """Get the problem with copies of its parameters which have the scales."""
+    problem.parameters = [deepcopy(p) for p in problem.parameters]
+    for parameter, scale in zip(problem.parameters, scales, strict=True):
+        parameter.scale = scale
+    return problem
+
+
+def test_the_problem_transforms_every_parameter_with_its_scale(
+    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
+) -> None:
+    """A parameter without a scale has the one of the settings."""
+    problem = _with_scales(
+        op_hctz_pk, [ParameterScaleType.LINEAR, None, ParameterScaleType.LOG]
+    )
+    with pytest.raises(ValueError, match="must be initialized first"):
+        problem.to_scale([1.0, 2.0, 3.0])
+
+    problem.initialize(fit_settings)
+    assert problem.parameter_scale is ParameterScaleType.LOG10
+    assert problem.scales_initialized == [
+        ParameterScaleType.LINEAR,
+        ParameterScaleType.LOG10,
+        ParameterScaleType.LOG,
+    ]
+    x = np.array([0.5, 100.0, np.e])
+    scaled = problem.to_scale(x)
+    np.testing.assert_allclose(scaled, [0.5, 2.0, 1.0])
+    np.testing.assert_allclose(problem.from_scale(scaled), x)
+
+    for values in ([1.0, 2.0], [1.0, 2.0, 3.0, 4.0], 1.0, [[1.0, 2.0, 3.0]]):
+        with pytest.raises(ValueError, match="one value per parameter"):
+            problem.to_scale(values)
+        with pytest.raises(ValueError, match="one value per parameter"):
+            problem.from_scale(values)
+
+
+def test_the_cost_does_not_depend_on_the_scales_of_the_parameters(
+    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
+) -> None:
+    """The scales are how the optimizer walks, not what it optimizes."""
+    problem = op_hctz_iv
+    problem.initialize(fit_settings)
+    x = np.asarray(problem.x0, dtype=float)
+    cost = problem.cost_least_square(problem.to_scale(x))
+
+    n = len(problem.parameters)
+    scales: list[ParameterScaleType | None] = [
+        [ParameterScaleType.LINEAR, ParameterScaleType.LOG, None][k % 3]
+        for k in range(n)
+    ]
+    problem = _with_scales(problem, scales)
+    problem.initialize(fit_settings, force=True)
+    assert problem.cost_least_square(problem.to_scale(x)) == pytest.approx(
+        cost, rel=1e-12
+    )
+
+
+def test_a_parameter_on_the_linear_scale_may_be_negative(
+    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
+) -> None:
+    """The scale of a parameter decides which bounds it may have."""
+    problem = _with_scales(op_hctz_pk, [ParameterScaleType.LINEAR, None, None])
+    problem.parameters[0].lower_bound = -1.0
+    problem.initialize(fit_settings)
+    lower = problem.to_scale([p.lower_bound for p in problem.parameters])
+    assert lower[0] == -1.0
+    assert lower[1] == pytest.approx(np.log10(problem.parameters[1].lower_bound))
+
+    problem.parameters[1].lower_bound = -1.0
+    with pytest.raises(
+        ValueError, match=rf"'LOG10'.*positive.*'{problem.parameters[1].pid}'"
+    ):
+        problem.initialize(fit_settings, force=True)
+
+
+def test_the_fisher_information_uses_the_scales_of_the_parameters(
+    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
+) -> None:
+    """The intervals are transformed back with the scale of every parameter."""
+    problem = op_hctz_iv
+    n = len(problem.parameters)
+    problem = _with_scales(problem, [ParameterScaleType.LINEAR] + [None] * (n - 1))
+    problem.initialize(fit_settings)
+    pset = ParameterSet.from_fit_parameters(
+        problem.parameters, x=np.asarray(problem.x0, dtype=float), sid="start"
+    )
+    fim = fisher_information(problem, fit_settings, pset)
+    assert fim.scale is ParameterScaleType.LOG10
+    assert fim.parameter_scales == problem.scales_initialized
+    assert fim.to_dict()["scales"] == ["LINEAR"] + ["LOG10"] * (n - 1)
+    np.testing.assert_allclose(fim.from_scale(fim.to_scale(fim.values)), fim.values)
+    lower, upper = fim.confidence_intervals()
+    errors = fim.standard_errors
+    if np.isfinite(errors[0]):
+        # symmetric on the linear scale, and not on a logarithmic one
+        assert fim.values[0] - lower[0] == pytest.approx(upper[0] - fim.values[0])
+
+    with pytest.raises(ValueError, match="one scale per parameter"):
+        replace(fim, scales=[ParameterScaleType.LINEAR])
+    assert replace(fim, scales=[]).parameter_scales == [ParameterScaleType.LOG10] * n
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -x tests/fit/test_parameter_scale.py`
Expected: FAIL: `test_a_parameter_has_the_scale_of_the_settings_by_default` (`AttributeError: 'FitParameter' object has no attribute 'scale'`)

- [ ] **Step 3: Apply this patch to `src/sbmlsim/fit/objects.py`**

Apply this patch to `src/sbmlsim/fit/objects.py`:

```diff
diff --git a/src/sbmlsim/fit/objects.py b/src/sbmlsim/fit/objects.py
index 6dd195c..bfd4c70 100644
--- a/src/sbmlsim/fit/objects.py
+++ b/src/sbmlsim/fit/objects.py
@@ -15,6 +15,7 @@ import numpy as np
 import pandas as pd

 from sbmlsim.data import Data
+from sbmlsim.fit.options import ParameterScaleType
 from sbmlsim.serialization import to_json
 from sbmlsim.units import Quantity

@@ -493,6 +494,7 @@ class FitParameter:
         unit: str | None = None,
         target: str | None = None,
         mappings: Any = None,
+        scale: ParameterScaleType | str | None = None,
     ):
         """Initialize FitParameter.

@@ -514,10 +516,39 @@ class FitParameter:
                 A selector is a callable and is not serialized: it must be a
                 module level function, because the workers of a parallel fit
                 unpickle the parameters.
+            scale: space the optimizer searches the parameter in, or its name.
+                `None` is the `parameter_scale` of the `FitSettings`. A
+                parameter which is negative or zero, e.g. a weight of a
+                network, is searched on the linear scale.

         Raises:
-            ValueError: if the bounds or the start value are inconsistent.
+            ValueError: if the bounds or the start value are inconsistent, if
+                a value is not a number, or if the scale is not a scale.
         """
+        for key, value in (("lower_bound", lower_bound), ("upper_bound", upper_bound)):
+            if value is None or np.isnan(value):
+                raise ValueError(
+                    f"FitParameter '{pid}': the '{key}' is '{value}', which is "
+                    f"not a number. A parameter without a bound has an "
+                    f"infinite one."
+                )
+        if start_value is not None and not np.isfinite(start_value):
+            raise ValueError(
+                f"FitParameter '{pid}': the start value '{start_value}' is not "
+                f"a finite number."
+            )
+        if isinstance(scale, str):
+            if scale not in ParameterScaleType.__members__:
+                raise ValueError(
+                    f"FitParameter '{pid}': the scale '{scale}' is not one of "
+                    f"{list(ParameterScaleType.__members__)}."
+                )
+            scale = ParameterScaleType[scale]
+        if scale is not None and not isinstance(scale, ParameterScaleType):
+            raise ValueError(
+                f"FitParameter '{pid}': the scale '{scale}' is not a "
+                f"`ParameterScaleType`."
+            )
         if lower_bound > upper_bound:
             raise ValueError(
                 f"FitParameter '{pid}': lower bound '{lower_bound}' is larger than "
@@ -536,6 +567,7 @@ class FitParameter:
         self.unit = unit
         self.target = target
         self.mappings = mappings
+        self.scale: ParameterScaleType | None = scale
         if unit is None:
             logger.warning(
                 "No unit provided for FitParameter '%s', assuming model units.",
@@ -567,6 +599,7 @@ class FitParameter:
             and _isclose(self.upper_bound, other.upper_bound)
             and self.unit == other.unit
             and self.target_id == other.target_id
+            and self.scale == other.scale
         )

     def __hash__(self) -> int:
@@ -596,6 +629,7 @@ class FitParameter:
             "upper_bound": self.upper_bound,
             "unit": self.unit,
             "target": self.target,
+            "scale": None if self.scale is None else self.scale.name,
         }

     @staticmethod
```

- [ ] **Step 4: Apply this patch to `src/sbmlsim/fit/optimization.py`**

Apply this patch to `src/sbmlsim/fit/optimization.py`:

```diff
diff --git a/src/sbmlsim/fit/optimization.py b/src/sbmlsim/fit/optimization.py
index a270f55..ec73770 100644
--- a/src/sbmlsim/fit/optimization.py
+++ b/src/sbmlsim/fit/optimization.py
@@ -461,6 +461,11 @@ class OptimizationProblem(ObjectJSONEncoder):
             return

         self.settings = settings
+        #: the space the optimizer searches every parameter in
+        self.scales: list[ParameterScaleType] = [
+            settings.parameter_scale if p.scale is None else p.scale
+            for p in self.parameters
+        ]
         self._validate_parameters()
         # initialize can be called more than once, e.g. for the report of a fit
         self._reset_mappings()
@@ -797,13 +802,70 @@ class OptimizationProblem(ObjectJSONEncoder):
         """Get the space the optimizer searches the parameters in."""
         return self.settings_initialized.parameter_scale

+    @property
+    def scales_initialized(self) -> list[ParameterScaleType]:
+        """Get the space the optimizer searches every parameter in.
+
+        The scale of a parameter is its `FitParameter.scale` and the
+        `parameter_scale` of the settings for a parameter without one.
+
+        Raises:
+            ValueError: if the problem was not initialized.
+        """
+        self.settings_initialized  # noqa: B018
+        return self.scales
+
+    def _scaled(self, x: Any, to_scale: bool) -> np.ndarray:
+        """Transform one value per parameter, every one with its own scale."""
+        values = np.asarray(x, dtype=float)
+        scales = self.scales_initialized
+        if values.shape != (len(scales),):
+            raise ValueError(
+                f"'{self.opid}': the transformation requires one value per "
+                f"parameter, but values of the shape '{values.shape}' are given "
+                f"for the '{len(scales)}' parameters '{self.pids}'."
+            )
+        if len(set(scales)) == 1:
+            # one scale, which is one call of numpy
+            scale = scales[0]
+            scaled = scale.to_scale(values) if to_scale else scale.from_scale(values)
+            return np.asarray(scaled, dtype=float)
+        return np.array(
+            [
+                scale.to_scale(value) if to_scale else scale.from_scale(value)
+                for scale, value in zip(scales, values, strict=True)
+            ],
+            dtype=float,
+        )
+
     def to_scale(self, x: Any) -> np.ndarray:
-        """Transform parameters of the model into the space of the optimizer."""
-        return np.asarray(self.parameter_scale.to_scale(x), dtype=float)
+        """Transform parameters of the model into the space of the optimizer.
+
+        Args:
+            x: one value per parameter, in the units of the model.
+
+        Returns:
+            The values in the space the optimizer searches, every parameter
+            with its own scale.
+
+        Raises:
+            ValueError: if `x` does not have one value per parameter.
+        """
+        return self._scaled(x, to_scale=True)

     def from_scale(self, x: Any) -> np.ndarray:
-        """Transform parameters of the optimizer into the units of the model."""
-        return np.asarray(self.parameter_scale.from_scale(x), dtype=float)
+        """Transform parameters of the optimizer into the units of the model.
+
+        Args:
+            x: one value per parameter, in the space the optimizer searches.
+
+        Returns:
+            The values in the units of the model.
+
+        Raises:
+            ValueError: if `x` does not have one value per parameter.
+        """
+        return self._scaled(x, to_scale=False)

     def _validate_parameters(self) -> None:
         """Check that the parameters can be optimized.
@@ -815,9 +877,8 @@ class OptimizationProblem(ObjectJSONEncoder):
         Raises:
             ValueError: if a bound or a start value does not suit the scale.
         """
-        scale = self.parameter_scale
-        space = f"'{scale.name}' parameter space"
-        for p in self.parameters:
+        for p, scale in zip(self.parameters, self.scales_initialized, strict=True):
+            space = f"'{scale.name}' parameter space"
             for key in ["lower_bound", "upper_bound"]:
                 value = getattr(p, key)
                 if not np.isfinite(value):
@@ -1260,10 +1321,13 @@ class OptimizationProblem(ObjectJSONEncoder):
             # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html#scipy.optimize.differential_evolution
             ts = time.time()
             try:
-                de_bounds_log = [
-                    (self.to_scale(p.lower_bound), self.to_scale(p.upper_bound))
-                    for k, p in enumerate(self.parameters)
-                ]
+                de_bounds_log = list(
+                    zip(
+                        self.to_scale([p.lower_bound for p in self.parameters]),
+                        self.to_scale([p.upper_bound for p in self.parameters]),
+                        strict=True,
+                    )
+                )
                 opt_result = scipy.optimize.differential_evolution(
                     func=self.cost_least_square, bounds=de_bounds_log, **kwargs
                 )
```

- [ ] **Step 5: Apply this patch to `src/sbmlsim/fit/fisher.py`**

Apply this patch to `src/sbmlsim/fit/fisher.py`:

```diff
diff --git a/src/sbmlsim/fit/fisher.py b/src/sbmlsim/fit/fisher.py
index c26d5f3..f431a96 100644
--- a/src/sbmlsim/fit/fisher.py
+++ b/src/sbmlsim/fit/fisher.py
@@ -56,7 +56,11 @@ class FisherInformation:
         pids: parameters, in the order of the matrix.
         values: values of the parameters, in the units of the model.
         scale: space the matrix is expressed in, i.e. the space the optimizer
-            searches, see `ParameterScaleType`.
+            searches, see `ParameterScaleType`. It is the scale of the
+            settings of the fit, i.e. of the parameters without a scale of
+            their own.
+        scales: the space of every parameter, in the order of the matrix.
+            Without them every parameter has the scale `scale`.
         matrix: the Fisher information `J' J`.
         cost: cost of the parameter set.
         n: number of data points of the fit.
@@ -74,17 +78,56 @@ class FisherInformation:
     alpha: float = 0.95
     rank_tolerance: float = DEFAULT_RANK_TOLERANCE
     units: list[str | None] = field(default_factory=list)
+    scales: list[ParameterScaleType] = field(default_factory=list)

     #: the covariance is read by the errors, the correlations and the table, so
     #: the warning of a rank deficient information is logged for the first of
     #: them and not once per reader
     _warned: bool = field(default=False, init=False, repr=False, compare=False)

+    def __post_init__(self) -> None:
+        """Check the scales of the parameters.
+
+        Raises:
+            ValueError: if the scales are given and not one per parameter.
+        """
+        if self.scales and len(self.scales) != len(self.pids):
+            raise ValueError(
+                f"'{self.opid}': the Fisher information requires one scale per "
+                f"parameter, but '{len(self.scales)}' scales are given for the "
+                f"'{len(self.pids)}' parameters '{self.pids}'."
+            )
+
     @property
     def k(self) -> int:
         """Get the number of parameters."""
         return len(self.pids)

+    @property
+    def parameter_scales(self) -> list[ParameterScaleType]:
+        """Get the space of every parameter, in the order of the matrix."""
+        return list(self.scales) if self.scales else [self.scale] * self.k
+
+    def to_scale(self, values: Any) -> np.ndarray:
+        """Transform one value per parameter into the space of the optimizer."""
+        return np.array(
+            [
+                scale.to_scale(value)
+                for scale, value in zip(self.parameter_scales, values, strict=True)
+            ],
+            dtype=float,
+        )
+
+    def from_scale(self, values: Any) -> np.ndarray:
+        """Transform one value per parameter into the units of the model."""
+        return np.array(
+            [
+                scale.from_scale(value)
+                for scale, value in zip(self.parameter_scales, values, strict=True)
+            ],
+            dtype=float,
+        )
+
     @property
     def sigma2(self) -> float:
         """Get the variance of the residuals, `2 cost / (n - k)`.
@@ -178,8 +221,9 @@ class FisherInformation:

         The interval is `θ ± t · SE` in the space the optimizer searches, with
         the quantile of the t distribution of `n - k` degrees of freedom, and
-        is transformed back into the units of the model. On a logarithmic scale
-        the interval is therefore not symmetric around the value.
+        is transformed back into the units of the model, every parameter with
+        its own scale. On a logarithmic scale the interval is therefore not
+        symmetric around the value.

         Returns:
             The lower and the upper bound in the units of the model.
@@ -189,12 +233,9 @@ class FisherInformation:
             nan = np.full(self.k, np.nan)
             return nan, nan
         quantile = float(student_t.ppf(0.5 + self.alpha / 2.0, dof))
-        scaled = self.scale.to_scale(self.values)
+        scaled = self.to_scale(self.values)
         delta = quantile * self.standard_errors
-        return (
-            np.asarray(self.scale.from_scale(scaled - delta), dtype=float),
-            np.asarray(self.scale.from_scale(scaled + delta), dtype=float),
-        )
+        return self.from_scale(scaled - delta), self.from_scale(scaled + delta)

     @property
     def summary_df(self) -> pd.DataFrame:
@@ -203,7 +244,7 @@ class FisherInformation:
         errors = self.standard_errors
         with np.errstate(divide="ignore", invalid="ignore"):
             # the error relative to the value, in the scaled space
-            cv = np.abs(errors / self.scale.to_scale(self.values)) * 100.0
+            cv = np.abs(errors / self.to_scale(self.values)) * 100.0
         units = self.units or [None] * self.k
         return pd.DataFrame(
             {
@@ -225,6 +266,7 @@ class FisherInformation:
             "pids": list(self.pids),
             "values": [float(v) for v in self.values],
             "scale": self.scale.name,
+            "scales": [scale.name for scale in self.parameter_scales],
             "matrix": [[float(v) for v in row] for row in self.matrix],
             "cost": self.cost,
             "n": self.n,
@@ -302,4 +344,5 @@ def fisher_information(
         alpha=alpha,
         rank_tolerance=rank_tolerance,
         units=[p.unit for p in problem.parameters],
+        scales=list(problem.scales_initialized),
     )
```

- [ ] **Step 6: Apply this patch to `src/sbmlsim/fit/identifiability.py`**

Apply this patch to `src/sbmlsim/fit/identifiability.py`:

```diff
diff --git a/src/sbmlsim/fit/identifiability.py b/src/sbmlsim/fit/identifiability.py
index e7c4407..c084ee8 100644
--- a/src/sbmlsim/fit/identifiability.py
+++ b/src/sbmlsim/fit/identifiability.py
@@ -747,9 +747,9 @@ ProfilePoint = tuple[np.ndarray, float, bool]
 def _scaled_bounds(problem: OptimizationProblem) -> tuple[np.ndarray, np.ndarray]:
     """Get the bounds of the parameters in the space of the optimizer.

-    The scans run in the space the fit searches, i.e. the
-    `parameter_scale` of its settings, so that a step of the scan is a step of
-    the optimizer.
+    The scans run in the space the fit searches, i.e. every parameter in its
+    own scale, which is the `parameter_scale` of the settings for a parameter
+    without one, so that a step of the scan is a step of the optimizer.
     """
     return (
         problem.to_scale([p.lower_bound for p in problem.parameters]),
@@ -980,11 +980,18 @@ def profile_likelihood(

     x = parameter_set.x(problem.pids)
     lower, upper = _scaled_bounds(problem)
-    if problem.parameter_scale.is_log and np.any(x <= 0.0):
+    negative = {
+        pid: (float(value), scale.name)
+        for pid, value, scale in zip(
+            problem.pids, x, problem.scales_initialized, strict=True
+        )
+        if scale.is_log and value <= 0.0
+    }
+    if negative:
         raise ValueError(
             f"'{problem.opid}': the parameters must be positive, the scans run in "
-            f"'{problem.parameter_scale.name}' space, got "
-            f"'{dict(zip(problem.pids, x, strict=True))}'."
+            f"the logarithmic space of a parameter, got the values and scales "
+            f"'{negative}'."
         )
     theta_optimum = problem.to_scale(x)
     outside = [
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/fit`
Expected: PASS

- [ ] **Step 8: Run the checks**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: `All checks passed!`, `... files already formatted`, `All checks passed!` (zero diagnostics)

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "fit: a parameter has a scale of its own"
```

### Task 5: Infinite bounds on the linear scale, which are not sampled

**Files:**
- Modify: `docs/fitting.md`
- Modify: `src/sbmlsim/fit/optimization.py`
- Modify: `src/sbmlsim/fit/sampling.py`
- Test: `tests/fit/test_parameter_scale.py`
- Test (create): `tests/fit/test_sampling.py`

**Interfaces:**
- Consumes: `OptimizationProblem.scales_initialized`, `_validate_parameters` of task 4.
- Produces: `create_samples(parameters, size, sampling, seed, min_bound)` without `max_bound`; `OptimizationProblem._validate_parameters(algorithm=None)`, called with the algorithm by `_optimize_single_run`.

The elements of a network have no bounds. A parameter on the linear scale may therefore have infinite bounds: scipy `least_squares` takes them, differential evolution samples a finite box and `_validate_parameters(algorithm)` rejects them for it at the start of a run. `create_samples` no longer replaces an infinite bound by `max_bound` (the argument is gone): such a parameter is not sampled and starts from its start value in every repeat, which it therefore needs, and the repeats differ in the parameters with finite bounds. The unit hypercube of all parameters is still drawn, so the samples of the bounded parameters do not depend on the unbounded ones; `test_the_samples_of_bounded_parameters_did_not_change` writes the sampling of the release before down once more and pins that the samples are the same (it passes against the unchanged `sampling.py` as well, which was verified). `test_an_infinite_bound_is_never_allowed` of the tests before becomes `test_an_infinite_bound_needs_the_linear_scale`: this is the one existing test whose expectation the spec changes.

- [ ] **Step 1: Create `tests/fit/test_sampling.py`**

Create `tests/fit/test_sampling.py`:

```python
"""Tests of the sampling of the start values of a fit."""

import logging

import numpy as np
import pytest
from scipy.stats import qmc

from sbmlsim.fit import FitParameter
from sbmlsim.fit.sampling import SamplingType, create_samples

SIZE = 7
SEED = 1234


def _bounded() -> list[FitParameter]:
    return [
        FitParameter("p1", 100.0, lower_bound=10.0, upper_bound=1e4, unit="mM"),
        FitParameter("p2", 0.0, lower_bound=-2.0, upper_bound=3.0, unit="mM"),
        FitParameter("p3", None, lower_bound=1e-3, upper_bound=1.0, unit="mM"),
    ]


def _unbounded() -> list[FitParameter]:
    return [
        FitParameter("w1", -0.5, unit="dimensionless"),
        FitParameter("w2", 0.0, lower_bound=-np.inf, upper_bound=1.0, unit="mM"),
        FitParameter("w3", 2.5, lower_bound=0.0, upper_bound=np.inf, unit="mM"),
    ]


def _expected(parameters: list[FitParameter], sampling: SamplingType) -> np.ndarray:
    """Get the samples as they were drawn before parameters could be unbounded.

    This is the sampling of `create_samples` of the release 0.7.2 for bounded
    parameters, written down once more: the unit hypercube of all parameters
    is drawn, and every column is stretched to the bounds of its parameter.
    """
    rng = np.random.default_rng(SEED)
    if sampling.is_lhs:
        x = qmc.LatinHypercube(d=len(parameters), rng=rng).random(n=SIZE)
    else:
        x = rng.random(size=(SIZE, len(parameters)))
    for k, p in enumerate(parameters):
        lb, ub = float(p.lower_bound), float(p.upper_bound)
        if sampling.is_log:
            lb = 1e-10 if lb <= 0.0 else lb
            x[:, k] = np.power(10, np.log10(lb) + x[:, k] * np.log10(ub / lb))
        else:
            x[:, k] = lb + x[:, k] * (ub - lb)
    return x


@pytest.mark.parametrize("sampling", list(SamplingType))
def test_the_samples_of_bounded_parameters_did_not_change(
    sampling: SamplingType,
) -> None:
    """The samples of a fit with bounds are the ones of the release before."""
    parameters = _bounded()
    samples = create_samples(parameters, size=SIZE, sampling=sampling, seed=SEED)
    assert list(samples.columns) == ["p1", "p2", "p3"]
    np.testing.assert_allclose(
        samples.to_numpy(), _expected(parameters, sampling), rtol=1e-14
    )
    for p in parameters:
        lower = 1e-10 if sampling.is_log and p.lower_bound <= 0 else p.lower_bound
        assert np.all(samples[p.pid] >= lower)
        assert np.all(samples[p.pid] <= p.upper_bound)


@pytest.mark.parametrize("sampling", list(SamplingType))
def test_a_parameter_without_a_bound_is_not_sampled(
    sampling: SamplingType, caplog: pytest.LogCaptureFixture
) -> None:
    """A parameter with an infinite bound starts from its start value."""
    parameters = [*_bounded(), *_unbounded()]
    with caplog.at_level(logging.WARNING, logger="sbmlsim.fit.sampling"):
        samples = create_samples(parameters, size=SIZE, sampling=sampling, seed=SEED)
    for p in _unbounded():
        np.testing.assert_array_equal(samples[p.pid], np.full(SIZE, p.start_value))
    # the bound is not replaced by a number which is sampled
    assert "infinite" not in caplog.text
    assert np.all(np.isfinite(samples.to_numpy()))
    # the repeats differ in the parameters which have bounds
    assert samples["p1"].nunique() == SIZE


@pytest.mark.parametrize("sampling", list(SamplingType))
def test_the_samples_do_not_depend_on_the_parameters_without_bounds(
    sampling: SamplingType,
) -> None:
    """The columns of the bounded parameters are the ones of the hypercube."""
    bounded = _bounded()
    w1, w2, w3 = _unbounded()
    parameters = [w1, bounded[0], w2, bounded[1], bounded[2], w3]
    samples = create_samples(parameters, size=SIZE, sampling=sampling, seed=SEED)
    expected = _expected(
        [
            FitParameter(p.pid, None, 1.0, 2.0, unit="mM")
            if np.isinf(p.lower_bound) or np.isinf(p.upper_bound)
            else p
            for p in parameters
        ],
        sampling,
    )
    for k in (1, 3, 4):
        np.testing.assert_allclose(samples.iloc[:, k], expected[:, k], rtol=1e-14)


def test_a_parameter_without_a_bound_needs_a_start_value() -> None:
    """There is nothing to start from without a bound and a start value."""
    parameters = [*_bounded(), FitParameter("w1", None, unit="dimensionless")]
    with pytest.raises(ValueError, match=r"'w1'.*\[-inf - inf\].*'start_value'"):
        create_samples(parameters, size=SIZE, seed=SEED)


def test_the_sampling_is_checked() -> None:
    """The size and the parameters of a sampling are given."""
    with pytest.raises(ValueError, match="'size' must be a positive integer"):
        create_samples(_bounded(), size=0)
    with pytest.raises(ValueError, match="'parameters' must not be empty"):
        create_samples([], size=SIZE)
    negative = [FitParameter("p1", -2.0, lower_bound=-3.0, upper_bound=-1.0, unit="mM")]
    with pytest.raises(ValueError, match=r"'p1'.*positive upper bound"):
        create_samples(negative, size=SIZE, sampling=SamplingType.LOGUNIFORM)
```

- [ ] **Step 2: Apply this patch to `tests/fit/test_parameter_scale.py`**

Apply this patch to `tests/fit/test_parameter_scale.py`:

```diff
diff --git a/tests/fit/test_parameter_scale.py b/tests/fit/test_parameter_scale.py
index 8b31901..c6d8a10 100644
--- a/tests/fit/test_parameter_scale.py
+++ b/tests/fit/test_parameter_scale.py
@@ -16,7 +16,7 @@ import pytest
 from sbmlsim.fit import FitParameter, FitSettings, ParameterSet
 from sbmlsim.fit.fisher import fisher_information
 from sbmlsim.fit.optimization import OptimizationProblem
-from sbmlsim.fit.options import ParameterScaleType
+from sbmlsim.fit.options import OptimizationAlgorithmType, ParameterScaleType

 #: values which span orders of magnitude, i.e. what a log scale is for
 VALUES = np.array([1e-6, 1.0, 25.0, 1e3])
@@ -118,18 +118,36 @@ def test_a_logarithm_needs_positive_bounds(
     assert problem.parameter_scale is ParameterScaleType.LINEAR


-def test_an_infinite_bound_is_never_allowed(
+def test_an_infinite_bound_needs_the_linear_scale(
     op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
 ) -> None:
-    """An optimizer cannot search an interval which has no end."""
+    """The logarithm of an interval without an end is not searched.
+
+    On the linear scale the local optimizer takes the infinite bound, the
+    global optimizer samples a finite box and rejects it.
+    """
     problem = op_hctz_pk
     problem.parameters = [deepcopy(p) for p in problem.parameters]
     problem.parameters[0].upper_bound = np.inf

-    for scale in ParameterScaleType:
-        with pytest.raises(ValueError, match="finite"):
+    for scale in [ParameterScaleType.LOG10, ParameterScaleType.LOG]:
+        with pytest.raises(ValueError, match="requires a finite 'upper_bound'"):
             problem.initialize(replace(fit_settings, parameter_scale=scale), force=True)

+    problem.initialize(
+        replace(fit_settings, parameter_scale=ParameterScaleType.LINEAR), force=True
+    )
+    with pytest.raises(ValueError, match=r"DIFFERENTIAL_EVOLUTION.*finite box"):
+        problem._validate_parameters(OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION)
+    problem._validate_parameters(OptimizationAlgorithmType.LEAST_SQUARE)
+
+    problem.parameters[0].start_value = None
+    with pytest.raises(ValueError, match="requires a 'start_value'"):
+        problem.initialize(
+            replace(fit_settings, parameter_scale=ParameterScaleType.LINEAR),
+            force=True,
+        )
+

 # --- THE SCALE OF A PARAMETER ---

@@ -265,6 +283,32 @@ def test_a_parameter_on_the_linear_scale_may_be_negative(
         problem.initialize(fit_settings, force=True)


+def test_a_fit_with_a_parameter_without_bounds(
+    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
+) -> None:
+    """The local optimizer takes an infinite bound on the linear scale."""
+    problem = op_hctz_iv
+    n = len(problem.parameters)
+    problem = _with_scales(problem, [ParameterScaleType.LINEAR] + [None] * (n - 1))
+    first = problem.parameters[0]
+    first.lower_bound, first.upper_bound = -np.inf, np.inf
+    problem.initialize(fit_settings)
+
+    starts = problem.start_values(size=3, seed=1)
+    assert [float(start[0]) for start in starts if start is not None] == [
+        first.start_value
+    ] * 3
+    fits, _ = problem.optimize(size=1, seed=1, max_nfev=3)
+    assert np.all(np.isfinite(fits[0].x))
+    assert np.isfinite(fits[0].cost)
+    assert fits[0].cost <= problem.cost_least_square(problem.to_scale(starts[0]))
+
+    fits, _ = problem.optimize(
+        size=1, seed=1, algorithm=OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION
+    )
+    assert "finite box" in fits[0].message
+
+
 def test_the_fisher_information_uses_the_scales_of_the_parameters(
     op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
 ) -> None:
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest -q -x tests/fit/test_sampling.py tests/fit/test_parameter_scale.py`
Expected: FAIL: `test_a_parameter_without_a_bound_is_not_sampled` (the column of `w1` is sampled between `-1e10` and `1e10`), `test_an_infinite_bound_needs_the_linear_scale` (`ValueError ... requires a finite 'upper_bound'` on the linear scale)

- [ ] **Step 4: Apply this patch to `src/sbmlsim/fit/sampling.py`**

Apply this patch to `src/sbmlsim/fit/sampling.py`:

```diff
diff --git a/src/sbmlsim/fit/sampling.py b/src/sbmlsim/fit/sampling.py
index ad0f705..2481b3e 100644
--- a/src/sbmlsim/fit/sampling.py
+++ b/src/sbmlsim/fit/sampling.py
@@ -40,11 +40,14 @@ def create_samples(
     sampling: SamplingType = SamplingType.LOGUNIFORM,
     seed: int | None = None,
     min_bound: float = 1e-10,
-    max_bound: float = 1e10,
 ) -> pd.DataFrame:
     """Create samples of start values from the bounds of the parameters.

-    Infinite bounds are replaced by the hard bounds `min_bound` and `max_bound`.
+    A parameter with an infinite bound has no interval to sample: it starts
+    from its start value in every sample, so the samples differ in the
+    parameters with finite bounds. The samples of these do not depend on the
+    parameters which are not sampled.
+
     Logarithmic sampling requires positive bounds, non-positive lower bounds are
     replaced by `min_bound`.

@@ -53,14 +56,15 @@ def create_samples(
         size: number of samples.
         sampling: type of sampling.
         seed: seed of the random number generator, for reproducible samples.
-        min_bound: hard lower bound, replaces an infinite or non-positive bound.
-        max_bound: hard upper bound, replaces an infinite bound.
+        min_bound: hard lower bound, replaces a non-positive bound of a
+            logarithmic sampling.

     Returns:
         DataFrame with one row per sample and one column per parameter.

     Raises:
-        ValueError: if the sampling type is unsupported or the bounds are invalid.
+        ValueError: if the sampling type is unsupported, the bounds are
+            invalid, or a parameter with an infinite bound has no start value.
     """
     if size < 1:
         raise ValueError(f"'size' must be a positive integer, but '{size}' given.")
@@ -81,12 +85,16 @@ def create_samples(
         raise ValueError(f"Unsupported SamplingType: '{sampling}'")

     for k, p in enumerate(parameters):
-        lb, ub = _sampling_bounds(
-            parameter=p,
-            sampling=sampling,
-            min_bound=min_bound,
-            max_bound=max_bound,
-        )
+        if np.isinf(p.lower_bound) or np.isinf(p.upper_bound):
+            if p.start_value is None:
+                raise ValueError(
+                    f"'{p.pid}': a parameter with the infinite bounds "
+                    f"[{p.lower_bound} - {p.upper_bound}] is not sampled and "
+                    f"requires a 'start_value'."
+                )
+            x[:, k] = p.start_value
+            continue
+        lb, ub = _sampling_bounds(parameter=p, sampling=sampling, min_bound=min_bound)

         # stretch sampling dimension from [0, 1) to [lb, ub)
         if sampling.is_log:
@@ -104,20 +112,12 @@ def _sampling_bounds(
     parameter: FitParameter,
     sampling: SamplingType,
     min_bound: float,
-    max_bound: float,
 ) -> tuple[float, float]:
-    """Resolve the bounds of a parameter to the finite interval which is sampled."""
+    """Resolve the finite bounds of a parameter to the interval which is sampled."""
     pid = parameter.pid
     lb = float(parameter.lower_bound)
     ub = float(parameter.upper_bound)

-    if np.isinf(lb):
-        lb = -max_bound if lb < 0 else max_bound
-        logger.warning("'%s': infinite lower bound set to '%s'", pid, lb)
-    if np.isinf(ub):
-        ub = max_bound if ub > 0 else -max_bound
-        logger.warning("'%s': infinite upper bound set to '%s'", pid, ub)
-
     if sampling.is_log:
         # logarithmic sampling requires positive bounds
         if lb <= 0.0:
```

- [ ] **Step 5: Apply this patch to `src/sbmlsim/fit/optimization.py`**

Apply this patch to `src/sbmlsim/fit/optimization.py`:

```diff
diff --git a/src/sbmlsim/fit/optimization.py b/src/sbmlsim/fit/optimization.py
index ec73770..558d88a 100644
--- a/src/sbmlsim/fit/optimization.py
+++ b/src/sbmlsim/fit/optimization.py
@@ -867,25 +867,49 @@ class OptimizationProblem(ObjectJSONEncoder):
         """
         return self._scaled(x, to_scale=False)

-    def _validate_parameters(self) -> None:
+    def _validate_parameters(
+        self, algorithm: OptimizationAlgorithmType | None = None
+    ) -> None:
         """Check that the parameters can be optimized.

         An optimization on a logarithmic scale, which is the default, requires
-        finite positive bounds and start values; on the linear scale the bounds
-        only have to be finite.
+        finite positive bounds and start values. A parameter on the linear
+        scale may have infinite bounds, which the local optimizer takes; the
+        global optimizer samples a finite box.
+
+        Args:
+            algorithm: the algorithm of the optimization, `None` for the
+                checks which hold for every algorithm.

         Raises:
-            ValueError: if a bound or a start value does not suit the scale.
+            ValueError: if a bound or a start value does not suit the scale
+                or the algorithm.
         """
         for p, scale in zip(self.parameters, self.scales_initialized, strict=True):
             space = f"'{scale.name}' parameter space"
             for key in ["lower_bound", "upper_bound"]:
                 value = getattr(p, key)
                 if not np.isfinite(value):
-                    raise ValueError(
-                        f"{self.opid}: the optimization requires a finite "
-                        f"'{key}', but FitParameter '{p.pid}' has '{value}'."
-                    )
+                    if scale.is_log:
+                        raise ValueError(
+                            f"{self.opid}: the optimization is performed in "
+                            f"{space}, which requires a finite '{key}', but "
+                            f"FitParameter '{p.pid}' has '{value}'."
+                        )
+                    if algorithm == OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION:
+                        raise ValueError(
+                            f"{self.opid}: the optimization with "
+                            f"'{algorithm.name}' samples a finite box and "
+                            f"requires a finite '{key}', but FitParameter "
+                            f"'{p.pid}' has '{value}'."
+                        )
+                    if p.start_value is None:
+                        raise ValueError(
+                            f"{self.opid}: FitParameter '{p.pid}' has the "
+                            f"infinite '{key}' '{value}' and is not sampled, so "
+                            f"it requires a 'start_value'."
+                        )
+                    continue
                 if scale.is_log and value <= 0.0:
                     raise ValueError(
                         f"{self.opid}: the optimization is performed in {space}, "
@@ -1281,6 +1305,7 @@ class OptimizationProblem(ObjectJSONEncoder):

         # the optimizer searches the scaled space, see `ParameterScaleType`
         x0log: np.ndarray = self.to_scale(x0)
+        self._validate_parameters(algorithm)

         if algorithm == OptimizationAlgorithmType.LEAST_SQUARE:
             # scipy least square optimizer
```

- [ ] **Step 6: Apply this patch to `docs/fitting.md`**

Apply this patch to `docs/fitting.md`:

```diff
diff --git a/docs/fitting.md b/docs/fitting.md
index 9b933d2..20b1bf6 100644
--- a/docs/fitting.md
+++ b/docs/fitting.md
@@ -118,7 +118,7 @@ print(FitParameter.parameters_to_df(fit_parameters))

 A `FitMappingCollection` without mappings uses all fit mappings of its experiment, they are resolved when the problem is initialized. `FitMappingCollection(use_mapping_weights=True)` weights the mappings by the weights of the `FitMapping` objects, e.g., the counts of the data, instead of the weights given here; setting both is an error.

-`FitSettings.parameter_scale` is the space the optimizer searches the parameters in: `LOG10` by default, because a rate constant spans orders of magnitude and an optimizer on the linear scale spends its steps on the largest parameters, and `LOG` or `LINEAR` if a problem wants them. The bounds, the start values and the fitted parameters are always on the linear scale, i.e. in the units of the model, only the search happens in the scaled space. A logarithm needs finite positive bounds and a positive start value, the linear scale only needs finite bounds.
+`FitSettings.parameter_scale` is the space the optimizer searches the parameters in: `LOG10` by default, because a rate constant spans orders of magnitude and an optimizer on the linear scale spends its steps on the largest parameters, and `LOG` or `LINEAR` if a problem wants them. The bounds, the start values and the fitted parameters are always on the linear scale, i.e. in the units of the model, only the search happens in the scaled space. A logarithm needs finite positive bounds and a positive start value. A parameter on the linear scale may have infinite bounds, which the local optimizer takes; the global optimizer samples a finite box and rejects them. `FitParameter.scale` gives one parameter a scale of its own, e.g. the linear scale for a parameter which is negative or zero while the others are searched on the logarithmic scale of the settings.

 The scale is a property of the optimization and not of the model or of the data, which is why it is part of the settings; PEtab v2 removed the `parameterScale` of its parameter table for the same reason.

@@ -216,7 +216,7 @@ The same settings are needed to report a fit, so they are stored with its result

 ## Running the optimization

-`run_optimization` samples `size` start points within the bounds (see `sbmlsim.fit.sampling`), runs the optimizer from every start point, in parallel on `n_cores`, and returns an `OptimizationResult`. The progress of the runs is shown on the console, with the runs which are done, the elapsed time and an estimate of the total runtime, e.g. `~ 0:12:30 total`; the estimate is the time per batch of `n_cores` runs times the number of batches, so it is there as soon as the first run is done and settles as more runs come back:
+`run_optimization` samples `size` start points within the bounds (see `sbmlsim.fit.sampling`; a parameter with an infinite bound is not sampled and starts from its start value), runs the optimizer from every start point, in parallel on `n_cores`, and returns an `OptimizationResult`. The progress of the runs is shown on the console, with the runs which are done, the elapsed time and an estimate of the total runtime, e.g. `~ 0:12:30 total`; the estimate is the time per batch of `n_cores` runs times the number of batches, so it is there as soon as the first run is done and settles as more runs come back:

 ```py
 from sbmlsim.fit.options import OptimizationAlgorithmType
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/fit`
Expected: PASS

- [ ] **Step 8: Run the checks**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: `All checks passed!`, `... files already formatted`, `All checks passed!` (zero diagnostics)

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "fit: a parameter on the linear scale may have infinite bounds, which are not sampled"
```

### Task 6: Derived changes of a problem and parameters which are not entities of the model

**Files:**
- Modify: `src/sbmlsim/fit/cli.py`
- Create: `src/sbmlsim/fit/derived.py`
- Modify: `src/sbmlsim/fit/objects.py`
- Modify: `src/sbmlsim/fit/optimization.py`
- Modify: `src/sbmlsim/fit/parameter_mapping.py`
- Test (create): `tests/fit/test_derived_changes.py`
- Test: `tests/sciml/test_package.py`

**Interfaces:**
- Consumes: `FitParameter.scale` of task 4, `ParameterMapping.indices_for`, `SimulatorSerial.uinfo`, `ExperimentRunner.Q_`.
- Produces: `sbmlsim.fit.derived.DerivedChanges` (runtime checkable protocol: `model: str`, `symbols() -> Collection[str]`, `targets() -> Collection[str]`, `check_parameters(targets: Collection[str]) -> None`, `derived_changes(values: Mapping[str, float], condition: str) -> dict[str, float]`); `sbmlsim.fit.objects.EXTERNAL_PREFIX = "sciml:"`, `FitParameter.is_external`, `FitParameter.entity_id`; `OptimizationProblem(..., hybridizations: Sequence[DerivedChanges] | None = None)`, `.hybridizations`, `.model_keys`, `.simulation_keys` (the ids of the model and the simulation of every mapping in its experiment), `.group_derived: list[list[tuple[DerivedChanges, dict[str, float]]]]`, `._derived_changes(k_group, simulation, simulator, quantities) -> dict[str, Quantity]`; `FitDefinition(hybridizations=())`.

This is the generic half of the fit of a hybrid problem, and nothing in it names a network. `sbmlsim.fit.derived.DerivedChanges` is the protocol of the hook: an object with a `model`, the `symbols()` it reads, the `targets()` it sets, `check_parameters(targets)` for what a fit must not write, and `derived_changes(values, condition)`. `OptimizationProblem(..., hybridizations=...)` takes such objects, checks them against the protocol, pickles them with the definition (`__getstate__`) and resolves them at `initialize` in `_group_derived_changes`: the hooks of the model of every simulation group, the values of the model their symbols name (read from the model as it was loaded, because a simulation changes the state of the roadrunner instance), and the checks which make a flat or wrong objective an error: a hook of a model no fit mapping is simulated with, a parameter which writes what a hook freezes or sets, a hook which reads what another hook sets (a hook reads the value of the model of its own target), two hooks of one target, and an external parameter which no hook reads. `_simulate_groups` calls `_derived_changes` after the changes of the parameters and after `normalize`, with the values in the units of the model in the order of precedence of the spec (fit parameters, changes of the simulation without the derived changes of the last evaluation, values of the model), and adds the changes as quantities of the unit of the target. A parameter whose target has the prefix `EXTERNAL_PREFIX = "sciml:"` is not an entity of the model: `ParameterMapping.changes_for` writes no change for it, `_store_model_parameters` takes its start value as the value of the model, and `FitParameter.is_external`/`entity_id` are the accessors. `FitDefinition` carries `hybridizations` so that a hybrid fit is defined like every other fit. The tests use a hook of the tests which scales an entity of the hctz model by an external factor, so they need no network.

- [ ] **Step 1: Create `tests/fit/test_derived_changes.py`**

Create `tests/fit/test_derived_changes.py`:

```python
"""Tests of the derived changes of a problem, with a hook of the tests.

The hook scales an entity of the model by a parameter of the fit which is
not an entity of the model, i.e. what a network before the simulation does
without a network.
"""

import pickle
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

import numpy as np
import pytest

from sbmlsim.fit import FitParameter, FitSettings
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.fit.derived import DerivedChanges
from sbmlsim.fit.objects import EXTERNAL_PREFIX
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ParameterScaleType

#: the entity of the hctz model the hook writes
TARGET = "KI__HCTZEX_k"

#: the parameter of the fit the hook reads, which is not an entity
FACTOR = "factor_k"


@dataclass(frozen=True)
class Scaling:
    """The change `target = factor * value of the model`.

    Attributes:
        model: id of the model in the experiment.
        target: the entity which is set.
        factor: the parameter of the fit which is read.
        frozen: ids the fit must not write.
        calls: the conditions the changes were calculated for.
    """

    model: str = "model"
    target: str = TARGET
    factor: str = FACTOR
    frozen: frozenset[str] = frozenset()
    calls: list[str] = field(default_factory=list, compare=False)

    def symbols(self) -> Collection[str]:
        return {self.factor, self.target}

    def targets(self) -> Collection[str]:
        return {self.target}

    def check_parameters(self, targets: Collection[str]) -> None:
        frozen = sorted(set(targets) & self.frozen)
        if frozen:
            raise ValueError(f"the parameters write {frozen}, which are frozen")

    def derived_changes(
        self, values: Mapping[str, float], condition: str
    ) -> dict[str, float]:
        self.calls.append(condition)
        return {self.target: values[self.factor] * values.get(self.target, 1.0)}


def _factor(start: float = 1.0) -> FitParameter:
    return FitParameter(
        FACTOR,
        start,
        lower_bound=0.0,
        upper_bound=np.inf,
        unit="dimensionless",
        target=f"{EXTERNAL_PREFIX}{FACTOR}",
        scale=ParameterScaleType.LINEAR,
    )


def _problem(
    definition: FitDefinition, parameters: list[FitParameter], **kwargs: object
) -> OptimizationProblem:
    """Build the iv problem with the parameters and the hybridizations."""
    return OptimizationProblem(
        opid="derived",
        mapping_collections=definition.collections(),
        fit_parameters=parameters,
        base_path=definition.base_path,
        data_path=definition.data_path,
        **kwargs,  # ty: ignore[invalid-argument-type]
    )


def test_the_hook_is_the_protocol() -> None:
    """A hybridization is what provides the derived changes."""
    assert isinstance(Scaling(), DerivedChanges)
    assert not isinstance(object(), DerivedChanges)
    with pytest.raises(TypeError, match="does not provide"):
        OptimizationProblem(
            "x",
            [],
            [_factor()],
            hybridizations=[object()],  # ty: ignore[invalid-argument-type]
        )


def test_an_external_parameter_writes_no_change_and_the_hook_reads_it(
    definition_hctz_iv: FitDefinition, fit_settings: FitSettings
) -> None:
    """The prediction with the factor is the prediction with the scaled entity."""
    scaling = Scaling()
    problem = _problem(definition_hctz_iv, [_factor(2.0)], hybridizations=[scaling])
    problem.initialize(fit_settings)
    assert problem.pids == [FACTOR]
    assert problem.parameter_mapping_initialized.changes_for(0, [1.0]) == {}
    nominal = float(problem.models[0].r[TARGET])
    assert problem.xmodel[0] == 2.0

    predictions = problem.predictions(np.array([2.0]))
    # once per simulation, with the id of the simulation
    assert scaling.calls == ["hctz_iv1", "hctz_iv35"]
    # the target of the hook was written, its value follows from the factor
    changes = problem.simulations[0].timecourses[0].changes
    assert changes[TARGET].magnitude == pytest.approx(2.0 * nominal)

    plain = _problem(
        definition_hctz_iv,
        [FitParameter(TARGET, 2.0 * nominal, 1e-10, 1.0, unit="1/ml")],
    )
    plain.initialize(fit_settings)
    expected = plain.predictions(np.array([2.0 * nominal]))
    for k, values in predictions.items():
        np.testing.assert_allclose(values, expected[k], rtol=1e-10)
    # the cost depends on the factor
    assert problem.cost_least_square(np.array([2.0])) != pytest.approx(
        problem.cost_least_square(np.array([1.0]))
    )


def test_the_hook_reads_the_changes_of_the_fit(
    definition_hctz_iv: FitDefinition, fit_settings: FitSettings
) -> None:
    """A value of the fit has precedence over the value of the model."""
    problem = _problem(
        definition_hctz_iv,
        [FitParameter(TARGET, 1e-4, 1e-10, 1.0, unit="1/ml")],
        hybridizations=[Scaling(target="Ka_dis_hctz", factor=TARGET)],
    )
    problem.initialize(fit_settings)
    model = problem.models[0]
    # the value of the model as it was loaded: a simulation changes the state
    nominal = float(model.r["Ka_dis_hctz"])
    problem.predictions(np.array([1e-4]))
    assert float(model.r["Ka_dis_hctz"]) != nominal
    changes = problem.simulations[0].timecourses[0].changes
    # the values of the fit reach the hook in the units of the model
    factor = problem.runner_initialized.Q_(1e-4, "1/ml").to(model.uinfo[TARGET])
    assert changes["Ka_dis_hctz"].magnitude == pytest.approx(factor.magnitude * nominal)
    # and the value of the model does not change between the evaluations
    problem.predictions(np.array([1e-4]))
    assert changes["Ka_dis_hctz"].magnitude == pytest.approx(factor.magnitude * nominal)


def test_a_problem_with_hooks_is_pickled(
    definition_hctz_iv: FitDefinition, fit_settings: FitSettings
) -> None:
    """The workers of a parallel fit get the definition with its hooks."""
    problem = _problem(definition_hctz_iv, [_factor(2.0)], hybridizations=[Scaling()])
    problem.initialize(fit_settings)
    restored = pickle.loads(pickle.dumps(problem))
    assert not restored.is_initialized
    assert restored.hybridizations == [Scaling()]
    restored.initialize(fit_settings)
    np.testing.assert_allclose(
        np.asarray(restored.residuals(restored.to_scale([2.0])), dtype=float),
        np.asarray(problem.residuals(problem.to_scale([2.0])), dtype=float),
    )


def test_a_definition_carries_its_hooks(definition_hctz_iv: FitDefinition) -> None:
    """A `FitDefinition` builds the problem with the hybridizations."""
    from dataclasses import replace

    definition = replace(
        definition_hctz_iv, parameters=[_factor()], hybridizations=[Scaling()]
    )
    assert definition.problem("hooked").hybridizations == [Scaling()]
    assert definition_hctz_iv.problem("plain").hybridizations == []


@pytest.mark.parametrize(
    ("parameters", "hybridizations", "message"),
    [
        ([_factor()], [], r"'factor_k' writes 'sciml:factor_k'.*no hybridization"),
        ([_factor()], [Scaling(model="other")], r"name the models \['other'\]"),
        (
            [_factor(), FitParameter(TARGET, 1e-4, 1e-10, 1.0, unit="1/ml")],
            [Scaling(frozen=frozenset({TARGET}))],
            "which are frozen",
        ),
        (
            [_factor()],
            [Scaling(), Scaling(target="Ka_dis_hctz", factor=TARGET)],
            r"reads \['KI__HCTZEX_k'\], which a hybridization sets",
        ),
        ([_factor()], [Scaling(), Scaling()], "two hybridizations.*set 'KI__HCTZEX_k'"),
        (
            [
                FitParameter(
                    FACTOR,
                    None,
                    0.0,
                    1.0,
                    unit="dimensionless",
                    target="sciml:x",
                    scale=ParameterScaleType.LINEAR,
                )
            ],
            [Scaling(factor="x")],
            "requires a 'start_value'",
        ),
    ],
)
def test_hooks_and_parameters_which_do_not_fit(
    definition_hctz_iv: FitDefinition,
    fit_settings: FitSettings,
    parameters: list[FitParameter],
    hybridizations: list[Scaling],
    message: str,
) -> None:
    """The problem refuses what would make the objective flat or wrong."""
    problem = _problem(definition_hctz_iv, parameters, hybridizations=hybridizations)
    with pytest.raises(ValueError, match=message):
        problem.initialize(fit_settings)


def test_a_hook_which_sets_no_entity(
    definition_hctz_iv: FitDefinition, fit_settings: FitSettings
) -> None:
    """A change of something which is not in the model is an error."""
    problem = _problem(
        definition_hctz_iv, [_factor()], hybridizations=[Scaling(target="nothing")]
    )
    problem.initialize(fit_settings)
    with pytest.raises(ValueError, match="sets 'nothing', which is not an entity"):
        problem.predictions(np.array([1.0]))


def test_an_external_target_names_something() -> None:
    """The prefix alone is not a target."""
    with pytest.raises(ValueError, match="names nothing"):
        FitParameter("x", 1.0, unit="dimensionless", target=EXTERNAL_PREFIX)
    parameter = FitParameter("x", 1.0, unit="dimensionless", target="sciml:x")
    assert parameter.is_external
    assert parameter.entity_id == "x"
    assert not FitParameter("x", 1.0, unit="dimensionless").is_external
```

- [ ] **Step 2: Apply this patch to `tests/sciml/test_package.py`**

Apply this patch to `tests/sciml/test_package.py`:

```diff
diff --git a/tests/sciml/test_package.py b/tests/sciml/test_package.py
index d7eda91..88d73c8 100644
--- a/tests/sciml/test_package.py
+++ b/tests/sciml/test_package.py
@@ -46,6 +46,7 @@ def test_the_package_does_not_import_the_networks() -> None:
     result = _python(
         "import sys; sys.modules['petab_sciml'] = None\n"
         "import sbmlsim, sbmlsim.fit, sbmlsim.testsuite, sbmlsim.fit.petab_v2\n"
+        "import sbmlsim.fit.cli, sbmlsim.fit.derived, sbmlsim.fit.runner\n"
         "assert 'sbmlsim.sciml' not in sys.modules\n"
     )
     assert result.returncode == 0, result.stderr
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest -q -x tests/fit/test_derived_changes.py tests/sciml/test_package.py`
Expected: FAIL at collection: `ModuleNotFoundError: No module named 'sbmlsim.fit.derived'`

- [ ] **Step 4: Create `src/sbmlsim/fit/derived.py`**

Create `src/sbmlsim/fit/derived.py`:

```python
"""Changes of a simulation which are calculated from the parameters of a fit.

A fit writes the values of its parameters into the model as the changes of a
simulation. A derived change is a change which is not a parameter but a
function of them: before every simulation the problem hands the values of the
parameters, of the changes of the simulation and of the model to the objects
it was given as `hybridizations`, and adds the changes they answer with to
the simulation.

`DerivedChanges` is what such an object provides. The neural networks of a
hybrid problem are the implementation, see
`sbmlsim.sciml.hybridization.Hybridization`: a network which runs before the
simulation calculates parameters and initial values of the model from
parameters of the fit.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class DerivedChanges(Protocol):
    """The changes of a simulation which follow from the values of a fit."""

    @property
    def model(self) -> str:
        """Get the id of the model in the experiment the changes belong to."""
        ...

    def symbols(self) -> Collection[str]:
        """Get the ids whose values `derived_changes` reads.

        Returns:
            The ids of entities of the model and of parameters of the fit
            which are not entities of the model.
        """
        ...

    def targets(self) -> Collection[str]:
        """Get the entities of the model `derived_changes` sets.

        Returns:
            The ids of the entities, or the selections of their
            concentrations.
        """
        ...

    def check_parameters(self, targets: Collection[str]) -> None:
        """Check the targets of the parameters of a fit.

        Args:
            targets: the entities the parameters of the fit write, without the
                prefix of a target which is not an entity of the model.

        Raises:
            ValueError: if the fit writes what the derived changes set or
                hold constant.
        """
        ...

    def derived_changes(
        self, values: Mapping[str, float], condition: str
    ) -> dict[str, float]:
        """Get the changes of a simulation for the values of the fit.

        Args:
            values: id -> value in the units of the model. The values are the
                ones of the parameters of the fit, of the changes of the
                simulation and of the model, in this order of precedence.
            condition: id of the simulation in its experiment.

        Returns:
            target -> value in the unit of the target in the model.

        Raises:
            ValueError: if a value is missing or a change is not a finite
                number.
        """
        ...
```

- [ ] **Step 5: Apply this patch to `src/sbmlsim/fit/objects.py`**

Apply this patch to `src/sbmlsim/fit/objects.py`:

```diff
diff --git a/src/sbmlsim/fit/objects.py b/src/sbmlsim/fit/objects.py
index bfd4c70..460a19e 100644
--- a/src/sbmlsim/fit/objects.py
+++ b/src/sbmlsim/fit/objects.py
@@ -24,6 +24,11 @@ if TYPE_CHECKING:

 logger = logging.getLogger(__name__)

+#: prefix of the target of a parameter which is not an entity of the model.
+#: No change of the simulation is written for it, the derived changes of the
+#: problem read its value, see `sbmlsim.fit.derived`
+EXTERNAL_PREFIX = "sciml:"
+

 def _isclose(a: float | None, b: float | None) -> bool:
     """Compare two optional floats, None only equals None."""
@@ -523,7 +528,8 @@ class FitParameter:

         Raises:
             ValueError: if the bounds or the start value are inconsistent, if
-                a value is not a number, or if the scale is not a scale.
+                a value is not a number, if the scale is not a scale, or if
+                the target is the prefix of an external target alone.
         """
         for key, value in (("lower_bound", lower_bound), ("upper_bound", upper_bound)):
             if value is None or np.isnan(value):
@@ -549,6 +555,11 @@ class FitParameter:
                 f"FitParameter '{pid}': the scale '{scale}' is not a "
                 f"`ParameterScaleType`."
             )
+        if target == EXTERNAL_PREFIX:
+            raise ValueError(
+                f"FitParameter '{pid}': the target '{target}' names nothing, an "
+                f"external target is '{EXTERNAL_PREFIX}<id>'."
+            )
         if lower_bound > upper_bound:
             raise ValueError(
                 f"FitParameter '{pid}': lower bound '{lower_bound}' is larger than "
@@ -579,6 +590,21 @@ class FitParameter:
         """Get the entity of the model the value is written to."""
         return self.target if self.target is not None else self.pid

+    @property
+    def is_external(self) -> bool:
+        """Check whether the parameter is not an entity of the model.
+
+        The target of such a parameter has the prefix `EXTERNAL_PREFIX`. The
+        fit writes no change for it, the derived changes of the problem read
+        its value.
+        """
+        return self.target_id.startswith(EXTERNAL_PREFIX)
+
+    @property
+    def entity_id(self) -> str:
+        """Get the target without the prefix of an external target."""
+        return self.target_id.removeprefix(EXTERNAL_PREFIX)
+
     @property
     def is_versioned(self) -> bool:
         """Check whether the parameter applies to a part of the data only."""
```

- [ ] **Step 6: Apply this patch to `src/sbmlsim/fit/parameter_mapping.py`**

Apply this patch to `src/sbmlsim/fit/parameter_mapping.py`:

```diff
diff --git a/src/sbmlsim/fit/parameter_mapping.py b/src/sbmlsim/fit/parameter_mapping.py
index 5e890ec..8e12987 100644
--- a/src/sbmlsim/fit/parameter_mapping.py
+++ b/src/sbmlsim/fit/parameter_mapping.py
@@ -256,10 +256,14 @@ class ParameterMapping:
                 residuals and referenced here.

         Returns:
-            The quantity by entity of the model.
+            The quantity by entity of the model. A target which is not an
+            entity of the model, see `FitParameter.is_external`, is not a
+            change: the derived changes of the problem read its value.
         """
         return {
-            target: quantities[index] for target, index in self._by_group[group].items()
+            target: quantities[index]
+            for target, index in self._by_group[group].items()
+            if not self.parameters[index].is_external
         }

     def coverage(self) -> list[CoverageRow]:
```

- [ ] **Step 7: Apply this patch to `src/sbmlsim/fit/optimization.py`**

Apply this patch to `src/sbmlsim/fit/optimization.py`:

```diff
diff --git a/src/sbmlsim/fit/optimization.py b/src/sbmlsim/fit/optimization.py
index 558d88a..bce0cc0 100644
--- a/src/sbmlsim/fit/optimization.py
+++ b/src/sbmlsim/fit/optimization.py
@@ -3,7 +3,7 @@
 import logging
 import time
 from collections import defaultdict
-from collections.abc import Callable, Sequence
+from collections.abc import Callable, Mapping, Sequence
 from pathlib import Path
 from typing import Any

@@ -14,6 +14,7 @@ from scipy import interpolate

 from sbmlsim.console import console
 from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
+from sbmlsim.fit.derived import DerivedChanges
 from sbmlsim.fit.helpers import _filters
 from sbmlsim.fit.objects import (
     UNUSED_KINDS,
@@ -176,18 +177,42 @@ class OptimizationProblem(ObjectJSONEncoder):
         fit_parameters: list[FitParameter],
         base_path: Path | None = None,
         data_path: Path | None = None,
+        hybridizations: Sequence[DerivedChanges] | None = None,
     ):
         """Optimization problem.

         The problem must be pickable for parallelization !
         So initialize must be run to create the non-pickable instances.

-        :param opid: id for optimization problem
-        :param mapping_collections:
-        :param fit_parameters:
+        Args:
+            opid: id for optimization problem.
+            mapping_collections: the fit mappings of the problem.
+            fit_parameters: the parameters of the fit.
+            base_path: directory the models of the experiments are relative
+                to.
+            data_path: directory of the data of the experiments.
+            hybridizations: the derived changes of the problem, i.e. changes
+                of a simulation which are calculated from the parameters of
+                the fit, see `sbmlsim.fit.derived`. The hybridizations of the
+                neural networks of a hybrid problem are what implements
+                them. They are part of the definition and are pickled with
+                it.
+
+        Raises:
+            ValueError: if the problem has no parameters or two parameters of
+                one id.
+            TypeError: if a hybridization does not provide the derived
+                changes.
         """
         super().__init__()
         self.opid: str = opid
+        self.hybridizations: list[DerivedChanges] = list(hybridizations or [])
+        for hybridization in self.hybridizations:
+            if not isinstance(hybridization, DerivedChanges):
+                raise TypeError(
+                    f"'{opid}': the hybridization '{hybridization}' does not "
+                    f"provide `sbmlsim.fit.derived.DerivedChanges`."
+                )
         self.mapping_collections = []
         for collection in mapping_collections:
             if collection.exclude:
@@ -247,6 +272,7 @@ class OptimizationProblem(ObjectJSONEncoder):
             fit_parameters=self.parameters,
             base_path=self.base_path,
             data_path=self.data_path,
+            hybridizations=self.hybridizations,
         )
         return fresh.__dict__

@@ -278,6 +304,13 @@ class OptimizationProblem(ObjectJSONEncoder):
         self.models: list[Any] = []
         self.simulations: list[Any] = []
         self.selections: list[Any] = []
+        #: id of the model and of the simulation of every mapping in its
+        #: experiment
+        self.model_keys: list[str] = []
+        self.simulation_keys: list[str] = []
+        #: the derived changes of every simulation group with the values of
+        #: the model they read, see `_group_derived_changes`
+        self.group_derived: list[list[tuple[DerivedChanges, dict[str, float]]]] = []
         # indices of the mappings which share a simulation, see `_group_mappings`
         self.mapping_groups: list[list[int]] = []
         #: which parameter writes which entity in which simulation, resolved
@@ -731,6 +764,8 @@ class OptimizationProblem(ObjectJSONEncoder):
                 self.models.append(model)
                 self.simulations.append(simulation)
                 self.selections.append(selections)
+                self.model_keys.append(task.model_id)
+                self.simulation_keys.append(task.simulation_id)

                 # store information
                 self.experiment_keys.append(sid)
@@ -788,6 +823,7 @@ class OptimizationProblem(ObjectJSONEncoder):
             ],
         )
         self._check_shared_simulation_bindings()
+        self._group_derived_changes()

         # set simulator instance with arguments
         simulator = SimulatorSerial(
@@ -998,14 +1034,126 @@ class OptimizationProblem(ObjectJSONEncoder):
                     f"for both."
                 )

+    def _group_derived_changes(self) -> None:
+        """Resolve the derived changes of every simulation group.
+
+        The derived changes of a group are the hybridizations of its model.
+        The values of the model which they read are stored here, from the
+        model as it was loaded: a simulation changes the state of the model.
+
+        Raises:
+            ValueError: if a hybridization names a model no fit mapping is
+                simulated with, if a parameter of the fit writes what a
+                hybridization sets or holds constant, if a hybridization
+                reads what another one sets, or if a parameter which is not
+                an entity of the model is read by no hybridization.
+        """
+        self.group_derived = []
+        unknown = sorted({h.model for h in self.hybridizations} - set(self.model_keys))
+        if unknown:
+            raise ValueError(
+                f"'{self.opid}': the hybridizations name the models {unknown}, "
+                f"but the fit mappings are simulated with the models "
+                f"{sorted(set(self.model_keys))}."
+            )
+        mapping = self.parameter_mapping_initialized
+        read: set[str] = set()
+        for k_group, group in enumerate(self.mapping_groups):
+            k0 = group[0]
+            hybridizations = [
+                h for h in self.hybridizations if h.model == self.model_keys[k0]
+            ]
+            targets = [
+                self.parameters[index].entity_id
+                for index in mapping.indices_for(k_group).values()
+            ]
+            derived: list[tuple[DerivedChanges, dict[str, float]]] = []
+            written: dict[str, int] = {}
+            for k, hybridization in enumerate(hybridizations):
+                hybridization.check_parameters(targets)
+                for target in hybridization.targets():
+                    if target in written:
+                        raise ValueError(
+                            f"'{self.opid}': two hybridizations of the model "
+                            f"'{hybridization.model}' set '{target}'."
+                        )
+                    written[target] = k
+            for hybridization in hybridizations:
+                symbols = set(hybridization.symbols())
+                # a hybridization reads the value of the model of what it sets
+                others = {
+                    t
+                    for t, k in written.items()
+                    if hybridizations[k] is not hybridization
+                }
+                chained = sorted(symbols & others)
+                if chained:
+                    raise ValueError(
+                        f"'{self.opid}': a hybridization of the model "
+                        f"'{hybridization.model}' reads {chained}, which a "
+                        f"hybridization sets. Derived changes are calculated "
+                        f"from the parameters of the fit and the model, not "
+                        f"from each other."
+                    )
+                read |= symbols
+                derived.append(
+                    (hybridization, self._model_values(self.models[k0], symbols))
+                )
+            self.group_derived.append(derived)
+
+        for parameter in self.parameters:
+            if parameter.is_external and parameter.entity_id not in read:
+                raise ValueError(
+                    f"'{self.opid}': FitParameter '{parameter.pid}' writes "
+                    f"'{parameter.target_id}', which is not an entity of a model "
+                    f"and which no hybridization reads. The objective does not "
+                    f"depend on the parameter."
+                )
+
+    @staticmethod
+    def _model_values(
+        model: RoadrunnerSBMLModel, symbols: set[str]
+    ) -> dict[str, float]:
+        """Get the values of the entities of a model which are symbols.
+
+        Args:
+            model: the model as it was loaded, with its changes.
+            symbols: ids, of which some are entities of the model.
+
+        Returns:
+            id -> value of the symbols which are entities of the model.
+
+        Raises:
+            ValueError: if the model is not loaded in roadrunner.
+        """
+        if model.r is None:
+            raise ValueError(f"Model '{model}' is not loaded in roadrunner.")
+        values: dict[str, float] = {}
+        for symbol in sorted(symbols):
+            try:
+                values[symbol] = float(model.r[symbol])
+            except (RuntimeError, TypeError, ValueError):
+                # not an entity of the model: a parameter of the fit or a
+                # constant of the hybridization
+                continue
+            if symbol in model.changes:
+                change = model.changes[symbol]
+                values[symbol] = float(
+                    change.magnitude if isinstance(change, Quantity) else change
+                )
+        return values
+
     def _store_model_parameters(self) -> None:
         """Store the initial values of the fitted parameters in the models.

         The values are read from the first model, a model which starts from
-        different values is reported.
+        different values is reported. A parameter which is not an entity of
+        the model starts from its start value.

         Raises:
-            ValueError: if a model is not loaded in roadrunner.
+            ValueError: if a model is not loaded in roadrunner, or if a
+                parameter which is not an entity of the model has no start
+                value.
         """
         for k_model, model in enumerate(self.models):
             if model.r is None:
@@ -1013,6 +1161,15 @@ class OptimizationProblem(ObjectJSONEncoder):

             for k, parameter in enumerate(self.parameters):
                 target = parameter.target_id
+                if parameter.is_external:
+                    if parameter.start_value is None:
+                        raise ValueError(
+                            f"'{self.opid}': FitParameter '{parameter.pid}' "
+                            f"writes '{target}', which is not an entity of the "
+                            f"model, so it requires a 'start_value'."
+                        )
+                    self.xmodel[k] = parameter.start_value
+                    continue
                 pid_value = model.r[target]
                 if target in model.changes:
                     change = model.changes[target]
@@ -1424,6 +1581,10 @@ class OptimizationProblem(ObjectJSONEncoder):
                 selections=sorted({s for k in indices for s in self.selections[k]})
             )
             simulation.normalize(uinfo=simulator.uinfo)
+            if self.group_derived[k_group]:
+                simulation.timecourses[0].changes.update(
+                    self._derived_changes(k_group, simulation, simulator, quantities)
+                )

             df: pd.DataFrame | None
             try:
@@ -1443,6 +1604,71 @@ class OptimizationProblem(ObjectJSONEncoder):

         return results

+    def _derived_changes(
+        self,
+        k_group: int,
+        simulation: TimecourseSim,
+        simulator: SimulatorSerial,
+        quantities: Sequence[Quantity],
+    ) -> dict[str, Quantity]:
+        """Get the derived changes of the simulation of a group.
+
+        The values the hybridizations calculate from are the values of the
+        parameters of the fit, the changes of the first timecourse of the
+        simulation and the values of the model, in this order of precedence
+        and in the units of the model.
+
+        Args:
+            k_group: index of the simulation group.
+            simulation: the simulation of the group with the changes of the
+                parameters, normalized to the units of the model.
+            simulator: simulator of the problem, with the model of the group.
+            quantities: the quantity of every parameter, in the order of the
+                parameter vector.
+
+        Returns:
+            The changes by entity of the model, in the unit of the entity.
+
+        Raises:
+            ValueError: if a hybridization cannot calculate its changes, or
+                if a target is not an entity of the model.
+        """
+        uinfo = simulator.uinfo
+        Q_ = self.runner_initialized.Q_
+        mapping = self.parameter_mapping_initialized
+        k0 = self.mapping_groups[k_group][0]
+        derived = self.group_derived[k_group]
+        written = {target for h, _ in derived for target in h.targets()}
+
+        changes: dict[str, float] = {
+            key: float(value.magnitude if isinstance(value, Quantity) else value)
+            for key, value in simulation.timecourses[0].changes.items()
+            # a derived change of the last evaluation is not a value
+            if key not in written
+        }
+        fitted: dict[str, float] = {}
+        for index in mapping.indices_for(k_group).values():
+            parameter = self.parameters[index]
+            quantity = quantities[index]
+            if not parameter.is_external and parameter.target_id in uinfo:
+                quantity = quantity.to(uinfo[parameter.target_id])
+            fitted[parameter.entity_id] = float(quantity.magnitude)
+
+        result: dict[str, Quantity] = {}
+        for hybridization, model_values in derived:
+            values: Mapping[str, float] = {**model_values, **changes, **fitted}
+            for target, value in hybridization.derived_changes(
+                values, condition=self.simulation_keys[k0]
+            ).items():
+                if target not in uinfo:
+                    raise ValueError(
+                        f"'{self.opid}': a hybridization of the model "
+                        f"'{hybridization.model}' sets '{target}', which is not "
+                        f"an entity of the model."
+                    )
+                result[target] = Q_(value, uinfo[target])
+        return result
+
     def _interpolate(self, k: int, df: pd.DataFrame) -> np.ndarray:
         """Get the simulation of a fit mapping at its reference data.

```

- [ ] **Step 8: Apply this patch to `src/sbmlsim/fit/cli.py`**

Apply this patch to `src/sbmlsim/fit/cli.py`:

```diff
diff --git a/src/sbmlsim/fit/cli.py b/src/sbmlsim/fit/cli.py
index 8195087..babd717 100644
--- a/src/sbmlsim/fit/cli.py
+++ b/src/sbmlsim/fit/cli.py
@@ -36,6 +36,7 @@ from typing import Any

 from sbmlsim import log
 from sbmlsim.fit import display
+from sbmlsim.fit.derived import DerivedChanges
 from sbmlsim.fit.fisher import FisherInformation, fisher_information
 from sbmlsim.fit.identifiability import (
     IdentifiabilityResult,
@@ -95,6 +96,9 @@ class FitDefinition:
         base_path: base path of the simulation experiments.
         data_path: path of the datasets of the simulation experiments.
         settings: settings of the fit.
+        hybridizations: the derived changes of the problem, see
+            `sbmlsim.fit.derived`, e.g. the hybridizations of its neural
+            networks.
     """

     mapping_collections: Callable[[], dict[str, list[FitMappingCollection]]]
@@ -102,6 +106,7 @@ class FitDefinition:
     base_path: Path
     data_path: Path
     settings: FitSettings = field(default_factory=FitSettings)
+    hybridizations: Sequence[DerivedChanges] = ()

     def collections(
         self, study_ids: Sequence[str] | None = None
@@ -155,6 +160,7 @@ class FitDefinition:
             fit_parameters=self.parameters,
             base_path=self.base_path,
             data_path=self.data_path,
+            hybridizations=self.hybridizations,
         )


```

- [ ] **Step 9: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/fit tests/sciml/test_package.py`
Expected: PASS

- [ ] **Step 10: Run the checks**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: `All checks passed!`, `... files already formatted`, `All checks passed!` (zero diagnostics)

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "fit: derived changes of a problem and parameters which are not entities of the model"
```

### Task 7: `Hybridization`: where a network sits

**Files:**
- Modify: `src/sbmlsim/sciml/errors.py`
- Create: `src/sbmlsim/sciml/hybridization.py`
- Test (create): `tests/data/models/lotka_volterra.xml`
- Test (create): `tests/sciml/hybrid.py`
- Test (create): `tests/sciml/test_hybridization.py`

**Interfaces:**
- Consumes: `Network`, `input_id`, `output_id`, `parse_io_id` of task 2; `formula_symbols`, `evaluate_formula`, `TIME` of task 3; `DerivedChanges` of task 6; `PLACEHOLDER` of the interpreter.
- Produces: `sbmlsim.sciml.errors.NetworkHybridizationError`; `sbmlsim.sciml.hybridization.NetworkPattern` (`PRE_INITIALIZATION`, `RHS`, `OBSERVABLE`, `.is_compiled`), `ALL_CONDITIONS`, `NetworkInput(formula=None, formulas=None, arrays=None)` with `.all_formulas()`, `.is_conditional`, `.shape`, `.formula_of(condition)`, `.array_of(condition)`, `Hybridization(network, pattern, model, inputs, outputs, frozen=frozenset(), constants={})` with `.input_shapes()`, `.output_shapes()`, `.symbols()`, `.targets()`, `.check_parameters(targets)`, `.input_values(values, condition)`, `.derived_changes(values, condition)`, `.validate(sbml_path)`, `.error(message)`; `input_shapes(network, inputs)`, `output_shapes(network, shapes)`, `entity_of(target)`, `read_model(sbml_path, network) -> (document, model)`. `tests/sciml/hybrid.py`: `MODEL_PATH`, `write_model(path, level, version)`, `feed_forward(...)`, `two_inputs(...)`, `convolution(...)`.

`Hybridization` is the native description of where a network sits and implements the `DerivedChanges` protocol of task 6. It differs from the dataclass of the spec in three points which the cases of the test suite force. First, `NetworkInput` has `formulas` next to `formula` and `arrays`: case 003 sets the inputs of a network before the simulation by its conditions (`cond1: net1_input1 = 10.0`, `cond2: net1_input1 = net1_input_pre1`), which is one formula per condition and not a change of the model (`net1_input1` is not an entity, the simulator would refuse it). Second, `constants` holds the values of the symbols of the formulas which are neither entities of the model nor parameters of the fit, i.e. the parameters of the parameter table such as `net1_input_pre1 = 1.0` of case 002, which keeps their ids for the exporter of phase 4. Third, a compiled network may have an input which is an array (cases 036 to 039): a constant array is a part of the compiled model, and the arrays of conditions are set before every simulation by `derived_changes`, so the spec's rule that a compiled network has only formulas as inputs is replaced by the rule that it has one formula per input. The condition of a simulation is the id of the simulation in its experiment, `ALL_CONDITIONS = "0"` names the value of every condition. The class validates everything it can without the model when it is created (the ids, the shapes of the inputs which the elements have to cover, the outputs against the shapes of a forward pass with zeros, the frozen elements, the patterns) and `validate(sbml_path)` checks the targets, the constants and the inputs against the model; every error is a `NetworkHybridizationError` which names the network and the input, output or target. `input_shapes` and `output_shapes` are module functions because the reader needs them before it has a hybridization. The Lotka-Volterra model of the test suite becomes the test model `tests/data/models/lotka_volterra.xml` (the compartment has `spatialDimensions="3"` so that libsbml converts it to level 2 for a test), and `tests/sciml/hybrid.py` builds it in other levels and builds small networks with random arrays.

- [ ] **Step 1: Create `tests/sciml/hybrid.py`**

Create `tests/sciml/hybrid.py`:

```python
"""A small hybrid problem for the tests: a model and its networks.

The model is the model of Lotka and Volterra of the PEtab SciML test suite:

    d prey / dt     = alpha * prey - beta * prey * predator
    d predator / dt = gamma * prey * predator - delta * predator

A network replaces `gamma` or sets it before the simulation.
"""

from pathlib import Path

import libsbml
import numpy as np
from petab_sciml import Input, Layer, NNModel, Node

from sbmlsim.sciml import Network

#: the model, with the species `prey` and `predator` and the parameters
#: `alpha`, `beta`, `gamma` and `delta`
MODEL_PATH = Path(__file__).parent.parent / "data" / "models" / "lotka_volterra.xml"


def write_model(path: Path, level: int = 3, version: int = 1) -> Path:
    """Write a copy of the model of Lotka and Volterra.

    Args:
        path: the file the model is written to.
        level: the level of SBML.
        version: the version of SBML.

    Returns:
        The path.
    """
    document = libsbml.readSBMLFromFile(str(MODEL_PATH))
    if (level, version) != (document.getLevel(), document.getVersion()):
        assert document.setLevelAndVersion(level, version)
    libsbml.writeSBMLToFile(document, str(path))
    return path


def _node(
    name: str, op: str, target: str, args: list, kwargs: dict | None = None
) -> Node:
    return Node(name=name, op=op, target=target, args=args, kwargs=kwargs or {})


def feed_forward(
    sid: str = "net1",
    n_inputs: int = 2,
    n_hidden: int = 3,
    n_outputs: int = 1,
    activation: str = "tanh",
    kwargs: dict | None = None,
    seed: int = 1,
) -> Network:
    """Build `layer2(activation(layer1(x)))` with random arrays.

    Args:
        sid: id of the network.
        n_inputs: number of inputs.
        n_hidden: number of units of the hidden layer.
        n_outputs: number of outputs.
        activation: the function between the layers.
        kwargs: the keyword arguments of the function.
        seed: seed of the arrays.

    Returns:
        The network.
    """
    rng = np.random.default_rng(seed)
    model = NNModel(
        nn_model_id=sid,
        inputs=[Input(input_id="input0")],
        layers=[
            Layer(
                layer_id="layer1",
                layer_type="Linear",
                args={"in_features": n_inputs, "out_features": n_hidden, "bias": True},
            ),
            Layer(
                layer_id="layer2",
                layer_type="Linear",
                args={"in_features": n_hidden, "out_features": n_outputs, "bias": True},
            ),
        ],
        forward=[
            _node("net_input", "placeholder", "net_input", []),
            _node("layer1", "call_module", "layer1", ["net_input"]),
            _node("act", "call_function", activation, ["layer1"], kwargs),
            _node("layer2", "call_module", "layer2", ["act"]),
            _node("output", "output", "output", ["layer2"]),
        ],
    )
    return Network(
        sid=sid,
        model=model,
        parameters={
            "layer1": {
                "weight": rng.normal(size=(n_hidden, n_inputs)),
                "bias": rng.normal(size=n_hidden),
            },
            "layer2": {
                "weight": rng.normal(size=(n_outputs, n_hidden)),
                "bias": rng.normal(size=n_outputs),
            },
        },
    )


def two_inputs(sid: str = "net6", seed: int = 2) -> Network:
    """Build `layer1(cat(x0, x1))` with an input of one and one of three elements.

    Args:
        sid: id of the network.
        seed: seed of the arrays.

    Returns:
        The network.
    """
    rng = np.random.default_rng(seed)
    model = NNModel(
        nn_model_id=sid,
        inputs=[Input(input_id="input0"), Input(input_id="input1")],
        layers=[
            Layer(
                layer_id="layer1",
                layer_type="Linear",
                args={"in_features": 4, "out_features": 1, "bias": True},
            )
        ],
        forward=[
            _node("x0", "placeholder", "x0", []),
            _node("x1", "placeholder", "x1", []),
            _node("cat", "call_function", "cat", [["x0", "x1"]], {"dim": 0}),
            _node("layer1", "call_module", "layer1", ["cat"]),
            _node("output", "output", "output", ["layer1"]),
        ],
    )
    return Network(
        sid=sid,
        model=model,
        parameters={
            # small weights, the model with the network stays a tame ODE
            "layer1": {
                "weight": 0.2 * rng.normal(size=(1, 4)),
                "bias": rng.normal(size=1),
            }
        },
    )


def convolution(sid: str = "net3", seed: int = 3) -> Network:
    """Build `layer2(flatten(layer1(x)))` with a convolution, numpy only.

    Args:
        sid: id of the network.
        seed: seed of the arrays.

    Returns:
        The network, which takes an input of the shape `(1, 4, 4)`.
    """
    rng = np.random.default_rng(seed)
    model = NNModel(
        nn_model_id=sid,
        inputs=[Input(input_id="input0")],
        layers=[
            Layer(
                layer_id="layer1",
                layer_type="Conv2d",
                args={"in_channels": 1, "out_channels": 1, "kernel_size": 3},
            ),
            Layer(layer_id="layer2", layer_type="Flatten", args={"start_dim": 0}),
            Layer(
                layer_id="layer3",
                layer_type="Linear",
                args={"in_features": 4, "out_features": 1, "bias": True},
            ),
        ],
        forward=[
            _node("net_input", "placeholder", "net_input", []),
            _node("layer1", "call_module", "layer1", ["net_input"]),
            _node("layer2", "call_module", "layer2", ["layer1"]),
            _node("layer3", "call_module", "layer3", ["layer2"]),
            _node("output", "output", "output", ["layer3"]),
        ],
    )
    return Network(
        sid=sid,
        model=model,
        parameters={
            "layer1": {
                "weight": rng.normal(size=(1, 1, 3, 3)),
                "bias": rng.normal(size=1),
            },
            "layer3": {"weight": rng.normal(size=(1, 4)), "bias": rng.normal(size=1)},
        },
    )
```

- [ ] **Step 2: Create `tests/data/models/lotka_volterra.xml`**

Create `tests/data/models/lotka_volterra.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="lv">
    <listOfCompartments>
      <compartment id="default" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="prey" compartment="default" initialAmount="0.44249296" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="predator" compartment="default" initialAmount="4.6280594" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="alpha" value="1.3" constant="true"/>
      <parameter id="beta" value="0.9" constant="true"/>
      <parameter id="gamma" value="0.8" constant="true"/>
      <parameter id="delta" value="1.8" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="v1" reversible="false" fast="false">
        <listOfProducts>
          <speciesReference species="prey" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <listOfModifiers>
          <modifierSpeciesReference species="prey"/>
        </listOfModifiers>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <times/>
              <ci> alpha </ci>
              <ci> prey </ci>
            </apply>
          </math>
        </kineticLaw>
      </reaction>
      <reaction id="v2" reversible="false" fast="false">
        <listOfReactants>
          <speciesReference species="prey" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfModifiers>
          <modifierSpeciesReference species="predator"/>
        </listOfModifiers>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <times/>
              <ci> beta </ci>
              <ci> prey </ci>
              <ci> predator </ci>
            </apply>
          </math>
        </kineticLaw>
      </reaction>
      <reaction id="v3" reversible="false" fast="false">
        <listOfProducts>
          <speciesReference species="predator" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <listOfModifiers>
          <modifierSpeciesReference species="prey"/>
          <modifierSpeciesReference species="predator"/>
        </listOfModifiers>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <times/>
              <ci> gamma </ci>
              <ci> prey </ci>
              <ci> predator </ci>
            </apply>
          </math>
        </kineticLaw>
      </reaction>
      <reaction id="v4" reversible="false" fast="false">
        <listOfReactants>
          <speciesReference species="predator" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <times/>
              <ci> delta </ci>
              <ci> predator </ci>
            </apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
```

- [ ] **Step 3: Create `tests/sciml/test_hybridization.py`**

Create `tests/sciml/test_hybridization.py`:

```python
"""Tests of the hybridization of a network and a model."""

import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

import libsbml
import numpy as np
import pytest

from sbmlsim.fit.derived import DerivedChanges
from sbmlsim.sciml import NetworkImportError
from sbmlsim.sciml.errors import NetworkHybridizationError
from sbmlsim.sciml.hybridization import (
    ALL_CONDITIONS,
    Hybridization,
    NetworkInput,
    NetworkPattern,
    entity_of,
)
from tests.sciml.hybrid import convolution, feed_forward, two_inputs, write_model

PRE = NetworkPattern.PRE_INITIALIZATION
RHS = NetworkPattern.RHS
OBSERVABLE = NetworkPattern.OBSERVABLE


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    """Write the model of Lotka and Volterra."""
    return write_model(tmp_path / "lv.xml")


def _inputs(first: str = "prey", second: str = "predator") -> dict[str, NetworkInput]:
    return {
        "net1__input0__0": NetworkInput(formula=first),
        "net1__input0__1": NetworkInput(formula=second),
    }


def _hybridization(pattern: NetworkPattern | str = RHS, **kwargs: Any) -> Hybridization:
    compiled = NetworkPattern(pattern).is_compiled if pattern != "ode" else True
    arguments: dict[str, Any] = {
        "network": feed_forward(),
        "pattern": pattern,
        "model": "lv",
        "inputs": _inputs() if compiled else _inputs("alpha", "2 * k"),
        "outputs": {"net1__output0__0": "gamma"},
    }
    if not compiled:
        arguments["constants"] = {"k": 0.5}
    arguments.update(kwargs)
    return Hybridization(**arguments)


# --- AN INPUT ---


def test_an_input_is_a_formula_or_arrays() -> None:
    """Exactly one of the three is given."""
    assert NetworkInput(formula="prey").all_formulas() == ["prey"]
    assert NetworkInput(formulas={"e1": "1.0", "e2": "k"}).all_formulas() == [
        "1.0",
        "k",
    ]
    assert NetworkInput(arrays={"e1": [1.0, 2.0]}).all_formulas() == []
    with pytest.raises(ValueError, match="but nothing is given"):
        NetworkInput()
    with pytest.raises(ValueError, match=r"but \['formula', 'arrays'\] is given"):
        NetworkInput(formula="prey", arrays={ALL_CONDITIONS: [1.0]})
    with pytest.raises(ValueError, match="formulas of an input name no condition"):
        NetworkInput(formulas={})
    with pytest.raises(ValueError, match="arrays of an input name no condition"):
        NetworkInput(arrays={})


@pytest.mark.parametrize("formula", ["", "prey +", "f(prey)"])
def test_the_formula_of_an_input_is_math(formula: str) -> None:
    """A formula which is not math is an error when the input is created."""
    with pytest.raises(ValueError, match="The formula"):
        NetworkInput(formula=formula)
    with pytest.raises(ValueError, match="The formula"):
        NetworkInput(formulas={"e1": formula})


def test_the_arrays_of_an_input() -> None:
    """The arrays have one shape and finite values, and are copies."""
    values = np.array([1.0, 2.0, 3.0])
    network_input = NetworkInput(arrays={"e1": values, "e2": [3, 2, 1]})
    values[0] = 5.0
    assert network_input.shape == (3,)
    np.testing.assert_array_equal(network_input.array_of("e1"), [1.0, 2.0, 3.0])
    stored = network_input.array_of("e1")
    assert stored is not None
    with pytest.raises(ValueError, match="read-only"):
        stored[0] = 5.0
    with pytest.raises(ValueError, match=r"one shape.*'e1': \(3,\).*'e2': \(2,\)"):
        NetworkInput(arrays={"e1": [1.0, 2.0, 3.0], "e2": [1.0, 2.0]})
    with pytest.raises(ValueError, match=r"condition 'e2'.*not finite"):
        NetworkInput(arrays={"e1": [1.0], "e2": [np.nan]})


def test_the_value_of_a_condition() -> None:
    """A condition which is not listed has the value of all conditions."""
    arrays = NetworkInput(arrays={ALL_CONDITIONS: [1.0], "e2": [2.0]})
    np.testing.assert_array_equal(arrays.array_of("e1"), [1.0])
    np.testing.assert_array_equal(arrays.array_of("e2"), [2.0])
    assert arrays.is_conditional
    assert arrays.formula_of("e1") is None
    assert NetworkInput(arrays={"e2": [2.0]}).array_of("e1") is None
    assert not NetworkInput(arrays={ALL_CONDITIONS: [1.0]}).is_conditional

    formulas = NetworkInput(formulas={"e1": "10.0", ALL_CONDITIONS: "k"})
    assert formulas.formula_of("e1") == "10.0"
    assert formulas.formula_of("e2") == "k"
    assert formulas.array_of("e1") is None
    assert NetworkInput(formulas={"e1": "10.0"}).formula_of("e2") is None
    assert NetworkInput(formula="k").formula_of("e2") == "k"
    assert not NetworkInput(formula="k").is_conditional


def test_the_equality_of_inputs() -> None:
    """Inputs are compared by their formulas and the elements of their arrays."""
    assert NetworkInput(formula="prey") == NetworkInput(formula="prey")
    assert NetworkInput(formula="prey") != NetworkInput(formula="predator")
    assert NetworkInput(formula="prey") != NetworkInput(formulas={"e1": "prey"})
    assert NetworkInput(arrays={"e1": [1.0, 2.0]}) == NetworkInput(
        arrays={"e1": np.array([1.0, 2.0])}
    )
    assert NetworkInput(arrays={"e1": [1.0, 2.0]}) != NetworkInput(
        arrays={"e1": [1.0, 3.0]}
    )
    assert NetworkInput(arrays={"e1": [1.0]}) != NetworkInput(arrays={"e2": [1.0]})
    assert NetworkInput(arrays={"e1": [1.0]}) != NetworkInput(formula="prey")
    assert NetworkInput(formula="prey") != "prey"


# --- THE HYBRIDIZATION AND ITS NETWORK ---


def test_a_hybridization() -> None:
    """The attributes are copies, the pattern is read from its value."""
    outputs = {"net1__output0__0": "gamma"}
    hybridization = _hybridization(pattern="rhs", outputs=outputs, frozen=[])
    outputs["net1__output0__0"] = "alpha"
    assert hybridization.pattern is RHS
    assert hybridization.outputs == {"net1__output0__0": "gamma"}
    assert hybridization.frozen == frozenset()
    assert hybridization.input_shapes() == [(2,)]
    assert hybridization.output_shapes() == [(1,)]
    assert isinstance(hybridization, DerivedChanges)


def test_the_patterns() -> None:
    """The networks of two patterns are compiled into the model."""
    assert not PRE.is_compiled
    assert RHS.is_compiled
    assert OBSERVABLE.is_compiled
    with pytest.raises(NetworkHybridizationError, match=r"'net1'.*'ode' is not one of"):
        _hybridization(pattern="ode")


def test_the_entity_of_a_target() -> None:
    """The target of a concentration names its species."""
    assert entity_of("[prey]") == "prey"
    assert entity_of("prey") == "prey"


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({"net1__input0__0": NetworkInput(formula="prey")}, r"\(1,\)\] do not fit"),
        (
            {
                "net1__input0__0": NetworkInput(formula="prey"),
                "net1__input0__2": NetworkInput(formula="prey"),
            },
            r"the input 0 has the shape \(3,\), but its elements \[\(1,\)\] are missing",
        ),
        ({}, "the input 0 of the forward pass is missing"),
        (
            {**_inputs(), "net1__input1__0": NetworkInput(formula="prey")},
            "is the input 1, but the forward pass has 1 inputs",
        ),
        ({"net1__in0__0": NetworkInput(formula="prey")}, "is not the id of an input"),
        ({"net1__input0": NetworkInput(formula="prey")}, "name the element"),
        (
            {"net1__input0__0": NetworkInput(arrays={"e1": [1.0, 2.0]})},
            r"is an element, but its arrays have the shape \(2,\)",
        ),
        (
            {
                "net1__input0": NetworkInput(arrays={"e1": [1.0, 2.0]}),
                "net1__input0__0": NetworkInput(formula="prey"),
            },
            r"the inputs \[0\] are given as an array and element by element",
        ),
        (
            {
                "net1__input0__0": NetworkInput(formula="prey"),
                "net1__input0__0_1": NetworkInput(formula="prey"),
            },
            "differ in their number of axes",
        ),
        ({"net1__input0__0": "prey"}, "is not a `NetworkInput`"),
        ({"net1__input0": NetworkInput(arrays={"e1": np.ones(3)})}, "do not fit"),
    ],
)
def test_inputs_which_do_not_fit_the_network(inputs: dict, message: str) -> None:
    """The inputs cover the inputs of the forward pass, the error names them."""
    with pytest.raises(NetworkHybridizationError, match=message) as excinfo:
        _hybridization(inputs=inputs)
    assert "net1" in str(excinfo.value)


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        ({}, "no output has a target"),
        ({"net1__output0__1": "gamma"}, r"not an element of the output 0.*\(1,\)"),
        ({"net1__output1__0": "gamma"}, "is the output 1, but the forward pass has 1"),
        ({"net1__output0": "gamma"}, "names no element"),
        ({"net1__out0__0": "gamma"}, "is not the id of an output"),
        ({"net1__output0__0": ""}, "is not an id"),
        ({"net1__output0__0": "[]"}, "is not an id"),
        ({"net1__output0__0": 1.0}, "is not an id"),
    ],
)
def test_outputs_which_do_not_fit_the_network(outputs: dict, message: str) -> None:
    """The outputs are elements of the outputs of the network."""
    with pytest.raises(NetworkHybridizationError, match=message) as excinfo:
        _hybridization(outputs=outputs)
    assert "net1" in str(excinfo.value)


def test_two_outputs_with_one_target() -> None:
    """An entity is set by one output."""
    network = feed_forward(n_outputs=2)
    with pytest.raises(NetworkHybridizationError, match=r"both set 'prey'"):
        Hybridization(
            network=network,
            pattern=PRE,
            model="lv",
            inputs=_inputs("alpha", "beta"),
            outputs={"net1__output0__0": "prey", "net1__output0__1": "[prey]"},
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frozen": {"net1__layer9__weight__0_0"}}, "are not elements of the network"),
        ({"constants": {"k": np.inf}}, "the constant 'k' is 'inf'"),
        ({"model": ""}, "is not an id"),
        ({"model": None}, "is not an id"),
    ],
)
def test_attributes_which_are_not_valid(kwargs: dict, message: str) -> None:
    """The error names the network and the attribute."""
    with pytest.raises(NetworkHybridizationError, match=message) as excinfo:
        _hybridization(**kwargs)
    assert "Network 'net1'" in str(excinfo.value)


def test_a_network_without_values() -> None:
    """A network is not evaluated with arrays which have no values."""
    network = feed_forward()
    parameters = {"layer1": dict(network.parameters["layer1"])}
    with pytest.raises(NetworkImportError, match=r"'layer2'.*has no values"):
        _hybridization(network=replace(network, parameters=parameters))


def test_a_network_which_is_compiled_is_evaluated_on_expressions() -> None:
    """A convolution runs before the simulation, not in the model."""
    inputs = {"net3__input0": NetworkInput(arrays={ALL_CONDITIONS: np.ones((1, 4, 4))})}
    outputs = {"net3__output0__0": "gamma"}
    hybridization = Hybridization(
        network=convolution(), pattern=PRE, model="lv", inputs=inputs, outputs=outputs
    )
    assert hybridization.output_shapes() == [(1,)]
    for pattern in (RHS, OBSERVABLE):
        with pytest.raises(
            NetworkHybridizationError, match=r"'net3'.*not evaluated on expressions"
        ):
            Hybridization(
                network=convolution(),
                pattern=pattern,
                model="lv",
                inputs=inputs,
                outputs=outputs,
            )


def test_a_network_which_is_compiled_has_one_formula_per_input() -> None:
    """A rule of a model does not depend on the condition."""
    inputs = {
        "net1__input0__0": NetworkInput(formulas={"e1": "prey", "e2": "predator"}),
        "net1__input0__1": NetworkInput(formula="predator"),
    }
    with pytest.raises(NetworkHybridizationError, match="one formula per input"):
        _hybridization(pattern=RHS, inputs=inputs)
    assert _hybridization(pattern=PRE, inputs=inputs).pattern is PRE


# --- THE DERIVED CHANGES ---


def test_the_derived_changes_are_the_forward_pass() -> None:
    """The outputs of the network at the inputs are the changes of the targets."""
    hybridization = _hybridization(pattern=PRE)
    network = hybridization.network
    changes = hybridization.derived_changes({"alpha": 1.3}, condition="e1")
    (expected,) = network.forward(np.array([1.3, 1.0]))
    assert changes == {"gamma": pytest.approx(expected[0])}
    assert hybridization.targets() == {"gamma"}
    assert hybridization.symbols() == {"alpha", "k", *network.parameter_ids()}


def test_the_derived_changes_use_the_values_of_the_elements() -> None:
    """An element which is not frozen has the value of the fit."""
    sid = "net1__layer2__bias__0"
    frozen = "net1__layer1__bias__0"
    hybridization = _hybridization(pattern=PRE, frozen={frozen})
    assert sid in hybridization.symbols()
    assert frozen not in hybridization.symbols()
    nominal = hybridization.derived_changes({"alpha": 1.3}, condition="e1")["gamma"]
    shifted = hybridization.derived_changes(
        {
            "alpha": 1.3,
            sid: float(hybridization.network.parameters["layer2"]["bias"][0]) + 2.0,
        },
        condition="e1",
    )["gamma"]
    assert shifted == pytest.approx(nominal + 2.0)
    # a value of a frozen element is not read
    ignored = hybridization.derived_changes(
        {"alpha": 1.3, frozen: 100.0}, condition="e1"
    )["gamma"]
    assert ignored == nominal
    # the network keeps its nominal values
    assert hybridization.derived_changes({"alpha": 1.3}, "e1")["gamma"] == nominal


def test_the_values_have_precedence_over_the_constants() -> None:
    """A constant is the value of a symbol which the fit does not give."""
    hybridization = _hybridization(pattern=PRE)
    with_constant = hybridization.input_values({"alpha": 1.3}, condition="e1")
    np.testing.assert_allclose(with_constant[0], [1.3, 1.0])
    with_value = hybridization.input_values({"alpha": 1.3, "k": 2.0}, condition="e1")
    np.testing.assert_allclose(with_value[0], [1.3, 4.0])


def test_the_inputs_of_a_condition() -> None:
    """Formulas and arrays are selected by the condition of the simulation."""
    network = two_inputs()
    hybridization = Hybridization(
        network=network,
        pattern=PRE,
        model="lv",
        inputs={
            "net6__input0__0": NetworkInput(formulas={"e1": "10.0", "e2": "alpha"}),
            "net6__input1": NetworkInput(
                arrays={"e1": [1.0, 2.0, 3.0], "e2": [3.0, 2.0, 1.0]}
            ),
        },
        outputs={"net6__output0__0": "gamma"},
    )
    first, second = hybridization.input_values({"alpha": 1.3}, condition="e1")
    np.testing.assert_allclose(first, [10.0])
    np.testing.assert_allclose(second, [1.0, 2.0, 3.0])
    first, second = hybridization.input_values({"alpha": 1.3}, condition="e2")
    np.testing.assert_allclose(first, [1.3])
    np.testing.assert_allclose(second, [3.0, 2.0, 1.0])
    e1 = hybridization.derived_changes({"alpha": 1.3}, condition="e1")["gamma"]
    e2 = hybridization.derived_changes({"alpha": 1.3}, condition="e2")["gamma"]
    assert e1 == pytest.approx(network.forward(first * 0 + 10.0, second[::-1])[0][0])
    assert e1 != e2

    with pytest.raises(
        NetworkHybridizationError,
        match=r"'net6'.*'net6__input0__0' has no formula for the condition 'e3'",
    ):
        hybridization.derived_changes({"alpha": 1.3}, condition="e3")


def test_an_array_without_the_condition() -> None:
    """An input without an array for a condition names the condition."""
    hybridization = Hybridization(
        network=two_inputs(),
        pattern=PRE,
        model="lv",
        inputs={
            "net6__input0__0": NetworkInput(formula="alpha"),
            "net6__input1": NetworkInput(arrays={"e1": [1.0, 2.0, 3.0]}),
        },
        outputs={"net6__output0__0": "gamma"},
    )
    with pytest.raises(
        NetworkHybridizationError,
        match=r"'net6__input1' has no array for the condition 'e2'.*\['e1'\]",
    ):
        hybridization.derived_changes({"alpha": 1.3}, condition="e2")


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({}, r"input 'net1__input0__0'.*uses \['alpha'\], which have no value"),
        ({"alpha": np.nan}, "which is not a finite number"),
        ({"alpha": np.array([1.0, 2.0])}, "which is not a finite number"),
    ],
)
def test_an_input_without_a_number(values: dict, message: str) -> None:
    """A symbol without a value and a value which is no number are errors."""
    with pytest.raises(NetworkHybridizationError, match=message) as excinfo:
        _hybridization(pattern=PRE).derived_changes(values, condition="e1")
    assert "Network 'net1'" in str(excinfo.value)


def test_an_output_which_is_not_finite() -> None:
    """An output which overflows is an error and not a change of the model."""
    hybridization = _hybridization(pattern=PRE, inputs=_inputs("alpha", "exp(alpha)"))
    with (
        pytest.raises(NetworkHybridizationError, match="input 'net1__input0__1'"),
        np.errstate(over="ignore"),
    ):
        hybridization.derived_changes({"alpha": 1e6}, condition="e1")


def test_the_derived_changes_of_a_compiled_network() -> None:
    """The model evaluates the network, the fit sets the arrays of a condition."""
    assert _hybridization(pattern=RHS).symbols() == frozenset()
    assert _hybridization(pattern=RHS).targets() == frozenset()
    assert _hybridization(pattern=RHS).derived_changes({}, "e1") == {}

    def hybridization(arrays: dict) -> Hybridization:
        return Hybridization(
            network=two_inputs(),
            pattern=RHS,
            model="lv",
            inputs={
                "net6__input0__0": NetworkInput(formula="prey"),
                "net6__input1": NetworkInput(arrays=arrays),
            },
            outputs={"net6__output0__0": "gamma"},
        )

    # one array for every condition is a part of the model
    constant = hybridization({ALL_CONDITIONS: [1.0, 2.0, 3.0]})
    assert constant.targets() == frozenset()
    assert constant.derived_changes({}, "e1") == {}

    conditional = hybridization({"e1": [1.0, 2.0, 3.0], "e2": [3.0, 2.0, 1.0]})
    ids = ["net6__input1__0", "net6__input1__1", "net6__input1__2"]
    assert conditional.targets() == set(ids)
    assert conditional.derived_changes({}, "e2") == dict(
        zip(ids, [3.0, 2.0, 1.0], strict=True)
    )
    with pytest.raises(NetworkHybridizationError, match="no array for the condition"):
        conditional.derived_changes({}, "e3")


def test_the_parameters_of_a_fit_are_checked() -> None:
    """A fit does not write what the network sets or holds constant."""
    frozen = "net1__layer1__bias__0"
    hybridization = _hybridization(pattern=PRE, frozen={frozen})
    hybridization.check_parameters(["alpha", "beta", "net1__layer1__bias__1"])
    with pytest.raises(NetworkHybridizationError, match=rf"\['{frozen}'\] are frozen"):
        hybridization.check_parameters(["alpha", frozen])
    with pytest.raises(NetworkHybridizationError, match=r"\['gamma'\] are set by"):
        hybridization.check_parameters(["alpha", "gamma"])
    with pytest.raises(NetworkHybridizationError, match=r"\['net1__output0__0'\]"):
        hybridization.check_parameters(["net1__output0__0"])


def test_a_hybridization_is_pickled() -> None:
    """A fit pickles its hybridizations for the workers, the copy is equal."""
    hybridization = Hybridization(
        network=two_inputs(),
        pattern=PRE,
        model="lv",
        inputs={
            "net6__input0__0": NetworkInput(formula="alpha"),
            "net6__input1": NetworkInput(arrays={"e1": [1.0, 2.0, 3.0]}),
        },
        outputs={"net6__output0__0": "gamma"},
        frozen={"net6__layer1__bias__0"},
        constants={"k": 1.0},
    )
    copy = pickle.loads(pickle.dumps(hybridization))
    assert copy == hybridization
    assert copy.derived_changes({"alpha": 1.3}, "e1") == hybridization.derived_changes(
        {"alpha": 1.3}, "e1"
    )
    assert copy != _hybridization()


# --- THE HYBRIDIZATION AND ITS MODEL ---


def test_a_hybridization_fits_its_model(model_path: Path) -> None:
    """The three patterns are valid for the model."""
    _hybridization(pattern=RHS).validate(model_path)
    _hybridization(pattern=PRE).validate(model_path)
    _hybridization(pattern=PRE, outputs={"net1__output0__0": "[prey]"}).validate(
        model_path
    )
    _hybridization(pattern=OBSERVABLE, outputs={"net1__output0__0": "y"}).validate(
        model_path
    )


@pytest.mark.parametrize(
    ("pattern", "kwargs", "message"),
    [
        (RHS, {"outputs": {"net1__output0__0": "kappa"}}, "'kappa'.*is not an entity"),
        (PRE, {"outputs": {"net1__output0__0": "kappa"}}, "'kappa'.*is not an entity"),
        (
            OBSERVABLE,
            {"outputs": {"net1__output0__0": "gamma"}},
            "the model 'lv.xml' has an entity 'gamma'",
        ),
        (
            OBSERVABLE,
            {"outputs": {"net1__output0__0": "[y]"}},
            "is the symbol of an observable and not a concentration",
        ),
        (RHS, {"outputs": {"net1__output0__0": "[prey]"}}, "is a concentration"),
        (PRE, {"outputs": {"net1__output0__0": "[gamma]"}}, "is a concentration"),
        (PRE, {"constants": {"k": 0.5, "alpha": 1.0}}, "'alpha' is an entity"),
        (RHS, {"inputs": _inputs("prey", "gamma")}, r"uses \['gamma'\], which the out"),
        (PRE, {"inputs": _inputs("alpha", "prey")}, "'prey', which is a species"),
        (PRE, {"inputs": _inputs("alpha", "time")}, "'time', which is the time"),
    ],
)
def test_a_hybridization_which_does_not_fit_its_model(
    model_path: Path, pattern: NetworkPattern, kwargs: dict, message: str
) -> None:
    """The error names the network and the target or the input."""
    hybridization = _hybridization(pattern=pattern, **kwargs)
    with pytest.raises(NetworkHybridizationError, match=message) as excinfo:
        hybridization.validate(model_path)
    assert "Network 'net1'" in str(excinfo.value)


def test_an_input_which_a_rule_sets(tmp_path: Path, model_path: Path) -> None:
    """An entity with a rule is not a constant of a simulation."""
    document = libsbml.readSBMLFromFile(str(model_path))
    model = document.getModel()
    model.getParameter("alpha").setConstant(False)
    rule = model.createAssignmentRule()
    rule.setVariable("alpha")
    rule.setMath(libsbml.parseL3Formula("2 * prey"))
    path = tmp_path / "rule.xml"
    libsbml.writeSBMLToFile(document, str(path))
    with pytest.raises(
        NetworkHybridizationError, match="'alpha', which is set by a rule"
    ):
        _hybridization(pattern=PRE).validate(path)


def test_a_model_which_cannot_be_read(tmp_path: Path) -> None:
    """A file which is missing or no model names the network and the file."""
    with pytest.raises(NetworkHybridizationError, match=r"'net1'.*does not exist"):
        _hybridization().validate(tmp_path / "missing.xml")
    path = tmp_path / "empty.xml"
    path.write_text("<sbml/>")
    with pytest.raises(NetworkHybridizationError, match=r"'net1'.*holds no SBML model"):
        _hybridization().validate(path)
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest -q -x tests/sciml/test_hybridization.py`
Expected: FAIL at collection: `ModuleNotFoundError: No module named 'sbmlsim.sciml.hybridization'`

- [ ] **Step 5: Apply this patch to `src/sbmlsim/sciml/errors.py`**

Apply this patch to `src/sbmlsim/sciml/errors.py`:

```diff
diff --git a/src/sbmlsim/sciml/errors.py b/src/sbmlsim/sciml/errors.py
index af20e1d..0a38f9b 100644
--- a/src/sbmlsim/sciml/errors.py
+++ b/src/sbmlsim/sciml/errors.py
@@ -45,3 +45,11 @@ class UnsupportedLayerError(NetworkError, NotImplementedError):
         super().__init__(
             f"Network '{network}', node '{node}': '{target}' is not supported, {reason}"
         )
+
+
+class NetworkHybridizationError(NetworkError, ValueError):
+    """A hybridization does not fit its network, its model or its fit.
+
+    The message names the network and, where it applies, the input, the
+    output or the target.
+    """
```

- [ ] **Step 6: Create `src/sbmlsim/sciml/hybridization.py`**

Create `src/sbmlsim/sciml/hybridization.py`:

```python
"""Where a network sits in a hybrid problem.

A `Hybridization` connects a `Network` to a model: what its inputs are, which
entities of the model its outputs set, and in which of three places it runs.

| pattern | inputs | outputs | executed by |
| --- | --- | --- | --- |
| `PRE_INITIALIZATION` | constants: formulas of parameters, arrays | parameters and initial values, set before the simulation | numpy, once per simulation |
| `RHS` | formulas of species, parameters and time, arrays | parameters of the rate equations | roadrunner, as assignment rules |
| `OBSERVABLE` | formulas of species, parameters and time, arrays | symbols of an observable | roadrunner, as assignment rules |

A network before the simulation is evaluated by `derived_changes`, which a
fit calls for every simulation with the values of its parameters. The
networks of the other two patterns are compiled into the model by
`sbmlsim.sciml.compiler`; `derived_changes` sets only their inputs which are
arrays of a condition.

The condition of a simulation is the id of the simulation in its experiment.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import libsbml
import numpy as np
from numpy.typing import ArrayLike

from sbmlsim.mathml import TIME, evaluate_formula, formula_symbols
from sbmlsim.sciml.backend import BackendKind
from sbmlsim.sciml.errors import NetworkHybridizationError, NetworkImportError
from sbmlsim.sciml.interpreter import PLACEHOLDER
from sbmlsim.sciml.network import (
    Network,
    input_id,
    output_id,
    parse_io_id,
)

logger = logging.getLogger(__name__)

#: the condition of the arrays and formulas which hold for every condition
ALL_CONDITIONS = "0"


class NetworkPattern(StrEnum):
    """The place of a network in a hybrid problem."""

    PRE_INITIALIZATION = "pre_initialization"
    RHS = "rhs"
    OBSERVABLE = "observable"

    @property
    def is_compiled(self) -> bool:
        """Check whether a network of the pattern is compiled into the model."""
        return self is not NetworkPattern.PRE_INITIALIZATION


@dataclass(frozen=True, eq=False)
class NetworkInput:
    """An input of a network or an element of one.

    Exactly one of the attributes is given.

    Attributes:
        formula: an L3 formula of SBML which holds for every condition, e.g.
            `prey`, `alpha * prey` or `0.5`.
        formulas: id of the condition -> formula, for an input which differs
            between the conditions. `ALL_CONDITIONS` is the formula of the
            conditions which are not listed.
        arrays: id of the condition -> values, `ALL_CONDITIONS` for the values
            of the conditions which are not listed. The arrays of an input
            have one shape.
    """

    formula: str | None = None
    formulas: Mapping[str, str] | None = None
    arrays: Mapping[str, ArrayLike] | None = None

    def __post_init__(self) -> None:
        """Validate the input and copy its arrays.

        Raises:
            ValueError: if not exactly one attribute is given, if a formula
                is not valid math, if no condition is given, or if the arrays
                differ in their shape or hold a value which is not finite.
        """
        given = [
            name
            for name in ("formula", "formulas", "arrays")
            if getattr(self, name) is not None
        ]
        if len(given) != 1:
            raise ValueError(
                f"An input is a formula, formulas of conditions or arrays of "
                f"conditions, but {given or 'nothing'} is given"
            )
        if self.formulas is not None:
            if not self.formulas:
                raise ValueError("The formulas of an input name no condition")
            object.__setattr__(self, "formulas", dict(self.formulas))
        if self.arrays is not None:
            if not self.arrays:
                raise ValueError("The arrays of an input name no condition")
            arrays = {}
            for condition, values in self.arrays.items():
                array = np.array(values, dtype=float)
                if not np.all(np.isfinite(array)):
                    raise ValueError(
                        f"The array of the condition '{condition}' holds values "
                        f"which are not finite"
                    )
                array.setflags(write=False)
                arrays[condition] = array
            shapes = {condition: a.shape for condition, a in arrays.items()}
            if len(set(shapes.values())) > 1:
                raise ValueError(
                    f"The arrays of an input have one shape, but the shapes of "
                    f"the conditions are {shapes}"
                )
            object.__setattr__(self, "arrays", arrays)
        for formula in self.all_formulas():
            formula_symbols(formula)

    def all_formulas(self) -> list[str]:
        """Get the formulas of the input, empty for an input of arrays."""
        if self.formula is not None:
            return [self.formula]
        return list((self.formulas or {}).values())

    @property
    def is_conditional(self) -> bool:
        """Check whether the input differs between the conditions."""
        values = self.formulas if self.formulas is not None else self.arrays
        return values is not None and set(values) != {ALL_CONDITIONS}

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Get the shape of the arrays, `None` for an input of formulas."""
        if self.arrays is None:
            return None
        return np.shape(next(iter(self.arrays.values())))

    def formula_of(self, condition: str) -> str | None:
        """Get the formula of a condition.

        Args:
            condition: id of the condition.

        Returns:
            The formula, `None` for an input of arrays and for a condition
            without a formula.
        """
        if self.formula is not None:
            return self.formula
        if self.formulas is None:
            return None
        return self.formulas.get(condition, self.formulas.get(ALL_CONDITIONS))

    def array_of(self, condition: str) -> np.ndarray | None:
        """Get the array of a condition.

        Args:
            condition: id of the condition.

        Returns:
            The array, `None` for an input of formulas and for a condition
            without an array.
        """
        if self.arrays is None:
            return None
        array = self.arrays.get(condition, self.arrays.get(ALL_CONDITIONS))
        return None if array is None else np.asarray(array, dtype=float)

    def __eq__(self, other: object) -> bool:
        """Check whether two inputs are equal, the arrays element by element."""
        if not isinstance(other, NetworkInput):
            return NotImplemented
        if self.formula != other.formula or self.formulas != other.formulas:
            return False
        if self.arrays is None or other.arrays is None:
            return self.arrays is None and other.arrays is None
        return self.arrays.keys() == other.arrays.keys() and all(
            np.array_equal(array, other.arrays[condition])
            for condition, array in self.arrays.items()
        )

    def __hash__(self) -> int:
        """Get the hash of the formula, equal inputs have one formula."""
        return hash(self.formula)


def input_shapes(
    network: Network, inputs: Mapping[str, NetworkInput]
) -> list[tuple[int, ...]]:
    """Get the shapes of the inputs of the forward pass of a network.

    The shape of an input which is given as an array is the shape of its
    arrays. The shape of an input which is given element by element follows
    from the indices of its elements, which have to cover it.

    Args:
        network: the network.
        inputs: id of the input -> the input, see `Hybridization`.

    Returns:
        The shape of every input, in the order of the forward pass.

    Raises:
        NetworkHybridizationError: if an id is not the id of an input, if an
            input is given as an array and element by element, if an input of
            the forward pass is missing, or if the elements of an input do
            not cover a shape.
    """
    sid = network.sid

    def error(message: str) -> NetworkHybridizationError:
        return NetworkHybridizationError(f"Network '{sid}': {message}")

    n_inputs = sum(node.op == PLACEHOLDER for node in network.model.forward)
    arrays: dict[int, tuple[int, ...]] = {}
    elements: dict[int, set[tuple[int, ...]]] = {}
    for key, network_input in inputs.items():
        if not isinstance(network_input, NetworkInput):
            raise error(f"the input '{key}' is not a `NetworkInput`")
        try:
            k, index = parse_io_id(sid, "input", key)
        except ValueError as err:
            raise NetworkHybridizationError(str(err)) from err
        if k >= n_inputs:
            raise error(
                f"'{key}' is the input {k}, but the forward pass has {n_inputs} inputs"
            )
        shape = network_input.shape
        if index is None:
            if shape is None:
                raise error(
                    f"the input '{key}' is a formula, which is the value of one "
                    f"element: name the element, e.g. '{input_id(sid, k, (0,))}'"
                )
            arrays[k] = shape
        else:
            if shape not in (None, ()):
                raise error(
                    f"the input '{key}' is an element, but its arrays have the "
                    f"shape {shape}"
                )
            elements.setdefault(k, set()).add(index)
    both = sorted(set(arrays) & set(elements))
    if both:
        raise error(f"the inputs {both} are given as an array and element by element")
    shapes: list[tuple[int, ...]] = []
    for k in range(n_inputs):
        if k in arrays:
            shapes.append(arrays[k])
            continue
        if k not in elements:
            raise error(
                f"the input {k} of the forward pass is missing, the inputs are "
                f"{sorted(inputs)}"
            )
        indices = elements[k]
        ndims = {len(index) for index in indices}
        if len(ndims) != 1:
            raise error(
                f"the elements of the input {k} differ in their number of axes: "
                f"{sorted(indices)}"
            )
        shape = tuple(
            max(index[axis] for index in indices) + 1 for axis in range(ndims.pop())
        )
        missing = sorted(set(np.ndindex(shape)) - indices)
        if missing:
            raise error(
                f"the input {k} has the shape {shape}, but its elements {missing} "
                f"are missing"
            )
        shapes.append(shape)
    return shapes


def output_shapes(
    network: Network, shapes: list[tuple[int, ...]]
) -> list[tuple[int, ...]]:
    """Get the shapes of the outputs of a network for inputs of given shapes.

    Args:
        network: the network, with the values of its arrays.
        shapes: the shape of every input, in the order of the forward pass.

    Returns:
        The shape of every output, in the order of the forward pass.

    Raises:
        NetworkHybridizationError: if the network cannot be evaluated on
            inputs of the shapes.
        NetworkImportError: if an array of the network has no values.
    """
    try:
        outputs = network.forward(*[np.zeros(shape) for shape in shapes])
    except NetworkImportError:
        raise
    except ValueError as err:
        raise NetworkHybridizationError(
            f"Network '{network.sid}': inputs of the shapes {shapes} do not fit "
            f"the network: {err}"
        ) from err
    return [output.shape for output in outputs]


def entity_of(target: str) -> str:
    """Get the entity of the model a target names.

    Args:
        target: the entity or the selection of its concentration, e.g. `prey`
            or `[prey]`.

    Returns:
        The id of the entity, e.g. `prey`.
    """
    if target.startswith("[") and target.endswith("]"):
        return target[1:-1]
    return target


@dataclass(frozen=True)
class Hybridization:
    """A network with its inputs, its outputs and its place in a problem.

    The hybridization is validated when it is created, as far as it can be
    without the model: `validate` checks it against the model.

    Attributes:
        network: the network.
        pattern: where the network sits.
        model: id of the model in the experiment.
        inputs: id of the input -> the input. The id is `<net>__input<k>` for
            an input which is given as an array and
            `<net>__input<k>__<index>` for an element of an input which is
            given element by element, see `sbmlsim.sciml.network.input_id`.
        outputs: id of the element of an output -> target, see
            `sbmlsim.sciml.network.output_id`. The target is an entity of the
            model for `RHS`, an entity or the selection of its concentration
            (`[prey]`) for `PRE_INITIALIZATION`, and the symbol an observable
            uses for `OBSERVABLE`, which the compiler adds to the model. An
            output without a target is not used.
        frozen: ids of the elements of the arrays which are not estimated.
        constants: id -> value of the symbols of the formulas which are
            neither entities of the model nor parameters of the fit.
    """

    network: Network
    pattern: NetworkPattern
    model: str
    inputs: Mapping[str, NetworkInput]
    outputs: Mapping[str, str]
    frozen: Collection[str] = field(default_factory=frozenset)
    constants: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the hybridization against its network.

        Raises:
            NetworkHybridizationError: if the pattern is not a pattern, if an
                id is not the id of an input or output of the network, if
                the inputs do not cover the inputs of the forward pass, if
                the outputs are not outputs of the network, if two outputs
                have one target, if an element which is frozen is not an
                element of the network, if a constant is not finite, or if a
                network which is compiled has a layer without expressions or
                an input which differs in its formula between conditions.
            NetworkImportError: if an array of the network has no values.
        """
        object.__setattr__(self, "pattern", self._error(NetworkPattern, self.pattern))
        object.__setattr__(self, "inputs", dict(self.inputs))
        object.__setattr__(self, "outputs", dict(self.outputs))
        object.__setattr__(self, "frozen", frozenset(self.frozen))
        object.__setattr__(
            self,
            "constants",
            {key: float(value) for key, value in self.constants.items()},
        )
        if not isinstance(self.model, str) or not self.model:
            raise self.error(f"the id of the model '{self.model}' is not an id")
        for sid, value in self.constants.items():
            if not np.isfinite(value):
                raise self.error(f"the constant '{sid}' is '{value}', not a number")
        unknown = sorted(set(self.frozen) - set(self.network.parameter_ids()))
        if unknown:
            raise self.error(
                f"the frozen elements {unknown} are not elements of the network"
            )
        self.network.check_arrays(self.network.parameters, complete=True)

        shapes = self.input_shapes()
        if self.pattern.is_compiled:
            if BackendKind.SYMPY not in self.network.backends():
                raise self.error(
                    f"the pattern '{self.pattern.value}' compiles the network into "
                    f"the model, but it has layers or functions which are not "
                    f"evaluated on expressions, e.g. a convolution. Such a "
                    f"network runs before the simulation, i.e. with the pattern "
                    f"'{NetworkPattern.PRE_INITIALIZATION.value}'"
                )
            for sid, network_input in self.inputs.items():
                if network_input.formulas is not None:
                    raise self.error(
                        f"the input '{sid}' has the formulas of the conditions "
                        f"{sorted(network_input.formulas)}, but a network of the "
                        f"pattern '{self.pattern.value}' is a part of the model "
                        f"and has one formula per input"
                    )
        self._check_outputs(shapes)

    def _error(self, enum: type[StrEnum], value: Any) -> Any:
        """Get the member of an enumeration, as an error of the hybridization."""
        try:
            return enum(value)
        except ValueError as err:
            raise self.error(
                f"'{value}' is not one of {[member.value for member in enum]}"
            ) from err

    def error(self, message: str) -> NetworkHybridizationError:
        """Get an error which names the network.

        Args:
            message: what is wrong.

        Returns:
            The error, to be raised.
        """
        return NetworkHybridizationError(f"Network '{self.network.sid}': {message}")

    def input_shapes(self) -> list[tuple[int, ...]]:
        """Get the shapes of the inputs of the forward pass, see `input_shapes`."""
        return input_shapes(self.network, self.inputs)

    def output_shapes(self) -> list[tuple[int, ...]]:
        """Get the shapes of the outputs of the network for its inputs."""
        return output_shapes(self.network, self.input_shapes())

    def _check_outputs(self, input_shapes: list[tuple[int, ...]]) -> None:
        """Check the outputs against the network, see `__post_init__`."""
        sid = self.network.sid
        if not self.outputs:
            raise self.error("no output has a target, the network is not used")
        shapes = self.output_shapes()
        targets: dict[str, str] = {}
        for key, target in self.outputs.items():
            try:
                k, index = parse_io_id(sid, "output", key)
            except ValueError as err:
                raise NetworkHybridizationError(str(err)) from err
            if index is None:
                raise self.error(
                    f"the output '{key}' names no element, e.g. "
                    f"'{output_id(sid, k, (0,))}'"
                )
            if k >= len(shapes):
                raise self.error(
                    f"'{key}' is the output {k}, but the forward pass has "
                    f"{len(shapes)} outputs"
                )
            if index not in set(np.ndindex(shapes[k])):
                raise self.error(
                    f"'{key}' is not an element of the output {k}, which has the "
                    f"shape {shapes[k]} for inputs of the shapes {input_shapes}"
                )
            if not isinstance(target, str) or not entity_of(target):
                raise self.error(f"the target '{target}' of '{key}' is not an id")
            entity = entity_of(target)
            if entity in targets:
                raise self.error(
                    f"the outputs '{targets[entity]}' and '{key}' both set '{entity}'"
                )
            targets[entity] = key

    # --- WHAT A FIT NEEDS ---

    def symbols(self) -> frozenset[str]:
        """Get the ids whose values `derived_changes` reads.

        Returns:
            The symbols of the formulas of the inputs and the ids of the
            elements which are not frozen, for a network before the
            simulation. A network which is compiled reads nothing, the model
            evaluates it.
        """
        if self.pattern.is_compiled:
            return frozenset()
        symbols = {
            symbol
            for network_input in self.inputs.values()
            for formula in network_input.all_formulas()
            for symbol in formula_symbols(formula)
        }
        elements = set(self.network.parameter_ids()) - set(self.frozen)
        return frozenset(symbols | elements)

    def targets(self) -> frozenset[str]:
        """Get the entities of the model `derived_changes` sets.

        Returns:
            The targets of the outputs for a network before the simulation,
            and the ids of the elements of the inputs which are arrays of
            conditions for a network which is compiled.
        """
        if not self.pattern.is_compiled:
            return frozenset(self.outputs.values())
        return frozenset(sid for sid, _ in self._conditional_elements(condition=None))

    def _conditional_elements(
        self, condition: str | None
    ) -> list[tuple[str, float | None]]:
        """Get the elements of the inputs which are arrays of conditions.

        Args:
            condition: id of the condition, `None` for the ids alone.

        Returns:
            The id of the element and its value in the condition, for every
            input which differs between the conditions.

        Raises:
            NetworkHybridizationError: if an input has no array for the
                condition.
        """
        elements: list[tuple[str, float | None]] = []
        for key, network_input in self.inputs.items():
            if network_input.arrays is None or not network_input.is_conditional:
                continue
            k, index = parse_io_id(self.network.sid, "input", key)
            array = None if condition is None else network_input.array_of(condition)
            if condition is not None and array is None:
                raise self.error(
                    f"the input '{key}' has no array for the condition "
                    f"'{condition}', it has arrays for "
                    f"{sorted(network_input.arrays)}"
                )
            shape = network_input.shape or ()
            for element in np.ndindex(shape):
                sid = (
                    key if index is not None else input_id(self.network.sid, k, element)
                )
                elements.append((sid, None if array is None else float(array[element])))
        return elements

    def check_parameters(self, targets: Collection[str]) -> None:
        """Check the targets of the parameters of a fit against the network.

        Args:
            targets: the entities the parameters of the fit write, without
                the prefix of a target which is not an entity of the model.

        Raises:
            NetworkHybridizationError: if a parameter writes an element which
                is frozen, an output or its target, or an element of an input.
        """
        frozen = sorted(set(targets) & set(self.frozen))
        if frozen:
            raise self.error(
                f"the elements {frozen} are frozen, but parameters of the fit "
                f"write them. An element is frozen or estimated"
            )
        outputs = {entity_of(target) for target in self.outputs.values()}
        outputs |= set(self.outputs)
        written = sorted({entity_of(target) for target in targets} & outputs)
        if written:
            raise self.error(
                f"{written} are set by the outputs of the network, but "
                f"parameters of the fit write them. An entity is estimated or "
                f"calculated by the network"
            )

    def input_values(
        self, values: Mapping[str, float], condition: str
    ) -> list[np.ndarray]:
        """Get the inputs of the network for the values of a simulation.

        Args:
            values: id -> value of the parameters of the fit, of the changes
                of the simulation and of the entities of the model, in the
                units of the model. The constants of the hybridization are
                the values of the symbols which are not part of it.
            condition: id of the condition of the simulation.

        Returns:
            The inputs, one array per input of the forward pass.

        Raises:
            NetworkHybridizationError: if an input has no formula or array
                for the condition, if a symbol of a formula has no value, or
                if the value of a formula is not a finite number.
        """
        variables = {**self.constants, **values}
        sid = self.network.sid
        inputs = [np.zeros(shape) for shape in self.input_shapes()]
        for key, network_input in self.inputs.items():
            k, index = parse_io_id(sid, "input", key)
            if network_input.arrays is not None:
                array = network_input.array_of(condition)
                if array is None:
                    raise self.error(
                        f"the input '{key}' has no array for the condition "
                        f"'{condition}', it has arrays for "
                        f"{sorted(network_input.arrays)}"
                    )
                if index is None:
                    inputs[k] = np.array(array, dtype=float)
                else:
                    inputs[k][index] = float(array)
                continue
            formula = network_input.formula_of(condition)
            if formula is None:
                raise self.error(
                    f"the input '{key}' has no formula for the condition "
                    f"'{condition}', it has formulas for "
                    f"{sorted(network_input.formulas or {})}"
                )
            try:
                value = np.asarray(evaluate_formula(formula, variables), dtype=float)
            except (TypeError, ValueError) as err:
                raise self.error(f"input '{key}': {err}") from err
            if value.shape != () or not np.isfinite(value):
                raise self.error(
                    f"input '{key}': the formula '{formula}' has the value "
                    f"'{value}', which is not a finite number"
                )
            if index is None:
                raise self.error(f"the input '{key}' is a formula without an element")
            inputs[k][index] = float(value)
        return inputs

    def derived_changes(
        self, values: Mapping[str, float], condition: str
    ) -> dict[str, float]:
        """Get the changes of a simulation which follow from its values.

        A network before the simulation is evaluated: its inputs are resolved
        with `input_values`, its arrays are the nominal values with the
        values of the elements which are not frozen, and its outputs are the
        changes of their targets. For a network which is compiled the changes
        are the elements of the inputs which are arrays of conditions.

        Args:
            values: id -> value, see `input_values`. The values of the
                elements of the network are read from it by their id.
            condition: id of the condition of the simulation.

        Returns:
            target -> value, in the unit of the target in the model.

        Raises:
            NetworkHybridizationError: if an input cannot be resolved, see
                `input_values`, or if an output is not a finite number.
        """
        if self.pattern.is_compiled:
            return {
                sid: float(value)
                for sid, value in self._conditional_elements(condition)
                if value is not None
            }
        ids = self.network.parameter_ids()
        elements = {
            sid: float(values[sid])
            for sid in ids
            if sid in values and sid not in self.frozen
        }
        outputs = self.network.forward(
            *self.input_values(values, condition),
            parameters=self.network.with_values(elements),
        )
        changes: dict[str, float] = {}
        for key, target in self.outputs.items():
            k, index = parse_io_id(self.network.sid, "output", key)
            value = float(outputs[k][index])
            if not np.isfinite(value):
                raise self.error(
                    f"the output '{key}' is '{value}' in the condition "
                    f"'{condition}', which is not a finite number"
                )
            changes[target] = value
        return changes

    # --- THE MODEL ---

    def validate(self, sbml_path: Path) -> None:
        """Check the hybridization against the model.

        Args:
            sbml_path: the SBML model the network is a part of, without the
                network.

        Raises:
            NetworkHybridizationError: if the model cannot be read, if a
                target of `RHS` or `PRE_INITIALIZATION` is not an entity of
                the model, if a target of `OBSERVABLE` or a constant is one,
                if a formula uses a symbol which is an output of the
                network, or if an input of `PRE_INITIALIZATION` depends on
                the time, on a species or on an entity which a rule sets.
        """
        _, model = read_model(sbml_path, self.network.sid)
        for key, target in self.outputs.items():
            entity = entity_of(target)
            exists = model.getElementBySId(entity) is not None
            if self.pattern is NetworkPattern.OBSERVABLE:
                if target != entity:
                    raise self.error(
                        f"the target '{target}' of '{key}' is the symbol of an "
                        f"observable and not a concentration"
                    )
                if exists:
                    raise self.error(
                        f"the target '{target}' of '{key}' is the symbol of an "
                        f"observable, which is added to the model, but the "
                        f"model '{sbml_path.name}' has an entity '{entity}'"
                    )
            elif not exists:
                raise self.error(
                    f"the target '{target}' of '{key}' is not an entity of the "
                    f"model '{sbml_path.name}'"
                )
            elif target != entity and (
                self.pattern is NetworkPattern.RHS or model.getSpecies(entity) is None
            ):
                raise self.error(
                    f"the target '{target}' of '{key}' is a concentration, "
                    f"which is the initial value of a species for the pattern "
                    f"'{NetworkPattern.PRE_INITIALIZATION.value}'"
                )
        for sid in self.constants:
            if model.getElementBySId(sid) is not None:
                raise self.error(
                    f"the constant '{sid}' is an entity of the model "
                    f"'{sbml_path.name}', the model gives its value"
                )
        outputs = {entity_of(target) for target in self.outputs.values()}
        for key, network_input in self.inputs.items():
            for formula in network_input.all_formulas():
                symbols = formula_symbols(formula)
                circular = sorted(symbols & outputs)
                if circular:
                    raise self.error(
                        f"the input '{key}' uses {circular}, which the outputs "
                        f"of the network set"
                    )
                if self.pattern is not NetworkPattern.PRE_INITIALIZATION:
                    continue
                for symbol in sorted(symbols):
                    reason = _varies(model, symbol)
                    if reason is not None:
                        raise self.error(
                            f"the input '{key}' of a network which runs before "
                            f"the simulation is a constant, but its formula "
                            f"'{formula}' uses '{symbol}', which is {reason}"
                        )


def read_model(
    sbml_path: Path, network: str
) -> tuple[libsbml.SBMLDocument, libsbml.Model]:
    """Read the model of an SBML file.

    Args:
        sbml_path: the SBML file.
        network: id of the network, for the message.

    Returns:
        The document and its model. The model is a part of the document and
        is valid as long as the document is referenced.

    Raises:
        NetworkHybridizationError: if the file does not exist or holds no
            model.
    """
    if not Path(sbml_path).is_file():
        raise NetworkHybridizationError(
            f"Network '{network}': the model '{sbml_path}' does not exist"
        )
    document: libsbml.SBMLDocument = libsbml.readSBMLFromFile(str(sbml_path))
    model: libsbml.Model | None = document.getModel()
    if model is None:
        raise NetworkHybridizationError(
            f"Network '{network}': '{sbml_path}' holds no SBML model"
        )
    return document, model


def _varies(model: libsbml.Model, symbol: str) -> str | None:
    """Get why the value of a symbol is not a constant of a simulation.

    Args:
        model: the model.
        symbol: a symbol of a formula.

    Returns:
        The reason, `None` for a symbol which is a constant.
    """
    if symbol == TIME:
        return "the time"
    if model.getSpecies(symbol) is not None:
        return "a species"
    if model.getRuleByVariable(symbol) is not None:
        return "set by a rule of the model"
    return None
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/sciml tests/fit/test_derived_changes.py`
Expected: PASS

- [ ] **Step 8: Run the checks**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: `All checks passed!`, `... files already formatted`, `All checks passed!` (zero diagnostics)

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "sciml: the hybridization of a network and a model"
```

### Task 8: The gradient is a difference of three or five points which keeps its step next to a bound

**Files:**
- Modify: `src/sbmlsim/fit/petab_v2/likelihood.py`
- Test: `tests/fit/test_petab_v2_likelihood.py`

**Interfaces:**
- Consumes: `log_likelihood`, `nominal_parameters`, `_check_problem` of `likelihood.py`.
- Produces: `GRADIENT_ORDERS = (2, 4)`, `stencil(value, h, lower_bound, upper_bound, order=2) -> list[tuple[float, float]]`, `gradient(problem, parameters=None, step=1e-6, order=2) -> pd.Series`.

Two findings of the prototype shape the gradient. The open minor of phase 2: the central branch shrank `h` to the distance to the nearer bound, so a value just inside a bound divided the error of a simulation by a vanishing step. `stencil` now gives the points and the weights of a difference which keeps its step: five points (`order=4`) with room `2h` on both sides, three points with room `h`, the one sided difference of three points with the full step towards the side which has room, and the secant of the interval when the bounds are closer than the steps. And the truncation error: with the three point difference at `step=1e-6` the gradient of the elements of the network of case 001 misses the reference by `0.89` against `tol_grad = 0.1`, the error falls a hundredfold with a tenfold smaller step, i.e. it is the `h^2` term with a very large third derivative of the log-likelihood of an oscillating model. The reference values of the suite are differences of five points (`FiniteDifferences.central_fdm(5, 1)`); `gradient(problem, order=4)` is the same scheme at four simulations per parameter and its error is flat between the steps `1e-5` and `1e-7`. The default stays the three point difference of the spec.

- [ ] **Step 1: Apply this patch to `tests/fit/test_petab_v2_likelihood.py`**

Apply this patch to `tests/fit/test_petab_v2_likelihood.py`:

```diff
diff --git a/tests/fit/test_petab_v2_likelihood.py b/tests/fit/test_petab_v2_likelihood.py
index de00953..05ad10d 100644
--- a/tests/fit/test_petab_v2_likelihood.py
+++ b/tests/fit/test_petab_v2_likelihood.py
@@ -2,6 +2,7 @@

 import logging
 from pathlib import Path
+from typing import Any

 import numpy as np
 import pandas as pd
@@ -21,6 +22,7 @@ from sbmlsim.fit.petab_v2.likelihood import (
     log_likelihood,
     noise_values,
     nominal_parameters,
+    stencil,
 )

 #: simulations and measurements of the case `sciml_problem_import/001` of the
@@ -521,7 +523,9 @@ def test_gradient_stays_inside_the_bounds(
     """A parameter smaller than the step is not simulated below its bound.

     `KI__HCTZEX_k` has the lower bound `1e-10`, the step `1e-6` of the
-    central difference would take it to a negative value.
+    central difference would take it to a negative value. The difference is
+    the forward one with the full step: a step which is shrunk to the distance
+    to the bound divides the error of the simulation by `1e-8`.
     """
     problem = op_unit_noise
     pid = "KI__HCTZEX_k"
@@ -539,8 +543,8 @@ def test_gradient_stays_inside_the_bounds(
         for x in evaluated
         for j, p in enumerate(problem.parameters)
     )
-    # the central difference with the step shrunk to the distance to the bound
-    assert min(x[k] for x in evaluated) == pytest.approx(parameter.lower_bound)
+    points = sorted({float(x[k]) for x in evaluated})
+    assert points == pytest.approx([1e-8, 1e-8 + 1e-6, 1e-8 + 2e-6], rel=1e-12)


 def test_gradient_at_a_bound_is_one_sided(
@@ -603,3 +607,76 @@ def test_gradient_requires_the_parameters_inside_their_bounds(
     outside = ParameterSet(sid="outside", values={**nominal.values, pid: 0.0})
     with pytest.raises(ValueError, match=rf"{pid}.*bounds"):
         gradient(problem, outside)
+
+
+# --- THE STENCIL OF A DIFFERENCE ---
+
+
+def _derivative(points: list[tuple[float, float]], f: Any) -> float:
+    return sum(weight * f(point) for point, weight in points)
+
+
+@pytest.mark.parametrize("order", [2, 4])
+def test_the_central_difference(order: int) -> None:
+    """A parameter with room on both sides has the central difference."""
+    points = stencil(1.0, 0.1, 0.0, 2.0, order=order)
+    assert len(points) == {2: 2, 4: 4}[order]
+    assert min(p for p, _ in points) == pytest.approx(1.0 - order / 2 * 0.1)
+    assert max(p for p, _ in points) == pytest.approx(1.0 + order / 2 * 0.1)
+    # exact for a polynomial of the order
+    assert _derivative(points, lambda x: x**order) == pytest.approx(order * 1.0)
+    assert _derivative(points, lambda x: 3.0 * x + 1.0) == pytest.approx(3.0)
+
+
+def test_the_difference_next_to_a_bound_keeps_its_step() -> None:
+    """A bound closer than the step gives a one sided difference of full step."""
+    forward = stencil(0.01, 0.1, 0.0, 2.0)
+    assert [p for p, _ in forward] == pytest.approx([0.01, 0.11, 0.21])
+    assert _derivative(forward, lambda x: x**2) == pytest.approx(0.02)
+    backward = stencil(1.99, 0.1, 0.0, 2.0, order=4)
+    assert [p for p, _ in backward] == pytest.approx([1.99, 1.89, 1.79])
+    assert _derivative(backward, lambda x: x**2) == pytest.approx(3.98)
+    # at the bound
+    assert [p for p, _ in stencil(0.0, 0.1, 0.0, 2.0)] == pytest.approx([0.0, 0.1, 0.2])
+    # room for the central difference of three points, not of five
+    assert len(stencil(0.15, 0.1, 0.0, 2.0, order=4)) == 2
+
+
+def test_the_difference_of_bounds_closer_than_the_steps() -> None:
+    """Bounds closer than the steps of a difference give the secant of the interval."""
+    points = stencil(0.5, 1.0, 0.0, 1.0)
+    assert [p for p, _ in points] == [0.0, 1.0]
+    assert _derivative(points, lambda x: x**2) == pytest.approx(1.0)
+
+
+@pytest.mark.parametrize(
+    ("kwargs", "message"),
+    [
+        ({"order": 3}, r"order.*\(2, 4\)"),
+        ({"h": 0.0}, "step of the difference must be positive"),
+        ({"value": 3.0}, "inside the bounds"),
+        ({"value": 1.0, "lower_bound": 1.0, "upper_bound": 1.0}, "room for a step"),
+    ],
+)
+def test_a_stencil_which_does_not_exist(kwargs: dict, message: str) -> None:
+    """The order, the step and the bounds of a difference are checked."""
+    arguments: dict[str, Any] = {
+        "value": 1.0,
+        "h": 0.1,
+        "lower_bound": 0.0,
+        "upper_bound": 2.0,
+    }
+    arguments.update(kwargs)
+    with pytest.raises(ValueError, match=message):
+        stencil(**arguments)
+
+
+def test_the_gradient_of_the_fourth_order(op_unit_noise: OptimizationProblem) -> None:
+    """The five point difference agrees with the three point one."""
+    problem = op_unit_noise
+    second = gradient(problem)
+    fourth = gradient(problem, order=4)
+    assert list(fourth.index) == problem.pids
+    np.testing.assert_allclose(fourth.to_numpy(), second.to_numpy(), rtol=1e-3)
+    with pytest.raises(ValueError, match=r"order of the gradient is one of \(2, 4\)"):
+        gradient(problem, order=3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -x tests/fit/test_petab_v2_likelihood.py`
Expected: FAIL at collection: `ImportError: cannot import name 'stencil' from 'sbmlsim.fit.petab_v2.likelihood'`

- [ ] **Step 3: Apply this patch to `src/sbmlsim/fit/petab_v2/likelihood.py`**

Apply this patch to `src/sbmlsim/fit/petab_v2/likelihood.py`:

```diff
diff --git a/src/sbmlsim/fit/petab_v2/likelihood.py b/src/sbmlsim/fit/petab_v2/likelihood.py
index b32f9e3..1668968 100644
--- a/src/sbmlsim/fit/petab_v2/likelihood.py
+++ b/src/sbmlsim/fit/petab_v2/likelihood.py
@@ -384,42 +384,141 @@ def log_likelihood(
     return total


+#: the orders of the differences of the gradient
+GRADIENT_ORDERS: tuple[int, ...] = (2, 4)
+
+
+def stencil(
+    value: float, h: float, lower_bound: float, upper_bound: float, order: int = 2
+) -> list[tuple[float, float]]:
+    """Get the points and the weights of the difference of a derivative.
+
+    The derivative is the sum of the weights times the values of the function
+    at the points. The points stay inside the bounds and keep the step:
+
+    | room | difference | error |
+    | --- | --- | --- |
+    | `2 h` on both sides, `order=4` | central, five points | `h^4` |
+    | `h` on both sides | central, three points | `h^2` |
+    | `2 h` above | forward, three points | `h^2` |
+    | `2 h` below | backward, three points | `h^2` |
+    | less | between the bounds | the distance of the bounds |
+
+    A step which is shrunk to the distance to a bound, which is what a
+    central difference next to a bound needs, divides the error of the
+    function by a vanishing step, so the difference is one sided with the
+    full step instead.
+
+    Args:
+        value: the value of the parameter.
+        h: the step.
+        lower_bound: lower bound of the parameter.
+        upper_bound: upper bound of the parameter.
+        order: order of the central difference, `2` or `4`.
+
+    Returns:
+        The points with their weights.
+
+    Raises:
+        ValueError: if the order is not `2` or `4`, if the step is not
+            positive, or if the value is outside the bounds or the bounds are
+            equal.
+    """
+    if order not in GRADIENT_ORDERS:
+        raise ValueError(
+            f"The order of the difference is one of {GRADIENT_ORDERS}, not '{order}'."
+        )
+    if not h > 0.0:
+        raise ValueError(f"The step of the difference must be positive, not '{h}'.")
+    below = value - lower_bound
+    above = upper_bound - value
+    if not (below >= 0.0 and above >= 0.0) or (below == 0.0 and above == 0.0):
+        raise ValueError(
+            f"the difference requires the value inside the bounds "
+            f"[{lower_bound} - {upper_bound}] with room for a step, but it is "
+            f"'{value}'"
+        )
+    if order == 4 and below >= 2.0 * h and above >= 2.0 * h:
+        return [
+            (value - 2.0 * h, 1.0 / (12.0 * h)),
+            (value - h, -8.0 / (12.0 * h)),
+            (value + h, 8.0 / (12.0 * h)),
+            (value + 2.0 * h, -1.0 / (12.0 * h)),
+        ]
+    if below >= h and above >= h:
+        return [(value - h, -0.5 / h), (value + h, 0.5 / h)]
+    if above >= 2.0 * h:
+        return [
+            (value, -1.5 / h),
+            (value + h, 2.0 / h),
+            (value + 2.0 * h, -0.5 / h),
+        ]
+    if below >= 2.0 * h:
+        return [
+            (value, 1.5 / h),
+            (value - h, -2.0 / h),
+            (value - 2.0 * h, 0.5 / h),
+        ]
+    # the bounds are closer than the steps of a difference
+    distance = upper_bound - lower_bound
+    return [(lower_bound, -1.0 / distance), (upper_bound, 1.0 / distance)]
+
+
 def gradient(
     problem: OptimizationProblem,
     parameters: ParameterSet | None = None,
     step: float = DEFAULT_STEP,
+    order: int = 2,
 ) -> pd.Series:
-    """Get the gradient of the log-likelihood by central finite differences.
+    """Get the gradient of the log-likelihood by finite differences.

     The differences are taken on the linear scale, i.e. in the units of the
     model, with the step `step * max(|x|, 1)` for a parameter of the value
-    `x`. The model is not simulated outside the bounds of a parameter: a step
-    which would leave them is shrunk to the distance to the nearer bound, and
-    a parameter at a bound has the one sided difference into the bounds. A
-    difference divides the error of a simulation by the step, so the
+    `x`, and are central differences, see `stencil`. The model is not
+    simulated outside the bounds of a parameter: next to a bound the
+    difference is one sided with the full step. A parameter without bounds
+    has the central difference.
+
+    A difference divides the error of a simulation by the step, so the
     problem is initialized with `FitSettings` of tight tolerances and
     `variable_step_size=False`: with a variable step size the data is
     interpolated on the steps of the integrator, which differ between two
     simulations, and the simulations of one problem differ by `1e-6` however
     tight the tolerances are. The gradient logs a warning in this case.

+    The error of the central difference of three points grows with the
+    square of the step and the third derivative. The log-likelihood of a
+    model which oscillates is strongly curved in the parameters of a network:
+    the difference of five points, `order=4`, has an error which grows with
+    the fourth power of the step, for four simulations per parameter instead
+    of two.
+
+    The gradient is defined inside the bounds only: a parameter outside its
+    bounds raises, while `log_likelihood` evaluates the same parameters.
+
     Args:
         problem: initialized optimization problem.
         parameters: parameters to evaluate the gradient at,
             `nominal_parameters` by default.
         step: relative step of the differences.
+        order: order of the central difference, `2` or `4`.

     Returns:
         The derivative of the log-likelihood by every parameter of the fit,
         indexed by the ids of the parameters.

     Raises:
-        ValueError: if the step is not positive, if a parameter is outside
-            its bounds or its bounds are equal, or if the log-likelihood
-            cannot be calculated, see `log_likelihood`.
+        ValueError: if the step is not positive, if the order is not `2` or
+            `4`, if a parameter is outside its bounds or its bounds are
+            equal, or if the log-likelihood cannot be calculated, see
+            `log_likelihood`.
     """
     if not step > 0.0:
         raise ValueError(f"The step of the gradient must be positive, not '{step}'.")
+    if order not in GRADIENT_ORDERS:
+        raise ValueError(
+            f"The order of the gradient is one of {GRADIENT_ORDERS}, not '{order}'."
+        )
     _check_problem(problem)
     if problem.settings_initialized.variable_step_size:
         logger.warning(
@@ -440,35 +539,30 @@ def gradient(
         )

     derivatives: dict[str, float] = {}
+    # the log-likelihood at the parameters, which a one sided difference uses
+    at_parameters: float | None = None
     for k, parameter in enumerate(problem.parameters):
         pid = parameter.pid
         value = float(x[k])
-        h = step * max(abs(value), 1.0)
-        lower_bound = float(parameter.lower_bound)
-        upper_bound = float(parameter.upper_bound)
-        below = value - lower_bound
-        above = upper_bound - value
-        if not (below >= 0.0 and above >= 0.0) or (below == 0.0 and above == 0.0):
-            raise ValueError(
-                f"'{problem.opid}': the gradient requires the parameter '{pid}' "
-                f"inside its bounds [{lower_bound} - {upper_bound}] with room "
-                f"for a difference, but it has the value '{value}'."
+        try:
+            points = stencil(
+                value=value,
+                h=step * max(abs(value), 1.0),
+                lower_bound=float(parameter.lower_bound),
+                upper_bound=float(parameter.upper_bound),
+                order=order,
             )
-        if below > 0.0 and above > 0.0:
-            # central, with the step shrunk to the distance to the nearer bound
-            h = min(h, below, above)
-            lower, upper = value - h, value + h
-        elif below == 0.0:
-            # at the lower bound, forward
-            h = min(h, above)
-            lower, upper = value, value + h
-        else:
-            # at the upper bound, backward
-            h = min(h, below)
-            lower, upper = value - h, value
-        # `value - h` rounds, the bound is where the step ends
-        lower, upper = max(lower, lower_bound), min(upper, upper_bound)
-        plus = log_likelihood(problem, shifted(pid, upper))
-        minus = log_likelihood(problem, shifted(pid, lower))
-        derivatives[pid] = (plus - minus) / (upper - lower)
+        except ValueError as err:
+            raise ValueError(
+                f"'{problem.opid}': the gradient of the parameter '{pid}': {err}."
+            ) from err
+        derivative = 0.0
+        for point, weight in points:
+            if point == value:
+                if at_parameters is None:
+                    at_parameters = log_likelihood(problem, pset)
+                derivative += weight * at_parameters
+            else:
+                derivative += weight * log_likelihood(problem, shifted(pid, point))
+        derivatives[pid] = derivative
     return pd.Series(derivatives, name="gradient", dtype=float)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/fit/test_petab_v2_likelihood.py tests/fit/test_fisher.py`
Expected: PASS

- [ ] **Step 5: Run the checks**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: `All checks passed!`, `... files already formatted`, `All checks passed!` (zero diagnostics)

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "likelihood: the gradient is a difference of three or five points which keeps its step next to a bound"
```

### Task 9: The compiler, the fit parameters of a network, and a hybrid fit in python

**Files:**
- Modify: `src/sbmlsim/sciml/__init__.py`
- Create: `src/sbmlsim/sciml/compiler.py`
- Modify: `src/sbmlsim/sciml/errors.py`
- Modify: `src/sbmlsim/sciml/parameters.py`
- Test (create): `tests/sciml/experiment.py`
- Test (create): `tests/sciml/test_compiler.py`
- Test (create): `tests/sciml/test_fit.py`
- Test: `tests/sciml/test_package.py`
- Test: `tests/sciml/test_parameters.py`

**Interfaces:**
- Consumes: `Hybridization`, `NetworkInput`, `NetworkPattern`, `ALL_CONDITIONS`, `read_model` of task 7; `SympyBackend`, `unit_id`, `input_id`, `output_id`, `parse_io_id` of task 2; `expression_to_astnode`, `formula_symbols`, `TIME` of task 3; `evaluate` of the interpreter; `EXTERNAL_PREFIX`, `ParameterScaleType`.
- Produces: `sbmlsim.sciml.errors.NetworkCompilationError`; `sbmlsim.sciml.compiler.MODEL_SUFFIX = "_sciml"`, `compiled_path(sbml_path, directory=None) -> Path`, `compile_network(sbml_path, hybridizations, output_path) -> Path`; `network_fit_parameters(network, estimate, bounds, values=None, external=False)`; the exports of `sbmlsim.sciml` (`Hybridization`, `NetworkInput`, `NetworkPattern`, `compile_network`, `compiled_path`, `NetworkCompilationError`, `NetworkHybridizationError`). `tests/sciml/experiment.py`: `LotkaVolterra` (class attribute `model_path`), `DATA`, `SIMULATIONS`, `collections(experiment)`.

`compile_network` writes the networks of the patterns `RHS` and `OBSERVABLE` into a copy of the model, in the five steps of the spec: the elements of the arrays are constant parameters with their nominal values (dimensionless), an input is a parameter with the assignment rule of its formula (parsed with `parseL3FormulaWithModel`, its symbols must be entities of the model or constants of the hybridization) or constant parameters for an array, the interpreter runs with `SympyBackend` and the `on_node` hook replaces the expressions of every node by the symbols of its units, each a parameter with the assignment rule of its expression, and every element of an output is a parameter with a rule, whose target gets the rule `target = output`. A target of `RHS` is a parameter, a compartment or a species of the model which no rule, initial assignment, event or reaction sets and which becomes `constant="false"`; a target of `OBSERVABLE` is a new parameter. The hybridizations of one network (its two patterns) are merged, the ids of every part are checked against the model and against each other (`net1__layer1__bias__0` of a node named `layer1__bias` collides with the bias), the model is validated before and after, and the expressions are written through `expression_to_astnode` of task 3, which rejects `erf` naming the node. The compiled model of a `Linear`-`tanh`-`Linear` network gives the forward pass at every time point of a simulation to `1e-12` in SBML level 3 and level 2, and every activation of both backends is math of SBML. `network_fit_parameters` gains `external` for a network before the simulation (targets `sciml:<id>`), gives every element the linear scale, and leaves the elements of a layer the forward pass does not call out with a warning, as the notes of phase 1 ask. `tests/sciml/experiment.py` is a hybrid problem defined in python without PEtab, whose class lives in a module because the workers of a parallel fit import it; `test_fit.py` runs the three patterns through `OptimizationProblem`, the gradient, a pickled copy and a parallel fit of two workers.

- [ ] **Step 1: Create `tests/sciml/test_compiler.py`**

Create `tests/sciml/test_compiler.py`:

```python
"""Tests of the compilation of networks into an SBML model.

The claim of the compilation is that the model evaluates the network: the
values of the assignment rules which roadrunner calculates are the forward
pass of the network at the same inputs.
"""

from pathlib import Path
from typing import Any

import libsbml
import numpy as np
import pytest
import roadrunner

from sbmlsim.sciml import (
    Hybridization,
    Network,
    NetworkCompilationError,
    NetworkHybridizationError,
    NetworkInput,
    NetworkPattern,
    compile_network,
    compiled_path,
)
from sbmlsim.sciml.backend import BackendKind
from sbmlsim.sciml.hybridization import ALL_CONDITIONS
from sbmlsim.sciml.layers import FUNCTIONS
from tests.sciml.hybrid import feed_forward, two_inputs, write_model

PRE = NetworkPattern.PRE_INITIALIZATION
RHS = NetworkPattern.RHS
OBSERVABLE = NetworkPattern.OBSERVABLE

#: relative and absolute tolerance of the rules against the forward pass
TOLERANCE = 1e-12

#: the activations which take the input alone, with their keyword arguments
ACTIVATIONS: dict[str, dict[str, Any]] = {
    name: {}
    for name, function in FUNCTIONS.items()
    if BackendKind.SYMPY in function.backends
    and name not in {"cat", "concat", "concatenate", "flatten"}
}
ACTIVATIONS["gelu"] = {"approximate": "tanh"}
ACTIVATIONS["softmax"] = {"dim": 0}
ACTIVATIONS["log_softmax"] = {"dim": 0}


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    """Write the model of Lotka and Volterra."""
    return write_model(tmp_path / "lv.xml")


def _hybridization(
    network: Network | None = None,
    pattern: NetworkPattern = RHS,
    target: str = "gamma",
    **kwargs: Any,
) -> Hybridization:
    network = feed_forward() if network is None else network
    arguments: dict[str, Any] = {
        "network": network,
        "pattern": pattern,
        "model": "lv",
        "inputs": {
            f"{network.sid}__input0__0": NetworkInput(formula="prey"),
            f"{network.sid}__input0__1": NetworkInput(formula="predator"),
        },
        "outputs": {f"{network.sid}__output0__0": target},
    }
    arguments.update(kwargs)
    return Hybridization(**arguments)


def _load(path: Path) -> roadrunner.RoadRunner:
    r = roadrunner.RoadRunner(str(path))
    r.integrator.absolute_tolerance = 1e-12
    r.integrator.relative_tolerance = 1e-12
    return r


def _edit(path: Path, tmp_path: Path, edit: Any) -> Path:
    """Write a copy of the model which was changed."""
    document = libsbml.readSBMLFromFile(str(path))
    edit(document.getModel())
    edited = tmp_path / "edited.xml"
    libsbml.writeSBMLToFile(document, str(edited))
    return edited


# --- THE COMPILED MODEL ---


def test_the_path_of_the_compiled_model(tmp_path: Path) -> None:
    """The model with the networks is `<stem>_sciml.xml`."""
    assert compiled_path(Path("models/lv.xml")) == Path("models/lv_sciml.xml")
    assert compiled_path(Path("models/lv.xml"), tmp_path) == tmp_path / "lv_sciml.xml"


@pytest.mark.parametrize(("level", "version"), [(3, 1), (3, 2), (2, 4)])
def test_the_rules_are_the_forward_pass(
    tmp_path: Path, level: int, version: int
) -> None:
    """The network in the right hand side is evaluated at every time point."""
    path = write_model(tmp_path / "lv.xml", level=level, version=version)
    network = feed_forward()
    compiled = compile_network(
        path, [_hybridization(network)], tmp_path / "out" / "lv_sciml.xml"
    )
    assert compiled == tmp_path / "out" / "lv_sciml.xml"
    assert compiled.is_file()

    r = _load(compiled)
    r.timeCourseSelections = ["time", "prey", "predator", "gamma", "net1__output0__0"]
    result = r.simulate(0, 5, 26)
    assert np.ptp(result[:, 3]) > 0.1
    for row in result:
        (expected,) = network.forward(np.array([row[1], row[2]]))
        assert row[3] == pytest.approx(expected[0], rel=TOLERANCE, abs=TOLERANCE)
        assert row[4] == row[3]


def test_the_model_which_was_compiled_is_not_changed(model_path: Path) -> None:
    """The network is written into a copy."""
    before = model_path.read_text()
    compile_network(model_path, [_hybridization()], compiled_path(model_path))
    assert model_path.read_text() == before


@pytest.mark.parametrize("activation", sorted(ACTIVATIONS))
def test_a_function_in_the_model(model_path: Path, activation: str) -> None:
    """Every function which is evaluated on expressions is math of SBML."""
    network = feed_forward(activation=activation, kwargs=ACTIVATIONS[activation])
    compiled = compile_network(
        model_path, [_hybridization(network)], compiled_path(model_path)
    )
    r = _load(compiled)
    for prey, predator in [(0.4, 4.6), (2.0, 0.1), (0.0, 0.0), (7.5, 3.0)]:
        r["init(prey)"] = prey
        r["init(predator)"] = predator
        r.reset()
        (expected,) = network.forward(np.array([prey, predator]))
        assert r["gamma"] == pytest.approx(expected[0], rel=TOLERANCE, abs=TOLERANCE)


def test_the_parameters_of_the_compiled_model(model_path: Path) -> None:
    """The elements, the inputs, the units and the outputs are parameters."""
    network = feed_forward()
    compiled = compile_network(
        model_path, [_hybridization(network)], compiled_path(model_path)
    )
    document = libsbml.readSBMLFromFile(str(compiled))
    model = document.getModel()
    ids = {model.getParameter(k).getId() for k in range(model.getNumParameters())} - {
        "alpha",
        "beta",
        "gamma",
        "delta",
    }
    elements = set(network.parameter_ids())
    inputs = {"net1__input0__0", "net1__input0__1"}
    units = {f"net1__layer1__{k}" for k in range(3)}
    units |= {f"net1__act__{k}" for k in range(3)}
    units |= {"net1__layer2__0"}
    assert ids == elements | inputs | units | {"net1__output0__0"}

    for sid, (layer, name, index) in network.parameter_ids().items():
        parameter = model.getParameter(sid)
        assert parameter.getConstant()
        # libsbml writes a number with 15 digits
        assert parameter.getValue() == pytest.approx(
            network.parameters[layer][name][index], rel=1e-14
        )
        assert parameter.getUnits() == "dimensionless"
    for sid in inputs | units | {"net1__output0__0", "gamma"}:
        assert not model.getParameter(sid).getConstant()
        assert model.getRuleByVariable(sid) is not None
    # one layer deep: the rule of a unit names the units of the node before
    rule = libsbml.formulaToL3String(model.getRuleByVariable("net1__act__1").getMath())
    assert rule == "tanh(net1__layer1__1)"
    assert libsbml.formulaToL3String(model.getRuleByVariable("gamma").getMath()) == (
        "net1__output0__0"
    )


def test_an_element_of_the_model_is_a_parameter_of_a_fit(model_path: Path) -> None:
    """A change of an element changes the output, as a fit does it."""
    network = feed_forward()
    compiled = compile_network(
        model_path, [_hybridization(network)], compiled_path(model_path)
    )
    r = _load(compiled)
    sid = "net1__layer1__weight__2_1"
    r[sid] = 0.25
    r.reset()
    parameters = network.with_values({sid: 0.25})
    x = np.array([r["prey"], r["predator"]])
    (expected,) = network.forward(x, parameters=parameters)
    assert r["gamma"] == pytest.approx(expected[0], rel=TOLERANCE, abs=TOLERANCE)
    assert r["gamma"] != pytest.approx(network.forward(x)[0][0])


def test_a_network_in_an_observable(model_path: Path) -> None:
    """The target of an observable is a parameter which is added."""
    network = feed_forward()
    hybridization = _hybridization(network, pattern=OBSERVABLE, target="net1_output1")
    compiled = compile_network(model_path, [hybridization], compiled_path(model_path))
    r = _load(compiled)
    r.timeCourseSelections = ["time", "prey", "predator", "net1_output1", "gamma"]
    result = r.simulate(0, 5, 11)
    for row in result:
        (expected,) = network.forward(np.array([row[1], row[2]]))
        assert row[3] == pytest.approx(expected[0], rel=TOLERANCE, abs=TOLERANCE)
        # the right hand side is the one of the model
        assert row[4] == 0.8


def test_a_network_in_the_right_hand_side_and_an_observable(model_path: Path) -> None:
    """The two hybridizations of one network are one network in the model."""
    network = feed_forward(n_outputs=2)
    inputs = {
        "net1__input0__0": NetworkInput(formula="prey + (alpha - 1.3)"),
        "net1__input0__1": NetworkInput(formula="2 * k"),
    }
    hybridizations = [
        _hybridization(
            network,
            pattern=RHS,
            inputs=inputs,
            outputs={"net1__output0__1": "gamma"},
            constants={"k": 0.5},
        ),
        _hybridization(
            network,
            pattern=OBSERVABLE,
            inputs=inputs,
            outputs={"net1__output0__0": "y"},
            constants={"k": 0.5},
        ),
    ]
    compiled = compile_network(model_path, hybridizations, compiled_path(model_path))
    r = _load(compiled)
    assert r["k"] == 0.5
    r.timeCourseSelections = ["time", "prey", "y", "gamma"]
    for row in r.simulate(0, 5, 11):
        (expected,) = network.forward(np.array([row[1], 1.0]))
        assert row[2] == pytest.approx(expected[0], rel=TOLERANCE, abs=TOLERANCE)
        assert row[3] == pytest.approx(expected[1], rel=TOLERANCE, abs=TOLERANCE)


def test_two_networks_in_one_model(model_path: Path) -> None:
    """All networks of a model are compiled in one pass."""
    net1 = feed_forward("net1", seed=1)
    net2 = feed_forward("net2", seed=2, activation="relu")
    compiled = compile_network(
        model_path,
        [
            _hybridization(net1, target="gamma"),
            _hybridization(net2, target="alpha"),
            _hybridization(feed_forward("net3"), pattern=PRE, target="beta"),
        ],
        compiled_path(model_path),
    )
    r = _load(compiled)
    x = np.array([r["prey"], r["predator"]])
    assert r["gamma"] == pytest.approx(net1.forward(x)[0][0], rel=TOLERANCE)
    assert r["alpha"] == pytest.approx(net2.forward(x)[0][0], rel=TOLERANCE)
    # a network before the simulation is not a part of the model
    assert r["beta"] == 0.9
    assert "net3__output0__0" not in r.model.getGlobalParameterIds()


def test_an_input_which_is_an_array(model_path: Path) -> None:
    """The elements of an array are constant parameters of the model."""
    network = two_inputs()

    def hybridization(arrays: dict[str, Any]) -> Hybridization:
        return Hybridization(
            network=network,
            pattern=RHS,
            model="lv",
            inputs={
                "net6__input0__0": NetworkInput(formula="prey"),
                "net6__input1": NetworkInput(arrays=arrays),
            },
            outputs={"net6__output0__0": "gamma"},
        )

    array = np.array([1.0, 2.0, 3.0])
    compiled = compile_network(
        model_path, [hybridization({ALL_CONDITIONS: array})], compiled_path(model_path)
    )
    r = _load(compiled)
    (expected,) = network.forward(np.array([r["prey"]]), array)
    assert r["gamma"] == pytest.approx(expected[0], rel=TOLERANCE)

    # the arrays of conditions: the model has the values of the first one and
    # the derived changes of a fit set the ones of the simulation
    conditional = hybridization({"e2": array[::-1], "e1": array})
    compiled = compile_network(model_path, [conditional], compiled_path(model_path))
    r = _load(compiled)
    assert [r[f"net6__input1__{k}"] for k in range(3)] == [1.0, 2.0, 3.0]
    for sid, value in conditional.derived_changes({}, "e2").items():
        r[sid] = value
    r.reset()
    (expected,) = network.forward(np.array([r["prey"]]), array[::-1])
    assert r["gamma"] == pytest.approx(expected[0], rel=TOLERANCE)


def test_the_time_is_an_input(model_path: Path) -> None:
    """A formula of an input uses the time of the model."""
    network = feed_forward()
    hybridization = _hybridization(
        network,
        inputs={
            "net1__input0__0": NetworkInput(formula="time"),
            "net1__input0__1": NetworkInput(formula="0.5"),
        },
    )
    compiled = compile_network(model_path, [hybridization], compiled_path(model_path))
    r = _load(compiled)
    r.timeCourseSelections = ["time", "gamma"]
    for time, gamma in r.simulate(0, 2, 5):
        (expected,) = network.forward(np.array([time, 0.5]))
        assert gamma == pytest.approx(expected[0], rel=TOLERANCE, abs=TOLERANCE)


# --- WHAT IS NOT COMPILED ---


def test_the_error_function_is_not_math_of_sbml(model_path: Path) -> None:
    """`gelu` with the error function names its node."""
    network = feed_forward(activation="gelu", kwargs={"approximate": "none"})
    with pytest.raises(
        NetworkCompilationError, match=r"Network 'net1', node 'act'.*\['erf'\]"
    ):
        compile_network(
            model_path, [_hybridization(network)], compiled_path(model_path)
        )
    assert not compiled_path(model_path).exists()


@pytest.mark.parametrize(
    "sid",
    [
        "net1__layer1__weight__0_0",
        "net1__input0__1",
        "net1__act__2",
        "net1__output0__0",
        "k",
    ],
)
def test_an_id_of_the_model_which_is_an_id_of_the_network(
    model_path: Path, tmp_path: Path, sid: str
) -> None:
    """An entity of the model named like a part of a network is an error."""

    def add(model: libsbml.Model) -> None:
        parameter = model.createParameter()
        parameter.setId(sid)
        parameter.setValue(1.0)
        parameter.setConstant(True)

    path = _edit(model_path, tmp_path, add)
    hybridization = _hybridization(constants={"q": 1.0} if sid == "k" else {"k": 1.0})
    if sid == "k":
        compile_network(path, [hybridization], compiled_path(path))
        hybridization = _hybridization(constants={"k": 1.0})
        with pytest.raises(NetworkHybridizationError, match="'k' is an entity"):
            compile_network(path, [hybridization], compiled_path(path))
        return
    with pytest.raises(
        NetworkCompilationError, match=rf"Network 'net1'.*'{sid}'.*entity of the model"
    ):
        compile_network(path, [hybridization], compiled_path(path))


def test_two_parts_of_a_network_with_one_id(model_path: Path) -> None:
    """The ids of the units of a node and of the elements of an array differ."""
    network = feed_forward()
    model = network.model.model_copy(deep=True)
    # the units of the node are `net1__layer1__bias__0`, like the bias
    model.forward[2].name = "layer1__bias"
    model.forward[3].args = ["layer1__bias"]
    renamed = Network(sid="net1", model=model, parameters=network.parameters)
    with pytest.raises(
        NetworkCompilationError,
        match=r"unit \(0,\) of the node 'layer1__bias' and the element "
        r"'net1__layer1__bias__0' both have the id 'net1__layer1__bias__0'",
    ):
        compile_network(
            model_path, [_hybridization(renamed)], compiled_path(model_path)
        )


def _rule(model: libsbml.Model) -> None:
    model.getParameter("gamma").setConstant(False)
    rule = model.createAssignmentRule()
    rule.setVariable("gamma")
    rule.setMath(libsbml.parseL3Formula("2 * alpha"))


def _initial_assignment(model: libsbml.Model) -> None:
    assignment = model.createInitialAssignment()
    assignment.setSymbol("gamma")
    assignment.setMath(libsbml.parseL3Formula("2 * alpha"))


def _event(model: libsbml.Model) -> None:
    model.getParameter("gamma").setConstant(False)
    event = model.createEvent()
    event.setId("e1")
    event.setUseValuesFromTriggerTime(True)
    trigger = event.createTrigger()
    trigger.setMath(libsbml.parseL3Formula("time > 1"))
    trigger.setInitialValue(True)
    trigger.setPersistent(True)
    assignment = event.createEventAssignment()
    assignment.setVariable("gamma")
    assignment.setMath(libsbml.parseL3Formula("1"))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_rule, "is set by a rule of the model"),
        (_initial_assignment, "has an initial assignment"),
        (_event, "is set by the event 'e1'"),
    ],
)
def test_a_target_which_the_model_sets(
    model_path: Path, tmp_path: Path, edit: Any, message: str
) -> None:
    """A target has one rule, the error names the target and what sets it."""
    path = _edit(model_path, tmp_path, edit)
    with pytest.raises(NetworkCompilationError, match=message) as excinfo:
        compile_network(path, [_hybridization()], compiled_path(path))
    assert "Network 'net1': the target 'gamma' of 'net1__output0__0'" in str(
        excinfo.value
    )


def test_a_target_which_is_a_species(model_path: Path, tmp_path: Path) -> None:
    """A species which a reaction changes is not set by a rule."""
    inputs = {
        "net1__input0__0": NetworkInput(formula="alpha"),
        "net1__input0__1": NetworkInput(formula="predator"),
    }
    with pytest.raises(
        NetworkCompilationError,
        match=r"'prey' of 'net1__output0__0' is a species which the reaction 'v1'",
    ):
        compile_network(
            model_path,
            [_hybridization(target="prey", inputs=inputs)],
            compiled_path(model_path),
        )

    def add(model: libsbml.Model) -> None:
        species = model.createSpecies()
        species.setId("food")
        species.setCompartment("default")
        species.setInitialAmount(1.0)
        species.setHasOnlySubstanceUnits(True)
        species.setBoundaryCondition(True)
        species.setConstant(True)

    path = _edit(model_path, tmp_path, add)
    network = feed_forward()
    compiled = compile_network(
        path, [_hybridization(network, target="food")], compiled_path(path)
    )
    r = _load(compiled)
    x = np.array([r["prey"], r["predator"]])
    assert r["food"] == pytest.approx(network.forward(x)[0][0], rel=TOLERANCE)


def test_a_target_which_is_a_reaction(model_path: Path) -> None:
    """A rule sets a parameter, a compartment or a species."""
    with pytest.raises(
        NetworkCompilationError,
        match="'v1' of 'net1__output0__0' is not a parameter, a compartment or",
    ):
        compile_network(
            model_path, [_hybridization(target="v1")], compiled_path(model_path)
        )


def test_a_formula_with_a_symbol_the_model_does_not_have(model_path: Path) -> None:
    """A symbol of a formula is an entity of the model or a constant."""
    hybridization = _hybridization(
        inputs={
            "net1__input0__0": NetworkInput(formula="prey * kappa"),
            "net1__input0__1": NetworkInput(formula="predator"),
        }
    )
    with pytest.raises(
        NetworkCompilationError,
        match=r"Network 'net1', input 'net1__input0__0'.*'prey \* kappa' uses "
        r"\['kappa'\]",
    ):
        compile_network(model_path, [hybridization], compiled_path(model_path))


def test_what_is_compiled_together(model_path: Path, tmp_path: Path) -> None:
    """The hybridizations of a call belong to one model and are compiled."""
    with pytest.raises(NetworkCompilationError, match="No network is compiled"):
        compile_network(model_path, [], compiled_path(model_path))
    with pytest.raises(
        NetworkCompilationError, match=r"No network.*\['pre_initialization'\]"
    ):
        compile_network(
            model_path, [_hybridization(pattern=PRE)], compiled_path(model_path)
        )
    with pytest.raises(NetworkCompilationError, match=r"name the models \['lv', 'x'\]"):
        compile_network(
            model_path,
            [_hybridization(), _hybridization(feed_forward("net2"), model="x")],
            compiled_path(model_path),
        )
    with pytest.raises(NetworkCompilationError, match="replaces the model"):
        compile_network(model_path, [_hybridization()], model_path)
    with pytest.raises(NetworkHybridizationError, match="does not exist"):
        compile_network(
            tmp_path / "missing.xml", [_hybridization()], compiled_path(model_path)
        )


def test_two_hybridizations_of_a_network_which_differ(model_path: Path) -> None:
    """A network is compiled once, its hybridizations share what they share."""
    network = feed_forward(n_outputs=2)
    first = _hybridization(network, outputs={"net1__output0__0": "gamma"})
    with pytest.raises(NetworkCompilationError, match="differ in 'inputs'"):
        compile_network(
            model_path,
            [
                first,
                _hybridization(
                    network,
                    outputs={"net1__output0__1": "alpha"},
                    inputs={
                        "net1__input0__0": NetworkInput(formula="prey"),
                        "net1__input0__1": NetworkInput(formula="prey"),
                    },
                ),
            ],
            compiled_path(model_path),
        )
    with pytest.raises(NetworkCompilationError, match="differ in 'network'"):
        compile_network(
            model_path,
            [
                first,
                _hybridization(
                    feed_forward(n_outputs=2, seed=5),
                    outputs={"net1__output0__1": "alpha"},
                ),
            ],
            compiled_path(model_path),
        )
    with pytest.raises(
        NetworkCompilationError, match=r"use the outputs \['net1__output0__0'\]"
    ):
        compile_network(
            model_path,
            [first, _hybridization(network, pattern=OBSERVABLE, target="y")],
            compiled_path(model_path),
        )
    with pytest.raises(
        NetworkCompilationError,
        match=r"the constant 'k' has the values '1\.0' and '2\.0'",
    ):
        compile_network(
            model_path,
            [
                _hybridization(
                    network,
                    outputs={"net1__output0__0": "gamma"},
                    constants={"k": 1.0},
                ),
                _hybridization(
                    network,
                    outputs={"net1__output0__1": "alpha"},
                    constants={"k": 2.0},
                ),
            ],
            compiled_path(model_path),
        )
```

- [ ] **Step 2: Apply this patch to `tests/sciml/test_parameters.py`**

Apply this patch to `tests/sciml/test_parameters.py`:

```diff
diff --git a/tests/sciml/test_parameters.py b/tests/sciml/test_parameters.py
index 532340a..c8adbbe 100644
--- a/tests/sciml/test_parameters.py
+++ b/tests/sciml/test_parameters.py
@@ -1,11 +1,13 @@
 """Tests of the nominal values and the fit parameters of a network."""

+import logging
 from itertools import pairwise

 import numpy as np
 import pytest
 from petab_sciml import Input, Layer, NNModel, Node

+from sbmlsim.fit.options import ParameterScaleType
 from sbmlsim.sciml import Network, NetworkImportError
 from sbmlsim.sciml.parameters import (
     covered_arrays,
@@ -237,11 +239,12 @@ def test_the_ids_are_the_ids_of_the_network() -> None:
     )


-def test_an_estimated_array_without_nominal_values() -> None:
-    """An estimated element needs a start value.
+def test_the_elements_of_a_layer_which_is_not_called(
+    caplog: pytest.LogCaptureFixture,
+) -> None:
+    """The elements of a layer the forward pass does not call are left out.

-    A layer which the forward pass does not call needs no values to be
-    evaluated, but its elements are not estimated without them.
+    The outputs do not depend on them, a fit cannot estimate them.
     """
     network = _network(with_values=False)
     model = network.model.model_copy(deep=True)
@@ -255,12 +258,32 @@ def test_an_estimated_array_without_nominal_values() -> None:
     network = Network(sid="net1", model=model)
     values = {"net1.block.layer1": 0.0, "net1.norm": 1.0, "net1.layer2": 0.5}
     assert "unused" not in nominal_parameters(network, values)
-    with pytest.raises(
-        NetworkImportError, match=r"'unused'.*'weight' is estimated and has no nominal"
-    ):
-        network_fit_parameters(
-            network, estimate={"net1.unused": True}, bounds={}, values=values
+    with caplog.at_level(logging.WARNING, logger="sbmlsim.sciml.parameters"):
+        parameters = network_fit_parameters(
+            network, estimate={"net1": True}, bounds={}, values=values
         )
+    assert "['unused']" in caplog.text
+    assert parameters
+    assert not [p.pid for p in parameters if "unused" in p.pid]
+    assert {p.scale for p in parameters} == {ParameterScaleType.LINEAR}
+    assert {p.target for p in parameters} == {None}
+
+
+def test_an_estimated_array_without_nominal_values() -> None:
+    """An estimated element needs a start value."""
+    network = _network(with_values=False)
+    with pytest.raises(NetworkImportError, match=r"has no values"):
+        network_fit_parameters(network, estimate={"net1": True}, bounds={})
+
+
+def test_the_elements_of_a_network_before_the_simulation() -> None:
+    """The elements are not entities of a model, their target says so."""
+    parameters = network_fit_parameters(
+        _network(), estimate={"net1": True}, bounds={}, external=True
+    )
+    assert parameters
+    assert all(p.target == f"sciml:{p.pid}" for p in parameters)
+    assert all(p.is_external and p.entity_id == p.pid for p in parameters)


 def test_a_layer_without_trainable_arrays() -> None:
```

- [ ] **Step 3: Create `tests/sciml/experiment.py`**

Create `tests/sciml/experiment.py`:

```python
"""A hybrid problem which is defined in python, without PEtab.

The experiment simulates the model of Lotka and Volterra and compares the
species with data. It is a module of its own, because the workers of a
parallel fit import the experiment class of a problem.
"""

from pathlib import Path
from typing import ClassVar

import pandas as pd

from sbmlsim.data import DataSet
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import FitData, FitMapping, FitMappingCollection
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import AbstractSim, Timecourse, TimecourseSim
from sbmlsim.task import Task
from tests.sciml.hybrid import MODEL_PATH

#: the species of the model at the times `1` to `10`, simulated with the
#: parameters of the model
DATA: dict[str, list[float]] = {
    "prey": [
        0.1996,
        0.4843,
        1.6064,
        5.4941,
        3.0782,
        0.1952,
        0.2965,
        0.9041,
        3.1091,
        8.8516,
    ],
    "predator": [
        0.9224,
        0.1947,
        0.0676,
        0.1419,
        6.6987,
        1.9956,
        0.3923,
        0.0994,
        0.0683,
        1.2238,
    ],
}

#: the simulations of the experiment, which are the conditions of the inputs
SIMULATIONS = ("e1", "e2")


class LotkaVolterra(SimulationExperiment):
    """The model of Lotka and Volterra, simulated twice and compared with data.

    Attributes:
        model_path: the model the experiment simulates, which a test replaces
            by the model with the networks.
    """

    model_path: ClassVar[Path] = MODEL_PATH

    def models(self) -> dict[str, AbstractModel | Path]:
        return {
            "lv": AbstractModel(
                source=self.model_path,
                language_type=AbstractModel.LanguageType.SBML,
            )
        }

    def datasets(self) -> dict[str, DataSet]:
        return {
            sid: DataSet.from_df(
                pd.DataFrame(
                    {
                        "time": [float(k) for k in range(1, 11)],
                        "time_unit": "second",
                        "value": values,
                        "value_unit": "dimensionless",
                    }
                ),
                ureg=self.ureg,
            )
            for sid, values in DATA.items()
        }

    def simulations(self) -> dict[str, AbstractSim]:
        return {
            sid: TimecourseSim([Timecourse(start=0.0, end=10.0, steps=100)])
            for sid in SIMULATIONS
        }

    def tasks(self) -> dict[str, Task]:
        return {f"task_{sid}": Task(model="lv", simulation=sid) for sid in SIMULATIONS}

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            f"{species}_{sid}": FitMapping(
                self,
                reference=FitData(self, dataset=species, xid="time", yid="value"),
                observable=FitData(self, task=f"task_{sid}", xid="time", yid=species),
            )
            for sid in SIMULATIONS
            for species in DATA
        }


def collections(
    experiment: type[SimulationExperiment] = LotkaVolterra,
) -> list[FitMappingCollection]:
    """Get the fit mappings of the experiment, one collection per simulation.

    Args:
        experiment: the experiment class.

    Returns:
        The collections.
    """
    return [
        FitMappingCollection(
            experiment=experiment,
            sid=sid,
            mappings=[f"{species}_{sid}" for species in DATA],
        )
        for sid in SIMULATIONS
    ]
```

- [ ] **Step 4: Create `tests/sciml/test_fit.py`**

Create `tests/sciml/test_fit.py`:

```python
"""Tests of a fit of a hybrid problem which is defined in python.

The problem is `tests.sciml.experiment.LotkaVolterra` with networks of the
three patterns, without PEtab.
"""

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from sbmlsim.fit import FitParameter, FitSettings
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.fit.petab_v2.likelihood import gradient, log_likelihood
from sbmlsim.fit.runner import run_optimization
from sbmlsim.sciml import (
    Hybridization,
    Network,
    NetworkInput,
    NetworkPattern,
    compile_network,
    compiled_path,
    network_fit_parameters,
)
from sbmlsim.sciml.hybridization import ALL_CONDITIONS
from tests.sciml.experiment import LotkaVolterra, collections
from tests.sciml.hybrid import MODEL_PATH, feed_forward, two_inputs

PRE = NetworkPattern.PRE_INITIALIZATION
RHS = NetworkPattern.RHS

#: settings with which two simulations of the problem agree
SETTINGS = FitSettings(
    parameter_scale=ParameterScaleType.LINEAR,
    variable_step_size=False,
    absolute_tolerance=1e-12,
    relative_tolerance=1e-12,
)

#: the parameters of the model which are fitted
MECHANISTIC = [
    FitParameter("alpha", 1.3, 0.0, 15.0, unit="dimensionless"),
    FitParameter("beta", 0.9, 0.0, 15.0, unit="dimensionless"),
]


def _problem(
    hybridizations: Sequence[Hybridization],
    parameters: Sequence[FitParameter],
    experiment: type[LotkaVolterra] = LotkaVolterra,
) -> OptimizationProblem:
    return OptimizationProblem(
        opid="hybrid",
        mapping_collections=collections(experiment),
        fit_parameters=[*MECHANISTIC, *parameters],
        base_path=MODEL_PATH.parent,
        data_path=MODEL_PATH.parent,
        hybridizations=list(hybridizations),
    )


def _before(network: Network, **kwargs: object) -> Hybridization:
    """Get the network before the simulation, `gamma = net(alpha, k)`."""
    arguments: dict[str, object] = {
        "network": network,
        "pattern": PRE,
        "model": "lv",
        "inputs": {
            f"{network.sid}__input0__0": NetworkInput(formula="alpha"),
            f"{network.sid}__input0__1": NetworkInput(formula="k"),
        },
        "outputs": {f"{network.sid}__output0__0": "gamma"},
        "constants": {"k": 0.5},
    }
    arguments.update(kwargs)
    return Hybridization(**arguments)  # ty: ignore[invalid-argument-type]


# --- BEFORE THE SIMULATION ---


def test_a_network_before_the_simulation() -> None:
    """The elements are parameters of the fit which are not in the model."""
    network = feed_forward()
    hybridization = _before(network)
    elements = network_fit_parameters(
        network, estimate={"net1": True}, bounds={}, external=True
    )
    problem = _problem([hybridization], elements)
    problem.initialize(SETTINGS)
    assert problem.pids == ["alpha", "beta", *network.parameter_ids()]
    assert problem.scales_initialized == [ParameterScaleType.LINEAR] * len(problem.pids)
    # the elements start from their nominal values
    x = np.asarray(problem.x0, dtype=float)
    np.testing.assert_allclose(problem.xmodel, x)

    problem.predictions(x)
    changes = problem.simulations[0].timecourses[0].changes
    (expected,) = network.forward(np.array([1.3, 0.5]))
    assert changes["gamma"].magnitude == pytest.approx(expected[0])

    # a plain problem with the value of gamma has the same predictions
    plain = OptimizationProblem(
        opid="plain",
        mapping_collections=collections(),
        fit_parameters=[
            *MECHANISTIC,
            FitParameter("gamma", float(expected[0]), unit="dimensionless"),
        ],
        base_path=MODEL_PATH.parent,
        data_path=MODEL_PATH.parent,
    )
    plain.initialize(SETTINGS)
    for k, values in problem.predictions(x).items():
        np.testing.assert_allclose(
            values, plain.predictions(np.array([1.3, 0.9, expected[0]]))[k], rtol=1e-9
        )

    # the objective depends on every element
    grad = gradient(problem, order=4)
    assert np.all(np.isfinite(grad.to_numpy()))
    assert np.all(grad[list(network.parameter_ids())].abs() > 0.0)
    assert np.isfinite(log_likelihood(problem))


def test_the_arrays_of_the_simulations() -> None:
    """Every simulation is evaluated with the arrays of its condition."""
    network = two_inputs()
    hybridization = Hybridization(
        network=network,
        pattern=PRE,
        model="lv",
        inputs={
            "net6__input0__0": NetworkInput(formula="alpha"),
            "net6__input1": NetworkInput(
                arrays={"e1": [1.0, 2.0, 3.0], "e2": [3.0, 2.0, 1.0]}
            ),
        },
        outputs={"net6__output0__0": "gamma"},
    )
    elements = network_fit_parameters(
        network, estimate={"net6": True}, bounds={}, external=True
    )
    problem = _problem([hybridization], elements)
    problem.initialize(SETTINGS)
    x = np.asarray(problem.x0, dtype=float)
    predictions = problem.predictions(x, indices=problem.indices())
    e1 = predictions[problem.mapping_keys.index("prey_e1")]
    e2 = predictions[problem.mapping_keys.index("prey_e2")]
    assert not np.allclose(e1, e2)
    for sid, array in (("e1", [1.0, 2.0, 3.0]), ("e2", [3.0, 2.0, 1.0])):
        k = problem.simulation_keys.index(sid)
        changes = problem.simulations[k].timecourses[0].changes
        (expected,) = network.forward(np.array([1.3]), np.array(array))
        assert changes["gamma"].magnitude == pytest.approx(expected[0])


def test_a_frozen_layer() -> None:
    """The elements of a frozen layer are not parameters and keep their values."""
    network = feed_forward()
    elements = network_fit_parameters(
        network,
        estimate={"net1": True, "net1.layer1": False},
        bounds={"net1": (-5.0, 5.0)},
        external=True,
    )
    frozen = set(network.parameter_ids()) - {p.pid for p in elements}
    assert frozen == {sid for sid in network.parameter_ids() if "layer1" in sid}
    problem = _problem([_before(network, frozen=frozen)], elements)
    problem.initialize(SETTINGS)
    assert all(np.isfinite(problem.bounds[0]))
    grad = gradient(problem)
    assert set(grad.index) == {"alpha", "beta", *(p.pid for p in elements)}

    with pytest.raises(ValueError, match=r"\['net1__layer1__bias__0'\] are frozen"):
        _problem(
            [_before(network, frozen=frozen)],
            [
                *elements,
                FitParameter(
                    "net1__layer1__bias__0",
                    0.0,
                    unit="dimensionless",
                    target="sciml:net1__layer1__bias__0",
                ),
            ],
        ).initialize(SETTINGS)


def test_a_parallel_fit_pickles_the_networks(tmp_path: Path) -> None:
    """The workers of a parallel fit get the hybridizations with the problem."""
    network = feed_forward()
    elements = network_fit_parameters(
        network, estimate={"net1.layer2": True}, bounds={}, external=True
    )
    problem = _problem([_before(network)], elements)
    restored = pickle.loads(pickle.dumps(problem))
    assert restored.hybridizations == problem.hybridizations
    assert restored.hybridizations[0].network == network

    result = run_optimization(
        problem,
        settings=SETTINGS,
        size=2,
        n_cores=2,
        seed=1,
        show_progress=False,
        max_nfev=2,
    )
    assert len(result.fits) == 2
    assert all(np.all(np.isfinite(fit.x)) for fit in result.fits)
    assert all("Error" not in fit.message for fit in result.fits)


# --- IN THE RIGHT HAND SIDE ---


def test_a_network_in_the_right_hand_side(tmp_path: Path) -> None:
    """The elements are parameters of the model with the network."""
    network = feed_forward()
    hybridization = Hybridization(
        network=network,
        pattern=RHS,
        model="lv",
        inputs={
            "net1__input0__0": NetworkInput(formula="prey"),
            "net1__input0__1": NetworkInput(formula="predator"),
        },
        outputs={"net1__output0__0": "gamma"},
    )
    compiled = compile_network(
        MODEL_PATH, [hybridization], compiled_path(MODEL_PATH, tmp_path)
    )

    class Compiled(LotkaVolterra):
        model_path: ClassVar[Path] = compiled

    elements = network_fit_parameters(network, estimate={"net1": True}, bounds={})
    assert all(not p.is_external for p in elements)
    problem = _problem([hybridization], elements, experiment=Compiled)
    problem.initialize(SETTINGS)
    # the network is a part of the model, nothing is derived
    assert problem.group_derived == [[(hybridization, {})], [(hybridization, {})]]
    x = np.asarray(problem.x0, dtype=float)
    np.testing.assert_allclose(problem.xmodel, x, rtol=1e-14)
    cost = problem.cost_least_square(x)
    shifted = x.copy()
    shifted[problem.pids.index("net1__layer2__bias__0")] += 0.1
    assert problem.cost_least_square(shifted) != pytest.approx(cost)
    grad = gradient(problem)
    assert np.all(grad[list(network.parameter_ids())].abs() > 0.0)


def test_the_arrays_of_the_simulations_of_a_compiled_network(tmp_path: Path) -> None:
    """The elements of an array of a condition are set before its simulation."""
    network = two_inputs()
    hybridization = Hybridization(
        network=network,
        pattern=RHS,
        model="lv",
        inputs={
            "net6__input0__0": NetworkInput(formula="prey"),
            "net6__input1": NetworkInput(
                arrays={"e1": [1.0, 2.0, 3.0], "e2": [3.0, 2.0, 1.0]}
            ),
        },
        outputs={"net6__output0__0": "gamma"},
    )
    compiled = compile_network(
        MODEL_PATH, [hybridization], compiled_path(MODEL_PATH, tmp_path)
    )

    class Compiled(LotkaVolterra):
        model_path: ClassVar[Path] = compiled

    problem = _problem([hybridization], [], experiment=Compiled)
    problem.initialize(SETTINGS)
    x = np.asarray(problem.x0, dtype=float)
    predictions = problem.predictions(x, indices=problem.indices())
    e1 = predictions[problem.mapping_keys.index("prey_e1")]
    e2 = predictions[problem.mapping_keys.index("prey_e2")]
    assert not np.allclose(e1, e2)
    k = problem.simulation_keys.index("e2")
    changes = problem.simulations[k].timecourses[0].changes
    assert [changes[f"net6__input1__{i}"].magnitude for i in range(3)] == [
        3.0,
        2.0,
        1.0,
    ]


def test_a_network_which_the_model_does_not_have() -> None:
    """The elements of a compiled network are entities of the compiled model."""
    network = feed_forward()
    elements = network_fit_parameters(network, estimate={"net1": True}, bounds={})
    with pytest.raises(RuntimeError, match="net1__layer1__weight__0_0"):
        _problem([], elements).initialize(SETTINGS)


def test_an_array_which_all_conditions_share(tmp_path: Path) -> None:
    """An array of all conditions is a part of the compiled model."""
    network = two_inputs()
    hybridization = Hybridization(
        network=network,
        pattern=RHS,
        model="lv",
        inputs={
            "net6__input0__0": NetworkInput(formula="prey"),
            "net6__input1": NetworkInput(arrays={ALL_CONDITIONS: [1.0, 2.0, 3.0]}),
        },
        outputs={"net6__output0__0": "gamma"},
    )
    compiled = compile_network(
        MODEL_PATH, [hybridization], compiled_path(MODEL_PATH, tmp_path)
    )

    class Compiled(LotkaVolterra):
        model_path: ClassVar[Path] = compiled

    problem = _problem([hybridization], [], experiment=Compiled)
    problem.initialize(SETTINGS)
    assert problem.group_derived == [[(hybridization, {})], [(hybridization, {})]]
    problem.predictions(np.asarray(problem.x0, dtype=float))
    assert "net6__input1__0" not in problem.simulations[0].timecourses[0].changes
```

- [ ] **Step 5: Apply this patch to `tests/sciml/test_package.py`**

Apply this patch to `tests/sciml/test_package.py`:

```diff
diff --git a/tests/sciml/test_package.py b/tests/sciml/test_package.py
index 88d73c8..59c7f9a 100644
--- a/tests/sciml/test_package.py
+++ b/tests/sciml/test_package.py
@@ -93,11 +93,18 @@ def test_the_package_does_not_import_torch() -> None:
 def test_the_exports() -> None:
     """The package exports what the user of a network needs, not the backends."""
     assert sorted(sbmlsim.sciml.__all__) == [
+        "Hybridization",
         "Network",
+        "NetworkCompilationError",
         "NetworkError",
+        "NetworkHybridizationError",
         "NetworkImportError",
+        "NetworkInput",
         "NetworkParameters",
+        "NetworkPattern",
         "UnsupportedLayerError",
+        "compile_network",
+        "compiled_path",
         "network_fit_parameters",
         "nominal_parameters",
     ]
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `uv run pytest -q -x tests/sciml/test_compiler.py tests/sciml/test_parameters.py tests/sciml/test_fit.py tests/sciml/test_package.py`
Expected: FAIL at collection: `ImportError: cannot import name 'compile_network' from 'sbmlsim.sciml'`

- [ ] **Step 7: Apply this patch to `src/sbmlsim/sciml/errors.py`**

Apply this patch to `src/sbmlsim/sciml/errors.py`:

```diff
diff --git a/src/sbmlsim/sciml/errors.py b/src/sbmlsim/sciml/errors.py
index 0a38f9b..80ff837 100644
--- a/src/sbmlsim/sciml/errors.py
+++ b/src/sbmlsim/sciml/errors.py
@@ -53,3 +53,11 @@ class NetworkHybridizationError(NetworkError, ValueError):
     The message names the network and, where it applies, the input, the
     output or the target.
     """
+
+
+class NetworkCompilationError(NetworkError, ValueError):
+    """A network cannot be compiled into a model.
+
+    The message names the network and, where it applies, the node or the
+    target, and the reason.
+    """
```

- [ ] **Step 8: Create `src/sbmlsim/sciml/compiler.py`**

Create `src/sbmlsim/sciml/compiler.py`:

```python
"""The compilation of networks into an SBML model.

A network in the right hand side of a model or in an observable is evaluated
by the integrator at every step. `compile_network` writes it into the model as
parameters with assignment rules, which roadrunner simulates like any other
rule:

1. Every element of the arrays is a constant parameter with its nominal value.
2. Every element of an input is a parameter with an assignment rule of its
   formula, or a constant parameter for an element of an array.
3. The forward pass runs on expressions (`SympyBackend`). After every node
   the expressions of its units are replaced by symbols, and every unit is a
   parameter with an assignment rule of its expression. The rules stay one
   layer deep, so the size of the model grows with the number of units and
   not with the depth of the network.
4. Every element of an output is a parameter with an assignment rule, and
   the target of an output has the assignment rule `target = output`.

The ids of the parameters are the ids of `sbmlsim.sciml.network`. The
expressions are written as the MathML of SBML by `sbmlmath`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import libsbml
import numpy as np
import sympy
from petab_sciml import Node

from sbmlsim.mathml import TIME, expression_to_astnode, formula_symbols
from sbmlsim.sciml.backend import SympyBackend
from sbmlsim.sciml.errors import NetworkCompilationError, NetworkHybridizationError
from sbmlsim.sciml.hybridization import (
    ALL_CONDITIONS,
    Hybridization,
    NetworkInput,
    NetworkPattern,
    read_model,
)
from sbmlsim.sciml.interpreter import evaluate
from sbmlsim.sciml.network import (
    Network,
    input_id,
    output_id,
    parse_io_id,
    unit_id,
)

logger = logging.getLogger(__name__)

#: suffix of the model which carries the networks
MODEL_SUFFIX = "_sciml"

#: the unit of the parameters of a network
UNIT = "dimensionless"


def compiled_path(sbml_path: Path, directory: Path | None = None) -> Path:
    """Get the path of the model which carries the networks of a model.

    Args:
        sbml_path: the model without the networks.
        directory: the directory the model is written to, the directory of
            the model by default.

    Returns:
        `<stem>_sciml.xml` in the directory.
    """
    sbml_path = Path(sbml_path)
    parent = sbml_path.parent if directory is None else Path(directory)
    return parent / f"{sbml_path.stem}{MODEL_SUFFIX}{sbml_path.suffix}"


class _Model:
    """The model the networks are written into.

    Attributes:
        document: the SBML document.
        model: its model.
        created: id of every parameter which was added -> what it is.
    """

    def __init__(self, sbml_path: Path, network: str) -> None:
        """Read the model, see `read_model`.

        Raises:
            NetworkCompilationError: if the model cannot be read or is not
                valid SBML.
        """
        try:
            self.document, self.model = read_model(sbml_path, network)
        except NetworkHybridizationError as err:
            raise NetworkCompilationError(str(err)) from err
        self.name = Path(sbml_path).name
        self.created: dict[str, str] = {}
        errors = self.errors()
        if errors:
            raise NetworkCompilationError(
                f"Network '{network}': the model '{self.name}' is not valid SBML: "
                f"{errors}"
            )

    def errors(self) -> str:
        """Check the model for consistency.

        Returns:
            The messages of the errors of the model, empty for a valid model.
        """
        self.document.getErrorLog().clearLog()
        self.document.checkConsistency()
        errors = [
            self.document.getError(k)
            for k in range(self.document.getNumErrors())
            if self.document.getError(k).getSeverity() >= libsbml.LIBSBML_SEV_ERROR
        ]
        return "; ".join(error.getMessage().strip() for error in errors)

    def has(self, sid: str) -> bool:
        """Check whether the model has an entity of an id."""
        return self.model.getElementBySId(sid) is not None

    def add_parameter(
        self, network: str, sid: str, what: str, value: float | None = None
    ) -> libsbml.Parameter:
        """Add a parameter of a network to the model.

        Args:
            network: id of the network.
            sid: id of the parameter.
            what: what the parameter is, for the messages.
            value: the value of a constant parameter, `None` for a parameter
                which an assignment rule sets.

        Returns:
            The parameter.

        Raises:
            NetworkCompilationError: if the model has an entity of the id, or
                if the id is the id of another parameter of a network.
        """
        if sid in self.created:
            raise NetworkCompilationError(
                f"Network '{network}': {what} and {self.created[sid]} both have "
                f"the id '{sid}'"
            )
        if self.has(sid):
            raise NetworkCompilationError(
                f"Network '{network}': {what} has the id '{sid}', which is the id "
                f"of an entity of the model '{self.name}'"
            )
        parameter: libsbml.Parameter = self.model.createParameter()
        parameter.setId(sid)
        parameter.setUnits(UNIT)
        parameter.setConstant(value is not None)
        if value is not None:
            parameter.setValue(float(value))
        self.created[sid] = what
        return parameter

    def add_rule(self, network: str, sid: str, what: str, math: Any) -> None:
        """Add the assignment rule of an entity.

        Args:
            network: id of the network.
            sid: id of the entity the rule sets.
            what: what the entity is, for the messages.
            math: the expression of the rule, or its syntax tree.

        Raises:
            NetworkCompilationError: if the expression has no MathML of SBML.
        """
        rule: libsbml.AssignmentRule = self.model.createAssignmentRule()
        rule.setVariable(sid)
        if isinstance(math, sympy.Basic):
            try:
                math = expression_to_astnode(math)
            except ValueError as err:
                raise NetworkCompilationError(
                    f"Network '{network}', {what}: {err}"
                ) from err
        if rule.setMath(math) != libsbml.LIBSBML_OPERATION_SUCCESS:
            raise NetworkCompilationError(
                f"Network '{network}', {what}: the math of the rule of '{sid}' "
                f"is not math of the model '{self.name}'"
            )

    def write(self, output_path: Path) -> Path:
        """Check the model and write it.

        Args:
            output_path: the path of the model.

        Returns:
            The path.

        Raises:
            NetworkCompilationError: if the model is not valid SBML.
        """
        errors = self.errors()
        if errors:
            raise NetworkCompilationError(
                f"The model '{self.name}' with the networks is not valid SBML: {errors}"
            )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not libsbml.writeSBMLToFile(self.document, str(output_path)):
            raise NetworkCompilationError(
                f"The model with the networks cannot be written to '{output_path}'"
            )
        return output_path


def _merge(hybridizations: Sequence[Hybridization]) -> list[Hybridization]:
    """Merge the hybridizations of one network, e.g. of its two patterns.

    Args:
        hybridizations: the hybridizations which are compiled.

    Returns:
        One hybridization per network with the outputs and the constants of
        all of them, in the order of the first hybridization of a network. The
        pattern of a merged hybridization is the one of the first, the
        patterns of its outputs are kept in `_patterns`.

    Raises:
        NetworkCompilationError: if two hybridizations of one network differ
            in the network, the model or the inputs, or if two constants of
            one id differ in their value.
    """
    merged: dict[str, Hybridization] = {}
    for hybridization in hybridizations:
        sid = hybridization.network.sid
        if sid not in merged:
            merged[sid] = hybridization
            continue
        first = merged[sid]
        for name in ("network", "model", "inputs"):
            if getattr(first, name) != getattr(hybridization, name):
                raise NetworkCompilationError(
                    f"Network '{sid}': two hybridizations of the network differ "
                    f"in '{name}', a network is compiled once"
                )
        try:
            merged[sid] = Hybridization(
                network=first.network,
                pattern=first.pattern,
                model=first.model,
                inputs=first.inputs,
                outputs={**first.outputs, **hybridization.outputs},
                frozen=set(first.frozen) | set(hybridization.frozen),
                constants=_constants(sid, first, hybridization),
            )
        except NetworkHybridizationError as err:
            raise NetworkCompilationError(str(err)) from err
        if len(merged[sid].outputs) != len(first.outputs) + len(hybridization.outputs):
            raise NetworkCompilationError(
                f"Network '{sid}': two hybridizations of the network use the "
                f"outputs {sorted(set(first.outputs) & set(hybridization.outputs))}"
            )
    return list(merged.values())


def _constants(sid: str, *hybridizations: Hybridization) -> dict[str, float]:
    """Get the constants of hybridizations, see `_merge`."""
    constants: dict[str, float] = {}
    for hybridization in hybridizations:
        for key, value in hybridization.constants.items():
            if key in constants and constants[key] != value:
                raise NetworkCompilationError(
                    f"Network '{sid}': the constant '{key}' has the values "
                    f"'{constants[key]}' and '{value}'"
                )
            constants[key] = value
    return constants


def _check_target(model: _Model, network: str, key: str, target: str) -> None:
    """Check that a rule can set an entity of the model, and let it vary.

    Args:
        model: the model.
        network: id of the network.
        key: id of the output which sets the target.
        target: id of the entity.

    Raises:
        NetworkCompilationError: if the entity is not a parameter, a
            compartment or a species, or if a rule, an initial assignment, an
            event or a reaction sets it.
    """
    libsbml_model = model.model
    element = (
        libsbml_model.getParameter(target)
        or libsbml_model.getCompartment(target)
        or libsbml_model.getSpecies(target)
    )
    prefix = f"Network '{network}': the target '{target}' of '{key}'"
    if element is None:
        raise NetworkCompilationError(
            f"{prefix} is not a parameter, a compartment or a species of the "
            f"model '{model.name}'"
        )
    if target in model.created:
        raise NetworkCompilationError(
            f"{prefix} is {model.created[target]} and not an entity of the model"
        )
    if libsbml_model.getRuleByVariable(target) is not None:
        raise NetworkCompilationError(f"{prefix} is set by a rule of the model")
    if libsbml_model.getInitialAssignmentBySymbol(target) is not None:
        raise NetworkCompilationError(
            f"{prefix} has an initial assignment, which a rule replaces"
        )
    for k in range(libsbml_model.getNumEvents()):
        event: libsbml.Event = libsbml_model.getEvent(k)
        for j in range(event.getNumEventAssignments()):
            if event.getEventAssignment(j).getVariable() == target:
                raise NetworkCompilationError(
                    f"{prefix} is set by the event '{event.getId()}' of the model"
                )
    species: libsbml.Species | None = libsbml_model.getSpecies(target)
    if species is not None and not species.getBoundaryCondition():
        for k in range(libsbml_model.getNumReactions()):
            reaction: libsbml.Reaction = libsbml_model.getReaction(k)
            if (
                reaction.getReactant(target) is not None
                or reaction.getProduct(target) is not None
            ):
                raise NetworkCompilationError(
                    f"{prefix} is a species which the reaction "
                    f"'{reaction.getId()}' changes"
                )
    element.setConstant(False)


def _symbols(ids: np.ndarray) -> np.ndarray:
    """Get the array of the symbols of an array of ids."""
    return np.asarray(np.frompyfunc(sympy.Symbol, 1, 1)(ids), dtype=object)


def _input_symbols(model: _Model, hybridization: Hybridization) -> list[np.ndarray]:
    """Add the inputs of a network to the model.

    Args:
        model: the model.
        hybridization: the hybridization of the network.

    Returns:
        The symbols of the elements of the inputs, one array per input.

    Raises:
        NetworkCompilationError: if a formula uses a symbol which is not an
            entity of the model, or if an id is taken.
    """
    sid = hybridization.network.sid
    shapes = hybridization.input_shapes()
    ids = [np.empty(shape, dtype=object) for shape in shapes]
    for k, shape in enumerate(shapes):
        for index in np.ndindex(shape):
            ids[k][index] = input_id(sid, k, index)

    for key, network_input in hybridization.inputs.items():
        k, index = parse_io_id(sid, "input", key)
        if network_input.arrays is not None:
            array = _compiled_array(network_input)
            for element in np.ndindex(array.shape):
                model.add_parameter(
                    sid,
                    ids[k][element if index is None else index],
                    f"the input '{key}'",
                    value=float(array[element]),
                )
            continue
        formula = network_input.formula
        if formula is None or index is None:
            # excluded by the validation of the hybridization
            raise NetworkCompilationError(
                f"Network '{sid}': the input '{key}' has no formula"
            )
        unknown = sorted(
            symbol
            for symbol in formula_symbols(formula)
            if symbol != TIME and not model.has(symbol)
        )
        if unknown:
            raise NetworkCompilationError(
                f"Network '{sid}', input '{key}': the formula '{formula}' uses "
                f"{unknown}, which are neither entities of the model "
                f"'{model.name}' nor constants of the hybridization"
            )
        math: libsbml.ASTNode | None = libsbml.parseL3FormulaWithModel(
            formula, model.model
        )
        if math is None:
            raise NetworkCompilationError(
                f"Network '{sid}', input '{key}': the formula '{formula}' is not "
                f"valid math: {libsbml.getLastParseL3Error()}"
            )
        model.add_parameter(sid, ids[k][index], f"the input '{key}'")
        model.add_rule(sid, ids[k][index], f"input '{key}'", math)
    return [_symbols(array) for array in ids]


def _compiled_array(network_input: NetworkInput) -> np.ndarray:
    """Get the values of an input of arrays the model is written with.

    Args:
        network_input: an input of arrays.

    Returns:
        The array of all conditions, and the array of the first condition for
        an input without one: a fit sets the values of a condition before
        every simulation, see `Hybridization.derived_changes`.
    """
    arrays = network_input.arrays or {}
    condition = ALL_CONDITIONS if ALL_CONDITIONS in arrays else sorted(arrays)[0]
    return np.asarray(arrays[condition], dtype=float)


def _element_symbols(model: _Model, network: Network) -> dict[str, dict[str, Any]]:
    """Add the elements of the arrays of a network to the model.

    Args:
        model: the model.
        network: the network.

    Returns:
        The arrays of the symbols of the elements, layer id -> array name ->
        symbols. The arrays which are not parameters, i.e. the running
        statistics of a normalization, are the numbers.
    """
    symbols: dict[str, dict[str, Any]] = {
        layer: {name: np.array(array, dtype=object) for name, array in arrays.items()}
        for layer, arrays in network.parameters.items()
    }
    for sid, (layer, name, index) in network.parameter_ids().items():
        if name not in network.parameters.get(layer, {}):
            continue
        model.add_parameter(
            network.sid,
            sid,
            f"the element '{sid}'",
            value=float(network.parameters[layer][name][index]),
        )
        symbols[layer][name][index] = sympy.Symbol(sid)
    return symbols


def _compile(model: _Model, hybridization: Hybridization) -> None:
    """Write one network into the model.

    Args:
        model: the model.
        hybridization: the hybridization of the network, with all its outputs.

    Raises:
        NetworkCompilationError: if the network cannot be compiled.
        UnsupportedLayerError: if a layer or function is not evaluated on
            expressions.
    """
    network = hybridization.network
    sid = network.sid
    for key, value in hybridization.constants.items():
        if key not in model.created:
            model.add_parameter(sid, key, f"the constant '{key}'", value=value)
    parameters = _element_symbols(model, network)
    inputs = _input_symbols(model, hybridization)

    def on_node(node: Node, value: np.ndarray) -> np.ndarray:
        """Replace the expressions of a node by the symbols of its units."""
        ids = np.empty(value.shape, dtype=object)
        for index in np.ndindex(value.shape):
            ids[index] = unit_id(sid, node.name, index)
            what = f"the unit {index} of the node '{node.name}'"
            model.add_parameter(sid, ids[index], what)
            model.add_rule(
                sid, ids[index], f"node '{node.name}'", sympy.sympify(value[index])
            )
        return _symbols(ids)

    try:
        outputs = evaluate(network.model, parameters, inputs, SympyBackend(), on_node)
    except ValueError as err:
        if isinstance(err, NetworkCompilationError):
            raise
        raise NetworkCompilationError(str(err)) from err

    for k, output in enumerate(outputs):
        for index in np.ndindex(output.shape):
            key = output_id(sid, k, index)
            model.add_parameter(sid, key, f"the output '{key}'")
            model.add_rule(sid, key, f"output '{key}'", sympy.sympify(output[index]))


def _compile_targets(
    model: _Model, hybridization: Hybridization, patterns: dict[str, NetworkPattern]
) -> None:
    """Write the rules of the targets of the outputs of a network.

    Args:
        model: the model.
        hybridization: the hybridization of the network, with all its outputs.
        patterns: id of the output -> the pattern of its hybridization.

    Raises:
        NetworkCompilationError: if a target cannot be set by a rule.
    """
    sid = hybridization.network.sid
    for key, target in hybridization.outputs.items():
        if patterns[key] is NetworkPattern.OBSERVABLE:
            model.add_parameter(
                sid, target, f"the target '{target}' of the output '{key}'"
            )
        else:
            _check_target(model, sid, key, target)
        model.add_rule(sid, target, f"target '{target}'", sympy.Symbol(key))


def compile_network(
    sbml_path: Path, hybridizations: Sequence[Hybridization], output_path: Path
) -> Path:
    """Write a model with the networks of its right hand side and observables.

    All networks of the patterns `RHS` and `OBSERVABLE` are added in one pass
    and one model is written. The hybridizations of the pattern
    `PRE_INITIALIZATION` are not a part of the model and are left out.

    Args:
        sbml_path: the SBML model without the networks.
        hybridizations: the hybridizations of the model.
        output_path: the path the model with the networks is written to, see
            `compiled_path`.

    Returns:
        The path of the model which was written.

    Raises:
        NetworkCompilationError: if no hybridization is compiled, if the
            hybridizations name different models, if an id of a network is an
            id of the model, if a target cannot be set by a rule, if an
            expression has no MathML of SBML, or if the model with the
            networks is not valid SBML. The message names the network.
        NetworkHybridizationError: if a hybridization does not fit the model,
            see `Hybridization.validate`.
        UnsupportedLayerError: if a layer or function is not evaluated on
            expressions.
    """
    compiled = [h for h in hybridizations if h.pattern.is_compiled]
    if not compiled:
        raise NetworkCompilationError(
            f"No network is compiled into '{sbml_path}': the patterns of the "
            f"hybridizations are {[str(h.pattern) for h in hybridizations]}"
        )
    models = sorted({h.model for h in compiled})
    if len(models) > 1:
        raise NetworkCompilationError(
            f"The networks {[h.network.sid for h in compiled]} are compiled "
            f"into one model, but name the models {models}"
        )
    if Path(output_path).resolve() == Path(sbml_path).resolve():
        raise NetworkCompilationError(
            f"The model with the networks replaces the model '{sbml_path}', "
            f"write it to another path"
        )
    for hybridization in compiled:
        hybridization.validate(Path(sbml_path))
    patterns = {key: h.pattern for h in compiled for key in h.outputs}

    model = _Model(Path(sbml_path), compiled[0].network.sid)
    merged = _merge(compiled)
    for hybridization in merged:
        _compile(model, hybridization)
    for hybridization in merged:
        _compile_targets(model, hybridization, patterns)
    path = model.write(output_path)
    logger.info(
        "The networks %s are compiled into '%s': %d parameters",
        [h.network.sid for h in merged],
        path.name,
        len(model.created),
    )
    return path
```

- [ ] **Step 9: Apply this patch to `src/sbmlsim/sciml/parameters.py`**

Apply this patch to `src/sbmlsim/sciml/parameters.py`:

```diff
diff --git a/src/sbmlsim/sciml/parameters.py b/src/sbmlsim/sciml/parameters.py
index 1e921d5..1113206 100644
--- a/src/sbmlsim/sciml/parameters.py
+++ b/src/sbmlsim/sciml/parameters.py
@@ -15,7 +15,8 @@ from collections.abc import Mapping

 import numpy as np

-from sbmlsim.fit.objects import FitParameter
+from sbmlsim.fit.objects import EXTERNAL_PREFIX, FitParameter
+from sbmlsim.fit.options import ParameterScaleType
 from sbmlsim.sciml.errors import NetworkImportError
 from sbmlsim.sciml.network import Network, NetworkParameters, copy_parameters

@@ -134,12 +135,17 @@ def network_fit_parameters(
     estimate: Mapping[str, bool],
     bounds: Mapping[str, tuple[float, float]],
     values: Mapping[str, float] | None = None,
+    external: bool = False,
 ) -> list[FitParameter]:
     """Create the fit parameters of the estimated elements of a network.

     `estimate`, `bounds` and `values` are given for the network, for a layer
     or for an array, and the more specific entry wins, see `covered_arrays`.

+    The elements of a layer which the forward pass does not call are left
+    out: the outputs of the network do not depend on them, so a fit cannot
+    estimate them.
+
     Args:
         network: the network with the values of its array file.
         estimate: key of the entry -> whether the elements are estimated. An
@@ -148,11 +154,15 @@ def network_fit_parameters(
             estimated element no entry covers is not bounded.
         values: key of the entry -> nominal value of the elements, which
             replaces the values of the array file.
+        external: whether the elements are not entities of a model, which is
+            the case for a network which runs before the simulation. The
+            target of such a parameter is `sciml:<id>`. The elements of a
+            network which is compiled into a model are entities of it.

     Returns:
         One parameter per estimated element, named by the id of the element,
-        with the nominal value as start value and the unit `dimensionless`, in the
-        order of `Network.parameter_ids`.
+        with the nominal value as start value, the linear scale and the unit
+        `dimensionless`, in the order of `Network.parameter_ids`.

     Raises:
         NetworkImportError: if a key does not name the network, a layer or an
@@ -164,9 +174,20 @@ def network_fit_parameters(
     bounded = resolve_entries(network, bounds)

     ids = network.parameter_ids()
+    used = set(network.used_layers())
+    unused = sorted(
+        {layer for layer, name in estimated if estimated[(layer, name)]} - used
+    )
+    if unused:
+        logger.warning(
+            "Network '%s': the layers %s are not called by the forward pass, "
+            "their elements are not estimated",
+            network.sid,
+            unused,
+        )
     fit_parameters: list[FitParameter] = []
     for sid, (layer, name, index) in ids.items():
-        if not estimated.get((layer, name), False):
+        if layer not in used or not estimated.get((layer, name), False):
             continue
         if name not in parameters.get(layer, {}):
             raise NetworkImportError(
@@ -181,6 +202,8 @@ def network_fit_parameters(
                 lower_bound=lower,
                 upper_bound=upper,
                 unit=ELEMENT_UNIT,
+                target=f"{EXTERNAL_PREFIX}{sid}" if external else None,
+                scale=ParameterScaleType.LINEAR,
             )
         )
     logger.info(
```

- [ ] **Step 10: Apply this patch to `src/sbmlsim/sciml/__init__.py`**

Apply this patch to `src/sbmlsim/sciml/__init__.py`:

```diff
diff --git a/src/sbmlsim/sciml/__init__.py b/src/sbmlsim/sciml/__init__.py
index 63328f6..4294e5f 100644
--- a/src/sbmlsim/sciml/__init__.py
+++ b/src/sbmlsim/sciml/__init__.py
@@ -5,9 +5,12 @@ which is what [PEtab SciML](https://github.com/PEtab-dev/petab_sciml)
 describes. This package is the native half of the support: `Network` is the
 architecture and the arrays of one network and `Network.forward` evaluates it
 with numpy; `nominal_parameters` and `network_fit_parameters` give the arrays
-and the fit parameters a problem describes. The package knows nothing of
-PEtab, the translation of a PEtab SciML problem will be part of
-`sbmlsim.fit.petab_v2`.
+and the fit parameters a problem describes. A `Hybridization` says where a
+network sits in a problem: before the simulation, where a fit evaluates it,
+or in the right hand side or an observable of the model, where
+`compile_network` writes it into the model. The package knows nothing of
+PEtab, the translation of a PEtab SciML problem is
+`sbmlsim.fit.petab_v2.sciml`.

 The layers are implemented against the backends of `sbmlsim.sciml.backend`,
 which are the extension point of the layers and not needed to use a network.
@@ -31,20 +34,31 @@ except ModuleNotFoundError as err:
         "with the extra 'sciml': pip install sbmlsim[sciml]"
     ) from err

+from sbmlsim.sciml.compiler import compile_network, compiled_path
 from sbmlsim.sciml.errors import (
+    NetworkCompilationError,
     NetworkError,
+    NetworkHybridizationError,
     NetworkImportError,
     UnsupportedLayerError,
 )
+from sbmlsim.sciml.hybridization import Hybridization, NetworkInput, NetworkPattern
 from sbmlsim.sciml.network import Network, NetworkParameters
 from sbmlsim.sciml.parameters import network_fit_parameters, nominal_parameters

 __all__ = [
+    "Hybridization",
     "Network",
+    "NetworkCompilationError",
     "NetworkError",
+    "NetworkHybridizationError",
     "NetworkImportError",
+    "NetworkInput",
     "NetworkParameters",
+    "NetworkPattern",
     "UnsupportedLayerError",
+    "compile_network",
+    "compiled_path",
     "network_fit_parameters",
     "nominal_parameters",
 ]
```

- [ ] **Step 11: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/sciml tests/fit`
Expected: PASS

- [ ] **Step 12: Run the checks**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: `All checks passed!`, `... files already formatted`, `All checks passed!` (zero diagnostics)

- [ ] **Step 13: Commit**

```bash
git add -A
git commit -m "sciml: networks are compiled into the model as assignment rules, and the fit parameters of a network"
```

### Task 10: The reader: one fit mapping per observable and experiment, the math of an observable

**Files:**
- Modify: `src/sbmlsim/fit/petab_v2/reader.py`
- Test (create): `tests/fit/test_petab_v2_reader.py`

**Interfaces:**
- Consumes: `expression_to_formula` of task 3, `PetabReader` as it is.
- Produces: `PetabReader._measurements` keyed by the key of the fit mapping, `PetabReader._observable_ids: dict[str, str]`, `PetabReader.observable_id(key) -> str`, `PetabReader.noise_model(key)`, `PetabReader.model_source(model_id=None) -> Path`.

Two silent wrong results of the reader which the cases of the suite hit before any network is involved. The reader keyed the measurements by observable, so an observable measured in two experiments (cases 003, 015, 038, 039: `prey_o` in `e1` and `e2`) got the twenty measurements of both experiments as one dataset of the first experiment. A fit mapping is now an observable in an experiment: its key is the id of the observable when it is measured in one experiment, which keeps every existing problem as it is, and `<observable>_<experiment>` otherwise, with an error when that key is the id of another observable; `observable_id(key)` gives the observable of a mapping back and `noise_model(key)` reads the noise of the mapping. And the reader handed the text of a `sympy` expression of PEtab to the L3 parser of libsbml for a formula observable: `log(prey)` of PEtab is the natural logarithm and `log(prey)` of a formula of SBML is the decadic one, and `str(x**2.0)` is not a formula of SBML at all. The formula is now written with `expression_to_formula` of task 3. `model_source(model_id=None)` gives the model the fit simulates, which task 11 extends by the compiled model.

- [ ] **Step 1: Create `tests/fit/test_petab_v2_reader.py`**

Create `tests/fit/test_petab_v2_reader.py`:

```python
"""Tests of the reader on a small problem: its observables and experiments.

The problem is the model of Lotka and Volterra of `tests/data/models` with
tables which are written here, so that a test controls every row.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from sbmlsim.fit import FitSettings
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.fit.petab_v2.reader import PetabReader

MODEL_PATH = Path(__file__).parent.parent / "data" / "models" / "lotka_volterra.xml"

SETTINGS = FitSettings(
    parameter_scale=ParameterScaleType.LINEAR,
    variable_step_size=False,
    absolute_tolerance=1e-12,
    relative_tolerance=1e-12,
)

#: the species at the times 1 to 10, simulated with the model
PREY = [0.1996, 0.4843, 1.6064, 5.4941, 3.0782, 0.1952, 0.2965, 0.9041, 3.1091, 8.8516]


def write_problem(
    directory: Path,
    observables: dict[str, str],
    experiments: dict[str, str | None],
    conditions: list[tuple[str, str, str]] = (),  # ty: ignore[invalid-parameter-default]
) -> Path:
    """Write a PEtab problem of the model with the observables and experiments.

    Every observable is measured in every experiment at the times `1` to
    `10`, with the values of `prey`.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "lv.xml").write_bytes(MODEL_PATH.read_bytes())

    def table(name: str, rows: list[dict[str, Any]]) -> None:
        pd.DataFrame(rows).to_csv(directory / f"{name}.tsv", sep="\t", index=False)

    table(
        "observables",
        [
            {
                "observableId": sid,
                "observableFormula": formula,
                "noiseFormula": 0.05,
                "noiseDistribution": "normal",
            }
            for sid, formula in observables.items()
        ],
    )
    table(
        "measurements",
        [
            {
                "observableId": sid,
                "experimentId": experiment_id,
                "measurement": value,
                "time": float(k + 1),
            }
            for experiment_id in experiments
            for sid in observables
            for k, value in enumerate(PREY)
        ],
    )
    table(
        "experiments",
        [
            {"experimentId": sid, "time": 0.0, "conditionId": condition or ""}
            for sid, condition in experiments.items()
        ],
    )
    table(
        "parameters",
        [
            {
                "parameterId": sid,
                "lowerBound": 0.0,
                "upperBound": 15.0,
                "nominalValue": value,
                "estimate": True,
            }
            for sid, value in [("alpha", 1.3), ("beta", 0.9)]
        ],
    )
    config: dict[str, Any] = {
        "format_version": "2.0.0",
        "model_files": {"lv": {"location": "lv.xml", "language": "sbml"}},
        "measurement_files": ["measurements.tsv"],
        "observable_files": ["observables.tsv"],
        "experiment_files": ["experiments.tsv"],
        "parameter_files": ["parameters.tsv"],
    }
    if conditions:
        table(
            "conditions",
            [
                {"conditionId": condition, "targetId": target, "targetValue": value}
                for condition, target, value in conditions
            ],
        )
        config["condition_files"] = ["conditions.tsv"]
    path = directory / "problem.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_an_observable_measured_in_several_experiments(tmp_path: Path) -> None:
    """One fit mapping per observable and experiment, named after both."""
    path = write_problem(
        tmp_path,
        observables={"prey_o": "prey", "predator_o": "predator"},
        experiments={"e1": "cond1", "e2": "cond2"},
        conditions=[("cond1", "delta", "1.8"), ("cond2", "delta", "3.6")],
    )
    reader = PetabReader.from_yaml(path)
    problem = reader.to_optimization_problem()
    assert [c.sid for c in problem.mapping_collections] == ["e1", "e2"]
    assert problem.mapping_collections[0].mappings == ["prey_o_e1", "predator_o_e1"]
    assert problem.mapping_collections[1].mappings == ["prey_o_e2", "predator_o_e2"]
    assert reader.observable_id("prey_o_e2") == "prey_o"
    with pytest.raises(ValueError, match="no fit mapping 'prey_o'"):
        reader.observable_id("prey_o")

    problem.initialize(SETTINGS)
    assert problem.simulation_keys == ["e1", "e1", "e2", "e2"]
    for k in problem.indices():
        assert len(problem.y_references[k]) == 10
    x = np.asarray(problem.x0, dtype=float)
    predictions = problem.predictions(x, indices=problem.indices())
    # the experiments differ in their condition
    assert not np.allclose(predictions[0], predictions[2])
    assert len(problem.noise_models) == 4
    assert all(noise is not None for noise in problem.noise_models)


def test_an_observable_measured_in_one_experiment_keeps_its_id(
    tmp_path: Path,
) -> None:
    """A problem with one experiment per observable reads as before."""
    path = write_problem(
        tmp_path,
        observables={"prey_o": "prey", "predator_o": "predator"},
        experiments={"e1": None},
    )
    problem = PetabReader.from_yaml(path).to_optimization_problem()
    assert problem.mapping_collections[0].mappings == ["prey_o", "predator_o"]


def test_a_mapping_named_like_an_observable(tmp_path: Path) -> None:
    """The key of a mapping is not the id of another observable."""
    path = write_problem(
        tmp_path,
        observables={"prey_o": "prey", "prey_o_e1": "predator"},
        experiments={"e1": None, "e2": None},
    )
    with pytest.raises(ValueError, match="'prey_o_e1', which is the id of another"):
        PetabReader.from_yaml(path)


def test_the_math_of_an_observable_is_translated(tmp_path: Path) -> None:
    """`log` of PEtab is the natural logarithm, `log` of SBML the decadic one."""
    path = write_problem(
        tmp_path,
        observables={
            "log_prey": "log(prey)",
            "log10_prey": "log10(prey)",
            "square": "prey^2",
            "prey_o": "prey",
        },
        experiments={"e1": None},
    )
    reader = PetabReader.from_yaml(path)
    reader.derived_dir = tmp_path / "derived"
    problem = reader.to_optimization_problem()
    problem.initialize(SETTINGS)
    predictions = problem.predictions(np.asarray(problem.x0, dtype=float))
    keys = problem.mapping_keys
    prey = predictions[keys.index("prey_o")]
    np.testing.assert_allclose(predictions[keys.index("log_prey")], np.log(prey))
    np.testing.assert_allclose(predictions[keys.index("log10_prey")], np.log10(prey))
    np.testing.assert_allclose(predictions[keys.index("square")], prey**2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -x tests/fit/test_petab_v2_reader.py`
Expected: FAIL: `test_an_observable_measured_in_several_experiments` (the collections hold `prey_o` and `predator_o` only, `AttributeError: 'PetabReader' object has no attribute 'observable_id'`), `test_the_math_of_an_observable_is_translated` (`log_prey` is the decadic logarithm)

- [ ] **Step 3: Apply this patch to `src/sbmlsim/fit/petab_v2/reader.py`**

Apply this patch to `src/sbmlsim/fit/petab_v2/reader.py`:

```diff
diff --git a/src/sbmlsim/fit/petab_v2/reader.py b/src/sbmlsim/fit/petab_v2/reader.py
index baf8044..87859a9 100644
--- a/src/sbmlsim/fit/petab_v2/reader.py
+++ b/src/sbmlsim/fit/petab_v2/reader.py
@@ -54,6 +54,7 @@ from sbmlsim.fit.petab_v2.observables import (
     observable_id as petab_observable_id,
 )
 from sbmlsim.fit.petab_v2.symbols import selection_of_formula, split_selection
+from sbmlsim.mathml import expression_to_formula
 from sbmlsim.model import AbstractModel
 from sbmlsim.simulation.timecourse import Timecourse, TimecourseSim
 from sbmlsim.task import Task
@@ -159,12 +160,34 @@ class PetabReader:
             self.uinfo.ureg if self.uinfo is not None else UnitRegistry()
         )

-        # measurements by observable, in the order of their time
+        #: the measurements of every fit mapping, in the order of their time.
+        #: A fit mapping is an observable in an experiment: its key is the id
+        #: of the observable, and `<observable>_<experiment>` for an
+        #: observable which is measured in several experiments
         self._measurements: dict[str, list[petab_v2.Measurement]] = {}
+        #: the observable of every fit mapping
+        self._observable_ids: dict[str, str] = {}
+        experiments: dict[str, list[str]] = {}
         for measurement in petab_problem.measurements:
-            self._measurements.setdefault(measurement.observable_id, []).append(
-                measurement
-            )
+            experiment_ids = experiments.setdefault(measurement.observable_id, [])
+            experiment_id = measurement.experiment_id or DEFAULT_EXPERIMENT
+            if experiment_id not in experiment_ids:
+                experiment_ids.append(experiment_id)
+        for measurement in petab_problem.measurements:
+            observable_id = measurement.observable_id
+            key = observable_id
+            if len(experiments[observable_id]) > 1:
+                experiment_id = measurement.experiment_id or DEFAULT_EXPERIMENT
+                key = f"{observable_id}_{experiment_id}"
+                if key in experiments:
+                    raise ValueError(
+                        f"The observable '{observable_id}' is measured in the "
+                        f"experiments {experiments[observable_id]}, its fit "
+                        f"mapping of the experiment '{experiment_id}' is "
+                        f"'{key}', which is the id of another observable."
+                    )
+            self._measurements.setdefault(key, []).append(measurement)
+            self._observable_ids[key] = observable_id
         for measurements in self._measurements.values():
             measurements.sort(key=lambda m: m.time)

@@ -343,7 +366,10 @@ class PetabReader:
         """Get the observables which are a formula and not an entity."""
         sbml_model = self._sbml_model()
         return {
-            observable.id: str(observable.formula)
+            # the math of PEtab is not the math of a formula of SBML: `log`
+            # is the natural logarithm in the one and the logarithm to the
+            # base 10 in the other
+            observable.id: expression_to_formula(observable.formula)
             for observable in self.petab_problem.observables
             if not is_entity(str(observable.formula), sbml_model)
         }
@@ -362,14 +388,37 @@ class PetabReader:
             The path of the model the fit simulates.
         """
         path = self._model_path(model)
+        if path in self._model_sources:
+            return self._model_sources[path]
+        derived_dir = self.derived_dir or path.parent
+        source = path
         formulas = self._formula_observables()
-        if not formulas:
-            return path
-        if path not in self._model_sources:
-            derived_dir = self.derived_dir or path.parent
-            derived = derived_dir / f"{path.stem}{MODEL_SUFFIX}{path.suffix}"
-            self._model_sources[path] = add_observables(path, formulas, derived)
-        return self._model_sources[path]
+        if formulas:
+            derived = derived_dir / f"{source.stem}{MODEL_SUFFIX}{source.suffix}"
+            source = add_observables(source, formulas, derived)
+        self._model_sources[path] = source
+        return source
+
+    def model_source(self, model_id: str | None = None) -> Path:
+        """Get the file of the model the fit simulates.
+
+        Args:
+            model_id: id of the model, the first model of the problem by
+                default.
+
+        Returns:
+            The path of the model, see `_model_source`.
+
+        Raises:
+            ValueError: if the problem has no model of the id.
+        """
+        for model in self.petab_problem.models:
+            if model_id is None or model.model_id == model_id:
+                return self._model_source(model)
+        raise ValueError(
+            f"The problem has no model '{model_id}', its models are "
+            f"{[m.model_id for m in self.petab_problem.models]}."
+        )

     @property
     def experiment_ids(self) -> list[str]:
@@ -603,9 +652,10 @@ class PetabReader:
         return None

     def datasets(self) -> dict[str, DataSet]:
-        """Get the datasets, one per observable, from its measurements."""
+        """Get the datasets, one per fit mapping, from its measurements."""
         datasets: dict[str, DataSet] = {}
-        for observable_id, measurements in self._measurements.items():
+        for key, measurements in self._measurements.items():
+            observable_id = self._observable_ids[key]
             info = self._observable_info(observable_id)
             # without the extension the data is in the units of the model,
             # which is what PEtab measures in
@@ -632,22 +682,25 @@ class PetabReader:
                 data["value_sd"] = errors
                 data["value_sd_unit"] = value_unit

-            datasets[dataset_id(observable_id)] = DataSet.from_df(
+            datasets[dataset_id(key)] = DataSet.from_df(
                 pd.DataFrame(data), ureg=self.ureg
             )
         return datasets

     def fit_mappings(self, experiment: SimulationExperiment) -> dict[str, FitMapping]:
-        """Get the fit mappings, one per observable of the problem.
+        """Get the fit mappings, one per observable and experiment.

         Args:
             experiment: experiment the mappings belong to.

         Returns:
-            The mappings by the id of their observable.
+            The mappings by their key, which is the id of their observable,
+            and `<observable>_<experiment>` for an observable which is
+            measured in several experiments.
         """
         mappings: dict[str, FitMapping] = {}
-        for observable_id, measurements in self._measurements.items():
+        for key, measurements in self._measurements.items():
+            observable_id = self._observable_ids[key]
             info = self._observable_info(observable_id)
             experiment_id = measurements[0].experiment_id or DEFAULT_EXPERIMENT
             task_id = f"task_{experiment_id}"
@@ -662,9 +715,9 @@ class PetabReader:
                 xid="time",
                 yid="value",
                 yid_sd="value_sd"
-                if "value_sd" in experiment._datasets[dataset_id(observable_id)].columns
+                if "value_sd" in experiment._datasets[dataset_id(key)].columns
                 else None,
-                dataset=dataset_id(observable_id),
+                dataset=dataset_id(key),
             )
             observable = FitData(
                 experiment,
@@ -672,36 +725,59 @@ class PetabReader:
                 yid=observable_yid,
                 task=task_id,
             )
-            mappings[observable_id] = FitMapping(
+            mappings[key] = FitMapping(
                 experiment,
                 reference=reference,
                 observable=observable,
                 weight=info.get("weight_mapping", 1.0),
-                noise=self.noise_model(observable_id),
+                noise=self.noise_model(key),
             )
         return mappings

-    def noise_model(self, observable_id: str) -> NoiseModel:
-        """Get the noise model of an observable of the problem.
+    def observable_id(self, key: str) -> str:
+        """Get the observable of a fit mapping of the problem.
+
+        Args:
+            key: key of the fit mapping, which is the id of its observable,
+                and `<observable>_<experiment>` for an observable which is
+                measured in several experiments.
+
+        Returns:
+            The id of the observable.
+
+        Raises:
+            ValueError: if the problem has no fit mapping of the key.
+        """
+        if key not in self._observable_ids:
+            raise ValueError(
+                f"The problem has no fit mapping '{key}', its fit mappings are "
+                f"{sorted(self._observable_ids)}."
+            )
+        return self._observable_ids[key]
+
+    def noise_model(self, key: str) -> NoiseModel:
+        """Get the noise model of a fit mapping of the problem.

         The noise model is read once, see `_read_noise_model`.

         Args:
-            observable_id: id of the observable.
+            key: key of the fit mapping, which is the id of its observable,
+                and `<observable>_<experiment>` for an observable which is
+                measured in several experiments.

         Returns:
-            The noise model of the fit mapping of the observable.
+            The noise model of the fit mapping.

         Raises:
-            ValueError: if the problem has no observable of the id, or if a
+            ValueError: if the problem has no fit mapping of the key, or if a
                 measurement does not have a value for every placeholder.
         """
-        if observable_id not in self._noise_models:
-            self._noise_models[observable_id] = self._read_noise_model(observable_id)
-        return self._noise_models[observable_id]
+        if key not in self._noise_models:
+            self._noise_models[key] = self._read_noise_model(key)
+        return self._noise_models[key]

-    def _read_noise_model(self, observable_id: str) -> NoiseModel:
-        """Read the noise model of an observable of the problem.
+    def _read_noise_model(self, key: str) -> NoiseModel:
+        """Read the noise model of a fit mapping of the problem.

         The noise formula and the distribution of the observable are kept as
         they are, with the noise parameters of its measurements as the values
@@ -711,15 +787,16 @@ class PetabReader:
         which PEtab estimates is one of them, see the `noise-parameters` gap.

         Args:
-            observable_id: id of the observable.
+            key: key of the fit mapping, see `noise_model`.

         Returns:
-            The noise model of the fit mapping of the observable.
+            The noise model of the fit mapping.

         Raises:
-            ValueError: if the problem has no observable of the id, or if a
+            ValueError: if the problem has no fit mapping of the key, or if a
                 measurement does not have a value for every placeholder.
         """
+        observable_id = self._observable_ids.get(key, key)
         if observable_id not in self._observables:
             raise ValueError(f"The problem has no observable '{observable_id}'.")
         observable = self._observables[observable_id]
@@ -728,7 +805,7 @@ class PetabReader:
         expressions: list[Any] = [observable.noise_formula]
         rows: list[tuple[float | str, ...]] = []
         if placeholders:
-            for measurement in self._measurements.get(observable_id, []):
+            for measurement in self._measurements.get(key, []):
                 expressions.extend(measurement.noise_parameters)
                 rows.append(
                     tuple(_formula(value) for value in measurement.noise_parameters)
@@ -875,16 +952,15 @@ class PetabReader:
     def _mapping_keys_of_condition(self, condition_id: str) -> set[str]:
         """Get the fit mappings of the experiments which use a condition.

-        A fit mapping of a problem which is read is named after its observable
-        and belongs to the experiment of its measurements, so the mappings of a
-        condition are the observables measured in the experiments which
-        reference it.
+        A fit mapping of a problem which is read is an observable in the
+        experiment of its measurements, so the mappings of a condition are the
+        ones of the experiments which reference it.

         Args:
             condition_id: id of the condition.

         Returns:
-            The ids of the fit mappings, which are the ids of the observables.
+            The keys of the fit mappings.
         """
         experiments = {
             experiment.id
@@ -894,8 +970,8 @@ class PetabReader:
             )
         }
         return {
-            observable_id
-            for observable_id, measurements in self._measurements.items()
+            key
+            for key, measurements in self._measurements.items()
             if (measurements[0].experiment_id or DEFAULT_EXPERIMENT) in experiments
         }

@@ -1031,29 +1107,29 @@ class PetabReader:
             The collections, in the order of the experiments of the problem.
         """
         by_experiment: dict[str, list[str]] = {}
-        for observable_id, measurements in self._measurements.items():
+        for key, measurements in self._measurements.items():
             experiment_id = measurements[0].experiment_id or DEFAULT_EXPERIMENT
-            by_experiment.setdefault(experiment_id, []).append(observable_id)
+            by_experiment.setdefault(experiment_id, []).append(key)

         collections: list[FitMappingCollection] = []
         for experiment_id, mappings in by_experiment.items():
             kinds = {
                 MappingKind(
-                    self._observable_info(observable_id).get(
+                    self._observable_info(self._observable_ids[key]).get(
                         "kind", MappingKind.TRAINING.value
                     )
                 )
-                for observable_id in mappings
+                for key in mappings
             }
             if len(kinds) > 1:
                 # an experiment whose observables are used differently is one
                 # collection per kind, the kind belongs to the selection
                 for kind in sorted(kinds):
                     selected = [
-                        observable_id
-                        for observable_id in mappings
+                        key
+                        for key in mappings
                         if MappingKind(
-                            self._observable_info(observable_id).get(
+                            self._observable_info(self._observable_ids[key]).get(
                                 "kind", MappingKind.TRAINING.value
                             )
                         )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/fit`
Expected: PASS

- [ ] **Step 5: Run the checks**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: `All checks passed!`, `... files already formatted`, `All checks passed!` (zero diagnostics)

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "petab: one fit mapping per observable and experiment, the math of an observable is the math of SBML"
```

### Task 11: The reader of PEtab SciML problems, the extension `sciml` and the gaps

**Files:**
- Modify: `src/sbmlsim/fit/petab_v2/extension.py`
- Modify: `src/sbmlsim/fit/petab_v2/gaps.py`
- Modify: `src/sbmlsim/fit/petab_v2/reader.py`
- Create: `src/sbmlsim/fit/petab_v2/sciml.py`
- Test: `tests/fit/test_petab_v2_noise.py`
- Test (create): `tests/sciml/petab.py`
- Test: `tests/sciml/test_package.py`
- Test (create): `tests/sciml/test_reader.py`

**Interfaces:**
- Consumes: `Network`, `Hybridization`, `NetworkInput`, `NetworkPattern`, `ALL_CONDITIONS`, `input_shapes`, `output_shapes`, `compile_network`, `compiled_path`, `network_fit_parameters`, `nominal_parameters`, `ELEMENT_UNIT` of tasks 7 and 9; `expression_to_formula`, `formula_symbols` of task 3; `EXTERNAL_PREFIX`, `FitParameter.scale`; `PetabReader` of task 10; `petab.v2.extensions.sciml.SciMLConfig`, `HybridizationTable`, `petab_sciml.constants.ALL_CONDITION_IDS`, `ARRAY`.
- Produces: `sbmlsim.fit.petab_v2.extension.SCIML_EXTENSION_ID`, `SCIML_EXTRA`, `sciml_installed()`, `known_extensions()`, `check_extensions(extensions, known=None)`; `sbmlsim.fit.petab_v2.sciml.SciMLProblemError(message, gap=None)`, `NetworkEntity`, `parse_entity(petab_id, model_entity_id)`, `read_sciml_config(extensions)`, `SciMLReader(petab_problem, config, base_path, simulations)` with `.networks`, `.entities`, `.inputs`, `.parameter_ids`, `.input_ids`, `.condition_ids`, `.hybridizations()`, `.network_parameters(sid)`, `.fit_parameters(unit_of)`, `parameter_scale(parameter)`; `PetabReader(..., sciml=None)`, `PetabReader.sciml`, `PetabReader.from_yaml` reading a SciML problem; the gaps `sciml-model-format`, `sciml-layer-sbml`, `sciml-training-mode`, `sciml-priors`, `sciml-parameter-scale` and `EVALUATION_MODE_LAYERS`. `tests/sciml/petab.py`: `write_problem(directory, networks, pre_initialization, mapping, hybridization, ...)`, `write_arrays`, `write_table`.

`sbmlsim.fit.petab_v2.sciml` translates the extension into the objects of `sbmlsim.sciml`; it is the only module of `sbmlsim.fit` which imports `sbmlsim.sciml`, and the reader imports it inside the methods which need it, so `import sbmlsim.fit` and every module of it work without the extra (`test_the_package_does_not_import_the_networks` pins it). `petab.v2.Problem.from_yaml` of `petab` 0.9.0 reads the networks of a SciML problem through PyTorch, which is no dependency: `PetabReader.from_yaml` checks the extensions, pops the `sciml` block, lets `petab` read the tables without it and hands the block to `SciMLReader`, which reads the NN YAML, the array files and the hybridization tables with `petab_sciml` and `petab.v2.extensions.sciml`; a problem which `petab` read with torch is read from its configuration. The translation: the rows of the mapping table are parsed into the parts of a network (`parse_entity`), the rows of the parameter table of a network go through `nominal_parameters` and `network_fit_parameters` (external for a network before the simulation), a prior on them is the gap `sciml-priors` and a format other than `YAML` the gap `sciml-model-format` (both raise a `SciMLProblemError` which carries the gap), an input is the row of the hybridization table, or the changes of the conditions as formulas of the simulations which start with them, or the parameter of the parameter table of its id (a constant when it is not estimated, an external fit parameter with the scale of the `parameterScale` column when it is), an array is selected by the condition the simulation starts with, and the outputs used by an observable are `OBSERVABLE` while the ones the hybridization table assigns are `RHS` or `PRE_INITIALIZATION`; the index of an output of PEtab names the element of one sample, so it is padded with zeros for leading axes of length one (`outputs[0][0]` of the `(1, 1)` output of the convolution of case 014). The reader compiles the networks of the right hand side and the observables into `<stem>_sciml.xml` in `derived_dir` before `observables.py` runs on it, skips the conditions of the array files and the changes which set the inputs of networks, reads the `parameterScale` column of a SciML problem for every parameter, and hands the hybridizations to the problem. `known_extensions()` adds `sciml` when `petab_sciml` is installed and `check_extensions` raises an `ImportError` naming `pip install sbmlsim[sciml]` for a required `sciml` block without it. `gaps.py` gets the five gaps of the spec, `gaps_of_problem` hits `sciml-parameter-scale` for a parameter with a scale and `sciml-training-mode` for a network with dropout or normalization layers, without importing `sbmlsim.sciml`.

- [ ] **Step 1: Create `tests/sciml/petab.py`**

Create `tests/sciml/petab.py`:

```python
"""A writer of small PEtab SciML problems for the tests of the reader."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml
from petab_sciml import NNModelStandard

from sbmlsim.sciml import Network
from tests.sciml.experiment import DATA
from tests.sciml.hybrid import MODEL_PATH

#: the parameters of the model of the parameter table, `lin` is the scale
#: the problems of the test suite carry
MECHANISTIC = [("alpha", 1.3), ("beta", 0.9), ("delta", 1.8)]


def write_table(
    path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] = ()
) -> Path:
    """Write the rows as a TSV table, with the columns for a table without rows."""
    df = pd.DataFrame(list(rows)) if rows else pd.DataFrame(columns=list(columns))
    df.to_csv(path, sep="\t", index=False)
    return path


def write_arrays(
    path: Path,
    networks: Sequence[Network] = (),
    inputs: Mapping[str, Mapping[str, np.ndarray]] | None = None,
) -> Path:
    """Write an array file with the arrays of networks and inputs.

    Args:
        path: the HDF5 file.
        networks: the networks whose arrays are written.
        inputs: id of the input -> id of the condition -> array.

    Returns:
        The path.
    """
    with h5py.File(path, "w") as f:
        f.create_group("metadata")["pytorch_format"] = True
        for network in networks:
            for layer, arrays in network.parameters.items():
                for name, array in arrays.items():
                    f[f"parameters/{network.sid}/{layer}/{name}"] = np.asarray(array)
        for input_id, conditions in (inputs or {}).items():
            for condition, array in conditions.items():
                f[f"inputs/{input_id}/{condition}"] = np.asarray(array, dtype=float)
    return path


def write_problem(
    directory: Path,
    networks: Sequence[Network],
    pre_initialization: Mapping[str, bool],
    mapping: Sequence[tuple[str, str]],
    hybridization: Sequence[tuple[str, str]],
    parameters: Sequence[Mapping[str, Any]] = (),
    observables: Mapping[str, str] | None = None,
    conditions: Sequence[tuple[str, str, str]] = (),
    experiments: Mapping[str, str | None] | None = None,
    inputs: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    formats: Mapping[str, str] | None = None,
    extra_yaml: Mapping[str, Any] | None = None,
) -> Path:
    """Write a PEtab SciML problem with the model of Lotka and Volterra.

    Args:
        directory: the directory the files are written into.
        networks: the networks of the problem, with their nominal values.
        pre_initialization: id of the network -> whether it runs before the
            simulation.
        mapping: the rows of the mapping table, `petabEntityId` and
            `modelEntityId`.
        hybridization: the rows of the hybridization table, `targetId` and
            `targetValue`.
        parameters: the rows of the parameter table in addition to the ones of
            `alpha`, `beta` and `delta` and of the arrays of the networks.
        observables: id of the observable -> its formula, `prey` and
            `predator` by default.
        conditions: the rows of the condition table, `conditionId`,
            `targetId` and `targetValue`.
        experiments: id of the experiment -> id of its condition, one
            experiment `e1` without a condition by default.
        inputs: the arrays of the inputs, see `write_arrays`.
        formats: id of the network -> its format, `YAML` by default.
        extra_yaml: entries added to the YAML of the problem.

    Returns:
        The path of the YAML of the problem.
    """
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "lv.xml"
    model_path.write_bytes(MODEL_PATH.read_bytes())
    for network in networks:
        NNModelStandard.save_data(
            data=network.model, filename=str(directory / f"{network.sid}.yaml")
        )
    array_files = [write_arrays(directory / "arrays.hdf5", networks, inputs).name]

    observables = observables or {"prey_o": "prey", "predator_o": "predator"}
    experiments = experiments or {"e1": None}
    write_table(
        directory / "observables.tsv",
        [
            {
                "observableId": sid,
                "observableFormula": formula,
                "noiseFormula": 0.05,
                "noiseDistribution": "normal",
            }
            for sid, formula in observables.items()
        ],
    )
    measurements = []
    for experiment_id in experiments:
        for sid, formula in observables.items():
            species = (
                "prey" if "prey" in formula or sid.startswith("prey") else "predator"
            )
            for k, value in enumerate(DATA[species]):
                measurements.append(
                    {
                        "observableId": sid,
                        "experimentId": experiment_id,
                        "measurement": value,
                        "time": float(k + 1),
                    }
                )
    write_table(directory / "measurements.tsv", measurements)
    write_table(
        directory / "experiments.tsv",
        [
            {"experimentId": sid, "time": 0.0, "conditionId": condition or ""}
            for sid, condition in experiments.items()
        ],
    )
    rows: list[dict[str, Any]] = [
        {
            "parameterId": sid,
            "parameterScale": "lin",
            "lowerBound": 0.0,
            "upperBound": 15.0,
            "nominalValue": value,
            "estimate": True,
        }
        for sid, value in MECHANISTIC
    ]
    rows.extend(
        {
            "parameterId": f"{network.sid}_ps",
            "parameterScale": "lin",
            "lowerBound": "-inf",
            "upperBound": "inf",
            "nominalValue": "array",
            "estimate": True,
        }
        for network in networks
    )
    rows.extend(dict(row) for row in parameters)
    write_table(directory / "parameters.tsv", rows)
    write_table(
        directory / "mapping.tsv",
        [
            {"petabEntityId": petab_id, "modelEntityId": model_id}
            for petab_id, model_id in [
                *mapping,
                *((f"{n.sid}_ps", f"{n.sid}.parameters") for n in networks),
            ]
        ],
    )
    write_table(
        directory / "hybridization.tsv",
        [{"targetId": target, "targetValue": value} for target, value in hybridization],
        columns=["targetId", "targetValue"],
    )
    config: dict[str, Any] = {
        "format_version": "2.0.0",
        "model_files": {"lv": {"location": "lv.xml", "language": "sbml"}},
        "measurement_files": ["measurements.tsv"],
        "observable_files": ["observables.tsv"],
        "experiment_files": ["experiments.tsv"],
        "parameter_files": ["parameters.tsv"],
        "mapping_files": ["mapping.tsv"],
        "extensions": {
            "sciml": {
                "version": "0.1.0",
                "required": True,
                "array_files": array_files,
                "hybridization_files": ["hybridization.tsv"],
                "neural_networks": {
                    network.sid: {
                        "location": f"{network.sid}.yaml",
                        "pre_initialization": bool(pre_initialization[network.sid]),
                        "format": (formats or {}).get(network.sid, "YAML"),
                    }
                    for network in networks
                },
            }
        },
    }
    if conditions:
        write_table(
            directory / "conditions.tsv",
            [
                {"conditionId": condition, "targetId": target, "targetValue": value}
                for condition, target, value in conditions
            ],
        )
        config["condition_files"] = ["conditions.tsv"]
    config.update(extra_yaml or {})
    path = directory / "problem.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path
```

- [ ] **Step 2: Create `tests/sciml/test_reader.py`**

Create `tests/sciml/test_reader.py`:

```python
"""Tests of the reader of PEtab SciML problems."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from petab_sciml import Layer, Node
from petab_sciml.constants import ALL_CONDITION_IDS

from sbmlsim.fit import FitSettings
from sbmlsim.fit.objects import EXTERNAL_PREFIX
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.fit.petab_v2 import GapKind, from_petab, gaps_of_problem
from sbmlsim.fit.petab_v2.gaps import GAPS_BY_ID
from sbmlsim.fit.petab_v2.likelihood import gradient, log_likelihood
from sbmlsim.fit.petab_v2.reader import PetabReader
from sbmlsim.fit.petab_v2.sciml import (
    NetworkEntity,
    SciMLProblemError,
    parse_entity,
)
from sbmlsim.sciml import (
    Network,
    NetworkHybridizationError,
    NetworkInput,
    NetworkPattern,
)
from sbmlsim.sciml.hybridization import ALL_CONDITIONS
from tests.sciml.hybrid import convolution, feed_forward, two_inputs
from tests.sciml.petab import write_problem

PRE = NetworkPattern.PRE_INITIALIZATION
RHS = NetworkPattern.RHS
OBSERVABLE = NetworkPattern.OBSERVABLE

SETTINGS = FitSettings(
    parameter_scale=ParameterScaleType.LINEAR,
    variable_step_size=False,
    absolute_tolerance=1e-12,
    relative_tolerance=1e-12,
)

#: the mapping and the hybridization of a network with two inputs and one
#: output which are species
INPUTS = [("net1_input1", "net1.inputs[0][0]"), ("net1_input2", "net1.inputs[0][1]")]
OUTPUT = [("net1_output1", "net1.outputs[0][0]")]
SPECIES = [("net1_input1", "prey"), ("net1_input2", "predator")]


def _read(path: Path) -> PetabReader:
    reader = PetabReader.from_yaml(path)
    reader.derived_dir = path.parent / "derived"
    return reader


# --- THE MAPPING TABLE ---


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("net1.inputs[0][1]", NetworkEntity("x", "net1", "inputs", 0, (1,))),
        ("net1.inputs[0][1][2]", NetworkEntity("x", "net1", "inputs", 0, (1, 2))),
        ("net1.inputs[2]", NetworkEntity("x", "net1", "inputs", 2, None)),
        ("net1.outputs[0][0]", NetworkEntity("x", "net1", "outputs", 0, (0,))),
        ("net1.parameters", NetworkEntity("x", "net1", "parameters", key="net1")),
        (
            "net1.parameters[layer1]",
            NetworkEntity("x", "net1", "parameters", key="net1.layer1"),
        ),
        (
            "net1.parameters[block.0].weight",
            NetworkEntity("x", "net1", "parameters", key="net1.block.0.weight"),
        ),
        ("prey", None),
        ("compartment.default", None),
    ],
)
def test_the_parts_of_a_network_of_the_mapping_table(
    model_id: str, expected: NetworkEntity | None
) -> None:
    """A `modelEntityId` names an input, an output or the parameters."""
    assert parse_entity("x", model_id) == expected


@pytest.mark.parametrize(
    "model_id",
    [
        "net1.inputs",
        "net1.inputs[a]",
        "net1.inputs[0]x",
        "net1.outputs[0]",
        "net1.parameters.weight",
        "net1.parameters[layer1]weight",
        "net1.parameters[]",
    ],
)
def test_a_part_which_is_not_one(model_id: str) -> None:
    """A row which names a network and no part of it is an error."""
    with pytest.raises(
        SciMLProblemError,
        match=f"'x' to '{model_id}'".replace("[", r"\[").replace("]", r"\]"),
    ):
        parse_entity("x", model_id)


# --- THE PATTERNS ---


def test_a_network_in_the_right_hand_side(tmp_path: Path) -> None:
    """The network is compiled into the model the fit simulates."""
    network = feed_forward()
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net1": False},
        mapping=[*INPUTS, *OUTPUT],
        hybridization=[*SPECIES, ("gamma", "net1_output1")],
    )
    reader = _read(path)
    assert reader.sciml is not None
    assert reader.sciml.networks["net1"] == network
    (hybridization,) = reader.sciml.hybridizations()
    assert hybridization.pattern is RHS
    assert hybridization.model == "lv"
    assert hybridization.inputs == {
        "net1__input0__0": NetworkInput(formula="prey"),
        "net1__input0__1": NetworkInput(formula="predator"),
    }
    assert hybridization.outputs == {"net1__output0__0": "gamma"}
    assert hybridization.frozen == frozenset()
    assert hybridization.constants == {}

    problem = reader.to_optimization_problem()
    assert problem.hybridizations == [hybridization]
    elements = [p for p in problem.parameters if p.pid.startswith("net1__")]
    assert len(elements) == len(network.parameter_ids())
    assert all(not p.is_external for p in elements)
    assert all(p.scale is ParameterScaleType.LINEAR for p in elements)
    assert all(p.unit == "dimensionless" for p in elements)
    assert [p.pid for p in problem.parameters[:3]] == ["alpha", "beta", "delta"]
    assert all(p.scale is ParameterScaleType.LINEAR for p in problem.parameters[:3])

    source = reader.model_source()
    assert source.name == "lv_sciml.xml"
    assert source.parent == tmp_path / "derived"
    problem.initialize(SETTINGS)
    assert np.isfinite(log_likelihood(problem))
    grad = gradient(problem)
    assert np.all(np.isfinite(grad.to_numpy()))
    assert np.all(grad[[p.pid for p in elements]].abs() > 0.0)


def test_a_network_before_the_simulation(tmp_path: Path) -> None:
    """The inputs are parameters of the table, the elements are external."""
    network = feed_forward()
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net1": True},
        mapping=[("k1", "net1.inputs[0][0]"), ("k2", "net1.inputs[0][1]"), *OUTPUT],
        hybridization=[("gamma", "net1_output1")],
        parameters=[
            {"parameterId": "k1", "nominalValue": 1.0, "estimate": False},
            {
                "parameterId": "k2",
                "nominalValue": 2.0,
                "estimate": True,
                "lowerBound": 0.1,
                "upperBound": 10.0,
                "parameterScale": "log10",
            },
        ],
    )
    reader = _read(path)
    assert reader.sciml is not None
    (hybridization,) = reader.sciml.hybridizations()
    assert hybridization.pattern is PRE
    assert hybridization.inputs == {
        "net1__input0__0": NetworkInput(formula="k1"),
        "net1__input0__1": NetworkInput(formula="k2"),
    }
    # the parameter which is not estimated is a constant, the estimated one
    # is a parameter of the fit which is not an entity of the model
    assert hybridization.constants == {"k1": 1.0}
    problem = reader.to_optimization_problem()
    by_id = {p.pid: p for p in problem.parameters}
    assert by_id["k2"].target == f"{EXTERNAL_PREFIX}k2"
    assert by_id["k2"].scale is ParameterScaleType.LOG10
    assert by_id["k2"].start_value == 2.0
    assert by_id["k2"].unit == "dimensionless"
    elements = [p for p in problem.parameters if p.pid.startswith("net1__")]
    assert all(p.target == f"{EXTERNAL_PREFIX}{p.pid}" for p in elements)
    assert reader.model_source() == tmp_path / "lv.xml"

    problem.initialize(SETTINGS)
    x = np.asarray(problem.x0, dtype=float)
    problem.predictions(x)
    changes = problem.simulations[0].timecourses[0].changes
    (expected,) = network.forward(np.array([1.0, 2.0]))
    assert changes["gamma"].magnitude == pytest.approx(expected[0])


def test_a_network_in_an_observable(tmp_path: Path) -> None:
    """The output is the symbol of the observable formula."""
    network = feed_forward()
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net1": False},
        mapping=[*INPUTS, *OUTPUT],
        hybridization=SPECIES,
        observables={"prey_o": "net1_output1 - 0.9 + prey", "predator_o": "predator"},
    )
    reader = _read(path)
    assert reader.sciml is not None
    (hybridization,) = reader.sciml.hybridizations()
    assert hybridization.pattern is OBSERVABLE
    assert hybridization.outputs == {"net1__output0__0": "net1_output1"}
    assert reader.model_source().name == "lv_sciml_observables.xml"
    problem = reader.to_optimization_problem()
    problem.initialize(SETTINGS)
    k = problem.mapping_keys.index("prey_o")
    assert problem.yid_observable[k] == "observable_prey_o"
    predictions = problem.predictions(np.asarray(problem.x0, dtype=float))
    assert np.all(np.isfinite(predictions[k]))
    assert not np.allclose(
        predictions[k], predictions[problem.mapping_keys.index("predator_o")]
    )


def test_a_network_with_outputs_of_two_patterns(tmp_path: Path) -> None:
    """A network in the right hand side and an observable is two hybridizations."""
    network = feed_forward(n_outputs=2)
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net1": False},
        mapping=[
            *INPUTS,
            ("net1_output1", "net1.outputs[0][0]"),
            ("net1_output2", "net1.outputs[0][1]"),
        ],
        hybridization=[*SPECIES, ("gamma", "net1_output2")],
        observables={"prey_o": "net1_output1", "predator_o": "predator"},
    )
    reader = _read(path)
    assert reader.sciml is not None
    observable, rhs = sorted(reader.sciml.hybridizations(), key=lambda h: h.pattern)
    assert rhs.pattern is RHS
    assert rhs.outputs == {"net1__output0__1": "gamma"}
    assert observable.pattern is OBSERVABLE
    assert observable.outputs == {"net1__output0__0": "net1_output1"}
    problem = reader.to_optimization_problem()
    problem.initialize(SETTINGS)
    assert np.isfinite(log_likelihood(problem))


def test_a_frozen_layer(tmp_path: Path) -> None:
    """A row of a layer which is not estimated freezes its elements."""
    network = feed_forward()
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net1": False},
        mapping=[*INPUTS, *OUTPUT, ("net1_layer1", "net1.parameters[layer1]")],
        hybridization=[*SPECIES, ("gamma", "net1_output1")],
        parameters=[
            {"parameterId": "net1_layer1", "nominalValue": "array", "estimate": False}
        ],
    )
    reader = _read(path)
    assert reader.sciml is not None
    (hybridization,) = reader.sciml.hybridizations()
    assert hybridization.frozen == {
        sid for sid in network.parameter_ids() if "layer1" in sid
    }
    problem = reader.to_optimization_problem()
    assert not [p for p in problem.parameters if "layer1" in p.pid]
    assert [p for p in problem.parameters if "layer2" in p.pid]


def test_the_nominal_values_of_the_parameter_table(tmp_path: Path) -> None:
    """A number in the parameter table sets the elements it covers."""
    network = feed_forward()
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net1": True},
        mapping=[
            ("k1", "net1.inputs[0][0]"),
            ("k2", "net1.inputs[0][1]"),
            *OUTPUT,
            ("net1_layer2_bias", "net1.parameters[layer2].bias"),
        ],
        hybridization=[("gamma", "net1_output1")],
        parameters=[
            {"parameterId": "k1", "nominalValue": 1.0, "estimate": False},
            {"parameterId": "k2", "nominalValue": 2.0, "estimate": False},
            {
                "parameterId": "net1_layer2_bias",
                "nominalValue": 0.25,
                "estimate": True,
                "lowerBound": -1.0,
                "upperBound": 1.0,
            },
        ],
    )
    reader = _read(path)
    assert reader.sciml is not None
    read = reader.sciml.networks["net1"]
    np.testing.assert_array_equal(read.parameters["layer2"]["bias"], [0.25])
    np.testing.assert_array_equal(
        read.parameters["layer1"]["weight"], network.parameters["layer1"]["weight"]
    )
    problem = reader.to_optimization_problem()
    by_id = {p.pid: p for p in problem.parameters}
    assert by_id["net1__layer2__bias__0"].start_value == 0.25


# --- THE CONDITIONS ---


def test_the_inputs_of_the_conditions(tmp_path: Path) -> None:
    """A condition sets an input, a number or a parameter of the table."""
    network = feed_forward()
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net1": True},
        mapping=[*INPUTS, *OUTPUT],
        hybridization=[("gamma", "net1_output1")],
        parameters=[
            {"parameterId": "k1", "nominalValue": 1.0, "estimate": False},
            {"parameterId": "k2", "nominalValue": 2.0, "estimate": False},
        ],
        conditions=[
            ("cond1", "net1_input1", "10.0"),
            ("cond1", "net1_input2", "20.0"),
            ("cond2", "net1_input1", "k1"),
            ("cond2", "net1_input2", "k2"),
        ],
        experiments={"e1": "cond1", "e2": "cond2"},
    )
    reader = _read(path)
    assert reader.sciml is not None
    (hybridization,) = reader.sciml.hybridizations()
    assert hybridization.inputs == {
        "net1__input0__0": NetworkInput(formulas={"e1": "10", "e2": "k1"}),
        "net1__input0__1": NetworkInput(formulas={"e1": "20", "e2": "k2"}),
    }
    assert hybridization.constants == {"k1": 1.0, "k2": 2.0}
    problem = reader.to_optimization_problem()
    # one fit mapping per observable and experiment
    assert sorted(
        problem.mapping_collections[0].mappings
        + problem.mapping_collections[1].mappings
    ) == [
        "predator_o_e1",
        "predator_o_e2",
        "prey_o_e1",
        "prey_o_e2",
    ]
    problem.initialize(SETTINGS)
    assert problem.simulation_keys == ["e1", "e1", "e2", "e2"]
    problem.predictions(np.asarray(problem.x0, dtype=float))
    for sid, inputs in (("e1", [10.0, 20.0]), ("e2", [1.0, 2.0])):
        k = problem.simulation_keys.index(sid)
        changes = problem.simulations[k].timecourses[0].changes
        (expected,) = network.forward(np.array(inputs))
        assert changes["gamma"].magnitude == pytest.approx(expected[0])
        # the conditions set no change of the model
        assert "net1_input1" not in changes


def test_the_arrays_of_the_conditions(tmp_path: Path) -> None:
    """An array file holds the array of every condition, or of all of them."""
    network = convolution()
    arrays = {"cond1": np.ones((1, 4, 4)), "cond2": 2.0 * np.ones((1, 4, 4))}
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net3": True},
        mapping=[("input0", "net3.inputs[0]"), ("net3_output1", "net3.outputs[0][0]")],
        hybridization=[("input0", "array"), ("gamma", "net3_output1")],
        inputs={"input0": arrays},
        experiments={"e1": "cond1", "e2": "cond2"},
    )
    reader = _read(path)
    assert reader.sciml is not None
    assert reader.sciml.condition_ids == {"cond1", "cond2"}
    (hybridization,) = reader.sciml.hybridizations()
    assert hybridization.inputs == {
        "net3__input0": NetworkInput(
            arrays={"e1": arrays["cond1"], "e2": arrays["cond2"]}
        )
    }
    # the output of the convolution has the axis of the batch
    assert hybridization.outputs == {"net3__output0__0": "gamma"}
    problem = reader.to_optimization_problem()
    problem.initialize(SETTINGS)
    problem.predictions(np.asarray(problem.x0, dtype=float))
    k = problem.simulation_keys.index("e2")
    (expected,) = network.forward(arrays["cond2"])
    assert problem.simulations[k].timecourses[0].changes["gamma"].magnitude == (
        pytest.approx(expected[0])
    )

    # one array for every condition
    path = write_problem(
        tmp_path / "all",
        networks=[network],
        pre_initialization={"net3": True},
        mapping=[("input0", "net3.inputs[0]"), ("net3_output1", "net3.outputs[0][0]")],
        hybridization=[("input0", "array"), ("gamma", "net3_output1")],
        inputs={"input0": {ALL_CONDITION_IDS: arrays["cond1"]}},
    )
    reader = _read(path)
    assert reader.sciml is not None
    (hybridization,) = reader.sciml.hybridizations()
    assert hybridization.inputs == {
        "net3__input0": NetworkInput(arrays={ALL_CONDITIONS: arrays["cond1"]})
    }


def test_an_array_of_a_network_in_the_right_hand_side(tmp_path: Path) -> None:
    """The arrays of the conditions are changes of the compiled model."""
    network = two_inputs()
    arrays = {"cond1": np.array([1.0, 2.0, 3.0]), "cond2": np.array([3.0, 2.0, 1.0])}
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net6": False},
        mapping=[
            ("net6_input1", "net6.inputs[0][0]"),
            ("net6_input2", "net6.inputs[1]"),
            ("net6_output1", "net6.outputs[0][0]"),
        ],
        hybridization=[
            ("net6_input1", "prey"),
            ("net6_input2", "array"),
            ("gamma", "net6_output1"),
        ],
        inputs={"net6_input2": arrays},
        experiments={"e1": "cond1", "e2": "cond2"},
    )
    reader = _read(path)
    problem = reader.to_optimization_problem()
    problem.initialize(SETTINGS)
    problem.predictions(np.asarray(problem.x0, dtype=float))
    k = problem.simulation_keys.index("e2")
    changes = problem.simulations[k].timecourses[0].changes
    assert [changes[f"net6__input1__{i}"].magnitude for i in range(3)] == [
        3.0,
        2.0,
        1.0,
    ]


# --- WHAT IS NOT READ ---


def _problem(tmp_path: Path, **kwargs: Any) -> Path:
    """Write the problem of a network in the right hand side, changed by kwargs."""
    arguments: dict[str, Any] = {
        "networks": [feed_forward()],
        "pre_initialization": {"net1": False},
        "mapping": [*INPUTS, *OUTPUT],
        "hybridization": [*SPECIES, ("gamma", "net1_output1")],
    }
    arguments.update(kwargs)
    return write_problem(tmp_path, **arguments)


def test_a_format_which_is_not_read(tmp_path: Path) -> None:
    """Only the format `YAML` is read, the others are a gap."""
    path = _problem(tmp_path, formats={"net1": "pytorch"})
    with pytest.raises(
        SciMLProblemError, match=r"'pytorch'.*sciml-model-format"
    ) as excinfo:
        _read(path)
    assert excinfo.value.gap == "sciml-model-format"


def test_a_prior_of_a_network(tmp_path: Path) -> None:
    """Priors on the parameters of a network are not read (#190)."""
    path = _problem(
        tmp_path,
        parameters=[
            {
                "parameterId": "net1_layer1",
                "nominalValue": "array",
                "estimate": True,
                "lowerBound": "-inf",
                "upperBound": "inf",
                "priorDistribution": "normal",
                "priorParameters": "0.0;1.0",
            }
        ],
        mapping=[*INPUTS, *OUTPUT, ("net1_layer1", "net1.parameters[layer1]")],
    )
    with pytest.raises(
        SciMLProblemError, match=r"prior 'normal'.*sciml-priors"
    ) as excinfo:
        _read(path)
    assert excinfo.value.gap == "sciml-priors"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"mapping": [*INPUTS, *OUTPUT, ("x", "net9.inputs[0][0]")]},
            r"names the network 'net9'",
        ),
        (
            {"hybridization": [*SPECIES, ("gamma", "2 * net1_output1")]},
            r"assigns 'gamma' the value '2\.0\*net1_output1'",
        ),
        (
            {"hybridization": [("net1_input1", "prey"), ("gamma", "net1_output1")]},
            "the input 'net1_input2' has no value",
        ),
        (
            {
                "hybridization": [
                    *SPECIES,
                    ("gamma", "net1_output1"),
                    ("alpha", "net1_output1"),
                ]
            },
            "assigns the output 'net1_output1' to 'gamma' and to 'alpha'",
        ),
        (
            {"hybridization": SPECIES},
            "no output of the network is used",
        ),
        (
            {"hybridization": [*SPECIES, ("gamma", "net1_output1"), ("gamma", "prey")]},
            "assigns 'gamma' twice",
        ),
        (
            {
                "pre_initialization": {"net1": True},
                "observables": {"prey_o": "net1_output1", "predator_o": "predator"},
                "hybridization": [],
                "mapping": [
                    ("k1", "net1.inputs[0][0]"),
                    ("k2", "net1.inputs[0][1]"),
                    *OUTPUT,
                ],
                "parameters": [
                    {"parameterId": "k1", "nominalValue": 1.0, "estimate": False},
                    {"parameterId": "k2", "nominalValue": 1.0, "estimate": False},
                ],
            },
            "used by an observable, but the network runs before the simulation",
        ),
        (
            {
                "mapping": [("net1_input1", "net1.inputs[0][0]"), *OUTPUT],
                "hybridization": [("net1_input1", "prey"), ("gamma", "net1_output1")],
            },
            r"inputs of the shapes \[\(1,\)\] do not fit",
        ),
    ],
)
def test_a_problem_which_cannot_be_read(
    tmp_path: Path, kwargs: dict[str, Any], message: str
) -> None:
    """The error names the network, the input, the output or the row."""
    path = _problem(tmp_path, **kwargs)
    with pytest.raises((SciMLProblemError, NetworkHybridizationError), match=message):
        _read(path).to_optimization_problem()


def test_a_convolution_in_the_right_hand_side(tmp_path: Path) -> None:
    """A network which is not evaluated on expressions is not compiled."""
    network = convolution()
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net3": False},
        mapping=[("input0", "net3.inputs[0]"), ("net3_output1", "net3.outputs[0][0]")],
        hybridization=[("input0", "array"), ("gamma", "net3_output1")],
        inputs={"input0": {ALL_CONDITION_IDS: np.ones((1, 4, 4))}},
    )
    with pytest.raises(NetworkHybridizationError, match="not evaluated on expressions"):
        _read(path).to_optimization_problem()


def test_from_petab(tmp_path: Path) -> None:
    """`from_petab` reads a problem with networks like any other."""
    path = _problem(tmp_path)
    problem, settings = from_petab(path)
    assert problem.hybridizations
    assert settings == FitSettings()
    problem.initialize(SETTINGS)
    assert np.isfinite(log_likelihood(problem))


# --- THE GAPS ---


def test_the_gaps_of_a_hybrid_problem(tmp_path: Path) -> None:
    """A problem with networks runs into the scale of a parameter."""
    problem, _ = from_petab(_problem(tmp_path))
    problem.initialize(SETTINGS)
    ids = {gap.id for gap in gaps_of_problem(problem)}
    assert "sciml-parameter-scale" in ids
    assert "sciml-training-mode" not in ids
    assert GAPS_BY_ID["sciml-parameter-scale"].kind is GapKind.EXTENSION
    for gap_id in ("sciml-model-format", "sciml-layer-sbml", "sciml-priors"):
        assert GAPS_BY_ID[gap_id].kind is GapKind.UNSUPPORTED
    assert GAPS_BY_ID["sciml-training-mode"].kind is GapKind.LOSSY


def test_the_gap_of_the_training_mode(tmp_path: Path) -> None:
    """A network with dropout is evaluated in evaluation mode."""
    network = feed_forward()
    model = network.model.model_copy(deep=True)
    model.layers.append(Layer(layer_id="drop", layer_type="Dropout", args={"p": 0.5}))
    model.forward.insert(
        2,
        Node(name="drop", op="call_module", target="drop", args=["layer1"], kwargs={}),
    )
    model.forward[3].args = ["drop"]
    with_dropout = Network(sid="net1", model=model, parameters=network.parameters)
    problem, _ = from_petab(_problem(tmp_path, networks=[with_dropout]))
    problem.initialize(SETTINGS)
    assert "sciml-training-mode" in {gap.id for gap in gaps_of_problem(problem)}


# --- THE INPUTS WHICH BITE ---


def test_a_problem_with_two_models(tmp_path: Path) -> None:
    """A problem with networks has one model, the error names the models."""
    path = _problem(
        tmp_path,
        extra_yaml={
            "model_files": {
                "lv": {"location": "lv.xml", "language": "sbml"},
                "lv2": {"location": "lv.xml", "language": "sbml"},
            }
        },
    )
    with pytest.raises(SciMLProblemError, match=r"one model.*\['lv', 'lv2'\]"):
        _read(path)


def test_a_condition_which_sets_the_input_of_a_compiled_network(tmp_path: Path) -> None:
    """A network in the right hand side has one formula per input."""
    path = _problem(
        tmp_path,
        hybridization=[("net1_input2", "predator"), ("gamma", "net1_output1")],
        conditions=[("cond1", "net1_input1", "prey"), ("cond2", "net1_input1", "1.0")],
        experiments={"e1": "cond1", "e2": "cond2"},
    )
    with pytest.raises(NetworkHybridizationError, match="one formula per input"):
        _read(path).to_optimization_problem()


def test_an_output_in_the_right_hand_side_and_an_observable(tmp_path: Path) -> None:
    """An output sets an entity or is a symbol of an observable, not both."""
    path = _problem(
        tmp_path, observables={"prey_o": "net1_output1", "predator_o": "predator"}
    )
    with pytest.raises(
        SciMLProblemError, match="'net1_output1' is used by an observable and assigned"
    ):
        _read(path).to_optimization_problem()


def test_a_problem_which_petab_read_with_torch(tmp_path: Path) -> None:
    """A problem `petab` read with its networks is read from its configuration."""
    pytest.importorskip("torch")
    from petab.v2 import Problem as PetabProblem

    petab_problem = PetabProblem.from_yaml(_problem(tmp_path))
    assert petab_problem.extensions.sciml is not None
    reader = PetabReader(petab_problem)
    reader.derived_dir = tmp_path / "derived"
    assert reader.sciml is not None
    assert reader.sciml.networks["net1"] == feed_forward()
    problem = reader.to_optimization_problem()
    assert len(problem.hybridizations) == 1
```

- [ ] **Step 3: Apply this patch to `tests/fit/test_petab_v2_noise.py`**

Apply this patch to `tests/fit/test_petab_v2_noise.py`:

```diff
diff --git a/tests/fit/test_petab_v2_noise.py b/tests/fit/test_petab_v2_noise.py
index 009f3cb..3dcb8fc 100644
--- a/tests/fit/test_petab_v2_noise.py
+++ b/tests/fit/test_petab_v2_noise.py
@@ -13,11 +13,13 @@ from petab.v1.yaml import load_yaml, write_yaml
 from sbmlsim.fit import FitSettings
 from sbmlsim.fit.objects import NoiseDistribution, NoiseModel, NoiseParameter
 from sbmlsim.fit.optimization import OptimizationProblem
-from sbmlsim.fit.petab_v2 import GapKind, gaps_of_problem, to_petab
+from sbmlsim.fit.petab_v2 import GapKind, extension, gaps_of_problem, to_petab
 from sbmlsim.fit.petab_v2.extension import (
     EXTENSION_ID,
     KNOWN_EXTENSIONS,
+    SCIML_EXTENSION_ID,
     check_extensions,
+    known_extensions,
 )
 from sbmlsim.fit.petab_v2.gaps import GAPS_BY_ID
 from sbmlsim.fit.petab_v2.likelihood import log_likelihood
@@ -229,7 +231,7 @@ def test_check_extensions() -> None:
         )
     # the ids are listed, not the representation of a list
     assert "'tool_a, tool_b'" in str(excinfo.value)
-    assert f"'{EXTENSION_ID}'" in str(excinfo.value)
+    assert f"it knows '{', '.join(sorted(known_extensions()))}'" in str(excinfo.value)
     assert "[" not in str(excinfo.value)
     # an extension which the caller knows is not foreign
     assert check_extensions({"tool_a": {"required": True}}, known={"tool_a"}) == []
@@ -263,25 +265,59 @@ def test_a_foreign_extension_which_is_not_required_is_ignored(
     assert len(problem.mapping_keys) == 4


-def test_the_sciml_extension_is_reported_and_not_a_missing_module(
-    petab_iv: Path,
+SCIML_BLOCK = {
+    "version": "0.1.0",
+    "required": True,
+    "array_files": [],
+    "hybridization_files": [],
+    "neural_networks": {},
+}
+
+
+def test_the_extension_of_the_networks_is_known_with_the_extra(
+    monkeypatch: pytest.MonkeyPatch,
+) -> None:
+    """The extension `sciml` is read when `petab_sciml` is installed."""
+    monkeypatch.setattr(extension, "sciml_installed", lambda: True)
+    assert known_extensions() == {EXTENSION_ID, SCIML_EXTENSION_ID}
+    assert check_extensions({SCIML_EXTENSION_ID: SCIML_BLOCK}) == []
+
+    monkeypatch.setattr(extension, "sciml_installed", lambda: False)
+    assert known_extensions() == {EXTENSION_ID}
+    assert check_extensions({SCIML_EXTENSION_ID: {"required": False}}) == [
+        SCIML_EXTENSION_ID
+    ]
+
+
+def test_the_missing_extra_is_named_and_not_a_missing_module(
+    petab_iv: Path, monkeypatch: pytest.MonkeyPatch
 ) -> None:
     """The extensions are checked before `petab` reads their files."""
-    _add_extension(
-        petab_iv,
-        "sciml",
-        {
-            "version": "0.1.0",
-            "required": True,
-            "array_files": [],
-            "hybridization_files": [],
-            "neural_networks": {},
-        },
-    )
-    with pytest.raises(ValueError, match=r"requires the extensions.*sciml"):
+    _add_extension(petab_iv, SCIML_EXTENSION_ID, SCIML_BLOCK)
+    monkeypatch.setattr(extension, "sciml_installed", lambda: False)
+    with pytest.raises(ImportError, match=r"pip install sbmlsim\[sciml\]"):
+        from_petab(petab_iv / "problem.yaml")
+    # the other extensions which are required are still named
+    _add_extension(petab_iv, "tool_a", {"version": "1.0.0", "required": True})
+    with pytest.raises(ImportError, match=r"pip install sbmlsim\[sciml\]"):
         from_petab(petab_iv / "problem.yaml")


+def test_a_problem_with_the_extension_and_without_networks_is_read(
+    petab_iv: Path,
+) -> None:
+    """The tables of a problem are read without its block of the networks."""
+    pytest.importorskip("petab_sciml")
+    _add_extension(petab_iv, SCIML_EXTENSION_ID, SCIML_BLOCK)
+    reader = PetabReader.from_yaml(petab_iv / "problem.yaml")
+    assert reader.sciml is not None
+    assert reader.sciml.networks == {}
+    problem = reader.to_optimization_problem()
+    assert problem.hybridizations == []
+    problem.initialize(reader.settings)
+    assert len(problem.mapping_keys) == 4
+
+
 def test_round_trip_keeps_the_noise_models(petab_iv: Path, tmp_path: Path) -> None:
     """A problem which is read and written again has the noise it had."""
     first, second, third = _observable_ids(petab_iv)[:3]
```

- [ ] **Step 4: Apply this patch to `tests/sciml/test_package.py`**

Apply this patch to `tests/sciml/test_package.py`:

```diff
diff --git a/tests/sciml/test_package.py b/tests/sciml/test_package.py
index 59c7f9a..af99d8d 100644
--- a/tests/sciml/test_package.py
+++ b/tests/sciml/test_package.py
@@ -47,7 +47,10 @@ def test_the_package_does_not_import_the_networks() -> None:
         "import sys; sys.modules['petab_sciml'] = None\n"
         "import sbmlsim, sbmlsim.fit, sbmlsim.testsuite, sbmlsim.fit.petab_v2\n"
         "import sbmlsim.fit.cli, sbmlsim.fit.derived, sbmlsim.fit.runner\n"
+        "from sbmlsim.fit.petab_v2.extension import known_extensions\n"
+        "assert known_extensions() == {'sbmlsim'}, known_extensions()\n"
         "assert 'sbmlsim.sciml' not in sys.modules\n"
+        "assert 'sbmlsim.fit.petab_v2.sciml' not in sys.modules\n"
     )
     assert result.returncode == 0, result.stderr

```

- [ ] **Step 5: Run the tests to verify they fail**

Run: `uv run pytest -q -x tests/sciml/test_reader.py tests/fit/test_petab_v2_noise.py tests/sciml/test_package.py`
Expected: FAIL at collection: `ModuleNotFoundError: No module named 'sbmlsim.fit.petab_v2.sciml'` and `ImportError: cannot import name 'known_extensions' from 'sbmlsim.fit.petab_v2.extension'`

- [ ] **Step 6: Apply this patch to `src/sbmlsim/fit/petab_v2/extension.py`**

Apply this patch to `src/sbmlsim/fit/petab_v2/extension.py`:

```diff
diff --git a/src/sbmlsim/fit/petab_v2/extension.py b/src/sbmlsim/fit/petab_v2/extension.py
index bf434be..4d170c4 100644
--- a/src/sbmlsim/fit/petab_v2/extension.py
+++ b/src/sbmlsim/fit/petab_v2/extension.py
@@ -14,6 +14,7 @@ through `sbmlsim` keeps the fit it started from, and
 read and fit with the objective PEtab defines.
 """

+import importlib.util
 from collections.abc import Collection, Mapping
 from typing import Any

@@ -23,8 +24,16 @@ from pydantic import Field
 #: id of the extension, the key of the block in the YAML of the problem
 EXTENSION_ID = "sbmlsim"

-#: the extensions the reader interprets. A problem which requires another one
-#: is rejected, see `check_extensions`
+#: id of the extension of PEtab SciML, i.e. of the neural networks of a
+#: hybrid problem, see `sbmlsim.fit.petab_v2.sciml`
+SCIML_EXTENSION_ID = "sciml"
+
+#: the extra of `sbmlsim` which reads the extension of PEtab SciML
+SCIML_EXTRA = "pip install sbmlsim[sciml]"
+
+#: the extensions the reader interprets without an extra. A problem which
+#: requires another one is rejected, see `known_extensions` and
+#: `check_extensions`
 KNOWN_EXTENSIONS: frozenset[str] = frozenset({EXTENSION_ID})

 #: version of the extension, raised when the block changes
@@ -104,9 +113,26 @@ def extension_of(config: Any) -> SbmlsimExtension | None:
     return SbmlsimExtension(**data)


+def sciml_installed() -> bool:
+    """Check whether the extra `sciml` is installed, i.e. `petab_sciml`."""
+    return importlib.util.find_spec("petab_sciml") is not None
+
+
+def known_extensions() -> frozenset[str]:
+    """Get the extensions the reader interprets in this environment.
+
+    Returns:
+        `KNOWN_EXTENSIONS`, and the extension of PEtab SciML when the extra
+        `sciml` is installed.
+    """
+    if sciml_installed():
+        return KNOWN_EXTENSIONS | {SCIML_EXTENSION_ID}
+    return KNOWN_EXTENSIONS
+
+
 def check_extensions(
     extensions: Mapping[str, Any] | None,
-    known: Collection[str] = KNOWN_EXTENSIONS,
+    known: Collection[str] | None = None,
 ) -> list[str]:
     """Check the extensions of a problem against the ones the reader knows.

@@ -120,7 +146,8 @@ def check_extensions(
             the dictionaries of the YAML or as the `ExtensionConfig` objects
             of a problem which was read, `None` for a problem without
             extensions.
-        known: ids of the extensions the reader interprets.
+        known: ids of the extensions the reader interprets,
+            `known_extensions` by default.

     Returns:
         The ids of the extensions which are to be ignored, i.e. the ones
@@ -128,8 +155,13 @@ def check_extensions(
         The reader logs them.

     Raises:
+        ImportError: if the problem requires the extension of PEtab SciML
+            and the extra `sciml` is not installed. The message names the
+            extra.
         ValueError: if the problem requires an extension which is not known.
     """
+    if known is None:
+        known = known_extensions()
     required: list[str] = []
     ignored: list[str] = []
     for extension_id, block in (extensions or {}).items():
@@ -144,6 +176,13 @@ def check_extensions(
         else:
             ignored.append(extension_id)

+    if SCIML_EXTENSION_ID in required:
+        raise ImportError(
+            f"The PEtab problem requires the extension '{SCIML_EXTENSION_ID}', "
+            f"i.e. it is a problem of PEtab SciML with neural networks. "
+            f"`sbmlsim` reads it with the package 'petab_sciml', which is "
+            f"installed with the extra 'sciml': {SCIML_EXTRA}"
+        )
     if required:
         raise ValueError(
             f"The PEtab problem requires the extensions '{', '.join(required)}', "
```

- [ ] **Step 7: Apply this patch to `src/sbmlsim/fit/petab_v2/gaps.py`**

Apply this patch to `src/sbmlsim/fit/petab_v2/gaps.py`:

```diff
diff --git a/src/sbmlsim/fit/petab_v2/gaps.py b/src/sbmlsim/fit/petab_v2/gaps.py
index a691e7d..cd9093c 100644
--- a/src/sbmlsim/fit/petab_v2/gaps.py
+++ b/src/sbmlsim/fit/petab_v2/gaps.py
@@ -29,6 +29,24 @@ if TYPE_CHECKING:

 logger = logging.getLogger(__name__)

+#: the layers of a network which behave differently in training mode
+EVALUATION_MODE_LAYERS: frozenset[str] = frozenset(
+    {
+        "Dropout",
+        "Dropout1d",
+        "Dropout2d",
+        "Dropout3d",
+        "AlphaDropout",
+        "FeatureAlphaDropout",
+        "BatchNorm1d",
+        "BatchNorm2d",
+        "BatchNorm3d",
+        "InstanceNorm1d",
+        "InstanceNorm2d",
+        "InstanceNorm3d",
+    }
+)
+

 class GapKind(StrEnum):
     """What the layer does about a difference to PEtab v2."""
@@ -215,6 +233,62 @@ GAPS: tuple[Gap, ...] = (
         "normal noise of the standard deviation of its data, or of `1.0` for "
         "data without errors, and has that noise model when it is read back",
     ),
+    Gap(
+        id="sciml-model-format",
+        kind=GapKind.UNSUPPORTED,
+        sbmlsim="a network is the NN YAML of PEtab SciML, which "
+        "`sbmlsim.sciml` evaluates and compiles",
+        petab="PEtab SciML also allows the formats `pytorch`, `equinox` and "
+        "`lux.jl`, i.e. a network in the code of a framework",
+        detail="the reader raises for a network which is not in the format "
+        "`YAML` and names the network: the code of a framework is not read",
+    ),
+    Gap(
+        id="sciml-layer-sbml",
+        kind=GapKind.UNSUPPORTED,
+        sbmlsim="a network in the right hand side or in an observable is "
+        "compiled into the model as assignment rules, which the MathML of "
+        "SBML expresses",
+        petab="a layer of the NN YAML is any layer of PEtab SciML, and a "
+        "network of any layers sits in the right hand side",
+        detail="a convolution, a pooling or a normalization layer, and `gelu` "
+        "with the error function, have no MathML: the reader raises for such a "
+        "network in the right hand side or in an observable and names the "
+        "network and the node. Such a network runs before the simulation",
+    ),
+    Gap(
+        id="sciml-training-mode",
+        kind=GapKind.LOSSY,
+        sbmlsim="a network is evaluated in evaluation mode: dropout is the "
+        "identity and the normalization layers use their stored statistics",
+        petab="PEtab SciML does not say in which mode a network is evaluated, "
+        "the reference values of its test suite are built in training mode "
+        "for dropout",
+        detail="the values of a problem with such a layer differ from the ones "
+        "of a tool which evaluates in training mode",
+    ),
+    Gap(
+        id="sciml-priors",
+        kind=GapKind.UNSUPPORTED,
+        sbmlsim="the objective of a fit has no priors (issue #190)",
+        petab="`priorDistribution` and `priorParameters` of a parameter of a "
+        "network, i.e. of all elements a row covers",
+        detail="the reader raises for a prior on the parameters of a network "
+        "and names the parameter. A problem with priors states a log-posterior, "
+        "which the log-likelihood is not",
+    ),
+    Gap(
+        id="sciml-parameter-scale",
+        kind=GapKind.EXTENSION,
+        sbmlsim="`FitParameter.scale`, the space the optimizer searches one "
+        "parameter in: the elements of a network are negative and zero and "
+        "are searched on the linear scale",
+        petab="PEtab v2 has no scale of a parameter. The problems of PEtab "
+        "SciML carry the column `parameterScale` of PEtab v1",
+        detail="the reader reads the column of a problem of PEtab SciML, the "
+        "elements of a network are on the linear scale, and the scale of a "
+        "parameter goes to the extension",
+    ),
     Gap(
         id="foreign-extension",
         kind=GapKind.UNSUPPORTED,
@@ -367,6 +441,15 @@ def gaps_of_problem(problem: "OptimizationProblem") -> list[Gap]:
         if any(parameter.estimate for parameter in noise.parameters):
             hits.add("noise-parameters")

+    if any(parameter.scale is not None for parameter in problem.parameters):
+        hits.add("sciml-parameter-scale")
+    for hybridization in problem.hybridizations:
+        # the layers of a network of `sbmlsim.sciml`, without importing it
+        model = getattr(getattr(hybridization, "network", None), "model", None)
+        layer_types = {layer.layer_type for layer in getattr(model, "layers", [])}
+        if layer_types & EVALUATION_MODE_LAYERS:
+            hits.add("sciml-training-mode")
+
     for k, xid in enumerate(problem.xid_observable):
         if xid != "time":
             logger.warning(
```

- [ ] **Step 8: Create `src/sbmlsim/fit/petab_v2/sciml.py`**

Create `src/sbmlsim/fit/petab_v2/sciml.py`:

```python
"""The neural networks of a PEtab SciML problem.

[PEtab SciML](https://github.com/PEtab-dev/petab_sciml) is the extension
`sciml` of PEtab v2 for hybrid problems, in which a model is combined with
neural networks. This module translates the extension into the objects of
`sbmlsim.sciml`, which know nothing of PEtab:

| PEtab SciML | `sbmlsim` |
| --- | --- |
| a network of the block `neural_networks` with its array files | `Network` |
| `pre_initialization`, the hybridization table and the observables | `Hybridization` and its `NetworkPattern` |
| the rows of the mapping table | the ids of the inputs, outputs and arrays |
| the rows of the parameter table of a network | `FitParameter` per element |
| an array of a condition | an array of the simulation of an experiment |

The module imports `sbmlsim.sciml` and with it `petab_sciml`, which is the
extra `sciml`. `sbmlsim.fit.petab_v2.reader` imports it only for a problem
with the extension.

`petab.v2.Problem.from_yaml` of `petab` 0.9.0 reads the networks of a problem
through PyTorch. `torch` is no dependency of `sbmlsim`, so the tables of a
problem are read by `petab` without the block of the extension, and the
networks, the hybridization tables and the array files are read here.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import sympy
from petab.v2 import Problem as PetabProblem
from petab.v2.extensions.sciml import HybridizationTable, SciMLConfig
from petab_sciml.constants import ALL_CONDITION_IDS, ARRAY

from sbmlsim.fit.objects import EXTERNAL_PREFIX, FitParameter
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.mathml import expression_to_formula, formula_symbols
from sbmlsim.sciml.hybridization import (
    ALL_CONDITIONS,
    Hybridization,
    NetworkInput,
    NetworkPattern,
    input_shapes,
    output_shapes,
)
from sbmlsim.sciml.network import (
    Network,
    input_id,
    load_array_data,
    output_id,
)
from sbmlsim.sciml.parameters import (
    ELEMENT_UNIT,
    network_fit_parameters,
    nominal_parameters,
)

logger = logging.getLogger(__name__)

#: the format of a network which is read
YAML_FORMAT = "yaml"

#: a `modelEntityId` of the mapping table which names a part of a network
ENTITY = re.compile(
    r"(?P<network>[^.\[\]\s]+)\.(?P<kind>inputs|outputs|parameters)(?P<rest>.*)"
)

#: the indices of an input or output, e.g. `[0][1]`
INDICES = re.compile(r"(?:\[\d+\])+")

#: the layer and the array of the parameters, e.g. `[layer1].weight`
LAYER = re.compile(r"\[(?P<layer>[^\]]+)\](?:\.(?P<array>[^.\[\]\s]+))?")

#: the scales of the column `parameterScale`, which PEtab v2 dropped and the
#: problems of PEtab SciML carry
SCALES: dict[str, ParameterScaleType] = {
    "lin": ParameterScaleType.LINEAR,
    "log": ParameterScaleType.LOG,
    "log10": ParameterScaleType.LOG10,
}


class SciMLProblemError(ValueError):
    """A PEtab SciML problem which cannot be read.

    Attributes:
        gap: id of the gap the problem runs into, see
            `sbmlsim.fit.petab_v2.gaps`, `None` for a problem which is not
            valid.
    """

    def __init__(self, message: str, gap: str | None = None) -> None:
        """Initialize the error.

        Args:
            message: what cannot be read.
            gap: id of the gap the problem runs into.
        """
        self.gap = gap
        super().__init__(message if gap is None else f"{message} (gap '{gap}')")


@dataclass(frozen=True)
class NetworkEntity:
    """A part of a network a row of the mapping table names.

    Attributes:
        petab_id: the `petabEntityId` of the row.
        network: id of the network.
        kind: `inputs`, `outputs` or `parameters`.
        k: position of the input or output, `None` for parameters.
        index: index of the element, `None` for an input which is an array
            and for parameters.
        key: the key of the parameters, e.g. `net1.layer1.weight`, see
            `sbmlsim.sciml.parameters`, `None` for an input or output.
    """

    petab_id: str
    network: str
    kind: str
    k: int | None = None
    index: tuple[int, ...] | None = None
    key: str | None = None


def parse_entity(petab_id: str, model_entity_id: str) -> NetworkEntity | None:
    """Read a row of the mapping table which names a part of a network.

    Args:
        petab_id: the `petabEntityId` of the row.
        model_entity_id: the `modelEntityId`, e.g. `net1.inputs[0][1]`,
            `net1.inputs[0]`, `net1.outputs[0][0]`, `net1.parameters`,
            `net1.parameters[layer1]` or `net1.parameters[layer1].weight`.

    Returns:
        The part of the network, `None` for a row which names an entity of a
        model.

    Raises:
        SciMLProblemError: if the row names a part of a network and is not
            one of the forms.
    """
    match = ENTITY.fullmatch(model_entity_id.strip())
    if match is None:
        return None
    network, kind, rest = match.group("network", "kind", "rest")
    error = SciMLProblemError(
        f"The mapping of '{petab_id}' to '{model_entity_id}' is not a part of a "
        f"network: an input is '{network}.inputs[0][1]' or '{network}.inputs[0]', "
        f"an output '{network}.outputs[0][0]' and the parameters "
        f"'{network}.parameters', '{network}.parameters[layer1]' or "
        f"'{network}.parameters[layer1].weight'"
    )
    if kind == "parameters":
        if not rest:
            return NetworkEntity(petab_id, network, kind, key=network)
        layer = LAYER.fullmatch(rest)
        if layer is None:
            raise error
        key = f"{network}.{layer.group('layer')}"
        if layer.group("array"):
            key = f"{key}.{layer.group('array')}"
        return NetworkEntity(petab_id, network, kind, key=key)

    if INDICES.fullmatch(rest) is None:
        raise error
    numbers = [int(number) for number in re.findall(r"\d+", rest)]
    index = tuple(numbers[1:]) if len(numbers) > 1 else None
    if kind == "outputs" and index is None:
        raise error
    return NetworkEntity(petab_id, network, kind, k=numbers[0], index=index)


def read_sciml_config(extensions: Mapping[str, Any] | None) -> SciMLConfig | None:
    """Get the block of the extension `sciml` of a problem.

    Args:
        extensions: the blocks of the extensions of the YAML of the problem.

    Returns:
        The configuration, `None` for a problem without the extension.

    Raises:
        SciMLProblemError: if the block is not a block of the extension.
    """
    block = (extensions or {}).get("sciml")
    if block is None:
        return None
    if isinstance(block, SciMLConfig):
        return block
    try:
        return SciMLConfig(**dict(block))
    except (TypeError, ValueError) as err:
        raise SciMLProblemError(
            f"The block 'sciml' of the problem is not valid: {err}"
        ) from err


class SciMLReader:
    """Read the networks of a PEtab SciML problem.

    Attributes:
        networks: the networks by their id, with the nominal values of the
            problem.
        entities: the parts of the networks the mapping table names, by their
            `petabEntityId`.
    """

    def __init__(
        self,
        petab_problem: PetabProblem,
        config: SciMLConfig,
        base_path: Path,
        simulations: Mapping[str, list[str]],
    ) -> None:
        """Read the networks, the hybridization tables and the array files.

        Args:
            petab_problem: the problem, read without the block of the
                extension.
            config: the block of the extension.
            base_path: directory the files of the problem are relative to.
            simulations: id of the simulation of every experiment -> the ids
                of the conditions the simulation starts with.

        Raises:
            SciMLProblemError: if the problem has not one model, if a network
                is not in the format `YAML`, if a row of the mapping table
                names a network the problem does not have, if a row of the
                hybridization table cannot be read, or if a parameter of a
                network has a prior.
            NetworkImportError: if a network or its arrays cannot be read.
        """
        self.petab_problem = petab_problem
        self.config = config
        self.base_path = Path(base_path)
        self.simulations = {key: list(ids) for key, ids in simulations.items()}
        if len(petab_problem.models) != 1:
            raise SciMLProblemError(
                f"A problem with networks has one model, but the problem has "
                f"the models {[m.model_id for m in petab_problem.models]}"
            )
        self.model = petab_problem.models[0]
        self.model_id: str = self.model.model_id
        self._parameters = {p.id: p for p in petab_problem.parameters}

        self._arrays = [
            load_array_data(self.base_path / str(path)) for path in config.array_files
        ]
        self.entities: dict[str, NetworkEntity] = {}
        for mapping in petab_problem.mappings:
            entity = parse_entity(mapping.petab_id, mapping.model_id or "")
            if entity is None:
                continue
            if entity.network not in (config.neural_networks or {}):
                raise SciMLProblemError(
                    f"The mapping of '{entity.petab_id}' names the network "
                    f"'{entity.network}', the networks of the problem are "
                    f"{sorted(config.neural_networks or {})}"
                )
            self.entities[entity.petab_id] = entity

        self._hybridization: dict[str, sympy.Basic] = {}
        for path in config.hybridization_files:
            for row in HybridizationTable.from_tsv(str(path), self.base_path).elements:
                if row.target_id in self._hybridization:
                    raise SciMLProblemError(
                        f"The hybridization table assigns '{row.target_id}' twice"
                    )
                if row.target_value is None:
                    raise SciMLProblemError(
                        f"The hybridization table assigns '{row.target_id}' no value"
                    )
                self._hybridization[row.target_id] = row.target_value

        self.networks: dict[str, Network] = {
            sid: self._network(sid) for sid in (config.neural_networks or {})
        }
        #: the inputs of every network by their id
        self.inputs: dict[str, dict[str, NetworkInput]] = {
            sid: {
                input_id(sid, entity.k or 0, entity.index): self._input(entity)
                for entity in self._entities(sid, "inputs")
            }
            for sid in self.networks
        }

    # --- THE NETWORKS ---

    def _entities(self, network: str, kind: str) -> list[NetworkEntity]:
        """Get the parts of a kind of a network, in the order of the table."""
        return [
            entity
            for entity in self.entities.values()
            if entity.network == network and entity.kind == kind
        ]

    def _network(self, sid: str) -> Network:
        """Read a network with the nominal values of the problem.

        Args:
            sid: id of the network.

        Returns:
            The network with the values of its array file, replaced by the
            values of the rows of the parameter table which are numbers.

        Raises:
            SciMLProblemError: if the network is not in the format `YAML`,
                or if a parameter of the network has a prior.
            NetworkImportError: if the network or its arrays cannot be read.
        """
        network_config = (self.config.neural_networks or {})[sid]
        if network_config.format.lower() != YAML_FORMAT:
            raise SciMLProblemError(
                f"Network '{sid}': the format '{network_config.format}' is not "
                f"read, only the format 'YAML'",
                gap="sciml-model-format",
            )
        network = Network.from_files(
            self.base_path / str(network_config.location), sid=sid
        )
        for path, data in zip(self.config.array_files, self._arrays, strict=True):
            if sid in data.parameters:
                network = replace(
                    network, parameters=network.read_arrays(self.base_path / str(path))
                )
        values: dict[str, float] = {}
        for entity in self._entities(sid, "parameters"):
            parameter = self._parameters.get(entity.petab_id)
            if parameter is None or entity.key is None:
                continue
            if getattr(parameter, "prior_distribution", None) is not None:
                raise SciMLProblemError(
                    f"Network '{sid}': the parameter '{parameter.id}' has the "
                    f"prior '{parameter.prior_distribution}', priors of the "
                    f"parameters of a network are not read",
                    gap="sciml-priors",
                )
            value = parameter.nominal_value
            if value is not None and value != ARRAY:
                values[entity.key] = float(value)
        return replace(network, parameters=nominal_parameters(network, values))

    def is_pre_initialization(self, sid: str) -> bool:
        """Check whether a network runs before the simulation."""
        return bool((self.config.neural_networks or {})[sid].pre_initialization)

    @property
    def parameter_ids(self) -> set[str]:
        """Get the ids of the parameter table which are arrays of networks."""
        return {e.petab_id for e in self.entities.values() if e.kind == "parameters"}

    @property
    def input_ids(self) -> set[str]:
        """Get the ids which are inputs of networks."""
        return {e.petab_id for e in self.entities.values() if e.kind == "inputs"}

    @property
    def condition_ids(self) -> set[str]:
        """Get the conditions the array files have arrays for."""
        return {
            condition
            for data in self._arrays
            for arrays in data.inputs.values()
            for condition in arrays
        } - {ALL_CONDITION_IDS}

    # --- THE HYBRIDIZATIONS ---

    def _conditions(self, by_condition: Mapping[str, Any], what: str) -> dict[str, Any]:
        """Translate the values of conditions into values of simulations.

        Args:
            by_condition: id of the condition -> value, `ALL_CONDITION_IDS`
                for the value of every condition.
            what: what the values are, for the message.

        Returns:
            id of the simulation -> value, `ALL_CONDITIONS` for the value of
            every simulation.

        Raises:
            SciMLProblemError: if a simulation starts with two conditions
                which have a value.
        """
        values: dict[str, Any] = {}
        if ALL_CONDITION_IDS in by_condition:
            values[ALL_CONDITIONS] = by_condition[ALL_CONDITION_IDS]
        for simulation, conditions in self.simulations.items():
            hits = [c for c in conditions if c in by_condition]
            if len(hits) > 1:
                raise SciMLProblemError(
                    f"The experiment '{simulation}' starts with the conditions "
                    f"{hits}, which all set {what}"
                )
            if hits:
                values[simulation] = by_condition[hits[0]]
        return values

    def _input(self, entity: NetworkEntity) -> NetworkInput:
        """Get an input of a network from the tables of the problem.

        The value of an input is, in this order, the row of the hybridization
        table, the changes of the conditions, and the parameter of the
        parameter table of its id.

        Args:
            entity: the input.

        Returns:
            The input.

        Raises:
            SciMLProblemError: if the input has no value, or if an array has
                no values in the array files.
        """
        petab_id = entity.petab_id
        value = self._hybridization.get(petab_id)
        if value is not None and str(value) == ARRAY:
            arrays = [
                data.inputs[petab_id]
                for data in self._arrays
                if petab_id in data.inputs
            ]
            if len(arrays) != 1:
                raise SciMLProblemError(
                    f"Network '{entity.network}': the input '{petab_id}' is an "
                    f"array, and {len(arrays)} array files have values for it"
                )
            return NetworkInput(
                arrays=self._conditions(
                    {
                        condition: np.asarray(array, dtype=float)
                        for condition, array in arrays[0].items()
                    },
                    f"the input '{petab_id}'",
                )
            )
        if value is not None:
            return NetworkInput(formula=expression_to_formula(value))

        by_condition = {
            condition.id: expression_to_formula(change.target_value)
            for condition in self.petab_problem.conditions
            for change in condition.changes
            if change.target_id == petab_id
        }
        if by_condition:
            formulas = self._conditions(by_condition, f"the input '{petab_id}'")
            if petab_id in self._parameters:
                formulas.setdefault(ALL_CONDITIONS, petab_id)
            return NetworkInput(formulas=formulas)
        if petab_id in self._parameters:
            return NetworkInput(formula=petab_id)
        raise SciMLProblemError(
            f"Network '{entity.network}': the input '{petab_id}' has no value, it "
            f"is neither assigned by the hybridization table or a condition nor "
            f"a parameter of the parameter table"
        )

    def _constants(self, inputs: Mapping[str, NetworkInput]) -> dict[str, float]:
        """Get the values of the symbols of inputs which are not in the model.

        Args:
            inputs: the inputs of a network.

        Returns:
            id -> nominal value of the parameters of the parameter table
            which the formulas use, which are not entities of the model and
            which are not estimated.
        """
        constants: dict[str, float] = {}
        for symbol in sorted(_symbols(inputs)):
            parameter = self._parameters.get(symbol)
            if parameter is None or self.model.has_entity_with_id(symbol):
                continue
            if parameter.estimate and not self.is_compiled_symbol(symbol):
                continue
            constants[symbol] = _nominal(parameter)
        return constants

    def is_compiled_symbol(self, symbol: str) -> bool:
        """Check whether a symbol is used by a network which is compiled.

        Such a symbol is a parameter of the model with the networks, so a
        parameter of the fit of its id is an entity of the model.

        Args:
            symbol: id of a parameter of the parameter table.
        """
        return any(
            symbol in _symbols(inputs)
            for sid, inputs in self.inputs.items()
            if not self.is_pre_initialization(sid)
        )

    def hybridizations(self) -> list[Hybridization]:
        """Get the hybridizations of the networks of the problem.

        The pattern of a network is `PRE_INITIALIZATION` when the problem
        says so. Otherwise an output which an observable uses is
        `OBSERVABLE` and an output which the hybridization table assigns to
        an entity of the model is `RHS`, and a network with outputs of both
        kinds has two hybridizations.

        Returns:
            The hybridizations, in the order of the networks of the problem.

        Raises:
            SciMLProblemError: if an input has no value, if an output is
                assigned to two targets or to a target and an observable, or
                if a network has no output which is used.
            NetworkHybridizationError: if the inputs and outputs do not fit
                a network.
        """
        observed = {
            str(symbol)
            for observable in self.petab_problem.observables
            for symbol in observable.formula.free_symbols
        }
        targets: dict[str, str] = {}
        for target, value in self._hybridization.items():
            output = self.entities.get(str(value))
            if output is None or output.kind != "outputs":
                continue
            if output.petab_id in targets:
                raise SciMLProblemError(
                    f"Network '{output.network}': the hybridization table assigns "
                    f"the output '{output.petab_id}' to '{targets[output.petab_id]}' "
                    f"and to '{target}'"
                )
            targets[output.petab_id] = target
        for target, value in self._hybridization.items():
            if target in self.entities or str(value) in targets:
                continue
            raise SciMLProblemError(
                f"The hybridization table assigns '{target}' the value '{value}'. "
                f"A row assigns a value to an input of a network or an output of "
                f"a network to an entity of the model"
            )

        hybridizations: list[Hybridization] = []
        for sid, network in self.networks.items():
            inputs = self.inputs[sid]
            outputs: dict[NetworkPattern, dict[str, str]] = {}
            shapes = output_shapes(network, input_shapes(network, inputs))
            for entity in self._entities(sid, "outputs"):
                key = output_id(sid, entity.k or 0, _index(entity, shapes))
                if entity.petab_id in targets:
                    pattern = (
                        NetworkPattern.PRE_INITIALIZATION
                        if self.is_pre_initialization(sid)
                        else NetworkPattern.RHS
                    )
                    target = self._target(targets[entity.petab_id], pattern)
                    outputs.setdefault(pattern, {})[key] = target
                if entity.petab_id in observed:
                    if self.is_pre_initialization(sid):
                        raise SciMLProblemError(
                            f"Network '{sid}': the output '{entity.petab_id}' is "
                            f"used by an observable, but the network runs "
                            f"before the simulation"
                        )
                    if entity.petab_id in targets:
                        raise SciMLProblemError(
                            f"Network '{sid}': the output '{entity.petab_id}' is "
                            f"used by an observable and assigned to "
                            f"'{targets[entity.petab_id]}' by the hybridization "
                            f"table. An output sets an entity of the model or "
                            f"is a symbol of an observable"
                        )
                    outputs.setdefault(NetworkPattern.OBSERVABLE, {})[key] = (
                        entity.petab_id
                    )
            if not outputs:
                raise SciMLProblemError(
                    f"Network '{sid}': no output of the network is used, neither "
                    f"by the hybridization table nor by an observable"
                )
            frozen = set(network.parameter_ids()) - {
                p.pid for p in self.network_parameters(sid)
            }
            hybridizations.extend(
                Hybridization(
                    network=network,
                    pattern=pattern,
                    model=self.model_id,
                    inputs=inputs,
                    outputs=pattern_outputs,
                    frozen=frozen,
                    constants=self._constants(inputs),
                )
                for pattern, pattern_outputs in outputs.items()
            )
        return hybridizations

    def _target(self, target: str, pattern: NetworkPattern) -> str:
        """Get the target of an output as `sbmlsim` names it.

        A network which runs before the simulation sets the initial value of
        a species as the model means it, i.e. the concentration of a species
        which is not amount based, which is the selection `[S]`.

        Args:
            target: the `targetId` of the hybridization table.
            pattern: the pattern of the hybridization.

        Returns:
            The entity, or the selection of the concentration of a species.
        """
        sbml_model = getattr(self.model, "sbml_model", None)
        if pattern is not NetworkPattern.PRE_INITIALIZATION or sbml_model is None:
            return target
        species = sbml_model.getSpecies(target)
        if species is not None and not species.getHasOnlySubstanceUnits():
            return f"[{target}]"
        return target

    # --- THE PARAMETERS ---

    def network_parameters(self, sid: str) -> list[FitParameter]:
        """Get the parameters of the fit which are elements of a network.

        Args:
            sid: id of the network.

        Returns:
            One parameter per estimated element, see
            `sbmlsim.sciml.parameters.network_fit_parameters`. The elements of
            a network which runs before the simulation are not entities of
            the model.
        """
        estimate: dict[str, bool] = {}
        bounds: dict[str, tuple[float, float]] = {}
        for entity in self._entities(sid, "parameters"):
            parameter = self._parameters.get(entity.petab_id)
            if parameter is None or entity.key is None:
                continue
            estimate[entity.key] = bool(parameter.estimate)
            bounds[entity.key] = (
                -np.inf if parameter.lb is None else float(parameter.lb),
                np.inf if parameter.ub is None else float(parameter.ub),
            )
        return network_fit_parameters(
            self.networks[sid],
            estimate=estimate,
            bounds=bounds,
            external=self.is_pre_initialization(sid),
        )

    def fit_parameters(
        self, unit_of: Callable[[str], str | None]
    ) -> list[FitParameter]:
        """Get the parameters of the fit which the networks add.

        These are the estimated elements of the networks and the estimated
        parameters of the parameter table which are inputs of networks and
        not entities of the model.

        Args:
            unit_of: gets the unit of an entity of the model.

        Returns:
            The parameters, the inputs first and the elements in the order of
            the networks.
        """
        parameters: list[FitParameter] = []
        symbols = {
            symbol
            for hybridization in self.hybridizations()
            for symbol in _symbols(hybridization.inputs)
        }
        for symbol in sorted(symbols):
            parameter = self._parameters.get(symbol)
            if (
                parameter is None
                or not parameter.estimate
                or self.model.has_entity_with_id(symbol)
            ):
                continue
            parameters.append(
                FitParameter(
                    pid=symbol,
                    start_value=_nominal(parameter),
                    lower_bound=-np.inf
                    if parameter.lb is None
                    else float(parameter.lb),
                    upper_bound=np.inf if parameter.ub is None else float(parameter.ub),
                    unit=unit_of(symbol) or ELEMENT_UNIT,
                    target=None
                    if self.is_compiled_symbol(symbol)
                    else f"{EXTERNAL_PREFIX}{symbol}",
                    scale=parameter_scale(parameter) or ParameterScaleType.LINEAR,
                )
            )
        for sid in self.networks:
            parameters.extend(self.network_parameters(sid))
        return parameters


def _symbols(inputs: Mapping[str, NetworkInput]) -> set[str]:
    """Get the symbols of the formulas of inputs."""
    return {
        symbol
        for network_input in inputs.values()
        for formula in network_input.all_formulas()
        for symbol in formula_symbols(formula)
    }


def _index(entity: NetworkEntity, shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
    """Get the index of an output of PEtab SciML in the output of the network.

    The index of PEtab SciML names an element of the output of one sample,
    the output of the network may have leading axes of the length one, i.e.
    the axis of the batch: `outputs[0][0]` of an output of the shape `(1, 1)`
    is the element `(0, 0)`.

    Args:
        entity: the output.
        shapes: the shapes of the outputs of the network.

    Returns:
        The index with a zero for every leading axis of the length one which
        it does not name. An index which does not fit is returned as it is,
        the hybridization names it.
    """
    index = entity.index or ()
    k = entity.k or 0
    if k >= len(shapes):
        return index
    missing = len(shapes[k]) - len(index)
    if missing > 0 and all(n == 1 for n in shapes[k][:missing]):
        return (0,) * missing + index
    return index


def _nominal(parameter: Any) -> float:
    """Get the nominal value of a parameter which is the value of an input.

    Args:
        parameter: parameter of the parameter table.

    Returns:
        The nominal value.

    Raises:
        SciMLProblemError: if the nominal value is not a finite number.
    """
    value = parameter.nominal_value
    if isinstance(value, int | float) and np.isfinite(value):
        return float(value)
    raise SciMLProblemError(
        f"The parameter '{parameter.id}' is the value of an input of a network "
        f"and has the nominal value '{value}', which is not a finite number"
    )


def parameter_scale(parameter: Any) -> ParameterScaleType | None:
    """Get the scale of a parameter of the parameter table.

    PEtab v2 has no scale of a parameter, it is a property of the
    optimization. The problems of PEtab SciML carry the column
    `parameterScale` of PEtab v1, which is read when it is there.

    Args:
        parameter: parameter of the parameter table.

    Returns:
        The scale, `None` for a parameter without one.

    Raises:
        SciMLProblemError: if the scale is not `lin`, `log` or `log10`.
    """
    extra = getattr(parameter, "model_extra", None) or {}
    value = extra.get("parameterScale")
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if str(value) not in SCALES:
        raise SciMLProblemError(
            f"The parameter '{parameter.id}' has the scale '{value}', which is "
            f"not one of {sorted(SCALES)}"
        )
    return SCALES[str(value)]
```

- [ ] **Step 9: Apply this patch to `src/sbmlsim/fit/petab_v2/reader.py`**

Apply this patch to `src/sbmlsim/fit/petab_v2/reader.py`:

```diff
diff --git a/src/sbmlsim/fit/petab_v2/reader.py b/src/sbmlsim/fit/petab_v2/reader.py
index 87859a9..4d9cc4d 100644
--- a/src/sbmlsim/fit/petab_v2/reader.py
+++ b/src/sbmlsim/fit/petab_v2/reader.py
@@ -16,7 +16,7 @@ is the case for a problem of another tool.

 import logging
 from pathlib import Path
-from typing import Any
+from typing import TYPE_CHECKING, Any

 import numpy as np
 import pandas as pd
@@ -39,11 +39,13 @@ from sbmlsim.fit.objects import (
     NoiseParameter,
 )
 from sbmlsim.fit.optimization import OptimizationProblem
-from sbmlsim.fit.options import FitSettings
+from sbmlsim.fit.options import FitSettings, ParameterScaleType
 from sbmlsim.fit.petab_v2.extension import (
+    SCIML_EXTENSION_ID,
     SbmlsimExtension,
     check_extensions,
     extension_of,
+    known_extensions,
 )
 from sbmlsim.fit.petab_v2.observables import (
     MODEL_SUFFIX,
@@ -60,6 +62,9 @@ from sbmlsim.simulation.timecourse import Timecourse, TimecourseSim
 from sbmlsim.task import Task
 from sbmlsim.units import UnitRegistry, UnitsInformation

+if TYPE_CHECKING:
+    from sbmlsim.fit.petab_v2.sciml import SciMLReader
+
 logger = logging.getLogger(__name__)

 #: unit of the time of a measurement if neither the `sbmlsim` extension nor
@@ -95,7 +100,14 @@ def dataset_id(observable_id: str) -> str:


 class PetabReader:
-    """Read a PEtab v2 problem as an `sbmlsim` simulation experiment and fit."""
+    """Read a PEtab v2 problem as an `sbmlsim` simulation experiment and fit.
+
+    Attributes:
+        sciml: the networks of a problem of PEtab SciML, `None` for a problem
+            without them.
+    """
+
+    sciml: "SciMLReader | None" = None

     def __init__(
         self,
@@ -103,6 +115,7 @@ class PetabReader:
         base_path: Path | None = None,
         name: str | None = None,
         derived_dir: Path | None = None,
+        sciml: Any | None = None,
     ):
         """Initialize the reader.

@@ -112,18 +125,31 @@ class PetabReader:
                 directory of its YAML file by default.
             name: name of the simulation experiment class which is created, the
                 id of the problem by default.
-            derived_dir: directory the model which carries the observables of
-                the problem is written to, next to the model by default. It is
-                only written if an observable of the problem is a formula.
+            derived_dir: directory the models the fit simulates are written
+                to, next to the model by default: the model which carries the
+                networks of the problem, `<stem>_sciml.xml`, and the model
+                which carries its observables, `<stem>_observables.xml`. A
+                model is only written if the problem has networks in the
+                model or an observable which is a formula.
+            sciml: the block of the extension of PEtab SciML, a `SciMLConfig`
+                or the dictionary of the YAML. It is the block of the
+                configuration of the problem by default, which a problem
+                carries that `petab` has read with its networks. `from_yaml`
+                reads the tables without the block and hands it over.

         Raises:
-            ValueError: if the problem has no model or no measurements, or if
-                it requires an extension `sbmlsim` does not know.
+            ImportError: if the problem has neural networks and the extra
+                `sciml` is not installed.
+            ValueError: if the problem has no model or no measurements, if
+                it requires an extension `sbmlsim` does not know, or if its
+                networks cannot be read, see
+                `sbmlsim.fit.petab_v2.sciml.SciMLReader`.
         """
         self.petab_problem = petab_problem
-        for extension_id in check_extensions(
-            getattr(petab_problem.config, "extensions", None)
-        ):
+        extensions = dict(getattr(petab_problem.config, "extensions", None) or {})
+        if sciml is not None:
+            extensions[SCIML_EXTENSION_ID] = sciml
+        for extension_id in check_extensions(extensions):
             logger.warning(
                 "The extension '%s' of the PEtab problem is not known to "
                 "`sbmlsim` and is ignored, which the problem allows: it is not "
@@ -202,6 +228,8 @@ class PetabReader:
         #: mappings are resolved again by every `initialize`
         self._noise_models: dict[str, NoiseModel] = {}

+        self.sciml = self._read_sciml(extensions)
+
     @staticmethod
     def from_yaml(yaml_file: Path, name: str | None = None) -> "PetabReader":
         """Read the problem of a PEtab YAML file.
@@ -214,15 +242,77 @@ class PetabReader:
             The reader of the problem.

         Raises:
+            ImportError: if the problem has neural networks and the extra
+                `sciml` is not installed.
             ValueError: if the problem requires an extension `sbmlsim` does
                 not know. The extensions are checked on the YAML, before
                 `petab` reads the problem: it needs the package of an
                 extension to read its files.
         """
         yaml_file = Path(yaml_file)
-        check_extensions((load_yaml(yaml_file) or {}).get("extensions"))
-        petab_problem = PetabProblem.from_yaml(yaml_file)
-        return PetabReader(petab_problem, base_path=yaml_file.parent, name=name)
+        config = load_yaml(yaml_file) or {}
+        extensions = dict(config.get("extensions") or {})
+        check_extensions(extensions)
+        sciml = extensions.pop(SCIML_EXTENSION_ID, None)
+        if sciml is None or SCIML_EXTENSION_ID not in known_extensions():
+            petab_problem = PetabProblem.from_yaml(yaml_file)
+            return PetabReader(petab_problem, base_path=yaml_file.parent, name=name)
+        # `petab` reads the networks of a problem through PyTorch, which is
+        # no dependency: it reads the tables, `SciMLReader` the networks
+        petab_problem = PetabProblem.from_yaml(
+            {**config, "extensions": extensions}, base_path=yaml_file.parent
+        )
+        return PetabReader(
+            petab_problem, base_path=yaml_file.parent, name=name, sciml=sciml
+        )
+
+    def _read_sciml(self, extensions: dict[str, Any]) -> "SciMLReader | None":
+        """Read the networks of a problem of PEtab SciML.
+
+        Args:
+            extensions: the blocks of the extensions of the problem.
+
+        Returns:
+            The reader of the networks, `None` for a problem without the
+            extension and for a problem whose extension is not required
+            while the extra `sciml` is not installed.
+
+        Raises:
+            ValueError: if the networks cannot be read.
+        """
+        if (
+            SCIML_EXTENSION_ID not in extensions
+            or SCIML_EXTENSION_ID not in known_extensions()
+        ):
+            return None
+        # the import needs the extra `sciml`
+        from sbmlsim.fit.petab_v2 import sciml
+
+        config = sciml.read_sciml_config(extensions)
+        if config is None:
+            return None
+        return sciml.SciMLReader(
+            petab_problem=self.petab_problem,
+            config=config,
+            base_path=self.base_path,
+            simulations=self._start_conditions(),
+        )
+
+    def _start_conditions(self) -> dict[str, list[str]]:
+        """Get the conditions every simulation starts with.
+
+        Returns:
+            id of the experiment -> the ids of the conditions of its first
+            period, none for the measurements which name no experiment.
+        """
+        conditions: dict[str, list[str]] = {
+            experiment_id: [] for experiment_id in self.experiment_ids
+        }
+        for experiment in self.petab_problem.experiments:
+            periods = sorted(experiment.periods, key=lambda p: p.time)
+            if periods:
+                conditions[experiment.id] = list(periods[0].condition_ids)
+        return conditions

     # --- INFORMATION OF THE EXTENSION ---

@@ -377,9 +467,12 @@ class PetabReader:
     def _model_source(self, model: Any) -> Path:
         """Get the file of the model the fit simulates.

-        The model of the problem is used as it is when every observable is an
-        entity of it, otherwise a copy which carries the observables is written
-        once, see `derived_dir`.
+        The model of the problem is used as it is when it has no networks
+        and every observable is an entity of it. Otherwise the networks of
+        the right hand side and of the observables are compiled into a copy,
+        `<stem>_sciml.xml`, and the observables which are formulas are
+        written into a copy of that, see `derived_dir`. The models are
+        written once.

         Args:
             model: model of the PEtab problem.
@@ -392,6 +485,20 @@ class PetabReader:
             return self._model_sources[path]
         derived_dir = self.derived_dir or path.parent
         source = path
+        if self.sciml is not None:
+            # the import needs the extra `sciml`
+            from sbmlsim.sciml.compiler import compile_network, compiled_path
+
+            hybridizations = [
+                hybridization
+                for hybridization in self.sciml.hybridizations()
+                if hybridization.pattern.is_compiled
+                and hybridization.model == model.model_id
+            ]
+            if hybridizations:
+                source = compile_network(
+                    path, hybridizations, compiled_path(path, derived_dir)
+                )
         formulas = self._formula_observables()
         if formulas:
             derived = derived_dir / f"{source.stem}{MODEL_SUFFIX}{source.suffix}"
@@ -511,12 +618,26 @@ class PetabReader:
             changes: dict[str, Any] = {}
             for condition_id in period.condition_ids:
                 condition = conditions.get(condition_id)
+                if (
+                    condition is None
+                    and self.sciml is not None
+                    and condition_id in self.sciml.condition_ids
+                ):
+                    # a condition of the array files, which selects the
+                    # arrays of the inputs of the networks
+                    continue
                 if condition is None:
                     raise ValueError(
                         f"The experiment '{experiment.id}' uses the condition "
                         f"'{condition_id}', which the problem does not define."
                     )
                 for change in condition.changes:
+                    if self.sciml is not None and (
+                        change.target_id in self.sciml.input_ids
+                    ):
+                        # the input of a network, which its hybridization
+                        # holds for the simulation of the experiment
+                        continue
                     if _is_number(change.target_value):
                         changes[change.target_id] = _to_float(change.target_value)
                         continue
@@ -1042,8 +1163,16 @@ class PetabReader:
         """
         versions = self._versions()
         parameters: list[FitParameter] = []
+        networks: list[FitParameter] = (
+            []
+            if self.sciml is None
+            else self.sciml.fit_parameters(lambda sid: self._unit_of(sid, None))
+        )
+        handled = {p.pid for p in networks}
+        if self.sciml is not None:
+            handled |= self.sciml.parameter_ids
         for parameter in self.petab_problem.parameters:
-            if not parameter.estimate:
+            if not parameter.estimate or parameter.id in handled:
                 continue
             target, keys = versions.get(parameter.id, (None, set()))
             if target is None and not self._in_model(parameter.id):
@@ -1084,9 +1213,30 @@ class PetabReader:
                     or self._unit_of(target or parameter.id, None),
                     target=target,
                     mappings=filter_keys(keys) if target is not None else None,
+                    scale=self._scale_of(parameter),
                 )
             )
-        return parameters
+        return parameters + networks
+
+    def _scale_of(self, parameter: Any) -> ParameterScaleType | None:
+        """Get the scale of a parameter of a problem of PEtab SciML.
+
+        PEtab v2 has no scale of a parameter. The problems of PEtab SciML
+        carry the column `parameterScale`, which is read for them.
+
+        Args:
+            parameter: parameter of the parameter table.
+
+        Returns:
+            The scale, `None` for a problem without networks and for a
+            parameter without a scale, which is the scale of the settings.
+        """
+        if self.sciml is None:
+            return None
+        # the import needs the extra `sciml`
+        from sbmlsim.fit.petab_v2.sciml import parameter_scale
+
+        return parameter_scale(parameter)

     def mapping_collections(
         self, experiment_class: type[SimulationExperiment]
@@ -1189,6 +1339,7 @@ class PetabReader:
             fit_parameters=self.fit_parameters(),
             base_path=self.base_path,
             data_path=self.base_path,
+            hybridizations=None if self.sciml is None else self.sciml.hybridizations(),
         )


```

- [ ] **Step 10: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/sciml tests/fit`
Expected: PASS

- [ ] **Step 11: Run the checks**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: `All checks passed!`, `... files already formatted`, `All checks passed!` (zero diagnostics)

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "petab: the reader of PEtab SciML problems and the gaps of hybrid problems"
```

### Task 12: The comparison of the cases `sciml_problem_import`, the baseline, the script and the cache

**Files:**
- Modify: `scripts/sciml_testsuite.py`
- Modify: `src/sbmlsim/sciml/testsuite.py`
- Modify: `src/sbmlsim/testsuite/cache.py`
- Test: `tests/data/sciml_baseline.json`
- Test: `tests/sciml/test_script.py`
- Test: `tests/sciml/test_suite_cases.py`
- Test: `tests/sciml/test_testsuite.py`
- Test: `tests/testsuite/test_cache.py`

**Interfaces:**
- Consumes: `PetabReader`, `SciMLProblemError` of task 11; `gradient`, `log_likelihood`, `nominal_parameters` of `likelihood.py`; `element_id`, `load_array_data`; `FitSettings`, `ParameterScaleType`.
- Produces: `sbmlsim.sciml.testsuite.PROBLEM_IMPORT_TOLERANCE`, `GRADIENT_STEP`, `GRADIENT_ORDER`, `MECHANISTIC`, `ProblemImportCase.settings()`, `.expected_simulations()`, `.expected_gradient()`, `.run() -> CaseResult`, `SciMLSuite.run()` with the group `sciml_problem_import`; `sbmlsim.testsuite.cache.STAGING_SUFFIX`, `STALE_AFTER`, `remove_stale(path, stale_after=STALE_AFTER) -> list[Path]`; `tests/data/sciml_baseline.json` with `n_cases: 96`.

`ProblemImportCase.run` reads the problem of a case with the reader, writes the models it derives into a temporary directory, initializes the problem with the linear scale, a fixed grid and the tolerances `PROBLEM_IMPORT_TOLERANCE = 1e-13` of the integrator, and compares the log-likelihood (`tol_llh`), the simulations of every fit mapping at its measurements (`tol_simulations`, the times must agree exactly, a reference value without a fit mapping is an error) and the gradient of five points at the step `1e-6` (`tol_grad`) with the reference values; a parameter of the reference without a fit parameter and the other way round is an error which names both, so the case compares something or says what is missing. A case which states a log-posterior is `UNSUPPORTED` before anything is read, a `SciMLProblemError` with a gap and an `UnsupportedLayerError` are `UNSUPPORTED`, any other error is `ERROR`. `SciMLSuite.run` runs the group, the baseline lists the 4 of 96 cases which do not pass and the tests of `test_testsuite.py` are one test per case. The sign of the gradient of the suite was checked on case 001: it is the gradient of the log-likelihood, `2538.78` for `alpha`, which `gradient` reproduces to `0.002`. The two gaps of the script from the notes: `unexpected_outcomes` reports a case of the baseline which was not run, and `run` without a baseline file says so instead of a bare `FileNotFoundError`. `cache.fetch` removes the staging directories a killed fetch of its target left behind, older than `STALE_AFTER` so that a fetch which is running keeps its own.

- [ ] **Step 1: Apply this patch to `tests/sciml/test_suite_cases.py`**

Apply this patch to `tests/sciml/test_suite_cases.py`:

```diff
diff --git a/tests/sciml/test_suite_cases.py b/tests/sciml/test_suite_cases.py
index 5b26f8c..8db5641 100644
--- a/tests/sciml/test_suite_cases.py
+++ b/tests/sciml/test_suite_cases.py
@@ -9,11 +9,22 @@ from pathlib import Path

 import h5py
 import numpy as np
+import pandas as pd
 import pytest
 import yaml
 from petab_sciml import Input, Layer, NNModel, NNModelStandard, Node

+from sbmlsim.fit.petab_v2.likelihood import (
+    gradient,
+    log_likelihood,
+)
+from sbmlsim.fit.petab_v2.likelihood import (
+    nominal_parameters as nominal_parameter_set,
+)
+from sbmlsim.fit.petab_v2.reader import PetabReader
 from sbmlsim.sciml.testsuite import (
+    GRADIENT_ORDER,
+    GRADIENT_STEP,
     SCIML_SUITE_COMMIT,
     CaseStatus,
     InitializationCase,
@@ -23,6 +34,8 @@ from sbmlsim.sciml.testsuite import (
     compare_arrays,
     parameter_key,
 )
+from tests.sciml.hybrid import feed_forward
+from tests.sciml.petab import write_problem

 WEIGHT = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
 BIAS = np.array([0.1, 0.2, 0.3])
@@ -582,3 +595,173 @@ def test_an_archive_which_is_not_the_suite(
     with pytest.raises(OSError, match=r"No directory 'ml_model_import'"):
         SciMLSuite.load("abc")
     assert SciMLSuite.cached("abc") is None
+
+
+def _problem_import_case(
+    tmp_path: Path,
+    llh: float | None = None,
+    gradient_of: dict[str, float] | None = None,
+    log_posterior: float | None = None,
+    frozen_layer: bool = False,
+) -> ProblemImportCase:
+    """Write a case of `sciml_problem_import` with the values of `sbmlsim`.
+
+    The reference values are calculated with `sbmlsim` itself, so a case
+    which is not changed passes; `llh`, `gradient_of` and `log_posterior`
+    replace them.
+    """
+    directory = tmp_path / "sciml_problem_import" / "001"
+    network = feed_forward()
+    mapping = [
+        ("net1_input1", "net1.inputs[0][0]"),
+        ("net1_input2", "net1.inputs[0][1]"),
+        ("net1_output1", "net1.outputs[0][0]"),
+    ]
+    parameters = []
+    if frozen_layer:
+        mapping.append(("net1_layer1", "net1.parameters[layer1]"))
+        parameters.append(
+            {"parameterId": "net1_layer1", "nominalValue": "array", "estimate": False}
+        )
+    write_problem(
+        directory / "petab",
+        networks=[network],
+        pre_initialization={"net1": False},
+        mapping=mapping,
+        hybridization=[
+            ("net1_input1", "prey"),
+            ("net1_input2", "predator"),
+            ("gamma", "net1_output1"),
+        ],
+        parameters=parameters,
+    )
+    reader = PetabReader.from_yaml(directory / "petab" / "problem.yaml")
+    reader.derived_dir = tmp_path / "derived"
+    problem = reader.to_optimization_problem()
+    case = ProblemImportCase(
+        cid="001",
+        path=directory,
+        problem_path=directory / "petab" / "problem.yaml",
+        llh=llh,
+        log_posterior=log_posterior,
+        simulation_files=[directory / "simulations.tsv"],
+        gradient_files={
+            "mech": directory / "grad_mech.tsv",
+            "net1": directory / "grad_net1.hdf5",
+        },
+        tol_llh=1e-3,
+        tol_simulations=1e-3,
+        tol_grad=0.1,
+    )
+    problem.initialize(case.settings())
+    parameters_nominal = nominal_parameter_set(problem)
+    if llh is None and log_posterior is None:
+        case = replace(case, llh=log_likelihood(problem, parameters_nominal))
+    predictions = problem.predictions(parameters_nominal.x(problem.pids))
+    rows = []
+    for k, values in predictions.items():
+        for time, value in zip(problem.x_references[k], values, strict=True):
+            rows.append(
+                {
+                    "observableId": reader.observable_id(problem.mapping_keys[k]),
+                    "experimentId": "e1",
+                    "simulation": value,
+                    "time": time,
+                }
+            )
+    pd.DataFrame(rows).to_csv(directory / "simulations.tsv", sep="\t", index=False)
+    grad = gradient(
+        problem, parameters_nominal, step=GRADIENT_STEP, order=GRADIENT_ORDER
+    )
+    grad.update(pd.Series(gradient_of or {}))
+    pd.DataFrame(
+        {
+            "parameterId": [p for p in grad.index if not p.startswith("net1__")],
+            "value": [grad[p] for p in grad.index if not p.startswith("net1__")],
+        }
+    ).to_csv(directory / "grad_mech.tsv", sep="\t", index=False)
+    arrays = {
+        layer: {name: np.zeros_like(array) for name, array in layer_arrays.items()}
+        for layer, layer_arrays in network.parameters.items()
+    }
+    for sid, (layer, name, index) in network.parameter_ids().items():
+        if sid in grad.index:
+            arrays[layer][name][index] = grad[sid]
+        else:
+            arrays[layer][name] = np.zeros(0)
+    with h5py.File(directory / "grad_net1.hdf5", "w") as f:
+        f.create_group("metadata")["pytorch_format"] = True
+        for layer, layer_arrays in arrays.items():
+            for name, array in layer_arrays.items():
+                f[f"parameters/net1/{layer}/{name}"] = array
+    return case
+
+
+def test_a_problem_import_case_passes(tmp_path: Path) -> None:
+    """A case whose reference values are the values of `sbmlsim` passes."""
+    case = _problem_import_case(tmp_path)
+    result = case.run()
+    assert result.passed, result.message
+    assert result.max_difference is not None
+    assert result.max_difference < 1e-3
+
+
+def test_a_problem_import_case_with_a_frozen_layer(tmp_path: Path) -> None:
+    """The gradient of a frozen layer is an empty array without a reference."""
+    result = _problem_import_case(tmp_path, frozen_layer=True).run()
+    assert result.passed, result.message
+
+
+def test_a_log_likelihood_outside_of_the_tolerance(tmp_path: Path) -> None:
+    """The log-likelihood is compared first and named."""
+    result = _problem_import_case(tmp_path, llh=1.0).run()
+    assert result.status is CaseStatus.TOLERANCE
+    assert result.message.startswith("the log-likelihood:")
+
+
+def test_a_gradient_outside_of_the_tolerance(tmp_path: Path) -> None:
+    """A derivative which differs by more than the tolerance is named."""
+    result = _problem_import_case(tmp_path, gradient_of={"alpha": 1e6}).run()
+    assert result.status is CaseStatus.TOLERANCE
+    assert result.message.startswith("the gradient:")
+
+
+def test_a_problem_import_case_with_priors_is_unsupported(tmp_path: Path) -> None:
+    """A case which states the log-posterior is not compared."""
+    result = _problem_import_case(tmp_path, log_posterior=-1.0).run()
+    assert result.status is CaseStatus.UNSUPPORTED
+    assert "sciml-priors" in result.message
+
+
+def test_a_problem_import_case_names_what_is_missing(tmp_path: Path) -> None:
+    """A reference value without a parameter, and the other way round, is an error."""
+    case = _problem_import_case(tmp_path)
+    df = pd.read_csv(case.gradient_files["mech"], sep="\t")
+    df.loc[len(df)] = ["kappa", 1.0]
+    df = df[df["parameterId"] != "beta"]
+    df.to_csv(case.gradient_files["mech"], sep="\t", index=False)
+    result = case.run()
+    assert result.status is CaseStatus.ERROR
+    assert "['kappa'] of the reference values are not estimated" in result.message
+    assert "['beta'] have no reference value" in result.message
+
+
+def test_a_problem_import_case_with_a_simulation_of_nothing(tmp_path: Path) -> None:
+    """A reference value of the simulations without a fit mapping is an error."""
+    case = _problem_import_case(tmp_path)
+    df = pd.read_csv(case.simulation_files[0], sep="\t")
+    df.loc[len(df)] = ["other", "e1", 1.0, 1.0]
+    df.to_csv(case.simulation_files[0], sep="\t", index=False)
+    result = case.run()
+    assert result.status is CaseStatus.ERROR
+    assert "1 of the 21 reference values" in result.message
+
+
+def test_a_problem_import_case_which_cannot_be_read(tmp_path: Path) -> None:
+    """A problem which is not read is an error which names the reason."""
+    case = _problem_import_case(tmp_path)
+    (case.path / "petab" / "net1.yaml").unlink()
+    result = case.run()
+    assert result.status is CaseStatus.ERROR
+    assert "net1.yaml" in result.message
+    assert "does not exist" in result.message
```

- [ ] **Step 2: Apply this patch to `tests/sciml/test_testsuite.py`**

Apply this patch to `tests/sciml/test_testsuite.py`:

```diff
diff --git a/tests/sciml/test_testsuite.py b/tests/sciml/test_testsuite.py
index 85fff7f..4d2e487 100644
--- a/tests/sciml/test_testsuite.py
+++ b/tests/sciml/test_testsuite.py
@@ -20,10 +20,12 @@ import pytest
 from sbmlsim.sciml.testsuite import (
     INITIALIZATION,
     MODEL_IMPORT,
+    PROBLEM_IMPORT,
     SCIML_SUITE_COMMIT,
     CaseResult,
     InitializationCase,
     ModelImportCase,
+    ProblemImportCase,
     SciMLSuite,
 )

@@ -46,6 +48,7 @@ def _case_ids(group: str) -> list[str]:

 MODEL_IMPORT_IDS = _case_ids(MODEL_IMPORT)
 INITIALIZATION_IDS = _case_ids(INITIALIZATION)
+PROBLEM_IMPORT_IDS = _case_ids(PROBLEM_IMPORT)


 @pytest.fixture(scope="session")
@@ -97,17 +100,28 @@ def test_initialization(cid: str, suite: SciMLSuite, baseline: dict) -> None:
     _check(case.run(), baseline)


+@pytest.mark.parametrize("cid", PROBLEM_IMPORT_IDS)
+def test_problem_import(cid: str, suite: SciMLSuite, baseline: dict) -> None:
+    """The log-likelihood, the simulations and the gradient of the case agree."""
+    case = ProblemImportCase.from_directory(suite.path / PROBLEM_IMPORT / cid)
+    _check(case.run(), baseline)
+
+
 def test_the_baseline_matches_the_suite(suite: SciMLSuite, baseline: dict) -> None:
     """The baseline was recorded for the commit which is pinned and cached."""
     assert baseline["suite_commit"] == SCIML_SUITE_COMMIT == suite.commit
     assert [f"{i:03d}" for i in range(1, 55)] == MODEL_IMPORT_IDS
     assert INITIALIZATION_IDS == ["001", "002", "003"]
-    assert baseline["n_cases"] == len(MODEL_IMPORT_IDS) + len(INITIALIZATION_IDS)
+    assert [f"{i:03d}" for i in range(1, 40)] == PROBLEM_IMPORT_IDS
+    assert baseline["n_cases"] == (
+        len(MODEL_IMPORT_IDS) + len(INITIALIZATION_IDS) + len(PROBLEM_IMPORT_IDS)
+    )
     assert baseline["n_passed"] == baseline["n_cases"] - len(
         baseline["expected_failures"]
     )
     keys = {f"{MODEL_IMPORT}/{cid}" for cid in MODEL_IMPORT_IDS}
     keys |= {f"{INITIALIZATION}/{cid}" for cid in INITIALIZATION_IDS}
+    keys |= {f"{PROBLEM_IMPORT}/{cid}" for cid in PROBLEM_IMPORT_IDS}
     assert set(baseline["expected_failures"]) <= keys


```

- [ ] **Step 3: Apply this patch to `tests/sciml/test_script.py`**

Apply this patch to `tests/sciml/test_script.py`:

```diff
diff --git a/tests/sciml/test_script.py b/tests/sciml/test_script.py
index 594656e..1df7c2b 100644
--- a/tests/sciml/test_script.py
+++ b/tests/sciml/test_script.py
@@ -93,6 +93,35 @@ def test_run_fails_when_a_case_differs_from_the_baseline(
     assert script.main(["run"]) == code


+def test_a_case_of_the_baseline_which_was_not_run(script: ModuleType) -> None:
+    """A baseline entry of a case which is not in the results is reported."""
+    baseline = {
+        "expected_failures": {
+            "ml_model_import/002": {"status": "tolerance", "reason": REASON},
+            "ml_model_import/099": {"status": "error", "reason": REASON},
+        }
+    }
+    assert script.unexpected_outcomes(RESULTS, baseline) == [
+        "ml_model_import/099: 'error' -> not run"
+    ]
+
+
+def test_run_without_a_baseline(
+    script: ModuleType,
+    monkeypatch: pytest.MonkeyPatch,
+    tmp_path: Path,
+    capsys: pytest.CaptureFixture[str],
+) -> None:
+    """`run` says that the baseline is missing instead of failing on the file."""
+    monkeypatch.setattr(script.SciMLSuite, "load", lambda commit: FakeSuite(RESULTS))
+    monkeypatch.setattr(script, "BASELINE_PATH", tmp_path / "missing.json")
+    assert script.main(["run"]) == 1
+    assert "does not exist" in capsys.readouterr().out
+    # `baseline` writes it
+    assert script.main(["baseline"]) == 0
+    assert (tmp_path / "missing.json").is_file()
+
+
 def test_the_baseline_keeps_a_reason_of_the_same_status(
     script: ModuleType, tmp_path: Path
 ) -> None:
```

- [ ] **Step 4: Apply this patch to `tests/testsuite/test_cache.py`**

Apply this patch to `tests/testsuite/test_cache.py`:

```diff
diff --git a/tests/testsuite/test_cache.py b/tests/testsuite/test_cache.py
index 82df9fd..eeab3c9 100644
--- a/tests/testsuite/test_cache.py
+++ b/tests/testsuite/test_cache.py
@@ -1,6 +1,8 @@
 """Tests of the download and the cache of a test suite."""

 import logging
+import os
+import time
 import zipfile
 from pathlib import Path

@@ -151,3 +153,29 @@ def test_a_member_outside_of_the_archive_stays_inside(

     assert (target / "001" / "a.txt").read_text() == "a"
     assert list(tmp_path.rglob("evil.txt")) == []
+
+
+def test_a_stale_staging_directory_is_removed(tmp_path: Path) -> None:
+    """A fetch removes what a killed fetch of its target left behind."""
+    url = _archive(tmp_path / "suite.zip", {"suite-1.0/cases/001/a.txt": "a"})
+    target = tmp_path / "cache" / "suite" / "1.0"
+    target.parent.mkdir(parents=True)
+    stale = target.parent / ".1.0.abc.incomplete"
+    stale.mkdir()
+    (stale / "archive.zip").write_text("x")
+    old = time.time() - 2 * cache.STALE_AFTER
+    os.utime(stale, (old, old))
+    fresh = target.parent / ".1.0.def.incomplete"
+    fresh.mkdir()
+    other = target.parent / ".2.0.abc.incomplete"
+    other.mkdir()
+    os.utime(other, (old, old))
+
+    cache.fetch(url, target, select=lambda staging: staging / "suite-1.0/cases")
+
+    assert sorted(p.name for p in target.parent.iterdir()) == [
+        ".1.0.def.incomplete",
+        ".2.0.abc.incomplete",
+        "1.0",
+    ]
+    assert cache.remove_stale(target / "x", stale_after=0.0) == []
```

- [ ] **Step 5: Apply this patch to `tests/data/sciml_baseline.json`**

Apply this patch to `tests/data/sciml_baseline.json`:

```diff
diff --git a/tests/data/sciml_baseline.json b/tests/data/sciml_baseline.json
index fa10ce5..453aa5f 100644
--- a/tests/data/sciml_baseline.json
+++ b/tests/data/sciml_baseline.json
@@ -1,11 +1,23 @@
 {
   "suite_commit": "0622bbfc5e12eb9b482659eabd1756ca0e87dfc8",
-  "n_cases": 57,
-  "n_passed": 56,
+  "n_cases": 96,
+  "n_passed": 92,
   "expected_failures": {
     "ml_model_import/020": {
       "status": "tolerance",
       "reason": "AlphaDropout: the reference values are the mean of 40000 forward passes in training mode, which for AlphaDropout is not the input. sbmlsim evaluates dropout in evaluation mode, where it is the identity (gap sciml-training-mode)"
+    },
+    "sciml_problem_import/032": {
+      "status": "unsupported",
+      "reason": "priors on the parameters of the network and of the model: the case states the log-posterior, which the log-likelihood of sbmlsim is not; priors wait for issue #190 (gap sciml-priors)"
+    },
+    "sciml_problem_import/033": {
+      "status": "unsupported",
+      "reason": "priors on the parameters of the network and of the model: the case states the log-posterior, which the log-likelihood of sbmlsim is not; priors wait for issue #190 (gap sciml-priors)"
+    },
+    "sciml_problem_import/034": {
+      "status": "unsupported",
+      "reason": "priors on the parameters of the network and of the model: the case states the log-posterior, which the log-likelihood of sbmlsim is not; priors wait for issue #190 (gap sciml-priors)"
     }
   }
 }
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `uv run pytest -q -x tests/sciml/test_suite_cases.py tests/sciml/test_script.py tests/testsuite/test_cache.py`
Expected: FAIL at collection: `ImportError: cannot import name 'GRADIENT_ORDER' from 'sbmlsim.sciml.testsuite'`; `test_a_stale_staging_directory_is_removed` (`AttributeError: module 'sbmlsim.testsuite.cache' has no attribute 'STALE_AFTER'`); `test_a_case_of_the_baseline_which_was_not_run` (the line is missing)

- [ ] **Step 7: Apply this patch to `src/sbmlsim/sciml/testsuite.py`**

Apply this patch to `src/sbmlsim/sciml/testsuite.py`:

```diff
diff --git a/src/sbmlsim/sciml/testsuite.py b/src/sbmlsim/sciml/testsuite.py
index 5365a58..f2b9031 100644
--- a/src/sbmlsim/sciml/testsuite.py
+++ b/src/sbmlsim/sciml/testsuite.py
@@ -21,6 +21,7 @@ name the axes of the arrays as they are stored, nothing is permuted.
 from __future__ import annotations

 import logging
+import tempfile
 from collections.abc import Iterator
 from dataclasses import dataclass, field, replace
 from enum import StrEnum
@@ -29,13 +30,29 @@ from typing import Any

 import h5py
 import numpy as np
+import pandas as pd
 import yaml
 from petab.v2.core import MappingTable, ParameterTable, ProblemConfig
 from petab.v2.extensions.sciml import SciMLConfig
 from petab_sciml.constants import ARRAY

+from sbmlsim.fit.options import FitSettings, ParameterScaleType
+from sbmlsim.fit.petab_v2.likelihood import (
+    gradient,
+    log_likelihood,
+)
+from sbmlsim.fit.petab_v2.likelihood import (
+    nominal_parameters as nominal_parameter_set,
+)
+from sbmlsim.fit.petab_v2.reader import DEFAULT_EXPERIMENT, PetabReader
+from sbmlsim.fit.petab_v2.sciml import SciMLProblemError
 from sbmlsim.sciml.errors import NetworkImportError, UnsupportedLayerError
-from sbmlsim.sciml.network import Network, NetworkParameters, load_array_data
+from sbmlsim.sciml.network import (
+    Network,
+    NetworkParameters,
+    element_id,
+    load_array_data,
+)
 from sbmlsim.sciml.parameters import nominal_parameters
 from sbmlsim.testsuite import cache

@@ -67,6 +84,22 @@ MODEL_IMPORT_TOLERANCE = 1e-3
 #: mean of forward passes in training mode
 DROPOUT_TOLERANCE = 1e-2

+#: absolute and relative tolerance of the integrator in the cases of the
+#: group `sciml_problem_import`. The reference values are simulated with the
+#: tolerances `1e-12`, and the error of a simulation enters the gradient
+#: multiplied by the sensitivities of the log-likelihood, which are of the
+#: order `1e4`
+PROBLEM_IMPORT_TOLERANCE = 1e-13
+
+#: relative step and order of the differences of the gradient. The reference
+#: values are central differences of five points
+GRADIENT_STEP = 1e-6
+GRADIENT_ORDER = 4
+
+#: the key of the gradient of the parameters which are not elements of a
+#: network in the `grad_files` of a case
+MECHANISTIC = "mech"
+

 class CaseStatus(StrEnum):
     """The outcome of a case."""
@@ -593,9 +626,11 @@ class InitializationCase:
 class ProblemImportCase:
     """A case of the group `sciml_problem_import`.

-    The case reads its files. Its comparison needs the log-likelihood, the
-    simulation and the gradient of a hybrid problem and is not part of this
-    class yet.
+    A case is a PEtab SciML problem with the log-likelihood, the simulations
+    at the measurements and the gradient of the log-likelihood at the nominal
+    values of its parameters. The reference values of the gradient are the
+    central differences of five points of a simulation with the tolerances
+    `1e-12`.

     Attributes:
         cid: the number of the case, e.g. `001`.
@@ -666,6 +701,228 @@ class ProblemImportCase:
             tol_grad=float(required(solutions, "tol_grad", path)),
         )

+    def settings(self) -> FitSettings:
+        """Get the settings the problem of the case is initialized with.
+
+        Returns:
+            Settings with the linear scale, a fixed grid and the tolerances
+            `PROBLEM_IMPORT_TOLERANCE`: two simulations on a variable grid
+            differ by more than a difference of the gradient resolves.
+        """
+        return FitSettings(
+            parameter_scale=ParameterScaleType.LINEAR,
+            variable_step_size=False,
+            absolute_tolerance=PROBLEM_IMPORT_TOLERANCE,
+            relative_tolerance=PROBLEM_IMPORT_TOLERANCE,
+        )
+
+    def expected_simulations(self) -> pd.DataFrame:
+        """Read the reference values of the simulations.
+
+        Returns:
+            The rows of the simulation files with the columns `observableId`,
+            `experimentId`, `time` and `simulation`.
+
+        Raises:
+            ValueError: if the case has no simulation file, or if a file lacks
+                a column.
+        """
+        if not self.simulation_files:
+            raise ValueError(f"The case '{self.path}' lists no simulation files")
+        frames = [
+            pd.read_csv(path, sep="\t", float_precision="round_trip")
+            for path in self.simulation_files
+        ]
+        df = pd.concat(frames, ignore_index=True)
+        missing = sorted(
+            {"observableId", "experimentId", "time", "simulation"} - set(df.columns)
+        )
+        if missing:
+            raise ValueError(
+                f"The simulation files of the case '{self.path}' have no columns "
+                f"{missing}"
+            )
+        return df
+
+    def expected_gradient(self) -> dict[str, float]:
+        """Read the reference values of the gradient.
+
+        Returns:
+            id of the parameter -> derivative of the log-likelihood. The id of
+            an element of a network is its id in `sbmlsim`, see
+            `sbmlsim.sciml.network.element_id`. An array which the file
+            stores as an empty array is frozen and has no derivative.
+
+        Raises:
+            ValueError: if the case has no gradient of the mechanistic
+                parameters, or if a file cannot be read.
+        """
+        if MECHANISTIC not in self.gradient_files:
+            raise ValueError(
+                f"The case '{self.path}' has no gradient of the mechanistic "
+                f"parameters, the key '{MECHANISTIC}' of 'grad_files'"
+            )
+        expected: dict[str, float] = {}
+        for key, path in self.gradient_files.items():
+            if key == MECHANISTIC:
+                df = pd.read_csv(path, sep="\t", float_precision="round_trip")
+                for pid, value in zip(df["parameterId"], df["value"], strict=True):
+                    expected[str(pid)] = float(value)
+                continue
+            data = load_array_data(path)
+            if key not in data.parameters:
+                raise ValueError(
+                    f"The gradient file '{path}' has no arrays of the network "
+                    f"'{key}', it has {sorted(data.parameters)}"
+                )
+            for layer, arrays in data.parameters[key].items():
+                for name, values in arrays.items():
+                    array = np.asarray(values, dtype=float)
+                    for index in np.ndindex(array.shape):
+                        expected[element_id(key, layer, name, index)] = float(
+                            array[index]
+                        )
+        return expected
+
+    def run(self) -> CaseResult:
+        """Read the problem and compare its values with the reference values.
+
+        The log-likelihood, the simulations at the measurements and the
+        gradient of the log-likelihood are compared, each with its tolerance.
+        The models the problem is simulated with are written into a
+        temporary directory.
+
+        Returns:
+            The outcome of the case. It does not raise: a problem with a gap
+            or a layer without an implementation is `UNSUPPORTED` and any
+            other error is `ERROR`.
+        """
+        if self.llh is None:
+            return CaseResult(
+                PROBLEM_IMPORT,
+                self.cid,
+                CaseStatus.UNSUPPORTED,
+                "the case states the log-posterior of a problem with priors, "
+                "which are not a part of the log-likelihood (gap 'sciml-priors')",
+            )
+        try:
+            with tempfile.TemporaryDirectory(prefix="sbmlsim-sciml-") as directory:
+                outcomes = self._compare(Path(directory))
+        except (UnsupportedLayerError, SciMLProblemError) as err:
+            if isinstance(err, SciMLProblemError) and err.gap is None:
+                return CaseResult(PROBLEM_IMPORT, self.cid, CaseStatus.ERROR, str(err))
+            return CaseResult(
+                PROBLEM_IMPORT, self.cid, CaseStatus.UNSUPPORTED, str(err)
+            )
+        except Exception as err:
+            return CaseResult(
+                PROBLEM_IMPORT,
+                self.cid,
+                CaseStatus.ERROR,
+                f"{type(err).__name__}: {err}",
+            )
+        return _worst(PROBLEM_IMPORT, self.cid, outcomes)
+
+    def _compare(self, directory: Path) -> list[tuple[CaseStatus, str, float | None]]:
+        """Compare the values of the problem with the reference values.
+
+        Args:
+            directory: the directory the models of the problem are written to.
+
+        Returns:
+            Outcome, message and largest difference of every comparison.
+        """
+        reader = PetabReader.from_yaml(self.problem_path)
+        reader.derived_dir = directory
+        problem = reader.to_optimization_problem(opid=f"case_{self.cid}")
+        problem.initialize(self.settings())
+        parameters = nominal_parameter_set(problem)
+
+        def outcome(
+            what: str, observed: Any, expected: Any, tolerance: float
+        ) -> tuple[CaseStatus, str, float | None]:
+            status, message, difference = compare_arrays(
+                np.asarray(observed, dtype=float),
+                np.asarray(expected, dtype=float),
+                tolerance,
+            )
+            return status, f"{what}: {message}" if message else "", difference
+
+        outcomes = [
+            outcome(
+                "the log-likelihood",
+                log_likelihood(problem, parameters),
+                self.llh,
+                self.tol_llh,
+            )
+        ]
+
+        expected = self.expected_simulations()
+        predictions = problem.predictions(parameters.x(problem.pids))
+        compared = 0
+        for k, prediction in predictions.items():
+            key = problem.mapping_keys[k]
+            observable_id = reader.observable_id(key)
+            experiment_id = problem.simulation_keys[k]
+            rows = expected[expected["observableId"] == observable_id]
+            if experiment_id != DEFAULT_EXPERIMENT:
+                rows = rows[rows["experimentId"] == experiment_id]
+            rows = rows.sort_values("time")
+            compared += len(rows)
+            what = f"the simulations of '{observable_id}' in '{experiment_id}'"
+            outcomes.append(
+                outcome(
+                    f"{what}, times",
+                    problem.x_references[k],
+                    rows["time"].to_numpy(),
+                    0.0,
+                )
+            )
+            outcomes.append(
+                outcome(
+                    what,
+                    prediction,
+                    rows["simulation"].to_numpy(),
+                    self.tol_simulations,
+                )
+            )
+        if compared != len(expected):
+            outcomes.append(
+                (
+                    CaseStatus.ERROR,
+                    f"{len(expected) - compared} of the {len(expected)} reference "
+                    f"values of the simulations belong to no fit mapping",
+                    None,
+                )
+            )
+
+        observed = gradient(
+            problem, parameters, step=GRADIENT_STEP, order=GRADIENT_ORDER
+        )
+        reference = self.expected_gradient()
+        missing = sorted(set(reference) - set(observed.index))
+        extra = sorted(set(observed.index) - set(reference))
+        if missing or extra:
+            outcomes.append(
+                (
+                    CaseStatus.ERROR,
+                    f"the gradient: the parameters {missing} of the reference "
+                    f"values are not estimated, and the estimated parameters "
+                    f"{extra} have no reference value",
+                    None,
+                )
+            )
+        shared = [pid for pid in observed.index if pid in reference]
+        outcomes.append(
+            outcome(
+                "the gradient",
+                [observed[pid] for pid in shared],
+                [reference[pid] for pid in shared],
+                self.tol_grad,
+            )
+        )
+        return outcomes
+

 @dataclass(frozen=True)
 class SciMLSuite:
@@ -785,12 +1042,13 @@ class SciMLSuite:
             yield ProblemImportCase.from_directory(self.path / PROBLEM_IMPORT / cid)

     def run(self) -> list[CaseResult]:
-        """Run the cases which are compared.
+        """Run the cases of the suite.

         Returns:
-            The results of the groups `ml_model_import` and `initialization`,
-            in this order.
+            The results of the groups `ml_model_import`, `initialization` and
+            `sciml_problem_import`, in this order.
         """
         results = [case.run() for case in self.model_import_cases()]
         results.extend(case.run() for case in self.initialization_cases())
+        results.extend(case.run() for case in self.problem_import_cases())
         return results
```

- [ ] **Step 8: Apply this patch to `src/sbmlsim/testsuite/cache.py`**

Apply this patch to `src/sbmlsim/testsuite/cache.py`:

```diff
diff --git a/src/sbmlsim/testsuite/cache.py b/src/sbmlsim/testsuite/cache.py
index 2ffbd30..418e303 100644
--- a/src/sbmlsim/testsuite/cache.py
+++ b/src/sbmlsim/testsuite/cache.py
@@ -14,6 +14,7 @@ import logging
 import os
 import shutil
 import tempfile
+import time
 import urllib.request
 import zipfile
 from collections.abc import Callable
@@ -21,6 +22,13 @@ from pathlib import Path

 logger = logging.getLogger(__name__)

+#: suffix of the staging directory of a fetch
+STAGING_SUFFIX = ".incomplete"
+
+#: age in seconds after which the staging directory of a fetch is stale, i.e.
+#: left behind by a fetch which was killed. A fetch takes minutes
+STALE_AFTER = 6 * 3600.0
+

 def cache_root() -> Path:
     """Get the directory the test suites are cached in.
@@ -75,8 +83,11 @@ def fetch(url: str, path: Path, select: Callable[[Path], Path]) -> Path:
             does not find the cases, or if they cannot be moved into place.
     """
     path.parent.mkdir(parents=True, exist_ok=True)
+    remove_stale(path)
     staging = Path(
-        tempfile.mkdtemp(prefix=f".{path.name}.", suffix=".incomplete", dir=path.parent)
+        tempfile.mkdtemp(
+            prefix=f".{path.name}.", suffix=STAGING_SUFFIX, dir=path.parent
+        )
     )
     archive = staging / "archive.zip"
     try:
@@ -95,3 +106,33 @@ def fetch(url: str, path: Path, select: Callable[[Path], Path]) -> Path:
     finally:
         shutil.rmtree(staging, ignore_errors=True)
     return path
+
+
+def remove_stale(path: Path, stale_after: float = STALE_AFTER) -> list[Path]:
+    """Remove the staging directories a killed fetch of a target left behind.
+
+    A fetch which is running has a staging directory as well, which is why
+    only a directory older than `stale_after` is removed.
+
+    Args:
+        path: the target of the fetch.
+        stale_after: age in seconds after which a staging directory is stale.
+
+    Returns:
+        The directories which were removed.
+    """
+    removed: list[Path] = []
+    now = time.time()
+    for staging in path.parent.glob(f".{path.name}.*{STAGING_SUFFIX}"):
+        if not staging.is_dir():
+            continue
+        try:
+            age = now - staging.stat().st_mtime
+        except OSError:
+            continue
+        if age < stale_after:
+            continue
+        logger.info("Removing the stale staging directory '%s'", staging)
+        shutil.rmtree(staging, ignore_errors=True)
+        removed.append(staging)
+    return removed
```

- [ ] **Step 9: Apply this patch to `scripts/sciml_testsuite.py`**

Apply this patch to `scripts/sciml_testsuite.py`:

```diff
diff --git a/scripts/sciml_testsuite.py b/scripts/sciml_testsuite.py
index fc54986..b57b3e5 100644
--- a/scripts/sciml_testsuite.py
+++ b/scripts/sciml_testsuite.py
@@ -64,7 +64,8 @@ def unexpected_outcomes(results: list[CaseResult], baseline: dict) -> list[str]:
         baseline: the content of the baseline.

     Returns:
-        One line per case, `<key>: '<expected>' -> '<observed>'`.
+        One line per case, `<key>: '<expected>' -> '<observed>'`, and one per
+        case of the baseline which was not run.
     """
     expected_failures = baseline["expected_failures"]
     lines: list[str] = []
@@ -73,6 +74,10 @@ def unexpected_outcomes(results: list[CaseResult], baseline: dict) -> list[str]:
         expected = CaseStatus.PASS.value if recorded is None else recorded["status"]
         if result.status.value != expected:
             lines.append(f"{result.key}: '{expected}' -> '{result.status.value}'")
+    keys = {result.key for result in results}
+    for key, recorded in expected_failures.items():
+        if key not in keys:
+            lines.append(f"{key}: '{recorded['status']}' -> not run")
     return lines


@@ -141,6 +146,12 @@ def main(argv: list[str] | None = None) -> int:
         console.print(f"PEtab SciML test suite '{suite.commit}': {suite.path}")
         return 0

+    if args.command == "run" and not BASELINE_PATH.is_file():
+        console.print(
+            f"[red]The baseline {BASELINE_PATH} does not exist, write it with "
+            f"`baseline` first[/red]"
+        )
+        return 1
     results = run(suite)
     if args.command == "baseline":
         write_baseline(results, suite, BASELINE_PATH)
```

- [ ] **Step 10: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/sciml tests/testsuite`
Expected: PASS

- [ ] **Step 11: Run the cases of the test suite**

Run: `uv run python scripts/sciml_testsuite.py download && uv run pytest -q -m sciml_testsuite tests/sciml/test_testsuite.py`
Expected: 98 passed (54 + 3 + 39 cases and the two tests of the baseline); `uv run python scripts/sciml_testsuite.py run` prints `92/96 cases pass` and exits with `0`

- [ ] **Step 12: Run the checks**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: `All checks passed!`, `... files already formatted`, `All checks passed!` (zero diagnostics)

- [ ] **Step 13: Commit**

```bash
git add -A
git commit -m "testsuite: the cases sciml_problem_import compare the log-likelihood, the simulations and the gradient"
```

### Task 13: The documentation of `docs/petab.md`

**Files:**
- Modify: `docs/petab.md`

**Interfaces:**
- Consumes: everything of the tasks before.
- Produces: nothing.

`docs/petab.md` stays true: the fit mappings of a problem which is read, the translation of the math of an observable, the scale of a parameter, the five point gradient and the one sided difference next to a bound, the extension `sciml` and the extra, and a section on hybrid problems with the three patterns, the compiled model, the test suite and the gaps. The API page, the examples and the table of the layers are phase 4. The measured cost of a compiled network goes into the section: a `Linear`-`tanh`-`Linear` network of 5 units per layer (51 elements) loads in `0.1 s` and simulates in `1.4 ms` against `0.6 ms` for the model without it, 20 units (501 elements) load in `2 s` and simulate in `9 ms`, 50 units (2751 elements) load in `44 s` and simulate in `32 ms`: roadrunner compiles every rule, so the time to load the model grows faster than the number of units.

- [ ] **Step 1: Apply this patch to `docs/petab.md`**

Apply this patch to `docs/petab.md`:

```diff
diff --git a/docs/petab.md b/docs/petab.md
index 146d25c..ea1b0b7 100644
--- a/docs/petab.md
+++ b/docs/petab.md
@@ -48,7 +48,7 @@ opt_result = run_optimization(problem=problem, settings=settings, size=5, n_core

 The reader builds a `SimulationExperiment` from the tables, in the same way the SED-ML parser builds one from a document: the models of the problem are its models, its experiments are the timecourse simulations, its observables are the fit mappings and the measurements of an observable are its dataset. `PetabReader` gives access to the parts.

-An experiment of PEtab is a simulation with its conditions and the observables which are measured in it, which is what a `FitMappingCollection` is, so the reader gives one collection back per experiment of the problem. A fit which is written and read again therefore comes back as one collection per simulation rather than as the collections it was defined with, which is the same fit of the same data.
+An experiment of PEtab is a simulation with its conditions and the observables which are measured in it, which is what a `FitMappingCollection` is, so the reader gives one collection back per experiment of the problem. A fit mapping is an observable in an experiment: it is named after its observable, and `<observable>_<experiment>` when the observable is measured in several experiments. The math of an observable formula is translated into the math of SBML, e.g. `log` of PEtab is the natural logarithm and `log` of a formula of SBML is the one to the base 10. A fit which is written and read again therefore comes back as one collection per simulation rather than as the collections it was defined with, which is the same fit of the same data.

 ## What PEtab does not express

@@ -70,7 +70,7 @@ console.print(gaps_table(gaps_of_problem(problem)))

 A gap is of one of three kinds:

-- **extension**: PEtab has no place for it and the extension carries it, i.e. the units, the settings of the fit, the kind of every mapping, the output grid of the timecourses, the metadata of a curve and the settings of the integrator. The scale the optimizer searches in is part of the settings, which is where PEtab v2 puts it as well: it removed the `parameterScale` of its parameter table because the scale is a property of the optimization and not of the problem, so the bounds and the start values are written on the linear scale. The round trip through `sbmlsim` is exact, a tool which reads the problem without the extension gets a valid PEtab problem which does not know these things.
+- **extension**: PEtab has no place for it and the extension carries it, i.e. the units, the settings of the fit, the kind of every mapping, the output grid of the timecourses, the metadata of a curve and the settings of the integrator. The scale the optimizer searches in is part of the settings, which is where PEtab v2 puts it as well: it removed the `parameterScale` of its parameter table because the scale is a property of the optimization and not of the problem, so the bounds and the start values are written on the linear scale. A parameter with a scale of its own (`FitParameter.scale`), e.g. an element of a neural network on the linear scale, is the `sciml-parameter-scale` gap. The round trip through `sbmlsim` is exact, a tool which reads the problem without the extension gets a valid PEtab problem which does not know these things.
 - **lossy**: the information is transformed. The noise model of PEtab is its objective and is only evaluated by `sbmlsim`, the weights of `sbmlsim` are not the standard deviation PEtab uses as the noise, a pre-simulation of a finite duration is not the pre-equilibration of PEtab, and the reader builds one simulation experiment for a problem, so a task selects the observables of the whole problem rather than those of the experiment a measurement came from.
 - **unsupported**: the export raises. A structural model change (`ModelChange.clamp_species`), an observable which is a python function and a mapping whose x is not the time of the simulation have no PEtab representation. The reader raises for a problem which requires the extension of another tool.

@@ -128,11 +128,17 @@ A parameter of the noise which the problem estimates is therefore evaluated and

 The log-likelihood is the one of the measurements, so a problem whose residuals are relative to the baseline of a curve (`ABSOLUTE_TO_BASELINE`, `NORMALIZED_TO_BASELINE`) has none and `log_likelihood` raises. The logarithmic distributions require positive measurements and simulations.

-`gradient` is the central finite difference of the log-likelihood on the linear scale, with the step `step * max(|x|, 1)` for every parameter of the fit, and returns a `pandas.Series` indexed by the ids of the parameters. The model is not simulated outside the bounds of a parameter: a step which would leave them is shrunk to the distance to the nearer bound, e.g. for a parameter smaller than the step, and a parameter at one of its bounds has the one sided difference into the bounds; a parameter outside its bounds is an error. A difference divides the error of a simulation by the step. With `variable_step_size=True` the data is interpolated on the steps of the integrator, which differ between two simulations, so the simulations of one problem differ by `1e-6` however tight the tolerances are and the gradient is noise; `gradient` logs a warning in this case.
+`gradient` is the central finite difference of the log-likelihood on the linear scale, with the step `step * max(|x|, 1)` for every parameter of the fit, and returns a `pandas.Series` indexed by the ids of the parameters. The difference is of three points by default and of five points with `order=4`, whose error grows with the fourth power of the step instead of the square: the log-likelihood of a model which oscillates is strongly curved in the parameters of a neural network, and the reference values of the PEtab SciML test suite are differences of five points. The model is not simulated outside the bounds of a parameter: next to a bound the difference is one sided with the full step, because a step which is shrunk to the distance to the bound divides the error of a simulation by a vanishing step; a parameter outside its bounds is an error. A difference divides the error of a simulation by the step. With `variable_step_size=True` the data is interpolated on the steps of the integrator, which differ between two simulations, so the simulations of one problem differ by `1e-6` however tight the tolerances are and the gradient is noise; `gradient` logs a warning in this case.

 ## Extensions of other tools

-A problem carries the extensions of any tool in the `extensions` block of its YAML, and `required` says whether the problem can be interpreted without one of them. The reader knows the `sbmlsim` extension. A problem which requires another extension is not read: `from_petab` raises a `ValueError` which names the extension, before `petab` reads the files of the problem. An extension which is not required is ignored and the log says so.
+A problem carries the extensions of any tool in the `extensions` block of its YAML, and `required` says whether the problem can be interpreted without one of them. The reader knows the `sbmlsim` extension and, with the extra `sciml` installed, the `sciml` extension of PEtab SciML. A problem which requires another extension is not read: `from_petab` raises a `ValueError` which names the extension, before `petab` reads the files of the problem; a problem which requires `sciml` without the extra raises an `ImportError` which names `pip install sbmlsim[sciml]`. An extension which is not required is ignored and the log says so.
+
+## Hybrid problems of PEtab SciML
+
+[PEtab SciML](https://github.com/PEtab-dev/petab_sciml) is the extension of PEtab v2 for hybrid problems, in which the model is combined with neural networks. `sbmlsim.sciml` is the native half: a `Network` is the architecture and the arrays of a network, which `Network.forward` evaluates with numpy, and a `Hybridization` says where it sits. A network before the simulation (`pre_initialization`) is evaluated once per simulation and its outputs are changes of the simulation, e.g. a parameter or an initial value; a network in the right hand side or in an observable is compiled into the model by `compile_network` as parameters with assignment rules, which roadrunner evaluates at every step. The reader translates a problem with the `sciml` extension into these objects: the model with the networks is written next to the model as `<stem>_sciml.xml`, the elements of the networks are parameters of the fit on the linear scale, and `from_petab` gives a problem which is simulated, evaluated and fitted like any other. The problems of PEtab SciML carry the `parameterScale` column of PEtab v1, which becomes the scale of the parameter. The networks are read from the NN YAML and the array files without `torch`, which `petab` needs for them. The cost of a compiled network is the cost of its rules: a `Linear`-`tanh`-`Linear` network of 5 units per layer (51 elements) loads in `0.1 s` and simulates 101 points in `1.4 ms` against `0.6 ms` for the model without it, 20 units per layer (501 elements) load in `2 s` and simulate in `9 ms`, and 50 units per layer (2751 elements) load in `44 s` and simulate in `32 ms`; roadrunner compiles every rule when it loads the model, so the time to load grows faster than the number of units, and a large network belongs before the simulation.
+
+The cases `sciml_problem_import` of the [PEtab SciML test suite](https://github.com/PEtab-dev/petab_sciml_testsuite) compare the log-likelihood, the simulations at the measurements and the gradient with the reference values: `tox r -e sciml` runs them, and `tests/data/sciml_baseline.json` lists the cases which do not pass with their reason. The cases with priors on the parameters of a network state a log-posterior, which the log-likelihood is not, and wait for issue #190 (`sciml-priors`). A network in the format `pytorch`, `equinox` or `lux.jl` (`sciml-model-format`), a layer without MathML in the right hand side (`sciml-layer-sbml`), and the training mode of dropout and the normalization layers (`sciml-training-mode`) are the other gaps. The exporter, the report and the examples of hybrid problems are not part of this release.

 ## The example

@@ -163,7 +169,7 @@ python -m examples.petab.benchmark --runs=8 --no-identifiability

 The example reads the converted problem, reports what PEtab cannot express about it, fits it, analyses the identifiability of the fitted parameters and writes the report of the fit. For `Perelson_Science1996` the clearance rate `c` of the virions is identifiable and the loss rate `delta` of the infected cells is not identifiable towards zero.

-The observables of `Boehm_JProteomeRes2014` are formulas over several species, e.g. `(100 * pApB + 200 * pApA * specC17) / (...)`, which is not what roadrunner selects. `sbmlsim.fit.petab_v2.observables` therefore writes a copy of the model in which every such observable is a parameter with an assignment rule, and the fit selects that parameter: the math of PEtab is the math of the model, so the formula is the rule. Simulated at the nominal parameters of the problem, the observables agree with the `simulatedData` of the collection to `2e-4` at a relative tolerance of `1e-9` of the integrator.
+The observables of `Boehm_JProteomeRes2014` are formulas over several species, e.g. `(100 * pApB + 200 * pApA * specC17) / (...)`, which is not what roadrunner selects. `sbmlsim.fit.petab_v2.observables` therefore writes a copy of the model in which every such observable is a parameter with an assignment rule, and the fit selects that parameter: the identifiers of the math of PEtab are the ones of the model, and the formula is translated into the math of SBML for the rule. Simulated at the nominal parameters of the problem, the observables agree with the `simulatedData` of the collection to `2e-4` at a relative tolerance of `1e-9` of the integrator.

 The parameters of a problem which are not estimated are applied to the model with the nominal values of the parameter table, which is what PEtab prescribes and which the model does not have to agree with.

```

- [ ] **Step 2: Build the documentation**

Run: `uv run zensical build --clean`
Expected: the build finishes without an error, `site/petab/index.html` holds the section "Hybrid problems of PEtab SciML"

- [ ] **Step 3: Run the checks**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: `All checks passed!`, `... files already formatted`, `All checks passed!` (zero diagnostics)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs: hybrid problems of PEtab SciML"
```


## After the last task

Run the whole suite once more from the tree of the last commit:

```bash
uv run pytest -q
uv run pytest -q -m sciml_testsuite tests/sciml/test_testsuite.py
uv run ruff check && uv run ruff format --check && uvx ty check
tox r -e ty
```

Expected: `1471 passed, 2 skipped` (the 12 warnings are the ones which exist before this branch: SALib, seaborn, pint), `98 passed` for the cases of the suite, zero diagnostics. Nothing is pushed.

Reported upstream by the owner, not by a task: `petab.v2.Problem.from_yaml` of `petab` 0.9.0 needs `torch` for a problem with networks (`NameError: nn` in `petab_sciml` without it), which is why `PetabReader.from_yaml` reads the tables without the `sciml` block.

## Passage for CLAUDE.md

Not applied by any task, the controller hands it to the owner. It replaces the sentence on `sbmlsim.sciml` in the section "Architecture" of the paragraph on `fit/` and adds a paragraph:

**`sciml/` - neural networks of hybrid problems (PEtab SciML).** `Network` (`sciml/network.py`) is a frozen dataclass of the architecture (the `NNModel` of `petab_sciml`) and the arrays of one network in the PyTorch layout; it is validated when it is created, compares its id, architecture and arrays, and `forward` evaluates it with numpy. Every element, unit, input and output has an id which is an SBML `SId` (`element_id`, `unit_id`, `input_id`, `output_id`, `parse_io_id`). `interpreter.evaluate` walks the forward pass with a backend of `sciml/backend.py`: `NumpyBackend` is the forward pass and `SympyBackend` evaluates the same layers on expressions; the layers of the package `layers` declare which backends they support (convolution, pooling and normalization are numpy only). `Hybridization` (`sciml/hybridization.py`) says where a network sits: `PRE_INITIALIZATION` runs before every simulation in numpy and its outputs are changes of the simulation (it implements the `DerivedChanges` protocol of `fit/derived.py`), `RHS` and `OBSERVABLE` are compiled into the model by `compile_network` (`sciml/compiler.py`) as parameters with assignment rules, one rule per unit and one layer deep, written to `<stem>_sciml.xml`; a `NetworkInput` is a formula, one formula per condition, or arrays per condition (`ALL_CONDITIONS = "0"`), the condition of a simulation is its id in the experiment, and the arrays of a condition of a compiled network are set by its derived changes. `network_fit_parameters` (`sciml/parameters.py`) gives one `FitParameter` per estimated element on the linear scale and with the unit `dimensionless`, with the target `sciml:<id>` for a network before the simulation (`external=True`). `sbmlsim.sciml` knows nothing of PEtab and is the extra `sciml` (`petab-sciml`, `h5py`, `pyyaml`); `fit/petab_v2/sciml.py` translates the extension `sciml` of a problem (read without `torch`, which `petab` needs for it) and the reader imports it lazily. `sciml/testsuite.py` holds the PEtab SciML test suite pinned by `SCIML_SUITE_COMMIT`: `ml_model_import` compares the forward pass, `initialization` the nominal values, `sciml_problem_import` the log-likelihood, the simulations and the five point gradient at the tolerances `1e-13`; `tests/data/sciml_baseline.json` is the baseline, `tox r -e sciml` runs the cases and `scripts/sciml_testsuite.py` is the command line.

Additions to the paragraph on `fit/`: `FitParameter.scale` is the scale of one parameter and `None` the `parameter_scale` of the settings, `problem.to_scale`/`from_scale` transform every parameter with its own scale and `FisherInformation` and the profiles use them; a parameter on the linear scale may have infinite bounds, which the local optimizer takes, differential evolution refuses and `fit/sampling.py` does not sample (it starts from the start value); a parameter whose target has the prefix `sciml:` (`EXTERNAL_PREFIX`) is not an entity of the model, `ParameterMapping` writes no change for it and the derived changes read it; `OptimizationProblem(hybridizations=...)` takes objects of the protocol `fit/derived.py::DerivedChanges`, which are pickled with the definition, resolved per simulation group at `initialize` (with the checks that nothing is flat: a parameter which writes what a hook sets or freezes, a hook which reads what another sets, an external parameter no hook reads) and called by `_simulate_groups` after the changes of the parameters with the values of the fit, of the simulation and of the model in the units of the model. `gradient` of `petab_v2/likelihood.py` has `order` (2 or 4, `stencil` gives the points), keeps its step next to a bound with a one sided difference and raises for a parameter outside its bounds. The reader keys its fit mappings by observable and by experiment (`<observable>_<experiment>` for an observable measured in several experiments), translates the math of an observable into the math of SBML (`log` of PEtab is the natural logarithm), compiles the networks of a problem into `<stem>_sciml.xml` before `observables.py` runs, reads the `parameterScale` column of a SciML problem, and `known_extensions()` adds `sciml` when the extra is installed. `mathml.py` gained `formula_expression`, `formula_symbols`, `evaluate_formula`, `expression_to_astnode` and `expression_to_formula`, which read a formula with libsbml and `sbmlmath` (a symbol for every identifier, also `beta`, `gamma` and `lambda`) and write an expression as math of SBML; `mathml.evaluate` hands the text of a formula to `sympify` and stays for `Data` of type `FUNCTION`.
