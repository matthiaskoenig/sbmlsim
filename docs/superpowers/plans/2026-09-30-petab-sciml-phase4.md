# PEtab SciML Phase 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A hybrid problem is written as PEtab SciML so that every read case of the test suite and every python defined hybrid fit round trips exactly, the console and the report show a network as one row per array, two examples fit and report hybrid problems, and the documentation describes it all.

**Architecture:** The exporter of `fit/petab_v2/export.py` writes the definition of a problem, not the state an evaluation left in it: `OptimizationProblem` keeps the changes of the first timecourses as defined, a derived model (compiled networks, formula observables) carries a record of what was added to it (`model/provenance.py`), which the exporter strips to write the model the problem was defined with, and the new `fit/petab_v2/sciml_export.py` (imported lazily, the extra `sciml`) translates the hybridizations into the `sciml` block, the tables, the NN YAML and the HDF5 arrays through `petab.v2.extensions.sciml` and `petab_sciml`. The console and the report get one `HookSummary` per hook of the problem (`fit/derived.py`, implemented by `Hybridization.summary`), which groups the elements of a network per array. Two examples under `examples/sciml/` and the documentation close the phase, with the cleanups the reviews of phase 3 parked.

**Tech Stack:** python >= 3.13, libroadrunner, libsbml, petab 0.9 (`petab.v2`, `petab.v2.extensions.sciml`), petab-sciml >= 0.0.3, h5py, sympy, sbmlmath, numpy, pandas, jinja2, rich, matplotlib; torch (CPU) only in the `dev` extra for validation with `petab`.

**Spec:** `docs/superpowers/specs/2026-09-30-petab-sciml-design.md`, sections "`export.py`", "Report and console", "The fit" (the report), "The test suite" (the round trip tests), "Phases" (row 4 and the paragraph below the table), and the two sections of amendments. The notes of the controller which this plan takes up are `.superpowers/sdd/2026-09-30-petab-sciml-phase3/progress.md` (the lines "parked" and "Note for phase 4") and the notes files of the scratchpad. The passage for `CLAUDE.md` is the last section of this plan; no task edits `CLAUDE.md`.

## Global Constraints

- python >= 3.13; the code runs on the `.venv` of the worktree with `uv run --no-sync`.
- Full type annotations and google style docstrings on every module, class and function of `src/` (ruff `D`); `examples/` and `tests/` are exempt from the docstring rules but not from the type check.
- `ruff check`, `ruff format --check` and `uvx ty check` at zero diagnostics after every task; suppress a diagnostic only with a rule specific `# ty: ignore[rule-name]`, never a blanket `# type: ignore`.
- Library code logs with `logging.getLogger(__name__)` and lazy `%s` formatting (ruff `G`), never prints; `examples/` print through `sbmlsim.console` and `sbmlsim.fit.display`.
- Never the em dash character in any file, use "-".
- No attribution of agents in commits or files: no `Co-Authored-By`, no "Generated with" lines.
- Never edit `CHANGELOG.md` or `CLAUDE.md`; release notes are not written (they belong to the release commit).
- Markdown without hard line wraps: a paragraph, a list item or a table row is one line.
- Tests run with `uv run --no-sync pytest -q -x <path>`; the full suite with `uv run --no-sync pytest -q` must pass with 0 warnings after every task; tests of a report use `mapping_figures=False`.
- `sbmlsim.sciml` knows nothing of PEtab; `sbmlsim.fit` imports `sbmlsim.sciml` lazily (`tests/sciml/test_package.py::test_the_package_does_not_import_the_networks` pins it); `fit/petab_v2` translates.
- A read problem runs serially (its experiment class is built at runtime), a python defined one in parallel.
- Every new public function validates its inputs and raises errors which name the network, the node, the target or the fit mapping.
- Examples write into the current working directory and never open a window; an example never commits derived files (`<stem>_sciml.xml`, `<stem>_observables.xml`), they go to `results/` which `.gitignore` covers.
- Commits on the current branch `petab-sciml-phase4`, never pushed.
- The documentation builds with `uv run --no-sync zensical build --clean` and reports `No issues found`.

## Review Focus

1. An export after an evaluation of the problem: `_simulate_groups` writes the values of the parameters and the derived changes into the first timecourse, so an export wrote them as condition rows (every element of a network as a condition of the experiment). The definition must be written; pinned in Task 2 (`test_export_after_an_evaluation_writes_the_definition`).
2. A PEtab model whose id is `model`, which the helpers of `petab_sciml` and the PEtab documentation use: the reader named the default experiment `model` and the experiment refused the duplicate key. Pinned in Task 2 (`test_a_model_named_model`).
3. An array of a network whose elements differ in `estimate` or in their bounds, which a python user can build by hand: PEtab SciML describes an array as a whole, so the export must refuse it and name the array (gap `sciml-partial-array`). Pinned in Task 3.
4. An observable id which shadows an entity of the model (a mapping key `prey` for the species `prey`): `petab` rejects such a problem, the exporter must rename it. Pinned in Task 3 (`test_an_observable_which_shadows_an_entity`).
5. A hook of a user which is not a `Hybridization` (any `DerivedChanges`): it cannot be written as PEtab SciML, the export must raise and name it instead of writing a problem without it. Pinned in Task 3 (`test_a_hook_which_is_no_network_is_refused`).

## What was prototyped

Every code block of the tasks 1 to 3 and of the tasks 5, 6, 8 and 9 was run in a copy of the branch before the plan was written. With it, the round trip of all 36 read cases of `sciml_problem_import` (001 to 031 and 035 to 039; 032 to 034 are not read, gap `sciml-priors`) is exact: the parameters (id, start value, bounds, unit, scale, target, versioning), the hybridizations (`==`, i.e. network, pattern, model, inputs with their arrays, outputs, frozen elements, constants), the data, the kinds and the weights of the mappings are equal, and the predictions and the log-likelihood of the problem which is read back are bit-identical to a second read of the original. The comparison is against a second read because the first model roadrunner loads in a process differs from every later one by about `1e-9` in the predictions (the JIT of the first load, measured in phase 3 as well); the exported files themselves do not depend on it. The five python defined problems of `tests/sciml/test_fit.py` (before the simulation, frozen layer, arrays per simulation, in the right hand side, arrays of a compiled network) round trip exactly as well. Three findings decided the design:

- Cases 009, 012, 013, 037 and 039 differed by `2e-9` when the observables were named `<experiment>__<mapping>`: the parameters of the observables in the derived model get other ids, and roadrunner orders its symbols by id. The observables are therefore named after the mapping keys when they are unique.
- Case 039 differed by `4.5e-9` because an observable measured in two experiments, which the reader splits into `prey_o_e1` and `prey_o_e2`, was written as two observables and so as two parameters of the derived model. Fit mappings with the same formula, the same noise and the key scheme of the reader are written as one observable again; the `observables` block of the `sbmlsim` extension is keyed by the fit mapping (version `0.2.0`), the reader looks a mapping up by its key first and by its observable second.
- The model without the networks: the compiled model and the model with the formula observables record what was added to them in an annotation of the model, `strip_derivation` inverts both. The alternative, a source path carried by the hybridization or the experiment, needs the source file at export time and does not cover the observables which `add_observables` bakes into the model of cases 037 and 039. The annotation is added to the annotation node in place: libsbml checks an annotation it is given as a whole against the `metaid` of the model, and the RDF annotation of the models of the suite has none.

- A constant of a hybridization which only a row of the hybridization table names (`net1__input0__1 = k`, `k` a row of the parameter table) is an extraneous parameter to the linter of `petab`, which counts the ids of the model, of the mapping table, of the observables and of the conditions and not the values of the hybridization table. An input whose formula is such a constant which feeds only this input is therefore written the way the suite writes it (case 002): the constant is the `petabEntityId` of the input and no row of the hybridization table, which the reader reads back exactly.

The neural ODE of the how-to of PEtab SciML (`petab_sciml.problem_utils.neural_ode`) is a network 2-10-10-2 with `tanh` which gives the rates of both species: `sbmlsim` reads such a problem and fits it (`from_petab`, `RHS` pattern, 162 elements, one evaluation 4 ms, one iteration of `least_squares` with the finite difference jacobian 0.7 s), but the files `create_neural_ode_problem` of `petab_sciml` 0.0.3 writes are not PEtab v2 (`neural_nets` instead of `neural_networks`, `estimate: 1`, `simulationConditionId`), `petab` itself rejects them, and a local least squares fit from a random initialization does not fit three oscillations (cost 91 to 70 in 150 iterations, 2 minutes). The example therefore defines the neural ODE in python with two hidden layers of five units (57 elements) on the first four seconds of the data, which three random starts fit to the noise in 15 s each; the docs say for which size the finite difference jacobian is usable.

---

### Task 1: The record of a derivation of a model

**Files:**
- Create: `src/sbmlsim/model/provenance.py`
- Modify: `src/sbmlsim/sciml/compiler.py:121-146` (`_Model.__init__`), `:271-294` (`_Model.write`), `:382-411` (`_check_target`)
- Modify: `src/sbmlsim/fit/petab_v2/observables.py:21-25` (imports), `:86-92` (before `checkConsistency`)
- Create: `docs/api/model.provenance.md`
- Modify: `docs/api/index.md` (table `sbmlsim.model`), `zensical.toml` (nav `sbmlsim.model`)
- Test: `tests/model/test_provenance.py`

**Interfaces:**
- Consumes: `libsbml`, `sbmlsim.sciml.compiler._Model` (`created`, `targets`), `sbmlsim.fit.petab_v2.observables.add_observables`.
- Produces: `Derivation(source: str, created: tuple[str, ...], targets: Mapping[str, bool])` with `xml()`, `derivation_of(model: libsbml.Model) -> Derivation | None`, `record_derivation(model, source: Path, created: Iterable[str], targets: Mapping[str, bool]) -> Derivation`, `strip_derivation(sbml_path: Path) -> tuple[libsbml.SBMLDocument, Derivation]`. Every model `compile_network` and `add_observables` write carries the record; Task 2 strips it.

- [ ] **Step 1: Write the failing tests**

Create `tests/model/test_provenance.py`:

```python
"""Tests of the record of a derivation of a model."""

from pathlib import Path

import libsbml
import pytest

from sbmlsim.fit.petab_v2.observables import add_observables
from sbmlsim.model.provenance import (
    NAMESPACE,
    Derivation,
    derivation_of,
    record_derivation,
    strip_derivation,
)
from sbmlsim.sciml import Hybridization, NetworkInput, NetworkPattern, compile_network
from tests.sciml.hybrid import MODEL_PATH, feed_forward

#: an RDF annotation without a `metaid`, as the models of the PEtab SciML
#: test suite carry it
RDF = (
    '<annotation><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/">'
    '<rdf:Description rdf:about="#x"><dc:creator>who</dc:creator></rdf:Description>'
    "</rdf:RDF></annotation>"
)


def _read(path: Path) -> libsbml.Model:
    return libsbml.readSBMLFromFile(str(path)).getModel()


def _rhs(network) -> Hybridization:  # noqa: ANN001
    return Hybridization(
        network=network,
        pattern=NetworkPattern.RHS,
        model="lv",
        inputs={
            "net1__input0__0": NetworkInput(formula="prey"),
            "net1__input0__1": NetworkInput(formula="predator"),
        },
        outputs={"net1__output0__0": "gamma"},
    )


def test_a_model_without_annotation_gets_the_record(tmp_path: Path) -> None:
    document = libsbml.readSBMLFromFile(str(MODEL_PATH))
    model = document.getModel()
    assert derivation_of(model) is None
    record_derivation(model, Path("lv.xml"), ["a", "b"], {"gamma": True})
    path = tmp_path / "derived.xml"
    libsbml.writeSBMLToFile(document, str(path))
    derivation = derivation_of(_read(path))
    assert derivation == Derivation("lv.xml", ("a", "b"), {"gamma": True})


def test_the_record_keeps_an_rdf_annotation_without_metaid(tmp_path: Path) -> None:
    document = libsbml.readSBMLFromFile(str(MODEL_PATH))
    model = document.getModel()
    assert model.setAnnotation(libsbml.XMLNode.convertStringToXMLNode(RDF)) == 0
    record_derivation(model, Path("lv.xml"), ["a"], {})
    path = tmp_path / "derived.xml"
    libsbml.writeSBMLToFile(document, str(path))
    model = _read(path)
    annotation = model.getAnnotation()
    names = {
        annotation.getChild(k).getName() for k in range(annotation.getNumChildren())
    }
    assert names == {"RDF", "derived"}
    assert derivation_of(model) == Derivation("lv.xml", ("a",), {})
    assert NAMESPACE in model.getAnnotationString()


def test_a_second_derivation_extends_the_record(tmp_path: Path) -> None:
    document = libsbml.readSBMLFromFile(str(MODEL_PATH))
    model = document.getModel()
    record_derivation(model, Path("lv.xml"), ["a"], {"gamma": True})
    record_derivation(model, Path("lv_sciml.xml"), ["b"], {"a": False, "beta": True})
    derivation = derivation_of(model)
    # the source stays, a target which the first derivation created is no target
    assert derivation == Derivation("lv.xml", ("a", "b"), {"gamma": True, "beta": True})
    annotation = model.getAnnotation()
    assert annotation.getNumChildren() == 1


def test_strip_gives_the_source_model_back(tmp_path: Path) -> None:
    network = feed_forward()
    compiled = compile_network(MODEL_PATH, [_rhs(network)], tmp_path / "lv_sciml.xml")
    derivation = derivation_of(_read(compiled))
    assert derivation is not None
    assert derivation.source == "lotka_volterra.xml"
    assert set(network.parameter_ids()) <= set(derivation.created)
    assert derivation.targets == {"gamma": True}

    document, stripped = strip_derivation(compiled)
    assert stripped == derivation
    model = document.getModel()
    source = _read(MODEL_PATH)
    assert model.getNumParameters() == source.getNumParameters()
    assert model.getNumRules() == source.getNumRules()
    assert (
        model.getParameter("gamma").getConstant()
        == source.getParameter("gamma").getConstant()
    )
    assert derivation_of(model) is None
    assert libsbml.writeSBMLToString(document) == libsbml.writeSBMLToString(
        libsbml.readSBMLFromFile(str(MODEL_PATH))
    )


def test_strip_of_observables_on_a_compiled_model(tmp_path: Path) -> None:
    compiled = compile_network(
        MODEL_PATH, [_rhs(feed_forward())], tmp_path / "lv_sciml.xml"
    )
    derived = add_observables(
        compiled, {"total": "prey + predator"}, tmp_path / "lv_sciml_observables.xml"
    )
    derivation = derivation_of(_read(derived))
    assert derivation is not None
    assert derivation.source == "lotka_volterra.xml"
    assert "observable_total" in derivation.created
    document, _ = strip_derivation(derived)
    assert (
        document.getModel().getNumParameters() == _read(MODEL_PATH).getNumParameters()
    )


def test_strip_refuses_a_model_which_is_not_derived() -> None:
    with pytest.raises(ValueError, match="is not derived from a model"):
        strip_derivation(MODEL_PATH)


def test_strip_names_a_created_parameter_which_is_missing(tmp_path: Path) -> None:
    document = libsbml.readSBMLFromFile(str(MODEL_PATH))
    record_derivation(document.getModel(), Path("lv.xml"), ["missing"], {})
    path = tmp_path / "derived.xml"
    libsbml.writeSBMLToFile(document, str(path))
    with pytest.raises(ValueError, match="has no parameter 'missing'"):
        strip_derivation(path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync pytest -q -x tests/model/test_provenance.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbmlsim.model.provenance'`

- [ ] **Step 3: Create the module**

Create `src/sbmlsim/model/provenance.py`:

```python
"""The record of what was derived from a model.

`sbmlsim` writes models which are derived from the model of a problem: the
model with the compiled networks of a hybrid problem
(`sbmlsim.sciml.compiler.compile_network`) and the model with the observables
which are formulas (`sbmlsim.fit.petab_v2.observables.add_observables`). A fit
simulates the derived model, an export writes the model the problem was
defined with. The derived model therefore carries what was added to it, as an
element of the annotation of the model in the namespace `NAMESPACE`:

    <derived xmlns="https://github.com/matthiaskoenig/sbmlsim/derived" source="lv.xml">
      <created>net1__layer1__weight__0_0 net1__output0__0</created>
      <target id="gamma" constant="true"/>
    </derived>

`created` lists the parameters which were added, with their rules, and
`target` lists the parameters of the source which got a rule, with the value
of `constant` they had. `strip_derivation` undoes both, which gives the source
model. A derivation of a derived model extends the record, so the source is
always the model the problem was defined with.

The element is added to the annotation of the model in place: libsbml checks
an annotation it is given as a whole against the `metaid` of the model, and
the RDF annotation of a model of the wild does not always have one.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import libsbml

logger = logging.getLogger(__name__)

#: the namespace of the record
NAMESPACE = "https://github.com/matthiaskoenig/sbmlsim/derived"

#: the element of the record
ELEMENT = "derived"


@dataclass(frozen=True)
class Derivation:
    """What was added to a model.

    Attributes:
        source: name of the file of the model the derivation started from.
        created: ids of the parameters which were added, in the order they
            were added; a rule of such a parameter was added with it.
        targets: id of a parameter of the source which got a rule -> whether
            it was constant before.
    """

    source: str
    created: tuple[str, ...] = ()
    targets: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze the attributes."""
        object.__setattr__(self, "created", tuple(self.created))
        object.__setattr__(self, "targets", dict(self.targets))

    def xml(self) -> str:
        """Get the record as the element of the annotation."""
        targets = "".join(
            f'<target id="{sid}" constant="{"true" if constant else "false"}"/>'
            for sid, constant in self.targets.items()
        )
        return (
            f'<{ELEMENT} xmlns="{NAMESPACE}" source="{self.source}">'
            f"<created>{' '.join(self.created)}</created>{targets}</{ELEMENT}>"
        )


def _record_index(model: libsbml.Model) -> int:
    """Get the index of the record in the annotation of a model, `-1` for none."""
    if not model.isSetAnnotation():
        return -1
    annotation: libsbml.XMLNode = model.getAnnotation()
    for k in range(annotation.getNumChildren()):
        child: libsbml.XMLNode = annotation.getChild(k)
        if child.getName() == ELEMENT and child.getURI() == NAMESPACE:
            return k
    return -1


def derivation_of(model: libsbml.Model) -> Derivation | None:
    """Read the record of a model.

    Args:
        model: the model.

    Returns:
        The derivation, `None` for a model which is not derived.

    Raises:
        ValueError: if the record is not valid.
    """
    k = _record_index(model)
    if k < 0:
        return None
    record: libsbml.XMLNode = model.getAnnotation().getChild(k)
    source = record.getAttrValue("source")
    if not source:
        raise ValueError(
            f"The model '{model.getId()}' has a record of its derivation without "
            f"the source"
        )
    created: list[str] = []
    targets: dict[str, bool] = {}
    for i in range(record.getNumChildren()):
        child: libsbml.XMLNode = record.getChild(i)
        if child.getName() == "created":
            text = child.getChild(0).getCharacters() if child.getNumChildren() else ""
            created.extend(text.split())
        elif child.getName() == "target":
            sid = child.getAttrValue("id")
            constant = child.getAttrValue("constant")
            if not sid or constant not in ("true", "false"):
                raise ValueError(
                    f"The model '{model.getId()}' has a record of its derivation "
                    f"with the target '{sid}' and constant '{constant}'"
                )
            targets[sid] = constant == "true"
    return Derivation(source=source, created=tuple(created), targets=targets)


def _remove_record(model: libsbml.Model) -> None:
    """Remove the record from the annotation of a model, if it has one."""
    k = _record_index(model)
    if k < 0:
        return
    annotation: libsbml.XMLNode = model.getAnnotation()
    annotation.removeChild(k)
    if annotation.getNumChildren() == 0:
        model.unsetAnnotation()


def record_derivation(
    model: libsbml.Model,
    source: Path,
    created: Iterable[str],
    targets: Mapping[str, bool],
) -> Derivation:
    """Write the record of a derivation into a model.

    A record the model has is extended: the source stays, the created ids
    and the targets are added. A target which the earlier record created is
    a created id and not a target.

    Args:
        model: the derived model.
        source: the file of the model the derivation started from.
        created: ids of the parameters which were added.
        targets: id of a parameter which got a rule -> whether it was
            constant.

    Returns:
        The record which was written.

    Raises:
        ValueError: if the record cannot be written.
    """
    earlier = derivation_of(model)
    if earlier is None:
        derivation = Derivation(
            source=Path(source).name, created=tuple(created), targets=dict(targets)
        )
    else:
        derivation = Derivation(
            source=earlier.source,
            created=(*earlier.created, *created),
            targets={
                **earlier.targets,
                **{
                    sid: constant
                    for sid, constant in targets.items()
                    if sid not in earlier.created
                },
            },
        )
        _remove_record(model)
    node = libsbml.XMLNode.convertStringToXMLNode(derivation.xml())
    if node is None:
        raise ValueError(
            f"The record of the derivation of '{source}' is not XML: {derivation.xml()}"
        )
    if model.isSetAnnotation():
        success = model.getAnnotation().addChild(node)
    else:
        success = model.setAnnotation(
            libsbml.XMLNode.convertStringToXMLNode(
                f"<annotation>{derivation.xml()}</annotation>"
            )
        )
    if success != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise ValueError(
            f"The record of the derivation of '{source}' cannot be written into "
            f"the model '{model.getId()}': "
            f"{libsbml.OperationReturnValue_toString(success)}"
        )
    return derivation


def strip_derivation(sbml_path: Path) -> tuple[libsbml.SBMLDocument, Derivation]:
    """Undo the derivation of a model.

    Args:
        sbml_path: the derived model.

    Returns:
        The document of the source model, i.e. the model without the
        parameters and rules which were added and without the record, and
        the record.

    Raises:
        ValueError: if the model cannot be read, is not derived, or a created
            parameter or a target is not in it.
    """
    document: libsbml.SBMLDocument = libsbml.readSBMLFromFile(str(sbml_path))
    model: libsbml.Model | None = document.getModel()
    if model is None:
        raise ValueError(f"'{sbml_path}' is not an SBML model")
    derivation = derivation_of(model)
    if derivation is None:
        raise ValueError(f"The model '{sbml_path}' is not derived from a model")
    for sid in derivation.created:
        model.removeRuleByVariable(sid)
        if model.removeParameter(sid) is None:
            raise ValueError(
                f"The model '{sbml_path}' has no parameter '{sid}', which its "
                f"derivation created"
            )
    for sid, constant in derivation.targets.items():
        model.removeRuleByVariable(sid)
        parameter: libsbml.Parameter | None = model.getParameter(sid)
        if parameter is None:
            raise ValueError(
                f"The model '{sbml_path}' has no parameter '{sid}', which its "
                f"derivation set"
            )
        parameter.setConstant(constant)
    _remove_record(model)
    logger.info(
        "The model '%s' is the model '%s' without %d parameters",
        Path(sbml_path).name,
        derivation.source,
        len(derivation.created),
    )
    return document, derivation
```

- [ ] **Step 4: Record the derivation in the compiler and in `add_observables`**

Apply to `src/sbmlsim/sciml/compiler.py`:

```diff
@@
 from sbmlsim.mathml import TIME, expression_to_astnode, formula_symbols
+from sbmlsim.model.provenance import record_derivation
 from sbmlsim.sciml.backend import SympyBackend
@@ class _Model:
         except NetworkHybridizationError as err:
             raise NetworkCompilationError(str(err)) from err
         self.name = Path(sbml_path).name
+        self.source_path = Path(sbml_path)
         self.created: dict[str, str] = {}
+        #: id of every target of the source which got a rule -> whether it
+        #: was constant before, the record of the derivation
+        self.rule_targets: dict[str, bool] = {}
         self.constants: dict[str, float] = {}
         self.targets: dict[str, tuple[str, str]] = {}
@@     def write(self, output_path: Path) -> Path:
         Raises:
             NetworkCompilationError: if the model is not valid SBML.
         """
+        record_derivation(
+            self.model, self.source_path, list(self.created), self.rule_targets
+        )
         errors = self.errors()
         if errors:
@@ def _check_target(model: _Model, network: str, key: str, target: str) -> None:
             f"{prefix} has an initial assignment in the model '{model.name}', "
             f"which the assignment rule of the network replaces"
         )
+    model.rule_targets[target] = bool(parameter.getConstant())
     parameter.setConstant(False)
```

Add to the docstring of `_Model` the attributes `source_path: the file of the model without the networks` and `rule_targets: id of every target of the source which got a rule -> whether it was constant before`.

Apply to `src/sbmlsim/fit/petab_v2/observables.py`:

```diff
@@
 import libsbml

+from sbmlsim.model.provenance import record_derivation
+
 logger = logging.getLogger(__name__)
@@ def add_observables(
         rule.setVariable(sid)
         rule.setMath(math)

+    record_derivation(model, sbml_path, [observable_id(pid) for pid in formulas], {})
     document.checkConsistency()
```

Extend the docstring of `add_observables` with the sentence: "The model records the parameters it got (`sbmlsim.model.provenance`), so that an export writes the model of the problem and the formulas."

- [ ] **Step 5: Run the tests**

Run: `uv run --no-sync pytest -q -x tests/model/test_provenance.py tests/sciml/test_compiler.py tests/fit/test_petab_v2_reader.py`
Expected: PASS

- [ ] **Step 6: The API page**

Create `docs/api/model.provenance.md`:

```markdown
# model.provenance

::: sbmlsim.model.provenance
```

In `docs/api/index.md`, table `sbmlsim.model`, add the row `| [model.provenance](model.provenance.md) | the record of what was added to a derived model, i.e. the compiled networks and the formula observables, and its inverse |`. In `zensical.toml`, nav `sbmlsim.model`, add `{ "provenance" = "api/model.provenance.md" },` after `model_resources`.

Run: `uv run --no-sync zensical build --clean 2>&1 | tail -3`
Expected: `No issues found`

- [ ] **Step 7: Lint, type check, commit**

Run: `uv run --no-sync ruff check && uv run --no-sync ruff format --check && uvx ty check && uv run --no-sync pytest -q`
Expected: zero diagnostics, all tests pass, 0 warnings

```bash
git add src/sbmlsim/model/provenance.py src/sbmlsim/sciml/compiler.py src/sbmlsim/fit/petab_v2/observables.py tests/model/test_provenance.py docs/api/model.provenance.md docs/api/index.md zensical.toml
git commit -m "model: a derived model records what was added to it, strip_derivation gives the source back"
```

---

### Task 2: The exporter writes the definition of a problem

**Files:**
- Modify: `src/sbmlsim/fit/optimization.py:310-313` (attributes), `:820-826` (before `ParameterMapping`)
- Modify: `src/sbmlsim/fit/petab_v2/export.py` (imports, `_add_models`, `_add_experiments`, `_periods`, `_add_observables_and_measurements`, `_add_extension`, `to_petab`)
- Modify: `src/sbmlsim/fit/petab_v2/reader.py:92` (`DEFAULT_EXPERIMENT`), `:349-353` (`_observable_info`), its four callers
- Modify: `src/sbmlsim/fit/petab_v2/extension.py:40` (`EXTENSION_VERSION`), docstring of `SbmlsimExtension.observables`
- Modify: `tests/fit/test_petab_v2.py:310-335` (`test_round_trip_keeps_the_data`), `tests/fit/test_petab_v2_dosing.py:58-68` (`_periods_of`)
- Test: `tests/fit/test_petab_v2_definition.py`

**Interfaces:**
- Consumes: `Derivation`, `derivation_of`, `strip_derivation` of Task 1; `PetabReader`, `PetabExporter`.
- Produces: `OptimizationProblem.defined_changes: list[dict[str, Any]]` (per fit mapping, the changes of its first timecourse as defined, set by `initialize`); `PetabExporter.derivations: dict[str, Derivation]`, `PetabExporter.info_keys: dict[int, str]`, `PetabExporter._simulation_ids() -> dict[str, str]`, `PetabExporter._periods(..., simulation_key: str, defined_changes: Mapping[str, Any], sbml_model=None)`, `PetabExporter._name_observables()`, `PetabExporter._observable_formula(k) -> str`, `PetabExporter._is_derived_observable(k) -> bool`; `PetabReader.observable_info(key: str) -> dict[str, Any]` (public, by fit mapping); `DEFAULT_EXPERIMENT = "default_experiment"`; the `sbmlsim` extension `0.2.0` whose `observables` block is keyed by the fit mapping and carries `observable`, and whose `parameters` carry `scale` when set. Task 3 adds the SciML parts to the same methods; the attribute `self.sciml` is created here as `None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/fit/test_petab_v2_definition.py`:

```python
"""Tests that the export writes the definition of a fit, not its state."""

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import petab.v2 as petab_v2
import pytest
import yaml

from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS
from sbmlsim.fit import FitSettings
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.fit.objects import FitParameter
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.fit.petab_v2 import PetabExporter, to_petab
from sbmlsim.fit.petab_v2.extension import (
    EXTENSION_VERSION,
    SbmlsimExtension,
    extension_of,
)
from sbmlsim.fit.petab_v2.likelihood import log_likelihood
from sbmlsim.fit.petab_v2.reader import DEFAULT_EXPERIMENT, PetabReader, from_petab
from test_petab_v2_reader import write_problem
from tests.fit.hooks import Scaling, factor_parameter


def _extension(config: Any) -> SbmlsimExtension:
    extension = extension_of(config)
    assert extension is not None
    return extension


def _hctz(settings: FitSettings, **replaced) -> OptimizationProblem:  # noqa: ANN003
    definition = dataclasses.replace(FIT_DEFINITIONS["PK"], **replaced)
    problem = definition.problem(opid="hctz_definition")
    problem.initialize(settings)
    return problem


def test_export_after_an_evaluation_writes_the_definition(
    fit_settings: FitSettings,
) -> None:
    """The values an evaluation writes into the timecourses are not conditions."""
    problem = _hctz(fit_settings)
    x = np.asarray(problem.x0, dtype=float)
    problem.cost_least_square(problem.to_scale(x))
    # the evaluation wrote the parameters into the first timecourses
    assert any(
        set(problem.pids) & set(simulation.timecourses[0].changes)
        for simulation in problem.simulations
    )
    assert not any(
        set(problem.pids) & set(changes) for changes in problem.defined_changes
    )

    petab_problem = PetabExporter(problem).to_problem()
    targets = {
        change.target_id
        for condition in petab_problem.conditions
        for change in condition.changes
    }
    assert not targets & set(problem.pids)


def test_the_observables_are_named_after_the_mappings(
    fit_settings: FitSettings,
) -> None:
    """Unique mapping keys are the observable ids, without the experiment."""
    problem = _hctz(fit_settings)
    exporter = PetabExporter(problem)
    petab_problem = exporter.to_problem()
    ids = {observable.id for observable in petab_problem.observables}
    assert ids == set(problem.mapping_keys)
    assert _extension(petab_problem.config).version == EXTENSION_VERSION
    info = _extension(petab_problem.config).observables
    assert set(info) == set(problem.mapping_keys)
    assert all(entry["observable"] == key for key, entry in info.items())


def test_an_observable_measured_in_two_experiments_is_written_once(
    tmp_path: Path,
) -> None:
    """The reader splits it into two mappings, the export joins them again."""
    path = write_problem(
        tmp_path / "problem", {"prey_o": "prey"}, {"e1": None, "e2": None}
    )
    problem, _ = from_petab(path)
    # the bounds of the problem start at zero, which the logarithmic scale
    # of the default settings refuses
    settings = FitSettings(parameter_scale=ParameterScaleType.LINEAR)
    problem.initialize(settings)
    assert sorted(problem.mapping_keys) == ["prey_o_e1", "prey_o_e2"]

    yaml_file = to_petab(problem, tmp_path / "export")
    petab_problem = petab_v2.Problem.from_yaml(yaml_file)
    assert [observable.id for observable in petab_problem.observables] == ["prey_o"]
    assert {m.experiment_id for m in petab_problem.measurements} == {"e1", "e2"}

    restored, _ = from_petab(yaml_file)
    restored.initialize(settings)
    reader = PetabReader.from_yaml(yaml_file)
    assert sorted(restored.mapping_keys) == ["prey_o_e1", "prey_o_e2"]
    assert reader.observable_info("prey_o_e1")["mapping"] == "prey_o_e1"
    assert log_likelihood(restored) == pytest.approx(log_likelihood(problem), rel=1e-9)


def test_a_formula_observable_is_written_as_its_formula(tmp_path: Path) -> None:
    """The model of the problem is written, not the one with the observable."""
    path = write_problem(
        tmp_path / "problem", {"total": "prey + predator"}, {"e1": None}
    )
    reader = PetabReader.from_yaml(path)
    reader.derived_dir = tmp_path / "derived"
    problem = reader.to_optimization_problem(opid="formula")
    settings = FitSettings(parameter_scale=ParameterScaleType.LINEAR)
    problem.initialize(settings)
    assert problem.yid_observable == ["observable_total"]

    yaml_file = to_petab(problem, tmp_path / "export")
    petab_problem = petab_v2.Problem.from_yaml(yaml_file)
    assert (tmp_path / "export" / "lv.xml").is_file()
    assert not (tmp_path / "export" / "lv_observables.xml").exists()
    assert not petab_problem.models[0].has_entity_with_id("observable_total")
    (observable,) = petab_problem.observables
    assert str(observable.formula) == "predator + prey"
    info = _extension(petab_problem.config).observables["total"]
    assert info["yid_observable"] is None

    restored, _ = from_petab(yaml_file)
    restored.initialize(settings)
    assert restored.yid_observable == ["observable_total"]
    assert log_likelihood(restored) == pytest.approx(log_likelihood(problem), rel=1e-9)


def test_a_model_named_model(tmp_path: Path) -> None:
    """The model id `model` of the PEtab documentation is not the default experiment."""
    path = write_problem(tmp_path / "problem", {"prey_o": "prey"}, {"e1": None})
    # the measurements name no experiment, and the model is named `model`
    measurements = pd.read_csv(tmp_path / "problem" / "measurements.tsv", sep="\t")
    measurements.drop(columns=["experimentId"]).to_csv(
        tmp_path / "problem" / "measurements.tsv", sep="\t", index=False
    )
    config = yaml.safe_load(path.read_text())
    config.pop("experiment_files")
    config["model_files"] = {"model": config["model_files"]["lv"]}
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    problem, _ = from_petab(path)
    problem.initialize(FitSettings(parameter_scale=ParameterScaleType.LINEAR))
    assert problem.simulation_keys == [DEFAULT_EXPERIMENT] * 1
    assert DEFAULT_EXPERIMENT != "model"


def test_the_scale_of_a_parameter_survives_the_round_trip(
    fit_settings: FitSettings, tmp_path: Path
) -> None:
    first = FIT_DEFINITIONS["PK"].parameters[0]
    parameters = [
        FitParameter(
            first.pid,
            first.start_value,
            first.lower_bound,
            first.upper_bound,
            unit=first.unit,
            target=first.target,
            mappings=first.mappings,
            scale=ParameterScaleType.LINEAR,
        ),
        *FIT_DEFINITIONS["PK"].parameters[1:],
    ]
    problem = _hctz(fit_settings, parameters=parameters)
    yaml_file = to_petab(problem, tmp_path / "export")
    extension = _extension(petab_v2.Problem.from_yaml(yaml_file).config)
    assert extension.parameters[parameters[0].pid]["scale"] == "LINEAR"
    restored, _ = from_petab(yaml_file)
    scales = {p.pid: p.scale for p in restored.parameters}
    assert scales[parameters[0].pid] is ParameterScaleType.LINEAR
    assert all(
        scale is None for pid, scale in scales.items() if pid != parameters[0].pid
    )


def test_an_external_parameter_is_no_condition(
    definition_hctz_iv: FitDefinition, fit_settings: FitSettings
) -> None:
    """A parameter which is no entity of the model is not assigned by a condition."""
    definition = dataclasses.replace(
        definition_hctz_iv, parameters=[factor_parameter()], hybridizations=[Scaling()]
    )
    problem = definition.problem(opid="external")
    problem.initialize(fit_settings)
    petab_problem = PetabExporter(problem).to_problem()
    targets = {
        change.target_id
        for condition in petab_problem.conditions
        for change in condition.changes
    }
    assert "factor_k" not in targets
    assert not any(target.startswith("sciml:") for target in targets)
```

Task 3 replaces the last test: from then on a hook which is no network is refused by the exporter.

`fit_settings` and `definition_hctz_iv` are fixtures of `tests/fit/conftest.py`; `FitParameter` is a plain class (`src/sbmlsim/fit/objects.py:485`), which is why the scale test builds the parameter with the constructor.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync pytest -q -x tests/fit/test_petab_v2_definition.py`
Expected: FAIL, the first test with `AttributeError: 'OptimizationProblem' object has no attribute 'defined_changes'`

- [ ] **Step 3: The problem keeps the defined changes**

Apply to `src/sbmlsim/fit/optimization.py`:

```diff
@@ class OptimizationProblem(ObjectJSONEncoder):
         self.models: list[Any] = []
         self.simulations: list[Any] = []
+        #: the changes of the first timecourse of every simulation as defined,
+        #: see `initialize`
+        self.defined_changes: list[dict[str, Any]] = []
         self.selections: list[Any] = []
@@ def initialize(self, settings: FitSettings, force: bool = False) -> None:
                 f"be '{MappingKind.TRAINING.value}'."
             )

+        #: the changes of the first timecourse of every fit mapping as the
+        #: experiment defines them: `_simulate_groups` writes the values of
+        #: the parameters and the derived changes into the timecourse, and an
+        #: export writes the definition
+        self.defined_changes = [
+            dict(simulation.timecourses[0].changes)
+            if isinstance(simulation, TimecourseSim) and simulation.timecourses
+            else {}
+            for simulation in self.simulations
+        ]
         self.parameter_mapping = ParameterMapping(
```

`TimecourseSim` is already imported in `optimization.py` (`from sbmlsim.simulation import TimecourseSim`).

- [ ] **Step 4: The reader: the default experiment and the info of a mapping**

Apply to `src/sbmlsim/fit/petab_v2/reader.py`:

```diff
@@
-DEFAULT_EXPERIMENT = "model"
+DEFAULT_EXPERIMENT = "default_experiment"
@@
-    def _observable_info(self, observable_id: str) -> dict[str, Any]:
-        """Get what the extension says about an observable, empty without one."""
+    def observable_info(self, key: str) -> dict[str, Any]:
+        """Get what the extension says about a fit mapping, empty without one.
+
+        Args:
+            key: key of the fit mapping. The block `observables` of the
+                extension is keyed by the fit mapping since its version
+                0.2.0, and by the observable before: an observable measured
+                in several experiments is one observable of several fit
+                mappings, each with its own kind and weight.
+        """
         if self.extension is None:
             return {}
-        return self.extension.observables.get(observable_id, {})
+        info = self.extension.observables.get(key)
+        if info is None:
+            info = self.extension.observables.get(self._observable_ids.get(key, ""))
+        return info or {}
```

and replace the four calls: in `datasets` and `fit_mappings` `info = self._observable_info(observable_id)` becomes `info = self.observable_info(key)`, and in `mapping_collections` both `self._observable_info(self._observable_ids[key]).get(...)` become `self.observable_info(key).get(...)`. Extend the comment on `DEFAULT_EXPERIMENT`: "It is not `model`, the id the PEtab documentation and the helpers of `petab_sciml` give the model, because the keys of the models, the simulations and the tasks of an experiment share one namespace."

Apply to `src/sbmlsim/fit/petab_v2/extension.py`:

```diff
-EXTENSION_VERSION = "0.1.0"
+EXTENSION_VERSION = "0.2.0"
```

and in the docstring of `SbmlsimExtension` replace the `observables:` line by: `observables: the fit mapping behind every observable, keyed by the fit mapping (an observable measured in several experiments is one observable of several fit mappings): the observable, its kind, the weight of the curve, the units of the data, the experiment and the task it belongs to and the metadata of the curve.` and the `parameters:` line by `parameters: unit, start value and, when set, the scale of every fit parameter which is no element of a network.`

- [ ] **Step 5: The exporter**

Apply to `src/sbmlsim/fit/petab_v2/export.py`:

```diff
@@
 import shutil
 from collections import defaultdict
+from collections.abc import Mapping
 from pathlib import Path
 from typing import Any

+import libsbml
 import numpy as np
@@
 from sbmlsim.fit.petab_v2.symbols import condition_target, observable_formula
+from sbmlsim.mathml import formula_expression
+from sbmlsim.model.provenance import Derivation, derivation_of, strip_derivation
 from sbmlsim.simulation.timecourse import Timecourse, TimecourseSim
@@ class PetabExporter:
         self.experiment_ids: dict[int, str] = {}
         self.observable_ids: dict[int, str] = {}
+        #: the key of every fit mapping in the block `observables` of the
+        #: extension: the key the reader gives the mapping when it reads the
+        #: problem again, see `_name_observables`
+        self.info_keys: dict[int, str] = {}
@@
         self.group_indices: dict[int, int] = {
             k: g for g, group in enumerate(problem.mapping_groups) for k in group
         }
+        #: the derivation of every model which is derived, by model id
+        self.derivations: dict[str, Derivation] = {}
+        #: the networks of the problem, `None` for a problem without them
+        self.sciml: Any = None
+
+    def _simulation_ids(self) -> dict[str, str]:
+        """Get the id of the PEtab experiment of every simulation of the problem.
+
+        The ids are the ones `_add_experiments` gives the experiments, computed
+        ahead of it: the networks key the arrays of their inputs by them.
+
+        Returns:
+            id of the simulation of a fit mapping -> id of the experiment.
+        """
+        problem = self.problem
+        ids: dict[str, str] = {}
+        exported = set(self.indices)
+        for collection_index, collection in enumerate(problem.mapping_collections):
+            indices = [
+                k for k in exported if problem.collection_indices[k] == collection_index
+            ]
+            groups: dict[tuple[int, int], list[int]] = defaultdict(list)
+            for k in sorted(indices):
+                groups[(id(problem.models[k]), id(problem.simulations[k]))].append(k)
+            for n, group in enumerate(groups.values()):
+                experiment_id = petab_id(collection.sid)
+                if len(groups) > 1:
+                    experiment_id = petab_id(collection.sid, f"sim{n}")
+                for k in group:
+                    ids.setdefault(problem.simulation_keys[k], experiment_id)
+        return ids
@@ def _add_models(self, petab_problem: PetabProblem) -> None:
             path = Path(source.path).resolve()
             sid = by_source.get(path)
             if sid is None:
-                sid = petab_id(model.sid or path.stem)
+                # the id the experiment gives the model, which is what the
+                # hybridizations of the problem name
+                sid = petab_id(self.problem.model_keys[k])
                 if sid in by_source.values():
                     sid = petab_id(sid, f"model{len(by_source)}")
                 # `petab.v2.Model` is the abstract base, the SBML model is the
-                # concrete class which reads a file
-                model_file = SbmlModel.from_file(path, model_id=sid)
-                model_file.rel_path = Path(path.name)
+                # concrete class which reads a file. A model which is derived
+                # from the model of the problem, i.e. carries compiled
+                # networks or observables, is written as its source
+                document: libsbml.SBMLDocument = libsbml.readSBMLFromFile(str(path))
+                name = path.name
+                sbml_model = document.getModel()
+                if sbml_model is not None and derivation_of(sbml_model) is not None:
+                    document, derivation = strip_derivation(path)
+                    self.derivations[sid] = derivation
+                    name = derivation.source
+                model_file = SbmlModel(sbml_document=document, model_id=sid)
+                model_file.rel_path = Path(name)
                 petab_problem.models.append(model_file)
@@ def _add_experiments(self, petab_problem: PetabProblem) -> None:
                     experiment_id=experiment_id,
                     simulation=simulation,
                     group_index=self.group_indices[k0],
+                    simulation_key=problem.simulation_keys[k0],
+                    defined_changes=problem.defined_changes[k0],
                     sbml_model=self.sbml_models.get(
@@ def _periods(
         simulation: TimecourseSim,
         group_index: int,
+        simulation_key: str,
+        defined_changes: Mapping[str, Any],
         sbml_model: Any = None,
     ) -> list[petab_v2.ExperimentPeriod]:
@@
                 `ParameterMapping.indices_for` resolves the binding for.
+            simulation_key: id of the simulation in its experiment, which is
+                the condition of the inputs of the networks.
+            defined_changes: the changes of the first timecourse as the
+                experiment defines them, see
+                `OptimizationProblem.defined_changes`.
             sbml_model: `libsbml.Model` of the problem, for the math of the
@@
             for target, index in sorted(mapping.indices_for(group_index).items()):
                 parameter = self.problem.parameters[index]
+                if parameter.is_external:
+                    # not an entity of the model, the networks read it
+                    continue
                 if has_renamed_targets([parameter]):
@@
+        # the inputs of the networks which differ between the conditions
+        # are changes of the condition of the first period, and the arrays
+        # of such inputs are keyed by its id
+        input_changes: list[petab_v2.Change] = []
+        needs_condition = False
+        if self.sciml is not None:
+            input_changes = self.sciml.input_changes(simulation_key)
+            needs_condition = self.sciml.needs_condition(simulation_key)
+
         periods: list[petab_v2.ExperimentPeriod] = []
         offset: float = simulation.time_offset
         for k, tc in enumerate(simulation.timecourses):
@@
             condition_ids: list[str] = []
-            tc_version_changes = version_changes if k == 0 else []
-            if tc.changes or tc_version_changes:
+            # the changes as the experiment defines them, not the values of
+            # the parameters an evaluation wrote into the timecourse
+            changes = defined_changes if k == 0 else tc.changes
+            tc_changes = [
+                petab_v2.Change(
+                    target_id=condition_target(target, sbml_model),
+                    target_value=_magnitude(value),
+                )
+                for target, value in changes.items()
+            ]
+            if k == 0:
+                tc_changes += version_changes + input_changes
+            if tc_changes or (k == 0 and needs_condition):
                 condition_id = petab_id(experiment_id, f"tc{k}")
-                _table(petab_problem, "condition_tables").conditions.append(
-                    petab_v2.Condition(
-                        id=condition_id,
-                        changes=[
-                            petab_v2.Change(
-                                target_id=condition_target(target, sbml_model),
-                                target_value=_magnitude(value),
-                            )
-                            for target, value in tc.changes.items()
-                        ]
-                        + tc_version_changes,
-                    )
-                )
+                if tc_changes:
+                    _table(petab_problem, "condition_tables").conditions.append(
+                        petab_v2.Condition(id=condition_id, changes=tc_changes)
+                    )
                 condition_ids.append(condition_id)
@@ def _add_observables_and_measurements(self, petab_problem: PetabProblem) -> None:
         """Add one observable per fit mapping with its reference data."""
         problem = self.problem
+        self._name_observables()
+        written: set[str] = set()
         for k in self.indices:
-            observable_id = petab_id(
-                problem.experiment_keys[k], problem.mapping_keys[k]
-            )
-            self.observable_ids[k] = observable_id
-
+            observable_id = self.observable_ids[k]
             # the noise model of the mapping, which is the one it was read
             # with or the standard deviation of its data: the noise parameter
             # of a measurement fills in the placeholder the observable declares
             noise = self._noise_model(k, observable_id)
-            sbml_model = self.sbml_models.get(self.model_ids[id(problem.models[k])])
-            _table(petab_problem, "observable_tables").observables.append(
-                petab_v2.Observable(
-                    id=observable_id,
-                    name=f"{problem.experiment_keys[k]}.{problem.mapping_keys[k]}",
-                    formula=observable_formula(problem.yid_observable[k], sbml_model),
-                    noise_formula=noise.formula,
-                    noise_distribution=noise.distribution.value,
-                    noise_placeholders=list(noise.placeholders),
+            if observable_id not in written:
+                written.add(observable_id)
+                _table(petab_problem, "observable_tables").observables.append(
+                    petab_v2.Observable(
+                        id=observable_id,
+                        name=f"{problem.experiment_keys[k]}.{problem.mapping_keys[k]}",
+                        formula=self._observable_formula(k),
+                        noise_formula=noise.formula,
+                        noise_distribution=noise.distribution.value,
+                        noise_placeholders=list(noise.placeholders),
+                    )
                 )
-            )

             experiment_id = self.experiment_ids.get(k)
```

Insert after `_add_observables_and_measurements` (before `_noise_model`):

```python
def _name_observables(self) -> None:
    """Name the observable of every fit mapping which is written.

    The id of an observable is the key of its fit mapping, and the key
    with its experiment where two experiments share a key. Fit mappings
    which observe the same thing with the same noise in different
    experiments are one observable measured in several experiments,
    which is what the reader splits into one fit mapping per experiment
    (`<observable>_<experiment>`): they are written as one observable
    again, so that the problem which was read keeps its observables. An
    observable which would shadow an entity of the model is prefixed.
    """
    problem = self.problem
    keys = [problem.mapping_keys[k] for k in self.indices]
    unique = len(set(keys)) == len(keys)
    content: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for k in self.indices:
        observable_id = (
            petab_id(problem.mapping_keys[k])
            if unique
            else petab_id(problem.experiment_keys[k], problem.mapping_keys[k])
        )
        self.observable_ids[k] = observable_id
        noise = noise_model_of(problem, k)
        content[
            (
                self.model_ids[id(problem.models[k])],
                self._observable_formula(k),
                noise.formula,
                noise.distribution,
                tuple(noise.placeholders),
            )
        ].append(k)
    for group in content.values():
        if len(group) < 2:
            continue
        experiments = [self.experiment_ids.get(k) for k in group]
        if len(set(experiments)) != len(group) or None in experiments:
            continue
        stems: set[str] = set()
        for k, experiment in zip(group, experiments, strict=True):
            key, suffix = problem.mapping_keys[k], f"_{experiment}"
            if not key.endswith(suffix):
                stems.clear()
                break
            stems.add(key[: -len(suffix)])
        if len(stems) != 1:
            continue
        stem = petab_id(next(iter(stems)))
        for k in group:
            self.observable_ids[k] = stem
    # an observable must not shadow an entity of the model
    for k in self.indices:
        sbml_model = self.sbml_models.get(self.model_ids[id(problem.models[k])])
        observable_id = self.observable_ids[k]
        if sbml_model is not None and sbml_model.getElementBySId(observable_id):
            self.observable_ids[k] = petab_id("observable", observable_id)
    # the key of a fit mapping in the extension is the key the reader
    # gives it: the observable, and `<observable>_<experiment>` for an
    # observable which is measured in several experiments
    experiments_of: dict[str, set[str | None]] = defaultdict(set)
    for k in self.indices:
        experiments_of[self.observable_ids[k]].add(self.experiment_ids.get(k))
    for k in self.indices:
        observable_id = self.observable_ids[k]
        self.info_keys[k] = (
            observable_id
            if len(experiments_of[observable_id]) == 1
            else f"{observable_id}_{self.experiment_ids.get(k)}"
        )


def _observable_formula(self, k: int) -> str:
    """Get the formula of the observable of a fit mapping.

    An observable of a model which is derived, i.e. a parameter with the
    formula of the observable as its rule which `add_observables` wrote,
    is written as that formula, because the model is written as its
    source. Every other observable is the math of its selection.

    Args:
        k: index of the fit mapping.

    Returns:
        The math of PEtab of the observable.

    Raises:
        ValueError: if the derived model has no rule for the observable.
    """
    problem = self.problem
    model_id = self.model_ids[id(problem.models[k])]
    sbml_model = self.sbml_models.get(model_id)
    selection = problem.yid_observable[k]
    if self._is_derived_observable(k):
        derived = libsbml.readSBMLFromFile(str(problem.models[k].source.path))
        rule = derived.getModel().getRuleByVariable(selection)
        if rule is None:
            raise ValueError(
                f"'{problem.opid}': the observable '{selection}' of the fit "
                f"mapping '{problem.mapping_keys[k]}' was added to the model "
                f"'{problem.models[k].source.path}' without a rule."
            )
        return petab_math_str(
            formula_expression(libsbml.formulaToL3String(rule.getMath()))
        )
    return observable_formula(selection, sbml_model)


def _is_derived_observable(self, k: int) -> bool:
    """Check whether the observable of a fit mapping was added to the model.

    Args:
        k: index of the fit mapping.

    Returns:
        Whether the selection of the mapping is a parameter which the
        derivation of its model created, see `_observable_formula`.
    """
    derivation = self.derivations.get(self.model_ids[id(self.problem.models[k])])
    return derivation is not None and (
        self.problem.yid_observable[k] in derivation.created
    )
```

Then in `_add_extension`:

```diff
         parameters = {
             parameter.pid: {
                 "unit": parameter.unit,
                 "start_value": parameter.start_value,
+                **(
+                    {"scale": parameter.scale.name}
+                    if parameter.scale is not None
+                    else {}
+                ),
             }
             for parameter in problem.parameters
         }
@@
         for k in self.indices:
             model = problem.models[k]
-            observables[self.observable_ids[k]] = {
+            # keyed by the fit mapping, several of which may share an
+            # observable, see `_name_observables`
+            observables[self.info_keys[k]] = {
+                "observable": self.observable_ids[k],
                 "experiment": problem.experiment_keys[k],
                 "mapping": problem.mapping_keys[k],
@@
                 "xid_observable": problem.xid_observable[k],
-                "yid_observable": problem.yid_observable[k],
+                # an observable of a derived model is written as its formula
+                # and selected as the reader adds it to the model again
+                "yid_observable": None
+                if self._is_derived_observable(k)
+                else problem.yid_observable[k],
```

and in `to_petab`:

```diff
     for k in exporter.indices:
         model = problem.models[k]
         source_path = model.source.path
+        if exporter.model_ids[id(model)] in exporter.derivations:
+            # a derived model is written as its source by `to_files`
+            continue
         if source_path is not None:
```

Update the module docstring of `export.py`: after "every model of the fit is a model of the problem," add "written as the model the problem was defined with when the fit simulates a derived one (compiled networks, formula observables, see `sbmlsim.model.provenance`)," and after "every fit mapping is one observable" add ", named after the mapping; mappings which observe one thing in several experiments are one observable".

- [ ] **Step 6: Update the two tests which knew the old ids and signature**

In `tests/fit/test_petab_v2_dosing.py::_periods_of`:

```diff
     exporter.problem = SimpleNamespace(  # ty: ignore[invalid-assignment]
         opid="dosing", parameter_mapping=None
     )
+    exporter.sciml = None
     periods = exporter._periods(
-        problem, experiment_id="dosing", simulation=simulation, group_index=0
+        problem,
+        experiment_id="dosing",
+        simulation=simulation,
+        group_index=0,
+        simulation_key="dosing",
+        defined_changes=simulation.timecourses[0].changes,
     )
```

In `tests/fit/test_petab_v2.py::test_round_trip_keeps_the_data`:

```diff
-    keys = {
-        f"{original.experiment_keys[k]}__{original.mapping_keys[k]}": k
-        for k in range(len(original.mapping_keys))
-    }
-    for i, key in enumerate(problem.mapping_keys):
-        k = keys[key]
+    # the observable of a fit mapping is named after the mapping, the
+    # extension says which mapping of which experiment it was
+    reader = PetabReader.from_yaml(petab_dir / "problem.yaml")
+    keys = {
+        (original.experiment_keys[k], original.mapping_keys[k]): k
+        for k in range(len(original.mapping_keys))
+    }
+    for i, key in enumerate(problem.mapping_keys):
+        info = reader.observable_info(key)
+        k = keys[(info["experiment"], info["mapping"])]
```

- [ ] **Step 7: Run the tests**

Run: `uv run --no-sync pytest -q -x tests/fit/test_petab_v2_definition.py tests/fit/test_petab_v2.py tests/fit/test_petab_v2_dosing.py tests/fit/test_petab_v2_reader.py tests/fit/test_petab_v2_noise.py tests/sciml`
Expected: PASS (one test skipped)

- [ ] **Step 8: Lint, type check, full suite, commit**

Run: `uv run --no-sync ruff check && uv run --no-sync ruff format --check && uvx ty check && uv run --no-sync pytest -q`
Expected: zero diagnostics, all tests pass, 0 warnings

```bash
git add src/sbmlsim/fit/optimization.py src/sbmlsim/fit/petab_v2/export.py src/sbmlsim/fit/petab_v2/reader.py src/sbmlsim/fit/petab_v2/extension.py tests/fit/test_petab_v2_definition.py tests/fit/test_petab_v2.py tests/fit/test_petab_v2_dosing.py
git commit -m "petab: the export writes the definition of a fit, the source of a derived model and one observable per mapping"
```

---

### Task 3: The exporter of the networks

**Files:**
- Create: `src/sbmlsim/fit/petab_v2/sciml_export.py`
- Modify: `src/sbmlsim/fit/petab_v2/export.py` (`__init__`, `_add_parameters`, `_add_extension`, `to_petab`)
- Modify: `src/sbmlsim/fit/petab_v2/gaps.py:280-291` (after `sciml-parameter-scale`)
- Modify: `tests/fit/test_petab_v2_definition.py` (the last test)
- Create: `docs/api/fit.petab_v2.sciml_export.md`, `docs/api/fit.petab_v2.sciml.md`
- Modify: `docs/api/index.md` (table `sbmlsim.fit`), `zensical.toml` (nav `sbmlsim.fit`)
- Test: `tests/sciml/test_export.py`

**Interfaces:**
- Consumes: `PetabExporter` of Task 2 (`self.sciml`, `self.info_keys`, `_simulation_ids`, `_periods` with `input_changes`/`needs_condition`), `Hybridization`, `Network`, `NetworkInput`, `parse_io_id`, `entity_of`, `network_fit_parameters` constants, `petab.v2.extensions.sciml`, `petab_sciml`.
- Produces: `SciMLExporter(problem, hybridizations, simulation_ids)` with `element_ids: set[str]`, `constants: dict[str, float]`, `hybridization_rows: list[HybridizationRow]`, `input_changes(simulation) -> list[petab_v2.Change]`, `needs_condition(simulation) -> bool`, `mapping_rows() -> list[petab_v2.Mapping]`, `parameter_rows() -> list[petab_v2.Parameter]`, `config() -> SciMLConfig`, `write(output_dir)`; module functions `petab_math(formula) -> str`, `petab_index(index, shape) -> str`, `parameters_id(network, layer=None, array=None) -> str`, `model_entity_id(network, layer=None, array=None) -> str`, `network_yaml(sid) -> str`, `network_arrays(sid) -> str`; the gap `sciml-partial-array`. `to_petab` of a hybrid problem writes `problem.yaml` with the `sciml` block, `hybridization.tsv`, `mapping.tsv`, `<net>.yaml` and `<net>_arrays.hdf5`.

- [ ] **Step 1: Write the failing tests**

Create `tests/sciml/test_export.py`:

```python
"""Tests of the export of a hybrid problem as PEtab SciML and its round trip.

The problems are the python defined problems of `tests.sciml.test_fit`. The
predictions of the problem which is read back are compared with a tolerance:
the first model roadrunner loads in a process differs by about `1e-9` from
every later one; the round trip of the cases of the test suite compares
against a second read and is exact, see `tests/sciml/test_testsuite.py`.
"""

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import petab.v2 as petab_v2
import pytest
import yaml

from sbmlsim.fit import FitParameter
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.fit.petab_v2 import to_petab
from sbmlsim.fit.petab_v2.export import PetabExporter
from sbmlsim.fit.petab_v2.likelihood import log_likelihood
from sbmlsim.fit.petab_v2.reader import PetabReader
from sbmlsim.fit.petab_v2.sciml_export import petab_index, petab_math
from sbmlsim.sciml import (
    Hybridization,
    NetworkInput,
    compile_network,
    compiled_path,
    network_fit_parameters,
)
from tests.fit.hooks import Scaling, factor_parameter
from tests.sciml.experiment import LotkaVolterra
from tests.sciml.hybrid import MODEL_PATH, feed_forward, two_inputs
from tests.sciml.test_fit import MECHANISTIC, PRE, RHS, SETTINGS, _before, _problem


def _read(
    yaml_file: Path, derived_dir: Path
) -> tuple[PetabReader, OptimizationProblem]:
    reader = PetabReader.from_yaml(yaml_file)
    reader.derived_dir = derived_dir
    problem = reader.to_optimization_problem(opid="restored")
    problem.initialize(SETTINGS)
    return reader, problem


def _tables(petab_dir: Path) -> dict[str, Any]:
    """Read the tables of an exported problem without `petab`, which needs torch."""
    tables = {
        name: pd.read_csv(
            petab_dir / f"{name}.tsv", sep="\t", dtype=str, keep_default_na=False
        )
        for name in ("parameters", "observables", "experiments", "mapping")
    }
    tables["config"] = yaml.safe_load((petab_dir / "problem.yaml").read_text())
    return tables


def _tuple(p: FitParameter) -> tuple:
    return (
        p.pid,
        p.start_value,
        p.lower_bound,
        p.upper_bound,
        p.unit,
        p.scale,
        p.target,
        p.is_versioned,
    )


def assert_round_trip(
    problem: OptimizationProblem, tmp_path: Path
) -> OptimizationProblem:
    """Write the problem, read it back and compare the two."""
    problem.initialize(SETTINGS)
    yaml_file = to_petab(problem, tmp_path / "petab")
    reader, restored = _read(yaml_file, tmp_path / "derived")

    assert [_tuple(p) for p in restored.parameters] == [
        _tuple(p) for p in problem.parameters
    ]
    assert restored.hybridizations == problem.hybridizations
    assert len(restored.mapping_keys) == len(problem.mapping_keys)

    keys = {
        (problem.experiment_keys[k], problem.mapping_keys[k]): k
        for k in range(len(problem.mapping_keys))
    }
    x = np.asarray(problem.x0, dtype=float)
    expected = problem.predictions(x)
    observed = restored.predictions(
        np.asarray(
            [dict(zip(problem.pids, x, strict=True))[pid] for pid in restored.pids]
        )
    )
    for i, key in enumerate(restored.mapping_keys):
        info = reader.observable_info(key)
        k = keys[(info["experiment"], info["mapping"])]
        assert np.array_equal(restored.x_references[i], problem.x_references[k])
        np.testing.assert_allclose(restored.y_references[i], problem.y_references[k])
        assert restored.mapping_kinds[i] == problem.mapping_kinds[k]
        assert restored.weights_curves[i] == problem.weights_curves[k]
        np.testing.assert_allclose(observed[i], expected[k], rtol=1e-7)
    assert log_likelihood(restored) == pytest.approx(log_likelihood(problem), rel=1e-8)
    return restored


# --- ROUND TRIPS ---


def test_a_network_before_the_simulation(tmp_path: Path) -> None:
    network = feed_forward()
    elements = network_fit_parameters(
        network, estimate={"net1": True}, bounds={}, external=True
    )
    restored = assert_round_trip(_problem([_before(network)], elements), tmp_path)
    (hybridization,) = restored.hybridizations
    assert hybridization.constants == {"k": 0.5}
    tables = _tables(tmp_path / "petab")
    rows = tables["parameters"].set_index("parameterId")
    assert rows.loc["net1__parameters", "estimate"] == "true"
    assert rows.loc["net1__parameters", "nominalValue"] == "array"
    assert rows.loc["k", "estimate"] == "false"
    assert float(rows.loc["k", "nominalValue"]) == 0.5
    sciml = tables["config"]["extensions"]["sciml"]
    assert sciml["neural_networks"]["net1"]["pre_initialization"] is True
    # the elements are not in the `sbmlsim` block
    assert set(tables["config"]["extensions"]["sbmlsim"]["parameters"]) == {
        "alpha",
        "beta",
    }


def test_a_frozen_layer_and_bounds(tmp_path: Path) -> None:
    network = feed_forward()
    elements = network_fit_parameters(
        network,
        estimate={"net1": True, "net1.layer1": False},
        bounds={"net1": (-5.0, 5.0)},
        external=True,
    )
    frozen = set(network.parameter_ids()) - {p.pid for p in elements}
    assert_round_trip(_problem([_before(network, frozen=frozen)], elements), tmp_path)
    rows = _tables(tmp_path / "petab")["parameters"].set_index("parameterId")
    # the most common row is the row of the network, the layer which differs
    # has a row of its own
    network_row = rows.loc["net1__parameters"]
    layer_row = rows.loc["net1__layer2__parameters"]
    assert {network_row["estimate"], layer_row["estimate"]} == {"true", "false"}
    estimated = layer_row if layer_row["estimate"] == "true" else network_row
    assert (float(estimated["lowerBound"]), float(estimated["upperBound"])) == (
        -5.0,
        5.0,
    )


def test_the_arrays_of_the_simulations(tmp_path: Path) -> None:
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
    assert_round_trip(_problem([hybridization], elements), tmp_path)
    # the arrays are keyed by the condition of the first period of the experiment
    experiments = _tables(tmp_path / "petab")["experiments"]
    assert sorted(experiments["conditionId"]) == ["e1__tc0", "e2__tc0"]


def _compiled(tmp_path: Path, hybridization: Hybridization) -> type[LotkaVolterra]:
    compiled = compile_network(
        MODEL_PATH, [hybridization], compiled_path(MODEL_PATH, tmp_path / "model")
    )

    class Compiled(LotkaVolterra):
        model_path: ClassVar[Path] = compiled

    return Compiled


def test_a_network_in_the_right_hand_side(tmp_path: Path) -> None:
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
    elements = network_fit_parameters(network, estimate={"net1": True}, bounds={})
    problem = _problem(
        [hybridization], elements, experiment=_compiled(tmp_path, hybridization)
    )
    assert_round_trip(problem, tmp_path)
    config = _tables(tmp_path / "petab")["config"]
    # the model of the problem is the model without the network
    assert config["model_files"]["lv"]["location"] == "lotka_volterra.xml"
    assert (
        "net1__output0__0"
        not in (tmp_path / "petab" / "lotka_volterra.xml").read_text()
    )
    assert (tmp_path / "petab" / "net1.yaml").is_file()
    assert (tmp_path / "petab" / "net1_arrays.hdf5").is_file()
    assert (tmp_path / "petab" / "hybridization.tsv").is_file()


def test_the_arrays_of_a_compiled_network(tmp_path: Path) -> None:
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
        frozen=set(network.parameter_ids()),
    )
    problem = _problem(
        [hybridization], [], experiment=_compiled(tmp_path, hybridization)
    )
    assert_round_trip(problem, tmp_path)


def test_an_observable_which_shadows_an_entity(tmp_path: Path) -> None:
    """The mapping keys `prey_e1` and `prey_e2` become one observable, not `prey`."""
    network = feed_forward()
    elements = network_fit_parameters(
        network, estimate={"net1": True}, bounds={}, external=True
    )
    assert_round_trip(_problem([_before(network)], elements), tmp_path)
    observables = _tables(tmp_path / "petab")["observables"]
    assert set(observables["observableId"]) == {
        "observable__prey",
        "observable__predator",
    }


def test_the_exported_problem_is_valid_petab(tmp_path: Path) -> None:
    """`petab` reads the networks of a problem through torch, the dev extra has it."""
    pytest.importorskip("torch")
    network = feed_forward()
    elements = network_fit_parameters(
        network, estimate={"net1": True}, bounds={}, external=True
    )
    problem = _problem([_before(network)], elements)
    problem.initialize(SETTINGS)
    yaml_file = to_petab(problem, tmp_path / "petab")
    issues = petab_v2.Problem.from_yaml(yaml_file).validate()
    assert not issues.has_errors(), str(issues)


# --- WHAT IS REFUSED ---


def test_a_partial_array_is_refused(tmp_path: Path) -> None:
    network = feed_forward()
    elements = network_fit_parameters(
        network, estimate={"net1": True}, bounds={}, external=True
    )
    # one element of the bias of the first layer is frozen
    elements = [p for p in elements if p.pid != "net1__layer1__bias__0"]
    problem = _problem([_before(network, frozen={"net1__layer1__bias__0"})], elements)
    problem.initialize(SETTINGS)
    with pytest.raises(ValueError, match="sciml-partial-array"):
        to_petab(problem, tmp_path / "petab")


def test_an_element_which_differs_from_the_network_is_refused(tmp_path: Path) -> None:
    network = feed_forward()
    elements = network_fit_parameters(
        network, estimate={"net1": True}, bounds={}, external=True
    )
    elements[0] = FitParameter(
        elements[0].pid, 99.0, unit="dimensionless", target=elements[0].target
    )
    problem = _problem([_before(network)], elements)
    problem.initialize(SETTINGS)
    with pytest.raises(ValueError, match=r"starts from 99\.0, but the network 'net1'"):
        to_petab(problem, tmp_path / "petab")


def test_an_element_on_another_scale_is_refused(tmp_path: Path) -> None:
    network = feed_forward()
    elements = network_fit_parameters(
        network, estimate={"net1": True}, bounds={}, external=True
    )
    # a positive element, which a logarithmic scale can search
    k = next(i for i, p in enumerate(elements) if float(p.start_value or 0.0) > 0.0)
    first = elements[k]
    elements[k] = FitParameter(
        first.pid,
        first.start_value,
        1e-6,
        10.0,
        unit="dimensionless",
        target=first.target,
        scale=ParameterScaleType.LOG10,
    )
    problem = _problem([_before(network)], elements)
    problem.initialize(SETTINGS)
    with pytest.raises(ValueError, match="has the scale 'LOG10'"):
        to_petab(problem, tmp_path / "petab")


def test_a_hook_which_is_no_network_is_refused() -> None:
    problem = OptimizationProblem(
        opid="hook",
        mapping_collections=_problem([], []).mapping_collections,
        fit_parameters=[*MECHANISTIC, factor_parameter()],
        base_path=MODEL_PATH.parent,
        data_path=MODEL_PATH.parent,
        hybridizations=[Scaling(model="lv", target="gamma")],
    )
    problem.initialize(SETTINGS)
    with pytest.raises(ValueError, match="is not a `Hybridization`"):
        PetabExporter(problem)


# --- THE HELPERS ---


def test_petab_math() -> None:
    assert petab_math("alpha + (prey - 1.3)") == "alpha + prey - 1.3"
    assert petab_math("ln(x)") == "log(x)"
    assert petab_math("10.0") == "10"
    assert petab_math("x / 3") == "x/3"


def test_petab_index() -> None:
    assert petab_index((0, 1), (2, 3)) == "[0][1]"
    assert petab_index((0, 0), (1, 1)) == "[0]"
    assert petab_index((0, 2), (1, 5)) == "[2]"
    assert petab_index((0,), (1,)) == "[0]"
```

`_problem`, `_before`, `MECHANISTIC`, `PRE`, `RHS` and `SETTINGS` are the helpers of `tests/sciml/test_fit.py`; importing them from the test module is fine, the module is a package module of `tests.sciml`. `petab_math` writes what `petab_math_str` of `petab` writes: `1.3` stays a decimal and `ln` becomes `log`.

Replace the body of `tests/fit/test_petab_v2_definition.py::test_an_external_parameter_is_no_condition` (the `petab_problem = ...` line and the assertions) by:

```python
    with pytest.raises(ValueError, match="is not a `Hybridization`"):
        PetabExporter(problem)
```

and rename it to `test_a_problem_with_a_hook_which_is_no_network_is_refused`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync pytest -q -x tests/sciml/test_export.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbmlsim.fit.petab_v2.sciml_export'`

- [ ] **Step 3: The gap**

Apply to `src/sbmlsim/fit/petab_v2/gaps.py`, after the gap `sciml-parameter-scale`:

```python
(
    Gap(
        id="sciml-partial-array",
        kind=GapKind.UNSUPPORTED,
        sbmlsim="every element of a network is its own `FitParameter`, so a "
        "fit can estimate some elements of an array or bound them differently",
        petab="a row of the parameter table of PEtab SciML describes the "
        "network, a layer or an array as a whole",
        detail="the export raises for an array whose elements differ in "
        "whether they are estimated or in their bounds and names the array. "
        "`network_fit_parameters` and `Hybridization.fit_parameters` describe "
        "the elements per network, layer and array, which is what PEtab "
        "expresses",
    ),
)
```

- [ ] **Step 4: Create the module**

Create `src/sbmlsim/fit/petab_v2/sciml_export.py`:

```python
"""Write the neural networks of a hybrid problem as PEtab SciML.

The inverse of `sbmlsim.fit.petab_v2.sciml`: the hybridizations of an
`OptimizationProblem` become the block `sciml` of the problem, the
hybridization table, the rows of the mapping table and of the parameter table
which name the networks, the NN YAML of every network and one array file per
network with its arrays and the arrays of its inputs.

| `sbmlsim` | PEtab SciML |
| --- | --- |
| `Network` | the NN YAML and the arrays of the array file |
| `Hybridization.pattern` | `pre_initialization` of the network, and the observables |
| an input which is a formula | a row of the hybridization table |
| an input with a formula per condition | a change of the condition of the experiment |
| an input which is arrays | `array` in the hybridization table, the arrays in the array file by condition |
| an output of `RHS` or `PRE_INITIALIZATION` | a row of the hybridization table which assigns the target |
| an output of `OBSERVABLE` | the symbol of the observable formula, mapped to the output |
| `frozen` and the bounds of the elements | the most general rows of the parameter table |
| `constants` | rows of the parameter table which are not estimated |

The module imports `sbmlsim.sciml` and with it `petab_sciml`, which is the
extra `sciml`. `sbmlsim.fit.petab_v2.export` imports it only for a problem
with hybridizations.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import petab.v2 as petab_v2
import sympy
from petab.v2.extensions.sciml import Hybridization as HybridizationRow
from petab.v2.extensions.sciml import (
    HybridizationTable,
    NeuralNetConfig,
    SciMLConfig,
)
from petab.v2.math import petab_math_str
from petab_sciml import ArrayData, ArrayDataStandard, Metadata, NNModelStandard
from petab_sciml.constants import ALL_CONDITION_IDS, ARRAY

from sbmlsim.fit.objects import FitParameter
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.fit.petab_v2.sciml import YAML_FORMAT
from sbmlsim.mathml import formula_expression
from sbmlsim.sciml.hybridization import (
    ALL_CONDITIONS,
    Hybridization,
    NetworkPattern,
    entity_of,
)
from sbmlsim.sciml.network import Network, parse_io_id
from sbmlsim.sciml.parameters import ELEMENT_UNIT

if TYPE_CHECKING:
    from sbmlsim.fit.optimization import OptimizationProblem

logger = logging.getLogger(__name__)

#: the file of the hybridization table
HYBRIDIZATION_FILE = "hybridization.tsv"

#: the suffix of the `petabEntityId` of the parameters of a network, a
#: layer or an array, e.g. `net1__parameters`, `net1__layer1__parameters`
PARAMETERS_SUFFIX = "parameters"


def network_yaml(sid: str) -> str:
    """Get the file of the NN YAML of a network."""
    return f"{sid}.yaml"


def network_arrays(sid: str) -> str:
    """Get the array file of a network, with its arrays and its inputs."""
    return f"{sid}_arrays.hdf5"


def petab_math(formula: str) -> str:
    """Get the math of PEtab of an L3 formula of SBML.

    Args:
        formula: the formula, e.g. `alpha + (prey - 1.3)`.

    Returns:
        The expression in the math of PEtab, in which `log` is the natural
        logarithm. A decimal of the formula is written as a decimal, not as
        the fraction the parser reads it as.

    Raises:
        ValueError: if the formula is not valid math.
    """
    expression = formula_expression(formula)

    def is_decimal(atom: sympy.Basic) -> bool:
        if not isinstance(atom, sympy.Rational) or atom.is_Integer:
            return False
        q = int(atom.q)
        while q % 10 == 0:
            q //= 10
        return q == 1

    expression = expression.replace(is_decimal, lambda atom: sympy.Float(atom))
    return petab_math_str(expression)


def petab_index(index: tuple[int, ...], shape: tuple[int, ...]) -> str:
    """Get the index of an element of an output as PEtab SciML writes it.

    Args:
        index: the index of the element in the output.
        shape: the shape of the output.

    Returns:
        The index, e.g. `[0][1]`, without the leading axes of length one,
        which are the axes of the batch, see `sbmlsim.fit.petab_v2.sciml`.
    """
    leading = 0
    while leading < len(shape) - 1 and shape[leading] == 1 and index[leading] == 0:
        leading += 1
    return "".join(f"[{i}]" for i in index[leading:])


def parameters_id(
    network: str, layer: str | None = None, array: str | None = None
) -> str:
    """Get the `petabEntityId` of the parameters of a network, a layer or an array."""
    parts = [network, layer, array, PARAMETERS_SUFFIX]
    return "__".join(part for part in parts if part)


def model_entity_id(
    network: str, layer: str | None = None, array: str | None = None
) -> str:
    """Get the `modelEntityId` of the parameters of a network, a layer or an array."""
    if layer is None:
        return f"{network}.parameters"
    if array is None:
        return f"{network}.parameters[{layer}]"
    return f"{network}.parameters[{layer}].{array}"


@dataclass
class ArrayRow:
    """The row of the parameter table an array of a network is described by."""

    estimate: bool
    lower: float
    upper: float

    def key(self) -> tuple[bool, float, float]:
        """Get the row as a hashable key."""
        return (self.estimate, self.lower, self.upper)


@dataclass
class ExportedNetwork:
    """A network of the problem with what is written for it.

    Attributes:
        network: the network.
        hybridizations: its hybridizations, one or two.
        pre_initialization: whether it runs before the simulation.
        arrays: the array file of the network, with the arrays of its inputs.
        parameter_rows: the rows of the parameter table, `petabEntityId` ->
            `modelEntityId` and the row.
    """

    network: Network
    hybridizations: list[Hybridization]
    pre_initialization: bool
    arrays: ArrayData
    parameter_rows: dict[str, tuple[str, ArrayRow]] = field(default_factory=dict)


class SciMLExporter:
    """Write the networks of an optimization problem as PEtab SciML.

    The exporter is created by `PetabExporter` for a problem with
    hybridizations and fills the tables of the problem it builds. It
    validates the hybridizations against the fit parameters of the problem:
    every element of a network which is a fit parameter has the value of the
    network as its start value, the linear scale and the unit
    `dimensionless`, and the elements of one array are either all estimated
    with one pair of bounds or all frozen.

    Attributes:
        problem: the problem which is exported.
        simulation_ids: id of the simulation of a fit mapping -> id of the
            PEtab experiment.
        networks: the networks by their id.
        element_ids: the ids of the elements of all networks, which are no
            rows of the parameter table and no part of the `sbmlsim` block.
        constants: the constants of the hybridizations which are rows of
            the parameter table, id -> value.
        constant_inputs: id of an input whose formula is a constant which
            feeds only this input -> the constant, which is the
            `petabEntityId` of the input.
        hybridization_rows: the rows of the hybridization table.
    """

    def __init__(
        self,
        problem: OptimizationProblem,
        hybridizations: Sequence[Any],
        simulation_ids: Mapping[str, str],
    ) -> None:
        """Initialize the exporter.

        Args:
            problem: the initialized problem.
            hybridizations: the hybridizations of the problem.
            simulation_ids: id of the simulation of a fit mapping (the
                condition of the inputs) -> id of the experiment of PEtab.

        Raises:
            ValueError: if a hybridization is not a `Hybridization` of
                `sbmlsim.sciml`, if a network is before the simulation and in
                the model, if two hybridizations of a network differ in the
                network, the model or the inputs, if an element which is a
                fit parameter differs from the network, if the elements of an
                array are not described by one row, or if two hybridizations
                give a constant different values.
        """
        self.problem = problem
        self.simulation_ids = dict(simulation_ids)
        by_network: dict[str, list[Hybridization]] = {}
        for hybridization in hybridizations:
            if not isinstance(hybridization, Hybridization):
                raise ValueError(
                    f"'{problem.opid}': the hybridization '{hybridization}' is not "
                    f"a `Hybridization` of `sbmlsim.sciml`, only networks are "
                    f"written as PEtab SciML."
                )
            by_network.setdefault(hybridization.network.sid, []).append(hybridization)

        parameters = {p.pid: p for p in problem.parameters}
        self.networks: dict[str, ExportedNetwork] = {}
        for sid, group in by_network.items():
            first = group[0]
            for other in group[1:]:
                if other.network != first.network:
                    raise ValueError(
                        f"'{problem.opid}': the network '{sid}' has two "
                        f"hybridizations with different networks, a network of "
                        f"a problem is one network."
                    )
                if other.model != first.model or other.inputs != first.inputs:
                    raise ValueError(
                        f"'{problem.opid}': the network '{sid}' has two "
                        f"hybridizations which differ in the model or the inputs."
                    )
            patterns = {h.pattern for h in group}
            pre = NetworkPattern.PRE_INITIALIZATION in patterns
            if pre and len(patterns) > 1:
                raise ValueError(
                    f"'{problem.opid}': the network '{sid}' runs before the "
                    f"simulation and is in the model, a network of PEtab SciML "
                    f"is one of the two."
                )
            frozen: set[str] = set()
            for h in group:
                frozen |= set(h.frozen)
            self.networks[sid] = ExportedNetwork(
                network=first.network,
                hybridizations=list(group),
                pre_initialization=pre,
                arrays=ArrayData(
                    metadata=Metadata(pytorch_format=True),
                    parameters={
                        sid: {
                            layer: {
                                name: np.asarray(array, dtype=float)
                                for name, array in arrays.items()
                            }
                            for layer, arrays in first.network.parameters.items()
                        }
                    },
                ),
                parameter_rows=self._parameter_rows(first.network, frozen, parameters),
            )
        self.element_ids: set[str] = {
            sid
            for exported in self.networks.values()
            for sid in exported.network.parameter_ids()
        }
        self.constants: dict[str, float] = {}
        for exported in self.networks.values():
            for h in exported.hybridizations:
                for key, value in h.constants.items():
                    if key in parameters:
                        continue
                    if key in self.constants and self.constants[key] != value:
                        raise ValueError(
                            f"'{problem.opid}': the constant '{key}' has the values "
                            f"'{self.constants[key]}' and '{value}' in two "
                            f"hybridizations."
                        )
                    self.constants[key] = value
        # an input whose formula is a constant which feeds only this input is
        # written the way PEtab SciML writes such an input: the constant is
        # the `petabEntityId` of the input and a row of the parameter table,
        # without a row of the hybridization table. A parameter which only a
        # hybridization row names is extraneous to the linter of `petab`
        formulas = Counter(
            network_input.formula
            for exported in self.networks.values()
            for network_input in exported.hybridizations[0].inputs.values()
            if network_input.formula in self.constants
        )
        self.constant_inputs: dict[str, str] = {
            key: network_input.formula
            for exported in self.networks.values()
            for key, network_input in exported.hybridizations[0].inputs.items()
            if network_input.formula is not None
            and formulas.get(network_input.formula) == 1
        }
        # built once: the arrays of the inputs go into the array files while
        # the rows are built
        self.hybridization_rows: list[HybridizationRow] = self._hybridization_rows()

    # --- THE PARAMETER ROWS ---

    def _parameter_rows(
        self,
        network: Network,
        frozen: set[str],
        parameters: Mapping[str, FitParameter],
    ) -> dict[str, tuple[str, ArrayRow]]:
        """Get the most general rows which describe the elements of a network.

        Args:
            network: the network.
            frozen: the ids of the frozen elements.
            parameters: the fit parameters of the problem by their id.

        Returns:
            `petabEntityId` -> `modelEntityId` and the row: one row for the
            network with the most common description of its arrays, and rows
            for the layers or the arrays which differ from it.

        Raises:
            ValueError: if an element which is a fit parameter does not carry
                the value of the network, the linear scale and the unit
                `dimensionless`, if an element is neither frozen nor a fit
                parameter, or if the elements of one array differ in whether
                they are estimated or in their bounds.
        """
        sid = network.sid
        used = set(network.used_layers())
        rows: dict[tuple[str, str], ArrayRow] = {}
        for element, (layer, name, index) in network.parameter_ids().items():
            if element in frozen or layer not in used:
                row = ArrayRow(False, -np.inf, np.inf)
            elif element in parameters:
                parameter = parameters[element]
                value = float(network.parameters[layer][name][index])
                if parameter.start_value != value:
                    raise ValueError(
                        f"'{self.problem.opid}': the element '{element}' starts "
                        f"from {parameter.start_value}, but the network '{sid}' "
                        f"carries {value}. The values of a network are the start "
                        f"values of its elements, set them on the network."
                    )
                if parameter.scale not in (None, ParameterScaleType.LINEAR):
                    raise ValueError(
                        f"'{self.problem.opid}': the element '{element}' has the "
                        f"scale '{parameter.scale.name}', the elements of a "
                        f"network are on the linear scale."
                    )
                if parameter.unit not in (None, ELEMENT_UNIT):
                    raise ValueError(
                        f"'{self.problem.opid}': the element '{element}' has the "
                        f"unit '{parameter.unit}', the elements of a network are "
                        f"'{ELEMENT_UNIT}'."
                    )
                row = ArrayRow(True, parameter.lower_bound, parameter.upper_bound)
            else:
                raise ValueError(
                    f"'{self.problem.opid}': the element '{element}' of the "
                    f"network '{sid}' is neither frozen nor a parameter of the fit."
                )
            first = rows.setdefault((layer, name), row)
            if first.key() != row.key():
                raise ValueError(
                    f"'{self.problem.opid}': the elements of the array '{name}' "
                    f"of the layer '{layer}' of the network '{sid}' differ in "
                    f"whether they are estimated or in their bounds "
                    f"({first} and {row} for '{element}'). A row of the parameter "
                    f"table of PEtab SciML describes an array as a whole "
                    f"(gap 'sciml-partial-array')."
                )

        # the most common row is the row of the network, the layers and the
        # arrays which differ get rows of their own
        counts = Counter(row.key() for row in rows.values())
        network_key = counts.most_common(1)[0][0]
        result: dict[str, tuple[str, ArrayRow]] = {
            parameters_id(sid): (model_entity_id(sid), ArrayRow(*network_key))
        }
        layers: dict[str, list[tuple[str, ArrayRow]]] = {}
        for (layer, name), row in rows.items():
            layers.setdefault(layer, []).append((name, row))
        for layer, arrays in layers.items():
            keys = {row.key() for _, row in arrays}
            if keys == {network_key}:
                continue
            if len(keys) == 1:
                result[parameters_id(sid, layer)] = (
                    model_entity_id(sid, layer),
                    ArrayRow(*next(iter(keys))),
                )
                continue
            for name, row in arrays:
                if row.key() != network_key:
                    result[parameters_id(sid, layer, name)] = (
                        model_entity_id(sid, layer, name),
                        row,
                    )
        return result

    # --- THE TABLES ---

    def condition_of(self, simulation: str) -> str:
        """Get the id of the condition of the first period of an experiment.

        Args:
            simulation: id of the simulation of the experiment.

        Returns:
            The id, which `PetabExporter._periods` gives the first period.
        """
        return f"{self.simulation_ids[simulation]}__tc0"

    def input_changes(self, simulation: str) -> list[petab_v2.Change]:
        """Get the changes of the condition of a simulation which set inputs.

        Args:
            simulation: id of the simulation.

        Returns:
            One change per input which has a formula for the condition of the
            simulation, i.e. per input which differs between the conditions.

        Raises:
            ValueError: if such an input has no formula for the simulation.
        """
        changes: list[petab_v2.Change] = []
        for exported in self.networks.values():
            for key, network_input in exported.hybridizations[0].inputs.items():
                if network_input.formulas is None:
                    continue
                formula = network_input.formula_of(simulation)
                if formula is None:
                    raise ValueError(
                        f"'{self.problem.opid}': the input '{key}' has no formula "
                        f"for the simulation '{simulation}', it has formulas for "
                        f"{sorted(network_input.formulas)}."
                    )
                changes.append(
                    petab_v2.Change(target_id=key, target_value=petab_math(formula))
                )
        return changes

    def needs_condition(self, simulation: str) -> bool:
        """Check whether the first period of an experiment needs a condition.

        An input which differs between the conditions needs the condition of
        the experiment: its formula is a change of it, its arrays are keyed
        by it in the array file.

        Args:
            simulation: id of the simulation.
        """
        return any(
            network_input.is_conditional
            for exported in self.networks.values()
            for network_input in exported.hybridizations[0].inputs.values()
        )

    def mapping_rows(self) -> list[petab_v2.Mapping]:
        """Get the rows of the mapping table which name parts of the networks.

        Returns:
            A row per input, per used output and per row of the parameter
            table of the networks. The `petabEntityId` of an input and of an
            output which sets an entity is its id in `sbmlsim`, the one of an
            input which is a constant of its own is the constant, the one of
            an output an observable uses is its target, i.e. the symbol of
            the observable formula.
        """
        rows: list[petab_v2.Mapping] = []
        for sid, exported in self.networks.items():
            hybridization = exported.hybridizations[0]
            for key in hybridization.inputs:
                k, index = parse_io_id(sid, "input", key)
                indices = "" if index is None else "".join(f"[{i}]" for i in index)
                rows.append(
                    petab_v2.Mapping(
                        petab_id=self.constant_inputs.get(key, key),
                        model_id=f"{sid}.inputs[{k}]{indices}",
                    )
                )
            shapes = hybridization.output_shapes()
            for h in exported.hybridizations:
                for key, target in h.outputs.items():
                    k, index = parse_io_id(sid, "output", key)
                    petab_id = target if h.pattern is NetworkPattern.OBSERVABLE else key
                    indices = petab_index(index or (), shapes[k])
                    rows.append(
                        petab_v2.Mapping(
                            petab_id=petab_id, model_id=f"{sid}.outputs[{k}]{indices}"
                        )
                    )
            for petab_id, (model_id, _) in exported.parameter_rows.items():
                rows.append(petab_v2.Mapping(petab_id=petab_id, model_id=model_id))
        return rows

    def _hybridization_rows(self) -> list[HybridizationRow]:
        """Get the rows of the hybridization table.

        Returns:
            A row per input which is a formula for every condition
            (`targetValue` is the math) or arrays (`targetValue` is `array`,
            the arrays go into the array file of the network keyed by the
            conditions), and a row per output of a network before the
            simulation or in the right hand side, which assigns the output
            to its target.
        """
        rows: list[HybridizationRow] = []
        for exported in self.networks.values():
            hybridization = exported.hybridizations[0]
            for key, network_input in hybridization.inputs.items():
                if key in self.constant_inputs:
                    continue
                if network_input.formula is not None:
                    rows.append(
                        HybridizationRow(
                            target_id=key,
                            target_value=petab_math(network_input.formula),
                        )
                    )
                elif network_input.arrays is not None:
                    rows.append(HybridizationRow(target_id=key, target_value=ARRAY))
                    exported.arrays.inputs[key] = {
                        (
                            ALL_CONDITION_IDS
                            if condition == ALL_CONDITIONS
                            else self.condition_of(condition)
                        ): np.asarray(array, dtype=float)
                        for condition, array in network_input.arrays.items()
                    }
            for h in exported.hybridizations:
                if h.pattern is NetworkPattern.OBSERVABLE:
                    continue
                for key, target in h.outputs.items():
                    rows.append(
                        HybridizationRow(target_id=entity_of(target), target_value=key)
                    )
        return rows

    def parameter_rows(self) -> list[petab_v2.Parameter]:
        """Get the rows of the parameter table of the networks and the constants.

        Returns:
            The rows of the arrays of every network with the nominal value
            `array`, and one row per constant of the hybridizations which is
            not estimated.
        """
        rows: list[petab_v2.Parameter] = []
        for exported in self.networks.values():
            for petab_id, (_, row) in exported.parameter_rows.items():
                rows.append(
                    petab_v2.Parameter(
                        id=petab_id,
                        lb=row.lower if row.estimate else None,
                        ub=row.upper if row.estimate else None,
                        nominal_value=ARRAY,
                        estimate=row.estimate,
                    )
                )
        for key, value in self.constants.items():
            rows.append(
                petab_v2.Parameter(
                    id=key, lb=None, ub=None, nominal_value=value, estimate=False
                )
            )
        return rows

    def config(self) -> SciMLConfig:
        """Get the block of the extension `sciml` of the problem."""
        return SciMLConfig(
            required=True,
            array_files=[Path(network_arrays(sid)) for sid in self.networks],
            hybridization_files=[Path(HYBRIDIZATION_FILE)],
            neural_networks={
                sid: NeuralNetConfig(
                    location=Path(network_yaml(sid)),
                    pre_initialization=exported.pre_initialization,
                    format=YAML_FORMAT.upper(),
                )
                for sid, exported in self.networks.items()
            },
        )

    def write(self, output_dir: Path) -> None:
        """Write the files of the networks into the directory of the problem.

        Args:
            output_dir: the directory of the problem. The NN YAML and the
                array file of every network and the hybridization table are
                written next to `problem.yaml`.
        """
        output_dir = Path(output_dir)
        for sid, exported in self.networks.items():
            NNModelStandard.save_data(
                exported.network.model, str(output_dir / network_yaml(sid))
            )
            ArrayDataStandard.save_data(
                exported.arrays, str(output_dir / network_arrays(sid))
            )
        HybridizationTable(self.hybridization_rows).to_tsv(
            output_dir / HYBRIDIZATION_FILE
        )
        logger.info(
            "The networks %s are written as PEtab SciML into '%s'",
            sorted(self.networks),
            output_dir,
        )
```

- [ ] **Step 5: Wire the exporter into `export.py`**

Apply to `src/sbmlsim/fit/petab_v2/export.py`:

```diff
@@
 from sbmlsim.fit.petab_v2.extension import (
     EXTENSION_ID,
+    SCIML_EXTENSION_ID,
     SbmlsimExtension,
 )
@@ class PetabExporter.__init__
         #: the networks of the problem, `None` for a problem without them
         self.sciml: Any = None
+        if problem.hybridizations:
+            # the import needs the extra `sciml`
+            from sbmlsim.fit.petab_v2.sciml_export import SciMLExporter
+
+            self.sciml = SciMLExporter(
+                problem,
+                problem.hybridizations,
+                simulation_ids=self._simulation_ids(),
+            )
@@ def _add_parameters(self, petab_problem: PetabProblem) -> None:
         no version applied.
         """
+        elements = self.sciml.element_ids if self.sciml is not None else set()
         for index, parameter in enumerate(self.problem.parameters):
+            if parameter.pid in elements:
+                # the rows of the networks describe the elements
+                continue
             nominal_value = (
@@
         self._add_noise_parameters(petab_problem)
+        if self.sciml is not None:
+            _table(petab_problem, "parameter_tables").parameters.extend(
+                self.sciml.parameter_rows()
+            )
+            _table(petab_problem, "mapping_tables").mappings.extend(
+                self.sciml.mapping_rows()
+            )
@@ def _add_extension(self, petab_problem: PetabProblem) -> None:
+        elements = self.sciml.element_ids if self.sciml is not None else set()
         parameters = {
             parameter.pid: {
@@
             }
             for parameter in problem.parameters
+            if parameter.pid not in elements
         }
@@
         petab_problem.config.extensions[EXTENSION_ID] = extension
+        if self.sciml is not None:
+            petab_problem.config.extensions[SCIML_EXTENSION_ID] = self.sciml.config()
@@ def to_petab(
     _set_table_paths(petab_problem)
     petab_problem.to_files(base_path=output_dir)
+    if exporter.sciml is not None:
+        exporter.sciml.write(output_dir)
     yaml_file = output_dir / YAML_FILE
```

`TABLE_FILES` and `_table` get the entry `"mapping_tables": "mapping.tsv"` and `"mapping_tables": petab_v2.MappingTable` (the mapping table is written by `to_files` when it has a `rel_path`). Extend the docstring of `PetabExporter` with: "A problem with hybridizations is written with its networks as PEtab SciML, see `sbmlsim.fit.petab_v2.sciml_export`, which needs the extra `sciml`."

- [ ] **Step 6: Run the tests**

Run: `uv run --no-sync pytest -q -x tests/sciml/test_export.py tests/fit/test_petab_v2_definition.py tests/fit/test_petab_v2.py tests/sciml/test_package.py`
Expected: PASS

- [ ] **Step 7: API pages**

Create `docs/api/fit.petab_v2.sciml.md` (`# fit.petab_v2.sciml` / `::: sbmlsim.fit.petab_v2.sciml`) and `docs/api/fit.petab_v2.sciml_export.md` (`# fit.petab_v2.sciml_export` / `::: sbmlsim.fit.petab_v2.sciml_export`). Add the rows `| [fit.petab_v2.likelihood](fit.petab_v2.likelihood.md) | the log-likelihood and its gradient, the noise models of PEtab v2 |` (the page exists, the table lacked the row), `| [fit.petab_v2.sciml](fit.petab_v2.sciml.md) | the networks of a PEtab SciML problem read into `Hybridization` objects |` and `| [fit.petab_v2.sciml_export](fit.petab_v2.sciml_export.md) | the hybridizations of a problem written as PEtab SciML |` after the row `fit.petab_v2.gaps` of the table `sbmlsim.fit` of `docs/api/index.md`, and the nav entries `{ "petab_v2.sciml" = "api/fit.petab_v2.sciml.md" }` and `{ "petab_v2.sciml_export" = "api/fit.petab_v2.sciml_export.md" }` after `petab_v2.likelihood` in `zensical.toml`.

Run: `uv run --no-sync zensical build --clean 2>&1 | tail -3`
Expected: `No issues found`

- [ ] **Step 8: Lint, type check, full suite, commit**

Run: `uv run --no-sync ruff check && uv run --no-sync ruff format --check && uvx ty check && uv run --no-sync pytest -q`
Expected: zero diagnostics, all tests pass, 0 warnings

```bash
git add src/sbmlsim/fit/petab_v2/sciml_export.py src/sbmlsim/fit/petab_v2/export.py src/sbmlsim/fit/petab_v2/gaps.py tests/sciml/test_export.py tests/fit/test_petab_v2_definition.py docs/api/fit.petab_v2.sciml.md docs/api/fit.petab_v2.sciml_export.md docs/api/index.md zensical.toml
git commit -m "petab: a hybrid problem is written as PEtab SciML, the round trip of a python defined fit is exact"
```

---

### Task 4: The round trip of every case of the test suite, the priors of a problem

**Files:**
- Modify: `src/sbmlsim/sciml/testsuite.py` (`ProblemImportCase`, after `_compare`)
- Modify: `tests/sciml/test_testsuite.py` (a test per case)
- Modify: `src/sbmlsim/fit/petab_v2/reader.py:1214-1229` (`fit_parameters`, the loop)
- Modify: `src/sbmlsim/fit/petab_v2/gaps.py` (after `sciml-priors`)
- Modify: `src/sbmlsim/fit/runner.py:325-330` (`GUARD_MESSAGE`)
- Modify: `tests/fit/test_petab_v2_reader.py` (a test of the warning)
- Modify: `docs/petab.md` (the list of the gaps)

**Interfaces:**
- Consumes: `to_petab`, `PetabReader`, `ProblemImportCase` (`settings()`, `problem_path`).
- Produces: `ProblemImportCase.round_trip(directory: Path) -> list[str]` (the differences between the problem read from the case and the problem read from its export, empty when exact); `tests/sciml/test_testsuite.py::test_problem_round_trip[cid]` with the marker `sciml_testsuite`; the gap `priors` (LOSSY) and the warning of the reader for a prior of a parameter it fits.

- [ ] **Step 1: The round trip of a case**

Add to `src/sbmlsim/sciml/testsuite.py`, in `ProblemImportCase` after `_compare`:

```python
    def round_trip(self, directory: Path) -> list[str]:
        """Write the problem of the case as PEtab SciML and read it back.

        The problem is read twice from the case and once from its export:
        the first model roadrunner loads in a process differs by about
        `1e-9` from every later one, so the export is compared with the
        second read, bit for bit.

        Args:
            directory: the directory the export and the derived models are
                written to.

        Returns:
            What differs between the problem of the case and the problem
            which is read back, one line each: the parameters (id, start
            value, bounds, unit, scale, target), the hybridizations, the
            data, the kinds and the weights of the fit mappings, the
            predictions and the log-likelihood. Empty when the round trip
            is exact.
        """
        settings = self.settings()

        def read(path: Path, derived: Path) -> tuple[PetabReader, OptimizationProblem]:
            reader = PetabReader.from_yaml(path)
            reader.derived_dir = derived
            problem = reader.to_optimization_problem(opid=f"case_{self.cid}")
            problem.initialize(settings)
            return reader, problem

        _, first = read(self.problem_path, directory / "first")
        yaml_file = to_petab(first, directory / "petab")
        _, original = read(self.problem_path, directory / "second")
        reader, restored = read(yaml_file, directory / "restored")

        def parameter(p: FitParameter) -> tuple[Any, ...]:
            return (
                p.pid,
                p.start_value,
                p.lower_bound,
                p.upper_bound,
                p.unit,
                p.scale,
                p.target,
                p.is_versioned,
            )

        differences: list[str] = []
        expected = [parameter(p) for p in original.parameters]
        observed = [parameter(p) for p in restored.parameters]
        if expected != observed:
            differences.append(
                f"the parameters differ: {sorted(set(expected) ^ set(observed))}"
            )
        if len(original.hybridizations) != len(restored.hybridizations):
            differences.append("the number of hybridizations differs")
        pairs = zip(original.hybridizations, restored.hybridizations, strict=False)
        for k, (h1, h2) in enumerate(pairs):
            for name in (
                "network",
                "pattern",
                "model",
                "inputs",
                "outputs",
                "frozen",
                "constants",
            ):
                if getattr(h1, name, None) != getattr(h2, name, None):
                    differences.append(
                        f"the hybridization {k} ({type(h1).__name__}): {name}"
                    )
        if len(original.mapping_keys) != len(restored.mapping_keys):
            differences.append("the number of fit mappings differs")
            return differences

        keys = {
            (original.experiment_keys[k], original.mapping_keys[k]): k
            for k in range(len(original.mapping_keys))
        }
        x = np.asarray(original.x0, dtype=float)
        values = dict(zip(original.pids, x, strict=True))
        predictions = original.predictions(x)
        restored_predictions = restored.predictions(
            np.asarray([values[pid] for pid in restored.pids], dtype=float)
        )
        for i, key in enumerate(restored.mapping_keys):
            info = reader.observable_info(key)
            k = keys.get((info.get("experiment"), info.get("mapping")))
            if k is None:
                differences.append(f"the fit mapping '{key}' is not one of the case")
                continue
            if not np.array_equal(restored.x_references[i], original.x_references[k]):
                differences.append(f"the times of '{key}'")
            if not np.array_equal(restored.y_references[i], original.y_references[k]):
                differences.append(f"the data of '{key}'")
            if restored.mapping_kinds[i] != original.mapping_kinds[k]:
                differences.append(f"the kind of '{key}'")
            if restored.weights_curves[i] != original.weights_curves[k]:
                differences.append(f"the weight of '{key}'")
            if not np.array_equal(restored_predictions[i], predictions[k]):
                worst = np.max(np.abs(restored_predictions[i] - predictions[k]))
                differences.append(f"the predictions of '{key}' differ by {worst:.2e}")
        llh, restored_llh = log_likelihood(original), log_likelihood(restored)
        if llh != restored_llh:
            differences.append(f"the log-likelihood: {llh} and {restored_llh}")
        return differences
```

with the imports `from sbmlsim.fit.objects import FitParameter`, `from sbmlsim.fit.optimization import OptimizationProblem` and `from sbmlsim.fit.petab_v2.export import to_petab` added to the module (it already imports `PetabReader`, `log_likelihood`, `np`, `Path`, `Any`).

- [ ] **Step 2: The test per case**

Add to `tests/sciml/test_testsuite.py` after `test_problem_import`:

```python
@pytest.mark.sciml_testsuite
@pytest.mark.parametrize("cid", PROBLEM_IMPORT_IDS)
def test_problem_round_trip(cid: str, suite: SciMLSuite, tmp_path: Path) -> None:
    """A case which is read is written as PEtab SciML and read back exactly."""
    case = next(c for c in suite.problem_import_cases() if c.cid == cid)
    if case.llh is None:
        pytest.skip("the case states a log-posterior and is not read (sciml-priors)")
    assert case.round_trip(tmp_path) == []
```

`PROBLEM_IMPORT_IDS` is the list the module parametrizes `test_problem_import` with.

Run: `uv run --no-sync pytest -q -x -m sciml_testsuite tests/sciml/test_testsuite.py -k round_trip -p no:randomly`
Expected: 36 passed, 3 skipped (the suite is in the cache `~/.cache/sbmlsim/petab-sciml-testsuite/`; `tox r -e sciml` downloads it otherwise). Also run `uv run --no-sync python scripts/sciml_testsuite.py run` if the script has a `run` command, to see that nothing else changed: 36 of 39 pass as before.

- [ ] **Step 3: The priors of a parameter the fit uses**

Apply to `src/sbmlsim/fit/petab_v2/reader.py`, in `fit_parameters`, after the `continue` of the parameter which is not an entity (the block which logs "is estimated by the problem but is not an entity of a model"):

```python
            if parameter.prior_distribution is not None:
                # the objective of `sbmlsim` has no priors, see the gap
                logger.warning(
                    "The parameter '%s' has the prior '%s', which the objective "
                    "of `sbmlsim` does not use (gap 'priors', issue #190).",
                    parameter.id,
                    parameter.prior_distribution,
                )
```

Add to `src/sbmlsim/fit/petab_v2/gaps.py` after the gap `sciml-priors`:

```python
(
    Gap(
        id="priors",
        kind=GapKind.LOSSY,
        sbmlsim="the objective of a fit is a weighted least squares, without "
        "priors on the parameters (issue #190)",
        petab="`priorDistribution` and `priorParameters` of a parameter of "
        "the parameter table",
        detail="the reader drops the prior of a parameter of the model with a "
        "warning which names the parameter, the fit and the log-likelihood do "
        "not use it. A prior on the parameters of a network is the gap "
        "`sciml-priors`, which raises",
    ),
)
```

Add to `tests/fit/test_petab_v2_reader.py`:

```python
def test_a_prior_of_a_parameter_is_dropped_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = write_problem(tmp_path / "problem", {"prey_o": "prey"}, {"e1": None})
    parameters = pd.read_csv(tmp_path / "problem" / "parameters.tsv", sep="\t")
    parameters["priorDistribution"] = ["normal", ""]
    parameters["priorParameters"] = ["1.0;0.5", ""]
    parameters.to_csv(tmp_path / "problem" / "parameters.tsv", sep="\t", index=False)
    with caplog.at_level(logging.WARNING, logger="sbmlsim.fit.petab_v2.reader"):
        problem, _ = from_petab(path)
    assert [p.pid for p in problem.parameters] == ["alpha", "beta"]
    assert "The parameter 'alpha' has the prior" in caplog.text
    assert "gap 'priors'" in caplog.text
```

(`import logging` at the top of the test module, and `from_petab` from `sbmlsim.fit.petab_v2.reader` if the module does not import it yet.)

Add to `docs/petab.md`, in the list of the gaps of the SciML section, the line `- `priors`: a prior on a parameter of the model is dropped with a warning which names the parameter, the objective of `sbmlsim` has none (issue #190)`; `tests/fit/test_petab_v2.py::test_gaps_are_documented` derives its list from `GAPS` and needs nothing.

- [ ] **Step 4: The message of the guard**

In `src/sbmlsim/fit/runner.py`, `GUARD_MESSAGE`, replace the last line by `"Use 'serial=True' to fit without worker processes."` (`n_cores=1` still starts one worker, see `run_optimization`). No test asserts the last line of the message.

- [ ] **Step 5: Lint, type check, full suite, commit**

Run: `uv run --no-sync ruff check && uv run --no-sync ruff format --check && uvx ty check && uv run --no-sync pytest -q`
Expected: zero diagnostics, all tests pass, 0 warnings

```bash
git add src/sbmlsim/sciml/testsuite.py tests/sciml/test_testsuite.py src/sbmlsim/fit/petab_v2/reader.py src/sbmlsim/fit/petab_v2/gaps.py src/sbmlsim/fit/runner.py tests/fit/test_petab_v2_reader.py docs/petab.md
git commit -m "testsuite: the round trip of every read case is exact, a prior of a parameter is the gap priors"
```

---

### Task 5: The summary of a hook and the console

**Files:**
- Modify: `src/sbmlsim/fit/derived.py:23-37` (imports), `:37-40` (before `DerivedChanges`), `DerivedChanges` (a method), `:114-127` (`describe`), after `describe`
- Modify: `src/sbmlsim/sciml/hybridization.py` (imports, `Hybridization`, before `symbols`)
- Modify: `src/sbmlsim/fit/display.py` (imports, after `parameters_table`, `print_parameters`)
- Modify: `src/sbmlsim/fit/runner.py:252-259`, `src/sbmlsim/fit/cli.py:555-562` and `:712-719` (the calls of `print_parameters`)
- Modify: `tests/fit/hooks.py` (`Scaling.summary`), `tests/sciml/test_fit.py` (use `fit_parameters`)
- Create: `docs/api/fit.derived.md`, `docs/api/fit.parameter_mapping.md`
- Modify: `docs/api/index.md`, `zensical.toml`
- Test: `tests/fit/test_display.py`, `tests/fit/test_derived_changes.py`, `tests/sciml/test_hybridization.py`, `tests/sciml/test_parameters.py`

**Interfaces:**
- Consumes: `DerivedChanges` protocol, `Hybridization`, `network_fit_parameters`, `display._table`, `_number`, `CoverageRow.pid`.
- Produces: `fit/derived.py`: `ParameterGroup(label: str, ids: tuple[str, ...])`, `HookSummary(name, kind, description, targets: tuple[str, ...], groups: tuple[ParameterGroup, ...])`, `DerivedChanges.summary() -> HookSummary` (a required member of the protocol), `hook_summaries(hooks) -> list[HookSummary]`, `group_parameters(parameters, summaries) -> tuple[list[FitParameter], list[tuple[ParameterGroup, list[FitParameter]]]]`, `describe` with the summary; `Hybridization.summary()` and `Hybridization.fit_parameters(estimate, bounds=None) -> tuple[list[FitParameter], Hybridization]`; `display.hooks_table(summaries)`, `display.groups_table(groups)`, `display.print_parameters(parameters, coverage=None, hooks=None)`, `display.ELEMENT_UNIT_LABEL`. Task 6 uses `group_parameters` and `hook_summaries` in the report.

- [ ] **Step 1: Write the failing tests**

Add to `tests/fit/test_derived_changes.py`:

```python
def test_the_summary_of_a_hook_and_the_groups() -> None:
    from sbmlsim.fit.derived import (
        HookSummary,
        ParameterGroup,
        describe,
        group_parameters,
        hook_summaries,
    )

    scaling = Scaling()
    (summary,) = hook_summaries([scaling])
    assert summary == HookSummary(
        name="scaling",
        kind="scaling",
        description=f"{TARGET} = {FACTOR} * {TARGET}",
        targets=(TARGET,),
        groups=(),
    )
    a, b, c = FitParameter("a", 1.0), FitParameter("b", 2.0), FitParameter("c", 3.0)
    grouped = HookSummary(
        "net",
        "rhs",
        "layer1 (Linear)",
        ("x",),
        (ParameterGroup("net.l.w", ("b", "z")),),
    )
    single, groups = group_parameters([a, b, c], [summary, grouped])
    assert single == [a, c]
    assert groups == [(ParameterGroup("net.l.w", ("b", "z")), [b])]
    assert describe(scaling)["summary"]["name"] == "scaling"
```

Add to `tests/sciml/test_hybridization.py` (the module has a `feed_forward` network and a hybridization builder; use its fixtures, the test below names the objects of `tests.sciml.hybrid`):

```python
def test_the_summary_of_a_hybridization() -> None:
    from tests.sciml.hybrid import feed_forward

    network = feed_forward()
    hybridization = Hybridization(
        network=network,
        pattern=NetworkPattern.RHS,
        model="lv",
        inputs={
            "net1__input0__0": NetworkInput(formula="prey"),
            "net1__input0__1": NetworkInput(formula="predator"),
        },
        outputs={"net1__output0__0": "gamma"},
    )
    summary = hybridization.summary()
    assert summary.name == "net1"
    assert summary.kind == "rhs"
    assert summary.description == "layer1 (Linear), layer2 (Linear)"
    assert summary.targets == ("gamma",)
    assert [group.label for group in summary.groups] == [
        "net1.layer1.weight",
        "net1.layer1.bias",
        "net1.layer2.weight",
        "net1.layer2.bias",
    ]
    assert summary.groups[0].ids == tuple(
        sid
        for sid in network.parameter_ids()
        if sid.startswith("net1__layer1__weight__")
    )


def test_fit_parameters_of_a_hybridization() -> None:
    from tests.sciml.hybrid import feed_forward

    network = feed_forward()
    before = Hybridization(
        network=network,
        pattern=NetworkPattern.PRE_INITIALIZATION,
        model="lv",
        inputs={
            "net1__input0__0": NetworkInput(formula="alpha"),
            "net1__input0__1": NetworkInput(formula="k"),
        },
        outputs={"net1__output0__0": "gamma"},
        constants={"k": 0.5},
    )
    parameters, frozen = before.fit_parameters(
        estimate={"net1.layer2": True}, bounds={"net1": (-2.0, 2.0)}
    )
    assert [p.pid for p in parameters] == [
        sid for sid in network.parameter_ids() if "layer2" in sid
    ]
    assert all(p.is_external for p in parameters)
    assert all((p.lower_bound, p.upper_bound) == (-2.0, 2.0) for p in parameters)
    assert frozen.frozen == set(network.parameter_ids()) - {p.pid for p in parameters}
    assert frozen.pattern is before.pattern and frozen.inputs == before.inputs
    # the elements of a compiled network are entities of the model
    in_model = Hybridization(
        network=network,
        pattern=NetworkPattern.RHS,
        model="lv",
        inputs={
            "net1__input0__0": NetworkInput(formula="prey"),
            "net1__input0__1": NetworkInput(formula="predator"),
        },
        outputs={"net1__output0__0": "gamma"},
    )
    parameters, _ = in_model.fit_parameters(estimate={"net1": True})
    assert all(not p.is_external for p in parameters)
    with pytest.raises(
        NetworkImportError, match="is not the network, a layer or an array"
    ):
        in_model.fit_parameters(estimate={"net1.layer9": True})
```

Add to `tests/fit/test_display.py`:

```python
def test_the_parameters_of_a_hook_are_one_row_per_array(
    capsys: pytest.CaptureFixture,
) -> None:
    from sbmlsim.fit.derived import HookSummary, ParameterGroup
    from sbmlsim.fit.display import groups_table, hooks_table, print_parameters

    elements = [FitParameter(f"net__l__w__{k}", float(k), -5.0, 5.0) for k in range(4)]
    ids = (*[p.pid for p in elements], "net__l__w__4")
    summary = HookSummary(
        name="net",
        kind="pre_initialization",
        description="l (Linear)",
        targets=("gamma",),
        groups=(ParameterGroup("net.l.w", ids),),
    )
    print_parameters(
        [FitParameter("alpha", 1.0, 0.0, 10.0), *elements], hooks=[summary]
    )
    out = capsys.readouterr().out
    assert "alpha" in out
    assert "net.l.w" in out
    assert "net__l__w__0" not in out
    table = groups_table([(summary.groups[0], elements)])
    assert table.row_count == 1
    assert hooks_table([summary]).row_count == 1
```

`capsys` captures the rich console, which writes to `sys.stdout`; `print_parameters` prints the table of the arrays with `print_wide`, so the label is not truncated.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync pytest -q -x tests/fit/test_derived_changes.py tests/sciml/test_hybridization.py tests/fit/test_display.py`
Expected: FAIL with `ImportError: cannot import name 'HookSummary'`

- [ ] **Step 3: The summary in `derived.py`**

Apply to `src/sbmlsim/fit/derived.py`:

```diff
-from collections.abc import Collection, Mapping, Sequence
+from collections.abc import Collection, Iterable, Mapping, Sequence
+from dataclasses import dataclass
 from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
@@
+@dataclass(frozen=True)
+class ParameterGroup:
+    """Parameters of a fit which a hook holds as one array.
+
+    The console and the report show the group as one row, because a network
+    adds hundreds of elements to a fit.
+
+    Attributes:
+        label: what the group is, e.g. `net1.layer1.weight`.
+        ids: the ids of its elements, estimated or not, in the order of the
+            array.
+    """
+
+    label: str
+    ids: tuple[str, ...]
+
+
+@dataclass(frozen=True)
+class HookSummary:
+    """What the console and the report say about a hook.
+
+    Attributes:
+        name: id of the hook, e.g. of the network.
+        kind: where it sits, e.g. the pattern of a network.
+        description: what it is made of, e.g. the layers of a network.
+        targets: the entities it sets.
+        groups: its parameters, as the groups the tables show.
+    """
+
+    name: str
+    kind: str
+    description: str
+    targets: tuple[str, ...]
+    groups: tuple[ParameterGroup, ...]
+
+
 @runtime_checkable
 class DerivedChanges(Protocol):
     """The changes of a simulation which follow from the values of a fit."""

+    def summary(self) -> HookSummary:
+        """Describe the hook for the console and the report."""
+        ...
+
     @property
     def model(self) -> str:
@@ def describe(hybridization: DerivedChanges) -> dict[str, Any]:
-    Returns:
-        The type, the model and the targets of the hook.
+    Returns:
+        The type, the model, the targets and the summary of the hook.
     """
+    summary = hybridization.summary()
     return {
         "type": type(hybridization).__name__,
         "model": hybridization.model,
         "targets": sorted(hybridization.targets()),
+        "summary": {
+            "name": summary.name,
+            "kind": summary.kind,
+            "description": summary.description,
+            "arrays": [group.label for group in summary.groups],
+        },
     }
+
+
+def hook_summaries(hooks: Iterable[DerivedChanges]) -> list[HookSummary]:
+    """Get the summaries of the hooks of a problem, in their order."""
+    return [hook.summary() for hook in hooks]
+
+
+def group_parameters(
+    parameters: Sequence[FitParameter], summaries: Iterable[HookSummary]
+) -> tuple[list[FitParameter], list[tuple[ParameterGroup, list[FitParameter]]]]:
+    """Split the parameters of a fit into single ones and the groups of the hooks.
+
+    Args:
+        parameters: the parameters of the fit.
+        summaries: the summaries of the hooks of the problem.
+
+    Returns:
+        The parameters which belong to no group, in their order, and every
+        group with the parameters of the fit which are its elements. A group
+        without a parameter of the fit is listed with none, i.e. an array
+        which is frozen.
+    """
+    by_id = {p.pid: p for p in parameters}
+    grouped: set[str] = set()
+    groups: list[tuple[ParameterGroup, list[FitParameter]]] = []
+    for summary in summaries:
+        for group in summary.groups:
+            members = [by_id[sid] for sid in group.ids if sid in by_id]
+            grouped.update(group.ids)
+            groups.append((group, members))
+    single = [p for p in parameters if p.pid not in grouped]
+    return single, groups
```

`FitParameter` is imported under `TYPE_CHECKING` in the module already; the module has `from __future__ import annotations`.

- [ ] **Step 4: `Hybridization.summary` and `Hybridization.fit_parameters`**

Apply to `src/sbmlsim/sciml/hybridization.py`:

```diff
-from dataclasses import dataclass, field
+from dataclasses import dataclass, field, replace
@@
+from sbmlsim.fit.derived import HookSummary, ParameterGroup
+from sbmlsim.fit.objects import FitParameter
 from sbmlsim.mathml import TIME, evaluate_formula, formula_symbols
@@
 from sbmlsim.sciml.network import (
@@
 )
+from sbmlsim.sciml.parameters import network_fit_parameters
```

(check the existing `from dataclasses import ...` line and add `replace` to it) and, in `Hybridization` under the comment `# --- WHAT A FIT NEEDS ---` before `symbols`:

```python
def fit_parameters(
    self,
    estimate: Mapping[str, bool],
    bounds: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[list[FitParameter], Hybridization]:
    """Get the parameters of a fit of the network and freeze the rest.

    The pattern decides whether the elements are entities of the model,
    see `sbmlsim.sciml.parameters.network_fit_parameters`, and every
    element which is not estimated is frozen.

    Args:
        estimate: key of the entry -> whether the elements are estimated,
            for the network, a layer or an array.
        bounds: key of the entry -> lower and upper bound, none by
            default.

    Returns:
        The parameters of the fit and the hybridization with the other
        elements frozen.

    Raises:
        NetworkImportError: if a key does not name the network, a layer
            or an array, or if an estimated element has no value.
    """
    parameters = network_fit_parameters(
        self.network,
        estimate=estimate,
        bounds=bounds or {},
        external=not self.pattern.is_compiled,
    )
    frozen = set(self.network.parameter_ids()) - {p.pid for p in parameters}
    return parameters, replace(self, frozen=frozen)


def summary(self) -> HookSummary:
    """Describe the network for the console and the report.

    Returns:
        The id of the network, its pattern, its layers with their types
        in the order of the forward pass, the targets of its outputs and
        one group per array of the layers the forward pass calls, with
        the ids of all elements of the array.
    """
    network = self.network
    types = {layer.layer_id: layer.layer_type for layer in network.model.layers}
    used = network.used_layers()
    groups: dict[tuple[str, str], list[str]] = {}
    for sid, (layer, name, _) in network.parameter_ids().items():
        if layer in used:
            groups.setdefault((layer, name), []).append(sid)
    return HookSummary(
        name=network.sid,
        kind=self.pattern.value,
        description=", ".join(f"{layer} ({types[layer]})" for layer in used),
        targets=tuple(sorted(self.outputs.values())),
        groups=tuple(
            ParameterGroup(label=f"{network.sid}.{layer}.{name}", ids=tuple(ids))
            for (layer, name), ids in groups.items()
        ),
    )
```

`sbmlsim.sciml.parameters` imports `sbmlsim.sciml.network` and `sbmlsim.fit.objects` only, so the import does not cycle; `tests/sciml/test_package.py` keeps pinning that `sbmlsim.fit` imports nothing of `sbmlsim.sciml` at import time.

Add to `tests/fit/hooks.py::Scaling`, before `symbols`:

```python
    def summary(self) -> HookSummary:
        return HookSummary(
            name="scaling",
            kind="scaling",
            description=f"{self.target} = {self.factor} * {self.target}",
            targets=(self.target,),
            groups=(),
        )
```

with `from sbmlsim.fit.derived import HookSummary`. In `tests/fit/test_derived_changes.py::test_a_problem_with_hooks_is_a_dict`, the expected dictionary of the hook gains `"summary": {"name": "scaling", "kind": "scaling", "description": f"{TARGET} = {FACTOR} * {TARGET}", "arrays": []}`.

In `tests/sciml/test_fit.py`, replace the two places which compute `frozen = set(network.parameter_ids()) - {p.pid for p in elements}` by hand (`test_a_frozen_layer`, `test_a_parallel_fit_pickles_the_networks`) by `elements, hybridization = _before(network).fit_parameters(estimate=..., bounds=...)` and use `hybridization` where `_before(network, frozen=frozen)` was; keep the assertion on `frozen` in `test_a_frozen_layer` as `hybridization.frozen == {...}`.

- [ ] **Step 5: The console**

Apply to `src/sbmlsim/fit/display.py`:

```diff
+import numpy as np
 import pandas as pd
@@
 from sbmlsim.console import console
+from sbmlsim.fit.derived import HookSummary, ParameterGroup, group_parameters
 from sbmlsim.fit.objects import FitParameter, MappingKind
```

After `parameters_table`:

```python
#: the unit of the elements of a network, as the tables show it
ELEMENT_UNIT_LABEL = "dimensionless"


def hooks_table(summaries: Iterable[HookSummary]) -> Table:
    """Get the table of the hooks of a problem, e.g. its networks."""
    table = _table("network", "pattern", "layers", "targets")
    for summary in summaries:
        table.add_row(
            summary.name,
            summary.kind,
            summary.description,
            ", ".join(summary.targets),
        )
    return table


def _array_values(members: Sequence[FitParameter]) -> list[str]:
    """Get the minimum, the maximum and the norm of the start values of an array."""
    values = np.asarray([p.start_value for p in members], dtype=float)
    if values.size == 0:
        return ["-", "-", "-"]
    return [
        _number(float(values.min())),
        _number(float(values.max())),
        _number(float(np.linalg.norm(values))),
    ]


def groups_table(
    groups: Sequence[tuple[ParameterGroup, Sequence[FitParameter]]],
) -> Table:
    """Get the table of the arrays of the networks, one row per array.

    An array is shown with the number of its elements, the number of them
    which are estimated and the minimum, the maximum and the norm of the
    start values of the estimated elements, and the bounds when the elements
    agree on them.
    """
    table = _table(
        "array", "elements", "estimated", "min", "max", "norm", "lower", "upper"
    )
    for group, members in groups:
        lower = {p.lower_bound for p in members}
        upper = {p.upper_bound for p in members}
        table.add_row(
            group.label,
            str(len(group.ids)),
            str(len(members)),
            *_array_values(members),
            _number(lower.pop()) if len(lower) == 1 else "-",
            _number(upper.pop()) if len(upper) == 1 else "-",
        )
    return table
```

and replace `print_parameters`:

```python
def print_parameters(
    parameters: Iterable[FitParameter],
    coverage: Sequence[CoverageRow] | None = None,
    hooks: Iterable[HookSummary] | None = None,
) -> None:
    """Print the section of the parameters which are optimized.

    The elements of a network are not listed one by one: the networks are
    printed with their pattern, their layers and their targets, and their
    arrays with the number of elements, the estimated ones and the range and
    the norm of the start values.

    Args:
        parameters: parameters of the fit.
        coverage: what every parameter reaches, from
            `sbmlsim.fit.parameter_mapping.ParameterMapping.coverage`. The
            coverage table is only printed when some parameter does not reach
            every simulation, so an ordinary fit is not given an all-`-`
            table. The elements of the networks are left out of it.
        hooks: the summaries of the hooks of the problem, see
            `sbmlsim.fit.derived.hook_summaries`.
    """
    parameters = list(parameters)
    summaries = list(hooks or [])
    single, groups = group_parameters(parameters, summaries)
    section(f"Parameters ({len(parameters)})", icon=ICON_PARAMETERS)
    if single:
        console.print(parameters_table(single))
    if summaries:
        console.print(hooks_table(summaries))
        print_wide(groups_table(groups))
    grouped = {p.pid for _, members in groups for p in members}
    rows = [row for row in (coverage or []) if row.pid not in grouped]
    if rows and any(row.uncovered_groups for row in rows):
        console.print(coverage_table(rows))
```

`print_wide` is the helper below in the module which prints a table at its own width; the label of an array (`net1.layer1.weight`) was truncated at the width of the console otherwise. If `print_wide` is defined after `print_parameters`, the call still works (the module is loaded before the function runs).

In `src/sbmlsim/fit/runner.py` (`run_optimization`) and in `src/sbmlsim/fit/cli.py` (`report_cli` and `identifiability_cli`), add `hooks=hook_summaries(problem.hybridizations),` to the three calls of `display.print_parameters(...)`, with `from sbmlsim.fit.derived import hook_summaries` in both modules.

- [ ] **Step 6: Run the tests**

Run: `uv run --no-sync pytest -q -x tests/fit/test_derived_changes.py tests/sciml/test_hybridization.py tests/sciml/test_parameters.py tests/fit/test_display.py tests/sciml/test_fit.py tests/fit/test_cli.py tests/fit/test_optimization.py`
Expected: PASS

- [ ] **Step 7: API pages**

Create `docs/api/fit.derived.md` (`# fit.derived` / `::: sbmlsim.fit.derived`) and `docs/api/fit.parameter_mapping.md` (`# fit.parameter_mapping` / `::: sbmlsim.fit.parameter_mapping`); add the rows `| [fit.parameter_mapping](fit.parameter_mapping.md) | which parameter writes which entity in which simulation, the coverage of versioned parameters |` and `| [fit.derived](fit.derived.md) | derived changes of a simulation, the protocol a network before the simulation implements, and the summary of a hook for the console and the report |` after the row `fit.parameters` of the table `sbmlsim.fit` of `docs/api/index.md` and the nav entries `{ "derived" = "api/fit.derived.md" }` (after `optimization`) and `{ "parameter_mapping" = "api/fit.parameter_mapping.md" }` (after `parameters`) to `zensical.toml`.

Run: `uv run --no-sync zensical build --clean 2>&1 | tail -3`
Expected: `No issues found`

- [ ] **Step 8: Lint, type check, full suite, commit**

Run: `uv run --no-sync ruff check && uv run --no-sync ruff format --check && uvx ty check && uv run --no-sync pytest -q`
Expected: zero diagnostics, all tests pass, 0 warnings

```bash
git add src/sbmlsim/fit/derived.py src/sbmlsim/sciml/hybridization.py src/sbmlsim/fit/display.py src/sbmlsim/fit/runner.py src/sbmlsim/fit/cli.py tests/fit/hooks.py tests/sciml/test_fit.py tests/fit/test_derived_changes.py tests/sciml/test_hybridization.py tests/fit/test_display.py docs/api/fit.derived.md docs/api/fit.parameter_mapping.md docs/api/index.md zensical.toml
git commit -m "fit: a hook describes itself, the console shows a network as one row per array"
```

---

### Task 6: The report of a hybrid fit

**Files:**
- Modify: `src/sbmlsim/fit/result.py:40-85` (`bound_warnings`), `:458-478` (`OptimizationResult.report`)
- Modify: `src/sbmlsim/fit/report.py` (imports, `_write_text_report`, `parameters_report`, `html_context`, after `html_context`, `_fisher_context`, `fit_info`)
- Modify: `src/sbmlsim/resources/templates/fit_report.html:63-67` (after the parameters card), `:213` (the correlation heading)
- Modify: `src/sbmlsim/fit/cli.py` (`identifiability_cli`, the call of `profile_likelihood`)
- Modify: `src/sbmlsim/fit/identifiability.py:400-465` (`ParameterProfile.crossing` and `evaluate`: `scale` without default)
- Modify: `tests/fit/test_identifiability.py` (the seven calls of `crossing`/`evaluate`)
- Test: `tests/sciml/test_report.py`, `tests/fit/test_optimization_result.py`

**Interfaces:**
- Consumes: `hook_summaries`, `group_parameters`, `ParameterGroup` of Task 5; `FitReport`, `FisherInformation.summary_df`, `bound_warnings`.
- Produces: `bound_warnings(parameters, x, scales, rtol=0.05, groups: Mapping[str, Collection[str]] | None = None)`, `OptimizationResult.report(path=None, print_output=True, groups=None)`, `FitReport.parameter_groups() -> dict[str, list[str]]`, `FitReport._array_rows(groups, psets)`, the template context keys `hooks`, `arrays`, `fisher.note`; `ParameterProfile.crossing(threshold, direction, scale)` and `evaluate(threshold, flatness_cost, scale)` with `scale` required; `identifiability_cli` profiles the parameters which are no elements of a network unless `-p` names them.

- [ ] **Step 1: Write the failing tests**

Create `tests/sciml/test_report.py`:

```python
"""Tests of the report of a hybrid fit: one row per array of a network."""

from pathlib import Path

import numpy as np

from sbmlsim.fit.fisher import fisher_information
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.fit.report import FitReport
from sbmlsim.fit.result import bound_warnings
from sbmlsim.sciml import network_fit_parameters
from tests.sciml.hybrid import feed_forward
from tests.sciml.test_fit import SETTINGS, _before, _problem


def _report(tmp_path: Path, fisher: bool = False) -> tuple[FitReport, dict]:
    network = feed_forward()
    elements = network_fit_parameters(
        network, estimate={"net1": True}, bounds={"net1": (-5.0, 5.0)}, external=True
    )
    problem = _problem([_before(network)], elements)
    problem.initialize(SETTINGS)
    values = dict(zip(problem.pids, np.asarray(problem.x0, dtype=float), strict=True))
    # one element at its bound
    values["net1__layer1__bias__0"] = 4.99
    pset = ParameterSet(sid="nominal", values=values)
    fim = fisher_information(problem, SETTINGS, pset) if fisher else None
    report = FitReport(problem, SETTINGS, pset, fisher=fim, mapping_figures=False)
    context = report.html_context(tmp_path, "report")
    return report, context


def test_the_overview_shows_the_network_and_its_arrays(tmp_path: Path) -> None:
    report, context = _report(tmp_path)
    assert [row["pid"] for row in context["parameters"]] == ["alpha", "beta"]
    (hook,) = context["hooks"]
    assert hook == {
        "name": "net1",
        "kind": "pre_initialization",
        "description": "layer1 (Linear), layer2 (Linear)",
        "targets": "gamma",
    }
    arrays = {row["label"]: row for row in context["arrays"]}
    assert set(arrays) == {
        "net1.layer1.weight",
        "net1.layer1.bias",
        "net1.layer2.weight",
        "net1.layer2.bias",
    }
    row = arrays["net1.layer1.weight"]
    assert (row["elements"], row["estimated"], row["lower"], row["upper"]) == (
        6,
        6,
        "-5",
        "5",
    )
    (values,) = row["set_values"]
    assert len(values) == 3 and all(value != "-" for value in values)
    assert context["bound_warnings"] == [
        "nominal: !1 of the 3 elements of 'net1.layer1.bias' within 5% of a bound!"
    ]
    assert "net1" in report.fit_info()["networks"]
    path = report.create(tmp_path / "out", name="report")
    html = (path / "index.html").read_text()
    assert "net1.layer1.weight" in html
    assert "net1__layer1__weight__0_0" not in html
    text = (path / "report.txt").read_text()
    assert "1 of the 3 elements of 'net1.layer1.bias'" in text


def test_the_fisher_table_has_one_row_per_array(tmp_path: Path) -> None:
    _, context = _report(tmp_path, fisher=True)
    fisher = context["fisher"]
    labels = [row[0] for row in fisher["rows"]]
    assert labels[:2] == ["alpha", "beta"]
    assert "net1.layer1.weight (6 elements)" in labels
    assert len(labels) == 2 + 4
    assert fisher["pids"] == ["alpha", "beta"]
    assert len(fisher["correlation"]) == 2
    assert "13 elements are left out" in fisher["note"]


def test_bound_warnings_count_the_elements_of_a_group() -> None:
    from sbmlsim.fit import FitParameter
    from sbmlsim.fit.options import ParameterScaleType

    parameters = [FitParameter(f"w{k}", 0.0, -1.0, 1.0) for k in range(3)]
    x = np.array([0.99, -0.99, 0.0])
    scales = [ParameterScaleType.LINEAR] * 3
    assert bound_warnings(parameters, x, scales) == [
        "!Optimal parameter 'w0' within 5% of upper bound!",
        "!Optimal parameter 'w1' within 5% of lower bound!",
    ]
    assert bound_warnings(
        parameters, x, scales, groups={"net.w": ["w0", "w1", "w2"]}
    ) == ["!2 of the 3 elements of 'net.w' within 5% of a bound!"]
```

`feed_forward()` has `n_hidden=3` and `n_inputs=2`: `layer1.weight` 3x2 = 6 elements, `layer1.bias` 3, `layer2.weight` 1x3 = 3, `layer2.bias` 1, in all 13, which the numbers of the test are.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync pytest -q -x tests/sciml/test_report.py`
Expected: FAIL with `KeyError: 'hooks'`

- [ ] **Step 3: `bound_warnings` with groups**

Apply to `src/sbmlsim/fit/result.py`:

```diff
-from collections.abc import Iterable, Sequence
+from collections.abc import Collection, Iterable, Mapping, Sequence
@@ def bound_warnings(
     scales: Sequence[ParameterScaleType],
     rtol: float = 0.05,
+    groups: Mapping[str, Collection[str]] | None = None,
 ) -> list[str]:
@@
         rtol: relative distance to a bound which is reported.
+        groups: label of a group -> the ids of the parameters which are its
+            elements, e.g. the arrays of a network. The elements of a group
+            are reported as one message which counts them, because a network
+            has hundreds.

     Returns:
-        Messages for the parameters which are within `rtol` of one of their bounds.
+        Messages for the parameters which are within `rtol` of one of their
+        bounds, and one message per group with such elements.
     """
+    grouped: dict[str, str] = {
+        pid: label for label, ids in (groups or {}).items() for pid in ids
+    }
     messages: list[str] = []
+    at_bound: dict[str, int] = {}
     for k, (p, scale) in enumerate(zip(parameters, scales, strict=True)):
@@
         for bound, name in [(lb, "lower"), (ub, "upper")]:
             if abs(value - bound) / span < rtol:
+                if p.pid in grouped:
+                    label = grouped[p.pid]
+                    at_bound[label] = at_bound.get(label, 0) + 1
+                    continue
                 messages.append(
                     f"!Optimal parameter '{p.pid}' within {rtol:.0%} of {name} bound!"
                 )
+    sizes = {label: len(ids) for label, ids in (groups or {}).items()}
+    for label, count in at_bound.items():
+        messages.append(
+            f"!{count} of the {sizes[label]} elements of '{label}' within "
+            f"{rtol:.0%} of a bound!"
+        )
     return messages
@@ class OptimizationResult
-    def report(self, path: Path | None = None, print_output: bool = True) -> str:
-        """Report of optimization."""
+    def report(
+        self,
+        path: Path | None = None,
+        print_output: bool = True,
+        groups: Mapping[str, Collection[str]] | None = None,
+    ) -> str:
+        """Report of optimization.
+
+        Args:
+            path: file the report is written to, none by default.
+            print_output: print the report.
+            groups: the groups of parameters which `bound_warnings` reports
+                as one, e.g. the arrays of a network.
+        """
@@
-        for msg in bound_warnings(self.parameters, xopt, self.scales):
+        for msg in bound_warnings(self.parameters, xopt, self.scales, groups=groups):
```

- [ ] **Step 4: The report**

Apply to `src/sbmlsim/fit/report.py`:

```diff
 from sbmlsim.fit import display
+from sbmlsim.fit.derived import ParameterGroup, group_parameters, hook_summaries
 from sbmlsim.fit.fisher import FisherInformation
@@ def _write_text_report(self, path: Path) -> None:
         if self.opt_result:
-            info.append(self.opt_result.report(path=None, print_output=False))
+            info.append(
+                self.opt_result.report(
+                    path=None, print_output=False, groups=self.parameter_groups()
+                )
+            )
@@ def parameters_report(self) -> str:
             for msg in bound_warnings(
                 self.problem.parameters,
                 self.x(pset),
                 self.problem.scales_initialized,
+                groups=self.parameter_groups(),
             ):
@@ def html_context(self, results_dir: Path, name: str) -> dict[str, Any]:
-        # the parameters with one column per set
+        # the parameters with one column per set; the elements of a network
+        # are one row per array, see `_array_rows`
         psets = list(self.parameter_sets)
+        summaries = hook_summaries(self.problem.hybridizations)
+        single, groups = group_parameters(self.problem.parameters, summaries)
         parameters = [
             {
@@
                 "unit": p.unit or "model",
             }
-            for p in self.problem.parameters
+            for p in single
         ]
+        hooks = [
+            {
+                "name": summary.name,
+                "kind": summary.kind,
+                "description": summary.description,
+                "targets": ", ".join(summary.targets),
+            }
+            for summary in summaries
+        ]
+        arrays = self._array_rows(groups, psets)
         warnings: list[str] = []
         for pset in psets:
             warnings.extend(
                 f"{pset.sid}: {message}"
                 for message in bound_warnings(
                     self.problem.parameters,
                     self.x(pset),
                     self.problem.scales_initialized,
+                    groups=self.parameter_groups(),
                 )
             )
@@
             "parameters": parameters,
-            "versioned_parameters": has_renamed_targets(self.problem.parameters),
+            "hooks": hooks,
+            "arrays": arrays,
+            "versioned_parameters": has_renamed_targets(single),
```

Insert after `html_context` (before `_fisher_context`):

```python
def _array_rows(
    self,
    groups: Sequence[tuple[ParameterGroup, Sequence[Any]]],
    psets: Sequence[ParameterSet],
) -> list[dict[str, Any]]:
    """Get the rows of the arrays of the networks for the overview.

    Args:
        groups: the arrays with the parameters of the fit which are their
            elements, see `sbmlsim.fit.derived.group_parameters`.
        psets: the parameter sets of the report.

    Returns:
        One row per array with the number of elements, the estimated
        ones, the bounds when the elements agree on them, and the
        minimum, the maximum and the norm of the values of every set.
    """
    rows: list[dict[str, Any]] = []
    for group, members in groups:
        pids = [p.pid for p in members]
        lower = {p.lower_bound for p in members}
        upper = {p.upper_bound for p in members}
        set_values: list[list[str]] = []
        for pset in psets:
            values = np.asarray(
                [pset.values.get(pid, float("nan")) for pid in pids], dtype=float
            )
            set_values.append(
                ["-", "-", "-"]
                if values.size == 0
                else [
                    f"{values.min():.4g}",
                    f"{values.max():.4g}",
                    f"{np.linalg.norm(values):.4g}",
                ]
            )
        rows.append(
            {
                "label": group.label,
                "elements": len(group.ids),
                "estimated": len(members),
                "lower": f"{lower.pop():.4g}" if len(lower) == 1 else "-",
                "upper": f"{upper.pop():.4g}" if len(upper) == 1 else "-",
                "set_values": set_values,
            }
        )
    return rows


def parameter_groups(self) -> dict[str, list[str]]:
    """Get the arrays of the networks as groups of parameters of the fit.

    Returns:
        label of the array -> the ids of its elements which are
        parameters of the fit, for `bound_warnings`.
    """
    _, groups = group_parameters(
        self.problem.parameters, hook_summaries(self.problem.hybridizations)
    )
    return {group.label: [p.pid for p in members] for group, members in groups}
```

In `_fisher_context`, replace from `correlation = fim.correlation` to the end of the method by:

```python
# the elements of a network are one row per array, with the range of
# their errors, and are left out of the correlation matrix
_, groups = group_parameters(
    self.problem.parameters, hook_summaries(self.problem.hybridizations)
)
grouped = {p.pid for _, members in groups for p in members}
rows: list[list[str]] = [
    [value if isinstance(value, str) else f"{value:.5g}" for value in row.values()]
    for row in df.to_dict(orient="records")
    if row["parameter"] not in grouped
]
for group, members in groups:
    sub = df[df["parameter"].isin([p.pid for p in members])]
    if not len(sub):
        continue
    n = len(sub)
    cells = {
        "parameter": f"{group.label} ({n} element{'s' if n != 1 else ''})",
        "value": f"norm {np.linalg.norm(sub['value'].to_numpy()):.4g}",
        "scale": "LINEAR",
        "se": f"{sub['se'].min():.3g} to {sub['se'].max():.3g}",
        "unit": display.ELEMENT_UNIT_LABEL,
    }
    rows.append([cells.get(column, "-") for column in df.columns])
keep = [i for i, pid in enumerate(fim.pids) if pid not in grouped]
correlation = fim.correlation
return {
    "info": info,
    "identifiable": fim.is_identifiable,
    "columns": [
        {"name": column, "hint": self.HINTS.get(column)} for column in df.columns
    ],
    "rows": rows,
    "eigenvalues": [f"{value:.4g}" for value in eigenvalues],
    "pids": [fim.pids[i] for i in keep],
    "correlation": [[f"{correlation.iloc[i, j]:.3f}" for j in keep] for i in keep],
    "note": (
        f"The correlation is shown for the {len(keep)} parameters which "
        f"are no elements of a network; the {fim.k - len(keep)} elements "
        f"are left out."
        if grouped
        else None
    ),
}
```

In `fit_info`, after the `"experiments"` entry, add:

```python
        summaries = hook_summaries(self.problem.hybridizations)
        if summaries:
            info["networks"] = ", ".join(f"{s.name} ({s.kind})" for s in summaries)
```

(before the `"base path"` entry, the dict literal becomes a dict built in steps: create `info` with the entries up to `"experiments"`, add `networks` when there are summaries, then `info["base path"]` and `info["data path"]`).

Apply to `src/sbmlsim/resources/templates/fit_report.html`, after the `</div>` which closes the parameters card (the one after `bound_warnings`):

```html
  {% if hooks %}
  <h3>Networks</h3>
  <div class="card">
    <table data-filter="row">
      <thead><tr>
        <th data-sort="text">network</th><th data-sort="text">pattern</th>
        <th data-sort="text">layers</th><th data-sort="text">targets</th>
      </tr></thead>
      <tbody>
      {% for hook in hooks %}
      <tr><td class="mono">{{ hook.name }}</td><td>{{ hook.kind }}</td><td class="mono">{{ hook.description }}</td><td class="mono">{{ hook.targets }}</td></tr>
      {% endfor %}
      </tbody>
    </table>
    <table data-filter="row">
      <thead><tr>
        <th data-sort="text">array</th>
        <th class="num" data-sort="num">elements</th><th class="num" data-sort="num">estimated</th>
        {% for pset in parameter_set_ids %}<th class="num" data-sort="num">{{ pset }} min</th><th class="num" data-sort="num">{{ pset }} max</th><th class="num" data-sort="num">{{ pset }} norm</th>{% endfor %}
        <th class="num" data-sort="num">lower</th><th class="num" data-sort="num">upper</th>
      </tr></thead>
      <tbody>
      {% for row in arrays %}
      <tr>
        <td class="mono">{{ row.label }}</td>
        <td class="num mono">{{ row.elements }}</td><td class="num mono">{{ row.estimated }}</td>
        {% for values in row.set_values %}{% for value in values %}<td class="num mono">{{ value }}</td>{% endfor %}{% endfor %}
        <td class="num mono">{{ row.lower }}</td><td class="num mono">{{ row.upper }}</td>
      </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
  {% endif %}
```

and after the `<h3>Correlation ...</h3>` line of the Fisher section: `{% if fisher.note %}<p>{{ fisher.note }}</p>{% endif %}`. Add the hints `"elements": "Number of elements of the array."`, `"estimated": "Number of elements of the array the fit adjusts."` and `"norm": "Euclidean norm of the values of the estimated elements."` to `FitReport.HINTS`.

- [ ] **Step 5: The profiles of a hybrid problem**

In `src/sbmlsim/fit/cli.py`, `identifiability_cli`, before `result = profile_likelihood(`:

```python
    # the elements of a network are profiled only when named: one scan per
    # element is impractical for hundreds of elements
    pids = options.parameter
    if pids is None:
        single, _ = group_parameters(
            problem.parameters, hook_summaries(problem.hybridizations)
        )
        if len(single) < len(problem.parameters):
            logger.info(
                "The %d elements of the networks are not profiled, name an "
                "element with --parameter to profile it.",
                len(problem.parameters) - len(single),
            )
        pids = [p.pid for p in single]
```

and pass `pids=pids` to `profile_likelihood`; import `group_parameters` next to `hook_summaries`. Update the help of `--parameter`: `"parameter to profile, all parameters which are no elements of a network by default; repeatable"`.

In `src/sbmlsim/fit/identifiability.py`, `ParameterProfile.crossing` and `ParameterProfile.evaluate`: remove the default `= ParameterScaleType.LOG10` of `scale`, so that a direct caller must say the scale of the parameter (the sign based fall back it replaced was wrong for a linear parameter around zero); the two internal callers pass it already. In `tests/fit/test_identifiability.py`, add `scale=ParameterScaleType.LOG10` to the calls at the lines 111, 123, 124, 130, 221, 232, 245 and 257 (import `ParameterScaleType` from `sbmlsim.fit.options` if the module does not).

- [ ] **Step 6: Run the tests**

Run: `uv run --no-sync pytest -q -x tests/sciml/test_report.py tests/fit/test_report.py tests/fit/test_optimization_result.py tests/fit/test_identifiability.py tests/fit/test_cli.py tests/fit/test_fisher.py`
Expected: PASS

- [ ] **Step 7: Lint, type check, full suite, commit**

Run: `uv run --no-sync ruff check && uv run --no-sync ruff format --check && uvx ty check && uv run --no-sync pytest -q`
Expected: zero diagnostics, all tests pass, 0 warnings

```bash
git add src/sbmlsim/fit/result.py src/sbmlsim/fit/report.py src/sbmlsim/resources/templates/fit_report.html src/sbmlsim/fit/cli.py src/sbmlsim/fit/identifiability.py tests/fit/test_identifiability.py tests/sciml/test_report.py
git commit -m "report: the networks of a fit in the overview, the arrays as rows, the bound warnings and the Fisher table per array"
```

---

### Task 7: The cleanups the reviews of phase 3 parked

**Files:**
- Modify: `src/sbmlsim/log.py` (a helper), `src/sbmlsim/fit/derived.py:130-138` (`_some`), `src/sbmlsim/sciml/hybridization.py` (`_some`, its two callers and the `missing[:5]` message in `_check_compiled_values`), `src/sbmlsim/fit/petab_v2/likelihood.py:453-462` (`_warn_fall_back`)
- Modify: `examples/sensitivity/sensitivity_example.py:186-195` (`__main__`)
- Modify: `tests/fit/test_parameter_mapping.py:299` (the cost pin)
- Modify: `docs/sensitivity.md:130-136` (the list of the methods)
- Test: `tests/test_log.py` (create if the module has no test file; `rg -l "enable_rich_logging" tests` finds one)

**Interfaces:**
- Consumes: nothing new.
- Produces: `sbmlsim.log.some_ids(ids: Sequence[str], n: int = 5) -> str`, used by the three modules which had a copy of it.

- [ ] **Step 1: Write the failing test**

Add to the test module of `sbmlsim.log` (create `tests/test_log.py` with the module docstring `"""Tests of the logging helpers."""` if none exists):

```python
from sbmlsim.log import some_ids


def test_some_ids_lists_the_first_ids_and_the_count() -> None:
    assert some_ids(["a", "b"]) == "['a', 'b']"
    assert (
        some_ids([f"x{k}" for k in range(7)], n=5)
        == "['x0', 'x1', 'x2', 'x3', 'x4'] ... (7 in total)"
    )
    assert some_ids([], n=5) == "[]"
```

Run: `uv run --no-sync pytest -q -x tests/test_log.py`
Expected: FAIL with `ImportError: cannot import name 'some_ids'`

- [ ] **Step 2: One helper for the three copies**

Add to `src/sbmlsim/log.py`:

```python
def some_ids(ids: Sequence[str], n: int = 5) -> str:
    """Get the first ids of a list and how many there are, for a message.

    The ids a message lists can be the elements of a network, which are
    hundreds or thousands.

    Args:
        ids: the ids.
        n: how many of them are listed.

    Returns:
        The ids as a list, followed by the count when there are more than `n`.
    """
    ids = list(ids)
    if len(ids) <= n:
        return str(ids)
    return f"{ids[:n]} ... ({len(ids)} in total)"
```

with `from collections.abc import Sequence` in the imports of the module. Then:

- `src/sbmlsim/fit/derived.py`: delete `_some` and replace its call `_some(missing)` by `some_ids(missing, n=20)`, with `from sbmlsim.log import some_ids`.
- `src/sbmlsim/sciml/hybridization.py`: delete `_some`, replace `_some(missing)` and `_some(others)` by `some_ids(missing)` and `some_ids(others)`, and in `_check_compiled_values` replace `f"the elements {missing[:5]}{' ...' if len(missing) > 5 else ''} "` by `f"the elements {some_ids(missing)} "`, with `from sbmlsim.log import some_ids`.
- `src/sbmlsim/fit/petab_v2/likelihood.py`, `_warn_fall_back`: replace the two lines which build `listed` by `listed = some_ids(names, n=20)`, with the import.

Run: `uv run --no-sync pytest -q -x tests/test_log.py tests/fit/test_derived_changes.py tests/sciml/test_hybridization.py tests/sciml/test_fit.py tests/fit/test_petab_v2_likelihood.py`
Expected: PASS

- [ ] **Step 3: The sensitivity example runs every method**

In `examples/sensitivity/sensitivity_example.py`, replace

```python
    sas = [
        # sa_local,
        sa_sampling,
        # sa_sobol,
        # sa_fast,
        # sa_morris,
    ]
    for sa in sas:
        sa.execute()
        sa.plot()
```

by

```python
    # every method, the documentation calls this the complete example. The
    # analyses take about a minute in all (measured: local 8 s, sampling
    # 10 s, Sobol 11 s, FAST 18 s, Morris 13 s on 90% of the cores)
    for sa in [sa_local, sa_sampling, sa_sobol, sa_fast, sa_morris]:
        sa.execute()
        sa.plot()
```

Run: `cd $(mktemp -d) && MPLBACKEND=Agg PYTHONPATH=/home/mkoenig/git/sbmlsim-sciml-phase3 uv run --no-sync --project /home/mkoenig/git/sbmlsim-sciml-phase3 python -m examples.sensitivity.sensitivity_example 2>&1 | tail -3; ls results/sensitivity`
Expected: the five directories `local`, `sampling`, `sobol`, `fast`, `morris`, no error.

- [ ] **Step 4: The cost pin and the list of the methods**

In `tests/fit/test_parameter_mapping.py`, line 299, change `rel=1e-3` to `rel=1e-4` and the sentence of the docstring which explains `1e-3` to: "`1e-4` still separates a systematic mis-binding, which moves this cost by orders of magnitude, from integrator noise, and is a hundred times the difference of the integrator."

In `docs/sensitivity.md`, move the paragraph `An output which does not vary over the samples has no Sobol or FAST indices: they are `nan` and the log names the output.` from between the `FAST` and the `Morris` items to after the `Morris` item, so that the list of the four methods is one list.

Run: `uv run --no-sync pytest -q -x tests/fit/test_parameter_mapping.py && uv run --no-sync zensical build --clean 2>&1 | tail -1`
Expected: PASS, `No issues found`

- [ ] **Step 5: Lint, type check, full suite, commit**

Run: `uv run --no-sync ruff check && uv run --no-sync ruff format --check && uvx ty check && uv run --no-sync pytest -q`
Expected: zero diagnostics, all tests pass, 0 warnings

```bash
git add src/sbmlsim/log.py src/sbmlsim/fit/derived.py src/sbmlsim/sciml/hybridization.py src/sbmlsim/fit/petab_v2/likelihood.py examples/sensitivity/sensitivity_example.py tests/fit/test_parameter_mapping.py tests/test_log.py docs/sensitivity.md
git commit -m "cleanup: one helper lists some ids, the sensitivity example runs every method, the cost pin is tighter"
```

---

### Task 8: The example of a problem which is read: Lotka-Volterra of case 001

**Files:**
- Create: `examples/sciml/__init__.py`, `examples/sciml/lotka_volterra/README.md`, `examples/sciml/lotka_volterra/` (the ten files of the case), `examples/sciml/lotka_volterra_fit.py`
- Modify: `tests/examples/test_example_scripts.py:23-40` (`SCRIPTS`), `examples/README.md` (the table)

**Interfaces:**
- Consumes: `PetabReader`, `to_petab`, `run_optimization`, `FitReport`, `fisher_information`, `hook_summaries`, `display`.
- Produces: `python -m examples.sciml.lotka_volterra_fit [--runs N] [--seed S] [--max-nfev N]`, which writes `results/lotka_volterra/` (the compiled model, the report and the exported problem) into the working directory.

- [ ] **Step 1: The files of the case**

The case is MIT licensed (`https://github.com/PEtab-dev/petab_sciml_testsuite/blob/main/LICENSE`). Copy its PEtab files from the cache of the suite:

```bash
SUITE=$(uv run --no-sync python -c "from sbmlsim.sciml.testsuite import SciMLSuite; print(SciMLSuite.load().path)")
mkdir -p examples/sciml/lotka_volterra
cp "$SUITE"/sciml_problem_import/001/petab/* examples/sciml/lotka_volterra/
ls examples/sciml/lotka_volterra
```

Expected: `experiments.tsv hybridization.tsv lv.xml mapping.tsv measurements.tsv net1.yaml net1_ps.hdf5 observables.tsv parameters.tsv problem.yaml` (`SciMLSuite.load()` downloads the suite when the cache is empty). Create `examples/sciml/__init__.py` empty, and `examples/sciml/lotka_volterra/README.md`:

```markdown
# Lotka-Volterra with a network in the right hand side

The case `sciml_problem_import/001` of the [PEtab SciML test suite](https://github.com/PEtab-dev/petab_sciml_testsuite) (commit `0622bbfc5e12eb9b482659eabd1756ca0e87dfc8`, MIT license), unchanged: the Lotka-Volterra model `lv.xml` whose interaction term of the predator, `gamma`, is the output of the feed forward network `net1.yaml` (three linear layers of five units with `tanh`) with the species as inputs, the arrays of the network in `net1_ps.hdf5`, and the tables of PEtab v2 with the `sciml` extension in `problem.yaml`.

`python -m examples.sciml.lotka_volterra_fit` reads it, fits it, reports it and writes it as PEtab SciML again.
```

- [ ] **Step 2: The example**

Create `examples/sciml/lotka_volterra_fit.py`:

```python
"""A PEtab SciML problem read, fitted, reported and written again.

    python -m examples.sciml.lotka_volterra_fit --runs=1

The problem is the case 001 of the PEtab SciML test suite,
`examples/sciml/lotka_volterra/`: the Lotka-Volterra model with a feed
forward network in the right hand side, which replaces the interaction term
of the predator. The example reads the problem, prints its networks, fits the
parameters of the model and the elements of the network together, reports the
fit, writes the problem as PEtab SciML again and reads it back: the
log-likelihoods of the two agree.

A problem which is read from PEtab builds its simulation experiment at
runtime, which the workers of a parallel fit cannot import, so the fit runs in
one process. The results are written into `results/lotka_volterra` of the
working directory, with the model which carries the network,
`lv_sciml.xml`.
"""

import argparse
import sys
from pathlib import Path

# run as a script (`python examples/sciml/lotka_volterra_fit.py`, the "run
# file" of an IDE) the repository is not on `sys.path`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sbmlsim.console import console
from sbmlsim.fit import display
from sbmlsim.fit.derived import hook_summaries
from sbmlsim.fit.fisher import fisher_information
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings, ParameterScaleType
from sbmlsim.fit.petab_v2 import gaps_of_problem, gaps_table, to_petab
from sbmlsim.fit.petab_v2.likelihood import log_likelihood
from sbmlsim.fit.petab_v2.reader import PetabReader
from sbmlsim.fit.report import FitReport
from sbmlsim.fit.runner import run_optimization

#: the problem of the case 001 of the test suite
PROBLEM_PATH = Path(__file__).parent / "lotka_volterra" / "problem.yaml"

#: the elements of a network are searched on the linear scale, and a
#: difference of the cost needs a fixed grid
FIT_SETTINGS = FitSettings(
    parameter_scale=ParameterScaleType.LINEAR,
    variable_step_size=False,
    absolute_tolerance=1e-10,
    relative_tolerance=1e-10,
)


def read(problem_path: Path, derived_dir: Path, opid: str) -> OptimizationProblem:
    """Read a problem, with the model which carries the network in `derived_dir`."""
    reader = PetabReader.from_yaml(problem_path)
    reader.derived_dir = derived_dir
    problem = reader.to_optimization_problem(opid=opid)
    problem.initialize(FIT_SETTINGS)
    return problem


def main() -> None:
    """Read, fit, report and write the problem."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--runs", type=int, default=1, help="optimization runs")
    parser.add_argument("--seed", type=int, default=1234, help="seed of the runs")
    parser.add_argument(
        "--max-nfev", type=int, default=50, help="evaluations of the cost per run"
    )
    options = parser.parse_args()
    output_dir = Path("results") / "lotka_volterra"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- READ ---
    problem = read(PROBLEM_PATH, output_dir, opid="lotka_volterra")
    display.section("Problem", icon=display.ICON_FIT)
    display.key_values(
        {
            "problem": PROBLEM_PATH,
            "fit mappings": len(problem.mapping_keys),
            "parameters": len(problem.parameters),
            "log-likelihood": f"{log_likelihood(problem):.6g}",
        }
    )
    display.print_parameters(
        problem.parameters, hooks=hook_summaries(problem.hybridizations)
    )
    console.print(gaps_table(gaps_of_problem(problem), title="What PEtab v2 loses"))

    # --- FIT ---
    # the step of the finite differences of the jacobian is absolute for the
    # elements, which are around zero; the default of `sbmlsim.fit.cli` (5%)
    # is made for parameters on a logarithmic scale
    opt_result = run_optimization(
        problem=problem,
        settings=FIT_SETTINGS,
        size=options.runs,
        seed=options.seed,
        serial=True,
        show_progress=False,
        diff_step=1e-4,
        x_scale="jac",
        max_nfev=options.max_nfev,
    )
    parameter_set = opt_result.parameter_set(0)

    # --- REPORT ---
    fisher = fisher_information(
        problem=problem, settings=FIT_SETTINGS, parameter_set=parameter_set
    )
    report = FitReport(
        problem=problem,
        settings=FIT_SETTINGS,
        parameter_sets=[parameter_set],
        opt_result=opt_result,
        fisher=fisher,
    )
    report.create(output_dir, name="report")

    # --- WRITE AND READ AGAIN ---
    display.section("PEtab SciML", icon=":package:")
    yaml_file = to_petab(problem, output_dir / "petab", settings=FIT_SETTINGS)
    restored = read(yaml_file, output_dir / "petab" / "derived", opid="restored")
    display.key_values(
        {
            "written": yaml_file,
            "log-likelihood": f"{log_likelihood(problem):.10g}",
            "read again": f"{log_likelihood(restored):.10g}",
        }
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it**

Run: `cd $(mktemp -d) && MPLBACKEND=Agg PYTHONPATH=/home/mkoenig/git/sbmlsim-sciml-phase3 uv run --no-sync --project /home/mkoenig/git/sbmlsim-sciml-phase3 python -m examples.sciml.lotka_volterra_fit 2>&1 | grep -v WARNING | tail -25`
Expected: the sections `Problem` (54 parameters, the table of the network `net1 rhs` with six arrays), `Optimization`, `Report` with the link and `PEtab SciML` with two equal log-likelihoods (`33.02907924`); about 7 s. `results/lotka_volterra/petab/` holds `problem.yaml`, `lv.xml`, `net1.yaml`, `net1_arrays.hdf5`, `hybridization.tsv` and `mapping.tsv`, and nothing was written into the repository (`git status --short` shows only the new example files).

- [ ] **Step 4: The test and the README**

In `tests/examples/test_example_scripts.py`, add `"examples.sciml.lotka_volterra_fit",` to `SCRIPTS` after `"examples.petab.benchmark",`. In `examples/README.md`, add the row `| `examples/sciml/` | hybrid problems of PEtab SciML: `lotka_volterra_fit.py` reads the case 001 of the test suite (`examples/sciml/lotka_volterra/`, a network in the right hand side), fits it in one process, reports it and writes it as PEtab SciML again; `neural_ode/` is a neural ODE defined in python and fitted in parallel |` after the `examples/petab/` row (the `neural_ode/` part is Task 9, write the row once).

Run: `uv run --no-sync pytest -q -x tests/examples/test_example_scripts.py -k lotka_volterra`
Expected: PASS

- [ ] **Step 5: Lint, type check, full suite, commit**

Run: `uv run --no-sync ruff check && uv run --no-sync ruff format --check && uvx ty check && uv run --no-sync pytest -q`
Expected: zero diagnostics, all tests pass, 0 warnings

```bash
git add examples/sciml tests/examples/test_example_scripts.py examples/README.md
git commit -m "examples: the Lotka-Volterra problem of the PEtab SciML test suite read, fitted, reported and written"
```

---

### Task 9: The example of a problem defined in python: a neural ODE fitted in parallel

**Files:**
- Create: `examples/sciml/neural_ode/__init__.py`, `examples/sciml/neural_ode/lotka_volterra_neural_ode.xml`, `examples/sciml/neural_ode/data.tsv`, `examples/sciml/neural_ode/network.py`, `examples/sciml/neural_ode/experiment.py`, `examples/sciml/neural_ode/fitting.py`
- Modify: `tests/examples/test_example_scripts.py` (`SCRIPTS`)

**Interfaces:**
- Consumes: `Hybridization.fit_parameters` of Task 5, `compile_network`, `input_id`, `output_id`, `run_fit`, `FitRun.report`, `AbstractModel(source, base_path=...)`.
- Produces: `python -m examples.sciml.neural_ode.fitting [--runs N] [--cores N] [--seed S] [--max-nfev N]`, which compiles the network into `results/neural_ode/` (the `base_path` of the problem, so the workers of the parallel fit load the same file) and writes the fit and its report into `results/neural_ode/fit/`.

- [ ] **Step 1: The model and the data**

Create `examples/sciml/neural_ode/__init__.py` empty. Create `examples/sciml/neural_ode/lotka_volterra_neural_ode.xml`, the model `create_neural_ode` of `petab_sciml` writes for the species `prey` and `predator` with the initial values of the Lotka-Volterra case, verbatim:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model>
    <listOfCompartments>
      <compartment size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="prey" initialAmount="0.44249296" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="predator" initialAmount="4.6280594" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="prey_param" value="0" constant="false"/>
      <parameter id="predator_param" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="prey">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <ci> prey_param </ci>
        </math>
      </rateRule>
      <rateRule variable="predator">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <ci> predator_param </ci>
        </math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>
```

Create `examples/sciml/neural_ode/data.tsv`, the first four seconds of the Lotka-Volterra system (`alpha=1.3, beta=0.9, gamma=0.8, delta=1.8`) with normal noise of `0.05` (seed 2026), verbatim:

```text
observable	time	value
prey	0.25	0.2176
prey	0.50	0.2106
prey	0.75	0.0917
prey	1.00	0.2694
prey	1.25	0.2649
prey	1.50	0.2732
prey	1.75	0.3533
prey	2.00	0.4994
prey	2.25	0.6324
prey	2.50	0.8588
prey	2.75	1.2159
prey	3.00	1.6321
prey	3.25	2.1880
prey	3.50	2.9850
prey	3.75	4.0763
prey	4.00	5.4634
predator	0.25	3.1335
predator	0.50	2.1297
predator	0.75	1.3859
predator	1.00	0.8537
predator	1.25	0.5901
predator	1.50	0.4451
predator	1.75	0.2690
predator	2.00	0.1873
predator	2.25	0.1710
predator	2.50	0.1942
predator	2.75	0.0448
predator	3.00	0.1350
predator	3.25	0.0013
predator	3.50	0.0757
predator	3.75	0.0276
predator	4.00	0.2094
```

(the columns are separated by tabs.)

- [ ] **Step 2: The network**

Create `examples/sciml/neural_ode/network.py`:

```python
"""The network of the neural ODE, defined without PyTorch.

The architecture is the one of the neural ODE how-to of PEtab SciML with
smaller layers: two hidden layers of `HIDDEN` units with `tanh`, from the two
species to their two rates. The arrays are drawn once from a seed; the small
weights of the last layer start the fit from a slow model.
"""

import numpy as np
from petab_sciml import Input, Layer, NNModel, Node

from sbmlsim.sciml import Network

#: id of the network, the prefix of the ids of its elements
NETWORK_ID = "net1"

#: units of the hidden layers, 57 elements in all
HIDDEN = 5


def _linear(layer_id: str, n_in: int, n_out: int) -> Layer:
    return Layer(
        layer_id=layer_id,
        layer_type="Linear",
        args={"in_features": n_in, "out_features": n_out, "bias": True},
    )


def _node(name: str, op: str, target: str, args: list) -> Node:
    return Node(name=name, op=op, target=target, args=args, kwargs={})


def build_network(seed: int = 1, hidden: int = HIDDEN) -> Network:
    """Build `layer3(tanh(layer2(tanh(layer1(x)))))` with random arrays.

    Args:
        seed: seed of the arrays.
        hidden: units of the two hidden layers.

    Returns:
        The network, with its nominal values.
    """
    rng = np.random.default_rng(seed)
    model = NNModel(
        nn_model_id=NETWORK_ID,
        inputs=[Input(input_id="input0")],
        layers=[
            _linear("layer1", 2, hidden),
            _linear("layer2", hidden, hidden),
            _linear("layer3", hidden, 2),
        ],
        forward=[
            _node("net_input", "placeholder", "net_input", []),
            _node("layer1", "call_module", "layer1", ["net_input"]),
            _node("tanh", "call_function", "tanh", ["layer1"]),
            _node("layer2", "call_module", "layer2", ["tanh"]),
            _node("tanh_1", "call_function", "tanh", ["layer2"]),
            _node("layer3", "call_module", "layer3", ["tanh_1"]),
            _node("output", "output", "output", ["layer3"]),
        ],
    )
    return Network(
        sid=NETWORK_ID,
        model=model,
        parameters={
            "layer1": {
                "weight": 0.5 * rng.normal(size=(hidden, 2)),
                "bias": 0.1 * rng.normal(size=hidden),
            },
            "layer2": {
                "weight": 0.5 * rng.normal(size=(hidden, hidden)),
                "bias": 0.1 * rng.normal(size=hidden),
            },
            "layer3": {
                "weight": 0.1 * rng.normal(size=(2, hidden)),
                "bias": 0.1 * rng.normal(size=2),
            },
        },
    )
```

- [ ] **Step 3: The experiment**

Create `examples/sciml/neural_ode/experiment.py`:

```python
"""The simulation experiment of the neural ODE: the model against the data.

The model `lotka_volterra_neural_ode.xml` has the species `prey` and
`predator` with the rate rules `d prey/dt = prey_param` and
`d predator/dt = predator_param`, which the network sets; it is the model
`create_neural_ode` of `petab_sciml` writes. The data are the first four
seconds of the Lotka-Volterra system with noise, `data.tsv`.

The experiment simulates the model with the network, which the fit compiles
into its `base_path` as `COMPILED_MODEL`: the workers of a parallel fit load
the model from that file.
"""

from pathlib import Path

import pandas as pd

from sbmlsim.data import DataSet
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import FitData, FitMapping
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import AbstractSim, Timecourse, TimecourseSim
from sbmlsim.task import Task

#: the directory of the example, with the model and the data
EXAMPLE_PATH = Path(__file__).parent

#: the model without the network
MODEL_PATH = EXAMPLE_PATH / "lotka_volterra_neural_ode.xml"

#: the model with the network, relative to the `base_path` of the problem
COMPILED_MODEL = "lotka_volterra_neural_ode_sciml.xml"

#: the species, which are observed and whose rates the network gives
SPECIES = ("prey", "predator")


class NeuralODE(SimulationExperiment):
    """The neural ODE simulated over the data of both species."""

    def models(self) -> dict[str, AbstractModel | Path]:
        return {
            "lv": AbstractModel(
                source=COMPILED_MODEL,
                base_path=self.base_path,
                language_type=AbstractModel.LanguageType.SBML,
            )
        }

    def datasets(self) -> dict[str, DataSet]:
        df = pd.read_csv(EXAMPLE_PATH / "data.tsv", sep="\t")
        return {
            str(species): DataSet.from_df(
                pd.DataFrame(
                    {
                        "time": rows["time"].to_numpy(),
                        "time_unit": "second",
                        "value": rows["value"].to_numpy(),
                        "value_unit": "dimensionless",
                    }
                ),
                ureg=self.ureg,
            )
            for species, rows in df.groupby("observable")
        }

    def simulations(self) -> dict[str, AbstractSim]:
        return {"sim": TimecourseSim([Timecourse(start=0.0, end=4.0, steps=100)])}

    def tasks(self) -> dict[str, Task]:
        return {"task_sim": Task(model="lv", simulation="sim")}

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            f"{species}_data": FitMapping(
                self,
                reference=FitData(self, dataset=species, xid="time", yid="value"),
                observable=FitData(self, task="task_sim", xid="time", yid=species),
            )
            for species in SPECIES
        }
```

- [ ] **Step 4: The fit**

Create `examples/sciml/neural_ode/fitting.py`:

```python
"""Fit of a neural ODE which is defined in python, in parallel.

    python -m examples.sciml.neural_ode.fitting --runs=2 --cores=2

The network sits in the right hand side: `compile_network` writes the model
with the network into `results/neural_ode` of the working directory, which is
the `base_path` of the problem, so the workers of the fit load the same file.
The elements are bounded, so every run starts from its own random values in
the bounds; an element without bounds starts from the value of the network in
every run. The fit uses a finite difference jacobian with a small step, which
the default step of `sbmlsim.fit.cli` (5%, made for parameters on a
logarithmic scale) is not. The results and the report are written into
`results/neural_ode/fit`.
"""

import argparse
import sys
from pathlib import Path

# run as a script (`python examples/sciml/neural_ode/fitting.py`, the "run
# file" of an IDE) the repository is not on `sys.path`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.sciml.neural_ode.experiment import (
    COMPILED_MODEL,
    EXAMPLE_PATH,
    MODEL_PATH,
    NeuralODE,
)
from examples.sciml.neural_ode.network import NETWORK_ID, build_network
from sbmlsim.fit import FitMappingCollection, FitSettings
from sbmlsim.fit.cli import FitDefinition, run_fit
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.sciml import (
    Hybridization,
    NetworkInput,
    NetworkPattern,
    compile_network,
    input_id,
    output_id,
)

#: where the fit writes, the model with the network included
RESULTS_PATH = Path("results") / "neural_ode"

NETWORK = build_network()

#: the network in the right hand side: the species are its inputs, the rates
#: its outputs. Every element is estimated within the bounds
HYBRIDIZATION = Hybridization(
    network=NETWORK,
    pattern=NetworkPattern.RHS,
    model="lv",
    inputs={
        input_id(NETWORK_ID, 0, (0,)): NetworkInput(formula="prey"),
        input_id(NETWORK_ID, 0, (1,)): NetworkInput(formula="predator"),
    },
    outputs={
        output_id(NETWORK_ID, 0, (0,)): "prey_param",
        output_id(NETWORK_ID, 0, (1,)): "predator_param",
    },
)
PARAMETERS, HYBRIDIZATION = HYBRIDIZATION.fit_parameters(
    estimate={NETWORK_ID: True}, bounds={NETWORK_ID: (-3.0, 3.0)}
)

#: the elements are searched on the linear scale, and a difference of the
#: cost needs a fixed grid
FIT_SETTINGS = FitSettings(
    parameter_scale=ParameterScaleType.LINEAR,
    variable_step_size=False,
    absolute_tolerance=1e-10,
    relative_tolerance=1e-10,
)


def collections() -> dict[str, list[FitMappingCollection]]:
    """Get the fit mappings, both species of the one experiment."""
    return {
        "neural_ode": [FitMappingCollection(experiment=NeuralODE, sid="neural_ode")]
    }


FIT_DEFINITIONS: dict[str, FitDefinition] = {
    "NODE": FitDefinition(
        mapping_collections=collections,
        parameters=PARAMETERS,
        base_path=RESULTS_PATH,
        data_path=EXAMPLE_PATH,
        settings=FIT_SETTINGS,
        hybridizations=[HYBRIDIZATION],
    )
}


def compile_model() -> Path:
    """Write the model with the network into the results directory."""
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    return compile_network(MODEL_PATH, [HYBRIDIZATION], RESULTS_PATH / COMPILED_MODEL)


def main() -> None:
    """Compile the model, fit the network and report the fit."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--runs", type=int, default=2, help="optimization runs")
    parser.add_argument("--cores", type=int, default=2, help="workers of the fit")
    parser.add_argument("--seed", type=int, default=1234, help="seed of the runs")
    parser.add_argument(
        "--max-nfev", type=int, default=100, help="evaluations of the cost per run"
    )
    options = parser.parse_args()

    compile_model()
    runs = run_fit(
        FIT_DEFINITIONS["NODE"],
        opid="neural_ode",
        size=options.runs,
        n_cores=options.cores,
        seed=options.seed,
        output_dir=RESULTS_PATH / "fit",
        diff_step=1e-4,
        x_scale="jac",
        max_nfev=options.max_nfev,
    )
    for run in runs.values():
        run.report(output_dir=RESULTS_PATH / "fit", show_titles=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run it**

Run: `cd $(mktemp -d) && MPLBACKEND=Agg PYTHONPATH=/home/mkoenig/git/sbmlsim-sciml-phase3 uv run --no-sync --project /home/mkoenig/git/sbmlsim-sciml-phase3 python -m examples.sciml.neural_ode.fitting 2>&1 | grep -v WARNING | tail -25`
Expected: the parameters section with the network `net1 rhs` and six arrays (57 elements, bounds -3 and 3), two runs on two workers, a best cost of the order of `1e-2` (the noise of the data; measured: `0.015` and `0.0005` to `0.015` for three seeds) in about 30 s, the report link, and the bound warnings per array (`!n of the m elements of 'net1.layer2.weight' within 5% of a bound!`) rather than per element. The elements start from random values in the bounds, so the two runs differ; `git status --short` shows nothing but the new files.

- [ ] **Step 6: The test**

In `tests/examples/test_example_scripts.py`, add `"examples.sciml.neural_ode.fitting",` after `"examples.sciml.lotka_volterra_fit",`.

Run: `uv run --no-sync pytest -q -x tests/examples/test_example_scripts.py -k sciml`
Expected: PASS (about 40 s for the two)

- [ ] **Step 7: Lint, type check, full suite, commit**

Run: `uv run --no-sync ruff check && uv run --no-sync ruff format --check && uvx ty check && uv run --no-sync pytest -q`
Expected: zero diagnostics, all tests pass, 0 warnings

```bash
git add examples/sciml/neural_ode tests/examples/test_example_scripts.py
git commit -m "examples: a neural ODE defined in python and fitted in parallel"
```

---

### Task 10: The documentation

**Files:**
- Modify: `docs/petab.md:137-163` (the section "Hybrid problems of PEtab SciML")
- Modify: `docs/fitting.md:283-330` (the section "Reporting the fit", one paragraph)
- Modify: `docs/references.md:133-138` (after the PEtab reference)
- Create: `docs/api/sciml.md`, `docs/api/sciml.network.md`, `docs/api/sciml.hybridization.md`, `docs/api/sciml.compiler.md`, `docs/api/sciml.parameters.md`, `docs/api/sciml.interpreter.md`, `docs/api/sciml.backend.md`, `docs/api/sciml.layers.md`, `docs/api/sciml.errors.md`, `docs/api/sciml.testsuite.md`, `docs/api/testsuite.cache.md`
- Modify: `docs/api/index.md` (a section `sbmlsim.sciml`, the table `sbmlsim.testsuite`), `zensical.toml` (nav)

**Interfaces:**
- Consumes: everything of the tasks 1 to 9 as it is on the branch; `sbmlsim.sciml.layers.LAYERS` and `FUNCTIONS` (the backends of every layer, `BackendKind.SYMPY` in `spec.backends` means "numpy, sympy").
- Produces: the documentation, which builds with `No issues found`.

- [ ] **Step 1: The SciML section of `docs/petab.md`**

In `docs/petab.md`, replace the paragraph `The exporter, the report and the examples of hybrid problems are not part of this release.` at the end of the section "Hybrid problems of PEtab SciML" by the following subsections (every paragraph one line, the tables as given):

```markdown
### Writing a hybrid problem

`to_petab` writes a problem with hybridizations as PEtab SciML: the model the problem was defined with (a model which carries compiled networks or formula observables records what was added to it, `sbmlsim.model.provenance`, and is written as its source), the `sciml` block of the YAML, the hybridization table, the rows of the mapping table which name the inputs, the outputs and the arrays of every network, the rows of the parameter table which say which arrays are estimated with which bounds (one row for the network when its arrays agree, rows for the layers or the arrays which differ), the NN YAML `<net>.yaml` and the array file `<net>_arrays.hdf5` with the arrays of the network and of its inputs, keyed by the condition of the experiment. The values of the network are its nominal values, so the exported problem carries them in the array file and no numeric rows; the scale of a parameter goes to the `sbmlsim` block. The round trip is exact: every case of the test suite which is read (`tox r -e sciml` runs `test_problem_round_trip`) and every python defined problem of `tests/sciml/test_export.py` reads back to the same parameters, hybridizations, data and log-likelihood. What cannot be written is refused with the name of the array: a fit which estimates some elements of an array, or bounds them differently, is the gap `sciml-partial-array`, because a row of PEtab SciML describes an array as a whole; a hook which is no network is refused as well.

`examples/sciml/lotka_volterra_fit.py` reads the case 001 of the test suite, fits it in one process (a problem which is read builds its experiment at runtime, which the workers of a parallel fit cannot import), reports it and writes it again; `examples/sciml/neural_ode/` defines a neural ODE in python, compiles the network into the `base_path` of the problem, so that the workers of a parallel fit load the same file, and fits it on two cores (`python -m examples.sciml.neural_ode.fitting --runs=2 --cores=2`). Both write into `results/` of the working directory.

### The report of a hybrid fit

The console and the report do not list the elements of a network one by one: the overview names every network with its pattern, its layers and its targets, and shows one row per array with the number of its elements, the number of them the fit estimates, the bounds when the elements agree on them and the minimum, the maximum and the norm of the values of every parameter set; the parameter table keeps the parameters of the model. The TSV and the JSON of the parameter sets keep every element. A warning about a parameter at a bound counts the elements of an array (`3 of the 25 elements of 'net1.layer2.weight' within 5% of a bound`), the table of the Fisher information shows an array as one row with the range of the standard errors of its elements and the correlation matrix leaves the elements out, and `identifiability_cli` profiles the parameters which are no elements of a network unless `--parameter` names an element: a profile is one optimization scan per parameter, which a network of hundreds of elements makes impractical, while the Fisher information is one jacobian and covers them.

### The size of a network

The optimizer stays a least squares fit whose jacobian is built by finite differences: an iteration costs one simulation per parameter, so a network with `n` elements costs `n + 1` simulations per iteration. Measured on the neural ODE of the example with a fixed grid and the tolerances `1e-10`: a network of 57 elements simulates in 2 ms, an iteration takes 0.15 s and a fit of the first four seconds of the Lotka-Volterra system from a random initialization converges in 80 evaluations, i.e. 15 s; a network of 162 elements simulates in 4 ms, an iteration takes 0.7 s, and the same fit over three oscillations does not converge from a random initialization within 150 iterations (2 minutes, the cost falls from 91 to 70), which is the well known difficulty of a neural ODE over a long window and not a matter of the jacobian. A network of a few hundred elements is the size a fit with the finite difference jacobian is made for; a network of thousands of elements (phase 3 measured 32 ms per simulation for 2751 elements in the right hand side, i.e. 90 s per iteration) needs the gradients by sensitivities or automatic differentiation and the optimizers of the machine learning frameworks, which are out of scope. The step of the finite differences matters: the default of `sbmlsim.fit.cli` (`diff_step=0.05`, relative, made for parameters on a logarithmic scale) is too coarse for elements around zero, the examples pass `diff_step=1e-4` and `x_scale="jac"` to `run_optimization`. An element without bounds starts from the value of the network in every run of a multi start; give the elements bounds when the runs should start from different values.

### The layers

The layers and the functions of PEtab SciML (`https://petab-sciml.readthedocs.io/latest/layers.html`) with the backends of `sbmlsim`: a layer of the `numpy` backend runs before the simulation, a layer of both backends is also compiled into the model, i.e. can sit in the right hand side or in an observable. `gelu` is evaluated on expressions but not compiled, the error function has no MathML (gap `sciml-layer-sbml`). Dropout is the identity and the normalization layers use their stored statistics, i.e. every network is evaluated in evaluation mode (gap `sciml-training-mode`).

| layer | PEtab.jl | AMICI | sbmlsim |
| --- | --- | --- | --- |
| `Linear` | yes | yes | numpy, sympy |
| `Bilinear` | yes | - | numpy, sympy |
| `Flatten` | yes | yes | numpy, sympy |
| `Dropout`, `Dropout1d`, `Dropout2d`, `Dropout3d`, `AlphaDropout`, `FeatureAlphaDropout` | yes | - | numpy, sympy (the identity) |
| `Conv1d`, `Conv2d`, `Conv3d` | yes | yes | numpy |
| `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d` | yes | yes | numpy |
| `MaxPool1d`, `MaxPool2d`, `MaxPool3d` | yes | yes | numpy |
| `AvgPool1d`, `AvgPool2d`, `AvgPool3d` | yes | yes | numpy |
| `LPPool1d`, `LPPool2d`, `LPPool3d` | yes | yes | numpy |
| `AdaptiveMaxPool1d`, `AdaptiveMaxPool2d`, `AdaptiveMaxPool3d` | yes | yes | numpy |
| `AdaptiveAvgPool1d`, `AdaptiveAvgPool2d`, `AdaptiveAvgPool3d` | yes | yes | numpy |
| `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d` | yes | - | numpy (stored statistics) |
| `InstanceNorm1d`, `InstanceNorm2d`, `InstanceNorm3d` | yes | - | numpy (stored statistics) |
| `LayerNorm` | yes | - | numpy |

| function | PEtab.jl | AMICI | sbmlsim |
| --- | --- | --- | --- |
| `relu`, `relu6`, `hardtanh`, `hardswish`, `hardsigmoid`, `leaky_relu` | yes | yes | numpy, sympy |
| `selu`, `elu`, `celu`, `softplus`, `softsign`, `tanhshrink`, `mish`, `silu` | yes | yes | numpy, sympy |
| `tanh`, `sigmoid`, `log_sigmoid` | yes | yes | numpy, sympy |
| `gelu` | yes | yes | numpy, sympy (not compiled: `erf` has no MathML) |
| `softmax`, `log_softmax` | yes | yes | numpy, sympy (see the size of `log_softmax` above) |
| `flatten`, `cat` (`concat`, `concatenate`) | yes | yes | numpy, sympy |
```

Check the columns `PEtab.jl` and `AMICI` against the page when the task runs (the page is the authority for those two columns; the rows of the normalization layers are supported by PEtab.jl according to its documentation, put `-` where the page says nothing) and the column `sbmlsim` against `LAYERS` and `FUNCTIONS` of `sbmlsim.sciml.layers`: `uv run --no-sync python -c "from sbmlsim.sciml.layers import LAYERS, FUNCTIONS; from sbmlsim.sciml.backend import BackendKind; print({n: BackendKind.SYMPY in s.backends for n, s in {**LAYERS, **FUNCTIONS}.items()})"`.

In the list of the gaps of the same section, add `- `sciml-partial-array`: a fit which estimates some elements of an array or bounds them differently, which a row of the parameter table of PEtab SciML cannot say; the export raises and names the array` (the `priors` line was added in Task 4). Replace the sentence of the section which says that the reader reads `scale` of the block by one which also says that the exporter writes it: "The scale of a parameter (`FitParameter.scale`) is written as `scale` into the `sbmlsim` block and read from it, and the problems of PEtab SciML carry the column `parameterScale` of PEtab v1, which becomes the scale of the parameter when the block does not give one."

- [ ] **Step 2: `docs/fitting.md` and `docs/references.md`**

In `docs/fitting.md`, section "Reporting the fit", add the paragraph: `A problem with neural networks (see [PEtab](petab.md#hybrid-problems-of-petab-sciml)) shows every network in the overview of the report with its pattern, its layers and its targets, and its arrays as one row each with the number of elements, the estimated ones, their bounds and the range and the norm of their values; the parameters of the model keep their rows and the TSV of the parameter sets keeps every element.`

In `docs/references.md`, after the PEtab reference block, add:

```markdown
**PEtab SciML.** The extension of PEtab for hybrid problems of a mechanistic model and neural networks, which `sbmlsim.fit.petab_v2` reads and writes and `sbmlsim.sciml` runs, see [PEtab](petab.md#hybrid-problems-of-petab-sciml).

> Persson S, Snelling B, Philipps M, Weindl D, Cvijovic M, Hasenauer J, Pathirana D, Fröhlich F.
> **PEtab SciML: an exchange format for specifying and training dynamic scientific machine learning models.**
> *arXiv.* 2026;2608.20184.
> [arXiv:2608.20184](https://arxiv.org/abs/2608.20184)
```

- [ ] **Step 3: The API pages**

Create one page per module, each of the form:

```markdown
# sciml.network

::: sbmlsim.sciml.network
```

for `sciml` (`::: sbmlsim.sciml`, the title `# sciml`), `sciml.network`, `sciml.hybridization`, `sciml.compiler`, `sciml.parameters`, `sciml.interpreter`, `sciml.backend`, `sciml.layers` (`::: sbmlsim.sciml.layers`), `sciml.errors`, `sciml.testsuite` and `testsuite.cache` (`::: sbmlsim.testsuite.cache`).

In `docs/api/index.md`, add before `## sbmlsim.testsuite`:

```markdown
## sbmlsim.sciml

The neural networks of hybrid problems, see [PEtab](../petab.md#hybrid-problems-of-petab-sciml). The package needs the extra `sciml`.

| module | description |
| --- | --- |
| [sciml](sciml.md) | the package: `Network`, `Hybridization`, `NetworkInput`, `NetworkPattern`, `compile_network`, `network_fit_parameters`, the id functions and the errors |
| [sciml.network](sciml.network.md) | `Network`, the architecture and the arrays of a network, its forward pass and the ids of its elements, inputs and outputs |
| [sciml.hybridization](sciml.hybridization.md) | `Hybridization`, where a network sits, its inputs and outputs, the derived changes of a network before the simulation |
| [sciml.compiler](sciml.compiler.md) | `compile_network`, a network in the right hand side or in an observable written into the model as assignment rules |
| [sciml.parameters](sciml.parameters.md) | the nominal values and the fit parameters of a network per network, layer or array |
| [sciml.interpreter](sciml.interpreter.md) | the walk over the forward pass of the NN YAML with a backend |
| [sciml.backend](sciml.backend.md) | the numpy backend of the forward pass and the sympy backend of the compiler |
| [sciml.layers](sciml.layers.md) | the layers and functions of PEtab SciML with the backends they support |
| [sciml.errors](sciml.errors.md) | the errors of the package |
| [sciml.testsuite](sciml.testsuite.md) | the PEtab SciML test suite: its cases, their comparison and the round trip |
```

and the row `| [testsuite.cache](testsuite.cache.md) | the download and the cache of a test suite, shared by the SBML Test Suite and the PEtab SciML test suite |` to the table `sbmlsim.testsuite`. In `zensical.toml`, add after the `sbmlsim.sensitivity` block:

```toml
    { "sbmlsim.sciml" = [
      { "sciml" = "api/sciml.md" },
      { "network" = "api/sciml.network.md" },
      { "hybridization" = "api/sciml.hybridization.md" },
      { "compiler" = "api/sciml.compiler.md" },
      { "parameters" = "api/sciml.parameters.md" },
      { "interpreter" = "api/sciml.interpreter.md" },
      { "backend" = "api/sciml.backend.md" },
      { "layers" = "api/sciml.layers.md" },
      { "errors" = "api/sciml.errors.md" },
      { "testsuite" = "api/sciml.testsuite.md" },
    ] },
```

and `{ "cache" = "api/testsuite.cache.md" },` to the `sbmlsim.testsuite` block.

- [ ] **Step 4: Build**

Run: `uv run --no-sync zensical build --clean 2>&1 | tail -5 && uv run --no-sync python scripts/llms_txt.py 2>&1 | tail -2`
Expected: `No issues found`; a page which mkdocstrings cannot render (a docstring with a section it does not know, a cross reference it cannot resolve) is a warning of the build and has to be fixed in the docstring, not silenced.

- [ ] **Step 5: Lint, full suite, commit**

Run: `uv run --no-sync ruff check && uv run --no-sync ruff format --check && uvx ty check && uv run --no-sync pytest -q`
Expected: zero diagnostics, all tests pass, 0 warnings

```bash
git add docs zensical.toml
git commit -m "docs: the export, the report and the examples of hybrid problems, the layers of PEtab SciML, the API of sbmlsim.sciml"
```

---

## Passage for CLAUDE.md

Not applied by any task; the owner adds it to `CLAUDE.md` with the release. It replaces the draft of the fix wave report of phase 3 (`.superpowers/sdd/2026-09-30-petab-sciml-phase3/fix-wave-report.md`, "Passage for CLAUDE.md (not applied)"), whose paragraphs on `sciml/` and on the additions to `fit/` and elsewhere stay as they are, with these changes and additions.

Append to the paragraph on `sciml/`: `Hybridization.fit_parameters(estimate, bounds)` gives the fit parameters of a network with `external` set by the pattern and the hybridization with the other elements frozen, which is what a python defined fit uses; `Hybridization.summary()` implements `DerivedChanges.summary()` (`fit/derived.py::HookSummary`, `ParameterGroup`): the name, the pattern, the layers, the targets and one group per array of the used layers, which the console (`display.print_parameters(..., hooks=hook_summaries(problem.hybridizations))`, one table of the networks and one row per array with the count, the estimated elements, min, max and norm) and the report (the same in the overview, `bound_warnings(..., groups=...)` counting the elements of an array, the Fisher table with one row per array and the correlation matrix without the elements, `identifiability_cli` profiling the parameters which are no elements unless `--parameter` names one) show instead of hundreds of element rows. `sciml/testsuite.py::ProblemImportCase.round_trip(directory)` writes a case as PEtab SciML and reads it back; the differences it answers with are empty for every read case (the comparison is against a second read of the original, because the first model roadrunner loads in a process differs by `1e-9` from every later one), `tests/sciml/test_testsuite.py::test_problem_round_trip` pins it under the `sciml_testsuite` marker.

Append to the paragraph on `fit/` (the PEtab layer): the exporter writes the definition of a problem and not the state an evaluation left in it, `OptimizationProblem.defined_changes` keeps the changes of the first timecourses as defined; a model the fit simulates which is derived from the model of the problem (compiled networks, formula observables) records what was added to it in an annotation (`model/provenance.py`: `record_derivation`, `derivation_of`, `strip_derivation`, written by `compile_network` and `add_observables`), and the exporter strips it and writes the source model with the formula of a derived observable; an observable is named after its fit mapping (with the experiment only where two experiments share a key), fit mappings with the same formula and noise in several experiments are one observable again (the inverse of the reader's `<observable>_<experiment>` split), and an observable which would shadow an entity of the model is prefixed with `observable__`; the `sbmlsim` extension is `0.2.0`, its `observables` block is keyed by the fit mapping and carries `observable`, and `parameters` carry `scale`; `PetabReader.observable_info(key)` is the lookup; `DEFAULT_EXPERIMENT` is `default_experiment` (a PEtab model named `model` collided with it). `fit/petab_v2/sciml_export.py::SciMLExporter` is the inverse of `SciMLReader`, created by `PetabExporter` for a problem with hybridizations (imported lazily, the extra `sciml`): it writes the `sciml` block, `hybridization.tsv`, the mapping rows (`<net>__input<k>__<index>` to `net.inputs[k][i]`, an output to `net.outputs[k][i]` without the leading axes of length one, the target of an `OBSERVABLE` output is its `petabEntityId`), the parameter rows (`<net>__parameters`, `<net>__<layer>__parameters`, `<net>__<layer>__<array>__parameters` with the most common description as the row of the network and rows for what differs, `nominalValue` `array`, the constants of the hybridizations as rows which are not estimated), `<net>.yaml` and `<net>_arrays.hdf5` (the arrays of the network and the arrays of its inputs keyed by the condition `<experiment>__tc0` of the first period, which the exporter creates for such an experiment), and refuses a partial array (gap `sciml-partial-array`), an element whose start value, scale or unit differs from the network, and a hook which is no `Hybridization`. The gap `priors` says the reader drops the prior of a parameter of the model with a warning (issue #190). `log.some_ids(ids, n)` lists the first ids of a message and the count.

Examples: `examples/sciml/lotka_volterra_fit.py` (the case 001 of the suite in `examples/sciml/lotka_volterra/`, read, fitted serially, reported, written and read again) and `examples/sciml/neural_ode/` (a neural ODE 2-5-5-2 defined in python with `Hybridization.fit_parameters`, compiled into the `base_path` of the problem so the workers load one file, fitted in parallel with `run_fit(..., diff_step=1e-4, x_scale="jac")` because the default step of 5% is made for logarithmic parameters); `tests/examples` runs both; the finite difference jacobian is usable for a few hundred elements (`docs/petab.md`, "The size of a network"). The sensitivity example runs all five analyses.
