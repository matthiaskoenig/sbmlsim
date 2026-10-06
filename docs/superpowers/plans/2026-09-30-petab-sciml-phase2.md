# PEtab SciML phase 2: log-likelihood and noise model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calculate the log-likelihood PEtab v2 defines, and its gradient, for an `OptimizationProblem`, keep the noise model of every observable through a round trip of a PEtab problem, and reject a problem which requires the extension of another tool.

**Architecture:** The noise formula and the noise distribution of an observable are a `NoiseModel` (`fit/objects.py`), which a `FitMapping` carries and `OptimizationProblem.initialize` collects as `problem.noise_models`; the reader fills it and the exporter writes it back. `fit/petab_v2/likelihood.py` has two layers: functions of arrays (`log_density`, `noise_values`, `default_noise_model`), which know nothing of a problem and are tested against values calculated by hand, and functions of a problem (`log_likelihood`, `gradient`), which simulate through the new `OptimizationProblem.predictions`. The optimizer is not touched, it stays a weighted least squares fit.

**Tech Stack:** python 3.13+, numpy, pandas, sympy, petab 0.9.0 (`petab.v2`, `petab.v2.math.sympify_petab`), libroadrunner, pytest with pytest-xdist, ruff, ty, zensical.

**Spec:** `docs/superpowers/specs/2026-09-30-petab-sciml-design.md`, phase 2 of the table "Phases", the sections "`likelihood.py`" and "`reader.py`" (first bullet). Phase 2 is independent of the neural network code of the phases 1, 3 and 4: nothing here imports `sbmlsim.sciml` or `petab_sciml`.

## Global Constraints

- python >= 3.13.
- Every module, class and function in `src/` carries full type annotations and a google style docstring (ruff `D`). `tests/` and `examples/` are exempt from the docstring rules, the tests of this plan have docstrings anyway.
- `uv run ruff check`, `uv run ruff format --check` and `uvx ty check` stay at zero diagnostics. Suppress a type diagnostic only with a rule specific `# ty: ignore[rule]`, never with a blanket `# type: ignore`. `ty` runs with `error-on-warning = true`.
- `ruff format` also formats the python blocks of markdown files. At the start of this plan `uv run ruff format --check` reports exactly one file, the spec `docs/superpowers/specs/2026-09-30-petab-sciml-design.md`; Task 1 formats it in its own commit, so that the check is at zero from then on.
- Library code logs with `logging.getLogger(__name__)` and lazy `%s` formatting (ruff `G`), it never prints.
- Never use the em dash character in any file, use "-".
- No attribution of agents in commits or files: no `Co-Authored-By`, no "Generated with" line.
- Never edit `CHANGELOG.md`.
- Markdown carries no hard line wraps: a paragraph, a list item or a table row is a single line.
- Tests run with `uv run pytest -q -x <path>`. Tests of a report use `mapping_figures=False` (this plan creates no report).
- Commits are made on the current branch and are never pushed.
- The log density is the one of PEtab v2, as `petab.v2.calculate.calculate_single_llh` of `petab` 0.9.0 implements it: the simulation is the median, the noise formula is the scale, and the logarithmic distributions are densities of the measurement `m`, i.e. they carry the `1/m` term. The reference value `llh = 33.02909543616689` of the case `sciml_problem_import/001` of the PEtab SciML test suite (`tol_llh = 1e-3`) pins the sign and the constant.
- The numbers of the tests are exact where the functions are functions of arrays. Two simulations of one problem agree to the tolerances of the integrator only with `variable_step_size=False`: with a variable step size the data is interpolated on the steps of the integrator and two runs differ by up to `3e-6` however tight the tolerances are. Every test which compares simulations uses `variable_step_size=False`, `absolute_tolerance=1e-12` and `relative_tolerance=1e-10`.
- How to apply a **Patch** block: it is a unified diff against the file as it is when the task starts. Apply it hunk by hunk with the Edit tool (the `-` lines and the context are the old text, the `+` lines and the context the new one), or save the block as a file and run `git apply <file>`. Do not change anything the diff does not change. A **Create** block is the complete content of a new file.

## Review Focus

- A measurement or a simulation which is zero or negative under `log-normal` or `log-laplace` (the amounts in the urine of the HCTZ problem start at `0.0`): a `ValueError` which names the fit mapping, not a `nan` or `-inf`. Tests: `test_log_density_requires_positive_values_for_a_log_distribution` (Task 1), `test_log_likelihood_of_a_log_distribution_requires_positive_data` (Task 3).
- `gradient` on a problem which was initialized with the default `variable_step_size=True`: the differences of its simulations are of the size of the step, so the result is noise. A reasonable person expects to be told: the gradient logs a warning which names the setting. Test: `test_gradient_warns_about_a_variable_step_size` (Task 3).
- A noise formula over something which is neither a placeholder nor a parameter, e.g. a species of the model or the time: the problem is read and fitted as before, the reader logs a warning, and `log_likelihood` raises a `ValueError` which names the symbol and the fit mapping. Test: `test_reader_reports_a_symbol_without_a_value` (Task 4), `test_noise_values_require_every_symbol` (Task 1).
- A parameter set, or a step of the gradient, which the model cannot simulate (a parameter at a bound of `0.0`, where `x - h` is negative): a `ValueError` which names the fit mapping and the parameters, not a `nan`. Tests: `test_predictions_of_a_failed_simulation_raise` (Task 2), `test_gradient_of_a_failed_simulation_raises` (Task 3).
- A problem of PEtab SciML (`sciml`, `required: true`) read before phase 3 exists: the error says that the problem requires an extension `sbmlsim` does not know, it is not the `ModuleNotFoundError: No module named 'petab_sciml'` which `petab` raises when it reads the files of the extension. Test: `test_the_sciml_extension_is_reported_and_not_a_missing_module` (Task 4).

## File Structure

| file | responsibility | task |
| --- | --- | --- |
| `src/sbmlsim/fit/objects.py` | `NoiseDistribution`, `NoiseParameter`, `NoiseModel`; `FitMapping(noise=...)` | 1, 2 |
| `src/sbmlsim/fit/__init__.py` | exports the three noise classes | 1 |
| `src/sbmlsim/fit/petab_v2/likelihood.py` | new: `log_density`, `noise_values`, `default_noise_model`, `noise_model_of`, `nominal_parameters`, `log_likelihood`, `gradient` | 1, 3 |
| `src/sbmlsim/fit/optimization.py` | `problem.noise_models`, `OptimizationProblem.predictions`, `_interpolate` | 2 |
| `src/sbmlsim/fit/petab_v2/__init__.py` | exports `log_likelihood`, `gradient` | 3 |
| `src/sbmlsim/fit/petab_v2/extension.py` | `KNOWN_EXTENSIONS`, `check_extensions` | 4 |
| `src/sbmlsim/fit/petab_v2/reader.py` | `PetabReader.noise_model`, the check of the extensions | 4 |
| `src/sbmlsim/fit/petab_v2/export.py` | writes the noise model and the parameters of the noise | 5 |
| `src/sbmlsim/fit/petab_v2/gaps.py` | gaps `noise-model`, `foreign-extension`, the detail of `noise-parameters` | 6 |
| `docs/petab.md`, `docs/api/fit.petab_v2.likelihood.md`, `zensical.toml`, `CLAUDE.md` | documentation | 6 |
| `tests/data/petab/sciml_001_llh.tsv` | new: measurements and simulations of the reference case | 1 |
| `tests/fit/test_petab_v2_likelihood.py` | new: the functions of arrays (Task 1), the functions of a problem (Task 3) | 1, 3 |
| `tests/fit/test_predictions.py` | new: the noise models and the predictions of a resolved problem | 2 |
| `tests/fit/test_petab_v2_noise.py` | new: reader and extensions (Task 4), export (Task 5), gaps (Task 6) | 4, 5, 6 |

Why the noise model is in `fit/objects.py` and not in `likelihood.py`: `FitMapping` carries it and `optimization.py` collects it, and both are imported by `petab_v2/export.py`, which `petab_v2/__init__.py` imports, so a class of `petab_v2` in the signature of `FitMapping` is a circular import. `likelihood.py` imports `OptimizationProblem` only for type checking, as `gaps.py` and `metrics.py` do.

---

### Task 1: The noise model and the log density

**Files:**
- Modify: `docs/superpowers/specs/2026-09-30-petab-sciml-design.md` (formatting of its python blocks only)
- Create: `tests/data/petab/sciml_001_llh.tsv`
- Create: `tests/fit/test_petab_v2_likelihood.py`
- Modify: `src/sbmlsim/fit/objects.py:78` (after `UNUSED_KINDS`)
- Modify: `src/sbmlsim/fit/__init__.py:16-38`
- Create: `src/sbmlsim/fit/petab_v2/likelihood.py`

**Interfaces:**
- Consumes: `petab.v2.math.sympify_petab(str) -> sympy.Basic`, which parses the math of PEtab (`^` is the power, `log` the natural logarithm) into symbols with `real=True`. Symbols are matched by `str(symbol)`.
- Produces, in `sbmlsim.fit.objects` and exported by `sbmlsim.fit`:
  - `class NoiseDistribution(StrEnum)` with `NORMAL = "normal"`, `LOG_NORMAL = "log-normal"`, `LAPLACE = "laplace"`, `LOG_LAPLACE = "log-laplace"` and the property `is_log -> bool`.
  - `@dataclass(frozen=True) class NoiseParameter(pid: str, value: float, estimate: bool = False, lower_bound: float | None = None, upper_bound: float | None = None)`.
  - `@dataclass(frozen=True) class NoiseModel(formula: str, distribution: NoiseDistribution = NORMAL, placeholders: tuple[str, ...] = (), placeholder_values: tuple[tuple[float | str, ...], ...] = (), parameters: tuple[NoiseParameter, ...] = (), observable: str | None = None)`. `placeholder_values[i][j]` is the value of `placeholders[j]` for the measurement `i`. It raises `ValueError` if a row does not have one value per placeholder.
- Produces, in `sbmlsim.fit.petab_v2.likelihood`:
  - `NOISE_PLACEHOLDER = "sd"`, `DEFAULT_SIGMA = 1.0`.
  - `log_density(measurement: ArrayLike, simulation: ArrayLike, sigma: ArrayLike, distribution: NoiseDistribution = NoiseDistribution.NORMAL) -> np.ndarray`, the log density of every measurement.
  - `noise_values(noise: NoiseModel, size: int, values: Mapping[str, float] | None = None, simulation: ArrayLike | None = None) -> np.ndarray`, the scale of the noise of every measurement. `values` are parameter values which override the nominal values of `noise.parameters`.
  - `default_noise_model(errors: ArrayLike | None) -> NoiseModel`.

The definitions, with the measurement `m`, the simulation `y` and the scale `s`:

| distribution | log density |
| --- | --- |
| `normal` | `-0.5 log(2 pi s^2) - 0.5 ((m - y) / s)^2` |
| `log-normal` | `-0.5 log(2 pi s^2 m^2) - 0.5 ((log m - log y) / s)^2` |
| `laplace` | `-log(2 s) - abs(m - y) / s` |
| `log-laplace` | `-log(2 s m) - abs(log m - log y) / s` |

The values of the tests for `m = 2`, `y = 3`, `s = 0.5` are `-2.2257913526447273`, `-1.2477424409910038`, `-2.0` and `-1.5040773967762742`. They were calculated by hand from the table and agree with `petab.v2.calculate.calculate_single_llh` to the last digit.

- [ ] **Step 1: Format the python blocks of the spec**

`uv run ruff format --check` fails on the branch because of the python blocks of the spec, which is not caused by this plan and is fixed first so that the check means something afterwards.

```bash
uv run ruff format --check
```

Expected: `1 file would be reformatted`, and the file is `docs/superpowers/specs/2026-09-30-petab-sciml-design.md`. If the check reports no file, skip the rest of this step.

```bash
uv run ruff format docs/superpowers/specs/2026-09-30-petab-sciml-design.md
git diff --stat
uv run ruff format --check
git add docs/superpowers/specs/2026-09-30-petab-sciml-design.md
git commit -m "Format the python blocks of the PEtab SciML design"
```

Expected: `git diff --stat` lists only the spec, the diff changes only whitespace and line breaks inside its python blocks, and the second check reports `files already formatted` without a file to reformat.

- [ ] **Step 2: Add the reference data**

The measurements of `petab/measurements.tsv` and the simulations of `simulations.tsv` of the case `sciml_problem_import/001` of the PEtab SciML test suite, joined by observable and time. The columns are separated by tabs. The observables of the case have `noiseFormula = 0.05` and `noiseDistribution = normal`, its `solutions.yaml` says `llh: 33.02909543616689` and `tol_llh: 0.001`.

**Create `tests/data/petab/sciml_001_llh.tsv`:**

````text
observableId	time	measurement	simulation
prey_o	1.0	0.17301723066954827	0.19971308977580363
prey_o	2.0	0.48917673783383914	0.4848624070907903
prey_o	3.0	1.643995531803569	1.608461064731753
prey_o	4.0	5.45196278566626	5.494412897414485
prey_o	5.0	2.9775220192442062	3.083784996961849
prey_o	6.0	0.18166337927167636	0.19525096436479264
prey_o	7.0	0.3481122259853288	0.2965509122049375
prey_o	8.0	0.9379191088947554	0.9054918836530768
prey_o	9.0	3.113240033804055	3.106111500481494
prey_o	10.0	8.863933242141234	8.839154061453035
predator_o	1.0	0.8474159434934865	0.9226939413096142
predator_o	2.0	0.2111345157163205	0.19263532796062602
predator_o	3.0	-0.025053892755649274	0.07072847039335985
predator_o	4.0	0.12501049397609923	0.13535116008928508
predator_o	5.0	6.700454554758639	6.6950730099708515
predator_o	6.0	2.007158288516007	1.9957113490583238
predator_o	7.0	0.42009248269510124	0.39133349944898516
predator_o	8.0	0.04803185161440761	0.09927080575204167
predator_o	9.0	0.12866939374360575	0.07136720076441437
predator_o	10.0	1.192783778293036	1.2192471831747158
````

Check that the file has tabs and 21 lines:

```bash
grep -c $'\t' tests/data/petab/sciml_001_llh.tsv
```

Expected: `21`.

- [ ] **Step 3: Write the failing tests**

**Create `tests/fit/test_petab_v2_likelihood.py`:**

````python
"""Tests of the log-likelihood of a problem."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sbmlsim.fit.objects import NoiseDistribution, NoiseModel, NoiseParameter
from sbmlsim.fit.petab_v2.likelihood import (
    default_noise_model,
    log_density,
    noise_values,
)

#: simulations and measurements of the case `sciml_problem_import/001` of the
#: PEtab SciML test suite, normal noise with the scale 0.05
SCIML_001 = Path(__file__).parents[1] / "data" / "petab" / "sciml_001_llh.tsv"
#: the `llh` of the `solutions.yaml` of the case and its `tol_llh`
SCIML_001_LLH = 33.02909543616689
SCIML_001_TOL = 1e-3


def test_log_density_normal() -> None:
    """The density of the normal distribution, calculated by hand."""
    # -0.5 * log(2 pi 0.25) - 0.5 * ((2 - 3) / 0.5)^2
    expected = -0.5 * np.log(2.0 * np.pi * 0.25) - 2.0
    assert expected == pytest.approx(-2.2257913526447273, rel=1e-15)
    density = log_density([2.0], [3.0], 0.5, NoiseDistribution.NORMAL)
    assert density == pytest.approx([expected], rel=1e-14)


def test_log_density_log_normal() -> None:
    """The simulation is the median and the density is the one of `m`."""
    # -0.5 * log(2 pi 0.25 * 4) - 0.5 * ((log 2 - log 3) / 0.5)^2
    expected = -0.5 * np.log(2.0 * np.pi) - 2.0 * np.log(1.5) ** 2
    assert expected == pytest.approx(-1.2477424409910038, rel=1e-15)
    density = log_density([2.0], [3.0], 0.5, NoiseDistribution.LOG_NORMAL)
    assert density == pytest.approx([expected], rel=1e-14)


def test_log_density_laplace() -> None:
    """The density of the Laplace distribution, calculated by hand."""
    # -log(2 * 0.5) - |2 - 3| / 0.5
    density = log_density([2.0], [3.0], 0.5, NoiseDistribution.LAPLACE)
    assert density == pytest.approx([-2.0], rel=1e-14)


def test_log_density_log_laplace() -> None:
    """The density of the log-Laplace distribution, calculated by hand."""
    # -log(2 * 0.5 * 2) - |log 2 - log 3| / 0.5
    expected = -np.log(2.0) - 2.0 * np.log(1.5)
    assert expected == pytest.approx(-1.5040773967762742, rel=1e-15)
    density = log_density([2.0], [3.0], 0.5, NoiseDistribution.LOG_LAPLACE)
    assert density == pytest.approx([expected], rel=1e-14)


@pytest.mark.parametrize("distribution", list(NoiseDistribution))
def test_log_density_is_a_density(distribution: NoiseDistribution) -> None:
    """The density of the measurement integrates to one.

    This is what tells the density of `m` from the density of `log m`, i.e.
    it pins the `1 / m` of the logarithmic distributions.
    """
    start = 1e-6 if distribution.is_log else -27.0
    m = np.linspace(start, start + 60.0, 600001)
    density = np.exp(log_density(m, np.full_like(m, 3.0), 0.25, distribution))
    assert np.trapezoid(density, m) == pytest.approx(1.0, abs=1e-4)


def test_log_density_has_its_maximum_at_the_measurement() -> None:
    """A simulation which hits the measurement is the most likely one."""
    for distribution in NoiseDistribution:
        at, off = log_density([2.0, 2.0], [2.0, 2.5], 0.5, distribution)
        assert at > off


def test_log_density_takes_a_scale_per_measurement() -> None:
    """The scale is one value or one per measurement."""
    density = log_density([1.0, 1.0], [1.0, 1.0], [0.5, 2.0])
    assert density == pytest.approx(
        [-0.5 * np.log(2.0 * np.pi * 0.25), -0.5 * np.log(2.0 * np.pi * 4.0)]
    )


@pytest.mark.parametrize("sigma", [0.0, -1.0, np.nan, np.inf])
def test_log_density_requires_a_positive_scale(sigma: float) -> None:
    """A scale which is not a positive number is an error and not a `nan`."""
    with pytest.raises(ValueError, match="positive finite"):
        log_density([1.0], [1.0], sigma)


def test_log_density_requires_positive_values_for_a_log_distribution() -> None:
    """The logarithm of a measurement which is not positive does not exist."""
    with pytest.raises(ValueError, match="measurement"):
        log_density([0.0], [1.0], 0.5, NoiseDistribution.LOG_NORMAL)
    with pytest.raises(ValueError, match="simulation"):
        log_density([1.0], [-1.0], 0.5, NoiseDistribution.LOG_LAPLACE)
    # the normal distribution takes them
    assert np.isfinite(log_density([-1.0], [0.0], 0.5)).all()


def test_log_density_requires_one_shape() -> None:
    """Every measurement has a simulation."""
    with pytest.raises(ValueError, match="shape"):
        log_density([1.0, 2.0], [1.0], 0.5)
    with pytest.raises(ValueError, match="scale"):
        log_density([1.0, 2.0], [1.0, 2.0], [0.5, 0.5, 0.5])


def test_log_likelihood_of_the_sciml_test_suite() -> None:
    """The `llh` of a case of the PEtab SciML test suite is reproduced.

    This pins the sign and the constant of the log-likelihood against a value
    which another tool calculated.
    """
    df = pd.read_csv(SCIML_001, sep="\t", float_precision="round_trip")
    assert len(df) == 20
    llh = float(np.sum(log_density(df.measurement, df.simulation, 0.05)))
    assert llh == pytest.approx(SCIML_001_LLH, abs=SCIML_001_TOL)
    assert llh == pytest.approx(SCIML_001_LLH, rel=1e-12)


def test_noise_values_of_a_number() -> None:
    """A noise formula which is a number is the scale of every measurement."""
    sigma = noise_values(NoiseModel(formula="0.05"), size=3)
    assert sigma.tolist() == [0.05, 0.05, 0.05]


def test_noise_values_of_a_placeholder() -> None:
    """A placeholder has a value per measurement."""
    noise = NoiseModel(
        formula="0.1 + 2 * sd",
        placeholders=("sd",),
        placeholder_values=((0.5,), (1.0,)),
    )
    assert noise_values(noise, size=2) == pytest.approx([1.1, 2.1])


def test_noise_values_of_a_parameter() -> None:
    """A parameter is evaluated at the given value, the nominal one without."""
    noise = NoiseModel(
        formula="sigma_a ^ 2",
        parameters=(NoiseParameter(pid="sigma_a", value=3.0, estimate=True),),
    )
    assert noise_values(noise, size=2) == pytest.approx([9.0, 9.0])
    assert noise_values(noise, size=2, values={"sigma_a": 2.0}) == pytest.approx(
        [4.0, 4.0]
    )


def test_noise_values_of_a_placeholder_which_is_a_parameter() -> None:
    """The value of a placeholder is a number or a formula of parameters."""
    noise = NoiseModel(
        formula="sd",
        placeholders=("sd",),
        placeholder_values=((0.5,), ("sigma_a",), ("2 * sigma_a",)),
        parameters=(NoiseParameter(pid="sigma_a", value=3.0),),
    )
    assert noise_values(noise, size=3) == pytest.approx([0.5, 3.0, 6.0])


def test_noise_values_of_the_observable() -> None:
    """A noise which is proportional to the simulation."""
    noise = NoiseModel(formula="0.1 * obs_a + 0.01", observable="obs_a")
    sigma = noise_values(noise, size=2, simulation=[1.0, 2.0])
    assert sigma == pytest.approx([0.11, 0.21])


def test_noise_values_of_a_placeholder_named_like_a_parameter() -> None:
    """A placeholder is the value of the measurement, whatever else has its name."""
    noise = NoiseModel(
        formula="sd",
        placeholders=("sd",),
        placeholder_values=((0.5,), (0.25,)),
        parameters=(NoiseParameter(pid="sd", value=3.0),),
    )
    sigma = noise_values(noise, size=2, values={"sd": 7.0})
    assert sigma.tolist() == [0.5, 0.25]


def test_noise_values_require_every_symbol() -> None:
    """A symbol without a value is named."""
    with pytest.raises(ValueError, match="k_unknown"):
        noise_values(NoiseModel(formula="2 * k_unknown"), size=2)


def test_noise_values_require_a_value_per_measurement() -> None:
    """The values of the placeholders are the ones of the measurements."""
    noise = NoiseModel(
        formula="sd", placeholders=("sd",), placeholder_values=((0.5,), (1.0,))
    )
    with pytest.raises(ValueError, match="'2' values"):
        noise_values(noise, size=3)


def test_noise_values_require_a_positive_scale() -> None:
    """A noise formula which is not positive is an error."""
    with pytest.raises(ValueError, match="positive finite"):
        noise_values(NoiseModel(formula="-0.05"), size=2)


def test_a_noise_model_requires_a_value_per_placeholder() -> None:
    """A measurement has a value for every placeholder."""
    with pytest.raises(ValueError, match="placeholders"):
        NoiseModel(
            formula="a + b", placeholders=("a", "b"), placeholder_values=((0.5,),)
        )


def test_default_noise_model() -> None:
    """Without a noise model the noise is the standard deviation of the data."""
    noise = default_noise_model(np.array([0.5, 0.25]))
    assert noise.distribution is NoiseDistribution.NORMAL
    assert noise_values(noise, size=2).tolist() == [0.5, 0.25]
    # and the scale is one for data without errors
    assert noise_values(default_noise_model(None), size=2).tolist() == [1.0, 1.0]
````

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest -q -x tests/fit/test_petab_v2_likelihood.py`

Expected: FAIL while the module is collected, with `ImportError: cannot import name 'NoiseDistribution' from 'sbmlsim.fit.objects'`.

- [ ] **Step 5: Add the noise model to the objects of a fit**

**Patch `src/sbmlsim/fit/objects.py`:**

````diff
--- a/src/sbmlsim/fit/objects.py
+++ b/src/sbmlsim/fit/objects.py
@@ -76,6 +76,96 @@

 #: kinds which a fit does not use at all, i.e. which are not even resolved
 UNUSED_KINDS: tuple[MappingKind, ...] = (MappingKind.EXCLUDED,)
+
+
+class NoiseDistribution(StrEnum):
+    """Distribution of the noise of a measurement.
+
+    These are the distributions of PEtab v2. The simulation is the median of
+    the distribution and the noise formula gives its scale: the standard
+    deviation of `normal`, the standard deviation of the logarithm of
+    `log-normal` and the scale `b` of `laplace` and, on the logarithm, of
+    `log-laplace`.
+    """
+
+    NORMAL = "normal"
+    LOG_NORMAL = "log-normal"
+    LAPLACE = "laplace"
+    LOG_LAPLACE = "log-laplace"
+
+    @property
+    def is_log(self) -> bool:
+        """Check whether the noise acts on the logarithm of the measurement."""
+        return self in {NoiseDistribution.LOG_NORMAL, NoiseDistribution.LOG_LAPLACE}
+
+
+@dataclass(frozen=True)
+class NoiseParameter:
+    """A parameter of a noise formula which is not an entity of a model.
+
+    Attributes:
+        pid: id of the parameter.
+        value: nominal value, which the log-likelihood uses unless the
+            parameter set it is evaluated at has a value for `pid`.
+        estimate: whether the problem the noise model was read from estimates
+            the parameter. `sbmlsim` does not estimate it, the flag and the
+            bounds are kept so that the problem is written as it was read.
+        lower_bound: lower bound of the estimation, `None` if there is none.
+        upper_bound: upper bound of the estimation, `None` if there is none.
+    """
+
+    pid: str
+    value: float
+    estimate: bool = False
+    lower_bound: float | None = None
+    upper_bound: float | None = None
+
+
+@dataclass(frozen=True)
+class NoiseModel:
+    """The noise of the measurements of a fit mapping.
+
+    The noise model does not enter the cost of a fit, which is a weighted
+    least squares fit. It is what the log-likelihood of a problem is
+    calculated with, see `sbmlsim.fit.petab_v2.likelihood`.
+
+    Attributes:
+        formula: the noise formula in the math of PEtab, e.g. `0.05`, `sd` or
+            `sigma_a + 0.1 * sd`. Its symbols are the `placeholders`, the
+            `parameters`, the parameters of the fit and the `observable`.
+        distribution: distribution of the noise.
+        placeholders: symbols of the formula which have a value per
+            measurement.
+        placeholder_values: for every measurement the values of the
+            placeholders, in the order of the measurements of the mapping. A
+            value is a number or a formula of parameters.
+        parameters: the parameters of the formula and of the placeholder
+            values which are not entities of a model, with their nominal value.
+        observable: symbol of the formula which stands for the simulation,
+            `None` if the formula has none.
+    """
+
+    formula: str
+    distribution: NoiseDistribution = NoiseDistribution.NORMAL
+    placeholders: tuple[str, ...] = ()
+    placeholder_values: tuple[tuple[float | str, ...], ...] = ()
+    parameters: tuple[NoiseParameter, ...] = ()
+    observable: str | None = None
+
+    def __post_init__(self) -> None:
+        """Check that every measurement has a value for every placeholder.
+
+        Raises:
+            ValueError: if a measurement has more or fewer values than the
+                noise model has placeholders.
+        """
+        for k, values in enumerate(self.placeholder_values):
+            if len(values) != len(self.placeholders):
+                raise ValueError(
+                    f"The noise formula '{self.formula}' has the placeholders "
+                    f"'{list(self.placeholders)}', but the measurement '{k}' "
+                    f"has the values '{list(values)}'."
+                )


 class FitMappingCollection:
````

**Patch `src/sbmlsim/fit/__init__.py`:**

````diff
--- a/src/sbmlsim/fit/__init__.py
+++ b/src/sbmlsim/fit/__init__.py
@@ -20,6 +20,9 @@
     FitParameter,
     MappingKind,
     MappingMetaData,
+    NoiseDistribution,
+    NoiseModel,
+    NoiseParameter,
 )
 from .options import FitSettings
 from .parameters import ParameterSet, ParameterSets
@@ -33,6 +36,9 @@
     "FitSettings",
     "MappingKind",
     "MappingMetaData",
+    "NoiseDistribution",
+    "NoiseModel",
+    "NoiseParameter",
     "ParameterSet",
     "ParameterSets",
 ]
````

- [ ] **Step 6: Write the functions of arrays**

**Create `src/sbmlsim/fit/petab_v2/likelihood.py`:**

````python
"""The log-likelihood of an optimization problem.

A fit of `sbmlsim` is a weighted least squares fit, PEtab defines the
objective of a problem as the likelihood of its measurements under a noise
model. This module calculates that log-likelihood for the evaluation of a
problem, the optimizer does not use it.

The noise model of a fit mapping is its `sbmlsim.fit.objects.NoiseModel`, i.e.
the noise formula and the distribution of the observable it was read from. The
simulation `y` is the median of the distribution of the measurement `m` and
the noise formula gives its scale `s`, which are the definitions of PEtab v2:

    normal        log p = -0.5 log(2 pi s^2)       - 0.5 ((m - y) / s)^2
    log-normal    log p = -0.5 log(2 pi s^2 m^2)   - 0.5 ((log m - log y) / s)^2
    laplace       log p = -log(2 s)                - |m - y| / s
    log-laplace   log p = -log(2 s m)              - |log m - log y| / s

The functions of arrays (`log_density`, `noise_values`) know nothing of a
problem, `log_likelihood` and `gradient` simulate one.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import sympy as sp
from numpy.typing import ArrayLike
from petab.v2.math import sympify_petab

from sbmlsim.fit.objects import NoiseDistribution, NoiseModel

logger = logging.getLogger(__name__)

#: the placeholder of the noise of a measurement, i.e. the standard deviation
#: of its data. PEtab v2 declares the placeholders of an observable in
#: `noisePlaceholders`; the `noiseParameter${n}_${observableId}` names of v1
#: are gone
NOISE_PLACEHOLDER = "sd"

#: scale of the noise of a fit mapping which has neither a noise model nor
#: errors on its data, in the unit of the observable
DEFAULT_SIGMA = 1.0


# --- THE FUNCTIONS OF ARRAYS ---


def log_density(
    measurement: ArrayLike,
    simulation: ArrayLike,
    sigma: ArrayLike,
    distribution: NoiseDistribution = NoiseDistribution.NORMAL,
) -> np.ndarray:
    """Get the log density of every measurement under its noise model.

    Args:
        measurement: the measured values.
        simulation: the simulated values at the measurements, the median of
            the distribution.
        sigma: scale of the noise, one value or one value per measurement.
        distribution: distribution of the noise.

    Returns:
        The log density of every measurement, see the module for the
        definitions.

    Raises:
        ValueError: if the arrays do not have one shape, if a scale is not a
            positive finite number, or if a measurement or a simulation of a
            logarithmic distribution is not positive.
    """
    m = np.asarray(measurement, dtype=float)
    y = np.asarray(simulation, dtype=float)
    if m.shape != y.shape:
        raise ValueError(
            f"The log density requires a simulation for every measurement, but "
            f"the measurements have the shape '{m.shape}' and the simulations "
            f"'{y.shape}'."
        )
    try:
        s = np.broadcast_to(np.asarray(sigma, dtype=float), m.shape)
    except ValueError as err:
        raise ValueError(
            f"The log density requires one scale of the noise or one per "
            f"measurement, but the measurements have the shape '{m.shape}' and "
            f"the scales '{np.shape(sigma)}'."
        ) from err
    if np.any(~np.isfinite(s)) or np.any(s <= 0.0):
        raise ValueError(
            f"The scale of the noise must be a positive finite number, but is "
            f"'{s[~np.isfinite(s) | (s <= 0.0)]}'."
        )

    distribution = NoiseDistribution(distribution)
    if distribution.is_log:
        for name, values in (("measurement", m), ("simulation", y)):
            if np.any(values <= 0.0):
                raise ValueError(
                    f"The distribution '{distribution.value}' requires positive "
                    f"values, but '{int(np.sum(values <= 0.0))}' of the "
                    f"'{name}' values are not: '{values[values <= 0.0]}'."
                )
        residual = (np.log(m) - np.log(y)) / s
        # the density of the measurement and not of its logarithm, i.e. the
        # jacobian `1 / m` of the transformation is part of it
        scale = s * m
    else:
        residual = (m - y) / s
        scale = s

    if distribution in {NoiseDistribution.NORMAL, NoiseDistribution.LOG_NORMAL}:
        return -0.5 * np.log(2.0 * np.pi * np.square(scale)) - 0.5 * np.square(residual)
    return -np.log(2.0 * scale) - np.abs(residual)


def _evaluate(formula: str | float, variables: Mapping[str, Any], context: str) -> Any:
    """Evaluate a formula of the math of PEtab on the values of its symbols.

    Args:
        formula: the formula, or a number.
        variables: value of every symbol, a number or an array.
        context: what the formula belongs to, for the message of an error.

    Returns:
        The value of the formula, an array if one of its symbols is one.

    Raises:
        ValueError: if a symbol of the formula has no value.
    """
    if isinstance(formula, int | float):
        return float(formula)
    expression = sympify_petab(formula)
    symbols = sorted(expression.free_symbols, key=str)
    missing = [str(symbol) for symbol in symbols if str(symbol) not in variables]
    if missing:
        raise ValueError(
            f"{context}: the formula '{formula}' uses '{missing}', which are "
            f"neither placeholders nor parameters with a value. A noise formula "
            f"is a number, a parameter or a formula of them."
        )
    function = sp.lambdify(symbols, expression, modules="numpy")
    return function(*[variables[str(symbol)] for symbol in symbols])


def noise_values(
    noise: NoiseModel,
    size: int,
    values: Mapping[str, float] | None = None,
    simulation: ArrayLike | None = None,
) -> np.ndarray:
    """Get the scale of the noise of every measurement of a fit mapping.

    The symbols of the noise formula are resolved in this order: a placeholder
    is the value of the measurement, the symbol of the observable is the
    simulation, a parameter is the value of `values` and, without one, the
    nominal value of the noise model. A parameter which a problem estimates is
    therefore evaluated at the value it is given, it is not estimated.

    Args:
        noise: noise model of the fit mapping.
        size: number of measurements.
        values: values of parameters by their id, e.g. the values of a
            parameter set.
        simulation: the simulated values at the measurements, for a noise
            formula which holds the observable.

    Returns:
        The scale of the noise, one value per measurement.

    Raises:
        ValueError: if the noise model has placeholders and not one value of
            them per measurement, if a symbol has no value, or if a scale is
            not a positive finite number.
    """
    context = f"noise formula '{noise.formula}'"
    parameters: dict[str, Any] = {p.pid: p.value for p in noise.parameters}
    parameters.update(values or {})

    variables: dict[str, Any] = dict(parameters)
    if noise.observable is not None and simulation is not None:
        variables[noise.observable] = np.asarray(simulation, dtype=float)
    if noise.placeholders:
        if len(noise.placeholder_values) != size:
            raise ValueError(
                f"{context}: '{len(noise.placeholder_values)}' values of the "
                f"placeholders '{list(noise.placeholders)}' for '{size}' "
                f"measurements."
            )
        for k, placeholder in enumerate(noise.placeholders):
            variables[placeholder] = np.array(
                [
                    float(_evaluate(row[k], parameters, context))
                    for row in noise.placeholder_values
                ],
                dtype=float,
            )

    sigma = np.array(
        np.broadcast_to(
            np.asarray(_evaluate(noise.formula, variables, context), dtype=float),
            (size,),
        )
    )
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError(
            f"{context}: the scale of the noise must be a positive finite "
            f"number, but is '{sigma[~np.isfinite(sigma) | (sigma <= 0.0)]}'."
        )
    return sigma


def default_noise_model(errors: ArrayLike | None) -> NoiseModel:
    """Get the noise model of a fit mapping which does not define one.

    The noise is normal with the standard deviation of the reference data,
    through the placeholder `NOISE_PLACEHOLDER`, and with `DEFAULT_SIGMA` for
    data without errors. This is what the export writes for such a mapping, so
    the log-likelihood of a problem and of its PEtab problem agree.

    Args:
        errors: the errors of the reference data of the mapping, i.e.
            `OptimizationProblem.y_errors[k]`, `None` if it has none.

    Returns:
        The noise model.
    """
    if errors is None:
        return NoiseModel(formula=repr(DEFAULT_SIGMA))
    return NoiseModel(
        formula=NOISE_PLACEHOLDER,
        placeholders=(NOISE_PLACEHOLDER,),
        placeholder_values=tuple(
            (float(error),) for error in np.asarray(errors, dtype=float)
        ),
    )
````

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/fit/test_petab_v2_likelihood.py tests/fit/test_objects.py`

Expected: PASS, `34 passed` (28 of the likelihood, 6 of the objects).

- [ ] **Step 8: Commit**

```bash
uv run ruff check && uv run ruff format --check && uvx ty check
git add tests/data/petab/sciml_001_llh.tsv tests/fit/test_petab_v2_likelihood.py src/sbmlsim/fit/objects.py src/sbmlsim/fit/__init__.py src/sbmlsim/fit/petab_v2/likelihood.py
git commit -m "Noise model of a fit mapping and the log density of PEtab v2"
```

Expected: the three checks report no diagnostics and the commit is created. The message has no attribution line.


### Task 2: The noise models and the predictions of a resolved problem

**Files:**
- Create: `tests/fit/test_predictions.py`
- Modify: `src/sbmlsim/fit/objects.py` (`FitMapping.__init__`, lines 300-333 before Task 1, which added 92 lines above it)
- Modify: `src/sbmlsim/fit/optimization.py:18-24` (import), `:252-281` (`_reset_mappings`), `:746` (`initialize`), `:1352` (new methods before `_interrupted_result`), `:1428-1441` (`residuals`)

**Interfaces:**
- Consumes: `NoiseModel`, `NoiseDistribution` of `sbmlsim.fit.objects` (Task 1).
- Produces:
  - `FitMapping(experiment, reference, observable, weight=None, metadata=None, noise: NoiseModel | None = None)` with the attribute `noise`.
  - `OptimizationProblem.noise_models: list[NoiseModel | None]`, one entry per resolved fit mapping, in the order of `mapping_keys`. It is reset by `_reset_mappings` and filled by `initialize`.
  - `OptimizationProblem.predictions(x: np.ndarray, indices: Sequence[int] | None = None) -> dict[int, np.ndarray]`. `x` are the parameter values on the linear scale in the order of `problem.pids`, `indices` are indices of fit mappings, `problem.training_indices` by default. The values are the simulation interpolated at `problem.x_references[k]`, without the baseline shift of the residual. It raises `ValueError` if the problem is not initialized, if no simulator is set or if the integration of a simulation failed.
  - `OptimizationProblem._interpolate(k: int, df: pd.DataFrame) -> np.ndarray`, which `residuals` now uses.

`residuals` does not change what it calculates. The interpolation it did inline moves into `_interpolate`, and the `console.print` of an interpolation error becomes a `logger.error`, because library code does not print. `console` stays imported, `report` uses it.

- [ ] **Step 1: Write the failing tests**

**Create `tests/fit/test_predictions.py`:**

````python
"""Tests of the predictions and the noise models of a resolved problem."""

import dataclasses

import numpy as np
import pytest

from examples.hctz_fitting.experiments.studies import Beermann1976
from sbmlsim.fit import FitMapping, FitSettings
from sbmlsim.fit.objects import NoiseDistribution, NoiseModel
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ResidualType


@pytest.fixture
def settings_tight() -> FitSettings:
    """Get settings with which two simulations of a problem agree.

    With a variable step size the output grid of a simulation is the steps of
    the integrator, which are not the same in two runs, and the data is
    interpolated on it: two simulations then differ by `1e-6` however tight
    the tolerances are. With a fixed grid they agree to the tolerances.
    """
    return FitSettings(
        residual=ResidualType.ABSOLUTE,
        variable_step_size=False,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-10,
    )


def test_a_fit_mapping_has_no_noise_model_by_default(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The noise model is optional, a fit does not need one."""
    op_hctz_iv.initialize(fit_settings)
    assert op_hctz_iv.noise_models == [None] * len(op_hctz_iv.mapping_keys)


def test_the_noise_model_of_a_fit_mapping_is_resolved(
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The problem holds the noise model of every mapping it resolves."""
    noise = NoiseModel(formula="0.05", distribution=NoiseDistribution.LAPLACE)
    fit_mappings = Beermann1976.fit_mappings

    def with_noise(self: Beermann1976) -> dict[str, FitMapping]:
        mappings = fit_mappings(self)
        mappings["fm_hctz_iv1_5_urine"].noise = noise
        return mappings

    monkeypatch.setattr(Beermann1976, "fit_mappings", with_noise)
    op_hctz_iv.initialize(fit_settings)

    k = op_hctz_iv.mapping_keys.index("fm_hctz_iv1_5_urine")
    assert op_hctz_iv.noise_models[k] is noise
    others = [n for i, n in enumerate(op_hctz_iv.noise_models) if i != k]
    assert others == [None] * (len(op_hctz_iv.mapping_keys) - 1)


def test_the_noise_models_are_resolved_again(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A second initialization does not append to the first."""
    op_hctz_iv.initialize(fit_settings)
    op_hctz_iv.initialize(fit_settings, force=True)
    assert len(op_hctz_iv.noise_models) == len(op_hctz_iv.mapping_keys)


def test_predictions_are_the_simulation_at_the_data(
    op_hctz_pk: OptimizationProblem, settings_tight: FitSettings
) -> None:
    """The predictions are what the residuals are calculated from."""
    op_hctz_pk.initialize(settings_tight)
    x = np.asarray(op_hctz_pk.x0, dtype=float)

    # all mappings, which is what the complete data of the residuals simulates
    indices = op_hctz_pk.indices()
    predictions = op_hctz_pk.predictions(x, indices=indices)
    assert sorted(predictions) == indices

    data = op_hctz_pk.residuals(op_hctz_pk.to_scale(x), complete_data=True)
    assert isinstance(data, dict)
    for k, prediction in predictions.items():
        assert prediction.shape == np.shape(op_hctz_pk.y_references[k])
        assert prediction == pytest.approx(
            np.asarray(data["y_obsip"][k]), rel=1e-8, abs=1e-12
        )


def test_predictions_of_the_training_data_by_default(
    op_hctz_pk: OptimizationProblem, settings_tight: FitSettings
) -> None:
    """Without indices the training data is simulated, as a fit does."""
    op_hctz_pk.initialize(settings_tight)
    x = np.asarray(op_hctz_pk.x0, dtype=float)

    predictions = op_hctz_pk.predictions(x)
    assert sorted(predictions) == op_hctz_pk.training_indices
    assert op_hctz_pk.validation_indices

    # the residuals of a fit without weights are `prediction - data`
    expected = np.concatenate(
        [
            predictions[k] - np.asarray(op_hctz_pk.y_references[k], dtype=float)
            for k in op_hctz_pk.training_indices
        ]
    )
    residuals = np.asarray(op_hctz_pk.residuals(op_hctz_pk.to_scale(x)), dtype=float)
    assert residuals == pytest.approx(expected, rel=1e-8, abs=1e-12)


def test_predictions_of_selected_mappings(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The mappings are selected by their indices."""
    op_hctz_pk.initialize(fit_settings)
    x = np.asarray(op_hctz_pk.x0, dtype=float)
    indices = op_hctz_pk.validation_indices
    assert indices
    assert sorted(op_hctz_pk.predictions(x, indices=indices)) == indices


def test_predictions_are_not_shifted_to_the_baseline(
    op_hctz_pk: OptimizationProblem, settings_tight: FitSettings
) -> None:
    """The residual of the settings does not change what is simulated."""
    x = np.asarray(op_hctz_pk.x0, dtype=float)
    op_hctz_pk.initialize(settings_tight)
    absolute = op_hctz_pk.predictions(x)
    op_hctz_pk.initialize(
        dataclasses.replace(settings_tight, residual=ResidualType.ABSOLUTE_TO_BASELINE)
    )
    baseline = op_hctz_pk.predictions(x)

    assert sorted(absolute) == sorted(baseline)
    # a curve which does not start at zero is where a shift would show
    assert any(prediction[0] != 0.0 for prediction in absolute.values())
    for k, prediction in absolute.items():
        assert baseline[k] == pytest.approx(prediction, rel=1e-8, abs=1e-12)


def test_predictions_require_an_initialized_problem(
    op_hctz_iv: OptimizationProblem,
) -> None:
    """A problem which is not initialized has no simulator."""
    with pytest.raises(ValueError, match="initialized"):
        op_hctz_iv.predictions(np.asarray(op_hctz_iv.x0, dtype=float))


def test_predictions_of_a_failed_simulation_raise(
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A simulation which failed has no prediction, and the error says so."""
    op_hctz_iv.initialize(fit_settings)
    simulator = op_hctz_iv.runner_initialized.simulator
    assert simulator is not None

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("CVODE failed")

    monkeypatch.setattr(simulator, "_timecourses", fail)
    with pytest.raises(ValueError, match="failed"):
        op_hctz_iv.predictions(np.asarray(op_hctz_iv.x0, dtype=float))
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -x tests/fit/test_predictions.py`

Expected: FAIL with `AttributeError: 'OptimizationProblem' object has no attribute 'noise_models'` or `... has no attribute 'predictions'`, depending on the test which runs first.

- [ ] **Step 3: Give the fit mapping its noise model**

**Patch `src/sbmlsim/fit/objects.py`:**

````diff
--- a/src/sbmlsim/fit/objects.py
+++ b/src/sbmlsim/fit/objects.py
@@ -402,6 +402,7 @@
         observable: FitData,
         weight: float | None = None,
         metadata: MappingMetaData | None = None,
+        noise: NoiseModel | None = None,
     ):
         """Initialize FitMapping.

@@ -415,12 +416,17 @@
             weight: weight of the fit mapping, the count of the reference data
                 is used if no weight is given.
             metadata: metadata of the mapping.
+            noise: noise model of the measurements, which the log-likelihood
+                of the problem uses. Without one the noise is normal with the
+                standard deviation of the reference data, see
+                `sbmlsim.fit.petab_v2.likelihood.default_noise_model`.
         """
         self.experiment = experiment
         self.reference = reference
         self.observable = observable
         self._weight = weight
         self.metadata = metadata
+        self.noise = noise

     @property
     def weight(self) -> float:
````

- [ ] **Step 4: Collect the noise models and add the predictions**

**Patch `src/sbmlsim/fit/optimization.py`:**

````diff
--- a/src/sbmlsim/fit/optimization.py
+++ b/src/sbmlsim/fit/optimization.py
@@ -21,6 +21,7 @@
     FitMappingCollection,
     FitParameter,
     MappingKind,
+    NoiseModel,
 )
 from sbmlsim.fit.options import (
     FitSettings,
@@ -266,6 +267,9 @@
         self.y_references: list[Any] = []
         self.y_errors: list[Any] = []
         self.y_errors_type: list[str | None] = []
+        #: the noise model of every mapping, `None` for a mapping without one,
+        #: see `sbmlsim.fit.petab_v2.likelihood`
+        self.noise_models: list[NoiseModel | None] = []
         # total weights for points (data points and curve weights)
         self.weights: list[Any] = []
         self.weights_points: list[Any] = []  # weights for data points based on errors
@@ -744,6 +748,7 @@
                 self.y_references.append(y_ref)
                 self.y_errors.append(y_ref_err)
                 self.y_errors_type.append(y_ref_err_type)
+                self.noise_models.append(mapping.noise)
                 # weights
                 self.weights.append(weight)
                 self.weights_points.append(weight_points)
@@ -1349,6 +1354,82 @@

         return results

+    def _interpolate(self, k: int, df: pd.DataFrame) -> np.ndarray:
+        """Get the simulation of a fit mapping at its reference data.
+
+        Args:
+            k: index of the fit mapping.
+            df: result of the simulation of the mapping.
+
+        Returns:
+            The observable, interpolated at the x values of the reference data.
+
+        Raises:
+            ValueError: if the reference data is outside of the simulation.
+        """
+        f = interpolate.interp1d(
+            x=df[self.xid_observable[k]],
+            y=df[self.yid_observable[k]],
+            copy=False,
+            assume_sorted=True,
+        )
+        try:
+            return np.asarray(f(self.x_references[k]), dtype=float)
+        except ValueError:
+            logger.error(
+                "Interpolation error in the fit mapping '%s.%s'.",
+                self.experiment_keys[k],
+                self.mapping_keys[k],
+            )
+            raise
+
+    def predictions(
+        self, x: np.ndarray, indices: Sequence[int] | None = None
+    ) -> dict[int, np.ndarray]:
+        """Get the simulation of fit mappings at their reference data.
+
+        The predictions are the values of the observable as they are
+        simulated, i.e. the baseline of a curve is not subtracted, whatever
+        the residual of the settings is.
+
+        Args:
+            x: values of the parameters in the units of the model, i.e. on the
+                linear scale, in the order of the parameters of the problem.
+            indices: indices of the fit mappings, the training data by
+                default.
+
+        Returns:
+            The prediction at the reference data of every mapping, by the
+            index of the mapping.
+
+        Raises:
+            ValueError: if no simulator is set or if the integration of a
+                simulation failed.
+        """
+        simulator: SimulatorSerial | None = self.runner_initialized.simulator
+        if simulator is None:
+            raise ValueError(f"No simulator set on OptimizationProblem '{self.opid}'.")
+        Q_ = self.runner_initialized.Q_
+
+        values = np.asarray(x, dtype=float)
+        quantities = [Q_(value, self.punits[ix]) for ix, value in enumerate(values)]
+        evaluated = set(self.training_indices if indices is None else indices)
+        results = self._simulate_groups(
+            simulator=simulator, quantities=quantities, evaluated=evaluated, x=values
+        )
+
+        predictions: dict[int, np.ndarray] = {}
+        for k in sorted(evaluated):
+            df = results[k]
+            if df is None:
+                raise ValueError(
+                    f"'{self.opid}': the simulation of the fit mapping "
+                    f"'{self.experiment_keys[k]}.{self.mapping_keys[k]}' failed "
+                    f"for the parameters '{dict(zip(self.pids, values, strict=True))}'."
+                )
+            predictions[k] = self._interpolate(k, df)
+        return predictions
+
     def _interrupted_result(
         self, err: Exception, x0log: np.ndarray
     ) -> RuntimeErrorOptimizeResult:
@@ -1427,18 +1508,7 @@

             df = results[k]
             if df is not None:
-                # interpolation of simulation results and requested time points
-                f = interpolate.interp1d(
-                    x=df[self.xid_observable[k]],
-                    y=df[self.yid_observable[k]],
-                    copy=False,
-                    assume_sorted=True,
-                )
-                try:
-                    y_obsip = f(self.x_references[k])
-                except ValueError as err:
-                    console.print(f"Interpolation error in mapping key: {mapping_key}")
-                    raise err
+                y_obsip = self._interpolate(k, df)

                 if self.residual in {
                     ResidualType.ABSOLUTE_TO_BASELINE,
````

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/fit/test_predictions.py`

Expected: PASS, `9 passed`.

- [ ] **Step 6: Run the tests of the fit, the residuals are its core**

Run: `uv run pytest -q -x tests/fit`

Expected: PASS, no failure and no error. The cost of every existing test is unchanged, `_interpolate` is the code `residuals` had inline.

- [ ] **Step 7: Commit**

```bash
uv run ruff check && uv run ruff format --check && uvx ty check
git add tests/fit/test_predictions.py src/sbmlsim/fit/objects.py src/sbmlsim/fit/optimization.py
git commit -m "Noise models and predictions of a resolved optimization problem"
```

Expected: the three checks report no diagnostics and the commit is created. The message has no attribution line.


### Task 3: The log-likelihood and its gradient

**Files:**
- Modify: `tests/fit/test_petab_v2_likelihood.py` (imports, and the tests of a problem appended)
- Modify: `src/sbmlsim/fit/petab_v2/likelihood.py` (imports, `DEFAULT_STEP`, and the functions of a problem appended)
- Modify: `src/sbmlsim/fit/petab_v2/__init__.py:15-43`
- Modify: `examples/hctz_fitting/fitting/petab_problem.py:35`, `:137-160`

**Interfaces:**
- Consumes:
  - `log_density`, `noise_values`, `default_noise_model` (Task 1).
  - `OptimizationProblem.predictions(x, indices=None) -> dict[int, np.ndarray]` and `OptimizationProblem.noise_models` (Task 2).
  - `OptimizationProblem.training_indices`, `.y_references`, `.y_errors`, `.pids`, `.parameters`, `.xmodel`, `.residual`, `.is_initialized`, `.settings_initialized`, `.experiment_keys`, `.mapping_keys` (existing).
  - `ParameterSet(sid, values, units)`, `ParameterSet.x(pids) -> np.ndarray`, which raises `KeyError` for a missing parameter, and `ParameterSet.from_fit_parameters(parameters, x, sid, provenance)` (existing, `fit/parameters.py`).
- Produces, in `sbmlsim.fit.petab_v2.likelihood`, the first two also exported by `sbmlsim.fit.petab_v2`:
  - `log_likelihood(problem: OptimizationProblem, parameters: ParameterSet | None = None) -> float`.
  - `gradient(problem: OptimizationProblem, parameters: ParameterSet | None = None, step: float = 1e-6) -> pd.Series`, indexed by `problem.pids`, named `gradient`.
  - `noise_model_of(problem: OptimizationProblem, k: int) -> NoiseModel`, the noise model of the mapping `k` or `default_noise_model(problem.y_errors[k])`.
  - `nominal_parameters(problem: OptimizationProblem) -> ParameterSet` with the `sid` `nominal`.
  - `DEFAULT_STEP = 1e-6`.

Decisions this task implements:

- The nominal value of a parameter is its `start_value`: the reader sets it from the `nominalValue` of the parameter table and the exporter writes it there. A parameter without a start value has the value of the model, `problem.xmodel[k]`.
- A parameter set may hold more than the parameters of the fit. A value for a parameter of a noise formula overrides its nominal value, which is how "a noise formula which holds an estimated parameter is evaluated at the value of the parameter set" is met.
- A problem whose residual is `ABSOLUTE_TO_BASELINE` or `NORMALIZED_TO_BASELINE` has no log-likelihood: `initialize` shifted `y_references` to the baseline, so the problem does not hold the measurements. `log_likelihood` raises.
- The step of the gradient is `step * max(|x|, 1)` on the linear scale, the rule of `fisher.jacobian`, which applies it in the space of the optimizer.
- The tests of the gradient compare it with `-J' r`: with a normal noise of scale one, the residual `ABSOLUTE`, no weights and the linear parameter scale, `llh = -n/2 log(2 pi) - cost`, so the log-likelihood is checked against the objective of the fit and the gradient against the jacobian of `fisher.py`.

- [ ] **Step 1: Write the failing tests**

**Patch `tests/fit/test_petab_v2_likelihood.py`:**

````diff
--- a/tests/fit/test_petab_v2_likelihood.py
+++ b/tests/fit/test_petab_v2_likelihood.py
@@ -1,16 +1,26 @@
 """Tests of the log-likelihood of a problem."""

+import logging
 from pathlib import Path

 import numpy as np
 import pandas as pd
 import pytest

+from sbmlsim.fit import FitSettings
+from sbmlsim.fit.cli import FitDefinition
+from sbmlsim.fit.fisher import jacobian
 from sbmlsim.fit.objects import NoiseDistribution, NoiseModel, NoiseParameter
+from sbmlsim.fit.optimization import OptimizationProblem
+from sbmlsim.fit.options import ParameterScaleType, ResidualType
+from sbmlsim.fit.parameters import ParameterSet
 from sbmlsim.fit.petab_v2.likelihood import (
     default_noise_model,
+    gradient,
     log_density,
+    log_likelihood,
     noise_values,
+    nominal_parameters,
 )

 #: simulations and measurements of the case `sciml_problem_import/001` of the
@@ -215,3 +225,244 @@
     assert noise_values(noise, size=2).tolist() == [0.5, 0.25]
     # and the scale is one for data without errors
     assert noise_values(default_noise_model(None), size=2).tolist() == [1.0, 1.0]
+
+
+# --- THE LOG-LIKELIHOOD OF A PROBLEM ---
+
+
+@pytest.fixture
+def settings_likelihood() -> FitSettings:
+    """Get settings with which the cost is the sum of squares of the data.
+
+    The integrator is tighter than in a fit and its output grid is fixed: a
+    finite difference of the log-likelihood divides the error of a simulation
+    by the step, and with a variable step size two simulations of a problem
+    differ by `1e-6`, see `sbmlsim.fit.petab_v2.likelihood.gradient`.
+    """
+    return FitSettings(
+        residual=ResidualType.ABSOLUTE,
+        parameter_scale=ParameterScaleType.LINEAR,
+        variable_step_size=False,
+        absolute_tolerance=1e-12,
+        relative_tolerance=1e-10,
+    )
+
+
+@pytest.fixture
+def op_unit_noise(
+    op_hctz_pk: OptimizationProblem, settings_likelihood: FitSettings
+) -> OptimizationProblem:
+    """Get the initialized problem with a normal noise of scale one."""
+    op_hctz_pk.initialize(settings_likelihood)
+    op_hctz_pk.noise_models = [
+        NoiseModel(formula="1.0") for _ in op_hctz_pk.mapping_keys
+    ]
+    return op_hctz_pk
+
+
+def test_the_hctz_problem_has_a_log_likelihood(
+    op_hctz_pk: OptimizationProblem, definition_hctz_pk: FitDefinition
+) -> None:
+    """The reference problem has a log-likelihood, with the settings of its fit."""
+    op_hctz_pk.initialize(definition_hctz_pk.settings)
+    assert all(noise is None for noise in op_hctz_pk.noise_models)
+
+    llh = log_likelihood(op_hctz_pk)
+    assert np.isfinite(llh)
+    assert llh < 0.0
+
+
+def test_log_likelihood_of_the_nominal_parameters_by_default(
+    op_hctz_pk: OptimizationProblem, settings_likelihood: FitSettings
+) -> None:
+    """The nominal values are the start values of the parameters."""
+    op_hctz_pk.initialize(settings_likelihood)
+    nominal = nominal_parameters(op_hctz_pk)
+    assert nominal.sid == "nominal"
+    assert nominal.values == {p.pid: p.start_value for p in op_hctz_pk.parameters}
+
+    llh = log_likelihood(op_hctz_pk)
+    assert llh == pytest.approx(log_likelihood(op_hctz_pk, nominal), rel=1e-8)
+    # and other parameters are another likelihood
+    other = ParameterSet(
+        sid="other", values={pid: 2.0 * v for pid, v in nominal.values.items()}
+    )
+    assert log_likelihood(op_hctz_pk, other) != pytest.approx(llh, rel=1e-3)
+
+
+def test_log_likelihood_is_the_sum_over_the_training_data(
+    op_hctz_pk: OptimizationProblem, settings_likelihood: FitSettings
+) -> None:
+    """The validation data and the outliers do not enter."""
+    op_hctz_pk.initialize(settings_likelihood)
+    assert op_hctz_pk.validation_indices
+    nominal = nominal_parameters(op_hctz_pk)
+    predictions = op_hctz_pk.predictions(nominal.x(op_hctz_pk.pids))
+    assert sorted(predictions) == op_hctz_pk.training_indices
+
+    expected = 0.0
+    for k in op_hctz_pk.training_indices:
+        errors = op_hctz_pk.y_errors[k]
+        sigma = errors if errors is not None else 1.0
+        expected += float(
+            np.sum(log_density(op_hctz_pk.y_references[k], predictions[k], sigma))
+        )
+    assert log_likelihood(op_hctz_pk) == pytest.approx(expected, rel=1e-8)
+
+
+def test_log_likelihood_of_a_unit_noise_is_the_cost(
+    op_unit_noise: OptimizationProblem,
+) -> None:
+    """With a normal noise of scale one the cost is the log-likelihood.
+
+    `llh = -n/2 log(2 pi) - 0.5 sum((y - m)^2)`, and the second term is the
+    cost of a fit of the absolute residuals without weights.
+    """
+    problem = op_unit_noise
+    nominal = nominal_parameters(problem)
+    x = nominal.x(problem.pids)
+    n = sum(len(problem.y_references[k]) for k in problem.training_indices)
+
+    cost = problem.cost_least_square(problem.to_scale(x))
+    assert log_likelihood(problem, nominal) == pytest.approx(
+        -0.5 * n * np.log(2.0 * np.pi) - cost, rel=1e-8
+    )
+
+
+def test_gradient_of_a_unit_noise_is_the_gradient_of_the_cost(
+    op_unit_noise: OptimizationProblem,
+) -> None:
+    """The gradient is `-J' r` of the residuals of the fit."""
+    problem = op_unit_noise
+    nominal = nominal_parameters(problem)
+    x = nominal.x(problem.pids)
+
+    residuals = np.asarray(problem.residuals(problem.to_scale(x)), dtype=float)
+    expected = -jacobian(problem, problem.to_scale(x)).T @ residuals
+
+    grad = gradient(problem, nominal)
+    assert list(grad.index) == problem.pids
+    # a difference of two log-likelihoods has the rounding error of their
+    # size, which is not relative to a small derivative
+    scale = float(np.max(np.abs(expected)))
+    assert grad.to_numpy() == pytest.approx(expected, rel=1e-4, abs=1e-6 * scale)
+    assert np.all(grad.to_numpy() != 0.0)
+
+
+def test_log_likelihood_uses_the_noise_model_of_a_mapping(
+    op_unit_noise: OptimizationProblem,
+) -> None:
+    """A parameter of the noise is evaluated at the value of the set."""
+    problem = op_unit_noise
+    nominal = nominal_parameters(problem)
+    n = sum(len(problem.y_references[k]) for k in problem.training_indices)
+    unit = log_likelihood(problem, nominal)
+
+    problem.noise_models = [
+        NoiseModel(
+            formula="sigma_a",
+            parameters=(NoiseParameter(pid="sigma_a", value=1.0, estimate=True),),
+        )
+        for _ in problem.mapping_keys
+    ]
+    assert log_likelihood(problem, nominal) == pytest.approx(unit, rel=1e-8)
+
+    # llh(s) = -n log(s) - n/2 log(2 pi) - cost / s^2
+    cost = -unit - 0.5 * n * np.log(2.0 * np.pi)
+    wide = ParameterSet(sid="wide", values={**nominal.values, "sigma_a": 2.0})
+    assert log_likelihood(problem, wide) == pytest.approx(
+        -n * np.log(2.0) - 0.5 * n * np.log(2.0 * np.pi) - cost / 4.0, rel=1e-8
+    )
+    # the gradient is the one of the parameters of the fit
+    assert list(gradient(problem, wide).index) == problem.pids
+
+
+def test_log_likelihood_requires_an_initialized_problem(
+    op_hctz_iv: OptimizationProblem,
+) -> None:
+    """The data of the problem has to be resolved."""
+    with pytest.raises(ValueError, match="initialize"):
+        log_likelihood(op_hctz_iv)
+
+
+def test_log_likelihood_requires_the_measurements(
+    op_hctz_iv: OptimizationProblem,
+) -> None:
+    """Data which is shifted to its baseline is not what the noise describes."""
+    op_hctz_iv.initialize(FitSettings(residual=ResidualType.ABSOLUTE_TO_BASELINE))
+    with pytest.raises(ValueError, match="baseline"):
+        log_likelihood(op_hctz_iv)
+
+
+def test_log_likelihood_names_the_mapping_of_an_error(
+    op_unit_noise: OptimizationProblem,
+) -> None:
+    """An error of a noise model says which mapping it belongs to."""
+    problem = op_unit_noise
+    k = problem.training_indices[0]
+    problem.noise_models[k] = NoiseModel(formula="k_unknown")
+    with pytest.raises(ValueError, match=problem.mapping_keys[k]):
+        log_likelihood(problem)
+
+
+def test_log_likelihood_of_a_log_distribution_requires_positive_data(
+    op_unit_noise: OptimizationProblem,
+) -> None:
+    """The amount in the urine is zero at the first measurement."""
+    problem = op_unit_noise
+    k = next(
+        k
+        for k in problem.training_indices
+        if np.any(np.asarray(problem.y_references[k]) <= 0.0)
+    )
+    problem.noise_models[k] = NoiseModel(
+        formula="0.5", distribution=NoiseDistribution.LOG_NORMAL
+    )
+    with pytest.raises(ValueError, match="positive") as excinfo:
+        log_likelihood(problem)
+    assert problem.mapping_keys[k] in str(excinfo.value)
+
+
+def test_gradient_of_a_failed_simulation_raises(
+    op_unit_noise: OptimizationProblem, monkeypatch: pytest.MonkeyPatch
+) -> None:
+    """A step which the model cannot simulate is an error and not a `nan`."""
+    simulator = op_unit_noise.runner_initialized.simulator
+    assert simulator is not None
+
+    def fail(*args: object, **kwargs: object) -> None:
+        raise RuntimeError("CVODE failed")
+
+    monkeypatch.setattr(simulator, "_timecourses", fail)
+    with pytest.raises(ValueError, match="failed"):
+        gradient(op_unit_noise)
+
+
+def test_log_likelihood_requires_the_parameters_of_the_fit(
+    op_unit_noise: OptimizationProblem,
+) -> None:
+    """A parameter set which lacks a parameter is an error."""
+    with pytest.raises(KeyError, match="does not contain"):
+        log_likelihood(op_unit_noise, ParameterSet(sid="empty", values={}))
+
+
+def test_gradient_warns_about_a_variable_step_size(
+    op_hctz_iv: OptimizationProblem,
+    fit_settings: FitSettings,
+    caplog: pytest.LogCaptureFixture,
+) -> None:
+    """The differences of simulations on a variable grid are noise."""
+    assert fit_settings.variable_step_size
+    op_hctz_iv.initialize(fit_settings)
+    with caplog.at_level(logging.WARNING, logger="sbmlsim.fit.petab_v2.likelihood"):
+        grad = gradient(op_hctz_iv)
+    assert "variable_step_size" in caplog.text
+    assert np.all(np.isfinite(grad.to_numpy()))
+
+
+def test_gradient_requires_a_positive_step(
+    op_unit_noise: OptimizationProblem,
+) -> None:
+    """A step of zero divides by zero."""
+    with pytest.raises(ValueError, match="step"):
+        gradient(op_unit_noise, step=0.0)
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -x tests/fit/test_petab_v2_likelihood.py`

Expected: FAIL while the module is collected, with `ImportError: cannot import name 'gradient' from 'sbmlsim.fit.petab_v2.likelihood'`.

- [ ] **Step 3: Write the functions of a problem**

**Patch `src/sbmlsim/fit/petab_v2/likelihood.py`:**

````diff
--- a/src/sbmlsim/fit/petab_v2/likelihood.py
+++ b/src/sbmlsim/fit/petab_v2/likelihood.py
@@ -23,14 +23,20 @@

 import logging
 from collections.abc import Mapping
-from typing import Any
+from typing import TYPE_CHECKING, Any

 import numpy as np
+import pandas as pd
 import sympy as sp
 from numpy.typing import ArrayLike
 from petab.v2.math import sympify_petab

 from sbmlsim.fit.objects import NoiseDistribution, NoiseModel
+from sbmlsim.fit.options import ResidualType
+from sbmlsim.fit.parameters import ParameterSet
+
+if TYPE_CHECKING:
+    from sbmlsim.fit.optimization import OptimizationProblem

 logger = logging.getLogger(__name__)

@@ -43,6 +49,10 @@
 #: scale of the noise of a fit mapping which has neither a noise model nor
 #: errors on its data, in the unit of the observable
 DEFAULT_SIGMA = 1.0
+
+#: relative step of the finite differences of the gradient, the rule of
+#: `sbmlsim.fit.fisher`
+DEFAULT_STEP = 1e-6


 # --- THE FUNCTIONS OF ARRAYS ---
@@ -236,3 +246,183 @@
             (float(error),) for error in np.asarray(errors, dtype=float)
         ),
     )
+
+
+# --- THE FUNCTIONS OF A PROBLEM ---
+
+
+def noise_model_of(problem: OptimizationProblem, k: int) -> NoiseModel:
+    """Get the noise model of a fit mapping of an initialized problem.
+
+    Args:
+        problem: initialized optimization problem.
+        k: index of the fit mapping.
+
+    Returns:
+        The noise model of the mapping, `default_noise_model` if it has none.
+    """
+    noise = problem.noise_models[k]
+    if noise is not None:
+        return noise
+    return default_noise_model(problem.y_errors[k])
+
+
+def nominal_parameters(problem: OptimizationProblem) -> ParameterSet:
+    """Get the nominal values of the parameters of a problem.
+
+    The nominal value of a parameter is its start value, which is the
+    `nominalValue` of the parameter table for a problem which was read, and
+    the value of the model for a parameter without a start value.
+
+    Args:
+        problem: initialized optimization problem.
+
+    Returns:
+        The parameter set `nominal`.
+    """
+    x = [
+        float(problem.xmodel[k]) if p.start_value is None else float(p.start_value)
+        for k, p in enumerate(problem.parameters)
+    ]
+    return ParameterSet.from_fit_parameters(
+        parameters=problem.parameters,
+        x=x,
+        sid="nominal",
+        provenance="nominal values of the problem",
+    )
+
+
+def _check_problem(problem: OptimizationProblem) -> None:
+    """Check that the log-likelihood of a problem is defined.
+
+    Raises:
+        ValueError: if the problem is not initialized, or if its residuals are
+            relative to the baseline of a curve.
+    """
+    if not problem.is_initialized:
+        raise ValueError(
+            f"'{problem.opid}': the log-likelihood requires the resolved "
+            f"mappings, call `initialize(settings)` first."
+        )
+    if problem.residual in {
+        ResidualType.ABSOLUTE_TO_BASELINE,
+        ResidualType.NORMALIZED_TO_BASELINE,
+    }:
+        raise ValueError(
+            f"'{problem.opid}': the residual '{problem.residual.name}' shifts "
+            f"the data to the baseline of its curve, so the problem does not "
+            f"hold the measurements the noise model describes. Initialize the "
+            f"problem with the residual 'ABSOLUTE' or 'NORMALIZED' for its "
+            f"log-likelihood."
+        )
+
+
+def log_likelihood(
+    problem: OptimizationProblem, parameters: ParameterSet | None = None
+) -> float:
+    """Get the log-likelihood of the training data of a problem.
+
+    The problem is simulated at the parameters and the log density of every
+    measurement of the training data is summed, see `log_density`. The noise
+    model of a fit mapping is `noise_model_of`. The settings of the fit, i.e.
+    the residual, the weights and the loss function, do not enter.
+
+    Args:
+        problem: initialized optimization problem.
+        parameters: parameters to evaluate the log-likelihood at, with the
+            values of the parameters of the fit and, optionally, of parameters
+            of the noise formulas. `nominal_parameters` by default.
+
+    Returns:
+        The log-likelihood.
+
+    Raises:
+        ValueError: if the problem is not initialized, if its residuals are
+            relative to the baseline, if a simulation failed or if the noise
+            model of a mapping cannot be evaluated.
+        KeyError: if the parameters lack a parameter of the fit.
+    """
+    _check_problem(problem)
+    pset = parameters if parameters is not None else nominal_parameters(problem)
+    predictions = problem.predictions(pset.x(problem.pids))
+
+    total = 0.0
+    for k in problem.training_indices:
+        key = f"{problem.experiment_keys[k]}.{problem.mapping_keys[k]}"
+        noise = noise_model_of(problem, k)
+        measurement = np.asarray(problem.y_references[k], dtype=float)
+        simulation = predictions[k]
+        try:
+            sigma = noise_values(
+                noise,
+                size=measurement.size,
+                values=pset.values,
+                simulation=simulation,
+            )
+            density = log_density(
+                measurement, simulation, sigma, distribution=noise.distribution
+            )
+        except ValueError as err:
+            raise ValueError(f"'{problem.opid}', fit mapping '{key}': {err}") from err
+        total += float(np.sum(density))
+    return total
+
+
+def gradient(
+    problem: OptimizationProblem,
+    parameters: ParameterSet | None = None,
+    step: float = DEFAULT_STEP,
+) -> pd.Series:
+    """Get the gradient of the log-likelihood by central finite differences.
+
+    The differences are taken on the linear scale, i.e. in the units of the
+    model, with the step `step * max(|x|, 1)` for a parameter of the value
+    `x`. A difference divides the error of a simulation by the step, so the
+    problem is initialized with `FitSettings` of tight tolerances and
+    `variable_step_size=False`: with a variable step size the data is
+    interpolated on the steps of the integrator, which differ between two
+    simulations, and the simulations of one problem differ by `1e-6` however
+    tight the tolerances are. The gradient logs a warning in this case.
+
+    Args:
+        problem: initialized optimization problem.
+        parameters: parameters to evaluate the gradient at,
+            `nominal_parameters` by default.
+        step: relative step of the differences.
+
+    Returns:
+        The derivative of the log-likelihood by every parameter of the fit,
+        indexed by the ids of the parameters.
+
+    Raises:
+        ValueError: if the step is not positive, or if the log-likelihood
+            cannot be calculated, see `log_likelihood`.
+    """
+    if not step > 0.0:
+        raise ValueError(f"The step of the gradient must be positive, not '{step}'.")
+    _check_problem(problem)
+    if problem.settings_initialized.variable_step_size:
+        logger.warning(
+            "'%s': the gradient is calculated with `variable_step_size=True`, "
+            "the differences of its simulations are of the size of the step "
+            "'%s'. Initialize the problem with `variable_step_size=False` and "
+            "tight tolerances for a gradient.",
+            problem.opid,
+            step,
+        )
+    pset = parameters if parameters is not None else nominal_parameters(problem)
+    x = pset.x(problem.pids)
+
+    def shifted(pid: str, value: float) -> ParameterSet:
+        """Get the parameter set with one value replaced."""
+        return ParameterSet(
+            sid=pset.sid, values={**pset.values, pid: value}, units=dict(pset.units)
+        )
+
+    derivatives: dict[str, float] = {}
+    for k, pid in enumerate(problem.pids):
+        h = step * max(abs(float(x[k])), 1.0)
+        plus = log_likelihood(problem, shifted(pid, float(x[k]) + h))
+        minus = log_likelihood(problem, shifted(pid, float(x[k]) - h))
+        derivatives[pid] = (plus - minus) / (2.0 * h)
+    return pd.Series(derivatives, name="gradient", dtype=float)
````

**Patch `src/sbmlsim/fit/petab_v2/__init__.py`:**

````diff
--- a/src/sbmlsim/fit/petab_v2/__init__.py
+++ b/src/sbmlsim/fit/petab_v2/__init__.py
@@ -25,6 +25,7 @@
     gaps_of_problem,
     gaps_table,
 )
+from sbmlsim.fit.petab_v2.likelihood import gradient, log_likelihood
 from sbmlsim.fit.petab_v2.reader import PetabReader, from_petab

 __all__ = [
@@ -39,5 +40,7 @@
     "from_petab",
     "gaps_of_problem",
     "gaps_table",
+    "gradient",
+    "log_likelihood",
     "to_petab",
 ]
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/fit/test_petab_v2_likelihood.py`

Expected: PASS, `42 passed`.

- [ ] **Step 5: Report the log-likelihood in the example of the reference problem**

**Patch `examples/hctz_fitting/fitting/petab_problem.py`:**

````diff
--- a/examples/hctz_fitting/fitting/petab_problem.py
+++ b/examples/hctz_fitting/fitting/petab_problem.py
@@ -32,7 +32,12 @@
 from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS
 from sbmlsim.console import console
 from sbmlsim.fit import display
-from sbmlsim.fit.petab_v2 import gaps_of_problem, gaps_table, to_petab
+from sbmlsim.fit.petab_v2 import (
+    gaps_of_problem,
+    gaps_table,
+    log_likelihood,
+    to_petab,
+)
 from sbmlsim.fit.petab_v2.extension import extension_of
 from sbmlsim.fit.petab_v2.reader import from_petab

@@ -137,6 +142,10 @@
     x = np.log10(np.asarray(problem.x0, dtype=float))
     cost = problem.cost_least_square(x)
     cost_petab = petab_problem_read.cost_least_square(x)
+    # the log-likelihood of the nominal parameters, with the noise model the
+    # export wrote, i.e. the standard deviation of the data
+    llh = log_likelihood(problem)
+    llh_petab = log_likelihood(petab_problem_read)
     display.key_values(
         {
             "collections": (
@@ -156,6 +165,7 @@
             ),
             "cost": f"{cost:.6f} -> {cost_petab:.6f}",
             "difference": f"{abs(cost - cost_petab) / cost:.2e} (relative)",
+            "log-likelihood": f"{llh:.4f} -> {llh_petab:.4f}",
         }
     )
     display.print_parameters(petab_problem_read.parameters)
````

- [ ] **Step 6: Run the example**

Run: `uv run pytest -q -x tests/examples/test_example_scripts.py -k petab_problem`

Expected: PASS, `1 passed`. Run the example itself once and read its output:

```bash
uv run python -m examples.hctz_fitting.fitting.petab_problem | tail -12
```

The example writes into `results/petab/PK` of the working directory, which git ignores.

Expected: the section "Read back" has the row `log-likelihood      -41500.5705 -> -41500.5705`, i.e. the HCTZ problem has a log-likelihood and the problem which is read back has the same one. The digits after the fourth may differ, the example runs with a variable step size.

- [ ] **Step 7: Commit**

```bash
uv run ruff check && uv run ruff format --check && uvx ty check
git add tests/fit/test_petab_v2_likelihood.py src/sbmlsim/fit/petab_v2/likelihood.py src/sbmlsim/fit/petab_v2/__init__.py examples/hctz_fitting/fitting/petab_problem.py
git commit -m "Log-likelihood of an optimization problem and its gradient"
```

Expected: the three checks report no diagnostics and the commit is created. The message has no attribution line.


### Task 4: The reader keeps the noise model and checks the extensions

**Files:**
- Create: `tests/fit/test_petab_v2_noise.py`
- Modify: `src/sbmlsim/fit/petab_v2/extension.py:17-26` (imports, `KNOWN_EXTENSIONS`), `:99` (`check_extensions` appended)
- Modify: `src/sbmlsim/fit/petab_v2/reader.py:23-24`, `:29-38` (imports), `:109-113` (`__init__`), `:160-165` (`from_yaml`), `:638-644` (`fit_mappings`), `:973` (`_formula` before `_is_number`)

**Interfaces:**
- Consumes:
  - `NoiseDistribution`, `NoiseModel`, `NoiseParameter`, `FitMapping(noise=...)` (Tasks 1 and 2).
  - `log_likelihood` (Task 3), in a test only.
  - `petab.v2.Observable.noise_formula: sympy.Basic`, `.noise_distribution: petab.v2.core.NoiseDistribution` (a `StrEnum` with the values of `NoiseDistribution`), `.noise_placeholders: list[sympy.Symbol]`; `petab.v2.Measurement.noise_parameters: list[sympy.Basic]`; `petab.v2.Parameter.id`, `.lb`, `.ub`, `.nominal_value`, `.estimate`.
  - `petab.v2.math.petab_math_str(expr) -> str`, the math of PEtab of a sympy expression, and `petab.v1.yaml.load_yaml(path) -> dict`, `write_yaml(dict, path)`.
- Produces:
  - `sbmlsim.fit.petab_v2.extension.KNOWN_EXTENSIONS: frozenset[str] = frozenset({"sbmlsim"})`. Phase 3 adds `sciml` to it.
  - `sbmlsim.fit.petab_v2.extension.check_extensions(extensions: Mapping[str, Any] | None, known: Collection[str] = KNOWN_EXTENSIONS) -> list[str]`, which raises `ValueError` for a required extension which is not known and returns the ids of the ones to ignore. It does not log, the reader does.
  - `PetabReader.noise_model(observable_id: str) -> NoiseModel`, and every `FitMapping` of `PetabReader.fit_mappings` carries it as `noise`.
  - `PetabReader._is_fit_parameter(parameter) -> bool`.

Decisions this task implements:

- The extensions are checked twice. `PetabReader.from_yaml` checks the YAML before `petab` reads the problem, because `petab` imports `petab_sciml` to read the files of a `sciml` block and fails with a `ModuleNotFoundError` before the reader sees the configuration. `PetabReader.__init__` checks the configuration of a problem which was not read from a file, and logs the extensions which are ignored. The warning is therefore logged once.
- A block without `required` is read as required.
- The standard deviation the reader derives from the first numeric noise parameter (`value_sd`, `reader.py:587-596`) stays as it is: the weights of a fit do not change.
- The noise formula is stored as text: a number as `repr(float)`, e.g. `0.05`, anything else as `petab_math_str`, e.g. `KI__HCTZEX_k + 0.1`.
- A parameter of the parameter table which a noise formula uses becomes a `NoiseParameter` unless it is a parameter of the fit, whose value comes from the parameter set. A symbol which is neither is logged and left to `log_likelihood`, which raises: reading such a problem must not fail, it is fitted today.

- [ ] **Step 1: Write the failing tests**

**Create `tests/fit/test_petab_v2_noise.py`:**

````python
"""Tests of the noise model and the extensions of a PEtab v2 problem."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from petab.v1.yaml import load_yaml, write_yaml

from sbmlsim.fit import FitSettings
from sbmlsim.fit.objects import NoiseDistribution, NoiseModel, NoiseParameter
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.petab_v2 import to_petab
from sbmlsim.fit.petab_v2.extension import (
    EXTENSION_ID,
    KNOWN_EXTENSIONS,
    check_extensions,
)
from sbmlsim.fit.petab_v2.likelihood import log_likelihood
from sbmlsim.fit.petab_v2.reader import PetabReader, from_petab

#: a parameter of the noise, which PEtab estimates and `sbmlsim` does not
SIGMA = NoiseParameter(
    pid="sigma_a", value=0.5, estimate=True, lower_bound=0.01, upper_bound=10.0
)


@pytest.fixture
def petab_iv(
    tmp_path: Path, op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> Path:
    """Write the iv problem, four fit mappings without errors on the data."""
    output_dir = tmp_path / "petab_iv"
    to_petab(op_hctz_iv, output_dir, settings=fit_settings)
    return output_dir


def _edit(path: Path, edit: Callable[[pd.DataFrame], pd.DataFrame | None]) -> None:
    """Edit a table of a problem in place, all of its values are text."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    edited = edit(df)
    (df if edited is None else edited).to_csv(path, sep="\t", index=False)


def _observable_ids(petab_dir: Path) -> list[str]:
    """Get the ids of the observables of a problem, in the order of the table."""
    df = pd.read_csv(petab_dir / "observables.tsv", sep="\t", dtype=str)
    return list(df["observableId"])


def _set_noise(petab_dir: Path, observable_id: str, **columns: str) -> None:
    """Set columns of the observable table for one observable."""

    def edit(df: pd.DataFrame) -> None:
        for column, value in columns.items():
            df.loc[df["observableId"] == observable_id, column] = value

    _edit(petab_dir / "observables.tsv", edit)


def _add_sigma(petab_dir: Path) -> None:
    """Add the parameter `SIGMA` of the noise to the parameter table."""

    def edit(df: pd.DataFrame) -> pd.DataFrame:
        row = dict.fromkeys(df.columns, "")
        row.update(
            parameterId=SIGMA.pid,
            lowerBound=str(SIGMA.lower_bound),
            upperBound=str(SIGMA.upper_bound),
            nominalValue=str(SIGMA.value),
            estimate="true",
        )
        return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    _edit(petab_dir / "parameters.tsv", edit)


def _add_extension(petab_dir: Path, extension_id: str, block: dict[str, Any]) -> None:
    """Add the block of an extension to the YAML of a problem."""
    yaml_file = petab_dir / "problem.yaml"
    config = load_yaml(yaml_file)
    config.setdefault("extensions", {})[extension_id] = block
    write_yaml(config, yaml_file)


# --- THE NOISE MODEL ---


def test_reader_keeps_a_number(petab_iv: Path) -> None:
    """The noise formula and the distribution of an observable are kept."""
    observable_id = _observable_ids(petab_iv)[0]
    _set_noise(
        petab_iv, observable_id, noiseFormula="0.05", noiseDistribution="laplace"
    )

    reader = PetabReader.from_yaml(petab_iv / "problem.yaml")
    assert reader.noise_model(observable_id) == NoiseModel(
        formula="0.05", distribution=NoiseDistribution.LAPLACE
    )


def test_reader_keeps_a_parameter_of_the_noise(petab_iv: Path) -> None:
    """A parameter of the noise is kept with its value and is not fitted."""
    observable_id = _observable_ids(petab_iv)[0]
    _set_noise(
        petab_iv,
        observable_id,
        noiseFormula=SIGMA.pid,
        noiseDistribution="log-normal",
    )
    _add_sigma(petab_iv)

    reader = PetabReader.from_yaml(petab_iv / "problem.yaml")
    assert reader.noise_model(observable_id) == NoiseModel(
        formula=SIGMA.pid,
        distribution=NoiseDistribution.LOG_NORMAL,
        parameters=(SIGMA,),
    )
    assert SIGMA.pid not in {p.pid for p in reader.fit_parameters()}


def test_reader_keeps_the_placeholders(petab_iv: Path) -> None:
    """The noise parameters of the measurements fill in the placeholders."""
    observable_id = _observable_ids(petab_iv)[0]
    _set_noise(petab_iv, observable_id, noiseFormula="2 * sd", noisePlaceholders="sd")
    _add_sigma(petab_iv)

    def edit(df: pd.DataFrame) -> None:
        rows = df.index[df["observableId"] == observable_id]
        assert len(rows) == 2
        df.loc[rows[0], "noiseParameters"] = "0.25"
        df.loc[rows[1], "noiseParameters"] = SIGMA.pid

    _edit(petab_iv / "measurements.tsv", edit)

    noise = PetabReader.from_yaml(petab_iv / "problem.yaml").noise_model(observable_id)
    assert noise.placeholders == ("sd",)
    assert noise.placeholder_values == ((0.25,), (SIGMA.pid,))
    assert noise.parameters == (SIGMA,)


def test_a_fit_parameter_is_not_a_parameter_of_the_noise(petab_iv: Path) -> None:
    """A parameter of the fit in a noise formula has the value of the set."""
    observable_id = _observable_ids(petab_iv)[0]
    _set_noise(petab_iv, observable_id, noiseFormula="0.1 + KI__HCTZEX_k")

    noise = PetabReader.from_yaml(petab_iv / "problem.yaml").noise_model(observable_id)
    assert noise.formula == "KI__HCTZEX_k + 0.1"
    assert noise.parameters == ()


def test_reader_reports_a_symbol_without_a_value(
    petab_iv: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A noise formula over an entity of the model is read and reported."""
    observable_id = _observable_ids(petab_iv)[0]
    _set_noise(petab_iv, observable_id, noiseFormula="0.1 * Vurine")

    problem, settings = from_petab(petab_iv / "problem.yaml")
    with caplog.at_level(logging.WARNING):
        problem.initialize(settings)
    assert "Vurine" in caplog.text
    # the fit does not need the noise, the log-likelihood does
    with pytest.raises(ValueError, match="Vurine"):
        log_likelihood(problem)


def test_the_resolved_problem_has_the_noise_models(petab_iv: Path) -> None:
    """The noise model of a fit mapping is part of the resolved problem."""
    observable_id = _observable_ids(petab_iv)[0]
    _set_noise(petab_iv, observable_id, noiseFormula="0.05")

    problem, settings = from_petab(petab_iv / "problem.yaml")
    problem.initialize(settings)
    assert len(problem.noise_models) == len(problem.mapping_keys)
    k = problem.mapping_keys.index(observable_id)
    assert problem.noise_models[k] == NoiseModel(formula="0.05")


# --- FOREIGN EXTENSIONS ---


def test_check_extensions() -> None:
    """A required extension which is not known raises, the others do not."""
    assert frozenset({EXTENSION_ID}) == KNOWN_EXTENSIONS
    assert check_extensions(None) == []
    assert check_extensions({}) == []
    assert check_extensions({EXTENSION_ID: {"required": True}}) == []
    assert check_extensions(
        {
            "tool_a": {"version": "1.0.0", "required": False},
            EXTENSION_ID: {"required": True},
            "tool_b": {"version": "1.0.0", "required": False},
        }
    ) == ["tool_a", "tool_b"]

    with pytest.raises(ValueError, match="tool_a"):
        check_extensions({"tool_a": {"version": "1.0.0", "required": True}})
    # an extension which the caller knows is not foreign
    assert check_extensions({"tool_a": {"required": True}}, known={"tool_a"}) == []


def test_check_extensions_reads_a_block_without_required_as_required() -> None:
    """A block which does not say is not ignored."""
    with pytest.raises(ValueError, match="tool_a"):
        check_extensions({"tool_a": {"version": "1.0.0"}})


def test_a_required_foreign_extension_raises(petab_iv: Path) -> None:
    """A problem which requires an extension of another tool is not read."""
    _add_extension(petab_iv, "tool_a", {"version": "1.0.0", "required": True})
    with pytest.raises(ValueError, match=r"requires the extensions.*tool_a"):
        from_petab(petab_iv / "problem.yaml")


def test_a_foreign_extension_which_is_not_required_is_ignored(
    petab_iv: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The problem is read and the log says what was ignored."""
    _add_extension(petab_iv, "tool_a", {"version": "1.0.0", "required": False})
    with caplog.at_level(logging.WARNING, logger="sbmlsim.fit.petab_v2.reader"):
        problem, settings = from_petab(petab_iv / "problem.yaml")
    messages = [r.getMessage() for r in caplog.records if "tool_a" in r.getMessage()]
    assert len(messages) == 1
    assert "ignored" in messages[0]

    problem.initialize(settings)
    assert len(problem.mapping_keys) == 4


def test_the_sciml_extension_is_reported_and_not_a_missing_module(
    petab_iv: Path,
) -> None:
    """The extensions are checked before `petab` reads their files."""
    _add_extension(
        petab_iv,
        "sciml",
        {
            "version": "0.1.0",
            "required": True,
            "array_files": [],
            "hybridization_files": [],
            "neural_networks": {},
        },
    )
    with pytest.raises(ValueError, match=r"requires the extensions.*sciml"):
        from_petab(petab_iv / "problem.yaml")
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -x tests/fit/test_petab_v2_noise.py`

Expected: FAIL while the module is collected, with `ImportError: cannot import name 'KNOWN_EXTENSIONS' from 'sbmlsim.fit.petab_v2.extension'`.

- [ ] **Step 3: Check the extensions of a problem**

**Patch `src/sbmlsim/fit/petab_v2/extension.py`:**

````diff
--- a/src/sbmlsim/fit/petab_v2/extension.py
+++ b/src/sbmlsim/fit/petab_v2/extension.py
@@ -14,6 +14,7 @@
 read and fit with the objective PEtab defines.
 """

+from collections.abc import Collection, Mapping
 from typing import Any

 from petab.v2.extensions import ExtensionConfig
@@ -21,6 +22,10 @@

 #: id of the extension, the key of the block in the YAML of the problem
 EXTENSION_ID = "sbmlsim"
+
+#: the extensions the reader interprets. A problem which requires another one
+#: is rejected, see `check_extensions`
+KNOWN_EXTENSIONS: frozenset[str] = frozenset({EXTENSION_ID})

 #: version of the extension, raised when the block changes
 EXTENSION_VERSION = "0.1.0"
@@ -97,3 +102,53 @@
     # the generic `ExtensionConfig` of a problem which was read from the YAML
     data = extension.model_dump() if hasattr(extension, "model_dump") else extension
     return SbmlsimExtension(**data)
+
+
+def check_extensions(
+    extensions: Mapping[str, Any] | None,
+    known: Collection[str] = KNOWN_EXTENSIONS,
+) -> list[str]:
+    """Check the extensions of a problem against the ones the reader knows.
+
+    PEtab says that a tool must reject a problem which requires an extension
+    it does not know and may ignore an extension which is not required (PEtab
+    v2, extensions). A block without `required` is read as required, which is
+    the safe reading of a block that does not say.
+
+    Args:
+        extensions: the blocks of the problem by the id of the extension, as
+            the dictionaries of the YAML or as the `ExtensionConfig` objects
+            of a problem which was read, `None` for a problem without
+            extensions.
+        known: ids of the extensions the reader interprets.
+
+    Returns:
+        The ids of the extensions which are to be ignored, i.e. the ones
+        which are not known and not required, in the order of the problem.
+        The reader logs them.
+
+    Raises:
+        ValueError: if the problem requires an extension which is not known.
+    """
+    required: list[str] = []
+    ignored: list[str] = []
+    for extension_id, block in (extensions or {}).items():
+        if extension_id in known:
+            continue
+        if isinstance(block, Mapping):
+            is_required = block.get("required", True)
+        else:
+            is_required = getattr(block, "required", True)
+        if is_required:
+            required.append(extension_id)
+        else:
+            ignored.append(extension_id)
+
+    if required:
+        raise ValueError(
+            f"The PEtab problem requires the extensions '{required}', which "
+            f"`sbmlsim` does not know (it knows '{sorted(known)}'). A required "
+            f"extension changes the mathematical interpretation of a problem, "
+            f"so the problem cannot be read without it."
+        )
+    return ignored
````

- [ ] **Step 4: Keep the noise model in the reader**

**Patch `src/sbmlsim/fit/petab_v2/reader.py`:**

````diff
--- a/src/sbmlsim/fit/petab_v2/reader.py
+++ b/src/sbmlsim/fit/petab_v2/reader.py
@@ -21,7 +21,9 @@
 import numpy as np
 import pandas as pd
 import petab.v2 as petab_v2
+from petab.v1.yaml import load_yaml
 from petab.v2 import Problem as PetabProblem
+from petab.v2.math import petab_math_str

 from sbmlsim.data import DataSet
 from sbmlsim.experiment import SimulationExperiment
@@ -32,10 +34,17 @@
     FitMappingCollection,
     FitParameter,
     MappingKind,
+    NoiseDistribution,
+    NoiseModel,
+    NoiseParameter,
 )
 from sbmlsim.fit.optimization import OptimizationProblem
 from sbmlsim.fit.options import FitSettings
-from sbmlsim.fit.petab_v2.extension import SbmlsimExtension, extension_of
+from sbmlsim.fit.petab_v2.extension import (
+    SbmlsimExtension,
+    check_extensions,
+    extension_of,
+)
 from sbmlsim.fit.petab_v2.observables import (
     MODEL_SUFFIX,
     add_observables,
@@ -107,9 +116,19 @@
                 only written if an observable of the problem is a formula.

         Raises:
-            ValueError: if the problem has no model or no measurements.
+            ValueError: if the problem has no model or no measurements, or if
+                it requires an extension `sbmlsim` does not know.
         """
         self.petab_problem = petab_problem
+        for extension_id in check_extensions(
+            getattr(petab_problem.config, "extensions", None)
+        ):
+            logger.warning(
+                "The extension '%s' of the PEtab problem is not known to "
+                "`sbmlsim` and is ignored, which the problem allows: it is not "
+                "required.",
+                extension_id,
+            )
         self.extension: SbmlsimExtension | None = extension_of(petab_problem.config)

         config_path = getattr(petab_problem.config, "base_path", None)
@@ -159,8 +178,15 @@

         Returns:
             The reader of the problem.
+
+        Raises:
+            ValueError: if the problem requires an extension `sbmlsim` does
+                not know. The extensions are checked on the YAML, before
+                `petab` reads the problem: it needs the package of an
+                extension to read its files.
         """
         yaml_file = Path(yaml_file)
+        check_extensions((load_yaml(yaml_file) or {}).get("extensions"))
         petab_problem = PetabProblem.from_yaml(yaml_file)
         return PetabReader(petab_problem, base_path=yaml_file.parent, name=name)

@@ -640,8 +666,115 @@
                 reference=reference,
                 observable=observable,
                 weight=info.get("weight_mapping", 1.0),
+                noise=self.noise_model(observable_id),
             )
         return mappings
+
+    def noise_model(self, observable_id: str) -> NoiseModel:
+        """Get the noise model of an observable of the problem.
+
+        The noise formula and the distribution of the observable are kept as
+        they are, with the noise parameters of its measurements as the values
+        of the placeholders. The parameters of the parameter table which the
+        formula uses and which the fit does not estimate are the parameters of
+        the noise model, with their nominal value: a parameter of the noise
+        which PEtab estimates is one of them, see the `noise-parameters` gap.
+
+        Args:
+            observable_id: id of the observable.
+
+        Returns:
+            The noise model of the fit mapping of the observable.
+
+        Raises:
+            ValueError: if the problem has no observable of the id, or if a
+                measurement does not have a value for every placeholder.
+        """
+        observables = {
+            observable.id: observable for observable in self.petab_problem.observables
+        }
+        if observable_id not in observables:
+            raise ValueError(f"The problem has no observable '{observable_id}'.")
+        observable = observables[observable_id]
+
+        placeholders = tuple(str(p) for p in observable.noise_placeholders)
+        expressions: list[Any] = [observable.noise_formula]
+        rows: list[tuple[float | str, ...]] = []
+        if placeholders:
+            for measurement in self._measurements.get(observable_id, []):
+                expressions.extend(measurement.noise_parameters)
+                rows.append(
+                    tuple(_formula(value) for value in measurement.noise_parameters)
+                )
+
+        symbols = sorted(
+            {
+                str(symbol)
+                for expression in expressions
+                for symbol in getattr(expression, "free_symbols", set())
+            }
+            - set(placeholders)
+        )
+        table = {parameter.id: parameter for parameter in self.petab_problem.parameters}
+        parameters: list[NoiseParameter] = []
+        for symbol in symbols:
+            if symbol == observable_id:
+                continue
+            parameter = table.get(symbol)
+            if parameter is None or not _is_number(parameter.nominal_value):
+                logger.warning(
+                    "The noise formula '%s' of the observable '%s' uses '%s', "
+                    "which is not a parameter of the parameter table with a "
+                    "nominal value. The log-likelihood of the problem cannot "
+                    "be calculated.",
+                    observable.noise_formula,
+                    observable_id,
+                    symbol,
+                )
+                continue
+            if self._is_fit_parameter(parameter):
+                # the value is the one of the parameter set
+                continue
+            parameters.append(
+                NoiseParameter(
+                    pid=parameter.id,
+                    value=_to_float(parameter.nominal_value),
+                    estimate=bool(parameter.estimate),
+                    lower_bound=float(parameter.lb)
+                    if parameter.lb is not None
+                    else None,
+                    upper_bound=float(parameter.ub)
+                    if parameter.ub is not None
+                    else None,
+                )
+            )
+
+        try:
+            return NoiseModel(
+                formula=str(_formula(observable.noise_formula)),
+                distribution=NoiseDistribution(str(observable.noise_distribution)),
+                placeholders=placeholders,
+                placeholder_values=tuple(rows),
+                parameters=tuple(parameters),
+                observable=observable_id if observable_id in symbols else None,
+            )
+        except ValueError as err:
+            raise ValueError(f"Observable '{observable_id}': {err}") from err
+
+    def _is_fit_parameter(self, parameter: Any) -> bool:
+        """Check whether a parameter of the problem is a parameter of the fit.
+
+        A parameter of the fit is estimated and is an entity of a model or a
+        version of one, see `_versions`. An estimated parameter which is
+        neither is a parameter of the noise or of an observable, which
+        `sbmlsim` does not fit.
+
+        Args:
+            parameter: parameter of the parameter table.
+        """
+        return bool(parameter.estimate) and (
+            parameter.id in self._versions() or self._in_model(parameter.id)
+        )

     def _selection_of(self, observable_id: str) -> str:
         """Get the selection of roadrunner which observes an observable.
@@ -970,6 +1103,21 @@
     return float(value)


+def _formula(value: Any) -> float | str:
+    """Get a value of PEtab as a number or as a formula of the math of PEtab.
+
+    Args:
+        value: sympy expression of a noise formula or of a noise parameter.
+
+    Returns:
+        The number as a `float` if the value is one, the formula as a string
+        otherwise.
+    """
+    if _is_number(value):
+        return _to_float(value)
+    return petab_math_str(value)
+
+
 def _is_number(value: Any) -> bool:
     """Check whether a value of PEtab is a finite number and not an id.

````

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/fit/test_petab_v2_noise.py`

Expected: PASS, `11 passed`.

- [ ] **Step 6: Run the tests of the PEtab layer**

Run: `uv run pytest -q -x tests/fit/test_petab_v2.py tests/fit/test_petab_v2_dosing.py tests/fit/test_petab_v2_symbols.py tests/examples/test_example_scripts.py`

Expected: PASS. The examples of `examples/petab/benchmark.py` read problems with estimated noise parameters, they run as before.

- [ ] **Step 7: Commit**

```bash
uv run ruff check && uv run ruff format --check && uvx ty check
git add tests/fit/test_petab_v2_noise.py src/sbmlsim/fit/petab_v2/extension.py src/sbmlsim/fit/petab_v2/reader.py
git commit -m "Reader keeps the noise model and rejects a required foreign extension"
```

Expected: the three checks report no diagnostics and the commit is created. The message has no attribution line.


### Task 5: The exporter writes the noise model

**Files:**
- Modify: `tests/fit/test_petab_v2_noise.py` (imports, and the tests of the export appended)
- Modify: `src/sbmlsim/fit/petab_v2/export.py:19-42` (imports), `:52-56` (`NOISE_PLACEHOLDER`), `:474-517` (`_add_observables_and_measurements`), `:521-544` (`_add_parameters`)

**Interfaces:**
- Consumes:
  - `noise_model_of(problem, k) -> NoiseModel` and `NOISE_PLACEHOLDER` of `sbmlsim.fit.petab_v2.likelihood` (Tasks 1 and 3).
  - `PetabReader.noise_model` (Task 4), through `from_petab`.
  - `petab.v2.Observable(id, name, formula, noise_formula, noise_distribution, noise_placeholders)`, `petab.v2.Measurement(..., noise_parameters)`, `petab.v2.Parameter(id, lb, ub, nominal_value, estimate)`.
- Produces:
  - `PetabExporter._noise_model(k: int, observable_id: str) -> NoiseModel`, the noise model a mapping is written with.
  - `PetabExporter._add_noise_parameters(petab_problem) -> None`.
  - `sbmlsim.fit.petab_v2.export.NOISE_PLACEHOLDER` stays importable from `export.py`, `tests/fit/test_petab_v2.py:30` imports it from there. It is defined in `likelihood.py` and re-exported with `import ... as NOISE_PLACEHOLDER`, which ruff reads as an explicit re-export.

Decisions this task implements:

- A mapping without a noise model is written as before: `noiseFormula = sd` with the placeholder `sd` and the standard deviation of every measurement, or `noiseFormula = 1.0` for data without errors. The code which did that inline is now `default_noise_model`, so the log-likelihood of a problem and of its export agree.
- The symbol of a noise formula which stands for the simulation is the id of the observable. The export names an observable `petab_id(experiment_key, mapping_key)`, which is not the id the problem was read with, so the symbol is renamed.
- A parameter of the noise is written once, with its nominal value, its bounds and `estimate` as it was read. Two noise models which give one parameter two values raise.
- A noise model which does not have the values of its placeholders for every measurement which is written raises. This happens when `initialize` dropped a measurement which is `nan`.

- [ ] **Step 1: Write the failing tests**

**Patch `tests/fit/test_petab_v2_noise.py`:**

````diff
--- a/tests/fit/test_petab_v2_noise.py
+++ b/tests/fit/test_petab_v2_noise.py
@@ -247,3 +247,80 @@
     )
     with pytest.raises(ValueError, match=r"requires the extensions.*sciml"):
         from_petab(petab_iv / "problem.yaml")
+
+
+def test_round_trip_keeps_the_noise_models(petab_iv: Path, tmp_path: Path) -> None:
+    """A problem which is read and written again has the noise it had."""
+    first, second, third = _observable_ids(petab_iv)[:3]
+    _set_noise(petab_iv, first, noiseFormula=SIGMA.pid, noiseDistribution="laplace")
+    _set_noise(petab_iv, second, noiseFormula="0.1 + 2 * sd", noisePlaceholders="sd")
+    _set_noise(petab_iv, third, noiseFormula=f"0.01 + 0.1 * {third}")
+    _add_sigma(petab_iv)
+
+    def edit(df: pd.DataFrame) -> None:
+        rows = df.index[df["observableId"] == second]
+        df.loc[rows, "noiseParameters"] = ["0.25", SIGMA.pid][: len(rows)]
+
+    _edit(petab_iv / "measurements.tsv", edit)
+
+    problem, settings = from_petab(petab_iv / "problem.yaml", opid="noise")
+    problem.initialize(settings)
+    llh = log_likelihood(problem)
+
+    yaml_file = to_petab(problem, tmp_path / "again", settings=settings)
+    again, settings_again = from_petab(yaml_file, opid="noise")
+    again.initialize(settings_again)
+
+    assert len(again.noise_models) == len(problem.noise_models)
+    for k, noise in enumerate(problem.noise_models):
+        assert noise is not None
+        written = again.noise_models[k]
+        assert written is not None
+        assert written.distribution is noise.distribution
+        assert written.placeholders == noise.placeholders
+        assert written.placeholder_values == noise.placeholder_values
+        assert written.parameters == noise.parameters
+        if noise.observable is None:
+            assert written.formula == noise.formula
+        else:
+            # the observable is written under the id of its fit mapping
+            assert written.observable == again.mapping_keys[k]
+            assert written.observable in written.formula
+
+    # the parameter of the noise is written once, as it was read
+    parameters = pd.read_csv(yaml_file.parent / "parameters.tsv", sep="\t")
+    sigma = parameters[parameters["parameterId"] == SIGMA.pid]
+    assert len(sigma) == 1
+    assert sigma["nominalValue"].iloc[0] == SIGMA.value
+    assert bool(sigma["estimate"].iloc[0]) is True
+    assert sigma["lowerBound"].iloc[0] == SIGMA.lower_bound
+
+    assert log_likelihood(again) == pytest.approx(llh, rel=1e-6)
+
+
+def test_round_trip_keeps_the_default_noise(
+    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings, tmp_path: Path
+) -> None:
+    """A problem without noise models has the same log-likelihood when read."""
+    op_hctz_pk.initialize(fit_settings)
+    yaml_file = to_petab(op_hctz_pk, tmp_path, settings=fit_settings)
+
+    problem, settings = from_petab(yaml_file)
+    problem.initialize(settings)
+    assert all(noise is not None for noise in problem.noise_models)
+    # up to the selections of the tasks, see the `selections` gap
+    assert log_likelihood(problem) == pytest.approx(
+        log_likelihood(op_hctz_pk), rel=1e-4
+    )
+
+
+def test_export_requires_a_value_per_measurement(
+    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings, tmp_path: Path
+) -> None:
+    """A noise model which does not cover the data is not written."""
+    op_hctz_iv.initialize(fit_settings)
+    op_hctz_iv.noise_models[0] = NoiseModel(
+        formula="sd", placeholders=("sd",), placeholder_values=((0.5,),)
+    )
+    with pytest.raises(ValueError, match="placeholders"):
+        to_petab(op_hctz_iv, tmp_path, settings=fit_settings)
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -x tests/fit/test_petab_v2_noise.py`

Expected: FAIL. The run stops at the first of two failures, which of them is first depends on the workers: `test_round_trip_keeps_the_noise_models` fails with `assert <NoiseDistribution.NORMAL: 'normal'> is <NoiseDistribution.LAPLACE: 'laplace'>` and `test_export_requires_a_value_per_measurement` with `DID NOT RAISE`. Without `-x` the result is `2 failed, 12 passed`: `test_round_trip_keeps_the_default_noise` passes already, the export writes the default noise today.

- [ ] **Step 3: Write the noise model and its parameters**

**Patch `src/sbmlsim/fit/petab_v2/export.py`:**

````diff
--- a/src/sbmlsim/fit/petab_v2/export.py
+++ b/src/sbmlsim/fit/petab_v2/export.py
@@ -16,6 +16,7 @@
 `sbmlsim.fit.petab_v2.gaps`.
 """

+import dataclasses
 import logging
 import re
 import shutil
@@ -25,10 +26,12 @@

 import numpy as np
 import petab.v2 as petab_v2
+import sympy as sp
 from petab.models.sbml_model import SbmlModel  # ty: ignore[unresolved-import]
 from petab.v2 import Problem as PetabProblem
-
-from sbmlsim.fit.objects import EVALUATED_KINDS, MappingKind
+from petab.v2.math import petab_math_str, sympify_petab
+
+from sbmlsim.fit.objects import EVALUATED_KINDS, MappingKind, NoiseModel
 from sbmlsim.fit.optimization import OptimizationProblem
 from sbmlsim.fit.options import FitSettings, WeightingCurvesType
 from sbmlsim.fit.parameter_mapping import has_renamed_targets
@@ -37,6 +40,10 @@
     SbmlsimExtension,
 )
 from sbmlsim.fit.petab_v2.gaps import Gap, GapKind, gaps_dict, gaps_of_problem
+from sbmlsim.fit.petab_v2.likelihood import (
+    NOISE_PLACEHOLDER as NOISE_PLACEHOLDER,
+)
+from sbmlsim.fit.petab_v2.likelihood import noise_model_of
 from sbmlsim.fit.petab_v2.symbols import condition_target, observable_formula
 from sbmlsim.simulation.timecourse import Timecourse, TimecourseSim
 from sbmlsim.units import Quantity
@@ -48,12 +55,6 @@

 #: name of the YAML file of the problem
 YAML_FILE = "problem.yaml"
-
-#: the placeholder of the noise of a measurement, i.e. the standard deviation
-#: of its data. PEtab v2 declares the placeholders of an observable in
-#: `noisePlaceholders`; the `noiseParameter${n}_${observableId}` names of v1
-#: are gone
-NOISE_PLACEHOLDER = "sd"

 #: the file of every table of the problem, `to_files` skips a table without one
 TABLE_FILES: dict[str, str] = {
@@ -480,21 +481,19 @@
             )
             self.observable_ids[k] = observable_id

-            errors = problem.y_errors[k]
-            # the noise of a measurement is its standard deviation, which the
-            # noise parameter of the measurement fills in. PEtab v2 declares the
-            # placeholders of an observable, the `noiseParameter${n}_${id}`
-            # names of v1 are gone
-            placeholders = [NOISE_PLACEHOLDER] if errors is not None else []
-            noise_formula = NOISE_PLACEHOLDER if errors is not None else "1.0"
+            # the noise model of the mapping, which is the one it was read
+            # with or the standard deviation of its data: the noise parameter
+            # of a measurement fills in the placeholder the observable declares
+            noise = self._noise_model(k, observable_id)
             sbml_model = self.sbml_models.get(self.model_ids[id(problem.models[k])])
             _table(petab_problem, "observable_tables").observables.append(
                 petab_v2.Observable(
                     id=observable_id,
                     name=f"{problem.experiment_keys[k]}.{problem.mapping_keys[k]}",
                     formula=observable_formula(problem.yid_observable[k], sbml_model),
-                    noise_formula=noise_formula,
-                    noise_placeholders=placeholders,
+                    noise_formula=noise.formula,
+                    noise_distribution=noise.distribution.value,
+                    noise_placeholders=list(noise.placeholders),
                 )
             )

@@ -510,11 +509,48 @@
                         time=float(times[i]),
                         measurement=float(values[i]),
                         observable_parameters=[],
-                        noise_parameters=[float(errors[i])]
-                        if errors is not None
+                        noise_parameters=list(noise.placeholder_values[i])
+                        if noise.placeholders
                         else [],
                     )
                 )
+
+    def _noise_model(self, k: int, observable_id: str) -> NoiseModel:
+        """Get the noise model a fit mapping is written with.
+
+        Args:
+            k: index of the fit mapping.
+            observable_id: id of the observable the mapping is written as. The
+                symbol of the noise formula which stands for the simulation is
+                the id of the observable, so it is renamed to this id.
+
+        Returns:
+            The noise model of the mapping, see
+            `sbmlsim.fit.petab_v2.likelihood.noise_model_of`.
+
+        Raises:
+            ValueError: if the noise model does not have the values of its
+                placeholders for every measurement which is written.
+        """
+        problem = self.problem
+        noise = noise_model_of(problem, k)
+        size = len(problem.y_references[k])
+        if noise.placeholders and len(noise.placeholder_values) != size:
+            raise ValueError(
+                f"'{problem.opid}': the noise model of the fit mapping "
+                f"'{problem.mapping_keys[k]}' has '{len(noise.placeholder_values)}' "
+                f"values of its placeholders '{list(noise.placeholders)}' for "
+                f"'{size}' measurements."
+            )
+        if noise.observable is None or noise.observable == observable_id:
+            return noise
+        expression = sympify_petab(noise.formula).subs(
+            sp.Symbol(noise.observable, real=True),
+            sp.Symbol(observable_id, real=True),
+        )
+        return dataclasses.replace(
+            noise, formula=petab_math_str(expression), observable=observable_id
+        )

     # --- PARAMETERS ---

@@ -542,6 +578,44 @@
                     estimate=True,
                 )
             )
+        self._add_noise_parameters(petab_problem)
+
+    def _add_noise_parameters(self, petab_problem: PetabProblem) -> None:
+        """Add the parameters of the noise models, every one of them once.
+
+        A parameter of a noise formula is a row of the parameter table, with
+        the nominal value the log-likelihood uses. It is written as estimated
+        if the problem it was read from estimates it, which `sbmlsim` does not
+        do, see the `noise-parameters` gap.
+
+        Raises:
+            ValueError: if two noise models give one parameter two values.
+        """
+        written: dict[str, float] = {}
+        for k in self.indices:
+            for parameter in noise_model_of(self.problem, k).parameters:
+                if parameter.pid in self.problem.pids:
+                    # a parameter of the fit, which is written already
+                    continue
+                if parameter.pid in written:
+                    if written[parameter.pid] != parameter.value:
+                        raise ValueError(
+                            f"'{self.problem.opid}': the parameter "
+                            f"'{parameter.pid}' of the noise has the values "
+                            f"'{written[parameter.pid]}' and '{parameter.value}' "
+                            f"in the noise models of two fit mappings."
+                        )
+                    continue
+                written[parameter.pid] = parameter.value
+                _table(petab_problem, "parameter_tables").parameters.append(
+                    petab_v2.Parameter(
+                        id=parameter.pid,
+                        lb=parameter.lower_bound,
+                        ub=parameter.upper_bound,
+                        nominal_value=parameter.value,
+                        estimate=parameter.estimate,
+                    )
+                )

     # --- EXTENSION ---

````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/fit/test_petab_v2_noise.py tests/fit/test_petab_v2.py tests/fit/test_petab_v2_dosing.py`

Expected: PASS, `40 passed` (14 of the noise, 20 of `test_petab_v2.py`, 6 of the dosing). `test_noise_placeholder_is_declared` and `test_exported_problem_is_valid_petab` of `test_petab_v2.py` are the tests which say that the export of the HCTZ problem did not change.

- [ ] **Step 5: Commit**

```bash
uv run ruff check && uv run ruff format --check && uvx ty check
git add tests/fit/test_petab_v2_noise.py src/sbmlsim/fit/petab_v2/export.py
git commit -m "Export writes the noise model of a fit mapping"
```

Expected: the three checks report no diagnostics and the commit is created. The message has no attribution line.


### Task 6: The gaps and the documentation

**Files:**
- Modify: `tests/fit/test_petab_v2_noise.py` (imports, and the tests of the gaps appended)
- Modify: `src/sbmlsim/fit/petab_v2/gaps.py:184-196` (`noise-parameters`, two new gaps after it), `:332` (`gaps_of_problem`)
- Modify: `docs/petab.md:74-75`, `:81-83`, `:118`
- Create: `docs/api/fit.petab_v2.likelihood.md`
- Modify: `zensical.toml:101`
- Modify: `CLAUDE.md:54` (the paragraph of `fit/`)

**Interfaces:**
- Consumes: `OptimizationProblem.noise_models` (Task 2), `NoiseParameter.estimate` (Task 1), everything the documentation names (Tasks 1 to 5).
- Produces: the gaps `noise-model` (`GapKind.LOSSY`) and `foreign-extension` (`GapKind.UNSUPPORTED`) in `GAPS` and `GAPS_BY_ID`. `gaps_of_problem` reports `noise-model` for a problem with a noise model and `noise-parameters` for one whose noise model has a parameter with `estimate=True`. `foreign-extension` is never reported for a problem: a problem which hits it is not read.

- [ ] **Step 1: Write the failing tests**

**Patch `tests/fit/test_petab_v2_noise.py`:**

````diff
--- a/tests/fit/test_petab_v2_noise.py
+++ b/tests/fit/test_petab_v2_noise.py
@@ -12,12 +12,13 @@
 from sbmlsim.fit import FitSettings
 from sbmlsim.fit.objects import NoiseDistribution, NoiseModel, NoiseParameter
 from sbmlsim.fit.optimization import OptimizationProblem
-from sbmlsim.fit.petab_v2 import to_petab
+from sbmlsim.fit.petab_v2 import GapKind, gaps_of_problem, to_petab
 from sbmlsim.fit.petab_v2.extension import (
     EXTENSION_ID,
     KNOWN_EXTENSIONS,
     check_extensions,
 )
+from sbmlsim.fit.petab_v2.gaps import GAPS_BY_ID
 from sbmlsim.fit.petab_v2.likelihood import log_likelihood
 from sbmlsim.fit.petab_v2.reader import PetabReader, from_petab

@@ -324,3 +325,29 @@
     )
     with pytest.raises(ValueError, match="placeholders"):
         to_petab(op_hctz_iv, tmp_path, settings=fit_settings)
+
+
+# --- THE GAPS ---
+
+
+def test_gaps_of_the_noise(petab_iv: Path) -> None:
+    """A problem with a noise model runs into the gaps of the noise."""
+    observable_id = _observable_ids(petab_iv)[0]
+    _set_noise(petab_iv, observable_id, noiseFormula=SIGMA.pid)
+    _add_sigma(petab_iv)
+
+    problem, settings = from_petab(petab_iv / "problem.yaml")
+    problem.initialize(settings)
+    ids = {gap.id for gap in gaps_of_problem(problem)}
+    assert {"noise-model", "noise-parameters"} <= ids
+
+
+def test_a_problem_without_noise_has_no_gap_of_it(
+    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
+) -> None:
+    """The gaps of the noise are the ones of a problem which has a noise."""
+    op_hctz_iv.initialize(fit_settings)
+    ids = {gap.id for gap in gaps_of_problem(op_hctz_iv)}
+    assert not {"noise-model", "noise-parameters", "foreign-extension"} & ids
+    assert GAPS_BY_ID["noise-model"].kind is GapKind.LOSSY
+    assert GAPS_BY_ID["foreign-extension"].kind is GapKind.UNSUPPORTED
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -x tests/fit/test_petab_v2_noise.py`

Expected: FAIL. The run stops at the first of two failures, which of them is first depends on the workers: `test_gaps_of_the_noise` fails with `'noise-model'` and `'noise-parameters'` as the extra items of the left set, `test_a_problem_without_noise_has_no_gap_of_it` with `KeyError: 'noise-model'`. Without `-x` the result is `2 failed, 14 passed`.

- [ ] **Step 3: Add the gaps**

**Patch `src/sbmlsim/fit/petab_v2/gaps.py`:**

````diff
--- a/src/sbmlsim/fit/petab_v2/gaps.py
+++ b/src/sbmlsim/fit/petab_v2/gaps.py
@@ -192,7 +192,38 @@
         detail="a parameter of a problem which is not an entity of a model is "
         "not fitted and the reader says which; the data of its observable is "
         "weighted by `FitSettings.weighting_points` instead. A fit of such a "
-        "problem is therefore not the fit PEtab describes",
+        "problem is therefore not the fit PEtab describes. A parameter of a "
+        "noise formula is kept in the noise model of its fit mapping with its "
+        "nominal value, its bounds and whether the problem estimates it, so it "
+        "is written as it was read; `log_likelihood` evaluates it at the value "
+        "of the parameter set it is given and at the nominal value without one",
+    ),
+    Gap(
+        id="noise-model",
+        kind=GapKind.LOSSY,
+        sbmlsim="the cost of a fit is a weighted sum of squares. The noise "
+        "formula and the noise distribution of a fit mapping are kept as its "
+        "`NoiseModel`, which `log_likelihood` evaluates and the optimizer does "
+        "not use",
+        petab="`noiseFormula` and `noiseDistribution` per observable are the "
+        "objective, i.e. the negative log likelihood of the measurements",
+        detail="the noise model of a problem which is read is written as it "
+        "was read, so the tables of a round trip agree and the log-likelihood "
+        "of a problem is compared with the one of other tools. The fit stays a "
+        "least squares fit, i.e. its optimum is not the maximum of the "
+        "likelihood. A fit mapping without a noise model is written with a "
+        "normal noise of the standard deviation of its data, or of `1.0` for "
+        "data without errors, and has that noise model when it is read back",
+    ),
+    Gap(
+        id="foreign-extension",
+        kind=GapKind.UNSUPPORTED,
+        sbmlsim="the reader interprets the `sbmlsim` extension of a problem",
+        petab="a problem carries the extensions of any tool, and `required` "
+        "says whether it can be interpreted without one of them",
+        detail="a problem which requires an extension `sbmlsim` does not know "
+        "is not read, the reader raises and names the extension. An extension "
+        "which is not required is ignored with a message in the log",
     ),
     Gap(
         id="x-observable",
@@ -328,6 +359,13 @@
                 hits.add("presimulation")
             if (tc.model_changes or tc.model_manipulations) and k >= 0:
                 hits.add("model-changes")
+
+    for noise in problem.noise_models:
+        if noise is None:
+            continue
+        hits.add("noise-model")
+        if any(parameter.estimate for parameter in noise.parameters):
+            hits.add("noise-parameters")

     for k, xid in enumerate(problem.xid_observable):
         if xid != "time":
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/fit/test_petab_v2_noise.py tests/fit/test_petab_v2.py`

Expected: PASS, `36 passed`. `test_gaps_are_documented` of `test_petab_v2.py` checks that the new gaps have an id of their own and all their texts.

- [ ] **Step 5: Write the documentation**

Every paragraph, list item and table row is one line. The section "The log-likelihood" and the section "Extensions of other tools" go in front of "The example".

**Patch `docs/petab.md`:**

````diff
--- a/docs/petab.md
+++ b/docs/petab.md
@@ -71,16 +71,72 @@
 A gap is of one of three kinds:

 - **extension**: PEtab has no place for it and the extension carries it, i.e. the units, the settings of the fit, the kind of every mapping, the output grid of the timecourses, the metadata of a curve and the settings of the integrator. The scale the optimizer searches in is part of the settings, which is where PEtab v2 puts it as well: it removed the `parameterScale` of its parameter table because the scale is a property of the optimization and not of the problem, so the bounds and the start values are written on the linear scale. The round trip through `sbmlsim` is exact, a tool which reads the problem without the extension gets a valid PEtab problem which does not know these things.
-- **lossy**: the information is transformed. The weights of `sbmlsim` are not the standard deviation PEtab uses as the noise, a pre-simulation of a finite duration is not the pre-equilibration of PEtab, and the reader builds one simulation experiment for a problem, so a task selects the observables of the whole problem rather than those of the experiment a measurement came from.
-- **unsupported**: the export raises. A structural model change (`ModelChange.clamp_species`), an observable which is a python function and a mapping whose x is not the time of the simulation have no PEtab representation.
+- **lossy**: the information is transformed. The noise model of PEtab is its objective and is only evaluated by `sbmlsim`, the weights of `sbmlsim` are not the standard deviation PEtab uses as the noise, a pre-simulation of a finite duration is not the pre-equilibration of PEtab, and the reader builds one simulation experiment for a problem, so a task selects the observables of the whole problem rather than those of the experiment a measurement came from.
+- **unsupported**: the export raises. A structural model change (`ModelChange.clamp_species`), an observable which is a python function and a mapping whose x is not the time of the simulation have no PEtab representation. The reader raises for a problem which requires the extension of another tool.

 The round trip of the HCTZ example keeps the settings, the parameters with their units, the mappings with their kinds and the reference data of every mapping, and its cost agrees to `7e-6`, which is the `selections` gap above.

 A parameter which is estimated separately for parts of the data is a condition of PEtab: the condition assigns the entity of the model the value of the estimated parameter, and the experiments of the subset reference it. The selector which chose the subset is a python callable and is not written; PEtab stores the resolution, so a problem which is read back selects the same fit mappings by their id. The fit, its cost and its parameters are the same, i.e. the round trip is exact in effect and not in source form.

+## The log-likelihood
+
+PEtab defines the objective of a problem as the likelihood of its measurements under a noise model, and `sbmlsim` fits by weighted least squares. `sbmlsim.fit.petab_v2.likelihood` calculates the log-likelihood of a problem for its evaluation, e.g. to compare a parameter set with the result of another tool. The optimizer does not use it.
+
+```py
+from dataclasses import replace
+from pathlib import Path
+
+from sbmlsim.fit.petab_v2 import from_petab, gradient, log_likelihood
+
+problem, settings = from_petab(Path("results") / "petab" / "problem.yaml")
+problem.initialize(settings)
+
+llh = log_likelihood(problem)
+llh_fit = log_likelihood(problem, parameters=opt_result.parameter_set())
+
+# a gradient needs an integrator which is more exact than its step
+problem.initialize(
+    replace(
+        settings,
+        variable_step_size=False,
+        absolute_tolerance=1e-12,
+        relative_tolerance=1e-10,
+    )
+)
+grad = gradient(problem)
+```
+
+`log_likelihood` simulates the problem at the parameters and sums the log density of every measurement of the training data; the validation data and the outliers do not enter, and neither do the residual, the weights and the loss function of the `FitSettings`. Without parameters it is evaluated at the nominal values, i.e. the start values of the parameters, which are the `nominalValue` of the parameter table of a problem which was read. The simulation `y` is the median of the distribution of the measurement `m` and the noise formula gives its scale `σ`, which are the definitions of PEtab v2:
+
+| `noiseDistribution` | log density of a measurement |
+| --- | --- |
+| `normal` | `-0.5 log(2π σ²) - 0.5 ((m - y) / σ)²` |
+| `log-normal` | `-0.5 log(2π σ² m²) - 0.5 ((log m - log y) / σ)²` |
+| `laplace` | `-log(2σ) - abs(m - y) / σ` |
+| `log-laplace` | `-log(2σ m) - abs(log m - log y) / σ` |
+
+The reader keeps the noise formula and the noise distribution of every observable as the `NoiseModel` of its fit mapping, `problem.noise_models` holds them after `initialize`, and the export writes them back, so a round trip keeps the noise. The symbols of a noise formula are resolved as follows:
+
+| symbol | value |
+| --- | --- |
+| a placeholder of `noisePlaceholders` | the `noiseParameters` of the measurement, a number or a formula of parameters |
+| the id of the observable | the simulation at the measurement |
+| a parameter of the fit | the value of the parameter set |
+| another parameter of the parameter table | the value of the parameter set if it has one, the `nominalValue` otherwise |
+
+A parameter of the noise which the problem estimates is therefore evaluated and not estimated, which is the `noise-parameters` gap: `log_likelihood(problem, ParameterSet(sid="fit", values={..., "sd_obs": 0.1}))` gives the log-likelihood another tool reports for its estimate. A noise formula over anything else, e.g. a species of the model, is read with a warning and `log_likelihood` raises for it. A fit mapping which has no noise model, i.e. every mapping of a problem which is defined in python, has a normal noise with the standard deviation of its reference data and with the scale `1.0` for data without errors, which is what the export writes for it. A `NoiseModel` is given to a `FitMapping` with its `noise` argument.
+
+The log-likelihood is the one of the measurements, so a problem whose residuals are relative to the baseline of a curve (`ABSOLUTE_TO_BASELINE`, `NORMALIZED_TO_BASELINE`) has none and `log_likelihood` raises. The logarithmic distributions require positive measurements and simulations.
+
+`gradient` is the central finite difference of the log-likelihood on the linear scale, with the step `step * max(|x|, 1)` for every parameter of the fit, and returns a `pandas.Series` indexed by the ids of the parameters. A difference divides the error of a simulation by the step. With `variable_step_size=True` the data is interpolated on the steps of the integrator, which differ between two simulations, so the simulations of one problem differ by `1e-6` however tight the tolerances are and the gradient is noise; `gradient` logs a warning in this case.
+
+## Extensions of other tools
+
+A problem carries the extensions of any tool in the `extensions` block of its YAML, and `required` says whether the problem can be interpreted without one of them. The reader knows the `sbmlsim` extension. A problem which requires another extension is not read: `from_petab` raises a `ValueError` which names the extension, before `petab` reads the files of the problem. An extension which is not required is ignored and the log says so.
+
 ## The example

-`examples/hctz_fitting/fitting/petab_problem.py` runs the layer on the reference problem, i.e. it reports the gaps of the fit, writes it, validates the problem with `petab`, lists which collection every experiment came from and reads the fit back:
+`examples/hctz_fitting/fitting/petab_problem.py` runs the layer on the reference problem, i.e. it reports the gaps of the fit, writes it, validates the problem with `petab`, lists which collection every experiment came from, reads the fit back and compares the cost and the log-likelihood of the two:

 ```bash
 python -m examples.hctz_fitting.fitting.petab_problem
@@ -115,7 +171,7 @@

 A simulation experiment which is read from a PEtab problem is created when the problem is read, and a class which is created cannot be pickled, so such a fit runs in one process: `n_cores=1`, which is what the example uses.

-Two things of the collection do not survive the conversion, and the example shows both. The parameter table of v1 has a `parameterScale`, which v2 removed and which is `FitSettings.parameter_scale` here, and the observable of the problem has a `log10-normal` noise distribution which the converter maps to `log-normal`; `sbmlsim` has no log noise, so the example fits the relative residuals (`ResidualType.NORMALIZED`) which describe a viral load over orders of magnitude in the same spirit. The problem also estimates the standard deviation `sd_task0_model0_perelson1_V` of its observable, which is not an entity of the model: `sbmlsim` fits the parameters of a model and weights the data, so it is not fitted and the reader says so.
+Two things of the collection do not survive the conversion, and the example shows both. The parameter table of v1 has a `parameterScale`, which v2 removed and which is `FitSettings.parameter_scale` here, and the observable of the problem has a `log10-normal` noise distribution which the converter maps to `log-normal`; the noise of an observable is what the log-likelihood is calculated with and not what `sbmlsim` fits, so the example fits the relative residuals (`ResidualType.NORMALIZED`) which describe a viral load over orders of magnitude in the same spirit. The problem also estimates the standard deviation `sd_task0_model0_perelson1_V` of its observable, which is not an entity of the model: `sbmlsim` fits the parameters of a model and weights the data, so it is not fitted and the reader says so.

 ## COMBINE archives

````

**Create `docs/api/fit.petab_v2.likelihood.md`:**

````markdown
# fit.petab_v2.likelihood

::: sbmlsim.fit.petab_v2.likelihood
````

**Patch `zensical.toml`:**

````diff
--- a/zensical.toml
+++ b/zensical.toml
@@ -99,6 +99,7 @@
       { "petab_v2.symbols" = "api/fit.petab_v2.symbols.md" },
       { "petab_v2.observables" = "api/fit.petab_v2.observables.md" },
       { "petab_v2.gaps" = "api/fit.petab_v2.gaps.md" },
+      { "petab_v2.likelihood" = "api/fit.petab_v2.likelihood.md" },
     ] },
     { "sbmlsim.sensitivity" = [
       { "analysis" = "api/sensitivity.analysis.md" },
````

**Insert into `CLAUDE.md`:** the paragraph which starts with ``**`fit/` `` is one line. Find this sentence in it, it occurs once:

````text
`gaps.py` is the catalogue of the differences with `gaps_of_problem` reporting the ones a given problem hits.
````

Insert the following text directly after it, separated from it and from the sentence which follows by one space. Change nothing else in the file, also not the characters of the lines this plan does not touch.

````text
`likelihood.py` is the log-likelihood PEtab defines as the objective, calculated for the evaluation of a problem and not used by the optimizer: `log_density` (the `normal`, `log-normal`, `laplace` and `log-laplace` density of PEtab v2, with the simulation as the median) and `noise_values` (the noise formula evaluated with sympy) are the functions of arrays, `log_likelihood(problem, parameters)` sums the log density over the training data at a parameter set, the nominal values by default, and `gradient` is its central finite difference, which needs `variable_step_size=False` and tight tolerances. The noise formula and the noise distribution of an observable are the `NoiseModel` of its `FitMapping` (`fit/objects.py`), which `initialize` collects as `problem.noise_models`, the reader fills and the exporter writes back; a mapping without one has a normal noise with the standard deviation of its data. `OptimizationProblem.predictions(x, indices)` is the simulation at the reference data. `extension.check_extensions` rejects a problem which requires the extension of another tool and the reader logs the ones it ignores.
````

- [ ] **Step 6: Build the documentation**

```bash
uv run zensical build --clean
grep -c "log-laplace" site/petab/index.html
ls site/api/fit.petab_v2.likelihood/index.html
```

Expected: the build ends with `No issues found`, the count is at least `1` and the page of the API exists. `site/` is ignored by git.

- [ ] **Step 7: Check the files of this plan for the em dash**

```bash
grep -l $'\u2014' src/sbmlsim/fit/objects.py src/sbmlsim/fit/__init__.py src/sbmlsim/fit/optimization.py src/sbmlsim/fit/petab_v2/*.py examples/hctz_fitting/fitting/petab_problem.py tests/fit/test_petab_v2_likelihood.py tests/fit/test_predictions.py tests/fit/test_petab_v2_noise.py tests/data/petab/sciml_001_llh.tsv docs/petab.md docs/api/fit.petab_v2.likelihood.md zensical.toml
grep -o '`likelihood.py` is the log-likelihood.*the ones it ignores\.' CLAUDE.md | grep -c $'\u2014'
```

Expected: the first command prints nothing, the second prints `0`. `CLAUDE.md` has the character in text this plan does not touch, which is why only the inserted text is checked.

- [ ] **Step 8: Run everything**

```bash
uv run pytest -q
uv run ruff check
uv run ruff format --check
uvx ty check
```

Expected: no failure and no error in the tests (the cases of the SBML Test Suite are deselected), and no diagnostic of the three checks.

- [ ] **Step 9: Commit**

```bash
uv run ruff check && uv run ruff format --check && uvx ty check
git add tests/fit/test_petab_v2_noise.py src/sbmlsim/fit/petab_v2/gaps.py docs/petab.md docs/api/fit.petab_v2.likelihood.md zensical.toml CLAUDE.md
git commit -m "Gaps and documentation of the log-likelihood and the noise model"
```

Expected: the three checks report no diagnostics and the commit is created. The message has no attribution line.


---

## Done when

- `uv run pytest -q -x tests/fit/test_petab_v2_likelihood.py tests/fit/test_predictions.py tests/fit/test_petab_v2_noise.py` passes, `67 passed`.
- `log_likelihood` of the HCTZ reference problem is a finite number (`test_the_hctz_problem_has_a_log_likelihood`, and the row `log-likelihood` of `python -m examples.hctz_fitting.fitting.petab_problem`).
- `uv run pytest -q`, `uv run ruff check`, `uv run ruff format --check` and `uvx ty check` are clean.

## Where this plan goes beyond the text of the spec

| spec | plan | reason |
| --- | --- | --- |
| the noise model is "kept in the resolved problem" | `NoiseModel` in `fit/objects.py`, `FitMapping.noise`, `problem.noise_models` | the resolved problem is built from the fit mappings, and a class of `petab_v2` in `objects.py` is a circular import |
| `log_likelihood` "simulates the problem" | `OptimizationProblem.predictions`, `residuals` uses the shared `_interpolate` | `residuals(complete_data=True)` simulates the validation data and the outliers as well and subtracts the baseline |
| the noise formula is "a number, a parameter of the parameter table or a formula of them" | the id of the observable is also a symbol, it is the simulation | `petab.v2.calculate.evaluate_noise_formula` substitutes it, and a noise which is proportional to the simulation is common |
| the gradient, risk "tight tolerances of the integrator" | `variable_step_size=False` is needed as well, `gradient` warns without it | with a variable step size two simulations of one problem differ by up to `3e-6` at tolerances of `1e-10` |
| the gaps of the spec are the ones of SciML | `noise-model`, `foreign-extension` and the detail of `noise-parameters` | the task of phase 2 asks for the matching entries |
| a foreign extension "raises" in the reader | `from_yaml` checks the YAML before `petab` reads the problem | `petab` fails on a `sciml` block with a `ModuleNotFoundError` first |
| silent | a residual to the baseline has no log-likelihood, `log_likelihood` raises | `initialize` shifts the data of such a problem |
