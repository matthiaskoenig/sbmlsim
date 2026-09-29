"""Tests of the noise model and the extensions of a PEtab v2 problem."""

import dataclasses
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
from sbmlsim.fit.petab_v2 import GapKind, gaps_of_problem, to_petab
from sbmlsim.fit.petab_v2.extension import (
    EXTENSION_ID,
    KNOWN_EXTENSIONS,
    check_extensions,
)
from sbmlsim.fit.petab_v2.gaps import GAPS_BY_ID
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


def test_a_fit_parameter_without_a_nominal_value_is_not_reported(
    petab_iv: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The value of a parameter of the fit is the one of the parameter set."""
    observable_id = _observable_ids(petab_iv)[0]
    _set_noise(petab_iv, observable_id, noiseFormula="0.1 + KI__HCTZEX_k")

    def edit(df: pd.DataFrame) -> None:
        df.loc[df["parameterId"] == "KI__HCTZEX_k", "nominalValue"] = ""

    _edit(petab_iv / "parameters.tsv", edit)

    reader = PetabReader.from_yaml(petab_iv / "problem.yaml")
    with caplog.at_level(logging.WARNING, logger="sbmlsim.fit.petab_v2.reader"):
        noise = reader.noise_model(observable_id)
    assert noise.parameters == ()
    assert "KI__HCTZEX_k" not in caplog.text


def test_reader_reports_a_symbol_without_a_value(
    petab_iv: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A noise formula over an entity of the model is read and reported once."""
    observable_id = _observable_ids(petab_iv)[0]
    _set_noise(petab_iv, observable_id, noiseFormula="0.1 * Vurine")

    with caplog.at_level(logging.WARNING, logger="sbmlsim.fit.petab_v2.reader"):
        problem, settings = from_petab(petab_iv / "problem.yaml")
        problem.initialize(settings)
        # the mappings are resolved again for other settings
        problem.initialize(dataclasses.replace(settings, absolute_tolerance=1e-12))
    messages = [r.getMessage() for r in caplog.records if "Vurine" in r.getMessage()]
    assert len(messages) == 1
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

    with pytest.raises(ValueError, match="tool_a") as excinfo:
        check_extensions(
            {
                "tool_a": {"version": "1.0.0", "required": True},
                "tool_b": {"version": "1.0.0", "required": True},
            }
        )
    # the ids are listed, not the representation of a list
    assert "'tool_a, tool_b'" in str(excinfo.value)
    assert f"'{EXTENSION_ID}'" in str(excinfo.value)
    assert "[" not in str(excinfo.value)
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


def test_round_trip_keeps_the_noise_models(petab_iv: Path, tmp_path: Path) -> None:
    """A problem which is read and written again has the noise it had."""
    first, second, third = _observable_ids(petab_iv)[:3]
    _set_noise(petab_iv, first, noiseFormula=SIGMA.pid, noiseDistribution="laplace")
    _set_noise(petab_iv, second, noiseFormula="0.1 + 2 * sd", noisePlaceholders="sd")
    _set_noise(petab_iv, third, noiseFormula=f"0.01 + 0.1 * {third}")
    _add_sigma(petab_iv)

    def edit(df: pd.DataFrame) -> None:
        rows = df.index[df["observableId"] == second]
        df.loc[rows, "noiseParameters"] = ["0.25", SIGMA.pid][: len(rows)]

    _edit(petab_iv / "measurements.tsv", edit)

    problem, settings = from_petab(petab_iv / "problem.yaml", opid="noise")
    problem.initialize(settings)
    llh = log_likelihood(problem)

    yaml_file = to_petab(problem, tmp_path / "again", settings=settings)
    again, settings_again = from_petab(yaml_file, opid="noise")
    again.initialize(settings_again)

    assert len(again.noise_models) == len(problem.noise_models)
    for k, noise in enumerate(problem.noise_models):
        assert noise is not None
        written = again.noise_models[k]
        assert written is not None
        assert written.distribution is noise.distribution
        assert written.placeholders == noise.placeholders
        assert written.placeholder_values == noise.placeholder_values
        assert written.parameters == noise.parameters
        if noise.observable is None:
            assert written.formula == noise.formula
        else:
            # the observable is written under the id of its fit mapping
            assert written.observable == again.mapping_keys[k]
            assert written.observable in written.formula

    # the parameter of the noise is written once, as it was read
    parameters = pd.read_csv(yaml_file.parent / "parameters.tsv", sep="\t")
    sigma = parameters[parameters["parameterId"] == SIGMA.pid]
    assert len(sigma) == 1
    assert sigma["nominalValue"].iloc[0] == SIGMA.value
    assert bool(sigma["estimate"].iloc[0]) is True
    assert sigma["lowerBound"].iloc[0] == SIGMA.lower_bound

    assert log_likelihood(again) == pytest.approx(llh, rel=1e-6)


def test_round_trip_keeps_the_default_noise(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings, tmp_path: Path
) -> None:
    """A problem without noise models has the same log-likelihood when read."""
    op_hctz_pk.initialize(fit_settings)
    yaml_file = to_petab(op_hctz_pk, tmp_path, settings=fit_settings)

    problem, settings = from_petab(yaml_file)
    problem.initialize(settings)
    assert all(noise is not None for noise in problem.noise_models)
    # up to the selections of the tasks, see the `selections` gap
    assert log_likelihood(problem) == pytest.approx(
        log_likelihood(op_hctz_pk), rel=1e-4
    )


def test_export_requires_a_value_per_measurement(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings, tmp_path: Path
) -> None:
    """A noise model which does not cover the data is not written."""
    op_hctz_iv.initialize(fit_settings)
    op_hctz_iv.noise_models[0] = NoiseModel(
        formula="sd", placeholders=("sd",), placeholder_values=((0.5,),)
    )
    with pytest.raises(ValueError, match="placeholders"):
        to_petab(op_hctz_iv, tmp_path, settings=fit_settings)


# --- THE GAPS ---


def test_gaps_of_the_noise(petab_iv: Path) -> None:
    """A problem with a noise model runs into the gaps of the noise."""
    observable_id = _observable_ids(petab_iv)[0]
    _set_noise(petab_iv, observable_id, noiseFormula=SIGMA.pid)
    _add_sigma(petab_iv)

    problem, settings = from_petab(petab_iv / "problem.yaml")
    problem.initialize(settings)
    ids = {gap.id for gap in gaps_of_problem(problem)}
    assert {"noise-model", "noise-parameters"} <= ids


def test_a_problem_without_noise_has_no_gap_of_it(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The gaps of the noise are the ones of a problem which has a noise."""
    op_hctz_iv.initialize(fit_settings)
    ids = {gap.id for gap in gaps_of_problem(op_hctz_iv)}
    assert not {"noise-model", "noise-parameters", "foreign-extension"} & ids
    assert GAPS_BY_ID["noise-model"].kind is GapKind.LOSSY
    assert GAPS_BY_ID["foreign-extension"].kind is GapKind.UNSUPPORTED
