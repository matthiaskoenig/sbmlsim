"""The kinds of a fit, run whole and split into units on the iv problem of HCTZ.

The fit, its profiles, its Fisher information, its report and its PEtab problem.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sbmlsim import run
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.run import ConfigurationError, PlanMismatch
from sbmlsim.run.fit_kinds import FitSpec, IdentifiabilitySpec, ProfileSpec
from sbmlsim.run.units import MissingUnits

PROJECT = "examples.hctz_fitting.project:PROJECT"
ROOT = Path(__file__).parents[2]

#: a fit of three repeats which stop after a few steps
FIT: dict[str, Any] = {
    "project": PROJECT,
    "subset": "IV",
    "repeats": 3,
    "seed": 1,
    "optimizer": {"max_nfev": 3},
}

#: the profiles of two parameters of a few points each
IDENTIFIABILITY: dict[str, Any] = {
    "project": PROJECT,
    "subset": "IV",
    "pids": ["GU__HCTZABS_k", "KI__HCTZEX_k"],
    "max_points": 2,
    "optimizer": {"max_nfev": 2},
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _no_simrun_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run of the tests is no task of simrun, whatever the environment."""
    for name in list(os.environ):
        if name.startswith("SIMRUN_"):
            monkeypatch.delenv(name)


@pytest.fixture(scope="module")
def fit_out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Get the out of a whole fit, which the other kinds start from."""
    out = tmp_path_factory.mktemp("fit") / "out"
    run.execute("fit", FIT, out, workers=1)
    return out


@pytest.fixture(scope="module")
def identifiability_out(
    fit_out: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Get the out of a whole identifiability analysis of the fit."""
    out = tmp_path_factory.mktemp("identifiability") / "out"
    run.execute(
        "identifiability", IDENTIFIABILITY, out, workers=1, inputs={"fit": fit_out}
    )
    return out


def _result(out: Path) -> OptimizationResult:
    return OptimizationResult.from_json(out / "IV" / "optimization_result.json")


def _assert_same_fits(a: OptimizationResult, b: OptimizationResult) -> None:
    np.testing.assert_array_equal(a.df_fits.cost.to_numpy(), b.df_fits.cost.to_numpy())
    np.testing.assert_array_equal(np.stack(a.df_fits.x), np.stack(b.df_fits.x))


def test_fit_writes_the_records(fit_out: Path) -> None:
    records = fit_out / "IV"
    assert sorted(p.name for p in (records / "runs").glob("*.json")) == [
        f"IV_run_{k}.json" for k in range(3)
    ]
    for name in ("optimization_result.json", "parameters.json", "fits.tsv"):
        assert (records / name).exists()
    assert not (fit_out / "adopted.json").exists()
    planned = FitSpec.model_validate(_json(fit_out / "spec.json"))
    assert (planned.opid, planned.seed, planned.units) == ("IV", 1, 3)
    summary = _json(fit_out / "summary.json")
    assert summary["opid"] == "IV"
    assert summary["repeats"] == 3
    assert 0 <= summary["converged"] <= 3
    assert summary["best_cost"] == pytest.approx(_result(fit_out).df_fits.cost.min())
    assert len(_json(records / "parameters.json")["sets"]) == 1


def test_fit_plan_draws_a_seed() -> None:
    planned = run.plan("fit", {"project": PROJECT, "subset": "IV", "repeats": 130})
    assert isinstance(planned, FitSpec)
    assert planned.seed is not None
    assert planned.opid == "IV"
    assert planned.units == 44
    assert run.count_units("fit", planned) == 44


def test_fit_plan_refuses_an_unknown_subset() -> None:
    with pytest.raises(ConfigurationError, match="NONE"):
        run.plan("fit", {"project": PROJECT, "subset": "NONE"})


def test_fit_split_equals_fit(fit_out: Path, tmp_path: Path) -> None:
    planned = run.plan("fit", FIT)
    out = tmp_path / "split"
    for index in range(run.count_units("fit", planned)):
        run.compute_unit("fit", planned, index, out)
    summary = run.merge("fit", planned, out)

    _assert_same_fits(_result(out), _result(fit_out))
    whole = _json(fit_out / "summary.json")
    assert summary == whole


def test_fit_merge_names_the_missing_units(tmp_path: Path) -> None:
    planned = run.plan("fit", FIT)
    out = tmp_path / "split"
    run.compute_unit("fit", planned, 1, out)
    with pytest.raises(MissingUnits) as info:
        run.merge("fit", planned, out)
    assert info.value.missing == (0, 2)


def test_fit_resume_runs_the_missing_repeats(
    fit_out: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    shutil.copytree(fit_out, out)
    (out / "IV" / "runs" / "IV_run_2.json").unlink()
    calls: list[int] = []
    optimize_run = OptimizationProblem.optimize_run

    def counted(self: OptimizationProblem, *args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("run", -1))
        return optimize_run(self, *args, **kwargs)

    monkeypatch.setattr(OptimizationProblem, "optimize_run", counted)
    run.execute("fit", FIT, out, workers=1, resume=True)

    assert calls == [2]
    _assert_same_fits(_result(out), _result(fit_out))


def test_fit_adopt_needs_the_identifiability() -> None:
    """An adopted fit carries the uncertainty of its parameters.

    The adoption itself is tested on the toy project, `test_fit_kinds_project`.
    """
    with pytest.raises(ConfigurationError, match="identifiability"):
        run.plan("fit", {**FIT, "repeats": 1, "adopt": True})


def test_fit_records_the_provenance_argument(tmp_path: Path) -> None:
    """`--provenance FILE` reaches the records of the fit, without the variable."""
    sources = [{"name": "model", "sha": "abc", "dirty": False}]
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps(sources), encoding="utf-8")
    out = tmp_path / "out"
    run.execute("fit", {**FIT, "repeats": 1}, out, workers=1, provenance=provenance)

    for path in (
        out / "IV" / "optimization_result.json",
        out / "IV" / "runs" / "IV_run_0.json",
    ):
        recorded = _json(path)["provenance"]
        assert recorded["sources"] == sources
        assert recorded["git"] is None


def test_unit_with_another_plan_is_refused(tmp_path: Path) -> None:
    planned = run.plan("fit", FIT)
    assert isinstance(planned, FitSpec)
    other = planned.model_copy(update={"units": 99})
    with pytest.raises(PlanMismatch):
        run.compute_unit("fit", other, 0, tmp_path / "out")
    with pytest.raises(PlanMismatch):
        run.merge("fit", other, tmp_path / "merge")


def _profiles(out: Path) -> dict[str, Any]:
    d = _json(out / "identifiability.json")
    del d["duration"]
    return d


def test_identifiability_split_equals_whole(
    fit_out: Path, identifiability_out: Path, tmp_path: Path
) -> None:
    planned = run.plan("identifiability", IDENTIFIABILITY)
    assert isinstance(planned, IdentifiabilitySpec)
    assert planned.units == 2
    out = tmp_path / "split"
    inputs = {"fit": fit_out}
    for index in range(run.count_units("identifiability", planned)):
        run.compute_unit("identifiability", planned, index, out, inputs=inputs)
    summary = run.merge("identifiability", planned, out, inputs=inputs)

    assert _profiles(out) == _profiles(identifiability_out)
    assert _json(out / "fisher.json") == _json(identifiability_out / "fisher.json")
    assert sorted(p.name for p in (out / "profiles").glob("*.json")) == [
        "GU__HCTZABS_k.json",
        "KI__HCTZEX_k.json",
    ]
    assert summary == _json(identifiability_out / "summary.json")
    assert summary["n_parameters"] == 2
    for name in ("optimization_result.json", "parameters.json"):
        assert (out / name).read_bytes() == (fit_out / "IV" / name).read_bytes()


def test_identifiability_plan_profiles_every_parameter() -> None:
    planned = run.plan("identifiability", {"project": PROJECT, "subset": "IV"})
    assert isinstance(planned, IdentifiabilitySpec)
    assert planned.pids == ["Ka_dis_hctz", "GU__HCTZABS_k", "KI__HCTZEX_k"]
    assert planned.units == 3


def test_identifiability_needs_the_fit(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="fit"):
        run.execute("identifiability", IDENTIFIABILITY, tmp_path / "out", workers=1)


def test_report_from_identifiability_out(
    identifiability_out: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    summary = run.execute(
        "report",
        {"project": PROJECT, "subset": "IV", "mapping_figures": False},
        out,
        inputs={"fit": identifiability_out},
    )
    assert (out / "report" / "index.html").exists()
    assert summary["report"] == "report"


def test_fisher_writes_fisher_json(fit_out: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    summary = run.execute(
        "fisher",
        {"project": PROJECT, "subset": "IV"},
        out,
        workers=1,
        inputs={"fit": fit_out},
    )
    fisher = _json(out / "fisher.json")
    assert fisher["pids"] == ["Ka_dis_hctz", "GU__HCTZABS_k", "KI__HCTZEX_k"]
    assert summary["rank"] == fisher_rank(out)
    for name in ("optimization_result.json", "parameters.json"):
        assert (out / name).exists()


def fisher_rank(out: Path) -> int:
    from sbmlsim.fit.cli import load_fit_records

    fisher = load_fit_records(out).fisher
    assert fisher is not None
    return fisher.rank


def test_petab_writes_a_problem(fit_out: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    summary = run.execute(
        "petab",
        {"project": PROJECT, "subset": "IV"},
        out,
        inputs={"fit": fit_out},
    )
    assert (out / "petab" / "problem.yaml").exists()
    assert summary["problem"] == "petab/problem.yaml"


def test_the_kinds_of_a_fit_are_registered() -> None:
    for name in ("fit", "identifiability", "fisher", "report", "petab"):
        assert run.get_kind(name).name == name
    assert run.get_kind("fit").cost_setting == "subset"
    assert not run.get_kind("report").uses_cores
    assert not run.get_kind("petab").uses_cores
    assert run.get_kind("identifiability").inputs == ("fit",)


def test_petab_without_a_fit_starts_from_the_start_values(tmp_path: Path) -> None:
    out = tmp_path / "out"
    run.execute("petab", {"project": PROJECT, "subset": "IV"}, out)
    assert (out / "petab" / "problem.yaml").exists()


def test_identifiability_defaults_are_the_profile_settings() -> None:
    from sbmlsim.fit.identifiability import ProfileSettings

    defaults = ProfileSettings().to_dict()
    spec = IdentifiabilitySpec(project=PROJECT, subset="IV")
    profile = ProfileSpec()
    # every setting of the profiles is a field of the spec
    names = set(defaults) - {"optimizer_kwargs"}
    assert names <= set(ProfileSpec.model_fields)
    for name in names:
        assert getattr(spec, name) == defaults[name], name
        assert getattr(profile, name) == defaults[name], name
    assert spec.optimizer == defaults["optimizer_kwargs"]
    assert profile.optimizer == defaults["optimizer_kwargs"]


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in ("OMP_NUM_THREADS", "MPLBACKEND")
        and not name.startswith("SIMRUN_")
    }
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-m", "sbmlsim.run", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def test_cli_fisher_with_the_input_of_a_fit(fit_out: Path, tmp_path: Path) -> None:
    spec = tmp_path / "spec.yaml"
    spec.write_text(f"project: {PROJECT}\nsubset: IV\n", encoding="utf-8")
    out = tmp_path / "out"
    result = _cli("fisher", str(spec), "--out", str(out), f"--input=fit={fit_out}")
    assert result.returncode == 0, result.stderr
    assert (out / "fisher.json").exists()

    missing = _cli("fisher", str(spec), "--out", str(tmp_path / "missing"))
    assert missing.returncode == 4, missing.stderr
    assert "input 'fit'" in missing.stderr


@pytest.mark.parametrize(
    "opid", ["../../escape", "/absolute", "a/b", "a\\b", ".", "..", "", "C:x"]
)
def test_an_opid_outside_of_the_out_is_refused(opid: str, tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="directory"):
        run.plan("fit", {**FIT, "opid": opid})
    planned = run.plan("fit", FIT)
    with pytest.raises(ConfigurationError, match="directory"):
        run.compute_unit(
            "fit", planned.model_copy(update={"opid": opid}), 0, tmp_path / "out"
        )
    assert list(tmp_path.parent.glob("escape")) == []


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"studies": ["NoStudy"]}, "NoStudy"),
        ({"optimizer": {"no_such_argument": 1}}, "no_such_argument"),
        ({"optimizer": {"seed": 2}}, "seed"),
        ({"optimizer": {"loss": "soft_l1"}}, "loss"),
    ],
)
def test_fit_plan_checks_the_studies_and_the_optimizer(
    settings: dict[str, Any], match: str
) -> None:
    with pytest.raises(ConfigurationError, match=match):
        run.plan("fit", {**FIT, **settings})


def test_fit_plan_accepts_the_arguments_of_the_runner() -> None:
    planned = run.plan("fit", {**FIT, "optimizer": {"xtol": 1e-6, "serial": True}})
    assert isinstance(planned, FitSpec)


@pytest.mark.parametrize(
    ("sampling", "name"),
    [("logUniform_lhs", "LOGUNIFORM_LHS"), ("START", "START"), (2, "UNIFORM")],
)
def test_fit_sampling_of_a_yaml_spec(
    sampling: str | int, name: str, tmp_path: Path
) -> None:
    """A sampling given by its name or value, as YAML gives it, runs in a unit."""
    planned = run.plan(
        "fit", {**FIT, "repeats": 1, "optimizer": {"sampling": sampling}}
    )
    assert isinstance(planned, FitSpec)
    assert planned.optimizer["sampling"] == name
    out = tmp_path / "out"
    run.compute_unit("fit", planned, 0, out)
    assert (out / "IV" / "runs" / "IV_run_0.json").exists()


@pytest.mark.parametrize("sampling", ["NOPE", 99, 1.5])
def test_fit_plan_refuses_an_unknown_sampling(sampling: Any) -> None:
    with pytest.raises(ConfigurationError, match="sampling"):
        run.plan("fit", {**FIT, "optimizer": {"sampling": sampling}})


def test_an_interrupted_fit_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl-C after some repeats finished keeps them and ends the run."""
    optimize_run = OptimizationProblem.optimize_run

    def interrupted(self: OptimizationProblem, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("run") == 1:
            raise KeyboardInterrupt
        return optimize_run(self, *args, **kwargs)

    monkeypatch.setattr(OptimizationProblem, "optimize_run", interrupted)
    out = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt):
        run.execute("fit", FIT, out, workers=1)
    assert sorted(p.name for p in (out / "IV" / "runs").glob("*.json")) == [
        "IV_run_0.json"
    ]
    assert _json(out / "run.json")["status"] == "failed"


def test_lost_repeats_are_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from sbmlsim.fit import cli as fit_cli

    def run_fit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"IV": SimpleNamespace(result=SimpleNamespace(interrupted=False))}

    monkeypatch.setattr(fit_cli, "run_fit", run_fit)
    with pytest.raises(RuntimeError, match=r"\[0, 1, 2\].*--resume"):
        run.execute("fit", FIT, tmp_path / "out", workers=1)


def test_identifiability_merge_names_the_missing_profiles(
    fit_out: Path, tmp_path: Path
) -> None:
    planned = run.plan("identifiability", IDENTIFIABILITY)
    out = tmp_path / "split"
    inputs = {"fit": fit_out}
    run.compute_unit("identifiability", planned, 1, out, inputs=inputs)
    with pytest.raises(MissingUnits) as info:
        run.merge("identifiability", planned, out, inputs=inputs)
    assert info.value.missing == (0,)


def test_identifiability_of_a_fit_of_some_studies(tmp_path: Path) -> None:
    """The analyses of a fit of one study evaluate the problem of that study."""
    fit = tmp_path / "fit"
    run.execute(
        "fit",
        {
            "project": PROJECT,
            "subset": "PK",
            "repeats": 1,
            "seed": 1,
            "studies": ["Beermann1976"],
            "optimizer": {"max_nfev": 2},
        },
        fit,
        workers=1,
    )
    out = tmp_path / "out"
    run.execute(
        "identifiability",
        {
            "project": PROJECT,
            "subset": "PK",
            "pids": ["KI__HCTZEX_k"],
            "max_points": 1,
            "reoptimize": False,
            "fisher": False,
        },
        out,
        workers=1,
        inputs={"fit": fit},
    )
    best = OptimizationResult.from_json(fit / "PK" / "optimization_result.json")
    profiles = _json(out / "identifiability.json")
    assert profiles["cost"] == pytest.approx(best.df_fits.cost.iloc[0], rel=1e-9)
