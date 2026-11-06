"""The kinds of scans, sensitivity and uncertainty analyses and experiments.

Run whole and split into units on the project of HCTZ: its scans `dose` and
`sobol`, its uncertainty analysis `cl` and its experiment `Beermann1976`.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
import xarray as xr

from examples.hctz_fitting import HCTZ_PATH, MODEL_PATH
from examples.hctz_fitting.project import PROJECT as HCTZ
from examples.hctz_fitting.project import observables, oral_dose
from sbmlsim import run
from sbmlsim.project import Project, ScanAnalysis
from sbmlsim.result import ScanResult
from sbmlsim.run import ConfigurationError, PlanMismatch
from sbmlsim.run.cli import EXIT_CONFIGURATION, main
from sbmlsim.run.scan_kinds import ScanSpec, SensitivitySpec, UncertaintySpec
from sbmlsim.run.units import MissingUnits
from sbmlsim.sensitivity import SensitivityResult
from sbmlsim.simulation import Scan
from sbmlsim.simulation.sampling import Uniform, random
from sbmlsim.simulator import Simulator
from sbmlsim.uncertainty import UncertaintyResult

PROJECT = "examples.hctz_fitting.project:PROJECT"
DOSE: dict[str, Any] = {"project": PROJECT, "analysis": "dose"}
SOBOL: dict[str, Any] = {"project": PROJECT, "analysis": "sobol"}
CL: dict[str, Any] = {"project": PROJECT, "analysis": "cl"}


def unseeded_scan() -> Scan:
    """Get a random design without a seed, which draws other points every time."""
    simulation = oral_dose()
    design = random(
        {"KI__HCTZEX_k": Uniform(relative=0.5)},
        4,
        seed=None,
        model=MODEL_PATH,
        simulation=simulation,
    )
    return Scan(simulation, [design])


#: a project whose scan has a design without a seed
UNSEEDED = Project(
    name="unseeded",
    root=HCTZ_PATH,
    scans={
        "random": ScanAnalysis(
            model=MODEL_PATH, scan=unseeded_scan, observables=observables
        )
    },
)

#: the uncertainty analysis `cl` of four draws, `few`, and of the adaptive
#: number of draws of a few rounds, `adaptive`
SMALL = Project(
    name="small",
    root=HCTZ_PATH,
    uncertainties={
        "few": HCTZ.uncertainty("cl").model_copy(update={"n": 4}),
        "adaptive": HCTZ.uncertainty("cl").model_copy(
            update={"n": None, "n_start": 4, "n_max": 8, "rel_tol": 1e-6}
        ),
    },
)
FEW: dict[str, Any] = {"project": f"{__name__}:SMALL", "analysis": "few"}
ADAPTIVE: dict[str, Any] = {"project": f"{__name__}:SMALL", "analysis": "adaptive"}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _no_simrun_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run of the tests is no task of simrun, whatever the environment."""
    for name in list(os.environ):
        if name.startswith("SIMRUN_"):
            monkeypatch.delenv(name)


@pytest.fixture(scope="module")
def dose_out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Get the out of a whole run of the dose scan."""
    out = tmp_path_factory.mktemp("dose") / "out"
    run.execute("scan", DOSE, out, workers=1)
    return out


def test_scan_writes_its_result(dose_out: Path) -> None:
    result = ScanResult.from_netcdf(dose_out / "result.nc")
    assert result.dims == ("dose",)
    assert "pk.cmax" in result.ds
    planned = ScanSpec.model_validate(_json(dose_out / "spec.json"))
    assert planned.units == 3
    assert planned.scan_hash is not None
    assert sorted(p.name for p in (dose_out / "blocks").iterdir()) == [
        "points-0-1.nc",
        "points-1-2.nc",
        "points-2-3.nc",
    ]
    assert _json(dose_out / "summary.json") == {"points": 3, "failed": 0}


def test_scan_split_equals_scan(dose_out: Path, tmp_path: Path) -> None:
    planned = run.plan("scan", DOSE)
    out = tmp_path / "split"
    for index in reversed(range(run.count_units("scan", planned))):
        run.compute_unit("scan", planned, index, out)
    summary = run.merge("scan", planned, out)

    assert summary == _json(dose_out / "summary.json")
    with (
        xr.open_dataset(out / "result.nc", engine="h5netcdf") as split,
        xr.open_dataset(dose_out / "result.nc", engine="h5netcdf") as whole,
    ):
        xr.testing.assert_identical(split.load(), whole.load())


def test_scan_resume_skips_blocks(
    dose_out: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    shutil.copytree(dose_out, out)
    (out / "blocks" / "points-1-2.nc").unlink()
    calls: list[range | None] = []
    simulate = Simulator.run

    def counted(self: Simulator, *args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("points"))
        return simulate(self, *args, **kwargs)

    monkeypatch.setattr(Simulator, "run", counted)
    run.execute("scan", DOSE, out, workers=1, resume=True)

    assert calls == [range(1, 2)]
    assert (out / "blocks" / "points-1-2.nc").exists()


def test_scan_merge_names_the_missing_units(tmp_path: Path) -> None:
    planned = run.plan("scan", DOSE)
    out = tmp_path / "split"
    run.compute_unit("scan", planned, 1, out)
    with pytest.raises(MissingUnits) as info:
        run.merge("scan", planned, out)
    assert info.value.missing == (0, 2)


def test_scan_with_another_plan_is_refused(tmp_path: Path) -> None:
    planned = run.plan("scan", DOSE)
    assert isinstance(planned, ScanSpec)
    for update in ({"units": 7}, {"scan_hash": "0" * 64}):
        other = planned.model_copy(update=update)
        with pytest.raises(PlanMismatch):
            run.compute_unit("scan", other, 0, tmp_path / f"unit-{len(update)}")
        with pytest.raises(PlanMismatch):
            run.merge("scan", other, tmp_path / f"merge-{len(update)}")


def test_scan_hash_is_stable() -> None:
    first = run.plan("scan", SOBOL)
    second = run.plan("scan", SOBOL)
    assert isinstance(first, ScanSpec) and isinstance(second, ScanSpec)
    assert first.scan_hash is not None
    assert first.scan_hash == second.scan_hash
    dose = run.plan("scan", DOSE)
    assert isinstance(dose, ScanSpec)
    assert first.scan_hash != dose.scan_hash


def test_plan_refuses_a_design_without_seed() -> None:
    with pytest.raises(ConfigurationError, match="give it a seed"):
        run.plan("scan", {"project": f"{__name__}:UNSEEDED", "analysis": "random"})


def test_plan_refuses_an_unknown_analysis() -> None:
    for kind in ("scan", "sensitivity", "uncertainty"):
        with pytest.raises(ConfigurationError, match="NONE"):
            run.plan(kind, {"project": PROJECT, "analysis": "NONE"})


def test_sensitivity_writes_indices(tmp_path: Path) -> None:
    out = tmp_path / "out"
    summary = run.execute("sensitivity", SOBOL, out, workers=1)
    indices = SensitivityResult.from_netcdf(out / "sensitivity.nc")
    assert "pk.cmax.S1" in indices.ds
    assert set(indices.ds.coords["parameter"].values) == {
        "Ka_dis_hctz",
        "KI__HCTZEX_k",
    }
    assert (out / "result.nc").exists()
    assert summary["points"] == 32
    assert summary["indices"] == "sobol"
    assert isinstance(run.plan("sensitivity", SOBOL), SensitivitySpec)


def test_sensitivity_needs_indices(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="indices"):
        run.plan("sensitivity", DOSE)
    spec = tmp_path / "spec.yaml"
    spec.write_text(f"project: {PROJECT}\nanalysis: dose\n", encoding="utf-8")
    code = main(["sensitivity", str(spec), "--out", str(tmp_path / "out")])
    assert code == EXIT_CONFIGURATION


def _uncertainty(out: Path) -> UncertaintyResult:
    return UncertaintyResult.from_netcdf(out / "uncertainty")


def _split_equals_whole(spec: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    """Run an uncertainty analysis whole and in units, compare and get the summary."""
    planned = run.plan("uncertainty", spec)
    assert isinstance(planned, UncertaintySpec)
    n_units = run.count_units("uncertainty", planned)
    whole_out = tmp_path / "whole"
    whole = run.execute("uncertainty", spec, whole_out, workers=1)

    out = tmp_path / "split"
    for index in range(n_units - 1):
        run.compute_unit("uncertainty", planned, index, out)
    with pytest.raises(MissingUnits) as info:
        run.merge("uncertainty", planned, out)
    assert info.value.missing == (n_units - 1,)
    run.compute_unit("uncertainty", planned, n_units - 1, out)
    assert run.merge("uncertainty", planned, out) == whole

    split, unsplit = _uncertainty(out), _uncertainty(whole_out)
    assert split.record == unsplit.record
    xr.testing.assert_identical(
        split.results["default"].ds, unsplit.results["default"].ds
    )
    return whole


def test_uncertainty_split_equals_whole_of_a_few_draws(tmp_path: Path) -> None:
    assert run.count_units("uncertainty", run.plan("uncertainty", FEW)) == 4
    assert _split_equals_whole(FEW, tmp_path) == {"n": 4, "converged": True}


@pytest.mark.slow
def test_uncertainty_split_equals_whole(tmp_path: Path) -> None:
    planned = run.plan("uncertainty", CL)
    assert isinstance(planned, UncertaintySpec)
    assert planned.seed == 1
    assert run.count_units("uncertainty", planned) == 32
    assert _split_equals_whole(CL, tmp_path) == {"n": 32, "converged": True}


def test_the_adaptive_propagation_is_one_unit(tmp_path: Path) -> None:
    assert run.count_units("uncertainty", run.plan("uncertainty", ADAPTIVE)) == 1
    summary = _split_equals_whole(ADAPTIVE, tmp_path)
    # the tolerance is not reached in the rounds of 4 and 8 draws
    assert summary == {"n": 8, "converged": False}


def test_uncertainty_plan_draws_a_seed() -> None:
    planned = run.plan("uncertainty", {**CL, "seed": None})
    assert isinstance(planned, UncertaintySpec)
    assert planned.seed == 1
    given = run.plan("uncertainty", {**CL, "seed": 7})
    assert isinstance(given, UncertaintySpec)
    assert given.seed == 7


def test_simulate_runs_one_experiment(tmp_path: Path) -> None:
    out = tmp_path / "out"
    summary = run.execute(
        "simulate", {"project": PROJECT, "experiments": ["Beermann1976"]}, out
    )
    assert summary == {"experiments": 1, "failed": 0}
    assert (out / "index.html").exists()
    assert (out / "Beermann1976").is_dir()
    events = [
        json.loads(line)
        for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    experiment = [e for e in events if e["event"] == "experiment"]
    assert experiment
    assert {e["sid"] for e in experiment} == {"Beermann1976"}
    assert {"simulations", "figures"} <= {e["phase"] for e in experiment}


def test_simulate_refuses_unknown_experiments() -> None:
    with pytest.raises(ConfigurationError, match="NONE"):
        run.plan("simulate", {"project": PROJECT, "experiments": ["NONE"]})
    with pytest.raises(ConfigurationError, match="experiments"):
        run.plan("simulate", {"project": f"{__name__}:UNSEEDED"})


def test_the_kinds_are_registered() -> None:
    for name in ("scan", "sensitivity", "uncertainty", "simulate"):
        assert run.get_kind(name).name == name
    assert run.get_kind("scan").splittable
    assert run.get_kind("sensitivity").splittable
    assert run.get_kind("uncertainty").splittable
    assert not run.get_kind("simulate").splittable


def test_the_record_of_a_design_has_its_seed() -> None:
    """The hash of a scan sees the seed of its design."""
    from examples.hctz_fitting.project import sobol_scan

    [dimension] = sobol_scan().to_dict()["dimensions"]
    assert dimension["design"]["options"]["seed"] == 1


def test_a_relative_model_is_relative_to_the_project() -> None:
    from sbmlsim.run.scan_kinds import _model

    assert _model(HCTZ, Path("model.xml")) == HCTZ_PATH / "model.xml"
    assert _model(HCTZ, MODEL_PATH) == MODEL_PATH
    assert _model(HCTZ, "model.xml") == "model.xml"
