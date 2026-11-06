"""The kinds of scans, sensitivity and uncertainty analyses and experiments.

Run whole and split into units on the project of HCTZ: its scans `dose` and
`sobol`, its uncertainty analysis `cl` and its experiment `Beermann1976`.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
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
from sbmlsim.simulation import Custom, Dimension, Observable, Scan
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
        "many": HCTZ.uncertainty("cl").model_copy(update={"n": 130}),
        "adaptive": HCTZ.uncertainty("cl").model_copy(
            update={"n": None, "n_start": 4, "n_max": 8, "rel_tol": 1e-6}
        ),
    },
)


def checked_excretion(time: np.ndarray, values: dict[str, Any]) -> float:
    """Get the renal excretion, which fails above 0.0035/ml."""
    k = float(np.max(values["KI__HCTZEX_k"]))
    if k > 0.0035:
        raise ValueError(f"the excretion {k} is too fast")
    return k


def failing_scan() -> Scan:
    """Get 24 renal excretions, of which the last 15 fail."""
    return Scan(
        oral_dose(),
        [
            Dimension(
                "excretion", values={"KI__HCTZEX_k": np.linspace(0.002, 0.006, 24)}
            )
        ],
    )


def failing_observables() -> list[Observable]:
    """Get the plasma concentration and the checked excretion."""
    return [
        *observables()[:1],
        Custom("checked", checked_excretion, "1/ml", symbols=["KI__HCTZEX_k"]),
    ]


#: a ragged scan of 24 points in 24 blocks whose points above 0.0035/ml fail
FAILING = Project(
    name="failing",
    root=HCTZ_PATH,
    scans={
        "failing": ScanAnalysis(
            model=MODEL_PATH,
            scan=failing_scan,
            observables=failing_observables,
            on_error="flag",
        )
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


def test_a_whole_scan_runs_its_blocks_on_its_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A whole run runs its blocks in spans and writes the files of its units.

    A block whose messages of failed points its span does not keep runs alone,
    here the block of the point 23.
    """
    from sbmlsim.result.scan import MAX_ERRORS, LostErrors

    spec = {"project": f"{__name__}:FAILING", "analysis": "failing"}
    calls: list[range | None] = []
    simulate = Simulator.run
    part = ScanResult.part

    def counted(self: Simulator, *args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("points"))
        assert self.n_workers == 2
        return simulate(self, *args, **kwargs)

    def lost(self: ScanResult, points: range) -> ScanResult:
        if points == range(23, 24):
            raise LostErrors("the messages of the point 23 are lost")
        return part(self, points)

    monkeypatch.setattr(Simulator, "run", counted)
    monkeypatch.setattr(ScanResult, "part", lost)
    whole = tmp_path / "whole"
    summary = run.execute("scan", spec, whole, workers=2)
    monkeypatch.undo()
    assert summary == {"points": 24, "failed": 15}
    # a span of two blocks for the two workers, the point 23 again alone
    assert calls == [range(k, k + 2) for k in range(0, 24, 2)] + [range(23, 24)]

    planned = run.plan("scan", spec)
    split = tmp_path / "split"
    for index in range(run.count_units("scan", planned)):
        run.compute_unit("scan", planned, index, split)
    run.merge("scan", planned, split)
    files = sorted(p.name for p in (whole / "blocks").iterdir())
    assert files == sorted(p.name for p in (split / "blocks").iterdir())
    assert len(files) == 24
    for name in [*(f"blocks/{f}" for f in files), "result.nc"]:
        xr.testing.assert_identical(
            ScanResult.from_netcdf(whole / name).ds,
            ScanResult.from_netcdf(split / name).ds,
        )
    result = ScanResult.from_netcdf(whole / "result.nc")
    assert "_point" in result.ds.dims
    assert len(result.ds.attrs["errors"]) == MAX_ERRORS


def six_scan() -> Scan:
    """Get 6 renal excretions, 6 blocks of one point."""
    return Scan(
        oral_dose(),
        [Dimension("excretion", values={"KI__HCTZEX_k": np.linspace(0.002, 0.004, 6)})],
    )


#: a ragged scan of 6 points in 6 blocks
SIX = Project(
    name="six",
    root=HCTZ_PATH,
    scans={"six": ScanAnalysis(model=MODEL_PATH, scan=six_scan)},
)


def test_a_whole_scan_keeps_its_blocks_span_by_span(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A span is as many blocks as the run has workers, its files kept once it ran."""
    spec = {"project": f"{__name__}:SIX", "analysis": "six"}
    whole = tmp_path / "whole"
    calls: list[tuple[range | None, list[str]]] = []
    simulate = Simulator.run

    def counted(self: Simulator, *args: Any, **kwargs: Any) -> Any:
        blocks = whole / "blocks"
        files = sorted(p.name for p in blocks.iterdir()) if blocks.exists() else []
        calls.append((kwargs.get("points"), files))
        return simulate(self, *args, **kwargs)

    monkeypatch.setattr(Simulator, "run", counted)
    run.execute("scan", spec, whole, workers=2)
    monkeypatch.undo()
    assert calls == [
        (range(2), []),
        (range(2, 4), ["points-0-1.nc", "points-1-2.nc"]),
        (
            range(4, 6),
            ["points-0-1.nc", "points-1-2.nc", "points-2-3.nc", "points-3-4.nc"],
        ),
    ]

    planned = run.plan("scan", spec)
    split = tmp_path / "split"
    for index in range(run.count_units("scan", planned)):
        run.compute_unit("scan", planned, index, split)
    run.merge("scan", planned, split)
    files = sorted(p.name for p in (whole / "blocks").iterdir())
    assert len(files) == 6
    for name in [*(f"blocks/{f}" for f in files), "result.nc"]:
        xr.testing.assert_identical(
            ScanResult.from_netcdf(whole / name).ds,
            ScanResult.from_netcdf(split / name).ds,
        )


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


def _designed(
    values: list[float],
    seed: int = 1,
    reference: float = 0.0037,
    covariate: float = 70.0,
) -> Scan:
    """Get a scan of a dimension drawn by a design and one of given values."""
    from sbmlsim.simulation.scan import Design

    design = Design(
        method="random",
        distributions={"KI__HCTZEX_k": {"type": "Uniform", "relative": 0.5}},
        options={"n": len(values), "seed": seed},
        references={"KI__HCTZEX_k": {"value": reference, "unit": "1/ml"}},
    )
    return Scan(
        oral_dose(),
        [
            Dimension(
                "draws",
                values={"KI__HCTZEX_k": values},
                coordinates={"weight": [covariate] * len(values)},
                design=design,
            ),
            Dimension("dose", values={"PODOSE_hctz": [10.0, 20.0]}),
        ],
    )


def test_scan_hash_is_the_identity_without_the_drawn_values() -> None:
    """A last bit of a drawn value or a reference, e.g. of another platform."""
    from sbmlsim.run.scan_kinds import scan_hash

    values = [0.002, 0.003]
    other = [0.002, float(np.nextafter(0.003, 1.0))]
    digest = scan_hash(_designed(values))
    assert scan_hash(_designed(other)) == digest
    assert scan_hash(_designed(values, reference=0.0037000000000000002)) == digest
    assert scan_hash(_designed(values, covariate=70.00000000000001)) == digest
    # the seed, the number of draws and the given values are the scan
    assert scan_hash(_designed(values, seed=2)) != digest
    assert scan_hash(_designed([0.002, 0.003, 0.004])) != digest
    given = _designed(values)
    changed = Scan(
        given.simulation,
        [
            given.dimensions[0],
            Dimension("dose", values={"PODOSE_hctz": [10.0, 20.000000000000004]}),
        ],
    )
    assert scan_hash(changed) != digest


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


def test_a_whole_propagation_runs_its_blocks_on_its_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The blocks of a whole run run in spans of two, as files of units."""
    calls: list[int] = []
    simulate = Simulator.run

    def counted(self: Simulator, model: Any, scan: Any, *args: Any, **kw: Any) -> Any:
        calls.append(len(scan))
        assert self.n_workers == 2
        return simulate(self, model, scan, *args, **kw)

    monkeypatch.setattr(Simulator, "run", counted)
    whole = tmp_path / "whole"
    summary = run.execute("uncertainty", FEW, whole, workers=2)
    monkeypatch.undo()
    assert calls == [2, 2]

    planned = run.plan("uncertainty", FEW)
    split = tmp_path / "split"
    for index in range(run.count_units("uncertainty", planned)):
        run.compute_unit("uncertainty", planned, index, split)
    assert run.merge("uncertainty", planned, split) == summary
    files = sorted(p.relative_to(whole) for p in (whole / "checkpoint").rglob("*.nc"))
    assert len(files) == 4
    assert files == sorted(
        p.relative_to(split) for p in (split / "checkpoint").rglob("*.nc")
    )
    for path in files:
        xr.testing.assert_identical(
            ScanResult.from_netcdf(whole / path).ds,
            ScanResult.from_netcdf(split / path).ds,
        )
    xr.testing.assert_identical(
        _uncertainty(whole).results["default"].ds,
        _uncertainty(split).results["default"].ds,
    )


def test_a_checkpoint_of_another_propagation_is_a_plan_mismatch(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    run.execute("uncertainty", FEW, out, workers=1)
    path = out / "checkpoint" / "default" / "draws-1-2.nc"
    ds = ScanResult.from_netcdf(path).ds
    ds.attrs["propagation"] = {**ds.attrs["propagation"], "on_error": "raise"}
    path.unlink()
    ScanResult(ds).to_netcdf(path)

    with pytest.raises(PlanMismatch, match="another propagation"):
        run.execute("uncertainty", FEW, out, workers=1, resume=True)
    with pytest.raises(PlanMismatch, match="another propagation"):
        run.compute_unit("uncertainty", run.plan("uncertainty", FEW), 1, out)
    with pytest.raises(PlanMismatch, match="another propagation"):
        run.merge("uncertainty", _json(out / "spec.json"), out)
    code = main(["uncertainty", *_spec_file(tmp_path), "--out", str(out), "--resume"])
    assert code == EXIT_CONFIGURATION


def _spec_file(tmp_path: Path) -> list[str]:
    """Write the spec FEW into a file, the argument of the command line."""
    path = tmp_path / "few.json"
    path.write_text(json.dumps(FEW), encoding="utf-8")
    return [str(path)]


def test_the_plan_of_more_draws_than_units_has_blocks_of_several_draws(
    tmp_path: Path,
) -> None:
    from sbmlsim.run.context import RunContext
    from sbmlsim.run.scan_kinds import _planned_uncertainty

    spec = {"project": f"{__name__}:SMALL", "analysis": "many"}
    planned = run.plan("uncertainty", spec)
    assert isinstance(planned, UncertaintySpec)
    assert run.count_units("uncertainty", planned) == 44
    ctx = RunContext(out=tmp_path, workers=1, resume=False, provenance=None)
    propagation, blocks = _planned_uncertainty(planned, ctx, workers=1)
    assert blocks is not None
    assert propagation.block_size == 3
    assert propagation.blocks() == blocks
    assert blocks[0] == range(3)
    assert blocks[-1] == range(129, 130)


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


def test_simulate_gives_the_workers_to_the_simulator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A serial run of a few experiments runs their scans on the workers."""
    from sbmlsim.experiment import runner

    given: dict[str, Any] = {}

    def run_experiments(experiments: list[Any], **kwargs: Any) -> list[Any]:
        given.update(kwargs, experiments=experiments)
        return []

    monkeypatch.setattr(runner, "run_experiments", run_experiments)
    assert HCTZ.experiments is not None
    names = [cls.__name__ for cls in HCTZ.experiments()[:3]]
    run.execute(
        "simulate",
        {"project": PROJECT, "experiments": names},
        tmp_path / "out",
        workers=3,
    )
    assert [cls.__name__ for cls in given["experiments"]] == names
    assert given["n_workers"] == 3
    assert given["simulator"].n_workers == 3


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
