"""The blocks of draws of a propagation and their checkpoints."""

import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
import pytest
import xarray as xr

from sbmlsim import Q
from sbmlsim.result.scan import MAX_ERRORS, ScanResult
from sbmlsim.simulation import PK, Change, Custom, Simulation
from sbmlsim.simulation.sampling import LogNormal
from sbmlsim.simulator import Simulator
from sbmlsim.uncertainty import (
    Propagation,
    Uncertain,
    UncertaintyModel,
    UncertaintyResult,
    propagate,
)
from tests.simulator.models import sbml_pk

OBSERVABLES = [PK("pk", "[C]", dose="PODOSE", route="oral")]
DESIGNS = ["sobol", "random", "lhs"]
#: the designs with a common grid of times and the ragged timecourses
CASES = [(design, 24) for design in DESIGNS] + [("sobol", None), ("lhs", None)]


def _scenarios(steps: int | None = 24) -> dict[str, Simulation]:
    """Get the scenarios, ragged timecourses without `steps`."""
    return {
        "control": Simulation(
            end=24, steps=steps, changes=[Change(0, {"PODOSE": Q(100, "mg")})]
        ),
        "treatment": Simulation(
            end=24,
            steps=steps,
            changes=[Change(0, {"PODOSE": Uncertain(Q(200, "mg"), cv=0.1)})],
        ),
    }


def _arguments(**kwargs: Any) -> dict[str, Any]:
    return {
        "model": sbml_pk(),
        "simulation": _scenarios(),
        "uq": UncertaintyModel(parameters={"ke": LogNormal(cv=0.2)}),
        "observables": OBSERVABLES,
        "keep": ["[C]", "pk"],
        "seed": 3,
        "simulator": Simulator(n_workers=1),
    } | kwargs


def _assert_equal(a: UncertaintyResult, b: UncertaintyResult) -> None:
    assert a.scenarios == b.scenarios
    for name in a.scenarios:
        xr.testing.assert_identical(a.results[name].ds, b.results[name].ds)
    assert a.record == b.record


@pytest.mark.parametrize(("design", "steps"), CASES)
def test_blocks_equal_the_whole_propagation(design: str, steps: int | None) -> None:
    simulation = _scenarios(steps)
    whole = propagate(**_arguments(n=24, design=design, simulation=simulation))
    blocks = propagate(
        **_arguments(n=24, design=design, simulation=simulation, block_size=10)
    )
    assert whole.results["control"].ds.sizes["draw"] == 24
    assert ("_point" in whole.results["control"].ds.dims) == (steps is None)
    _assert_equal(whole, blocks)


@pytest.mark.parametrize(("design", "steps"), CASES)
def test_blocks_of_separate_instances_merge(
    design: str, steps: int | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation = _scenarios(steps)
    whole = propagate(**_arguments(n=24, design=design, simulation=simulation))
    arguments = _arguments(
        n=24, design=design, simulation=simulation, checkpoint=tmp_path, block_size=10
    )
    blocks = Propagation(**arguments).blocks()
    assert blocks == [range(0, 10), range(10, 20), range(20, 24)]  # noqa: PIE808
    for block in blocks:
        Propagation(**arguments).run_block(block)
    if steps is None:
        sizes = {
            ScanResult.from_netcdf(path).ds.sizes["_point"]
            for path in (tmp_path / "control").iterdir()
        }
        assert len(sizes) > 1  # blocks of other lengths
    assert sorted(p.name for p in (tmp_path / "control").iterdir()) == [
        "draws-0-10.nc",
        "draws-10-20.nc",
        "draws-20-24.nc",
    ]

    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("A checkpointed block is simulated again.")

    monkeypatch.setattr(Simulator, "run", fail)
    merged = Propagation(**arguments).run()
    _assert_equal(whole, merged)


def test_adaptive_rounds_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(
        n=None,
        n_start=16,
        n_max=64,
        rel_tol=0.01,
        outputs=["pk.cmax"],
        checkpoint=tmp_path,
    )
    first = propagate(**arguments)
    assert [entry["n"] for entry in first.record["history"]] == [16, 32, 64]
    for scenario in first.scenarios:
        assert sorted(p.name for p in (tmp_path / scenario).iterdir()) == [
            "draws-0-16.nc",
            "draws-16-32.nc",
            "draws-32-64.nc",
        ]
        (tmp_path / scenario / "draws-32-64.nc").unlink()

    calls: list[int] = []
    run = Simulator.run

    def counted(self: Simulator, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return run(self, *args, **kwargs)

    monkeypatch.setattr(Simulator, "run", counted)
    second = propagate(**arguments)
    assert len(calls) == 2  # the last round of both scenarios
    _assert_equal(first, second)
    assert second.record["history"] == first.record["history"]


def test_the_adaptive_propagation_is_unchanged_by_a_checkpoint(
    tmp_path: Path,
) -> None:
    arguments = _arguments(n=None, n_start=16, n_max=32, outputs=["pk.cmax"])
    _assert_equal(propagate(**arguments), propagate(**arguments, checkpoint=tmp_path))


def test_blocks_need_n() -> None:
    with pytest.raises(ValueError, match="number of draws"):
        Propagation(**_arguments(n=None, outputs=["pk.cmax"])).blocks()


def test_the_blocks_of_n() -> None:
    assert Propagation(**_arguments(n=24)).blocks() == [range(24)]
    assert Propagation(**_arguments(n=20, block_size=10)).blocks() == [
        range(10),
        range(10, 20),
    ]
    with pytest.raises(ValueError, match="block size"):
        Propagation(**_arguments(n=24, block_size=0))


@pytest.mark.parametrize(
    ("argument", "value"), [("n", 0), ("n_start", 0), ("n_max", -1)]
)
def test_numbers_of_draws_must_be_positive(argument: str, value: int) -> None:
    with pytest.raises(ValueError, match=f"`{argument}`"):
        Propagation(**_arguments(**{argument: value}))


def test_a_drawn_seed_is_kept() -> None:
    propagation = Propagation(**_arguments(n=4, seed=None))
    assert isinstance(propagation.seed, int)
    assert propagation.run().record["seed"] == propagation.seed


@pytest.mark.parametrize("block", [range(0), range(20, 30), range(0, 10, 2)])
def test_a_block_outside_the_draws_raises(block: range) -> None:
    with pytest.raises(ValueError, match="block"):
        Propagation(**_arguments(n=24)).run_block(block)


@pytest.mark.parametrize(
    ("name", "directory"),
    [
        ("high dose", "high%20dose"),
        (".", "%2E"),
        ("..", "%2E%2E"),
        ("a/b", "a%2Fb"),
        ("a\\b", "a%5Cb"),
        (".hidden", ".hidden"),
        ("dose 50%", "dose%2050%25"),
    ],
)
def test_a_scenario_directory_encodes_its_name(
    name: str, directory: str, tmp_path: Path
) -> None:
    simulation = _scenarios()["control"]
    arguments = _arguments(n=4, simulation={name: simulation}, checkpoint=tmp_path)
    first = propagate(**arguments)
    assert [p.name for p in tmp_path.iterdir()] == [directory]
    assert unquote(directory) == name
    _assert_equal(first, propagate(**arguments))


def test_a_scenario_with_a_checkpoint_needs_a_name(tmp_path: Path) -> None:
    simulation = _scenarios()["control"]
    with pytest.raises(ValueError, match="needs a name"):
        Propagation(**_arguments(n=4, simulation={"": simulation}, checkpoint=tmp_path))
    Propagation(**_arguments(n=4, simulation={"": simulation}))


def test_scenario_names_must_differ_in_more_than_case(tmp_path: Path) -> None:
    simulation = _scenarios()["control"]
    with pytest.raises(ValueError, match="case"):
        Propagation(
            **_arguments(
                n=4, simulation={"A": simulation, "a": simulation}, checkpoint=tmp_path
            )
        )


#: a distribution which is not relative, so the draws do not depend on the model
ABSOLUTE = UncertaintyModel(parameters={"ke": LogNormal(median=0.2, cv=0.2)})


@pytest.mark.parametrize(
    ("changed", "differ"),
    [
        ({"seed": 4}, "scan"),
        ({"observables": [], "keep": ["[C]"]}, "keep, observables"),
        ({"time": np.linspace(0, 24, 7)}, "time"),
        (
            {"simulator": Simulator(n_workers=1, relative_tolerance=1e-8)},
            "integrator_settings",
        ),
        ({"model": sbml_pk(ke=0.3)}, "model"),
    ],
)
def test_a_checkpoint_of_another_propagation_is_refused(
    changed: dict[str, Any], differ: str, tmp_path: Path
) -> None:
    propagate(**_arguments(n=8, uq=ABSOLUTE, checkpoint=tmp_path))
    path = tmp_path / "control" / "draws-0-8.nc"
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"'{path}' is a block of another propagation, it differs in {differ}:"
        ),
    ):
        propagate(**_arguments(n=8, uq=ABSOLUTE, checkpoint=tmp_path, **changed))


def test_a_checkpoint_keeps_no_attribute_of_its_own(tmp_path: Path) -> None:
    result = propagate(**_arguments(n=8, checkpoint=tmp_path))
    assert "propagation" not in result.results["control"].ds.attrs
    _assert_equal(propagate(**_arguments(n=8)), result)


def test_blocks_emit_events(tmp_path: Path) -> None:
    import json

    from sbmlsim import events

    path = tmp_path / "progress.jsonl"
    with events.recording(path):
        propagate(**_arguments(n=20, design="sobol", block_size=10))
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    blocks = [
        (x["scenario"], x["start"], x["stop"])
        for x in lines
        if x["event"] == "propagation.block"
    ]
    assert sorted(blocks) == sorted(
        (s, a, b) for s in ("control", "treatment") for a, b in ((0, 10), (10, 20))
    )


def test_a_block_is_checkpointed_once_every_scenario_has_its_file(
    tmp_path: Path,
) -> None:
    propagation = Propagation(
        **_arguments(n=4, block_size=2, checkpoint=tmp_path / "checkpoint")
    )
    first, second = propagation.blocks()
    assert not propagation.checkpointed(first)
    propagation.run_block(first)
    assert propagation.checkpointed(first)
    assert not propagation.checkpointed(second)
    # a block of one scenario only is not checkpointed
    next((tmp_path / "checkpoint").glob("*/draws-0-2.nc")).unlink()
    assert not propagation.checkpointed(first)
    assert not Propagation(**_arguments(n=4)).checkpointed(first)


def fails_for_a_large_dose(time: np.ndarray, values: dict[str, Any]) -> float:
    """Fail for a large concentration, i.e. a large dose."""
    if np.max(values["[C]"]) > 13.0:
        raise ValueError("a large dose")
    return 1.0


def _failing(**kwargs: Any) -> dict[str, Any]:
    """Get a propagation of an uncertain dose of which about half the draws fail."""
    return _arguments(
        simulation=Simulation(
            end=24, changes=[Change(0, {"PODOSE": Uncertain(Q(200, "mg"), cv=0.1)})]
        ),
        uq=UncertaintyModel(),
        observables=[
            *OBSERVABLES,
            Custom("check", fails_for_a_large_dose, "dimensionless", symbols=["[C]"]),
        ],
        keep=["[C]", "pk", "check"],
        **kwargs,
    )


@pytest.mark.parametrize(
    "case", ["grid", "ragged", "failing"], ids=["grid", "ragged", "failing"]
)
def test_a_run_of_blocks_writes_the_checkpoints_of_the_blocks(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Consecutive blocks run in one run per scenario, their files as run alone."""
    if case == "failing":
        n, size = 32, 4
        arguments = _failing(n=n, block_size=size)
    else:
        n, size = 12, 3
        steps = 24 if case == "grid" else None
        arguments = _arguments(
            n=n, design="lhs", simulation=_scenarios(steps), block_size=size
        )
    pooled_arguments: dict[str, Any] = {
        **arguments,
        "checkpoint": tmp_path / "pooled",
        "simulator": Simulator(n_workers=2),
    }
    pooled = Propagation(**pooled_arguments)
    blocks = pooled.blocks()
    assert len(blocks) == n // size
    calls: list[int] = []
    simulate = Simulator.run

    def counted(
        self: Simulator, model: Any, scan: Any, *args: Any, **kwargs: Any
    ) -> Any:
        calls.append(len(scan))
        return simulate(self, model, scan, *args, **kwargs)

    monkeypatch.setattr(Simulator, "run", counted)
    pooled.run_blocks(blocks)
    monkeypatch.undo()
    n_scenarios = len(pooled.scenarios)
    # one run of the draws per scenario, then the blocks of the failed draws
    # whose messages the run does not keep
    assert calls.count(n) == n_scenarios
    assert set(calls) <= {n, size}

    for block in blocks:
        Propagation(**arguments, checkpoint=tmp_path / "alone").run_block(block)
    pooled_files = sorted(
        p.relative_to(tmp_path / "pooled") for p in (tmp_path / "pooled").rglob("*.nc")
    )
    alone_files = sorted(
        p.relative_to(tmp_path / "alone") for p in (tmp_path / "alone").rglob("*.nc")
    )
    assert pooled_files == alone_files
    assert len(pooled_files) == len(blocks) * n_scenarios
    for path in pooled_files:
        xr.testing.assert_identical(
            ScanResult.from_netcdf(tmp_path / "pooled" / path).ds,
            ScanResult.from_netcdf(tmp_path / "alone" / path).ds,
        )
    unblocked: dict[str, Any] = {**arguments, "block_size": None}
    whole = propagate(**unblocked)
    _assert_equal(pooled.run(), whole)
    if case == "failing":
        failed = int(whole.results["default"].ds["status"].sum())
        assert failed > MAX_ERRORS
        assert size in calls
    if case == "ragged":
        sizes = {
            ScanResult.from_netcdf(tmp_path / "pooled" / path).ds.sizes["_point"]
            for path in pooled_files
        }
        assert len(sizes) > 1
