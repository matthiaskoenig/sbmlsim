"""The blocks of draws of a propagation and their checkpoints."""

from pathlib import Path
from typing import Any

import pytest
import xarray as xr

from sbmlsim import Q
from sbmlsim.simulation import PK, Change, Simulation
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


def _scenarios() -> dict[str, Simulation]:
    return {
        "control": Simulation(
            end=24, steps=24, changes=[Change(0, {"PODOSE": Q(100, "mg")})]
        ),
        "treatment": Simulation(
            end=24,
            steps=24,
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


@pytest.mark.parametrize("design", DESIGNS)
def test_blocks_equal_the_whole_propagation(design: str) -> None:
    whole = propagate(**_arguments(n=24, design=design))
    blocks = propagate(**_arguments(n=24, design=design, block_size=10))
    assert whole.results["control"].ds.sizes["draw"] == 24
    _assert_equal(whole, blocks)


@pytest.mark.parametrize("design", DESIGNS)
def test_blocks_of_separate_instances_merge(
    design: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    whole = propagate(**_arguments(n=24, design=design))
    blocks = Propagation(
        **_arguments(n=24, design=design, checkpoint=tmp_path, block_size=10)
    ).blocks()
    assert blocks == [range(0, 10), range(10, 20), range(20, 24)]  # noqa: PIE808
    for block in blocks:
        Propagation(
            **_arguments(n=24, design=design, checkpoint=tmp_path, block_size=10)
        ).run_block(block)
    assert sorted(p.name for p in (tmp_path / "control").iterdir()) == [
        "draws-0-10.nc",
        "draws-10-20.nc",
        "draws-20-24.nc",
    ]

    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("A checkpointed block is simulated again.")

    monkeypatch.setattr(Simulator, "run", fail)
    merged = Propagation(
        **_arguments(n=24, design=design, checkpoint=tmp_path, block_size=10)
    ).run()
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


def test_a_drawn_seed_is_kept() -> None:
    propagation = Propagation(**_arguments(n=4, seed=None))
    assert isinstance(propagation.seed, int)
    assert propagation.run().record["seed"] == propagation.seed


@pytest.mark.parametrize("block", [range(0), range(20, 30), range(0, 10, 2)])
def test_a_block_outside_the_draws_raises(block: range) -> None:
    with pytest.raises(ValueError, match="block"):
        Propagation(**_arguments(n=24)).run_block(block)


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b", ".hidden", "a b"])
def test_a_scenario_name_must_be_a_file_name(name: str, tmp_path: Path) -> None:
    simulation = _scenarios()["control"]
    with pytest.raises(ValueError, match="file name"):
        Propagation(
            **_arguments(n=4, simulation={name: simulation}, checkpoint=tmp_path)
        )
    # without a checkpoint the name is no file name
    Propagation(**_arguments(n=4, simulation={name: simulation}))


def test_scenario_names_must_differ_in_more_than_case(tmp_path: Path) -> None:
    simulation = _scenarios()["control"]
    with pytest.raises(ValueError, match="case"):
        Propagation(
            **_arguments(
                n=4, simulation={"A": simulation, "a": simulation}, checkpoint=tmp_path
            )
        )


def test_a_checkpoint_of_another_propagation_is_refused(tmp_path: Path) -> None:
    propagate(**_arguments(n=8, checkpoint=tmp_path))
    with pytest.raises(ValueError, match="another propagation"):
        propagate(**_arguments(n=8, seed=4, checkpoint=tmp_path))
