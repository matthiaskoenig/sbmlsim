"""A kind of the tests: squares of consecutive numbers, summed by the merge.

`DummySpec` sums the squares of `seed + i` for `i < n`; the plan draws the seed
and splits the numbers into `unit_blocks(n)`, a unit writes the squares of its
block into `parts/<index>.json` and the merge sums the parts into `total.json`.
The summary carries the thread variable `OMP_NUM_THREADS`, which the command
line sets. The kind is referenced as `tests.run.kinds_dummy:DUMMY`.
"""

import json
import os
import secrets
from pathlib import Path
from typing import Any

from sbmlsim import events
from sbmlsim.run import Kind, RunContext, SplitSpec
from sbmlsim.run.units import MissingUnits, unit_blocks


class DummySpec(SplitSpec):
    n: int
    seed: int | None = None
    data: Path | None = None
    fit: Path | None = None
    project: str | None = None
    fail: bool = False
    invalid: bool = False


def plan(spec: DummySpec) -> DummySpec:
    seed = spec.seed if spec.seed is not None else secrets.randbelow(2**31)
    return spec.model_copy(update={"seed": seed, "units": len(unit_blocks(spec.n))})


def _part(ctx: RunContext, index: int) -> Path:
    return ctx.out / "parts" / f"{index}.json"


def compute_unit(spec: DummySpec, index: int, ctx: RunContext) -> None:
    assert spec.seed is not None
    block = unit_blocks(spec.n)[index]
    ctx.write_json(f"parts/{index}.json", [(spec.seed + i) ** 2 for i in block])
    events.emit("dummy.unit", index=index)


def merge(spec: DummySpec, ctx: RunContext) -> dict[str, Any]:
    assert spec.units is not None
    missing = [i for i in range(spec.units) if not _part(ctx, i).exists()]
    if missing:
        raise MissingUnits(missing, spec.units)
    total = sum(
        sum(json.loads(_part(ctx, i).read_text(encoding="utf-8")))
        for i in range(spec.units)
    )
    ctx.write_json("total.json", total)
    return {
        "total": total,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mplbackend": os.environ.get("MPLBACKEND"),
        "workers": ctx.workers,
    }


def run(spec: DummySpec, ctx: RunContext) -> dict[str, Any]:
    if spec.fail:
        raise RuntimeError("the dummy kind fails")
    if spec.invalid:
        DummySpec.model_validate({"n": "no number"})
    assert spec.units is not None
    for index in range(spec.units):
        if not (ctx.resume and _part(ctx, index).exists()):
            compute_unit(spec, index, ctx)
    return merge(spec, ctx)


DUMMY = Kind(
    name="dummy",
    spec=DummySpec,
    run=run,
    plan=plan,
    compute_unit=compute_unit,
    merge=merge,
    inputs=("fit",),
    uses_cores=False,
)
