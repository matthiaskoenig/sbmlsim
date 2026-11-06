"""The actions of the toy project: a table of the rates of the variants."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sbmlsim.project import Project
from tests.project_toy.factory import RATES


def write_rates(
    project: Project, out: Path, /, *, workers: int, factor: float = 1.0
) -> Mapping[str, Any]:
    """Write the rates of the variants times a factor into `rates.tsv` of the out."""
    lines = ["variant\trate"]
    lines += [
        f"{name}\t{RATES[path.name] * factor:g}"
        for name, path in project.variants.items()
    ]
    (out / "rates.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": len(project.variants), "workers": workers}


def nothing(project: Project, out: Path, /, *, workers: int) -> None:
    """Do nothing, an action without a summary."""
