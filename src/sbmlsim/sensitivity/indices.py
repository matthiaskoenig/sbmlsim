"""The sensitivity analyses on the result of a scan.

The analyses read the record of the design of a dimension of a result, see
`design_of`.
"""

from __future__ import annotations

from collections.abc import Collection

from sbmlsim.result import ScanResult
from sbmlsim.simulation.sampling import Design


def design_of(
    result: ScanResult, methods: Collection[str], dim: str | None = None
) -> tuple[str, Design]:
    """Find the dimension of a design of a method in a result.

    Args:
        result: the result of a scan with a design of `sbmlsim.simulation.sampling`.
        methods: the methods an analysis takes, e.g. `{"sobol"}`.
        dim: the id of the dimension, needed when there are several.

    Returns:
        The id of the dimension and its record.

    Raises:
        ValueError: if no dimension has a design of the methods, several have
            and `dim` is not given, or `dim` has none.
    """
    found = {
        str(d["id"]): Design.from_dict(d["design"])
        for d in result.ds.attrs.get("scan", {}).get("dimensions", [])
        if d.get("design") and d["design"]["method"] in methods
    }
    if dim is not None:
        if dim not in found:
            raise ValueError(
                f"The dimension '{dim}' has no design of {sorted(methods)}: "
                f"{sorted(found)}."
            )
        return dim, found[dim]
    if not found:
        raise ValueError(
            f"The result has no dimension with a design of {sorted(methods)}; "
            f"create one with sbmlsim.simulation.sampling."
        )
    if len(found) > 1:
        raise ValueError(
            f"The result has the designs {sorted(found)} of {sorted(methods)}; "
            f"choose one with dim=."
        )
    ((name, design),) = found.items()
    return name, design
