"""The credibility of many models in one table, `verify scorecard`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from sbmlsim.verify.common import UsageError, load_project_of, report

logger = logging.getLogger(__name__)


def evidence_extensions(args: argparse.Namespace) -> list[Any]:
    """Get the evidence extensions of the arguments.

    The extensions of the project (`--project`) come first, then those of
    `--extensions`, each a reference `module:attribute` to an extension or to
    an iterable of them.

    Raises:
        UsageError: for a reference which cannot be resolved, an object which is
            no extension, or two extensions of one name.
    """
    from sbmlsim.credibility.extensions import EvidenceExtension
    from sbmlsim.project import resolve

    extensions: list[Any] = []
    if args.project:
        extensions += load_project_of(args).evidence_extensions()
    for reference in args.extensions:
        try:
            found = resolve(reference)
        except (ValueError, ImportError, AttributeError) as err:
            raise UsageError(f"--extensions {reference}: {err}") from err
        extensions += [found] if isinstance(found, EvidenceExtension) else list(found)
    wrong = [e for e in extensions if not isinstance(e, EvidenceExtension)]
    if wrong:
        raise UsageError(f"{wrong!r} are no evidence extensions")
    names = [e.name for e in extensions]
    if repeated := sorted({n for n in names if names.count(n) > 1}):
        raise UsageError(f"the evidence extensions {repeated} are given twice")
    return extensions


def scorecard_command(options: argparse.Namespace) -> int:
    """Read the credibility records of many models, compare and qualify them.

    Prints the scorecard (model, context of use, verdict, verified), the pooled
    qualification (`--pool`) and the regressions against a baseline.

    Returns:
        0, 1 for a regression against the baseline.

    Raises:
        UsageError: for records, a baseline or extensions which cannot be read.
    """
    import pandas as pd

    from sbmlsim.credibility.scorecard import compare, load_records, pool, scorecard

    extensions = evidence_extensions(options)
    pooled_by = None
    if options.pool:
        pooled_by = {e.name: e for e in extensions}.get(options.pool)
        if pooled_by is None:
            raise UsageError(
                f"--pool {options.pool} is none of the extensions "
                f"{[e.name for e in extensions]}"
            )
    elif options.qualification:
        raise UsageError("--qualification needs --pool, the extension which pools")
    try:
        records = load_records(Path(path) for path in options.records)
        baseline = (
            None
            if options.baseline is None
            else pd.read_csv(options.baseline, sep="\t", keep_default_na=False)
        )
    except (FileNotFoundError, ValueError, KeyError) as err:
        raise UsageError(str(err)) from err
    table = scorecard(records, extensions)
    qualified = (
        None
        if pooled_by is None
        else pool(records, pooled_by, {"field": options.field})
    )
    if options.pool and qualified is None:
        raise UsageError(f"The extension {options.pool} pools no qualification.")
    if options.out:
        table.to_csv(options.out, sep="\t", index=False, lineterminator="\n")
    if options.qualification and qualified is not None:
        qualified.to_csv(
            options.qualification, sep="\t", index=False, lineterminator="\n"
        )
    report(f"{len(records)} models, {len(table)} contexts of use")
    for row in table.to_dict("records"):
        verified = "" if row["verified"] is None else f", verified {row['verified']}"
        report(f"  {row['model']}/{row['context']}: {row['verdict']}{verified}")
    if qualified is not None:
        _report_qualification(qualified, options.field)
    if baseline is None:
        return 0
    regressions = compare(baseline, table, options.tolerance, extensions)
    report(f"{len(regressions)} regressions against {options.baseline}")
    for regression in regressions:
        report(f"  {regression}")
    return 1 if regressions else 0


def _report_qualification(qualified: Any, field: str) -> None:
    """Report the pooled qualification of the ratio pairs by a field."""
    pooled = qualified[qualified["model"] == "all"]
    if len(pooled):
        report(f"qualification by {field}:")
        for row in pooled.to_dict("records"):
            report(
                f"  {row['value']}: {row['n_within']} of {row['n_pairs']} ratio "
                f"pairs within the limits of Guest ({row['within']:.0f} %), GMFE "
                f"{row['gmfe']:.3g}"
            )
