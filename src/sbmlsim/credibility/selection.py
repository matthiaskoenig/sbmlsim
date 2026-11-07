"""The data points of the evidence of the contexts of use of a model.

`select` applies the declaration (`Credibility`) to the table of the adopted fits
(`FitTable`): the points of every evidence, the limitation points and the
validation points which no evidence claims. Everything is keyed by the
experiment and the key of a mapping, never by the key alone, which two studies
can share. The rules:

- A mapping which is training data of one adopted fit is fitted data in every
  fit: its points in another fit, where it may be validation data, are no
  validation either.
- The quantities of a context of use restrict its evidence to their observables,
  compared without the brackets of a concentration (`[A]` is `A`); an assessed
  ratio quantity by the observable of its arms, a declared ratio without a
  pairing restricts nothing.
- The evidence extensions check the decisive points of every context of use
  (`sbmlsim.credibility.extensions.EvidenceExtension.selection_problems`),
  e.g. a study of a ratio with more than one reference arm is an error.
- The limitations, of the model and of every context, are data the model does
  not describe: they are subtracted from every other evidence of every context
  and counted; a limitation of fitted data is an error.
- A validation of fitted data is an error, its outliers are dropped and counted.
- A calibration keeps the fitted points of its selection, it counts the
  subtracted limitation points it would have kept.
- `Unclaimed` takes the validation points which no limitation and no other
  evidence of any context selects; two `Unclaimed` of different contexts may
  take the same points.
- An evidence which selects no point is an error, and so is an evidence of a
  fit the model has not adopted.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from sbmlsim.credibility.declaration import (
    ContextOfUse,
    Credibility,
    Evidence,
    EvidenceRole,
    Unclaimed,
)

if TYPE_CHECKING:
    from sbmlsim.credibility.extensions import EvidenceExtension
    from sbmlsim.credibility.records import FitTable
    from sbmlsim.fit.helpers import MappingFilter

__all__ = [
    "EvidenceSelection",
    "SelectionError",
    "Selections",
    "normalize_observable",
    "select",
]

type _Mask = np.ndarray[tuple[int], np.dtype[np.bool_]]


def normalize_observable(observable: str) -> str:
    """Get an observable without the brackets of a concentration.

    Args:
        observable: the observable, e.g. `[A]`, `A` or `pk.cmax`.

    Returns:
        The observable, e.g. `A` for `[A]`.
    """
    if observable.startswith("[") and observable.endswith("]"):
        return observable[1:-1]
    return observable


@dataclass(frozen=True, kw_only=True, eq=False)
class EvidenceSelection:
    """The data points of an evidence of a context of use.

    Attributes:
        context: the id of the context of use.
        evidence: the evidence.
        rows: the selected rows of the table of the fits (`FitTable.table`).
        outliers_dropped: the points of kind outlier which a validation
            selected and dropped.
        limitation_subtracted: the limitation points the evidence selected and
            which were subtracted.
    """

    context: str
    evidence: Evidence
    rows: pd.Index
    outliers_dropped: int
    limitation_subtracted: int


@dataclass(frozen=True, kw_only=True, eq=False)
class Selections:
    """The data points of the evidence of every context of use of a model.

    Attributes:
        evidence: the selection of every evidence, in the order of the contexts
            and their evidence.
        limitations: the rows of the limitations of the model and of every
            context.
        unclaimed: the validation points of no evidence and no limitation, with
            the columns `fit`, `experiment`, `mapping` and `n` (the points).
        limitation_evidence: the selection of every limitation, the ones of the
            model first, with the context `""`, then the ones of the contexts.
    """

    evidence: tuple[EvidenceSelection, ...]
    limitations: pd.Index
    unclaimed: pd.DataFrame
    limitation_evidence: tuple[EvidenceSelection, ...] = ()


class SelectionError(ValueError):
    """The evidence of a declaration selects invalid data.

    Attributes:
        problems: every problem of the selection.
    """

    def __init__(self, problems: list[str]) -> None:
        """Create the error.

        Args:
            problems: every problem of the selection.
        """
        self.problems = list(problems)
        super().__init__(
            "The evidence of the credibility declaration selects invalid data:\n"
            + "\n".join(f"- {problem}" for problem in self.problems)
        )


class _Selector:
    """The masks of the rows of a table of the fits, collecting the problems."""

    def __init__(self, fits: FitTable) -> None:
        """Index the table.

        Args:
            fits: the table of the fits.
        """
        table = fits.table
        self.fits = fits
        self.problems: list[str] = []
        self.size = len(table)
        self.fit = table["fit"].astype(str).to_numpy()
        self.subsets = sorted(set(self.fit))
        self.experiments = table["experiment"].astype(str).to_numpy()
        self.mappings = table["mapping"].astype(str).to_numpy()
        self.observables = (
            table["observable"].astype(str).map(normalize_observable).to_numpy()
        )
        kinds = table["kind"].astype(str).to_numpy()
        self.training: _Mask = kinds == "training"
        self.validation: _Mask = kinds == "validation"
        self.outlier: _Mask = kinds == "outlier"
        self.pairs = list(zip(self.experiments, self.mappings, strict=True))
        self.unique_pairs = list(dict.fromkeys(self.pairs))
        # the fits a mapping is training data of, it is fitted in every fit
        self.trained_in: dict[tuple[str, str], set[str]] = {}
        for pair, fit, training in zip(
            self.pairs, self.fit, self.training, strict=True
        ):
            if training:
                self.trained_in.setdefault(pair, set()).add(str(fit))
        self.fitted: _Mask = np.fromiter(
            (p in self.trained_in for p in self.pairs), bool, self.size
        )

    def none(self) -> _Mask:
        """Get the mask of no row."""
        return np.zeros(self.size, dtype=bool)

    def rows(self, mask: _Mask) -> pd.Index:
        """Get the rows of a mask as the index of the table."""
        return self.fits.table.index[mask]

    def describe(self, mask: _Mask) -> str:
        """Get the mappings of the rows of a mask, e.g. `CredA.fm_dose (DECAY)`."""
        names = {
            f"{experiment}.{mapping} ({fit})"
            for experiment, mapping, fit in zip(
                self.experiments[mask], self.mappings[mask], self.fit[mask], strict=True
            )
        }
        return ", ".join(sorted(names))

    def describe_fitted(self, mask: _Mask) -> str:
        """Get the fitted mappings of the rows of a mask with the fits they are trained in.

        E.g. `CredB.fm_dose (PD, training in PK)`.
        """
        names = {
            f"{experiment}.{mapping} ({fit}, training in "
            f"{', '.join(sorted(self.trained_in[(experiment, mapping)]))})"
            for experiment, mapping, fit in zip(
                self.experiments[mask], self.mappings[mask], self.fit[mask], strict=True
            )
        }
        return ", ".join(sorted(names))

    def accepted(self, selection: MappingFilter) -> _Mask:
        """Get the rows of the mappings a selection accepts, one call per mapping."""
        keep: set[tuple[str, str]] = {
            (experiment, mapping)
            for experiment, mapping in self.unique_pairs
            if selection(mapping, self.fits.lookup(experiment, mapping))
        }
        return np.fromiter((p in keep for p in self.pairs), bool, self.size)

    def of_fit(self, evidence: Evidence, who: str) -> _Mask | None:
        """Get the rows of the fit of an evidence, `None` for an unknown fit."""
        if evidence.fit is None:
            return np.ones(self.size, dtype=bool)
        if evidence.fit not in self.subsets:
            self.problems.append(
                f"{who}: the fit '{evidence.fit}' is no subset of the adopted "
                f"fits, the subsets are {self.subsets}"
            )
            return None
        return self.fit == evidence.fit

    def searched(self, evidence: Evidence) -> list[str]:
        """Get the subsets of the fits an evidence is searched in."""
        return self.subsets if evidence.fit is None else [evidence.fit]

    def quantities(self, cou: ContextOfUse) -> _Mask:
        """Get the rows of the quantities of a context, a problem per unknown one.

        A ratio quantity is matched by the observable of its arms, a declared
        ratio without a pairing against no observable.
        """
        known = set(self.observables)
        absolute = [q for q in cou.quantities if not q.ratio]
        ids = {normalize_observable(q.id) for q in absolute}
        self.problems.extend(
            f"context of use '{cou.id}': the quantity '{q.id}' matches no "
            f"observable of the fits, the observables are {sorted(known)}"
            for q in absolute
            if normalize_observable(q.id) not in known
        )
        for q in cou.ratios:
            if q.pairing is None:
                continue
            observable = normalize_observable(q.pairing.observable)
            ids.add(observable)
            if observable not in known:
                self.problems.append(
                    f"context of use '{cou.id}': the ratio '{q.id}' matches no "
                    f"observable of the fits with '{q.pairing.observable}', the "
                    f"observables are {sorted(known)}"
                )
        return np.isin(self.observables, sorted(ids))

    def base(self, evidence: Evidence, who: str, quantities: _Mask) -> _Mask | None:
        """Get the rows of the mappings an evidence accepts.

        Restricted to the quantities and to the fit of the evidence; `None` for
        an evidence of an unknown fit.
        """
        fit = self.of_fit(evidence, who)
        if fit is None:
            return None
        if isinstance(evidence.selection, Unclaimed):
            raise TypeError(f"{who}: an unclaimed evidence has no selection")
        return self.accepted(evidence.selection) & quantities & fit

    def limitation(self, evidence: Evidence, who: str) -> _Mask:
        """Get the rows of a limitation, a problem for fitted data or none."""
        if isinstance(evidence.selection, Unclaimed):
            self.problems.append(
                f"{who}: a limitation selects mappings, it cannot be unclaimed"
            )
            return self.none()
        mask = self.base(evidence, who, np.ones(self.size, dtype=bool))
        if mask is None:
            return self.none()
        if not mask.any():
            self.problems.append(
                f"{who}: the limitation selects no point of the fits "
                f"{self.searched(evidence)}"
            )
        fitted = mask & self.fitted
        if fitted.any():
            self.problems.append(
                f"{who}: a limitation is data the model does not describe, but "
                f"it selects fitted data: {self.describe_fitted(fitted)}"
            )
        return mask

    def empty(self, mask: _Mask, cou: ContextOfUse, evidence: Evidence) -> None:
        """Add a problem if an evidence selects no point."""
        if not mask.any():
            self.problems.append(
                f"context of use '{cou.id}': the evidence '{evidence.name}' "
                f"selects no point of the fits {self.searched(evidence)}"
            )


def select(
    credibility: Credibility,
    fits: FitTable,
    extensions: Sequence[EvidenceExtension] = (),
) -> Selections:
    """Select the data points of the evidence of every context of use.

    A mapping (its experiment and key) which is training data of any adopted fit
    counts as fitted in every fit: a validation which selects its points, in
    any fit, is an error which names the fits it is training data of, and
    neither `Unclaimed` nor the unclaimed validation points take them. Two
    `Unclaimed` of different contexts may take the same points. See the module
    for the other rules.

    Args:
        credibility: the declaration.
        fits: the table of the adopted fits, see `evaluate_fits` and
            `mapping_table`.
        extensions: the evidence extensions of the project, which check the
            decisive points of every context of use.

    Returns:
        The points of every evidence, the limitation points and the unclaimed
        validation points.

    Raises:
        SelectionError: with every problem of the selection.
    """
    selector = _Selector(fits)
    limitation_evidence: list[EvidenceSelection] = []

    def limitation(context: str, evidence: Evidence, who: str) -> _Mask:
        """Get the rows of a limitation and keep its selection."""
        mask = selector.limitation(evidence, who)
        limitation_evidence.append(
            EvidenceSelection(
                context=context,
                evidence=evidence,
                rows=selector.rows(mask),
                outliers_dropped=0,
                limitation_subtracted=0,
            )
        )
        return mask

    # a limitation, of the model or of a context, is data the model does not
    # describe and no evidence of any context
    limitations = selector.none()
    for evidence in credibility.limitations:
        limitations |= limitation("", evidence, f"model limitation '{evidence.name}'")
    quantities: dict[str, _Mask] = {}
    for cou in credibility.contexts:
        for evidence in cou.limitations:
            limitations |= limitation(
                cou.id,
                evidence,
                f"context of use '{cou.id}': the limitation '{evidence.name}'",
            )
        quantities[cou.id] = selector.quantities(cou)

    # the evidence with a selection first, `Unclaimed` takes what it leaves
    selected: dict[tuple[int, int], EvidenceSelection] = {}
    claimed = selector.none()
    for i, cou in enumerate(credibility.contexts):
        for j, evidence in enumerate(cou.evidence):
            if isinstance(evidence.selection, Unclaimed):
                continue
            who = f"context of use '{cou.id}': the evidence '{evidence.name}'"
            if evidence.role is EvidenceRole.LIMITATION:
                selector.problems.append(
                    f"{who} has the role limitation, a limitation of a context "
                    f"is one of its limitations"
                )
                continue
            base = selector.base(evidence, who, quantities[cou.id])
            if base is None:
                continue
            subtracted = base & limitations
            rows = base & ~limitations
            outliers = selector.none()
            fitted = selector.none()
            if evidence.role is EvidenceRole.VALIDATION:
                fitted = rows & selector.fitted
                if fitted.any():
                    selector.problems.append(
                        f"{who} validates on fitted data: "
                        f"{selector.describe_fitted(fitted)}"
                    )
                outliers = rows & selector.outlier & ~selector.fitted
                rows = rows & selector.validation & ~selector.fitted
            else:
                # a calibration keeps training points, a limitation is none
                rows = rows & selector.training
                subtracted &= selector.training
            if not fitted.any():
                # a validation of fitted data alone is one problem, not two
                selector.empty(rows, cou, evidence)
            claimed |= rows
            selected[(i, j)] = EvidenceSelection(
                context=cou.id,
                evidence=evidence,
                rows=selector.rows(rows),
                outliers_dropped=int(outliers.sum()),
                limitation_subtracted=int(subtracted.sum()),
            )

    # the validation points of a mapping which another fit fitted are fitted
    unclaimed = selector.validation & ~selector.fitted & ~limitations & ~claimed
    for i, cou in enumerate(credibility.contexts):
        for j, evidence in enumerate(cou.evidence):
            if not isinstance(evidence.selection, Unclaimed):
                continue
            who = f"context of use '{cou.id}': the evidence '{evidence.name}'"
            fit = selector.of_fit(evidence, who)
            if fit is None:
                continue
            base = quantities[cou.id] & selector.validation & ~selector.fitted & fit
            within = evidence.selection.within
            if within is not None:
                base &= selector.accepted(within)
            rows = base & ~limitations & ~claimed
            selector.empty(rows, cou, evidence)
            unclaimed &= ~rows
            selected[(i, j)] = EvidenceSelection(
                context=cou.id,
                evidence=evidence,
                rows=selector.rows(rows),
                outliers_dropped=0,
                limitation_subtracted=int((base & limitations).sum()),
            )

    _check_extensions(credibility, fits, selected, selector, extensions)
    if selector.problems:
        raise SelectionError(selector.problems)
    return Selections(
        evidence=tuple(selected[key] for key in sorted(selected)),
        limitations=selector.rows(limitations),
        unclaimed=_count(fits.table, unclaimed),
        limitation_evidence=tuple(limitation_evidence),
    )


def _check_extensions(
    credibility: Credibility,
    fits: FitTable,
    selected: dict[tuple[int, int], EvidenceSelection],
    selector: _Selector,
    extensions: Sequence[EvidenceExtension],
) -> None:
    """Add the problems of the extensions with the decisive points of every context."""
    if not extensions:
        return
    for i, cou in enumerate(credibility.contexts):
        rows = pd.Index([])
        for (k, _), selection in selected.items():
            if k == i and selection.evidence.role is cou.decisive_role:
                rows = rows.union(selection.rows)
        decisive = fits.table.loc[rows]
        for extension in extensions:
            selector.problems.extend(
                f"context of use '{cou.id}': {problem}"
                for problem in extension.selection_problems(cou, decisive, fits.lookup)
            )


def _count(table: pd.DataFrame, mask: _Mask) -> pd.DataFrame:
    """Get the points of the rows of a mask per fit and mapping."""
    keys: list[Hashable] = ["fit", "experiment", "mapping"]
    return (
        table.loc[mask]
        .groupby(keys, sort=True)
        .size()
        .rename("n")
        .reset_index()
        .astype({"n": int})
    )
