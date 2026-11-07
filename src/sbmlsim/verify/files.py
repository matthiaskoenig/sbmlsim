"""Compare PNG figures and SBML models of two directories, `verify png` and `verify sbml`."""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable
from pathlib import Path

import libsbml
import numpy as np

from sbmlsim.verify.common import UsageError, report


def _files(directory: Path, pattern: str) -> dict[Path, Path]:
    """Get the files below a directory by their path relative to it."""
    return {p.relative_to(directory): p for p in sorted(directory.rglob(pattern))}


def _directories(old: Path, new: Path) -> str | None:
    """Get the message about a directory which does not exist, or `None`."""
    for directory in (old, new):
        if not directory.is_dir():
            return f"{directory} is not a directory"
    return None


def png(args: argparse.Namespace) -> int:
    """Compare the PNG figures of two directories pixel by pixel.

    The directories are searched recursively and the figures are paired by the
    path relative to the directory; with `--sids` an old figure is paired with
    the new figure of the SId of its file name (`sbmlsim.experiment.to_sid`).

    Returns:
        0 if all figures are identical, 1 if some differ or exist on one side
        only.

    Raises:
        UsageError: if a directory does not exist, no figure was found or two
            old figures have the same SId.
    """
    from PIL import Image

    old, new = Path(args.old), Path(args.new)
    if message := _directories(old, new):
        raise UsageError(message)

    def selected(path: Path) -> bool:
        return not args.names or any(s in path.name for s in args.names)

    old_files = {k: v for k, v in _files(old, "*.png").items() if selected(v)}
    new_files = {k: v for k, v in _files(new, "*.png").items() if selected(v)}
    if args.sids:
        from sbmlsim.experiment import to_sid

        by_sid: dict[Path, Path] = {}
        for key, path in old_files.items():
            renamed = key.with_name(to_sid(key.stem) + key.suffix)
            if renamed in by_sid:
                raise UsageError(f"two old figures have the same SId: {renamed}")
            by_sid[renamed] = path
        old_files = by_sid
    if not old_files and not new_files:
        raise UsageError(f"no PNG figures in {old} and {new}")
    same = differing = 0
    for key in sorted(old_files.keys() | new_files.keys()):
        if key not in new_files:
            report(f"only in old: {key}")
            differing += 1
            continue
        if key not in old_files:
            report(f"only in new: {key}")
            differing += 1
            continue
        a = np.asarray(Image.open(old_files[key]))
        b = np.asarray(Image.open(new_files[key]))
        if a.shape == b.shape and np.array_equal(a, b):
            same += 1
            continue
        fraction = (a != b).any(axis=-1).mean() if a.shape == b.shape else 1.0
        report(f"differs: {key} {a.shape} {b.shape}, pixels {fraction:.2e}")
        differing += 1
    report(f"{same} identical, {differing} differ")
    return 1 if differing else 0


# --- sbml -----------------------------------------------------------------------


def sbml_elements(path: Path) -> dict[str, object] | None:
    """Get the elements of an SBML model which determine its simulation.

    Returns:
        The elements by kind and id, `None` if the file has no SBML model (the
        layouts the factories write next to the models are no SBML).
    """

    def formula(math_: libsbml.ASTNode | None) -> str | None:
        return libsbml.formulaToL3String(math_) if math_ is not None else None

    def references(items: Iterable[libsbml.SpeciesReference]) -> tuple:
        return tuple(sorted((r.getSpecies(), r.getStoichiometry()) for r in items))

    # the document owns the model, it has to stay referenced while the model is read
    document = libsbml.readSBMLFromFile(str(path))
    model = document.getModel()
    if model is None:
        return None
    elements: dict[str, object] = {}
    for p in model.getListOfParameters():
        # an unset value is NaN, as a value of NaN
        value = p.getValue()
        elements[f"parameter {p.getId()}"] = (value, p.getConstant(), p.getUnits())
    for s in model.getListOfSpecies():
        elements[f"species {s.getId()}"] = (
            s.getInitialAmount() if s.isSetInitialAmount() else None,
            s.getInitialConcentration() if s.isSetInitialConcentration() else None,
            s.getCompartment(),
            s.getHasOnlySubstanceUnits(),
            s.getBoundaryCondition(),
            s.getConstant(),
            s.getSubstanceUnits(),
        )
    for c in model.getListOfCompartments():
        size = c.getSize()
        elements[f"compartment {c.getId()}"] = (size, c.getConstant(), c.getUnits())
    for r in model.getListOfRules():
        elements[f"rule {r.getVariable()} {r.getTypeCode()}"] = formula(r.getMath())
    for a in model.getListOfInitialAssignments():
        elements[f"initial assignment {a.getSymbol()}"] = formula(a.getMath())
    for r in model.getListOfReactions():
        law = r.getKineticLaw()
        elements[f"reaction {r.getId()}"] = (
            formula(law.getMath()) if law else None,
            references(r.getListOfReactants()),
            references(r.getListOfProducts()),
            tuple(sorted(m.getSpecies() for m in r.getListOfModifiers())),
        )
    for e in model.getListOfEvents():
        elements[f"event {e.getId()}"] = (
            formula(e.getTrigger().getMath()),
            tuple(
                (a.getVariable(), formula(a.getMath()))
                for a in e.getListOfEventAssignments()
            ),
        )
    for u in model.getListOfUnitDefinitions():
        elements[f"unit {u.getId()}"] = tuple(
            sorted(
                (x.getKind(), x.getExponent(), x.getScale(), x.getMultiplier())
                for x in u.getListOfUnits()
            )
        )
    return {k: _nan_equal(v) for k, v in elements.items()}


def _nan_equal(value: object) -> object:
    """Replace NaN by a string, so that NaN equals NaN."""
    if isinstance(value, tuple):
        return tuple(_nan_equal(v) for v in value)
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def sbml(args: argparse.Namespace) -> int:
    """Compare SBML models: two files, or the files of the same path in two dirs.

    The directories are searched recursively and the files are paired by the
    path relative to the directory.

    Returns:
        0 if the models are equal, 1 if they differ or exist on one side only.

    Raises:
        UsageError: if a directory does not exist or no file was found.
    """
    old, new = Path(args.old), Path(args.new)
    if old.is_dir() or new.is_dir():
        if message := _directories(old, new):
            raise UsageError(message)
        names = sorted(_files(old, "*.xml").keys() | _files(new, "*.xml").keys())
        if not names:
            raise UsageError(f"no XML files in {old} and {new}")
        pairs = [(old / name, new / name) for name in names]
    else:
        pairs = [(old, new)]
    differing = 0
    for a_path, b_path in pairs:
        label = str(a_path.relative_to(old)) if old.is_dir() else a_path.name
        if not a_path.exists() or not b_path.exists():
            report(f"{label}: only in {'new' if b_path.exists() else 'old'}")
            differing += 1
            continue
        a, b = sbml_elements(a_path), sbml_elements(b_path)
        if a is None or b is None:
            report(f"{label}: no SBML model")
            continue
        changed = [k for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)]
        report(f"{label}: {len(changed)} of {len(a.keys() | b.keys())} differ")
        for k in changed:
            report(f"  {k}\n    old: {a.get(k)}\n    new: {b.get(k)}")
        differing += len(changed)
    return 1 if differing else 0


# --- scorecard ------------------------------------------------------------------
