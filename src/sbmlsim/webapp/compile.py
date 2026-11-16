"""The SBML of a model compiled to WebAssembly with sbmlsim-core.

The compiler of sbmlsim-core (`sbmlsim_core.compile_sbml`, the python package
`sbmlsim-core`) reads SBML level 3 core, flat and without a required package.
`prepare_sbml` brings a model there as sbmlsim simulates it: a hierarchical
model is flattened (`comp`), the uncertainties of `distrib` (the distributions
of the fitted parameters which a factory writes into a model) are stripped,
since they are metadata of the parameters and change no simulation, and a
model below level 3 is converted to level 3 version 2. A math which uses a
distribution function of `distrib` would change a simulation, so such a model
is refused, naming the element.

`compile_sbml` compiles the prepared model into `model.wasm` and its
description `model.json` (the contract `model/2` of sbmlsim-core), which
`write_compiled` writes into a directory, the layout of `sbmlsim-core compile`
which the JavaScript host loads. The compiled model is deterministic: the same
SBML and the same release of sbmlsim-core give the same bytes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import libsbml

from sbmlsim.model.model_sbml import _flatten, _is_hierarchical, to_level3

__all__ = [
    "DESCRIPTION_FILE",
    "WASM_FILE",
    "CompiledModel",
    "WebAppError",
    "compile_sbml",
    "prepare_sbml",
    "write_compiled",
]

#: the module of a compiled model in its directory
WASM_FILE: Final = "model.wasm"
#: the description of a compiled model in its directory
DESCRIPTION_FILE: Final = "model.json"
#: the URL of the csymbols of the distribution functions of `distrib`
_DISTRIB_SYMBOLS: Final = "http://www.sbml.org/sbml/symbols/distrib/"


class WebAppError(ValueError):
    """A model or a project the web app cannot be written of."""


@dataclass(frozen=True)
class CompiledModel:
    """A model compiled by sbmlsim-core.

    Attributes:
        wasm: the bytes of `model.wasm`.
        description: the text of `model.json`, its description `model/2`.
    """

    wasm: bytes
    description: str


def _element_name(element: libsbml.SBase) -> str:
    """Get the type and the id of an element, its target or its nearest parent with one."""
    kind = element.getElementName()
    for getter in ("getVariable", "getSymbol"):
        target = getattr(element, getter, None)
        if target is not None and target():
            return f"{kind} of '{target()}'"
    while element is not None:
        if element.isSetIdAttribute() and element.getIdAttribute():
            owner = element.getElementName()
            name = f"{owner} '{element.getIdAttribute()}'"
            return name if owner == kind else f"{kind} of {name}"
        element = element.getParentSBMLObject()
    return kind


def _in_distrib(element: libsbml.SBase | None) -> bool:
    """Check whether an element is one of `distrib` or inside one."""
    while element is not None:
        if element.getPackageName() == "distrib":
            return True
        element = element.getParentSBMLObject()
    return False


def _distrib_functions(doc: libsbml.SBMLDocument) -> list[str]:
    """Get the elements whose math uses a distribution function of `distrib`."""
    found = []
    elements: libsbml.SBaseList = doc.getListOfAllElements()
    for index in range(elements.getSize()):
        element: libsbml.SBase = elements.get(index)
        is_set = getattr(element, "isSetMath", None)
        if is_set is None or not is_set() or _in_distrib(element):
            continue
        mathml = libsbml.writeMathMLToString(element.getMath()) or ""  # ty: ignore[missing-attribute]
        position = mathml.find(_DISTRIB_SYMBOLS)
        if position >= 0:
            function = mathml[position + len(_DISTRIB_SYMBOLS) :].split('"', 1)[0]
            found.append(f"{_element_name(element)} ({function})")
    return found


def prepare_sbml(sbml: str, *, base_path: Path | None = None) -> str:
    """Prepare the SBML of a model for the compiler of sbmlsim-core.

    Args:
        sbml: the SBML of the model.
        base_path: the directory the external models of a hierarchical model
            are relative to, the current directory without.

    Returns:
        The SBML flattened, without `distrib`, in level 3 version 2 if it was
        below level 3, see the module.

    Raises:
        WebAppError: if libsbml cannot read, flatten or convert the model, or
            if a math uses a distribution function of `distrib`.
    """
    try:
        if _is_hierarchical(sbml):
            sbml = _flatten(sbml, None, base_path)
        sbml = to_level3(sbml, "model")
    except ValueError as err:
        raise WebAppError(str(err)) from err
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml)
    if doc.getModel() is None:
        raise WebAppError("libsbml cannot read the model.")
    used = _distrib_functions(doc)
    if used:
        raise WebAppError(
            "A math of the model uses a distribution function of distrib, which "
            f"would change its simulation: {', '.join(used)}."
        )
    stripped = False
    for index in reversed(range(doc.getNumPlugins())):
        plugin: libsbml.SBasePlugin = doc.getPlugin(index)
        uri = plugin.getURI()
        if "/distrib/" in uri:
            status = doc.enablePackage(uri, plugin.getPrefix(), False)
            if status != libsbml.LIBSBML_OPERATION_SUCCESS:
                raise WebAppError(f"The package distrib ({uri}) cannot be stripped.")
            stripped = True
    return str(libsbml.writeSBMLToString(doc)) if stripped else sbml


def compile_sbml(sbml: str, *, base_path: Path | None = None) -> CompiledModel:
    """Compile a model with sbmlsim-core after preparing it, see the module.

    Args:
        sbml: the SBML of the model.
        base_path: the directory of the external models of a hierarchical
            model, see `prepare_sbml`.

    Returns:
        The compiled model.

    Raises:
        WebAppError: if the model cannot be prepared or compiled, with the
            message of the compiler, which names the feature it refuses.
    """
    import sbmlsim_core

    prepared = prepare_sbml(sbml, base_path=base_path)
    try:
        compiled = sbmlsim_core.compile_sbml(prepared)
    except sbmlsim_core.CompileError as err:
        raise WebAppError(f"sbmlsim-core cannot compile the model: {err}") from err
    description = json.dumps(
        compiled.description, indent=1, ensure_ascii=False, allow_nan=False
    )
    return CompiledModel(wasm=compiled.wasm, description=f"{description}\n")


def write_compiled(compiled: CompiledModel, directory: Path) -> Path:
    """Write a compiled model into a directory, `model.wasm` and `model.json`.

    Args:
        compiled: the compiled model.
        directory: the directory, created if it does not exist.

    Returns:
        The directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / WASM_FILE).write_bytes(compiled.wasm)
    (directory / DESCRIPTION_FILE).write_text(compiled.description, encoding="utf-8")
    return directory
