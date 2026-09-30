"""The `sbmlsim` extension of a PEtab v2 problem.

PEtab v2 problems carry extensions, i.e., a block in the YAML of the problem
which a tool reads if it knows it and ignores if it does not. The `sbmlsim`
extension holds what the tables of PEtab do not express, see
`sbmlsim.fit.petab_v2.gaps`: the units, the settings of the fit, what a fit
does with a mapping and the structure of the timecourses.

The extension is `required`, because the settings it carries are the objective
of the fit: a tool which does not know it has to reject the problem rather than
fit the same data with another objective without saying so. A round trip
through `sbmlsim` keeps the fit it started from, and
`to_petab(..., required_extension=False)` writes a problem which other tools
read and fit with the objective PEtab defines.
"""

import importlib.util
from collections.abc import Collection, Mapping
from typing import Any

from petab.v2.extensions import ExtensionConfig
from pydantic import Field

#: id of the extension, the key of the block in the YAML of the problem
EXTENSION_ID = "sbmlsim"

#: id of the extension of PEtab SciML, i.e. of the neural networks of a
#: hybrid problem, see `sbmlsim.fit.petab_v2.sciml`
SCIML_EXTENSION_ID = "sciml"

#: the extra of `sbmlsim` which reads the extension of PEtab SciML
SCIML_EXTRA = "pip install sbmlsim[sciml]"

#: the extensions the reader interprets without an extra. A problem which
#: requires another one is rejected, see `known_extensions` and
#: `check_extensions`
KNOWN_EXTENSIONS: frozenset[str] = frozenset({EXTENSION_ID})

#: version of the extension, raised when the block changes
EXTENSION_VERSION = "0.2.0"


class SbmlsimExtension(ExtensionConfig):
    """What PEtab v2 does not express about an `sbmlsim` fit.

    PEtab says that an extension which changes the mathematical interpretation
    of a problem must be `required`, and that a tool must reject a problem
    which requires an extension it does not know but may ignore one which is
    not required (PEtab v2, extensions). The settings this extension carries,
    i.e. the residual, the loss function and the weighting, are the objective
    `sbmlsim` optimizes, so a problem which is read without them is fitted with
    a different objective on the same data. It is therefore `required`, and a
    tool which does not know `sbmlsim` has to say so instead of fitting the
    problem differently without telling anyone.

    `to_petab(..., required_extension=False)` writes it as not required, which
    is what a problem meant for other tools wants: they read the tables and
    optimize the objective PEtab defines.

    Attributes:
        version: version of the extension.
        required: whether a tool needs the extension to interpret the problem,
            `True` because the settings it carries are the objective of the fit.
        opid: id of the optimization problem.
        settings: the `FitSettings` of the fit as a dictionary.
        parameters: unit, start value and, when set, the scale of every fit
            parameter which is no element of a network.
        observables: the fit mapping behind every observable, keyed by the fit
            mapping (an observable measured in several experiments is one
            observable of several fit mappings): the observable, its kind, the
            weight of the curve, the units of the data, the experiment and the
            task it belongs to and the metadata of the curve.
        experiments: the structure of the `TimecourseSim` behind every PEtab
            experiment, i.e. the timecourses with their steps and what is
            discarded, and the fit mapping collection it belongs to.
        collections: the `FitMappingCollection` objects of the fit, i.e. the id
            of the collection, the simulation experiment class its mappings
            come from and what the fit does with them.
        models: the settings of the integrator per model.
        inputs: the formula of an input of a network for the simulations
            without a formula of their own, in the math of PEtab, by the id
            of the input. The conditions of these experiments repeat it, see
            `sbmlsim.fit.petab_v2.sciml_export`.
        gaps: what this export lost, see `sbmlsim.fit.petab_v2.gaps`.
    """

    version: str = EXTENSION_VERSION
    required: bool = True

    opid: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, dict[str, Any]] = Field(default_factory=dict)
    observables: dict[str, dict[str, Any]] = Field(default_factory=dict)
    experiments: dict[str, dict[str, Any]] = Field(default_factory=dict)
    collections: dict[str, dict[str, Any]] = Field(default_factory=dict)
    models: dict[str, dict[str, Any]] = Field(default_factory=dict)
    inputs: dict[str, str] = Field(default_factory=dict)
    gaps: list[dict[str, Any]] = Field(default_factory=list)


def extension_of(config: Any) -> SbmlsimExtension | None:
    """Get the `sbmlsim` extension of a PEtab problem configuration.

    Args:
        config: `ProblemConfig` of a PEtab v2 problem, `None` if the problem
            was not read from a YAML file.

    Returns:
        The extension, or `None` if the problem does not carry one.
    """
    extensions = getattr(config, "extensions", None)
    if not extensions:
        return None
    extension = extensions.get(EXTENSION_ID)
    if extension is None:
        return None
    if isinstance(extension, SbmlsimExtension):
        return extension
    # the generic `ExtensionConfig` of a problem which was read from the YAML
    data = extension.model_dump() if hasattr(extension, "model_dump") else extension
    return SbmlsimExtension(**data)


def sciml_installed() -> bool:
    """Check whether the extra `sciml` is installed, i.e. `petab_sciml`."""
    return importlib.util.find_spec("petab_sciml") is not None


def known_extensions() -> frozenset[str]:
    """Get the extensions the reader interprets in this environment.

    Returns:
        `KNOWN_EXTENSIONS`, and the extension of PEtab SciML when the extra
        `sciml` is installed.
    """
    if sciml_installed():
        return KNOWN_EXTENSIONS | {SCIML_EXTENSION_ID}
    return KNOWN_EXTENSIONS


def check_extensions(
    extensions: Mapping[str, Any] | None,
    known: Collection[str] | None = None,
) -> list[str]:
    """Check the extensions of a problem against the ones the reader knows.

    PEtab says that a tool must reject a problem which requires an extension
    it does not know and may ignore an extension which is not required (PEtab
    v2, extensions). A block without `required` is read as required, which is
    the safe reading of a block that does not say.

    Args:
        extensions: the blocks of the problem by the id of the extension, as
            the dictionaries of the YAML or as the `ExtensionConfig` objects
            of a problem which was read, `None` for a problem without
            extensions.
        known: ids of the extensions the reader interprets,
            `known_extensions` by default.

    Returns:
        The ids of the extensions which are to be ignored, i.e. the ones
        which are not known and not required, in the order of the problem.
        The reader logs them.

    Raises:
        ImportError: if the problem requires the extension of PEtab SciML
            and the extra `sciml` is not installed. The message names the
            extra.
        ValueError: if the problem requires an extension which is not known.
    """
    if known is None:
        known = known_extensions()
    required: list[str] = []
    ignored: list[str] = []
    for extension_id, block in (extensions or {}).items():
        if extension_id in known:
            continue
        if isinstance(block, Mapping):
            is_required = block.get("required", True)
        else:
            is_required = getattr(block, "required", True)
        if is_required:
            required.append(extension_id)
        else:
            ignored.append(extension_id)

    if SCIML_EXTENSION_ID in required:
        raise ImportError(
            f"The PEtab problem requires the extension '{SCIML_EXTENSION_ID}', "
            f"i.e. it is a problem of PEtab SciML with neural networks. "
            f"`sbmlsim` reads it with the package 'petab_sciml', which is "
            f"installed with the extra 'sciml': {SCIML_EXTRA}"
        )
    if required:
        raise ValueError(
            f"The PEtab problem requires the extensions '{', '.join(required)}', "
            f"which `sbmlsim` does not know (it knows '{', '.join(sorted(known))}'). "
            f"A required extension changes the mathematical interpretation of a "
            f"problem, so the problem cannot be read without it."
        )
    return ignored
