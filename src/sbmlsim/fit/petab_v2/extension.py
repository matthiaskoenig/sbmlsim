"""The `sbmlsim` extension of a PEtab v2 problem.

PEtab v2 problems carry extensions, i.e., a block in the YAML of the problem
which a tool reads if it knows it and ignores if it does not. The `sbmlsim`
extension holds what the tables of PEtab do not express, see
`sbmlsim.fit.petab_v2.gaps`: the units, the settings of the fit, what a fit
does with a mapping and the structure of the timecourses.

A PEtab problem with this extension is a valid PEtab problem, i.e., other tools
read the model, the measurements and the parameters as usual, and a round trip
through `sbmlsim` keeps the fit it started from.
"""

from typing import Any

from petab.v2.extensions import ExtensionConfig
from pydantic import Field

#: id of the extension, the key of the block in the YAML of the problem
EXTENSION_ID = "sbmlsim"

#: version of the extension, raised when the block changes
EXTENSION_VERSION = "0.1.0"


class SbmlsimExtension(ExtensionConfig):
    """What PEtab v2 does not express about an `sbmlsim` fit.

    PEtab says that an extension which changes the mathematical interpretation
    of a problem must be `required`, and that a tool must reject a problem
    which requires an extension it does not know but may ignore one which is
    not required (PEtab v2, extensions). This extension is on the line: the
    tables alone are a valid PEtab problem of the same data, but its settings,
    i.e. the residual, the loss function and the weighting, are the objective
    `sbmlsim` optimizes, so a tool which ignores them fits the same data with a
    different objective.

    `required` is therefore a choice of the export and defaults to `False`,
    which keeps the problem portable: another tool fits it with the objective
    PEtab defines, which is a different fit of the same data. `required=True`
    is the strict reading of the specification, where a tool which does not
    know `sbmlsim` must reject the problem instead of fitting it differently.

    Attributes:
        version: version of the extension.
        required: whether the problem needs it to be interpreted, always
            `False`, the tables alone are a valid problem.
        opid: id of the optimization problem.
        settings: the `FitSettings` of the fit as a dictionary.
        parameters: unit and start value per fit parameter.
        observables: the fit mapping behind every observable, i.e. its kind, the
            weight of the curve, the units of the data, the experiment and the
            task it belongs to and the metadata of the curve.
        experiments: the structure of the `TimecourseSim` behind every PEtab
            experiment, i.e. the timecourses with their steps and what is
            discarded, and the fit mapping collection it belongs to.
        collections: the `FitMappingCollection` objects of the fit, i.e. the id
            of the collection, the simulation experiment class its mappings
            come from and what the fit does with them.
        models: the settings of the integrator per model.
        gaps: what this export lost, see `sbmlsim.fit.petab_v2.gaps`.
    """

    version: str = EXTENSION_VERSION
    required: bool = False

    opid: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, dict[str, Any]] = Field(default_factory=dict)
    observables: dict[str, dict[str, Any]] = Field(default_factory=dict)
    experiments: dict[str, dict[str, Any]] = Field(default_factory=dict)
    collections: dict[str, dict[str, Any]] = Field(default_factory=dict)
    models: dict[str, dict[str, Any]] = Field(default_factory=dict)
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
