"""The kinds of the models and the actions of a project.

Every spec names a project, `project` (`module:attribute` of a
`sbmlsim.project.Project`).

- `factory` (`FactorySpec`) builds the models of the project with its factory
  into the out, with the layout of the package (`Project.create_models(out)`:
  the factory writes into `out / factory_dir.relative_to(root)`), and compares
  every output of the factory (`Project.compared_outputs`: the declared
  `factory_outputs`, else the variants and every file the factory wrote which
  the package has, with a warning) with the one of the package, an SBML file
  without the dates of its model history (`Project.changed_outputs`). The
  files of the package are never written; the summary names the number of the
  compared outputs, whether they are declared and the outputs which differ,
  `changed`, so a run shows whether the committed models are the ones the
  factory builds; a rebuild with nothing to compare fails.
- `action` (`ActionSpec`) runs an action of the project (`Project.actions`),
  an analysis of its own, with the arguments of the spec as keywords, see
  `sbmlsim.project.ActionFunction`; its summary is the mapping the action
  returns with the name of the action, `action`. The plan checks that the project has the
  action and that its function takes the arguments, so a misspelled argument
  is a configuration error before anything runs.

The modules of the factory and of the actions are imported when a kind plans
or runs, not when this module is imported.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import Field

from sbmlsim.run.kinds import (
    ConfigurationError,
    Kind,
    Spec,
    load_project_ref,
    register,
)

if TYPE_CHECKING:
    from sbmlsim.project import ActionFunction, Project
    from sbmlsim.run.context import RunContext

__all__ = [
    "ACTION",
    "FACTORY",
    "ActionSpec",
    "FactorySpec",
]

logger = logging.getLogger(__name__)


class FactorySpec(Spec):
    """The spec of a build of the models of a project with its factory.

    Attributes:
        project: the project, `module:attribute`.
    """

    project: str


class ActionSpec(Spec):
    """The spec of a run of an action of a project.

    Attributes:
        project: the project, `module:attribute`.
        name: the name of the action, a key of `Project.actions`.
        arguments: the arguments of the action, JSON values by the names of
            the keywords of its function; they are passed as they are, a
            relative path among them is not resolved against the directory of
            a spec file but relative to the current directory of the run.
    """

    project: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


# ----------------------------------------------------------------------------
# factory
# ----------------------------------------------------------------------------
def _factory_project(spec: FactorySpec) -> Project:
    """Get the project of a build and check that it has a factory.

    Raises:
        ConfigurationError: for a project without a factory, whose factory is
            not resolved or which has no directory of its factory.
    """
    project = load_project_ref(spec.project)
    try:
        project.load_factory()
    except (ImportError, AttributeError, TypeError, ValueError) as err:
        raise ConfigurationError(
            f"The models of '{spec.project}' are not built: {err}"
        ) from err
    if project.factory_dir is None:
        raise ConfigurationError(
            f"The project '{spec.project}' has neither a directory of its factory "
            "nor model variants: give factory_dir."
        )
    return project


def _plan_factory(spec: FactorySpec) -> FactorySpec:
    """Check that the project has a factory, the spec is planned as given."""
    _factory_project(spec)
    return spec


def _run_factory(spec: FactorySpec, ctx: RunContext) -> dict[str, Any]:
    """Build the models into the out and compare the outputs with the package.

    Raises:
        RuntimeError: if the rebuild has nothing to compare, neither declared
            outputs, variants nor a file of the package.
    """
    project = _factory_project(spec)
    directory = project.create_models(ctx.out)
    declared = bool(project.factory_outputs)
    if not declared:
        logger.warning(
            "The project '%s' declares no factory_outputs: the rebuild compares "
            "the variants and the files the factory wrote which the package has.",
            spec.project,
        )
    compared = project.compared_outputs(ctx.out)
    if not compared:
        raise RuntimeError(
            f"The rebuild of '{spec.project}' has nothing to compare: the project "
            "declares no factory_outputs and no variants, and the factory wrote no "
            "file of the package."
        )
    changed = [path.as_posix() for path in project.changed_outputs(ctx.out)]
    for path in changed:
        logger.warning("The output '%s' of the factory differs from the package.", path)
    return {
        "directory": directory.relative_to(ctx.out).as_posix(),
        "outputs": len(compared),
        "declared": declared,
        "changed": changed,
    }


# ----------------------------------------------------------------------------
# action
# ----------------------------------------------------------------------------
def _action(spec: ActionSpec) -> tuple[Project, ActionFunction]:
    """Get the project and the function of the action of a spec.

    Raises:
        ConfigurationError: for an unknown action, a function which is not
            resolved and arguments the function does not take.
    """
    project = load_project_ref(spec.project)
    try:
        function = project.load_action(spec.name)
    except KeyError as err:
        raise ConfigurationError(str(err.args[0])) from err
    except (ImportError, AttributeError, TypeError, ValueError) as err:
        raise ConfigurationError(
            f"The action '{spec.name}' of '{spec.project}' is not resolved: {err}"
        ) from err
    try:
        signature = inspect.signature(function)
    except TypeError, ValueError:
        return project, function
    try:
        signature.bind(project, project.root, workers=1, **spec.arguments)
    except TypeError as err:
        raise ConfigurationError(
            f"The action '{spec.name}' of '{spec.project}' does not take the "
            f"arguments {sorted(spec.arguments)}: {err}; an action is called as "
            "`function(project, out, *, workers, **arguments)`."
        ) from err
    return project, function


def _plan_action(spec: ActionSpec) -> ActionSpec:
    """Check the action and its arguments, the spec is planned as given."""
    _action(spec)
    return spec


def _run_action(spec: ActionSpec, ctx: RunContext) -> dict[str, Any]:
    """Run the action into the out.

    Raises:
        TypeError: if the action returns neither a mapping nor `None`.
    """
    project, function = _action(spec)
    workers = ctx.workers if ctx.workers is not None else 1
    result = function(project, ctx.out, workers=workers, **spec.arguments)
    if result is None:
        return {"action": spec.name}
    if not isinstance(result, Mapping):
        raise TypeError(
            f"The action '{spec.name}' returned a {type(result).__name__}, not "
            "its summary, a mapping, or None."
        )
    return {**result, "action": spec.name}


#: the models of a project built by its factory and compared with the package
FACTORY: Kind[FactorySpec] = Kind(
    name="factory",
    spec=FactorySpec,
    run=_run_factory,
    plan=_plan_factory,
    uses_cores=False,
)

#: an action of a project
ACTION: Kind[ActionSpec] = Kind(
    name="action", spec=ActionSpec, run=_run_action, plan=_plan_action
)

for _kind in (FACTORY, ACTION):
    register(_kind)
