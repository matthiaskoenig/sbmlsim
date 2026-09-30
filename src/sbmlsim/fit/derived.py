"""Changes of a simulation which are calculated from the parameters of a fit.

A fit writes the values of its parameters into the model as the changes of a
simulation. A derived change is a change which is not a parameter but a
function of them: before every simulation the problem hands the values of the
parameters, of the changes of the simulation and of the model to the objects
it was given as `hybridizations`, and adds the changes they answer with to
the simulation.

`DerivedChanges` is what such an object provides. The neural networks of a
hybrid problem are the implementation, see
`sbmlsim.sciml.hybridization.Hybridization`: a network which runs before the
simulation calculates parameters and initial values of the model from
parameters of the fit.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class DerivedChanges(Protocol):
    """The changes of a simulation which follow from the values of a fit."""

    @property
    def model(self) -> str:
        """Get the id of the model in the experiment the changes belong to."""
        ...

    def symbols(self) -> Collection[str]:
        """Get the ids whose values `derived_changes` reads.

        Returns:
            The ids of entities of the model and of parameters of the fit
            which are not entities of the model.
        """
        ...

    def targets(self) -> Collection[str]:
        """Get the entities of the model `derived_changes` sets.

        Returns:
            The ids of the entities, or the selections of their
            concentrations.
        """
        ...

    def check_parameters(self, targets: Collection[str]) -> None:
        """Check the targets of the parameters of a fit.

        Args:
            targets: the entities the parameters of the fit write, without the
                prefix of a target which is not an entity of the model.

        Raises:
            ValueError: if the fit writes what the derived changes set or
                hold constant.
        """
        ...

    def derived_changes(
        self, values: Mapping[str, float], condition: str
    ) -> dict[str, float]:
        """Get the changes of a simulation for the values of the fit.

        Args:
            values: id -> value in the units of the model. The values are the
                ones of the parameters of the fit, of the changes of the
                simulation and of the model, in this order of precedence.
            condition: id of the simulation in its experiment.

        Returns:
            target -> value in the unit of the target in the model.

        Raises:
            ValueError: if a value is missing or a change is not a finite
                number.
        """
        ...
