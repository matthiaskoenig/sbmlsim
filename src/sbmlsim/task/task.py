"""Tasks."""


class Task:
    """Tasks combine models with simulations.

    This allows to execute the same simulation with different
    model variants.
    """

    def __init__(
        self,
        model: str,
        simulation: str,
        sid: str | None = None,
        name: str | None = None,
    ):
        """Initialize Task.

        Args:
            model: Key of the model.
            simulation: Key of the simulation.
            sid: Identifier of the task.
            name: Name of the task.

        Raises:
            ValueError: If model or simulation are not string keys.
        """
        if not isinstance(model, str):
            raise ValueError(
                f"Reference to a model must be a string model key, "
                f"but found: '{model}' of type '{type(model)}'"
            )

        if not isinstance(simulation, str):
            raise ValueError(
                f"Reference to a simulation must be a string "
                f"simulation key, "
                f"but found: '{simulation}' of type '{type(simulation)}'"
            )

        self.model_id = model
        self.simulation_id = simulation

        self.sid = sid if sid else f"{model}__{simulation}"
        self.name = name

    def __repr__(self) -> str:
        """Get representation."""
        return f"Task(model={self.model_id} simulation={self.simulation_id})"

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary.

        Returns:
            Dictionary with model and simulation keys.
        """
        return {
            "model": self.model_id,
            "simulation": self.simulation_id,
        }
