"""Handling of algorithms and algorithm parameters."""

import logging

from pymetadata.ontologies import KISAO, KISAOType

from sbmlsim.simulation.base import BaseObject

logger = logging.getLogger(__name__)


class AlgorithmParameter(BaseObject):
    """AlgorithmParameter.

    The AlgorithmParameter class allows to parameterize a particular simulation
    algorithm. The set of possible parameters for a particular instance is determined
    by the algorithm that is referenced by the kisaoID of the enclosing algorithm
    element.
    """

    def __init__(
        self,
        kisao: KISAOType,
        value: str | float,
        sid: str = None,
        name: str = None,
    ):
        """Initialize AlgorithmParameter."""
        term: KISAO = KISAO.validate(kisao)
        term_name: str = KISAO.get_name(term)
        if name:
            if name != term_name:
                logger.warning("Using name '{name}' instead of '{term_name}'.")
            else:
                name = term_name

        super().__init__(sid=sid, name=name)
        self.kisao: KISAO = term
        self.value: str = str(value)

    def __repr__(self) -> str:
        """Get string representation."""
        return f"AlgorithmParameter('{self.name}' = {self.value} | {self.kisao})"


class Algorithm(BaseObject):
    """Algorithm class."""

    def __init__(
        self,
        kisao: KISAOType,
        parameters: list[AlgorithmParameter] | None = None,
        sid: str | None = None,
        name: str | None = None,
    ):
        """Initialize Algorithm."""
        term: KISAO = KISAO.validate(kisao)
        term_name: str = KISAO.get_name(term)
        if name:
            if name != term_name:
                logger.warning("Using name '{name}' instead of '{term_name}'.")
            else:
                name = term_name

        super().__init__(sid, name)
        self.kisao: KISAO = kisao
        self.parameters: list[AlgorithmParameter] | None = parameters

    def __repr__(self) -> str:
        """Get string representation."""
        return f"Algorithm({self.name}, {self.kisao}, parameters={self.parameters})"
