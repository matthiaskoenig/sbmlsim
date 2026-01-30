"""Handling of algorithms and algorithm parameters."""

from typing import Optional, Union

from pymetadata.metadata import KISAO, KISAOType
from pymetadata import log

from sbmlsim.simulation.base import BaseObject


logger = log.get_logger(__name__)


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
        value: Union[str, float],
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

        super(AlgorithmParameter, self).__init__(sid=sid, name=name)
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
        parameters: Optional[list[AlgorithmParameter]] = None,
        sid: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """Initialize Algorithm."""
        term: KISAO = KISAO.validate(kisao)
        term_name: str = KISAO.get_name(term)
        if name:
            if name != term_name:
                logger.warning("Using name '{name}' instead of '{term_name}'.")
            else:
                name = term_name

        super(Algorithm, self).__init__(sid, name)
        self.kisao: KISAO = kisao
        self.parameters: Optional[list[AlgorithmParameter]] = parameters

    def __repr__(self) -> str:
        """Get string representation."""
        return f"Algorithm({self.name}, {self.kisao}, parameters={self.parameters})"
