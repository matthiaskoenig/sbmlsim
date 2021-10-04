"""Handling of algorithms and algorithm parameters."""

from typing import List, Optional, Union

from pymetadata.metadata import KISAO, KISAOType
from sbmlutils import log

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
        self, kisao: KISAOType, value: Union[str, float], sid: str = None, name: str = None
    ):
        term: KISAO = KISAO.validate(kisao)
        name: str = KISAO.get_name(term)

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
        parameters: Optional[List[AlgorithmParameter]] = None,
        sid: Optional[str] = None,
        name: Optional[str] = None,
    ):
        term: KISAO = KISAO.validate(kisao)
        term_name: str = KISAO.name_kisao(term)
        if name:
            if name != term_name:
                logger.warning("Using name")

        super(Algorithm, self).__init__(sid, name)
        self.kisao: KISAO = kisao
        self.parameters: Optional[List[AlgorithmParameter]] = parameters

    def __repr__(self) -> str:
        """Get string representation."""
        return f"Algorithm({self.name}, {self.kisao}, parameters={self.parameters})"


if __name__ == "__main__":
    ap = AlgorithmParameter(sid=None, name=None, kisao="KISAO:0000211", value=1e-7)
    print(ap)
    ap = AlgorithmParameter(sid=None, name=None, kisao="KISAO_0000211", value=1e-7)
    print(ap)
    ap = AlgorithmParameter(kisao="KISAO:0000211", value=1e-7)
    print(ap)
    ap = AlgorithmParameter(kisao=KISAO.KISAO_0000211, value=1e-7)
    print(ap)
    ap = AlgorithmParameter(kisao=KISAO.ABSOLUTE_TOLERANCE, value=1e-7)
    print(ap)
