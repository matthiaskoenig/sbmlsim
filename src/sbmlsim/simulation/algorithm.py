"""Handling of algorithms and algorithm parameters."""

import logging
from typing import List, Union, Optional


from sbmlsim.simulation.base import BaseObject
from sbmlsim.simulation.kisaos import name_kisao, validate_kisao


logger = logging.getLogger(__name__)


class AlgorithmParameter(BaseObject):
    """AlgorithmParameter.
    
    The AlgorithmParameter class allows to parameterize a particular simulation algorithm. The set of
    possible parameters for a particular instance is determined by the algorithm that is referenced by the
    kisaoID of the enclosing algorithm element.

    TODO: add annotation
    https://identifiers.org/biomodels.kisao:KISAO_0000057
    """

    def __init__(
        self, kisao: str, value: Union[str, float], sid: str = None, name: str = None
    ):
        kisao = validate_kisao(kisao)
        name = name_kisao(kisao, name)

        super(AlgorithmParameter, self).__init__(sid=sid, name=name)
        self.kisao: str = kisao
        self.value: str = str(value)

    def __repr__(self) -> str:
        """Get string representation."""
        return f"AlgorithmParameter('{self.name}' = {self.value} | {self.kisao})"


class Algorithm(BaseObject):
    """Algorithm class.
    
    TODO: add annotation
    https://identifiers.org/biomodels.kisao:KISAO_0000057
    """
    def __init__(
        self, kisao: str, parameters: Optional[List[AlgorithmParameter]] = None, 
        sid: Optional[str] = None, name: Optional[str] = None,
    ):
        kisao: str = validate_kisao(kisao)
        name: str = name_kisao(kisao, name)

        super(Algorithm, self).__init__(sid, name)
        self.kisao: str = kisao
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
    ap = AlgorithmParameter(kisao="absolute tolerance", value=1e-7)
    print(ap)
