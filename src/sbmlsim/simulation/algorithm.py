"""Handling of algorithms and algorithm parameters.

1. global algorithm parameters
2. simulation algorithm parameters
3. nested lists
4. kisao terms

FIXME: add annotation
	https://identifiers.org/biomodels.kisao:KISAO_0000057
"""
import logging
import re
from typing import List, Union

from kisao import Kisao, utils
from kisao.data_model import AlgorithmSubstitutionPolicy
from pronto import Term

from sbmlsim.simulation.base import BaseObject


logger = logging.getLogger(__name__)
kisao_ontology = Kisao()
kisao_pattern = re.compile(r"^KISAO_\d{7}$")


kisao_lookup = {
    "absolute tolerance": "KISAO_0000211",
}


def validate_kisao(kisao: str) -> str:
    """Validates and normalizes kisao id against pattern."""
    if not kisao.startswith("KISAO"):
        # try lookup by name
        kisao = kisao_lookup.get(kisao, kisao)

    if kisao.startswith("KISAO:"):
        kisao = kisao.replace(":", "_")

    if not re.match(kisao_pattern, kisao):
        raise ValueError(
            f"kisao '{kisao}' does not match kisao pattern " f"'{kisao_pattern}'."
        )

    # term = kisao_ontology.get_term(kisao)
    # check that term exists

    return kisao


def name_kisao(kisao: str, name: str = None) -> str:
    """Get name for kisao id."""
    term: Term = kisao_ontology.get_term(kisao)
    if not term:
        raise ValueError(f"No term in kisao ontology for: '{kisao}'")

    kisao_name = term.name
    if kisao_name != term.name:
        logger.warning(f"Name '{name}' does not match kisao name '{kisao_name}'")
    if name:
        return name
    else:
        return kisao_name


class AlgorithmParameter(BaseObject):
    """AlgorithmParameter."""

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
    def __init__(
        self, sid: str, name: str, kisao: str, parameters: List[AlgorithmParameter]
    ):
        kisao = validate_kisao(kisao)
        name = name_kisao(kisao, name)

        super(Algorithm, self).__init__(sid, name)
        self.kisao: str = kisao
        self.parameters: List[AlgorithmParameter] = parameters

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
