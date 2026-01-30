"""Test Algorithm and AlgorithmParameters."""
import pytest

from sbmlsim.simulation.algorithm import KISAO, Algorithm, AlgorithmParameter, KISAOType


@pytest.mark.parametrize(
    "kisao, term",
    [
        ("KISAO:0000211", KISAO.KISAO_0000211),
        ("KISAO_0000211", KISAO.KISAO_0000211),
        (KISAO.KISAO_0000211, KISAO.KISAO_0000211),
        (KISAO.ABSOLUTE_TOLERANCE, KISAO.KISAO_0000211),
    ],
)
def test_algorithm_parameter(kisao: KISAOType, term: KISAO) -> None:
    """Test creation of AlgorithmParameters."""
    print(kisao)
    ap = AlgorithmParameter(sid=None, name=None, kisao=kisao, value=1e-7)
    assert ap
    assert isinstance(ap, AlgorithmParameter)
    assert ap.kisao == term
    assert ap.value == str(1e-7)


def test_algorithm() -> None:
    """Test creation of Algorithms."""

    algorithm = Algorithm(
        sid="algorithm",
        name="algorithm name",
        kisao=KISAO.CVODE,
        parameters=[
            AlgorithmParameter(kisao=KISAO.ABSOLUTE_TOLERANCE, value=1e-7),
            AlgorithmParameter(kisao=KISAO.RELATIVE_TOLERANCE, value=1e-7),
        ],
    )
    assert algorithm
    assert algorithm.sid == "algorithm"
    assert algorithm.name == "algorithm name"
    assert len(algorithm.parameters) == 2
    for p in algorithm.parameters:
        assert p
        assert isinstance(p, AlgorithmParameter)
