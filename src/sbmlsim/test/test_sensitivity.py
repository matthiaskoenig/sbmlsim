import pytest

from sbmlsim.examples import example_sensitivity
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation.sensititvity import ModelSensitivity, SensitivityType
from sbmlsim.test import MODEL_REPRESSILATOR


def test_sensitivity_example():
    example_sensitivity.run_sensitivity()


def test_sensitivity():
    model = RoadrunnerSBMLModel(MODEL_REPRESSILATOR)

    p_ref = ModelSensitivity.reference_dict(
        model=model, stype=SensitivityType.PARAMETER_SENSITIVITY
    )
    assert len(p_ref) == 7
    p_keys = ["KM", "eff", "n", "ps_0", "ps_a", "tau_mRNA", "tau_prot"]
    for key in p_keys:
        assert key in p_ref

    s_ref = ModelSensitivity.reference_dict(
        model=model, stype=SensitivityType.SPECIES_SENSITIVITY
    )
    assert len(s_ref) == 1
    s_keys = ["Y"]
    for key in s_keys:
        assert key in s_ref

    all_ref = ModelSensitivity.reference_dict(
        model=model, stype=SensitivityType.All_SENSITIVITY
    )
    print(all_ref)
    assert len(all_ref) == 8
    all_keys = s_keys + p_keys
    for key in all_keys:
        assert key in all_ref


def test_sensitivity_change():
    model = RoadrunnerSBMLModel(MODEL_REPRESSILATOR)
    p_ref = ModelSensitivity.reference_dict(
        model=model, stype=SensitivityType.PARAMETER_SENSITIVITY
    )
    plus = ModelSensitivity.apply_change_to_dict(p_ref, change=0.1)
    minus = ModelSensitivity.apply_change_to_dict(p_ref, change=-0.1)
    for key in ["KM", "eff", "n", "ps_0", "ps_a", "tau_mRNA", "tau_prot"]:
        assert pytest.approx(1.1 * p_ref[key].magnitude, plus[key].magnitude)
        assert pytest.approx(0.9 * p_ref[key].magnitude, minus[key].magnitude)
