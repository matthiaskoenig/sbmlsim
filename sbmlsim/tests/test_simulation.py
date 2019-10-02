import os
import sbmlsim
from sbmlsim.tests.settings import DATA_PATH


def test_simulate():
    model_path = os.path.join(DATA_PATH, 'models', 'body19_livertoy_flat.xml')
    r = sbmlsim.load_model(model_path)
    s = sbmlsim.simulate(r, start=0, end=100, steps=100)
    assert s is not None
