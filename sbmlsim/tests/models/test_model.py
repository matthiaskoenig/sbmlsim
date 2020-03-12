import pytest

from sbmlsim.tests.constants import MODEL_REPRESSILATOR
from sbmlsim.models.model import Model


def test_model_creation():
    model = Model(sid="model1", source=MODEL_REPRESSILATOR)
    assert model
    assert model.sid == "model1"
    assert model.source == MODEL_REPRESSILATOR


def test_model_creation_with_changes():
    model = Model(sid="model1", source=MODEL_REPRESSILATOR,
                  changes=[])
    assert model
    assert len(model.changes) == 0
