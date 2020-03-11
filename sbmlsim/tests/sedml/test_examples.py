from sbmlsim.combine.sedml.examples import create_dependent_variable, create_nested_algoritm

import pytest
import libsbml
import libsedml
import importlib


@pytest.fixture()
def resource():
    importlib.reload(libsedml)
    yield "resource"
    importlib.reload(libsbml)


def test_create_dependent_variable(resource):
    create_dependent_variable.create_dependent_variable_example()


