import libsbml

from sbmlsim.processing.mathml import formula_to_astnode, _get_variables

def test_variables_1():
    astnode = formula_to_astnode("x + y")
    variables = _get_variables(astnode)
    assert len(variables) == 2
    assert 'x' in variables
    assert 'y' in variables

def test_variables_2():
    astnode = formula_to_astnode("sin(x) + 2.0 * y/x * exp(10)")
    variables = _get_variables(astnode)
    print(variables)
    assert len(variables) == 2
    assert 'x' in variables
    assert 'y' in variables



