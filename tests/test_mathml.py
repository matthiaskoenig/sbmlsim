"""Test MathML functionality."""

import numpy as np
import pytest

from sbmlsim.mathml import (
    _get_variables,
    evaluate,
    formula_to_astnode,
    parse_mathml_str,
)


def test_variables_1() -> None:
    """Test variables."""
    astnode = formula_to_astnode("x + y")
    variables = _get_variables(astnode)
    assert len(variables) == 2
    assert "x" in variables
    assert "y" in variables


def test_variables_2() -> None:
    """Test variables."""
    astnode = formula_to_astnode("sin(x) + 2.0 * y/x * exp(10)")
    variables = _get_variables(astnode)
    print(variables)
    assert len(variables) == 2
    assert "x" in variables
    assert "y" in variables


def test_evaluate() -> None:
    """Test evaluate."""
    astnode = formula_to_astnode("x + 2.5 * y")
    res = evaluate(astnode=astnode, variables={"x": 1.0, "y": 2.0})
    assert res == pytest.approx(6.0)


def test_evaluate_array() -> None:
    """Test evaluate array."""
    astnode = formula_to_astnode("x + 2.5 * y")
    res = evaluate(
        astnode=astnode,
        variables={
            "x": np.array([1.0, 2.0]),
            "y": np.array([2.0, 3.0]),
        },
    )
    assert np.allclose(res, np.array([6.0, 9.5]))


def test_mathml_str() -> None:
    """Test MathML string."""
    mathml_str = """
           <math xmlns="http://www.w3.org/1998/Math/MathML">
                <piecewise>
                  <piece>
                    <cn type="integer"> 8 </cn>
                    <apply>
                      <lt/>
                      <ci> x </ci>
                      <cn type="integer"> 4 </cn>
                    </apply>
                  </piece>
                  <piece>
                    <cn> 0.1 </cn>
                    <apply>
                      <and/>
                      <apply>
                        <leq/>
                        <cn type="integer"> 5 </cn>
                        <ci> x </ci>
                      </apply>
                      <apply>
                        <lt/>
                        <ci> x </ci>
                        <cn type="integer"> 6 </cn>
                      </apply>
                    </apply>
                  </piece>
                  <otherwise>
                    <cn type="integer"> 8 </cn>
                  </otherwise>
                </piecewise>
              </math>
    """
    _ = parse_mathml_str(mathml_str)


def test_max_and_min_of_one_array_reduce_it() -> None:
    """`max(Y)` of a function of data is the largest value of the array."""
    import numpy as np

    from sbmlsim import mathml

    y = np.array([1.0, 2.0, 4.0])
    ast = mathml.formula_to_astnode("Y/max(Y)")
    np.testing.assert_allclose(mathml.evaluate(ast, {"Y": y}), [0.25, 0.5, 1.0])
    ast = mathml.formula_to_astnode("Y - min(Y)")
    np.testing.assert_allclose(mathml.evaluate(ast, {"Y": y}), [0.0, 1.0, 3.0])
    ast = mathml.formula_to_astnode("max(Y, 2)")
    np.testing.assert_allclose(mathml.evaluate(ast, {"Y": y}), [2.0, 2.0, 4.0])


def test_max_of_a_padded_array_ignores_the_padding() -> None:
    """The padding of a ragged result is no value of `max`."""
    import numpy as np

    from sbmlsim import mathml

    y = np.array([1.0, 4.0, np.nan])
    ast = mathml.formula_to_astnode("Y/max(Y)")
    np.testing.assert_allclose(mathml.evaluate(ast, {"Y": y})[:2], [0.25, 1.0])
