"""
Helper functions for evaluation of MathML expressions.
Using sympy to evalutate the expressions.
"""
from typing import Set, Dict, Tuple
import logging

import libsbml
from sympy import Symbol, sympify
from sympy.core.compatibility import exec_


def formula_to_astnode(formula: str):
    astnode = libsbml.parseL3Formula(formula)
    if not astnode:
        logging.error("Formula could not be parsed: '{}'".format(formula))
        logging.error(libsbml.getLastParseL3Error())
    return astnode


def parse_formula(formula: str):
    astnode = formula_to_astnode(formula)
    return parse_astnode(astnode)


def parse_mathml_str(mathml_str: str):
    astnode = libsbml.readMathMLFromString(mathml_str)  # type: libsbml.ASTNode
    return parse_astnode(astnode)


def parse_astnode(astnode: libsbml.ASTNode):
    """
    An AST node in libSBML is a recursive tree structure; each node has a type,
    a pointer to a value, and a list of children nodes. Each ASTNode node may
    have none, one, two, or more children depending on its type. There are
    node types to represent numbers (with subtypes to distinguish integer,
    real, and rational numbers), names (e.g., constants or variables),
    simple mathematical operators, logical or relational operators and
    functions.

    see also: http://sbml.org/Software/libSBML/docs/python-api/libsbml-math.html

    :param mathml:
    :return:
    """

    formula = libsbml.formulaToL3String(astnode)
    formula = formula.replace("piecewise", 'Piecewise')
    # formula = formula.replace("&&", 'and')
    # formula = formula.replace("||", 'or')

    # TODO: some rewrites necessary for Sympy
    # Piecewise

    print(formula)

    # [1] iterate over ASTNode and figure out variables
    variables = _get_variables(astnode)
    print(variables)


    # [2] create sympy expressions with variables and formula
    # necessary to map the expression trees
    # create symbols

    # additional methods
    ns = {}
    exec_('from sbmlsim.processing.mathml_functions import piecewise', ns)
    # FIXME: rewrite the piecwise function

    from sympy import Symbol
    for variable in variables:
        ns[variable] = Symbol(variable)

    expr = sympify(formula, locals=ns)
    print(expr)

    return expr


def evaluate(astnode, variables={}, array=False):
    """Evaluate the astnode with values """
    pass


def _get_variables(astnode: libsbml.ASTNode, variables=None) -> Set:
    """Adds variable names to the variables."""
    if variables is None:
        variables = set()

    num_children = astnode.getNumChildren()
    if num_children == 0:
        if astnode.isName():
            name = astnode.getName()
            variables.add(name)
    else:
        for k in range(num_children):
            child = astnode.getChild(k)  # type: libsbml.ASTNode
            _get_variables(child, variables=variables)

    return variables



if __name__ == "__main__":

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
                        <cn type="integer"> 4 </cn>
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
    parse_mathml_str(mathml_str)

    astnode = formula_to_astnode("x + y")
    variables = _get_variables(astnode)
    print(variables)

    expr = parse_formula("x + y")
    print(type(expr))



    '''
    # evaluate the function with the values
    astnode = libsbml.readMathMLFromString(mathmlStr)

    y = 5
    res = evaluateMathML(astnode,
                         variables={'x': y})
    print('Result:', res)
    '''