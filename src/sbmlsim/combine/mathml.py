"""Helper functions for evaluation of MathML expressions.

Using sympy to evaluate the expressions.
"""
from typing import Any, Dict, Set, Tuple

import libsedml
from sbmlutils import log
from sympy import Symbol, lambdify, sympify


logger = log.get_logger(__name__)


def formula_to_astnode(formula: str) -> libsedml.ASTNode:
    """Parse ASTNode from formula."""
    astnode = libsedml.parseL3Formula(formula)
    if not astnode:
        logger.error("Formula could not be parsed: '{}'".format(formula))
        logger.error(libsedml.getLastParseL3Error())
    return astnode


def astnode_to_formula(astnode: libsedml.ASTNode) -> str:
    """Write ASTNode as formula."""
    formula = libsedml.formulaToL3String(astnode)
    return formula


def parse_mathml_str(mathml_str: str):
    """Parse MathML string."""
    astnode: libsedml.AstNode = libsedml.readMathMLFromString(mathml_str)
    return parse_astnode(astnode)


def parse_formula(formula: str) -> libsedml.ASTNode:
    """Parse formula to ASTNode."""
    astnode = formula_to_astnode(formula)
    return parse_astnode(astnode)


def parse_astnode(astnode: libsedml.ASTNode) -> Any:
    """Parse ASTNode.

    An AST node in libSBML is a recursive tree structure; each node has a type,
    a pointer to a value, and a list of children nodes. Each ASTNode node may
    have none, one, two, or more children depending on its type. There are
    node types to represent numbers (with subtypes to distinguish integer,
    real, and rational numbers), names (e.g., constants or variables),
    simple mathematical operators, logical or relational operators and
    functions.

    see also: http://sbml.org/Software/libSBML/docs/python-api/libsedml-math.html

    :param mathml:
    :return:
    """
    formula = libsedml.formulaToL3String(astnode)

    # iterate over ASTNode and figure out variables
    # variables = _get_variables(astnode)

    # create sympy expression
    expr = expr_from_formula(formula)

    # print(formula, expr)
    return expr


def expr_from_formula(formula: str):
    """Parse sympy expression from given formula string."""

    # [2] create sympy expressions with variables and formula
    # necessary to map the expression trees
    # create symbols
    formula = replace_piecewise(formula)
    formula = formula.replace("&&", "&")
    formula = formula.replace("||", "|")

    # additional methods
    # ns = {}
    # symbols = []
    # exec_('from sbmlsim.processing.mathml_functions import piecewise', ns)
    # from sympy import Symbol
    # for variable in sorted(variables):
    #    symbol = Symbol(variable)
    #    ns[variable] = symbol
    #    symbols.append(symbol)
    # expr = sympify(formula, locals=ns)
    expr = sympify(formula)

    return expr


def evaluate(astnode: libsedml.ASTNode, variables: Dict):
    """Evaluate the astnode with values."""
    expr = parse_astnode(astnode)
    f = lambdify(args=list(expr.free_symbols), expr=expr)
    res = f(**variables)
    return res


def _get_variables(astnode: libsedml.ASTNode, variables=None) -> Set[str]:
    """Add variable names to the variables."""
    variables: Set
    if variables is None:
        variables = set()

    num_children = astnode.getNumChildren()
    if num_children == 0:
        if astnode.isName():
            name = astnode.getName()
            variables.add(name)
    else:
        for k in range(num_children):
            child = astnode.getChild(k)  # type: libsedml.ASTNode
            _get_variables(child, variables=variables)

    return variables


def replace_piecewise(formula):
    """Replace libsedml piecewise with sympy piecewise."""
    while True:
        index = formula.find("piecewise(")
        if index == -1:
            break

        # process piecewise
        search_idx = index + 9

        # init counters
        bracket_open = 0
        pieces = []
        piece_chars = []

        while search_idx < len(formula):
            c = formula[search_idx]
            if c == ",":
                if bracket_open == 1:
                    pieces.append("".join(piece_chars).strip())
                    piece_chars = []
            else:
                if c == "(":
                    if bracket_open != 0:
                        piece_chars.append(c)
                    bracket_open += 1
                elif c == ")":
                    if bracket_open != 1:
                        piece_chars.append(c)
                    bracket_open -= 1
                else:
                    piece_chars.append(c)

            if bracket_open == 0:
                pieces.append("".join(piece_chars).strip())
                break

            # next character
            search_idx += 1

        # find end index
        if (len(pieces) % 2) == 1:
            pieces.append("True")  # last condition is True
        sympy_pieces = []
        for k in range(0, int(len(pieces) / 2)):
            sympy_pieces.append(f"({pieces[2*k]}, {pieces[2*k+1]})")
        new_str = f"Piecewise({','.join(sympy_pieces)})"
        formula = formula.replace(formula[index : search_idx + 1], new_str)

    return formula


if __name__ == "__main__":

    # Piecewise in sympy
    # https://docs.sympy.org/latest/modules/functions/elementary.html#piecewise
    # Piecewise((expr, cond), (expr, cond), … )
    # necessary to do a rewrite of the piecewise function
    expr = expr_from_formula("piecewise(8, x < 4, 0.1, (4 <= x) && (x < 6), 8)")
    expr = expr_from_formula(
        "Piecewise((8, x < 4), (0.1, (x >= 5) & (x < 6)), (8, True))"
    )

    print(expr)

    # evaluate expression
    expr = parse_formula("x + y")
    print(expr.free_symbols, type(expr))

    """
    # evaluate the function with the values
    astnode = libsedml.readMathMLFromString(mathmlStr)

    y = 5
    res = evaluateMathML(astnode,
                         variables={'x': y})
    print('Result:', res)
    """

    """
    * The Boolean function symbols '&&' (and), '||' (or), '!' (not),
    and '!=' (not equals) may be used.
    """
