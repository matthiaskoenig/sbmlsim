from pint import UnitRegistry

from sbmlsim.data import Data





class Function(object):
    """ Functional data calculation.

    The idea ist to provide an object which can calculate a generic math function
    based on given input symbols.

    Important challenge is to handle the correct functional evaluation.
    """
    index = "abs_glc_bw",  # name to access (something not existing in data or task)

    f = "absorption_glc" / self.Q_(70, "kg"),  # when is this evaluated? How does this work with units
    task = "task_ogtt",
    dataset = None,
    data = {
        'absorption_glc': Data(self, index="absorption_glc", task="task_ogtt")
    }

if __name__ == "__main__":
    # TODO: example