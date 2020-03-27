"""
Document the loss function to handle outliers.

 The purpose of the loss function rho(s) is to reduce the influence of
    outliers on the solution.
"""



class AdjustableParameter():
    def __init__(self, sid, min=None, max=None, x0=None):
        self.sid = sid
        self.min = min
        self.max = max
        self.x0 = x0


class ParameterFitting(object):

    def __init__(self, fit_mappings, adjustable_parameters):
        self.fit_mappings = fit_mappings
        self.adjustable_parameters = adjustable_parameters




    def simulation_experiments(self):
        # number of simulation experiments
        -> defines simulations


    def residuals():
        """
        Calculates for every simulation experiment the residuals between
        Simulation and data.
        :return:
        """
        pass




    def optimize(self):


