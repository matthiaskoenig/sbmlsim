"""
Perform ode integration.
Can use different simulation backends like RoadRunner or COPASI.

"""
from __future__ import print_function, division

import os
import sys
import traceback

from simapp.models import SimulatorType, MethodType, SimulationStatus

import solve_fba
import solve_io
import solve_ode
from multiscale.multiscale_settings import MULTISCALE_GALACTOSE_RESULTS


def run_simulations(simulations, task):
    """ Performs the simulations based on the given solver.
        Switches to the respective subcode for the individual solvers.
    """
    # switch method and simulatorType
    solve_io.create_simulation_directory(task)
    method_type = task.method.method_type

    if method_type == MethodType.FBA:
        solve_fba(simulations)
    elif method_type == MethodType.ODE:
        if task.integrator == SimulatorType.COPASI:
            raise NotImplemented
            # solve_ode.solve_copasi(simulations)
        elif task.integrator == SimulatorType.ROADRUNNER:
            solve_ode.solve_roadrunner(simulations)
    else:
        raise SimulationException('Method not supported: {}'.format(method_type))


#########################################################################
# Simulation Exceptions
#########################################################################
class SimulationException(Exception):
    """ Custom class for all Simulation problems. """
    pass


def simulation_exception(simulation):
    """ Handling exceptions during simulation.
    :param simulation: django Simulation object
    :return: None
    """

    print('-' * 60)
    print('*** Exception in ODE integration ***')

    path = os.path.join(MULTISCALE_GALACTOSE_RESULTS, 'ERROR_{}.log'.format(simulation.pk))
    with open(path, 'a') as f_err:
        traceback.print_exc(file=f_err)
    traceback.print_exc(file=sys.stdout)

    print('-'*60)

    # update simulation status
    simulation.status = SimulationStatus.ERROR
    simulation.save()


