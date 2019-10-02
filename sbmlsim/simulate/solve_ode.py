"""
Perform ode integrations with roadrunner for simulations in the database.
"""

from __future__ import print_function, division
import time

from django.utils import timezone
from roadrunner import SelectionRecord
from simapp.models import ParameterType, SettingKey, SimulationStatus, ResultType

import roadrunner_tools as rt
import solve
import solve_io


def solve_roadrunner(simulations):
    """ Integrate simulations with RoadRunner.

    :param simulations: list of database simulations belonging to same task.
    :return: None
    """
    try:
        # read SBML
        comp_model = simulations[0].task.model
        rr = rt.MyRunner(comp_model.filepath)
    except RuntimeError:
        for sim in simulations:
            solve.simulation_exception(sim)
        raise

    # set the selection
    # FIXME:  this has to be provided from the outside (must be part of the simulation), i.e.
    sel = ['time'] \
        + ["".join(["[", item, "]"]) for item in rr.model.getBoundarySpeciesIds()] \
        + ["".join(["[", item, "]"]) for item in rr.model.getFloatingSpeciesIds()] \
        + [item for item in rr.model.getReactionIds() if item.startswith('H')]
    rr.selections = sel

    # integrator settings
    settings = simulations[0].task.method.get_settings_dict()
    rr.set_integrator_settings(
        variable_step_size=settings[SettingKey.VAR_STEPS],
        stiff=settings[SettingKey.STIFF],
        absolute_tolerance=settings[SettingKey.ABS_TOL],
        relative_tolerance=settings[SettingKey.REL_TOL]
    )
    # simulations
    for sim in simulations:
        _solve_roadrunner_single(rr, sim,
                                 start=settings[SettingKey.T_START],
                                 end=settings[SettingKey.T_END],
                                 steps=settings.get(SettingKey.STEPS, None))


def _solve_roadrunner_single(rr, sim, start, end, steps=None, hdf5=True, csv=False):
    """ Integrate single roadrunner simulation from start to end.

    :param rr: MyRunner instance with loaded model
    :param sim: django simulation
    :param start: start time
    :param end: end time
    :param hdf5: create hdf5 output
    :param csv: create csv output
    :return:
    """
    # TODO: refactor with roadrunner tools (as much logic as possible there)
    try:
        tstart_total = time.time()
        sim.time_assign = timezone.now()  # correction due to bulk assignment

        # make a concentration backup
        conc_backup = rr.store_concentrations()

        # set all parameters in the model and store the changes for revert
        changes = dict()
        for p in sim.parameters.all():
            if p.parameter_type == ParameterType.GLOBAL_PARAMETER:
                name = str(p.key)
                changes[name] = rr.model[name]
                rr.model[name] = p.value
                # print('set', name, ' = ', p.value)

        # recalculate the initial assignments
        rr.reset(SelectionRecord.INITIAL_GLOBAL_PARAMETER)

        # restore concentrations
        rr._set_concentrations(conc_backup)

        # apply concentration changes
        for p in sim.parameters.all():
            if p.parameter_type in [ParameterType.NONE_SBML_PARAMETER, ParameterType.GLOBAL_PARAMETER]:
                continue

            name = str(p.key)
            if p.parameter_type == ParameterType.BOUNDARY_INIT:
                name = '[{}]'.format(name)
            elif p.parameter_type == ParameterType.FLOATING_INIT:
                name = 'init([{}])'.format(name)

            changes[name] = rr.model[name]
            rr.model[name] = p.value

        # ode integration
        tstart_int = time.time()
        if steps is not None:
            s = rr.simulate(start=start, end=end, steps=steps)
        else:
            s = rr.simulate(start=start, end=end)
        time_integration = time.time() - tstart_int

        # complete model reset
        rr.reset(SelectionRecord.ALL)

        # store CSV
        if csv:
            csv_file = solve_io.csv_file(sim)
            solve_io.save_csv(csv_file, data=s, header=rr.selections)
            solve_io.store_result_db(sim, filepath=csv_file, result_type=ResultType.CSV)

        # store HDF5
        if hdf5:
            h5_file = solve_io.hdf5_file(sim)
            solve_io.save_hdf5(h5_file, data=s, header=rr.selections)
            solve_io.store_result_db(sim, filepath=h5_file, result_type=ResultType.HDF5)

        # update simulation
        sim.time_sim = timezone.now()
        sim.status = SimulationStatus.DONE
        sim.save()
        time_total = time.time()-tstart_total

        print('Time: [{:.4f}|{:.4f}]'.format(time_total, time_integration))

    except:
        solve.simulation_exception(sim)
