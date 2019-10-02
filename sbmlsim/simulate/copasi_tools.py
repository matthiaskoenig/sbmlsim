"""
Module for generating odesim config files for COPASI.

Config files are stored in ini format with different sections.
The set parameters are handeled in the [Parameters] section,
the Integration settings in the [Timecourse] section. Additional
information is stored in the [Simulation] section.

    ############################
    [Simulation]
    sbml = Dilution
    timecoure_id = tc1234
    pars_id = pars1234
    timestamp = 2014-03-27

    [Timecourse]
    t0 = 0.0
    dur = 100.0
    steps = 1000
    rTol = 1E-6
    aTol = 1E-6

    [Parameters]
    flow_sin = 60E-6
    PP__gal = 0.00012
    ############################
"""

from __future__ import print_function, division
import datetime
import time
import solve
import solve_io
# TODO: refactor and update with copasi python bindings

def create_config_file(sim, fname):
    """
    Creates a config file for the odesim in ini format.
    :param sim:
    :param fname:
    :return:
    """
    task = sim.task
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    
    # create the config sections
    lines = []
    lines += ['[Simulation]\n']
    lines += ["demo = {}\n".format(task.model.sbml_id)]
    lines += ["author = Matthias Koenig\n"]
    lines += ["time = {}\n".format(timestamp)]
    lines += ['Simulation = {}\n'.format(sim.pk)]
    lines += ["Task = {}\n".format(task.pk)]
    lines += ["SBML = {}\n".format(task.model.pk)]
    lines += ["\n"]
    
    lines += ["[Timecourse]\n"]
    lines += ["t0 = {}\n".format(task.tstart)]
    lines += ["dur = {}\n".format(task.tend)]
    lines += ["steps = {}\n".format(task.steps)]
    lines += ["rTol = {}\n".format(task.relTol)]
    lines += ["aTol = {}\n".format(task.absTol)]
    lines += ["\n"]
    
    lines += "[Parameters]\n"
    for p in sim.parameters.all():
        lines += ["{} = {}\n".format(p.key, p.value)]
    lines += ["\n"]
    
    lines += "[Settings]\n"
    for s in task.method.settings.all():
        lines += ["{} = {}\n".format(s.key, s.value)]
    
    # write the file
    with open(fname, 'w') as f:
        f.writelines(lines)
    f.close()
    return fname

def config_filename(sim, folder):
    sbml_id = sim.task.model.sbml_id
    return ''.join([folder, "/", sbml_id, "_Sim", str(sim.pk), '_config.ini'])


def solve_copasi(simulations):
    """ Integrate simulations with Copasi. """
    # TODO: Update to latest Copasi source & test.
    # TODO: Use the python interface to solve the problem.
    task = simulations[0].task

    filepath = task.model.filepath
    model_id = task.model.model_id
    for sim in simulations:
        try:
            sim.time_assign = timezone.now()                      # correction due to bulk assignment
            config_file = solve_io.store_config_file(sim, SIM_DIR)  # create the copasi config file for settings & changes
            csv_file = "".join([SIM_DIR, "/", str(sim.task), '/', model_id, "_Sim", str(sim.pk), '_copasi.csv'])

            # run an operating system command
            # call(["ls", "-l"])
            call_command = COPASI_EXEC + " -s " + filepath + " -c " + config_file + " -t " + csv_file
            print(call_command)
            call(shlex.split(call_command))

            solve_io.store_timecourse_db(sim, filepath=csv_file, ftype=solve_io.FileType.CSV)
        except Exception:
            solve.simulation_exception(sim)

################################################################################
if __name__ == "__main__":
    import django
    django.setup()
    
    from multiscale.multiscale_settings import SIM_DIR
    from simapp.models import Simulation
    
    sim = Simulation.objects.all()[0]
    fname = config_filename(sim, SIM_DIR) 
    create_config_file(sim, fname)
    print(fname)
