#!/usr/bin/python
"""
Module for running/starting fitting.

Starts processes on the n_cores which listen for available simulations.
The odesim settings and parameters determine the actual odesim.
The simulator supports parallalization by putting different simulations
on different CPUs. 

-------------------------------------------------------------------------------------
How multiprocessing works, in a nutshell:

    Process() spawns (fork or similar on Unix-like systems) a copy of the 
    original program.
    The copy communicates with the original to figure out that 
        (a) it's a copy and 
        (b) it should go off and invoke the target= function (see below).
    At this point, the original and copy are now different and independent, 
    and can run simultaneously.

Since these are independent processes, they now have independent Global Interpreter Locks 
(in CPython) so both can use up to 100% of a CPU on a multi-cpu box, as long as they don't 
contend for other lower-level (OS) resources. That's the "multiprocessing" part.
-------------------------------------------------------------------------------------
"""
import logging
import multiprocessing
import os
import numpy as np
import socket
from sbmlsim.fit.fit import run_optimization, analyze_optimization
from sbmlsim.fit import FitExperiment, FitParameter
from sbmlsim.simulator import SimulatorSerial

from sbmlsim.fit.optimization import OptimizationProblem, SamplingType, OptimizerType
from sbmlsim.fit.analysis import OptimizationResult
from sbmlsim.examples.experiments.midazolam.experiments.mandema1992 import Mandema1992
from sbmlsim.utils import timeit
from sbmlsim.examples.experiments.midazolam import MIDAZOLAM_PATH
RESULTS_PATH = MIDAZOLAM_PATH / "results"
DATA_PATH = MIDAZOLAM_PATH / "data"


from sbmlsim.examples.experiments.midazolam.experiments.mandema1992 import Mandema1992


logger = logging.getLogger(__name__)


op_kwargs1 = {
    "opid": "mid1oh_iv",
    "base_path": MIDAZOLAM_PATH,
    "data_path": DATA_PATH,
    "fit_experiments": [
            FitExperiment(experiment=Mandema1992, mappings=["fm4"])
        ],
    "fit_parameters": [
            # distribution parameters
            FitParameter(parameter_id="ftissue_mid1oh", start_value=1.0,
                         lower_bound=1, upper_bound=1E5,
                         unit="liter/min"),
            FitParameter(parameter_id="fup_mid1oh", start_value=0.1,
                         lower_bound=0.01, upper_bound=0.3,
                         unit="dimensionless"),
            # mid1oh kinetics
            FitParameter(parameter_id="KI__MID1OHEX_Vmax", start_value=100,
                         lower_bound=1E-1, upper_bound=1E4,
                         unit="mmole/min"),
        ]
}

op_mandema1992 = {
    "opid": "mandema1992",
    "base_path": MIDAZOLAM_PATH,
    "data_path": DATA_PATH,
    "fit_experiments": [
        # FitExperiment(experiment=Mandema1992, mappings=["fm1"]),
        FitExperiment(experiment=Mandema1992, mappings=["fm1", "fm3", "fm4"]),
    ],
    "fit_parameters": [
        # liver
        FitParameter(parameter_id="LI__MIDIM_Vmax", start_value=0.1,
                     lower_bound=1E-3, upper_bound=1E6,
                     unit="mmole_per_min"),
        FitParameter(parameter_id="LI__MID1OHEX_Vmax", start_value=0.1,
                     lower_bound=1E-3, upper_bound=1E6,
                     unit="mmole_per_min"),
        FitParameter(parameter_id="LI__MIDOH_Vmax", start_value=100,
                     lower_bound=10, upper_bound=200, unit="mmole_per_min"),
        # kidneys
        FitParameter(parameter_id="KI__MID1OHEX_Vmax", start_value=100,
                     lower_bound=1E-1, upper_bound=1E4,
                     unit="mmole/min"),

        # distribution
        FitParameter(parameter_id="ftissue_mid", start_value=2000,
                      lower_bound=1, upper_bound=1E5,
                      unit="liter/min"),
        FitParameter(parameter_id="fup_mid", start_value=0.1,
                      lower_bound=0.05, upper_bound=0.3,
                      unit="dimensionless"),
        # distribution parameters
        FitParameter(parameter_id="ftissue_mid1oh", start_value=1.0,
                     lower_bound=1, upper_bound=1E5,
                     unit="liter/min"),
        FitParameter(parameter_id="fup_mid1oh", start_value=0.1,
                     lower_bound=0.01, upper_bound=0.3,
                     unit="dimensionless"),
    ],
}


def worker(size, seed):
    """ Creates a worker for the cpu which listens for available simulations. """
    ip = get_ip_address()

    while True:
        print('{:<20} <Run fit>'.format(ip))

        simulator = SimulatorSerial()
        # op = OptimizationProblem(simulator=simulator, **op_kwargs)
        op = OptimizationProblem(simulator=simulator, **op_mandema1992)

        opt_res = run_optimization(
            op, size=size, seed=seed,
            verbose=False,
            optimizer=OptimizerType.LEAST_SQUARE,
            sampling=SamplingType.LOGUNIFORM_LHS,
            diff_step=0.05,
        )
        return opt_res

@timeit
def fit_parallel(n_cores: int, size=30):
    # Lock for syncronization between processes (but locks)
    lock = multiprocessing.Lock()

    pool = multiprocessing.Pool(processes=n_cores)
    sizes = [size]*n_cores
    seeds = list(np.random.randint(low=1, high=2000, size=n_cores))

    # mapping multiple arguments
    opt_results = pool.starmap(worker, zip(sizes, seeds))

    # single worker
    # opt_results = pool.map(worker, sizes)

    # combine simulation results
    return OptimizationResult.combine(opt_results)


def get_ip_address():
    """ Returns the IP adress for the given computer. """
    return socket.gethostbyname(socket.gethostname())


def info(title):
    print(title)
    print('module name:', __name__)
    if hasattr(os, 'getppid'):  # only available on Unix
        print('parent process:', os.getppid())
    print('process id:', os.getpid())


if __name__ == "__main__":     
    """
    Starting the simulations on the local computer.
    Call with --cpu option if not using 100% resources    
    
    """
    from optparse import OptionParser
    import math
    parser = OptionParser()
    parser.add_option("-c", "--cpu", dest="cpu_load",
                      help="CPU load between 0 and 1, i.e. 0.5 uses half the n_cores")

    (options, args) = parser.parse_args()

    # Handle the number of cores
    print('#'*60)
    print('# Simulator')
    print('#'*60)
    n_cores = multiprocessing.cpu_count()
    print('CPUs: ', n_cores)
    if options.cpu_load:
        n_cores = int(math.floor(float(options.cpu_load)*n_cores))
    print('Used CPUs: ', n_cores)
    print('#'*60)

    size = 30
    opt_res = fit_parallel(n_cores=n_cores)
    # opt_res.report()
    analyze_optimization(opt_res)

