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
from typing import Dict
import os
import numpy as np
import socket
from sbmlsim.fit.fit import run_optimization, analyze_optimization

from sbmlsim.simulator import SimulatorSerial
from sbmlsim.fit.optimization import OptimizationProblem, SamplingType, OptimizerType
from sbmlsim.fit.analysis import OptimizationResult
from sbmlsim.utils import timeit

logger = logging.getLogger(__name__)


def worker(size, seed, opdict):
    """ Creates a worker for the cpu which listens for available simulations. """
    ip = get_ip_address()

    while True:
        print('{:<20} <Run fit>'.format(ip))

        simulator = SimulatorSerial()
        # op = OptimizationProblem(simulator=simulator, **op_kwargs)
        op = OptimizationProblem(simulator=simulator, **opdict)

        opt_res = run_optimization(
            op, size=size, seed=seed,
            verbose=False,
            optimizer=OptimizerType.LEAST_SQUARE,
            sampling=SamplingType.LOGUNIFORM_LHS,
            diff_step=0.05,
        )
        return opt_res


@timeit
def fit_parallel(n_cores: int, size: int, op_dict: Dict):
    """

    :param n_cores: number of workers
    :param size: number of optimizations per worker
    :param op_dict: optimization problem
    :return:
    """
    # Lock for syncronization between processes (but locks)
    lock = multiprocessing.Lock()

    pool = multiprocessing.Pool(processes=n_cores)
    sizes = [size]*n_cores
    opdicts = [op_dict] * n_cores  # FIXME: copy?
    seeds = list(np.random.randint(low=1, high=2000, size=n_cores))

    # mapping multiple arguments
    opt_results = pool.starmap(worker, zip(sizes, seeds, opdicts))

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

    # FIXME: load op problem from file or JSON
    """
    size = 10
    op_dict = op_mid1oh_iv
    opt_res = fit_parallel(n_cores=n_cores, size=size, op_dict=op_dict)
    # opt_res.report()
    analyze_optimization(opt_res)
    """

